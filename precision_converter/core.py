"""Converting state dicts: precision casts, scaled FP8, EMA extraction and error measurement."""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Collection, Iterable, Iterator

import torch

from . import fp8
from .io import (
    UnsafeCheckpointError,  # noqa: F401  (re-exported)
    SafetensorsWriter,
    ShardedWriter,
    header_size_bound,
    load_checkpoint,  # noqa: F401  (re-exported)
    tensor_dtype,
    tensor_shape,
)

# Target precisions. "keep" leaves every tensor in the dtype it already has. "fp8_scaled" stores
# Linear and Embedding weights as FP8 with a per-tensor scale, in ComfyUI's format.
PRECISIONS: dict[str, torch.dtype | None] = {
    "keep": None,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8_e4m3fn": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "fp8_scaled": torch.float8_e4m3fn,
}

CONVERSION_TYPES = ("full", "ema-only", "no-ema")

EMA_PREFIX = "model_ema."
MODEL_PREFIX = "model."


@dataclass
class ConversionReport:
    converted: int = 0  # tensors cast to the target precision
    unchanged: int = 0  # values kept as they were: non-float, already the target dtype, or matched by keep_original
    dropped: int = 0  # EMA copies and EMA bookkeeping removed from the output
    ema_swapped: int = 0  # weights replaced by their EMA copy
    dequantized: int = 0  # scaled FP8 weights expanded with their scales
    scaled: int = 0  # weights stored as scaled FP8
    fallback: list[str] = field(default_factory=list)  # kept at the fallback precision because of max_error
    clamped: dict[str, int] = field(default_factory=dict)  # key -> number of values clamped into the target range
    errors: dict[str, float] = field(default_factory=dict)  # key -> relative error, when measured


@dataclass
class CheckpointInfo:
    tensors: int
    parameters: int
    bytes: int
    dtypes: dict[str, tuple[int, int]]  # dtype -> (tensor count, bytes)
    ema_tensors: int
    other_keys: list[str]  # keys whose value isn't a tensor
    formats: list[str] = field(default_factory=list)  # recognised layouts, e.g. "ComfyUI scaled FP8"


def _ema_key(key: str, state_dict: Mapping) -> str | None:
    """Find the EMA copy of a ``model.`` weight.

    Latent-diffusion style EMA (Stable Diffusion and friends) stores ``model.a.b.c`` as
    ``model_ema.abc`` because buffer names can't contain dots; other setups keep the dots.
    """
    rest = key[len(MODEL_PREFIX):]
    for candidate in (EMA_PREFIX + rest.replace(".", ""), EMA_PREFIX + rest):
        if candidate in state_dict:
            return candidate
    return None


def _cast(t: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, int]:
    """Cast a float tensor, clamping finite values that don't fit the target range.

    A plain cast turns out-of-range values into inf (fp16) or NaN (fp8 on older PyTorch).
    Returns the cast tensor and how many values were clamped.
    """
    limit = torch.finfo(dtype).max
    if torch.finfo(t.dtype).max <= limit or fp8.abs_max(t) <= limit and not _has_inf(t, dtype):
        return t.to(dtype), 0
    # fp8 tensors have few kernels, so do the clamping in fp32.
    src = t.float() if t.element_size() == 1 else t
    out_of_range = src.abs() > limit
    if dtype is not torch.float8_e4m3fn:  # e4m3fn has no inf, so infs get clamped too
        out_of_range &= torch.isfinite(src)
    clamped = int(out_of_range.sum())
    if clamped:
        src = torch.where(out_of_range, src.clamp(-limit, limit), src)
    return src.to(dtype), clamped


def _has_inf(t: torch.Tensor, dtype: torch.dtype) -> bool:
    # Only matters for e4m3fn, which can't store inf: those values have to be clamped as well.
    return dtype is torch.float8_e4m3fn and bool(torch.isinf(t.float() if t.element_size() == 1 else t).any())


def _restored(t: torch.Tensor, scale: torch.Tensor | None = None):
    """A function returning chunks of ``t`` (times ``scale``) as FP32, for error measurement."""
    flat = t.reshape(-1)

    def chunk(start: int, end: int) -> torch.Tensor:
        values = flat[start:end].float()
        return values * scale.float() if scale is not None else values

    return chunk


def _unconverted(value: object, dequantized: bool, dtype: torch.dtype | None) -> object:
    """A value that keeps its dtype. A dequantized weight has no original dtype left (it is FP32
    after applying its scale), so it is stored in bf16, what these models compute in."""
    if dequantized and dtype is not torch.float32 and isinstance(value, torch.Tensor):
        return value.to(torch.bfloat16)
    return value


def _matches(key: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(key, pattern) for pattern in patterns)


def iter_convert(
    state_dict: Mapping,
    precision: str = "keep",
    conversion: str = "full",
    keep_original: Iterable[str] = (),
    *,
    max_error: float | None = None,
    fallback: str = "keep",
    measure: bool = False,
    keys: Iterable[str] | None = None,
    quantize_only: Collection[str] | None = None,
    report: ConversionReport | None = None,
) -> Iterator[tuple[str, object]]:
    """Yield the converted entries of a state dict one at a time.

    Same options as :func:`convert_state_dict`. ``keys`` restricts the output to some keys (one
    shard, say) while EMA copies and scales are still looked up in the whole ``state_dict``.
    Statistics are added to ``report``.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}, expected one of {', '.join(PRECISIONS)}")
    if fallback not in PRECISIONS or fallback == "fp8_scaled":
        raise ValueError(f"unknown fallback precision {fallback!r}")
    if conversion not in CONVERSION_TYPES:
        raise ValueError(f"unknown conversion {conversion!r}, expected one of {', '.join(CONVERSION_TYPES)}")
    report = report if report is not None else ConversionReport()
    dtype = PRECISIONS[precision]
    fallback_dtype = PRECISIONS[fallback]
    keep_original = list(keep_original)
    measure = measure or max_error is not None
    # Scaled FP8 input only means something together with its scales, so expand it before any cast.
    scaled = fp8.find_scaled_weights(state_dict) if precision != "keep" else fp8.ScaledWeights()

    for key in state_dict if keys is None else keys:
        if key in scaled.extra:
            continue
        if conversion != "full" and key.startswith(EMA_PREFIX):
            report.dropped += 1
            continue
        source_key = key
        if conversion == "ema-only" and key.startswith(MODEL_PREFIX):
            ema_key = _ema_key(key, state_dict)
            if ema_key is not None:
                source_key = ema_key
                report.ema_swapped += 1
        value = state_dict[source_key]

        dequantized = source_key in scaled.scales
        if dequantized:
            weight = value.view(scaled.stored_as_bytes[source_key]) if source_key in scaled.stored_as_bytes else value
            value = fp8.dequantize(weight, state_dict[scaled.scales[source_key]])  # FP32
            report.dequantized += 1

        if (
            dtype is None
            or not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or _matches(key, keep_original)
            or (value.dtype == dtype and precision != "fp8_scaled")
        ):
            yield key, _unconverted(value, dequantized, dtype)
            report.unchanged += 1
            continue

        quantize = (
            key in quantize_only and value.dim() >= 2
            if quantize_only is not None
            else fp8.is_quantizable(key, {key: value})
        )
        if precision == "fp8_scaled" and quantize:
            q, scale = fp8.quantize_scaled(value, dtype)
            error = fp8.relative_error(value, _restored(q, scale)) if measure else None
            if error is not None:
                report.errors[key] = error
            if max_error is None or error <= max_error:
                module = key[: -len(".weight")]
                yield key, q
                yield module + ".weight_scale", scale
                yield module + ".comfy_quant", fp8.comfy_quant_entry(dtype)
                report.converted += 1
                report.scaled += 1
                continue
            report.fallback.append(key)
            target = fallback_dtype
        elif precision == "fp8_scaled":
            target = fallback_dtype  # norms, biases, convolutions: ComfyUI can't load them as FP8
        else:
            target = dtype

        if target is None or value.dtype == target:
            yield key, _unconverted(value, dequantized, dtype)
            report.unchanged += 1
            continue

        out, clamped = _cast(value, target)
        if measure and target is dtype:
            error = fp8.relative_error(value, _restored(out))
            report.errors[key] = error
            if max_error is not None and error > max_error:
                report.fallback.append(key)
                if fallback_dtype is None or value.dtype == fallback_dtype:
                    yield key, _unconverted(value, dequantized, dtype)
                    report.unchanged += 1
                    continue
                out, clamped = _cast(value, fallback_dtype)
        if clamped:
            report.clamped[key] = clamped
        yield key, out
        report.converted += 1


def convert_state_dict(
    state_dict: Mapping,
    precision: str = "keep",
    conversion: str = "full",
    keep_original: Iterable[str] = (),
    *,
    max_error: float | None = None,
    fallback: str = "keep",
    measure: bool = False,
    quantize_only: Collection[str] | None = None,
) -> tuple[dict[str, object], ConversionReport]:
    """Convert the floating-point tensors of a state dict to ``precision``.

    ``conversion`` is "full" (everything), "ema-only" (swap weights for their EMA copies and
    drop the EMA keys) or "no-ema" (drop the EMA keys). Integer and bool tensors are never cast,
    and keys matching a glob in ``keep_original`` keep their dtype. Scaled FP8 weights (ComfyUI,
    DeepSeek-style block scales, compressed-tensors) are dequantized before converting.

    With ``measure``, the relative error of every converted tensor is recorded in the report.
    With ``max_error``, tensors whose error would exceed it are stored at ``fallback`` precision
    instead ("keep" keeps their dtype). With ``fp8_scaled``, ``quantize_only`` names the weights to
    quantize (for example from :func:`fp8.reference_layers`) instead of every Linear weight.
    """
    report = ConversionReport()
    converted = dict(
        iter_convert(
            state_dict, precision, conversion, keep_original,
            max_error=max_error, fallback=fallback, measure=measure, quantize_only=quantize_only, report=report,
        )
    )
    return converted, report


def output_size_bound(state_dict: Mapping, keys: Iterable[str], precision: str, metadata) -> int:
    """Header space to reserve for writing these keys converted to ``precision``."""
    entries = []
    for key in keys:
        shape = tensor_shape(state_dict, key) or []
        entries.append((key, shape))
        if precision == "fp8_scaled" and key.endswith(".weight") and len(shape) >= 2:
            module = key[: -len(".weight")]
            entries += [(module + ".weight_scale", []), (module + ".comfy_quant", [64])]
    return header_size_bound(entries, metadata)


def save_checkpoint(
    state_dict: Mapping,
    path: str | os.PathLike,
    *,
    metadata: Mapping[str, str] | None = None,
    wrap: bool = True,
) -> list[str]:
    """Save a state dict, as safetensors if ``path`` ends in .safetensors.

    safetensors files are written one tensor at a time. PyTorch files are saved as
    ``{"state_dict": ...}`` like Lightning checkpoints unless ``wrap`` is False. The file is
    written next to its destination first and moved into place once complete. Returns the keys
    left out because safetensors can only store tensors.
    """
    path = Path(path)
    if path.suffix == ".safetensors":
        tensors = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
        reserve = header_size_bound(((k, list(v.shape)) for k, v in tensors.items()), metadata)
        _, _, skipped = write_safetensors(state_dict.items(), path, reserve=reserve, metadata=metadata)
        return skipped
    tmp = path.with_name(path.name + ".tmp")
    try:
        data = dict(state_dict)
        torch.save({"state_dict": data} if wrap else data, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return []


def clean_metadata(metadata: Mapping[str, str] | None, precision: str) -> dict[str, str] | None:
    """Drop header metadata that would make ComfyUI treat converted weights as still quantized."""
    if not metadata or precision == "keep":
        return dict(metadata) if metadata else None
    return {k: v for k, v in metadata.items() if k != "_quantization_metadata"}


def write_safetensors(
    entries: Iterable[tuple[str, object]],
    path: str | os.PathLike,
    *,
    reserve: int,
    metadata: Mapping[str, str] | None = None,
    max_shard_bytes: int | None = None,
) -> tuple[list[Path], int, list[str]]:
    """Stream converted entries into a safetensors file, or into shards with an index.

    Returns the files written, the number of tensor bytes, and the non-tensor keys left out.
    """
    path = Path(path)
    metadata = {"format": "pt", **(metadata or {})}
    tmp = path.with_name(path.name + ".tmp")
    if max_shard_bytes:
        writer: SafetensorsWriter | ShardedWriter = ShardedWriter(
            path, max_bytes=max_shard_bytes, reserve=reserve, metadata=metadata
        )
    else:
        writer = SafetensorsWriter(tmp, reserve=reserve, metadata=metadata)
    skipped: list[str] = []
    try:
        for key, value in entries:
            if isinstance(value, torch.Tensor):
                writer.add(key, value)
            else:
                skipped.append(key)
        nbytes = writer.nbytes
        if isinstance(writer, ShardedWriter):
            files = writer.close()
        else:
            writer.close()
            os.replace(tmp, path)
            files = [path]
    except BaseException:
        writer.abort()
        tmp.unlink(missing_ok=True)
        raise
    return files, nbytes, skipped


def detect_formats(state_dict: Mapping, metadata: Mapping[str, str] | None = None) -> list[str]:
    """Name the layouts recognised in a state dict, for ``--info``."""
    keys = list(state_dict)
    formats = []
    if any(k.endswith(".comfy_quant") for k in keys):
        formats.append("ComfyUI scaled FP8 (comfy_quant)")
    if fp8.LEGACY_MARKER in state_dict or (metadata and "_quantization_metadata" in metadata):
        formats.append("older ComfyUI scaled FP8")
    if any(k.endswith(".weight_scale_inv") for k in keys):
        formats.append("block-scaled FP8 (DeepSeek / Qwen style)")
    elif any(k.endswith(".weight_scale") for k in keys) and not any(k.endswith(".comfy_quant") for k in keys):
        formats.append("FP8 with weight scales (compressed-tensors style)")
    if any(k.startswith(EMA_PREFIX) for k in keys):
        formats.append("EMA weights (model_ema.*)")
    if any(k.startswith("lora_unet_") or k.startswith("lora_te") or ".lora_A." in k or ".lora_down." in k for k in keys):
        formats.append("LoRA")
    return formats


def describe(state_dict: Mapping, metadata: Mapping[str, str] | None = None) -> CheckpointInfo:
    """Count the tensors, parameters and bytes of a state dict per dtype, without reading data."""
    dtypes: dict[str, tuple[int, int]] = {}
    tensors = parameters = total = ema = 0
    other: list[str] = []
    for key in state_dict:
        dtype = tensor_dtype(state_dict, key)
        if dtype is None:
            other.append(key)
            continue
        numel = 1
        for dim in tensor_shape(state_dict, key):
            numel *= dim
        size = numel * torch.empty((), dtype=dtype).element_size()
        name = str(dtype).removeprefix("torch.")
        count, nbytes = dtypes.get(name, (0, 0))
        dtypes[name] = (count + 1, nbytes + size)
        tensors += 1
        parameters += numel
        total += size
        ema += key.startswith(EMA_PREFIX)
    return CheckpointInfo(tensors, parameters, total, dtypes, ema, other, detect_formats(state_dict, metadata))
