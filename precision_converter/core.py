"""Load, convert and save PyTorch checkpoints and safetensors files."""

from __future__ import annotations

import fnmatch
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Target precisions. "keep" leaves every tensor in the dtype it already has.
PRECISIONS: dict[str, torch.dtype | None] = {
    "keep": None,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8_e4m3fn": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}

CONVERSION_TYPES = ("full", "ema-only", "no-ema")

EMA_PREFIX = "model_ema."
MODEL_PREFIX = "model."


class UnsafeCheckpointError(Exception):
    """The checkpoint holds pickled objects that a weights-only load refuses to run."""


@dataclass
class ConversionReport:
    converted: int = 0  # tensors cast to the target dtype
    unchanged: int = 0  # values kept as they were: non-float, already the target dtype, or matched by keep_original
    dropped: int = 0  # EMA copies and EMA bookkeeping removed from the output
    ema_swapped: int = 0  # weights replaced by their EMA copy
    clamped: dict[str, int] = field(default_factory=dict)  # key -> number of values clamped into the target range


@dataclass
class CheckpointInfo:
    tensors: int
    parameters: int
    bytes: int
    dtypes: dict[str, tuple[int, int]]  # dtype -> (tensor count, bytes)
    ema_tensors: int
    other_keys: list[str]  # keys whose value isn't a tensor


def load_checkpoint(
    path: str | os.PathLike, *, unsafe: bool = False
) -> tuple[dict[str, object], dict[str, str] | None]:
    """Load a state dict and, for safetensors files, its metadata.

    PyTorch checkpoints are loaded with ``weights_only=True`` unless ``unsafe`` is set,
    because full unpickling can run arbitrary code from the file.
    """
    path = Path(path)
    if path.suffix == ".safetensors":
        with safe_open(path, framework="pt", device="cpu") as f:
            return {key: f.get_tensor(key) for key in f.keys()}, f.metadata()

    try:
        obj = torch.load(path, map_location="cpu", weights_only=not unsafe)
    except pickle.UnpicklingError as e:
        raise UnsafeCheckpointError(
            f"{path} contains pickled objects that a safe (weights-only) load refuses. "
            "If you trust where this file came from, run again with --unsafe-load."
        ) from e

    if isinstance(obj, Mapping) and isinstance(obj.get("state_dict"), Mapping):
        obj = obj["state_dict"]
    if not isinstance(obj, Mapping):
        raise ValueError(f"{path} doesn't contain a state dict (found {type(obj).__name__})")
    return dict(obj), None


def _ema_key(key: str, state_dict: Mapping[str, object]) -> str | None:
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
    # fp8 tensors have few kernels, so do the range check in fp32.
    src = t.float() if t.element_size() == 1 else t
    limit = torch.finfo(dtype).max
    clamped = 0
    if torch.finfo(t.dtype).max > limit:
        out_of_range = src.abs() > limit
        if dtype is not torch.float8_e4m3fn:  # e4m3fn has no inf, so infs get clamped too
            out_of_range &= torch.isfinite(src)
        clamped = int(out_of_range.sum())
        if clamped:
            src = torch.where(out_of_range, src.clamp(-limit, limit), src)
    return src.to(dtype), clamped


def convert_state_dict(
    state_dict: Mapping[str, object],
    precision: str = "keep",
    conversion: str = "full",
    keep_original: Iterable[str] = (),
) -> tuple[dict[str, object], ConversionReport]:
    """Convert the floating-point tensors of a state dict to ``precision``.

    ``conversion`` is "full" (everything), "ema-only" (swap weights for their EMA copies
    and drop the EMA keys) or "no-ema" (drop the EMA keys). Integer and bool tensors are
    never cast, and keys matching a glob in ``keep_original`` keep their dtype.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}, expected one of {', '.join(PRECISIONS)}")
    if conversion not in CONVERSION_TYPES:
        raise ValueError(f"unknown conversion {conversion!r}, expected one of {', '.join(CONVERSION_TYPES)}")
    dtype = PRECISIONS[precision]
    keep_original = list(keep_original)
    report = ConversionReport()
    converted: dict[str, object] = {}

    for key, value in state_dict.items():
        if conversion != "full" and key.startswith(EMA_PREFIX):
            report.dropped += 1
            continue
        if conversion == "ema-only" and key.startswith(MODEL_PREFIX):
            ema_key = _ema_key(key, state_dict)
            if ema_key is not None:
                value = state_dict[ema_key]
                report.ema_swapped += 1

        if (
            dtype is None
            or not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or value.dtype == dtype
            or any(fnmatch.fnmatchcase(key, pattern) for pattern in keep_original)
        ):
            converted[key] = value
            report.unchanged += 1
            continue

        converted[key], clamped = _cast(value, dtype)
        report.converted += 1
        if clamped:
            report.clamped[key] = clamped

    return converted, report


def save_checkpoint(
    state_dict: Mapping[str, object],
    path: str | os.PathLike,
    *,
    metadata: Mapping[str, str] | None = None,
) -> list[str]:
    """Save a state dict, as safetensors if ``path`` ends in .safetensors.

    PyTorch files are saved as ``{"state_dict": ...}`` like Lightning checkpoints. The file
    is written next to its destination first and moved into place once complete.
    Returns the keys left out because safetensors can only store tensors.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    skipped: list[str] = []
    try:
        if path.suffix == ".safetensors":
            tensors: dict[str, torch.Tensor] = {}
            seen: set[int] = set()
            for key, value in state_dict.items():
                if not isinstance(value, torch.Tensor):
                    skipped.append(key)
                    continue
                value = value.contiguous()
                # safetensors refuses tensors that share memory, e.g. tied weights.
                ptr = value.untyped_storage().data_ptr()
                if value.numel() and ptr in seen:
                    value = value.clone()
                seen.add(value.untyped_storage().data_ptr())
                tensors[key] = value
            save_file(tensors, tmp, metadata={"format": "pt", **(metadata or {})})
        else:
            torch.save({"state_dict": dict(state_dict)}, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return skipped


def describe(state_dict: Mapping[str, object]) -> CheckpointInfo:
    """Count the tensors, parameters and bytes of a state dict per dtype."""
    dtypes: dict[str, tuple[int, int]] = {}
    tensors = parameters = total = ema = 0
    other: list[str] = []
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            other.append(key)
            continue
        size = value.numel() * value.element_size()
        name = str(value.dtype).removeprefix("torch.")
        count, nbytes = dtypes.get(name, (0, 0))
        dtypes[name] = (count + 1, nbytes + size)
        tensors += 1
        parameters += value.numel()
        total += size
        ema += key.startswith(EMA_PREFIX)
    return CheckpointInfo(tensors, parameters, total, dtypes, ema, other)
