"""Scaled FP8: reading the formats other tools write, and writing ComfyUI's.

A scaled FP8 weight is stored as FP8 values plus a scale; the real weight is ``fp8 * scale``.
The layouts this module understands:

- ComfyUI (v0.4.0+): ``<m>.weight`` FP8, ``<m>.weight_scale`` FP32 scalar and
  ``<m>.comfy_quant`` holding ``{"format": "float8_e4m3fn"}`` as UTF-8 bytes. Optional
  ``<m>.input_scale``. Spec: https://github.com/Comfy-Org/ComfyUI/blob/master/QUANTIZATION.md
- Older ComfyUI ``scaled_fp8`` files: ``<m>.scale_weight`` scalar and a ``scaled_fp8`` marker.
- Block-scaled FP8 (DeepSeek-V3/R1, Qwen3 FP8 and others): ``<m>.weight_scale_inv`` with one
  scale per 128x128 block.
- compressed-tensors FP8 (vLLM / llm-compressor): ``<m>.weight_scale``, per tensor or per row.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch

from .io import tensor_dtype, tensor_shape

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
SCALE_SUFFIXES = (".weight_scale_inv", ".weight_scale", ".scale_weight")
# Entries that only describe quantization; they go away when weights are dequantized.
QUANT_EXTRA_SUFFIXES = (".comfy_quant", ".input_scale", ".scale_input")
LEGACY_MARKER = "scaled_fp8"
COMFY_FORMATS = {torch.float8_e4m3fn: "float8_e4m3fn", torch.float8_e5m2: "float8_e5m2"}
# Most of Comfy-Org's fp8_scaled text encoders (Gemma 4, Qwen3-VL, Llama 3.1, T5) map each layer's
# largest weight to 416 instead of FP8's maximum of 448, which ComfyUI itself and many of their
# diffusion models use. The accuracy is the same either way; copying it lets a conversion of the
# same bf16 file match their file byte for byte.
COMFY_ORG_FP8_MAX = 416.0

CHUNK = 1 << 24  # elements per chunk for whole-tensor reductions, to bound temporary memory


@dataclass
class ScaledWeights:
    scales: dict[str, str] = field(default_factory=dict)  # weight key -> scale key
    stored_as_bytes: dict[str, torch.dtype] = field(default_factory=dict)  # FP8 weights saved as uint8
    extra: set[str] = field(default_factory=set)  # keys that only describe the quantization


def find_scaled_weights(state_dict: Mapping) -> ScaledWeights:
    """Find the FP8 weights that come with a scale, and the entries that describe them."""
    keys = set(state_dict)
    found = ScaledWeights()
    for key in keys:
        if not key.endswith(".weight"):
            continue
        module = key[: -len(".weight")]
        scale_key = next((module + s for s in SCALE_SUFFIXES if module + s in keys), None)
        if scale_key is None:
            continue
        dtype = tensor_dtype(state_dict, key)
        if dtype in FP8_DTYPES:
            found.scales[key] = scale_key
        elif dtype == torch.uint8 and module + ".comfy_quant" in keys:
            fmt = _comfy_format(state_dict[module + ".comfy_quant"])
            fp8 = next((d for d, name in COMFY_FORMATS.items() if name == fmt), None)
            if fp8 is not None:  # older files store FP8 weights as raw bytes
                found.scales[key] = scale_key
                found.stored_as_bytes[key] = fp8

    if found.scales:
        modules = {k[: -len(".weight")] for k in found.scales}
        found.extra = set(found.scales.values())
        found.extra |= {m + s for m in modules for s in QUANT_EXTRA_SUFFIXES if m + s in keys}
        if LEGACY_MARKER in keys:
            found.extra.add(LEGACY_MARKER)
    unsupported = {
        k for k in keys if k.endswith(".comfy_quant") and k[: -len(".comfy_quant")] + ".weight" not in found.scales
    }
    if unsupported:
        formats = sorted({_comfy_format(state_dict[k]) for k in unsupported})
        raise ValueError(
            f"{len(unsupported)} layers use ComfyUI quantization formats this tool can't dequantize yet: "
            f"{', '.join(formats)}. Only float8_e4m3fn and float8_e5m2 are supported."
        )
    return found


def _comfy_format(value: torch.Tensor) -> str:
    try:
        return str(json.loads(bytes(value.tolist()).decode("utf-8")).get("format"))
    except (ValueError, AttributeError):
        return "unreadable"


def _block_size(weight_shape: list[int], scale_shape: list[int]) -> tuple[int, int] | None:
    """Infer the block size of a block-scaled weight, e.g. (128, 128)."""
    if len(weight_shape) != 2 or len(scale_shape) != 2:
        return None
    for size in (128, 64, 256, 32, 512, 16):
        if all(math.ceil(w / size) == s for w, s in zip(weight_shape, scale_shape)):
            return size, size
    return None


def dequantize(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Turn an FP8 weight and its scale back into FP32 values."""
    w = weight.float()
    scale = scale.float()
    rows = w.shape[0] if w.dim() else 1
    if scale.numel() == 1:
        return w.mul_(scale.reshape(()))
    if scale.dim() == w.dim() == 2 and scale.shape == (rows, 1):  # one scale per output row
        return w.mul_(scale)
    if scale.dim() == 1 and scale.shape[0] == rows:
        return w.mul_(scale.reshape(-1, *([1] * (w.dim() - 1))))
    block = _block_size(list(w.shape), list(scale.shape))
    if block is not None:  # one scale per block, e.g. 128x128
        r, c = block
        expanded = scale.repeat_interleave(r, 0)[: w.shape[0]].repeat_interleave(c, 1)[:, : w.shape[1]]
        return w.mul_(expanded)
    raise ValueError(f"can't apply a scale of shape {list(scale.shape)} to a weight of shape {list(w.shape)}")


def abs_max(t: torch.Tensor) -> float:
    """Largest finite absolute value, computed in chunks to avoid a full-size copy."""
    flat = t.reshape(-1)
    best = 0.0
    for chunk in flat.split(CHUNK):
        chunk = chunk.float().abs()
        chunk = chunk[torch.isfinite(chunk)]
        if chunk.numel():
            best = max(best, chunk.max().item())
    return best


def quantize_scaled(
    t: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn, fp8_max: float | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to FP8 with one FP32 scale for the whole tensor, so the largest value maps to
    ``fp8_max``: by default the top of the FP8 range (448 for float8_e4m3fn). Returns the FP8
    tensor and the scale."""
    limit = torch.finfo(dtype).max
    if fp8_max is not None and not 0 < fp8_max <= limit:
        raise ValueError(f"fp8_max must be in (0, {limit:g}] for {dtype}, got {fp8_max}")
    amax = abs_max(t)
    scale = amax / (fp8_max or limit) if amax > 0 else 1.0
    out = torch.empty(t.shape, dtype=dtype)
    src, dst = t.reshape(-1), out.reshape(-1)
    for start in range(0, src.numel(), CHUNK):
        chunk = src[start : start + CHUNK].float() / scale
        dst[start : start + CHUNK] = chunk.clamp_(-limit, limit).to(dtype)
    return out, torch.tensor(scale, dtype=torch.float32)


def comfy_quant_entry(dtype: torch.dtype) -> torch.Tensor:
    """The ``<module>.comfy_quant`` tensor marking a layer as scaled FP8 for ComfyUI."""
    config = json.dumps({"format": COMFY_FORMATS[dtype]})
    return torch.tensor(list(config.encode("utf-8")), dtype=torch.uint8)


def reference_layers(header: Mapping[str, Mapping]) -> set[str]:
    """The weights a quantized reference file stores in low precision, from its header: layers with
    a ``comfy_quant`` entry (any format, FP8 or int8) and FP8 weights."""
    layers = set()
    for key, entry in header.items():
        if key.endswith(".comfy_quant"):
            layers.add(key[: -len(".comfy_quant")] + ".weight")
        elif key.endswith(".weight") and str(entry.get("dtype", "")).startswith("F8"):
            layers.add(key)
    return layers


def reference_fp8_max(header: Mapping[str, Mapping], read: Callable[[str], torch.Tensor]) -> float | None:
    """:data:`COMFY_ORG_FP8_MAX` if a reference file maps each layer's largest weight to 416, judged
    from its two smallest FP8 weights with a per-tensor scale; otherwise None, for the default 448.
    ``read(key)`` returns one tensor of the reference."""
    sampled = []
    for key, entry in header.items():
        if not key.endswith(".weight") or entry.get("dtype") != "F8_E4M3":
            continue
        module = key[: -len(".weight")]
        scale = next((header[module + s] for s in (".weight_scale", ".scale_weight") if module + s in header), None)
        if scale is not None and math.prod(scale.get("shape", [])) == 1:
            start, end = entry["data_offsets"]
            sampled.append((end - start, key))
    sampled = sorted(sampled)[:2]
    if not sampled or sampled[-1][0] > 1 << 26:  # don't download big layers just for this
        return None
    tops = {read(key).float().abs().max().item() for _, key in sampled}
    return COMFY_ORG_FP8_MAX if tops == {COMFY_ORG_FP8_MAX} else None


def is_quantizable(key: str, state_dict: Mapping) -> bool:
    """Whether a tensor becomes scaled FP8 with ``-p fp8_scaled``: the 2-D weight of a Linear
    layer. Embedding tables and embedding projections (any key containing "embed") stay in
    higher precision, as in Comfy-Org's own fp8_scaled releases."""
    shape = tensor_shape(state_dict, key)
    dtype = tensor_dtype(state_dict, key)
    return (
        key.endswith(".weight")
        and "embed" not in key.lower()
        and shape is not None
        and len(shape) == 2
        and dtype is not None
        and dtype.is_floating_point
    )


def relative_error(original: torch.Tensor, restored_chunks) -> float:
    """``||original - restored|| / ||original||``, computed in chunks.

    ``restored_chunks(start, end)`` returns the restored FP32 values for that flat range."""
    flat = original.reshape(-1)
    diff = norm = 0.0
    for start in range(0, flat.numel(), CHUNK):
        o = flat[start : start + CHUNK].double()
        r = restored_chunks(start, start + o.numel()).double()
        finite = torch.isfinite(o) & torch.isfinite(r)
        diff += ((o - r)[finite] ** 2).sum().item()
        norm += (o[finite] ** 2).sum().item()
    if norm == 0:
        return 0.0 if diff == 0 else 1.0
    return math.sqrt(diff / norm)
