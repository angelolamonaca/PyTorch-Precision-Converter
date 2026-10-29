"""ComfyUI's INT8 format (``int8_tensorwise``), with or without ConvRot.

A weight ``[out, in]`` is stored as ``<m>.weight`` int8, ``<m>.weight_scale`` FP32 (one scale per output row,
``[out, 1]``, or a scalar) and ``<m>.comfy_quant`` holding
``{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`` as UTF-8 bytes. With ConvRot, the
weight is first rotated by a normalized Hadamard matrix in groups of ``convrot_groupsize`` input features; ComfyUI
rotates the activations the same way at run time, so the product is unchanged while outliers are spread over each
group. Spec: https://github.com/Comfy-Org/ComfyUI/blob/master/QUANTIZATION.md

Rotation and rounding follow comfy-kitchen's reference implementation in FP32, so converting the bf16 file
Comfy-Org released gives back the int8 values of their int8_convrot file.
"""

from __future__ import annotations

import json
import math

import torch

FORMAT = "int8_tensorwise"
GROUP_SIZE = 256  # ComfyUI's default convrot_groupsize, used by every Comfy-Org int8_convrot file
CHUNK = 1 << 24  # values per chunk of rows, to bound temporary memory

# Kronecker powers of this symmetric 4x4 block give the "regular" Hadamard matrices ConvRot uses.
_H4 = ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1))
_HADAMARD: dict[int, torch.Tensor] = {}


def hadamard(size: int) -> torch.Tensor:
    """The normalized regular Hadamard matrix of ``size`` (a power of 4) in FP32. It is symmetric and
    orthogonal, so it is its own inverse."""
    if size < 4 or size & (size - 1) or math.log(size, 4) % 1:
        raise ValueError(f"the ConvRot group size must be a power of 4, got {size}")
    if size not in _HADAMARD:
        block = torch.tensor(_H4, dtype=torch.float32)
        h = block
        while h.shape[0] < size:
            h = torch.kron(h, block)
        _HADAMARD[size] = h / size**0.5
    return _HADAMARD[size]


def rotate(weight: torch.Tensor, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """``weight @ H.T`` in groups of ``group_size`` columns, in FP32. Rotating twice gives the input back."""
    rows, cols = weight.shape
    grouped = weight.float().reshape(rows, cols // group_size, group_size)
    return (grouped @ hadamard(group_size).T).reshape(rows, cols)


def can_quantize(shape: list[int] | tuple[int, ...] | None, group_size: int = GROUP_SIZE) -> bool:
    """Whether a weight of this shape can be stored as int8_convrot: 2-D, with a whole number of groups."""
    return shape is not None and len(shape) == 2 and shape[1] % group_size == 0


def quantize_convrot(t: torch.Tensor, group_size: int = GROUP_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate a 2-D weight and quantize it to int8 with one FP32 scale per row. Returns the int8 tensor and
    the ``[rows, 1]`` scales."""
    if not can_quantize(list(t.shape), group_size):
        raise ValueError(f"int8_convrot needs a 2-D weight with columns divisible by {group_size}, got {list(t.shape)}")
    rows, cols = t.shape
    q = torch.empty((rows, cols), dtype=torch.int8)
    scale = torch.empty((rows, 1), dtype=torch.float32)
    step = max(1, CHUNK // cols)
    for start in range(0, rows, step):
        rotated = rotate(t[start : start + step], group_size)
        s = (rotated.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_(min=1e-30)
        q[start : start + step] = rotated.div_(s).round_().clamp_(-128, 127).to(torch.int8)
        scale[start : start + step] = s
    return q, scale


def dequantize(q: torch.Tensor, scale: torch.Tensor, group_size: int = 0) -> torch.Tensor:
    """Turn an int8 weight and its scale back into FP32 values, rotating back when ``group_size`` > 0."""
    values = q.float() * scale.float()
    if group_size and values.dim() == 2:
        out = torch.empty_like(values)
        step = max(1, CHUNK // values.shape[1])
        for start in range(0, values.shape[0], step):
            out[start : start + step] = rotate(values[start : start + step], group_size)
        return out
    return values


def relative_error(original: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, group_size: int) -> float:
    """``||original - restored|| / ||original||`` for an int8 weight, computed in chunks of rows."""
    diff = norm = 0.0
    step = max(1, CHUNK // max(1, original.shape[-1]))
    for start in range(0, original.shape[0], step):
        o = original[start : start + step].double()
        s = scale[start : start + step] if scale.dim() else scale
        r = dequantize(q[start : start + step], s, group_size).double()
        finite = torch.isfinite(o) & torch.isfinite(r)
        diff += ((o - r)[finite] ** 2).sum().item()
        norm += (o[finite] ** 2).sum().item()
    if norm == 0:
        return 0.0 if diff == 0 else 1.0
    return math.sqrt(diff / norm)


def comfy_quant_entry(group_size: int = GROUP_SIZE) -> torch.Tensor:
    """The ``<module>.comfy_quant`` tensor marking a layer as int8_convrot for ComfyUI."""
    config = json.dumps({"format": FORMAT, "convrot": True, "convrot_groupsize": group_size})
    return torch.tensor(list(config.encode("utf-8")), dtype=torch.uint8)


def convrot_group_size(config: dict) -> int:
    """The ConvRot group size of an ``int8_tensorwise`` comfy_quant config, or 0 without rotation. Like
    ComfyUI, reads the keys at the top level or under ``params``."""
    params = config.get("params") if isinstance(config.get("params"), dict) else {}
    if not config.get("convrot", params.get("convrot", False)):
        return 0
    return int(config.get("convrot_groupsize", params.get("convrot_groupsize", GROUP_SIZE)))
