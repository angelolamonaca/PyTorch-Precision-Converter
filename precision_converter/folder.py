"""Converting a whole model folder (Hugging Face Transformers or Diffusers layout)."""

from __future__ import annotations

import fnmatch
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Collection, Iterable

import torch

from .core import ConversionReport, clean_metadata, iter_convert, output_size_bound, write_safetensors
from .io import WEIGHT_SUFFIXES, ShardedSafetensors, open_checkpoint

SKIPPED_DIRS = {".git", ".cache", "__pycache__"}
CONFIG_DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


@dataclass
class FolderReport:
    report: ConversionReport = field(default_factory=ConversionReport)
    converted_files: list[str] = field(default_factory=list)
    copied_files: list[str] = field(default_factory=list)
    updated_configs: list[str] = field(default_factory=list)
    input_bytes: int = 0
    output_bytes: int = 0


def _files(root: Path) -> list[Path]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIPPED_DIRS)
        found += [Path(dirpath) / name for name in sorted(filenames)]
    return found


def convert_folder(
    src: str | os.PathLike,
    dst: str | os.PathLike,
    *,
    precision: str,
    conversion: str = "full",
    keep_original: Iterable[str] = (),
    max_error: float | None = None,
    fallback: str = "keep",
    measure: bool = False,
    skip_files: Iterable[str] = (),
    quantize_only: Collection[str] | None = None,
    fp8_max: float | None = None,
    unsafe: bool = False,
    progress: Callable[[str], None] | None = None,
) -> FolderReport:
    """Convert every weight file in ``src`` into the same layout under ``dst``.

    Other files are copied. Shard indexes are rewritten to match the new shards, and
    ``config.json`` files get the new ``torch_dtype`` (fp32/fp16/bf16) and lose their
    ``quantization_config`` once FP8 weights have been dequantized. Files matching a glob in
    ``skip_files`` (relative to ``src``, e.g. ``vae/*``) are copied unchanged.
    """
    src, dst = Path(src), Path(dst)
    if dst.resolve() == src.resolve() or src.resolve() in dst.resolve().parents:
        raise ValueError("the output folder can't be the input folder or inside it")
    skip_files = list(skip_files)
    result = FolderReport()
    files = _files(src)

    # Shards are converted one at a time, but EMA copies and FP8 scales can sit in another
    # shard, so lookups go through the whole set.
    shard_sets: dict[Path, ShardedSafetensors] = {}
    for index in (f for f in files if f.name.endswith(".safetensors.index.json")):
        sharded = ShardedSafetensors(index)
        for name in sharded.shard_names:
            shard_sets[(index.parent / name).resolve()] = sharded
    written_keys: dict[Path, tuple[list[str], int]] = {}  # output shard -> keys, tensor bytes
    converted_dirs: set[Path] = set()

    for path in files:
        rel = path.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        rel_posix = rel.as_posix()
        if path.name.endswith(".index.json") or (path.name == "config.json"):
            continue  # written after the weights
        is_weight = path.suffix in WEIGHT_SUFFIXES and not any(fnmatch.fnmatch(rel_posix, g) for g in skip_files)
        if not is_weight:
            shutil.copy2(path, out)
            result.copied_files.append(rel_posix)
            continue

        if progress:
            progress(rel_posix)
        result.input_bytes += path.stat().st_size
        options = dict(
            precision=precision, conversion=conversion, keep_original=keep_original,
            max_error=max_error, fallback=fallback, measure=measure, quantize_only=quantize_only,
            fp8_max=fp8_max, report=result.report,
        )
        if path.suffix == ".safetensors":
            own = open_checkpoint(path)
            sharded = shard_sets.get(path.resolve())
            lookup: Mapping = sharded if sharded is not None else own.tensors
            keys = list(own.tensors)
            entries = iter_convert(lookup, keys=keys, **options)
            reserve = output_size_bound(lookup, keys, precision, own.metadata)
            metadata = clean_metadata(own.metadata, precision)
            _, nbytes, _ = write_safetensors(entries, out, reserve=reserve, metadata=metadata)
            written_keys[out] = (_keys_in(out), nbytes)
        else:
            checkpoint = open_checkpoint(path, unsafe=unsafe)
            data = dict(iter_convert(checkpoint.tensors, **options))
            tmp = out.with_name(out.name + ".tmp")
            torch.save({"state_dict": data} if checkpoint.wrapped else data, tmp)
            os.replace(tmp, out)
            written_keys[out] = (list(data), sum(v.numel() * v.element_size() for v in data.values() if isinstance(v, torch.Tensor)))
        result.output_bytes += out.stat().st_size
        result.converted_files.append(rel_posix)
        converted_dirs.add(rel.parent)

    for path in files:
        rel = path.relative_to(src)
        out = dst / rel
        if path.name.endswith(".index.json"):
            _rewrite_index(path, out, written_keys)
        elif path.name == "config.json":
            if rel.parent in converted_dirs and _rewrite_config(path, out, precision, result.report.dequantized > 0):
                result.updated_configs.append(rel.as_posix())
            else:
                shutil.copy2(path, out)
                result.copied_files.append(rel.as_posix())
    return result


def _keys_in(path: Path) -> list[str]:
    with open(path, "rb") as f:
        size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(size))
    return [k for k in header if k != "__metadata__"]


def _rewrite_index(src: Path, out: Path, written: dict[Path, tuple[list[str], int]]) -> None:
    index = json.loads(src.read_text())
    weight_map: dict[str, str] = {}
    total = 0
    for name in dict.fromkeys(index.get("weight_map", {}).values()):
        shard = out.parent / name
        if shard in written:
            keys, nbytes = written[shard]
            weight_map.update(dict.fromkeys(keys, name))
            total += nbytes
    index.setdefault("metadata", {})["total_size"] = total
    index["weight_map"] = weight_map
    out.write_text(json.dumps(index, indent=2) + "\n")


def _rewrite_config(src: Path, out: Path, precision: str, dequantized: bool) -> bool:
    """Update dtype fields in a Transformers config. Returns False if nothing needed changing."""
    try:
        config = json.loads(src.read_text())
    except ValueError:
        return False
    if not isinstance(config, dict):
        return False
    changed = False
    if precision in CONFIG_DTYPES:
        for field_name in ("torch_dtype", "dtype"):
            if isinstance(config.get(field_name), str) and config[field_name] != CONFIG_DTYPES[precision]:
                config[field_name] = CONFIG_DTYPES[precision]
                changed = True
    if dequantized and precision != "fp8_scaled" and "quantization_config" in config:
        del config["quantization_config"]
        changed = True
    if changed:
        out.write_text(json.dumps(config, indent=2) + "\n")
    return changed
