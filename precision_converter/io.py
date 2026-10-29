"""Reading and writing checkpoints: safetensors files (single or sharded) and PyTorch pickles."""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch
from safetensors import safe_open

SAFETENSORS_DTYPES: dict[torch.dtype, str] = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.float8_e5m2: "F8_E5M2",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}
for _name, _code in (("uint16", "U16"), ("uint32", "U32"), ("uint64", "U64")):
    if hasattr(torch, _name):  # PyTorch 2.3+
        SAFETENSORS_DTYPES[getattr(torch, _name)] = _code
DTYPES_BY_CODE = {code: dtype for dtype, code in SAFETENSORS_DTYPES.items()}

WEIGHT_SUFFIXES = (".safetensors", ".ckpt", ".pt", ".pth", ".bin")

# Keys that hold the weights inside a training checkpoint, tried in this order.
NESTED_STATE_DICT_KEYS = ("state_dict", "model", "module", "model_state_dict", "net")


class UnsafeCheckpointError(Exception):
    """The checkpoint holds pickled objects that a weights-only load refuses to run."""


class SafetensorsFile(Mapping):
    """A safetensors file as a read-only mapping. Tensors are read when accessed, so a
    conversion only needs memory for what it keeps, not for the whole file."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self._file = safe_open(str(self.path), framework="pt", device="cpu")
        self._keys = list(self._file.keys())
        self._key_set = set(self._keys)
        self.metadata: dict[str, str] | None = self._file.metadata()

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self._key_set:
            raise KeyError(key)
        return self._file.get_tensor(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: object) -> bool:
        return key in self._key_set

    def shape(self, key: str) -> list[int]:
        return list(self._file.get_slice(key).get_shape())

    def dtype(self, key: str) -> torch.dtype:
        return DTYPES_BY_CODE[self._file.get_slice(key).get_dtype()]


class ShardedSafetensors(Mapping):
    """The shards listed in a ``*.safetensors.index.json``, read as one mapping."""

    def __init__(self, index_path: str | os.PathLike):
        self.path = Path(index_path)
        index = json.loads(self.path.read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self.index_metadata: dict = index.get("metadata", {})
        self._shards: dict[str, SafetensorsFile] = {}

    def shard(self, name: str) -> SafetensorsFile:
        if name not in self._shards:
            self._shards[name] = SafetensorsFile(self.path.parent / name)
        return self._shards[name]

    @property
    def shard_names(self) -> list[str]:
        return list(dict.fromkeys(self.weight_map.values()))

    @property
    def metadata(self) -> dict[str, str] | None:
        names = self.shard_names
        return self.shard(names[0]).metadata if names else None

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.shard(self.weight_map[key])[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.weight_map)

    def __len__(self) -> int:
        return len(self.weight_map)

    def __contains__(self, key: object) -> bool:
        return key in self.weight_map

    def shape(self, key: str) -> list[int]:
        return self.shard(self.weight_map[key]).shape(key)

    def dtype(self, key: str) -> torch.dtype:
        return self.shard(self.weight_map[key]).dtype(key)


def tensor_dtype(state_dict: Mapping, key: str) -> torch.dtype | None:
    """The dtype of a tensor without reading its data, or None for non-tensors."""
    if hasattr(state_dict, "dtype"):
        return state_dict.dtype(key)
    value = state_dict[key]
    return value.dtype if isinstance(value, torch.Tensor) else None


def tensor_shape(state_dict: Mapping, key: str) -> list[int] | None:
    """The shape of a tensor without reading its data, or None for non-tensors."""
    if hasattr(state_dict, "shape"):
        return state_dict.shape(key)
    value = state_dict[key]
    return list(value.shape) if isinstance(value, torch.Tensor) else None


@dataclass
class Checkpoint:
    tensors: Mapping  # key -> tensor (or other value, for PyTorch checkpoints)
    metadata: dict[str, str] | None  # safetensors header metadata
    nested_key: str | None = None  # where the weights were found inside a PyTorch checkpoint
    wrapped: bool = False  # the weights were stored as {"state_dict": ...}


def is_index_file(path: Path) -> bool:
    return path.name.endswith(".index.json")


def open_checkpoint(
    path: str | os.PathLike, *, unsafe: bool = False, key: str | None = None
) -> Checkpoint:
    """Open a checkpoint for reading.

    safetensors files and ``*.safetensors.index.json`` shard sets are read lazily. PyTorch
    checkpoints are memory-mapped when their format allows it and loaded with
    ``weights_only=True`` unless ``unsafe`` is set, because full unpickling can run arbitrary
    code from the file. ``key`` picks the state dict inside a training checkpoint; without it,
    ``state_dict``, ``model``, ``module``, ``model_state_dict`` and ``net`` are tried.
    """
    path = Path(path)
    if is_index_file(path):
        if not path.name.endswith(".safetensors.index.json"):
            raise ValueError(f"{path}: only safetensors shard indexes can be read as one file")
        sharded = ShardedSafetensors(path)
        return Checkpoint(sharded, sharded.metadata)
    if path.suffix == ".safetensors":
        f = SafetensorsFile(path)
        return Checkpoint(f, f.metadata)

    try:
        try:  # PyTorch 2.1 only memory-maps when given a str path
            obj = torch.load(str(path), map_location="cpu", weights_only=not unsafe, mmap=True)
        except RuntimeError as e:
            if "mmap" not in str(e):
                raise
            obj = torch.load(str(path), map_location="cpu", weights_only=not unsafe)  # legacy format
    except pickle.UnpicklingError as e:
        raise UnsafeCheckpointError(
            f"{path} contains pickled objects that a safe (weights-only) load refuses. "
            "If you trust where this file came from, run again with --unsafe-load."
        ) from e

    tensors, nested_key = _find_state_dict(obj, key, path)
    return Checkpoint(dict(tensors), None, nested_key, wrapped=nested_key == "state_dict")


def _find_state_dict(obj: object, key: str | None, path: Path) -> tuple[Mapping, str | None]:
    if not isinstance(obj, Mapping):
        raise ValueError(f"{path} doesn't contain a state dict (found {type(obj).__name__})")
    top_level = ", ".join(map(str, list(obj)[:20]))
    if key is not None:
        if not isinstance(obj.get(key), Mapping):
            raise ValueError(f"{path} has no state dict under {key!r} (top-level keys: {top_level})")
        return obj[key], key
    if isinstance(obj.get("state_dict"), Mapping):
        return obj["state_dict"], "state_dict"
    if any(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj, None
    for candidate in NESTED_STATE_DICT_KEYS[1:]:
        if isinstance(obj.get(candidate), Mapping):
            return obj[candidate], candidate
    raise ValueError(
        f"{path} has no tensors at the top level. Pick the state dict with --key "
        f"(top-level keys: {top_level})"
    )


def load_checkpoint(
    path: str | os.PathLike, *, unsafe: bool = False, key: str | None = None
) -> tuple[dict[str, object], dict[str, str] | None]:
    """Load a state dict into memory and, for safetensors files, its metadata."""
    checkpoint = open_checkpoint(path, unsafe=unsafe, key=key)
    return dict(checkpoint.tensors.items()), checkpoint.metadata


def _tensor_bytes(tensor: torch.Tensor) -> memoryview:
    # reshape(-1) because 0-dim tensors can't be viewed as another dtype.
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().data


def header_size_bound(entries: Iterable[tuple[str, list[int]]], metadata: Mapping[str, str] | None) -> int:
    """An upper bound for the JSON header of a safetensors file with these tensors."""
    size = len(json.dumps(dict(metadata or {}))) + 64
    for key, shape in entries:
        # "key":{"dtype":"F8_E4M3","shape":[...],"data_offsets":[<20 digits>,<20 digits>]},
        size += len(json.dumps(key)) + len(json.dumps(shape)) + 90
    return size


class SafetensorsWriter:
    """Writes a safetensors file one tensor at a time, so memory use stays at one tensor.

    The header goes first in the file but is only known at the end, so space for it is
    reserved up front (``reserve`` bytes, from :func:`header_size_bound`) and padded with
    spaces, which the safetensors format allows.
    """

    def __init__(self, path: str | os.PathLike, *, reserve: int, metadata: Mapping[str, str] | None = None):
        self.path = Path(path)
        self.metadata = dict(metadata) if metadata else None
        self._reserve = math.ceil(reserve / 8) * 8
        self._entries: dict[str, dict] = {}
        self._offset = 0
        self._file = open(self.path, "wb")
        self._file.seek(8 + self._reserve)

    @property
    def nbytes(self) -> int:
        return self._offset

    @property
    def keys(self) -> list[str]:
        return list(self._entries)

    def add(self, key: str, tensor: torch.Tensor) -> None:
        if key in self._entries:
            raise ValueError(f"duplicate tensor name {key!r}")
        if tensor.dtype not in SAFETENSORS_DTYPES:
            raise ValueError(f"{key}: safetensors can't store {tensor.dtype}")
        data = _tensor_bytes(tensor)
        self._file.write(data)
        end = self._offset + data.nbytes
        self._entries[key] = {
            "dtype": SAFETENSORS_DTYPES[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [self._offset, end],
        }
        self._offset = end

    def close(self) -> None:
        header: dict = {"__metadata__": self.metadata} if self.metadata else {}
        header.update(self._entries)
        raw = json.dumps(header, separators=(",", ":")).encode()
        if len(raw) > self._reserve:
            self._file.close()
            raise RuntimeError(f"safetensors header is {len(raw)} bytes, more than the {self._reserve} reserved")
        self._file.seek(0)
        self._file.write(struct.pack("<Q", self._reserve))
        self._file.write(raw + b" " * (self._reserve - len(raw)))
        self._file.close()

    def abort(self) -> None:
        self._file.close()


def parse_size(text: str) -> int:
    """Parse sizes like ``5GB``, ``500MB`` or ``2GiB`` into bytes."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)(I?)B?\s*", text.upper())
    if not m:
        raise ValueError(f"can't read {text!r} as a size; use something like 5GB or 500MB")
    number, unit, binary = m.groups()
    base = 1024 if binary else 1000
    return int(float(number) * base ** " KMGT".index(unit or " "))


class ShardedWriter:
    """Writes tensors into numbered shards of at most ``max_bytes`` plus a Hugging Face style
    ``<name>.safetensors.index.json``. If everything fits in one shard, writes a single file."""

    def __init__(self, path: str | os.PathLike, *, max_bytes: int, reserve: int, metadata: Mapping[str, str] | None):
        self.path = Path(path)  # the logical file, e.g. out/model.safetensors
        self.max_bytes = max_bytes
        self.reserve = reserve
        self.metadata = metadata
        self._done: list[SafetensorsWriter] = []
        self._current: SafetensorsWriter | None = None
        self.files: list[Path] = []

    def _tmp(self, i: int) -> Path:
        return self.path.with_name(f"{self.path.stem}.part{i}.tmp")

    def add(self, key: str, tensor: torch.Tensor) -> None:
        size = tensor.numel() * tensor.element_size()
        if self._current is not None and self._current.nbytes and self._current.nbytes + size > self.max_bytes:
            self._current.close()
            self._done.append(self._current)
            self._current = None
        if self._current is None:
            self._current = SafetensorsWriter(self._tmp(len(self._done)), reserve=self.reserve, metadata=self.metadata)
        self._current.add(key, tensor)

    @property
    def nbytes(self) -> int:
        return sum(w.nbytes for w in self._done) + (self._current.nbytes if self._current else 0)

    def close(self) -> list[Path]:
        if self._current is not None:
            self._current.close()
            self._done.append(self._current)
            self._current = None
        if not self._done:
            self._done.append(SafetensorsWriter(self._tmp(0), reserve=self.reserve, metadata=self.metadata))
            self._done[0].close()
        if len(self._done) == 1:
            os.replace(self._done[0].path, self.path)
            self.files = [self.path]
            return self.files
        total = len(self._done)
        weight_map = {}
        for i, writer in enumerate(self._done, start=1):
            name = f"{self.path.stem}-{i:05d}-of-{total:05d}.safetensors"
            os.replace(writer.path, self.path.with_name(name))
            self.files.append(self.path.with_name(name))
            weight_map.update(dict.fromkeys(writer.keys, name))
        index = {"metadata": {"total_size": self.nbytes}, "weight_map": weight_map}
        index_path = self.path.with_name(self.path.name + ".index.json")
        index_path.write_text(json.dumps(index, indent=2) + "\n")
        self.files.append(index_path)
        return self.files

    def abort(self) -> None:
        for writer in self._done + ([self._current] if self._current else []):
            writer.abort()
            writer.path.unlink(missing_ok=True)
