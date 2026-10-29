"""Convert the tensor precision of PyTorch checkpoints and safetensors files."""

__version__ = "2.3.0"

from .core import (  # noqa: E402
    CONVERSION_TYPES,
    PRECISIONS,
    CheckpointInfo,
    ConversionReport,
    convert_state_dict,
    describe,
    iter_convert,
    save_checkpoint,
    write_safetensors,
)
from .folder import FolderReport, convert_folder  # noqa: E402
from .fp8 import dequantize, quantize_scaled  # noqa: E402
from .int8 import quantize as quantize_int8, quantize_convrot  # noqa: E402
from .io import (  # noqa: E402
    Checkpoint,
    SafetensorsFile,
    ShardedSafetensors,
    UnsafeCheckpointError,
    load_checkpoint,
    open_checkpoint,
)

__all__ = [
    "CONVERSION_TYPES",
    "PRECISIONS",
    "Checkpoint",
    "CheckpointInfo",
    "ConversionReport",
    "FolderReport",
    "SafetensorsFile",
    "ShardedSafetensors",
    "UnsafeCheckpointError",
    "convert_folder",
    "convert_state_dict",
    "dequantize",
    "describe",
    "iter_convert",
    "load_checkpoint",
    "open_checkpoint",
    "quantize_convrot",
    "quantize_int8",
    "quantize_scaled",
    "save_checkpoint",
    "write_safetensors",
]
