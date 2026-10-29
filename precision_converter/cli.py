"""Command-line interface: ``precision-convert``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from . import __version__
from .core import (
    CONVERSION_TYPES,
    PRECISIONS,
    UnsafeCheckpointError,
    convert_state_dict,
    describe,
    load_checkpoint,
    save_checkpoint,
)

EXAMPLES = """\
examples:
  precision-convert model.safetensors -p fp16 -st
  precision-convert sd-model.ckpt -p fp16 -t ema-only -st
  precision-convert lora.safetensors -p bf16 -o lora-bf16.safetensors
  precision-convert flux.safetensors -p fp8_e4m3fn -st --keep-original "*norm*"
  precision-convert model.ckpt --info
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="precision-convert",
        description="Convert the tensor precision of PyTorch checkpoints and safetensors files.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input", nargs="?", help="Model file: .safetensors or a PyTorch checkpoint (.ckpt, .pt, .pth, .bin)."
    )
    parser.add_argument("-f", "--file", help="Same as INPUT, kept for compatibility with 1.x.")
    parser.add_argument(
        "-p",
        "--precision",
        default="keep",
        choices=PRECISIONS,
        help="Target precision for floating-point tensors (default: keep, which changes no dtype).",
    )
    parser.add_argument(
        "-t",
        "--type",
        default="full",
        choices=CONVERSION_TYPES,
        help="full: keep everything (default). ema-only: use the EMA weights and drop the EMA copies. "
        "no-ema: drop the EMA copies.",
    )
    parser.add_argument(
        "-st", "--safe-tensors", action="store_true", help="Save in safetensors format (default: PyTorch .ckpt)."
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path. Its extension picks the format (.safetensors or anything else for PyTorch). "
        "Default: <input>-<type>-<precision> next to the input.",
    )
    parser.add_argument(
        "--keep-original",
        metavar="PATTERN",
        action="append",
        default=[],
        help='Leave tensors whose key matches this glob in their original dtype, e.g. "first_stage_model.*". '
        "Repeatable.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the output file if it already exists.")
    parser.add_argument(
        "--unsafe-load",
        action="store_true",
        help="Load PyTorch checkpoints with full unpickling (weights_only=False). Unpickling can run code "
        "hidden in the file: only use this for files you trust.",
    )
    parser.add_argument("--info", action="store_true", help="Describe the input file and exit without converting.")
    parser.add_argument("-v", "--verbose", action="store_true", help="List the tensors whose values were clamped.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1000
    raise AssertionError("unreachable")


def default_output(input_path: Path, conversion: str, precision: str, safetensors: bool) -> Path:
    parts = [input_path.stem, conversion] + ([precision] if precision != "keep" else [])
    return input_path.with_name("-".join(parts) + (".safetensors" if safetensors else ".ckpt"))


def print_info(path: Path, state_dict: dict, metadata: dict | None) -> None:
    info = describe(state_dict)
    print(f"{path}: {info.tensors:,} tensors, {info.parameters:,} values, {format_bytes(info.bytes)}")
    for dtype, (count, nbytes) in sorted(info.dtypes.items(), key=lambda item: -item[1][1]):
        print(f"  {dtype:<16} {count:>7,} {'tensor ' if count == 1 else 'tensors'}  {format_bytes(nbytes):>10}")
    if info.ema_tensors:
        print(f"  EMA: {info.ema_tensors:,} model_ema.* tensors (-t ema-only or -t no-ema removes them)")
    if info.other_keys:
        print(f"  Non-tensor keys: {', '.join(info.other_keys)}")
    if metadata:
        print(f"  Metadata: {len(metadata)} entries ({', '.join(sorted(metadata)[:8])}{', ...' if len(metadata) > 8 else ''})")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.input and args.file:
        parser.error("give the input file either as INPUT or with -f, not both")
    input_path = Path(args.input or args.file or "model.ckpt")
    if not input_path.is_file():
        parser.error(f"input file not found: {input_path}")

    if args.output:
        output_path = Path(args.output)
        if args.safe_tensors and output_path.suffix != ".safetensors":
            parser.error("--safe-tensors needs an --output path ending in .safetensors")
    else:
        output_path = default_output(input_path, args.type, args.precision, args.safe_tensors)

    try:
        state_dict, metadata = load_checkpoint(input_path, unsafe=args.unsafe_load)
    except (UnsafeCheckpointError, ValueError, OSError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.info:
        print_info(input_path, state_dict, metadata)
        return 0

    if output_path.resolve() == input_path.resolve():
        parser.error("the output would replace the input file; choose another --output")
    if output_path.exists() and not args.overwrite:
        parser.error(f"{output_path} already exists; pass --overwrite to replace it")

    converted, report = convert_state_dict(state_dict, args.precision, args.type, args.keep_original)

    if args.precision == "keep":
        summary = f"Kept every dtype as it was ({report.unchanged:,} entries)"
    else:
        summary = f"Converted {report.converted:,} tensors to {args.precision}, left {report.unchanged:,} unchanged"
    if report.dropped:
        summary += f", dropped {report.dropped:,} EMA entries"
    print(summary)
    if args.type == "ema-only":
        if report.ema_swapped:
            print(f"Used the EMA weights for {report.ema_swapped:,} tensors")
        else:
            print("warning: no EMA weights found; the output keeps the regular weights", file=sys.stderr)
    if report.clamped:
        limit = torch.finfo(PRECISIONS[args.precision]).max
        print(
            f"warning: {len(report.clamped):,} tensors had values outside the {args.precision} range "
            f"(±{limit:g}); they were clamped instead of becoming inf/NaN",
            file=sys.stderr,
        )
        if args.verbose:
            for key, count in report.clamped.items():
                print(f"  {key}: {count:,} values", file=sys.stderr)

    skipped = save_checkpoint(converted, output_path, metadata=metadata)
    if skipped:
        print(f"warning: safetensors only stores tensors, left out: {', '.join(skipped)}", file=sys.stderr)

    before, after = input_path.stat().st_size, output_path.stat().st_size
    print(f"Saved {output_path} ({format_bytes(after)}, {after / before:.0%} of {format_bytes(before)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
