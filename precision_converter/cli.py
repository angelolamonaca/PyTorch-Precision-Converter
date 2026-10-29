"""Command-line interface: ``precision-convert``."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

import safetensors
import torch

from . import __version__
from .core import (
    CONVERSION_TYPES,
    PRECISIONS,
    ConversionReport,
    clean_metadata,
    describe,
    iter_convert,
    output_size_bound,
    write_safetensors,
)
from .folder import convert_folder
from .fp8 import reference_layers
from .io import (
    RenamedMapping,
    ShardedSafetensors,
    UnsafeCheckpointError,
    infer_renames,
    is_index_file,
    open_checkpoint,
    parse_size,
    read_header,
)

REPO_URL = "https://github.com/angelolamonaca/PyTorch-Precision-Converter"
STAR_LINE = f"⭐ If precision-convert saved you time, a GitHub star helps others find it: {REPO_URL}"

EXAMPLES = f"""\
examples:
  precision-convert model.safetensors -p fp16                  halve the size
  precision-convert model.ckpt -p fp16 -t ema-only -st         SD training checkpoint -> EMA weights, fp16
  precision-convert flux.safetensors -p fp8_scaled             ComfyUI fp8_scaled (per-tensor scales)
  precision-convert finetune.safetensors -p fp8_scaled --like https://huggingface.co/.../official_fp8_scaled.safetensors
                                                               same layers as an official fp8_scaled release
  precision-convert DeepSeek-V3/ -p bf16                       FP8 Hugging Face model folder -> BF16, on CPU
  precision-convert model.safetensors.index.json -o model.safetensors       merge shards into one file
  precision-convert model.safetensors -o out/model.safetensors --max-shard-size 5GB   split into shards
  precision-convert model.safetensors -p fp8_e4m3fn --max-error 0.05 --fallback bf16
                                                               fp8, but keep layers that lose >5% in bf16
  precision-convert model.safetensors -p fp8_scaled --report   show which layers lose the most precision
  precision-convert model.ckpt --info                          what's inside (dtypes, EMA, FP8 formats)

exit codes: 0 success, 1 the conversion failed, 2 invalid arguments
docs: {REPO_URL}
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="precision-convert",
        description="Convert PyTorch checkpoints, safetensors files and Hugging Face model folders to "
        "fp32, fp16, bf16 or fp8; extract EMA weights; dequantize scaled FP8; merge or split shards.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="A .safetensors file, a PyTorch checkpoint (.ckpt, .pt, .pth, .bin), a *.safetensors.index.json "
        "shard index, or a model folder.",
    )
    parser.add_argument("-f", "--file", help="Same as INPUT, kept for compatibility with 1.x.")
    parser.add_argument(
        "-p",
        "--precision",
        default="keep",
        choices=PRECISIONS,
        help="Target precision for floating-point tensors (default: keep, which changes no dtype). fp8_scaled "
        "stores Linear and Embedding weights as fp8_e4m3fn with a per-tensor scale, in ComfyUI's format.",
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
        "-st",
        "--safe-tensors",
        action="store_true",
        help="Save in safetensors format. Default: safetensors for safetensors input, .ckpt otherwise.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path. Its extension picks the format (.safetensors or anything else for PyTorch). For a "
        "folder, the output folder. Default: next to the input, named after the type and precision.",
    )
    parser.add_argument(
        "--keep-original",
        metavar="PATTERN",
        action="append",
        default=[],
        help='Leave tensors whose key matches this glob in their original dtype, e.g. "first_stage_model.*". '
        "Repeatable.",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        metavar="E",
        help="Automatic mixed precision: store a tensor at --fallback precision instead when converting it "
        "would change it by more than E (relative error, e.g. 0.05 for 5%%).",
    )
    parser.add_argument(
        "--fallback",
        default="keep",
        choices=[p for p in PRECISIONS if p != "fp8_scaled"],
        help="Precision for tensors rejected by --max-error and, with fp8_scaled, for tensors that can't be "
        "stored as scaled FP8 (norms, biases, convolutions). Default: keep their dtype.",
    )
    parser.add_argument(
        "--like",
        metavar="REFERENCE",
        help="With fp8_scaled: quantize exactly the layers that this quantized file quantizes, e.g. Comfy-Org's "
        "official fp8_scaled or int8 file of the same model. A local path or an https URL; only the header is read.",
    )
    parser.add_argument(
        "--rename",
        metavar="OLD=NEW",
        action="append",
        default=[],
        help='Replace a tensor name prefix, e.g. "model.language_model.=model.". Repeatable. With --like, '
        "prefixes that differ from the reference are found automatically.",
    )
    parser.add_argument("--report", action="store_true", help="Measure and list the tensors that lose the most precision.")
    parser.add_argument(
        "--max-shard-size",
        metavar="SIZE",
        help="Split safetensors output into shards of at most SIZE (e.g. 5GB) with a Hugging Face style index.",
    )
    parser.add_argument(
        "--skip-files",
        metavar="GLOB",
        action="append",
        default=[],
        help='Folder input only: copy weight files matching this glob unchanged, e.g. "vae/*". Repeatable.',
    )
    parser.add_argument("--key", help="Read the state dict from this top-level key of a PyTorch checkpoint.")
    parser.add_argument("--info", action="store_true", help="Describe the input file and exit without converting.")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON on stdout, for scripts and agents.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only print warnings and errors.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the output if it already exists.")
    parser.add_argument(
        "--unsafe-load",
        action="store_true",
        help="Load PyTorch checkpoints with full unpickling (weights_only=False). Unpickling can run code "
        "hidden in the file: only use this for files you trust.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="List the tensors that were clamped or kept back.")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__} (torch {torch.__version__}, safetensors {safetensors.__version__})",
    )
    return parser


def plural(n: int, noun: str) -> str:
    """``plural(1, "tensor")`` -> "1 tensor", ``plural(3, "tensor")`` -> "3 tensors"."""
    return f"{n:,} {noun}" if n == 1 else f"{n:,} {noun[:-1] + 'ies' if noun.endswith('y') else noun + 's'}"


def format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1000
    raise AssertionError("unreachable")


def default_output(input_path: Path, conversion: str, precision: str, safetensors: bool) -> Path:
    stem = input_path.name[: -len(".safetensors.index.json")] if is_index_file(input_path) else input_path.stem
    parts = [stem, conversion] + ([precision] if precision != "keep" else [])
    return input_path.with_name("-".join(parts) + (".safetensors" if safetensors else ".ckpt"))


def default_folder_output(input_path: Path, conversion: str, precision: str) -> Path:
    parts = ([precision] if precision != "keep" else []) + ([conversion] if conversion != "full" else [])
    return input_path.with_name(f"{input_path.name}-{'-'.join(parts) or 'converted'}")


class Console:
    """Human-readable output on stdout, warnings on stderr; JSON mode prints one object at the end."""

    def __init__(self, json_mode: bool, quiet: bool):
        self.json_mode = json_mode
        self.quiet = quiet or json_mode

    def say(self, text: str = "") -> None:
        if not self.quiet:
            print(text)

    def warn(self, text: str) -> None:
        print(f"warning: {text}", file=sys.stderr)

    def error(self, text: str) -> int:
        if self.json_mode:
            print(json.dumps({"ok": False, "error": text}))
        print(f"error: {text}", file=sys.stderr)
        return 1


def summarize(report: ConversionReport, args: argparse.Namespace, console: Console) -> None:
    if args.precision == "keep":
        console.say(f"Kept every dtype as it was ({plural(report.unchanged, 'entry')})")
    else:
        console.say(f"Converted {plural(report.converted, 'tensor')} to {args.precision}, left {report.unchanged:,} unchanged")
    if report.dropped:
        console.say(f"Dropped {plural(report.dropped, 'EMA entry')}")
    if report.dequantized:
        console.say(f"Dequantized {plural(report.dequantized, 'scaled FP8 weight')}")
    if report.scaled:
        console.say(f"Stored {plural(report.scaled, 'weight')} as scaled FP8 (ComfyUI format, needs ComfyUI v0.4.0+)")
    if args.type == "ema-only":
        if report.ema_swapped:
            console.say(f"Used the EMA weights for {plural(report.ema_swapped, 'tensor')}")
        else:
            console.warn("no EMA weights found; the output keeps the regular weights")
    if report.fallback:
        where = "their original dtype" if args.fallback == "keep" else args.fallback
        console.say(f"Kept {plural(len(report.fallback), 'tensor')} at {where}: converting them would change them by more "
                    f"than {args.max_error:g}")
        if args.verbose:
            for key in report.fallback:
                console.say(f"  {key}  ({report.errors.get(key, float('nan')):.4f})")
    if report.clamped:
        limit = torch.finfo(PRECISIONS[args.precision]).max
        console.warn(
            f"{plural(len(report.clamped), 'tensor')} had values outside the {args.precision} range (±{limit:g}); they "
            "were clamped instead of becoming inf/NaN"
        )
        if args.verbose:
            for key, count in report.clamped.items():
                print(f"  {key}: {count:,} values", file=sys.stderr)
    if args.report and report.errors:
        errors = sorted(report.errors.items(), key=lambda item: -item[1])
        values = [e for _, e in errors]
        console.say(
            f"Precision loss over {plural(len(values), 'converted tensor')} (relative error): "
            f"mean {statistics.fmean(values):.4f}, median {statistics.median(values):.4f}, max {values[0]:.4f}"
        )
        console.say("Largest:")
        for key, error in errors[:10]:
            console.say(f"  {error:.4f}  {key}")


def report_json(report: ConversionReport) -> dict:
    data = asdict(report)
    errors = sorted(report.errors.items(), key=lambda item: -item[1])
    data["errors"] = (
        {
            "mean": statistics.fmean(e for _, e in errors),
            "max": errors[0][1],
            "largest": [{"key": k, "error": e} for k, e in errors[:20]],
        }
        if errors
        else None
    )
    return data


def print_info(path: Path, checkpoint, console: Console) -> None:
    info = describe(checkpoint.tensors, checkpoint.metadata)
    if console.json_mode:
        data = asdict(info)
        data.update(
            input=str(path),
            metadata=checkpoint.metadata,
            nested_key=checkpoint.nested_key,
            shards=len(checkpoint.tensors.shard_names) if isinstance(checkpoint.tensors, ShardedSafetensors) else None,
        )
        print(json.dumps({"ok": True, **data}, indent=2))
        return
    print(f"{path}: {info.tensors:,} tensors, {info.parameters:,} values, {format_bytes(info.bytes)}")
    if isinstance(checkpoint.tensors, ShardedSafetensors):
        print(f"  Shards: {len(checkpoint.tensors.shard_names)}")
    if checkpoint.nested_key:
        print(f"  Weights stored under the {checkpoint.nested_key!r} key")
    for dtype, (count, nbytes) in sorted(info.dtypes.items(), key=lambda item: -item[1][1]):
        print(f"  {dtype:<16} {count:>7,} {'tensor ' if count == 1 else 'tensors'}  {format_bytes(nbytes):>10}")
    for name in info.formats:
        print(f"  Format: {name}")
    if info.ema_tensors:
        print(f"  EMA: {info.ema_tensors:,} model_ema.* tensors (-t ema-only or -t no-ema removes them)")
    if info.other_keys:
        print(f"  Non-tensor keys: {', '.join(info.other_keys)}")
    metadata = checkpoint.metadata
    if metadata:
        names = sorted(metadata)
        print(f"  Metadata: {len(metadata)} entries ({', '.join(names[:8])}{', ...' if len(names) > 8 else ''})")


def run_folder(args: argparse.Namespace, parser: argparse.ArgumentParser, input_path: Path, console: Console) -> int:
    for flag, name in ((args.safe_tensors, "--safe-tensors"), (args.max_shard_size, "--max-shard-size"),
                       (args.key, "--key"), (args.info, "--info")):
        if flag:
            parser.error(f"{name} works on files, not folders")
    output = Path(args.output) if args.output else default_folder_output(input_path, args.type, args.precision)
    if output.exists() and not args.overwrite:
        parser.error(f"{output} already exists; pass --overwrite to replace it")
    try:
        result = convert_folder(
            input_path, output, precision=args.precision, conversion=args.type,
            keep_original=args.keep_original, max_error=args.max_error, fallback=args.fallback,
            measure=args.report, skip_files=args.skip_files, quantize_only=args.like_layers, unsafe=args.unsafe_load,
            progress=lambda name: console.say(f"Converting {name}"),
        )
    except (UnsafeCheckpointError, ValueError, OSError, RuntimeError) as e:
        return console.error(str(e))
    if not result.converted_files:
        console.warn("no weight files found in the folder; everything was copied")
    summarize(result.report, args, console)
    if result.updated_configs:
        console.say(f"Updated {', '.join(result.updated_configs)}")
    ratio = f", {result.output_bytes / result.input_bytes:.0%} of {format_bytes(result.input_bytes)}" if result.input_bytes else ""
    console.say(f"Saved {output}/ ({plural(len(result.converted_files), 'weight file')}, {format_bytes(result.output_bytes)}{ratio}; "
                f"{plural(len(result.copied_files), 'other file')} copied)")
    if console.json_mode:
        data = {"ok": True, "input": str(input_path), "output": str(output), "precision": args.precision,
                "type": args.type, **report_json(result.report), "converted_files": result.converted_files,
                "copied_files": result.copied_files, "updated_configs": result.updated_configs,
                "input_bytes": result.input_bytes, "output_bytes": result.output_bytes}
        print(json.dumps(data, indent=2))
    elif not args.quiet:
        print(STAR_LINE, file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console(args.json, args.quiet)

    if args.input and args.file:
        parser.error("give the input either as INPUT or with -f, not both")
    input_path = Path(args.input or args.file or "model.ckpt")
    if not input_path.exists():
        parser.error(f"input not found: {input_path}")
    if args.max_error is not None and not 0 < args.max_error:
        parser.error("--max-error must be a positive number, e.g. 0.05")
    max_shard_bytes = None
    if args.max_shard_size:
        try:
            max_shard_bytes = parse_size(args.max_shard_size)
        except ValueError as e:
            parser.error(str(e))
    renames = {}
    for rule in args.rename:
        old, sep, new = rule.partition("=")
        if not sep or not old:
            parser.error(f"--rename takes OLD=NEW, got {rule!r}")
        renames[old] = new
    args.like_layers = None
    reference = None
    if args.like:
        if args.precision != "fp8_scaled":
            parser.error("--like only works with -p fp8_scaled")
        try:
            reference = read_header(args.like)
            args.like_layers = reference_layers(reference)
        except (OSError, ValueError) as e:
            return console.error(f"can't read the header of {args.like}: {e}")
        if not args.like_layers:
            return console.error(f"{args.like} has no quantized layers to copy")
    if input_path.is_dir():
        if renames:
            parser.error("--rename works on files, not folders")
        return run_folder(args, parser, input_path, console)
    if args.skip_files:
        parser.error("--skip-files only works with a folder as input")

    try:
        checkpoint = open_checkpoint(input_path, unsafe=args.unsafe_load, key=args.key)
    except (UnsafeCheckpointError, ValueError, OSError, RuntimeError) as e:
        return console.error(str(e))
    if args.info:
        print_info(input_path, checkpoint, console)
        return 0

    sources = (
        [input_path.parent / name for name in checkpoint.tensors.shard_names]
        if isinstance(checkpoint.tensors, ShardedSafetensors)
        else [input_path]
    )
    input_is_safetensors = input_path.suffix == ".safetensors" or is_index_file(input_path)
    if args.output:
        output_path = Path(args.output)
        if args.safe_tensors and output_path.suffix != ".safetensors":
            parser.error("--safe-tensors needs an --output path ending in .safetensors")
    else:
        output_path = default_output(input_path, args.type, args.precision, args.safe_tensors or input_is_safetensors)
    as_safetensors = output_path.suffix == ".safetensors"
    if max_shard_bytes and not as_safetensors:
        parser.error("--max-shard-size needs safetensors output")
    if args.precision in ("fp8_scaled", "fp8_e4m3fn", "fp8_e5m2") and not as_safetensors:
        console.warn("fp8 weights in a PyTorch pickle are unusual; most tools expect fp8 models as .safetensors")
    if any(output_path.resolve() == s.resolve() for s in sources):
        parser.error("the output would replace the input file; choose another --output")
    index_path = output_path.with_name(output_path.name + ".index.json")
    if (output_path.exists() or (max_shard_bytes and index_path.exists())) and not args.overwrite:
        parser.error(f"{output_path} already exists; pass --overwrite to replace it")
    if checkpoint.nested_key and checkpoint.nested_key != "state_dict":
        console.say(f"Using the weights under the {checkpoint.nested_key!r} key")
    if reference is not None and not renames:
        renames = infer_renames(checkpoint.tensors, reference)
        for old, new in renames.items():
            console.say(f"Renaming {old!r} to {new!r} to match the reference")
    if renames:
        try:
            checkpoint.tensors = RenamedMapping(checkpoint.tensors, renames)
        except ValueError as e:
            return console.error(str(e))
    if args.like_layers is not None:
        found = args.like_layers & set(checkpoint.tensors)
        if not found:
            example = next(iter(sorted(args.like_layers)))
            return console.error(
                f"none of the {len(args.like_layers):,} layers quantized in {args.like} exist in {input_path}; "
                f"the tensor names differ (the reference has e.g. {example!r})"
            )
        missing = len(args.like_layers) - len(found)
        console.say(f"Copying the layer choice of {args.like}: {plural(len(found), 'layer')} to quantize")
        if missing:
            console.warn(f"{plural(missing, 'layer')} quantized in the reference {'is' if missing == 1 else 'are'} not in the input")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = ConversionReport()
    entries = iter_convert(
        checkpoint.tensors, args.precision, args.type, args.keep_original,
        max_error=args.max_error, fallback=args.fallback, measure=args.report,
        quantize_only=args.like_layers, report=report,
    )
    try:
        if as_safetensors:
            reserve = output_size_bound(checkpoint.tensors, list(checkpoint.tensors), args.precision, checkpoint.metadata)
            files, _, skipped = write_safetensors(
                entries, output_path, reserve=reserve,
                metadata=clean_metadata(checkpoint.metadata, args.precision), max_shard_bytes=max_shard_bytes,
            )
            if skipped:
                console.warn(f"safetensors only stores tensors, left out: {', '.join(skipped)}")
        else:
            data = dict(entries)
            wrap = output_path.suffix == ".ckpt" or checkpoint.wrapped
            tmp = output_path.with_name(output_path.name + ".tmp")
            try:
                torch.save({"state_dict": data} if wrap else data, tmp)
                tmp.replace(output_path)
            finally:
                tmp.unlink(missing_ok=True)
            files = [output_path]
    except (ValueError, OSError, RuntimeError) as e:
        return console.error(str(e))

    summarize(report, args, console)
    before = sum(s.stat().st_size for s in sources)
    after = sum(f.stat().st_size for f in files)
    shards = [f for f in files if f.suffix == ".safetensors"]
    where = f"{output_path}" if len(shards) <= 1 else f"{len(shards)} shards + {index_path.name} in {output_path.parent}"
    console.say(f"Saved {where} ({format_bytes(after)}, {after / before:.0%} of {format_bytes(before)})")
    if console.json_mode:
        data = {"ok": True, "input": str(input_path), "output": str(output_path), "files": [str(f) for f in files],
                "precision": args.precision, "type": args.type, **report_json(report),
                "input_bytes": before, "output_bytes": after}
        print(json.dumps(data, indent=2))
    elif not args.quiet:
        print(STAR_LINE, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
