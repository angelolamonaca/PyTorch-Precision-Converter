# Changelog

## 2.2.0

### Added
- `-p int8_convrot`: ComfyUI's int8 ConvRot format (`int8_tensorwise` with `convrot`, ComfyUI v0.27.0+), the
  format of Comfy-Org's `*_int8_convrot.safetensors` releases. Linear weights are rotated by a Hadamard matrix in
  groups of 256 input features and stored as int8 with one FP32 scale per row. Rotation and rounding follow
  comfy-kitchen's reference code: converting Comfy-Org's bf16 Qwen-Image 2.1 with `--like` their int8 file gives
  back that file apart from rounding ties (99.99999% of int8 values identical, scales within 1 float32 ULP).
- `--like` works with `-p int8_convrot` too, to make int8_convrot files of fine-tunes with the official layers.
- ComfyUI int8 input (`int8_tensorwise`, with or without ConvRot) is dequantized, so `-p bf16` turns
  int8_convrot files back into bf16; `--info` recognises them.
- Python API: `quantize_convrot`.

## 2.1.0

### Added
- `-p fp8_scaled`: ComfyUI's per-layer scaled FP8 format (`comfy_quant`, ComfyUI v0.4.0+). Linear weights get a
  per-tensor scale, which keeps small weights accurate where a plain fp8 cast loses them. Embeddings, norms,
  biases and convolutions stay in their original precision.
- `--like REFERENCE` (a path or URL; only the header and two small layers are read) quantizes exactly the layers
  that an official fp8_scaled, int8 or fp8 file quantizes, for fine-tunes of models Comfy-Org has released, scales
  them the way the reference does (most Comfy-Org text encoders map each layer's largest weight to 416 instead of
  448), and renames tensors to the reference's names when they differ by a prefix. `--rename OLD=NEW` renames by
  hand.
- Scaled FP8 input is dequantized before converting: block-scaled FP8 (DeepSeek-V3/R1, Qwen3 FP8),
  compressed-tensors FP8, and all three ComfyUI layouts.
- Model folders as input: every weight file is converted, shard indexes are rewritten, `config.json` gets the new
  `torch_dtype` and loses `quantization_config` after dequantizing, and other files are copied. `--skip-files`
  copies matching weight files unchanged.
- Shard indexes (`*.safetensors.index.json`) as input, to merge shards into one file; `--max-shard-size` splits
  safetensors output into shards with an index.
- `--report` measures each tensor's relative error; `--max-error` with `--fallback` stores the tensors that would
  lose too much at another precision.
- `--key` picks the state dict inside a training checkpoint; `model`, `module`, `model_state_dict` and `net` are
  now found automatically, like `state_dict`.
- `--json` output, `-q/--quiet`, and torch/safetensors versions in `--version`.
- `--info` names the layouts it recognises (scaled FP8 variants, EMA, LoRA) and shows nested keys and shards.
- Python API: `open_checkpoint`, `iter_convert`, `write_safetensors`, `convert_folder`, `quantize_scaled`,
  `dequantize`; the package ships type hints.

### Changed
- safetensors files are read lazily and written one tensor at a time, and PyTorch checkpoints are memory-mapped,
  so memory use stays near one tensor instead of the whole model.
- safetensors input produces safetensors output by default.
- Output to `.pt`, `.pth` or `.bin` stays a plain state dict when the input was one; `.ckpt` output is still
  wrapped in `{"state_dict": ...}`.
- A star request is printed to stderr after a successful conversion; `--quiet` and `--json` leave it out.

### Fixed
- Scaled FP8 weights were cast without applying their scales, so 2.0.0 produced wrong weights for scaled FP8
  input. They are now dequantized.

## 2.0.0

### Fixed
- `-t ema-only` now outputs the EMA weights. 1.x kept the regular weights and saved the EMA weights under
  keys no model loads, so its "EMA-only" files contained the non-EMA weights.
- Only floating-point tensors are converted. 1.x also cast integer tensors such as `position_ids` and
  `num_updates` to floats.
- Values outside the target range are clamped, with a warning, instead of becoming `inf`.
- Saving as safetensors works for checkpoints with tied weights, non-contiguous tensors or non-tensor values.
- Output names are built correctly for paths without an extension or with dots in folder names.

### Changed
- The default precision is `keep`, which changes no dtype. `-p fp32` now converts fp16 and bf16 tensors to
  fp32, where 1.x left them unchanged. Output names leave out the precision when it is `keep`.
- PyTorch checkpoints load with `weights_only=True`. `--unsafe-load` enables full unpickling for trusted files.
- An existing output file is only replaced with `--overwrite`, and the input file is never replaced.
- Output is written to a temporary file and moved into place once complete.

### Added
- `pip install`-able package with a `precision-convert` command and a Python API.
- `fp8_e4m3fn` and `fp8_e5m2` precisions.
- safetensors metadata is copied to safetensors output.
- `--output`, `--keep-original`, `--info`, `--overwrite`, `--verbose` and `--version`.
- Tests and CI.

## 1.0.0

- First release.
