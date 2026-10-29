# Changelog

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
