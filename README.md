# PyTorch Precision Converter

[![Tests](https://github.com/angelolamonaca/PyTorch-Precision-Converter/actions/workflows/ci.yml/badge.svg)](https://github.com/angelolamonaca/PyTorch-Precision-Converter/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/angelolamonaca/PyTorch-Precision-Converter/blob/main/LICENSE)

Make PyTorch checkpoints and `safetensors` files **half the size (a quarter with fp8)** by converting them to
`fp16`, `bf16` or `fp8`, and strip the EMA weights you don't need for inference. One command, no model code
needed, runs on the CPU.

```console
$ precision-convert model.ckpt -p fp16 -t ema-only -st
Converted 280 tensors to fp16, left 1 unchanged, dropped 242 EMA entries
Used the EMA weights for 240 tensors
Saved model-ema-only-fp16.safetensors (29.93 MB, 27% of 109.23 MB)
```

## Why not just call `.half()`?

A one-line cast quietly breaks things. This tool handles the details:

- **Only floating-point tensors are cast.** Integer and bool tensors like `position_ids` or step counters keep
  their dtype instead of turning into floats.
- **No silent `inf` or `NaN`.** fp16 tops out at 65,504 and fp8 at 448. Values beyond the range are clamped
  to it and you get a warning, rather than infinities appearing in your weights.
- **Real EMA extraction.** `-t ema-only` puts the EMA weights where the model loads them from, for Stable
  Diffusion and other latent-diffusion checkpoints (`model_ema.*`), then drops the duplicates.
- **safetensors metadata survives.** LoRA training info (`ss_*` keys) and other metadata is copied to the output.
- **Safe by default.** PyTorch checkpoints load with `weights_only=True`, so a malicious `.ckpt` can't run code.
  Files that need full unpickling require an explicit `--unsafe-load`.
- **Your files are safe too.** Output is written to a temporary file and moved into place when complete, the
  input is never replaced, and existing files are only overwritten with `--overwrite`.
- **Mixed precision when you need it.** `--keep-original "first_stage_model.*"` keeps matching tensors (here the
  VAE) in their original dtype.

## Install

```bash
pip install git+https://github.com/angelolamonaca/PyTorch-Precision-Converter
```

This installs the `precision-convert` command (`python -m precision_converter` works too). It needs Python 3.9+,
PyTorch 2.1+ and `safetensors`.

## Usage

```bash
precision-convert INPUT [-p PRECISION] [-t TYPE] [-st] [-o OUTPUT] [options]
```

| Option | What it does |
|---|---|
| `INPUT` | `.safetensors` file or PyTorch checkpoint (`.ckpt`, `.pt`, `.pth`, `.bin`). `-f INPUT` works too. |
| `-p`, `--precision` | `fp32`, `fp16`, `bf16`, `fp8_e4m3fn`, `fp8_e5m2`, or `keep` (default: change no dtypes). |
| `-t`, `--type` | `full` keeps everything (default). `ema-only` uses the EMA weights. `no-ema` drops the EMA weights. |
| `-st`, `--safe-tensors` | Save as `.safetensors` instead of a PyTorch `.ckpt`. |
| `-o`, `--output` | Output path; its extension picks the format. Default: `<input>-<type>-<precision>` next to the input. |
| `--keep-original PATTERN` | Leave tensors whose key matches the glob in their original dtype. Repeatable. |
| `--info` | Show the tensors, dtypes, sizes, EMA and metadata in the file, then exit. |
| `--unsafe-load` | Fully unpickle PyTorch checkpoints. Only for files you trust. |
| `--overwrite` | Replace the output file if it exists. |
| `-v`, `--verbose` | List each tensor that had values clamped. |

## Recipes

**Shrink a Stable Diffusion training checkpoint for inference** (EMA weights, fp16, safetensors):

```bash
precision-convert model.ckpt -p fp16 -t ema-only -st
```

**Convert a LoRA to fp16 and keep its training metadata:**

```bash
precision-convert my-lora.safetensors -p fp16 -o my-lora-fp16.safetensors
```

**Store a large model in fp8** for UIs that load fp8 weights, such as ComfyUI, keeping normalization layers
precise:

```bash
precision-convert model.safetensors -p fp8_e4m3fn -st --keep-original "*norm*"
```

**Keep the VAE in fp32** (fp16 VAEs can produce NaNs and black images in some models):

```bash
precision-convert model.safetensors -p fp16 -st --keep-original "first_stage_model.*"
```

**Turn a pickled checkpoint into safetensors without changing precision:**

```bash
precision-convert model.ckpt -st
```

**Check what's inside a file before converting:**

```console
$ precision-convert model.ckpt --info
model.ckpt: 523 tensors, 27,274,319 values, 109.10 MB
  float32              521 tensors   109.10 MB
  int64                  1 tensor        616 B
  int32                  1 tensor          4 B
  EMA: 242 model_ema.* tensors (-t ema-only or -t no-ema removes them)
```

## Which precision should I pick?

| Precision | Bytes per value | Largest value | Good for |
|---|---|---|---|
| `fp32` | 4 | 3.4 × 10³⁸ | Training, reference copies, upcasting half-precision files |
| `bf16` | 2 | 3.4 × 10³⁸ | The safe 2-byte choice: same range as fp32, so nothing overflows |
| `fp16` | 2 | 65,504 | GPUs without fast bf16, and tools that expect fp16 |
| `fp8_e4m3fn` | 1 | 448 | Smallest files, for inference in tools that load fp8 weights; some quality loss |
| `fp8_e5m2` | 1 | 57,344 | fp8 with more range and less precision than `e4m3fn` |

The fp8 options cast each value directly, without per-tensor scaling factors.

## Python API

```python
from precision_converter import convert_state_dict, load_checkpoint, save_checkpoint

state_dict, metadata = load_checkpoint("model.safetensors")
converted, report = convert_state_dict(state_dict, precision="bf16", conversion="no-ema")
save_checkpoint(converted, "model-bf16.safetensors", metadata=metadata)

print(f"{report.converted} tensors converted, {len(report.clamped)} had values clamped")
```

`convert_state_dict` takes any mapping of tensors, so `model.state_dict()` works too.

## Upgrading from 1.x

`python convert.py ...` still works, but 2.0 fixes bugs that changed what earlier versions wrote:

- `-t ema-only` used to keep the regular weights and store the EMA weights under keys no model loads. It now
  outputs the actual EMA weights.
- Integer tensors are no longer cast to floating point, and overflowing values are clamped instead of
  becoming `inf`.
- The default precision is now `keep`. `-p fp32` used to change nothing; it now converts fp16 and bf16 tensors
  to fp32. With the default precision the output name no longer ends in `-fp32`.
- Checkpoints are loaded with `weights_only=True`. Use `--unsafe-load` for trusted files that need more.
- Existing output files are no longer overwritten unless you pass `--overwrite`.

See [CHANGELOG.md](https://github.com/angelolamonaca/PyTorch-Precision-Converter/blob/main/CHANGELOG.md) for details.

## Contributing

Issues and pull requests are welcome. To run the tests:

```bash
pip install -e ".[test]"
pytest
```

## License

MIT. See [LICENSE](https://github.com/angelolamonaca/PyTorch-Precision-Converter/blob/main/LICENSE).
