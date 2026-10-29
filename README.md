# PyTorch Precision Converter

[![PyPI](https://img.shields.io/pypi/v/pytorch-precision-converter)](https://pypi.org/project/pytorch-precision-converter/)
[![Tests](https://github.com/angelolamonaca/PyTorch-Precision-Converter/actions/workflows/ci.yml/badge.svg)](https://github.com/angelolamonaca/PyTorch-Precision-Converter/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/angelolamonaca/PyTorch-Precision-Converter/blob/main/LICENSE)

Convert PyTorch checkpoints, `safetensors` files and whole Hugging Face model folders to **fp16, bf16, fp8 or
ComfyUI `fp8_scaled`, `int8_convrot` and `int8`**, turn **FP8 models (DeepSeek, Qwen) back into BF16 on a CPU**, merge or
split **sharded safetensors**, and extract **EMA weights**, with one command and without loading the model into RAM.

```bash
pip install pytorch-precision-converter
```

```console
$ precision-convert Qwen3-0.6B-FP8/ -p bf16
Converting model.safetensors
Converted 196 tensors to bf16, left 115 unchanged
Dequantized 196 scaled FP8 weights
Updated config.json
Saved Qwen3-0.6B-FP8-bf16/ (1 weight file, 1.50 GB, 141% of 1.06 GB; 6 other files copied)
```

That output matches Qwen's official BF16 release: the same 311 tensors, 115 of them bit-identical and the 196
dequantized ones within FP8's rounding error (2.7%). It took 4 seconds and 262 MB of RAM on a Mac CPU.

## What it does

| Task | Command |
|---|---|
| [Convert a model to fp16 or bf16](#convert-a-model-to-fp16-or-bf16) | `precision-convert model.safetensors -p fp16` |
| [Make a ComfyUI `fp8_scaled` model](#make-a-comfyui-fp8_scaled-model) | `precision-convert model.safetensors -p fp8_scaled` |
| [fp8_scaled of a fine-tune, like the official release](#make-an-fp8_scaled-file-of-a-fine-tune-like-the-official-release) | `precision-convert finetune.safetensors -p fp8_scaled --like OFFICIAL_URL` |
| [Make a ComfyUI `int8_convrot` model](#make-a-comfyui-int8_convrot-model) | `precision-convert finetune.safetensors -p int8_convrot --like OFFICIAL_URL` |
| [Make a ComfyUI `int8` model for a T4](#make-a-comfyui-int8-model-for-a-t4) | `precision-convert finetune.safetensors -p int8 --like OFFICIAL_URL` |
| [Convert FP8 models to BF16 (DeepSeek, Qwen, ComfyUI)](#convert-fp8-models-to-bf16-deepseek-qwen-comfyui) | `precision-convert DeepSeek-V3/ -p bf16` |
| [Convert a Hugging Face model folder](#convert-a-hugging-face-model-folder) | `precision-convert Llama-3.1-8B/ -p fp16` |
| [Merge sharded safetensors into one file](#merge-sharded-safetensors-into-one-file) | `precision-convert model.safetensors.index.json -o model.safetensors` |
| [Split a safetensors file into shards](#split-a-safetensors-file-into-shards) | `precision-convert model.safetensors -o out/model.safetensors --max-shard-size 5GB` |
| [Convert .ckpt to .safetensors](#convert-ckpt-to-safetensors) | `precision-convert model.ckpt -st` |
| [Extract EMA weights](#extract-ema-weights-from-a-stable-diffusion-checkpoint) | `precision-convert model.ckpt -p fp16 -t ema-only -st` |
| [Find the layers that lose precision](#find-the-layers-that-lose-the-most-precision) | `precision-convert model.safetensors -p fp8_scaled --report` |
| [Automatic mixed precision](#keep-sensitive-layers-in-higher-precision-automatically) | `precision-convert model.safetensors -p fp8_e4m3fn --max-error 0.05 --fallback bf16` |
| [See what's inside a checkpoint](#see-whats-inside-a-checkpoint) | `precision-convert model.ckpt --info` |

It also handles the details a quick script gets wrong:

- **Low memory.** Tensors are read and written one at a time, so converting a 24 GB model doesn't need 24 GB of RAM.
- **Only floating-point tensors are cast.** `position_ids`, step counters and masks keep their integer dtype.
- **No silent `inf` or `NaN`.** Values beyond fp16's 65,504 or fp8's 448 are clamped, with a warning.
- **Metadata survives.** safetensors metadata (LoRA `ss_*` training info, `modelspec.*`) is copied to the output.
- **Safe by default.** `.ckpt`/`.pt` files load with `weights_only=True`, so a malicious pickle can't run code.
- **Your files are safe.** Output goes to a temporary file first, the input is never replaced, and nothing is
  overwritten without `--overwrite`.
- **Runs anywhere.** CPU only: no GPU, CUDA or Triton needed. Linux, macOS and Windows, Python 3.9+, PyTorch 2.1+.

### Tested on real models

| Model | Conversion | Result |
|---|---|---|
| [Qwen/Qwen3-0.6B-FP8](https://huggingface.co/Qwen/Qwen3-0.6B-FP8) (block-scaled FP8) | folder → bf16 | Same tensors as the official BF16 release; 115 bit-identical, 196 within 2.66% |
| [RedHatAI/Qwen3-0.6B-FP8-dynamic](https://huggingface.co/RedHatAI/Qwen3-0.6B-FP8-dynamic) (compressed-tensors) | folder → bf16 | Same tensors as the official BF16 release; 115 bit-identical, 196 within 2.60% |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | → `fp8_scaled` | 1.50 GB → 0.91 GB, 2.65% mean error, embeddings kept in bf16 |
| [Gemma 4 E4B](https://huggingface.co/Comfy-Org/gemma-4) text encoder, Comfy-Org's bf16 | → `fp8_scaled`, `--like` the official fp8_scaled | Every weight and scale bit-identical to the official file; 16.02 GB → 9.06 GB in 56 s, 4.1 GB peak memory |
| [Qwen3-VL 8B](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) text encoder, Comfy-Org's bf16 | → `fp8_scaled`, `--like` the official fp8_scaled ([Boogu-Image](https://huggingface.co/Comfy-Org/Boogu-Image)) | Unquantized tensors and 99.95% of FP8 bytes bit-identical to the official file, the rest rounding ties; 17.53 GB → 10.59 GB in 46 s |
| [Qwen-Image 2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1), Comfy-Org's bf16 | → `fp8_scaled`, `--like` the int8 release | The int8 release's 192 layers in FP8, 2.65% mean error; 14.23 GB → 7.25 GB in 33 s, 0.77 GB peak memory |
| [Qwen-Image 2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1), Comfy-Org's bf16 | → `int8_convrot`, `--like` the official int8_convrot | Same 649 tensors as the official file; every unquantized tensor and `comfy_quant` entry bit-identical, 99.99999% of int8 values identical (the other 644 of 7 billion are rounding ties), scales within 1 float32 ULP; 14.23 GB → 7.26 GB in 33 s, 0.65 GB peak memory |
| [MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image), Comfy-Org's bf16 | → `int8_convrot`, `--like` the official int8_convrot | Same 757 tensors as the official file, including its 42 per-layer int8-attention settings; every unquantized tensor and `comfy_quant` entry bit-identical, 99.99999% of int8 values identical (the other 420 of 4.8 billion are rounding ties), scales within 1 float32 ULP; 10.16 GB → 5.41 GB |
| [MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image), Comfy-Org's bf16 | → `int8`, `--like` the official int8_convrot | The official file's layers and settings; every int8 value and scale identical to comfy-kitchen's own quantizer (`quantize_int8_rowwise` on the FP32 weights); 10.16 GB → 5.41 GB |
| [Llama-3.1-8B-Lexi-Uncensored-V2](https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2), 4 shards | → one `fp8_scaled` file, `--like` Comfy-Org's Llama 3.1 8B | Same 291 tensors, dtypes and shapes as the official file; 16.06 GB → 9.08 GB in 48 s, 0.81 GB peak memory |
| [Huihui-Qwen3-VL-8B-Instruct-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated), 4 shards | → one `fp8_scaled` file, `--like` Comfy-Org's Qwen3-VL 8B | Renamed to ComfyUI's names; same 1,254 tensors, dtypes and shapes as the official file; 17.53 GB → 10.59 GB in 45 s |
| Comfy-Org `fp8_scaled` releases (Krea-2, Flux Krea) | header inspection | Uses the three layouts this tool reads and writes; tests cover each one |

Times and peak memory footprint measured with `/usr/bin/time -l` on an Apple M3 Pro CPU with 18 GB of RAM. The
MOSS-Image files were made on the benchmark machines.

### Benchmarks: which file to load in ComfyUI

Speed, VRAM and image fidelity of bf16, fp8, fp8_scaled, int8_convrot and NVFP4 files on Ampere, Ada and Blackwell
GPUs (RTX 30/40/50-class), with every image, the per-image data and the scripts:
[ComfyUI quantization benchmarks](https://huggingface.co/datasets/anla1234-itachi/comfyui-quant-bench). First model:
MOSS-Image, where Comfy-Org's int8_convrot file was the fastest or close to it and the closest to bf16 on all three.
On a T4, an `int8` file made with this tool beat it ([numbers](#make-a-comfyui-int8-model-for-a-t4)).

## How to

### Convert a model to fp16 or bf16

```bash
precision-convert model.safetensors -p fp16
precision-convert model.safetensors -p bf16 -o model-bf16.safetensors
```

Works the same for `.ckpt`, `.pt`, `.pth` and `.bin` files. Without `-o`, the output goes next to the input as
`model-full-fp16.safetensors`.

### Make a ComfyUI `fp8_scaled` model

```bash
precision-convert flux1-dev.safetensors -p fp8_scaled -o flux1-dev-fp8_scaled.safetensors
```

`fp8_scaled` stores Linear weights as `float8_e4m3fn` with a per-tensor FP32 scale, in the per-layer
`comfy_quant` format [documented by ComfyUI](https://github.com/Comfy-Org/ComfyUI/blob/master/QUANTIZATION.md).
That is the format of Comfy-Org's own `*_fp8_scaled.safetensors` releases; ComfyUI hasn't published the tool it
uses to make them. The file is half the size of bf16 and, unlike a plain fp8 cast, keeps small weights accurate
([numbers below](#why-fp8_scaled-beats-a-plain-fp8-cast)). By default every Linear weight is quantized, while
biases, norms, convolutions and embeddings stay in their original precision (`--fallback bf16` stores those in
bf16).

Put the file in `ComfyUI/models/diffusion_models` (or `text_encoders`) and load it with the default
`weight_dtype`. Don't pick `fp8_e4m3fn` in the loader: the file is already FP8. Needs ComfyUI v0.4.0 or newer.

### Make an fp8_scaled file of a fine-tune, like the official release

Comfy-Org chooses the quantized layers per model by hand. In Gemma 4 E4B, for example, they quantize the MLPs and
a few attention layers but leave the audio and vision towers in bf16. To make an fp8_scaled file of a fine-tune,
uncensored or abliterated version of a model they have released, copy their choice with `--like`:

```bash
precision-convert gemma4_e4b_uncensored.safetensors -p fp8_scaled \
  --like https://huggingface.co/Comfy-Org/gemma-4/resolve/main/text_encoders/gemma4_e4b_it_fp8_scaled.safetensors
```

`--like` takes a local file or a URL and reads only the header and two small layers, so nothing big is downloaded.
The reference can be fp8_scaled, int8 or plain fp8; its quantized layers become fp8_scaled in the output, scaled
the way the reference scales them (most Comfy-Org text encoders map each layer's largest weight to 416 instead of
FP8's maximum, 448). Converting Comfy-Org's own bf16 Gemma 4 E4B this way gives back their fp8_scaled file: every
weight and scale comes out [bit-identical](#tested-on-real-models).

When names differ only by a prefix, such as Hugging Face's `model.language_model.*` against ComfyUI's `model.*`,
they are renamed to the reference's (`--rename OLD=NEW` does the same by hand). A shard index works as input, so a
Hugging Face fine-tune becomes one ComfyUI file:

```bash
precision-convert Huihui-Qwen3-VL-8B-Instruct-abliterated/model.safetensors.index.json -p fp8_scaled \
  --like https://huggingface.co/Comfy-Org/Boogu-Image/resolve/main/text_encoders/qwen3vl_8b_fp8_scaled.safetensors \
  -o qwen3vl_8b_abliterated_fp8_scaled.safetensors
```

### Make a ComfyUI `int8_convrot` model

```bash
precision-convert qwen_image_finetune.safetensors -p int8_convrot \
  --like https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_int8_convrot.safetensors
```

`int8_convrot` is the format Comfy-Org now ships most new models in: each Linear weight is rotated by a Hadamard
matrix in groups of 256 input features, which spreads out outliers, and stored as int8 with one FP32 scale per
row. ComfyUI rotates the activations the same way while it runs, so the result is the same as without rotation. In
our [benchmark](#benchmarks-which-file-to-load-in-comfyui) it was the fastest format or close to it, and the closest
to bf16, on Ampere, Ada and Blackwell GPUs.

With `--like`, the same layers as Comfy-Org's int8 file are quantized, and the rotation and rounding follow
comfy-kitchen's reference code: converting Comfy-Org's own bf16 Qwen-Image 2.1 or MOSS-Image gives back their
int8_convrot file apart from rounding ties ([numbers](#tested-on-real-models)). Without `--like`, every Linear weight whose input size
is a multiple of 256 is quantized. The file needs ComfyUI v0.27.0 or newer (v0.30.0 when embeddings are quantized,
as in Comfy-Org's int8 text encoders).

`--like` also copies the reference's per-layer settings. Comfy-Org's newer int8 files, such as MOSS-Image's, carry
`comfy_attention.config` entries that switch ComfyUI (v0.38.0+) to int8 attention in those layers. Without them, the
same int8 MOSS-Image file took 2.3× longer per image on a T4 and 6–10% longer on A10G, L40S and RTX PRO 6000 GPUs.

### Make a ComfyUI `int8` model for a T4

```bash
precision-convert moss_image_finetune.safetensors -p int8 \
  --like https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/diffusion_models/moss_image_sft_int8_convrot.safetensors
```

`int8` is ComfyUI's int8 format without ConvRot: the same int8 weights with one FP32 scale per row, but no Hadamard
rotation, so ComfyUI doesn't rotate the activations at every layer either. Values and scales are identical to
comfy-kitchen's own quantizer run on the FP32 weights. The file needs ComfyUI v0.27.0 or newer.

On a T4 that rotation costs more than it gains. MOSS-Image at 1024×1024, 25 steps, ComfyUI `0df1646`, on the same
GPUs and prompts (median seconds per image, and median LPIPS against the bf16 image of the same prompt and seed):

| GPU | bf16 | `int8_convrot` (Comfy-Org) | `int8` (this tool) |
|---|---|---|---|
| T4 | 671.7 s | 107.7 s · LPIPS 0.079 | **98.1 s** · 0.076 |
| A10G | 45.0 s | **24.5 s** · 0.069 | 30.1 s · 0.104 |
| L40S | 15.4 s | **7.7 s** · 0.045 | 9.6 s · 0.072 |
| RTX PRO 6000 | 8.45 s | **4.63 s** · 0.067 | 5.40 s · 0.092 |

On the T4, `int8` took 9% less time than `int8_convrot` and stayed about as close to bf16 (median LPIPS). On
Ampere, Ada and Blackwell GPUs it took 17–25% longer and drifted further, so use `int8_convrot` there. A T4 has no
bf16 hardware, so ComfyUI runs MOSS-Image in fp32 on it by default: both int8 files were over 6× faster than that,
and `--fp16-unet` 5.5× (median LPIPS 0.011).

### Convert FP8 models to BF16 (DeepSeek, Qwen, ComfyUI)

```bash
precision-convert DeepSeek-V3.1/ -p bf16
precision-convert Qwen3-8B-FP8/ -p bf16
precision-convert flux1-dev-fp8_scaled.safetensors -p bf16
```

Scaled FP8 weights are multiplied back by their scales, which a plain dtype cast doesn't do. Recognised layouts:

- **Block-scaled FP8** (`weight_scale_inv`, one scale per 128×128 block): DeepSeek-V3, DeepSeek-R1, Qwen3 FP8 and
  other models in that format. Scales stored in a different shard than their weight are found through the index.
- **ComfyUI `fp8_scaled`**: per-layer `comfy_quant`, the `_quantization_metadata` header variant, and the older
  `scaled_fp8` / `scale_weight` format.
- **ComfyUI int8** (`int8_tensorwise`, with or without ConvRot), as in Comfy-Org's `*_int8_convrot.safetensors`.
- **compressed-tensors FP8** (`weight_scale` per tensor or per row), as written by llm-compressor for vLLM.

For a model folder, `config.json` loses its `quantization_config` and gets the new `torch_dtype`, so
Transformers loads the result as a normal BF16 model. Big models are converted shard by shard, so RAM use stays
low, but you need disk space for the BF16 copy (about twice the FP8 size).

### Convert a Hugging Face model folder

```bash
precision-convert Llama-3.1-8B/ -p fp16
precision-convert stable-diffusion-xl-base-1.0/ -p fp16 --skip-files "vae/*"
```

Every weight file in the folder (and its subfolders) is converted into a new folder (`Llama-3.1-8B-fp16/`), keeping
the same file names. Shard indexes are rewritten to match, `config.json` gets the new `torch_dtype`, and
everything else (tokenizers, configs, READMEs) is copied. `--skip-files` copies matching weight files unchanged,
for example to keep a VAE in fp32.

### Merge sharded safetensors into one file

```bash
precision-convert model.safetensors.index.json -o model.safetensors
precision-convert transformer/diffusion_pytorch_model.safetensors.index.json -p bf16 -o transformer.safetensors
```

The shards listed in the index are read as one model and written as one file, optionally converted on the way.
Tensor names are kept as they are.

### Split a safetensors file into shards

```bash
precision-convert model.safetensors -o out/model.safetensors --max-shard-size 5GB
```

Writes `model-00001-of-0000N.safetensors` files plus a `model.safetensors.index.json` in the Hugging Face layout,
for example to upload a large model to the Hub. Sizes accept `GB`, `MB`, `GiB` and so on.

### Convert .ckpt to .safetensors

```bash
precision-convert model.ckpt -st
```

Loads the checkpoint safely (`weights_only=True`) and writes `model-full.safetensors` without changing any dtype.
Some old checkpoints store extra Python objects; if you trust the file, `--unsafe-load` reads them anyway.
Weights nested under a key (`model`, `module`, `state_dict`...) are found automatically, or pick one with
`--key`, for example `--key state_dict_ema` for timm's EMA weights.

### Extract EMA weights from a Stable Diffusion checkpoint

```bash
precision-convert model.ckpt -p fp16 -t ema-only -st
```

`-t ema-only` puts the EMA weights (`model_ema.*`) in place of the regular ones and drops the duplicates, for
Stable Diffusion and other latent-diffusion training checkpoints. `-t no-ema` just drops them.

### Find the layers that lose the most precision

```console
$ precision-convert Qwen3-0.6B/model.safetensors -p fp8_scaled --report
Converted 197 tensors to fp8_scaled, left 114 unchanged
Stored 197 weights as scaled FP8 (ComfyUI format, needs ComfyUI v0.4.0+)
Precision loss over 197 converted tensors (relative error): mean 0.0265, median 0.0265, max 0.0266
Largest:
  0.0266  model.layers.23.self_attn.q_proj.weight
  0.0266  model.layers.26.self_attn.v_proj.weight
  ...
```

The relative error is `‖converted − original‖ / ‖original‖` per tensor, measured on the weights, with no data or
GPU needed. With `fp8_scaled` it stays near 2.7% for almost every layer; a plain fp8 cast or fp16 overflow shows up
here as much larger numbers.

### Keep sensitive layers in higher precision automatically

```bash
precision-convert model.safetensors -p fp8_e4m3fn --max-error 0.05 --fallback bf16
```

Every tensor whose conversion error would exceed 5% is stored in bf16 instead, and the rest in fp8: mixed
precision chosen from the weights themselves. `--keep-original PATTERN` does the same by name, for example
`--keep-original "*norm*"`.

### See what's inside a checkpoint

```console
$ precision-convert Qwen3-0.6B-FP8/model.safetensors --info
Qwen3-0.6B-FP8/model.safetensors: 507 tensors, 751,659,264 values, 1.06 GB
  bfloat16             311 tensors   622.51 MB
  float8_e4m3fn        196 tensors   440.40 MB
  Format: block-scaled FP8 (DeepSeek / Qwen style)
  Metadata: 1 entries (format)
```

`--info` also recognises ComfyUI scaled FP8, EMA weights, LoRAs, nested training checkpoints and shard indexes.
Add `--json` for machine-readable output.

### Convert a LoRA and keep its metadata

```bash
precision-convert my-lora.safetensors -p fp16 -o my-lora-fp16.safetensors
```

## Why `fp8_scaled` beats a plain fp8 cast

`float8_e4m3fn` has 3 mantissa bits and its smallest normal number is 2⁻⁶ ≈ 0.0156. Trained weights are often
smaller than that, so a plain cast pushes them into the low-precision subnormal range. Scaling each tensor so its
largest value lands on fp8's maximum (448) uses the full range instead. Relative error measured on random
512×512 weights:

| Weight std | fp16 | bf16 | fp8 cast (`fp8_e4m3fn`) | `fp8_scaled` |
|---|---|---|---|---|
| 0.1 | 0.02% | 0.17% | 2.6% | 2.6% |
| 0.02 | 0.02% | 0.17% | 3.3% | 2.6% |
| 0.005 | 0.02% | 0.17% | 11.3% | 2.6% |
| 0.001 | 0.02% | 0.17% | 56.1% | 2.6% |

Run `--report` on your own model to see the real numbers.

## Memory use

Converting a 1.6 GB fp32 file to fp16 on macOS, measured with `/usr/bin/time -l` (peak memory footprint, which
leaves out file pages the OS can reclaim):

| | Peak memory | Time |
|---|---|---|
| A typical script (`load_file`, `.half()`, `save_file`) | 0.96 GB | 1.1 s |
| `precision-convert -p fp16` | 0.39 GB | 1.4 s |
| `precision-convert -p fp8_scaled` | 0.36 GB | 1.4 s |

The typical script's memory grows with the model; `precision-convert` stays at about one tensor plus PyTorch
itself, whatever the model size.

## Which precision should I pick?

| Precision | Bytes per value | Largest value | Good for |
|---|---|---|---|
| `fp32` | 4 | 3.4 × 10³⁸ | Training, reference copies, upcasting |
| `bf16` | 2 | 3.4 × 10³⁸ | The safe 2-byte choice: same range as fp32, nothing overflows |
| `fp16` | 2 | 65,504 | GPUs without fast bf16, and tools that expect fp16 |
| `int8_convrot` | 1 (+ a scale per row) | 127 × scale | ComfyUI: in our benchmark the fastest or close to it, and the closest to bf16, on RTX 30, 40 and 50-class GPUs |
| `int8` | 1 (+ a scale per row) | 127 × scale | ComfyUI on a T4: 9% faster than `int8_convrot` in our benchmark, slower on newer GPUs |
| `fp8_scaled` | 1 (+ a scale per layer) | 448 × scale | ComfyUI: smallest files with the best fp8 quality |
| `fp8_e4m3fn` | 1 | 448 | Tools that load plain fp8 weights; loses precision on small weights |
| `fp8_e5m2` | 1 | 57,344 | Plain fp8 with more range and less precision |
| `keep` | | | Change no dtype: for merging, splitting, EMA extraction or `.ckpt` → `.safetensors` |

## All options

```bash
precision-convert INPUT [-p PRECISION] [-t TYPE] [-o OUTPUT] [options]
```

| Option | What it does |
|---|---|
| `INPUT` | A `.safetensors` file, a PyTorch checkpoint (`.ckpt`, `.pt`, `.pth`, `.bin`), a `*.safetensors.index.json`, or a model folder. |
| `-p`, `--precision` | `fp32`, `fp16`, `bf16`, `fp8_e4m3fn`, `fp8_e5m2`, `fp8_scaled`, `int8`, `int8_convrot`, or `keep` (default). |
| `-t`, `--type` | `full` (default), `ema-only` (use the EMA weights) or `no-ema` (drop them). |
| `-o`, `--output` | Output file or folder. The extension picks the format: `.safetensors` or PyTorch. |
| `-st`, `--safe-tensors` | Save as safetensors (the default for safetensors input). |
| `--keep-original PATTERN` | Leave tensors whose name matches the glob in their original dtype. Repeatable. |
| `--max-error E` | Store tensors whose relative error would exceed `E` at the `--fallback` precision. |
| `--fallback PRECISION` | Precision for those tensors and, with `fp8_scaled`, `int8` or `int8_convrot`, for tensors that can't be quantized. Default: keep. |
| `--like REFERENCE` | With `fp8_scaled`, `int8` or `int8_convrot`: quantize the layers a reference file quantizes, and rename to its names. Path or URL; reads only the header and two small layers. |
| `--rename OLD=NEW` | Replace a tensor name prefix. Repeatable. |
| `--report` | Measure and list the tensors that lose the most precision. |
| `--max-shard-size SIZE` | Split safetensors output into shards of at most `SIZE` (e.g. `5GB`) plus an index. |
| `--skip-files GLOB` | Folder input: copy matching weight files unchanged. Repeatable. |
| `--key KEY` | Read the weights from this top-level key of a PyTorch checkpoint. |
| `--info` | Describe the input and exit. |
| `--json` | Print the result as one JSON object on stdout. |
| `-q`, `--quiet` | Print only warnings and errors. |
| `--overwrite` | Replace the output if it exists. |
| `--unsafe-load` | Fully unpickle PyTorch checkpoints. Only for files you trust. |
| `-v`, `--verbose` | List the tensors that were clamped or kept back. |

## Python API

```python
from precision_converter import convert_state_dict, convert_folder, load_checkpoint, save_checkpoint

state_dict, metadata = load_checkpoint("model.safetensors")
converted, report = convert_state_dict(state_dict, precision="fp8_scaled", measure=True)
save_checkpoint(converted, "model-fp8_scaled.safetensors", metadata=metadata)
print(f"{report.scaled} layers in fp8, worst error {max(report.errors.values()):.3f}")

convert_folder("Qwen3-8B-FP8", "Qwen3-8B-BF16", precision="bf16")
```

`convert_state_dict` takes any mapping of tensors, including `model.state_dict()`. For large files,
`open_checkpoint` reads tensors lazily and `iter_convert` plus `write_safetensors` stream the conversion.

## Use from scripts and AI agents

`precision-convert` never prompts, so it is safe to run unattended.

- `--json` prints one JSON object with the output paths, sizes, tensor counts and errors. Failures print
  `{"ok": false, "error": "..."}` and exit with code 1.
- `--info --json` describes a file without converting it: dtypes, sizes, detected formats and metadata.
- Exit codes: 0 success, 1 the conversion failed, 2 invalid arguments.
- `precision-convert --help` lists examples for each task.

## FAQ

**Does converting to fp16 or bf16 reduce quality?** Barely: the table above shows 0.02% (fp16) and 0.17% (bf16)
relative error, below what affects outputs in practice. Most models are released and run in half precision.

**fp16 or bf16?** bf16 has fp32's range, so nothing overflows; prefer it on GPUs that support it (NVIDIA Ampere and
newer). Choose fp16 for older GPUs or tools that require it.

**My fp16 model makes NaNs or black images.** Some VAEs overflow in fp16. Keep the VAE in fp32 with
`--keep-original "first_stage_model.*"` (single-file checkpoints) or `--skip-files "vae/*"` (Diffusers folders), or
use bf16.

**Is it safe to convert `.ckpt` files from the internet?** Yes by default: they load with `weights_only=True`, which
refuses pickled code. Only `--unsafe-load` turns that off.

**Do I need a GPU?** No. Everything runs on the CPU.

## Support the project

If precision-convert saved you time, a ⭐ [on GitHub](https://github.com/angelolamonaca/PyTorch-Precision-Converter)
helps other people find it. Bug reports and feature requests are welcome in the
[issues](https://github.com/angelolamonaca/PyTorch-Precision-Converter/issues).

## Upgrading

See the [changelog](https://github.com/angelolamonaca/PyTorch-Precision-Converter/blob/main/CHANGELOG.md). From 2.0:
safetensors input now produces safetensors output by default, and scaled FP8 input is dequantized correctly. From
1.x: `-t ema-only` outputs the real EMA weights, the default precision is `keep`, and checkpoints load with
`weights_only=True`. `python convert.py` still works.

## Contributing

```bash
pip install -e ".[test]"
pytest
```

## License

MIT. See [LICENSE](https://github.com/angelolamonaca/PyTorch-Precision-Converter/blob/main/LICENSE).
