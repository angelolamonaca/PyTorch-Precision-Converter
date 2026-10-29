# Notes for coding agents

- Layout: `precision_converter/io.py` reads and writes files (lazy safetensors, shard sets, streaming writer),
  `fp8.py` handles scaled FP8 (and finds every quantized layout in an input), `int8.py` ComfyUI's int8 / ConvRot,
  `core.py` converts state dicts, `folder.py` converts model folders, `cli.py` is the `precision-convert` command.
- Tests: `pip install -e ".[test]"` then `pytest`. They run on the CPU in a few seconds. Add a test for every
  behaviour change.
- Compatibility: CI runs Python 3.9 with PyTorch 2.1 and safetensors 0.4.3, so avoid newer syntax and APIs
  (no `match`, keep `from __future__ import annotations` in modules).
- Memory: new code paths should stream tensors with `iter_convert` and `write_safetensors` rather than build
  whole state dicts.
- Docs: user-visible changes go in the README's "How to" section and in CHANGELOG.md.
