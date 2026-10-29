import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from precision_converter import convert_state_dict, open_checkpoint
from precision_converter.core import clean_metadata
from precision_converter.fp8 import dequantize, find_scaled_weights, quantize_scaled
from precision_converter.io import SafetensorsWriter, ShardedWriter, header_size_bound, parse_size


def write(path, tensors, metadata=None):
    writer = SafetensorsWriter(path, reserve=header_size_bound(((k, list(v.shape)) for k, v in tensors.items()), metadata),
                               metadata=metadata)
    for key, value in tensors.items():
        writer.add(key, value)
    writer.close()


def test_streaming_writer_round_trips_every_dtype(tmp_path):
    tensors = {
        "f32": torch.randn(3, 4),
        "bf16": torch.randn(5).bfloat16(),
        "fp8": torch.randn(2, 2).to(torch.float8_e4m3fn),
        "e5m2": torch.randn(2).to(torch.float8_e5m2),
        "scalar": torch.tensor(0.5),
        "empty": torch.zeros(0, 3),
        "mask": torch.tensor([True, False]),
        "ids": torch.arange(77),
        "view": torch.randn(4, 6).t(),  # non-contiguous
    }
    write(tmp_path / "m.safetensors", tensors, {"note": "hi"})
    loaded = load_file(tmp_path / "m.safetensors")
    for key, value in tensors.items():
        assert loaded[key].dtype == value.dtype and loaded[key].shape == value.shape
        assert torch.equal(loaded[key].float(), value.float())
    with safe_open(tmp_path / "m.safetensors", framework="pt") as f:
        assert f.metadata() == {"note": "hi"}


def test_lazy_file_reports_shape_and_dtype_without_loading(tmp_path):
    save_file({"w": torch.zeros(3, 7, dtype=torch.float16)}, tmp_path / "m.safetensors")
    tensors = open_checkpoint(tmp_path / "m.safetensors").tensors
    assert tensors.shape("w") == [3, 7] and tensors.dtype("w") == torch.float16
    assert "w" in tensors and len(tensors) == 1


def test_sharded_writer_splits_and_indexes(tmp_path):
    tensors = {f"layer{i}.weight": torch.randn(256) for i in range(6)}  # 1 KB each
    writer = ShardedWriter(tmp_path / "model.safetensors", max_bytes=2048, reserve=4096, metadata={"format": "pt"})
    for key, value in tensors.items():
        writer.add(key, value)
    files = writer.close()
    names = sorted(f.name for f in files)
    assert names == [
        "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors", "model.safetensors.index.json",
    ]
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == 6 * 1024
    merged = open_checkpoint(tmp_path / "model.safetensors.index.json").tensors
    assert all(torch.equal(merged[k], v) for k, v in tensors.items())
    assert not list(tmp_path.glob("*.tmp"))


def test_sharded_writer_writes_one_file_when_everything_fits(tmp_path):
    writer = ShardedWriter(tmp_path / "model.safetensors", max_bytes=10**9, reserve=1024, metadata=None)
    writer.add("w", torch.ones(4))
    assert writer.close() == [tmp_path / "model.safetensors"]


@pytest.mark.parametrize("text, expected", [("5GB", 5 * 10**9), ("500MB", 5 * 10**8), ("2GiB", 2 * 2**30), ("1.5G", 15 * 10**8), ("123", 123)])
def test_parse_size(text, expected):
    assert parse_size(text) == expected


def test_nested_state_dict_keys(tmp_path):
    torch.save({"model": {"w": torch.ones(2)}, "optimizer": {"state": {}}, "epoch": 3}, tmp_path / "a.pt")
    checkpoint = open_checkpoint(tmp_path / "a.pt")
    assert checkpoint.nested_key == "model" and list(checkpoint.tensors) == ["w"]

    torch.save({"state_dict": {"w": torch.ones(2)}, "state_dict_ema": {"w": torch.zeros(2)}}, tmp_path / "timm.pth")
    ema = open_checkpoint(tmp_path / "timm.pth", key="state_dict_ema").tensors
    assert torch.equal(ema["w"], torch.zeros(2))

    torch.save({"epoch": 3, "config": {"lr": 0.1}}, tmp_path / "none.pt")
    with pytest.raises(ValueError, match="--key"):
        open_checkpoint(tmp_path / "none.pt")


# Scaled FP8 -----------------------------------------------------------------------------------


def small_weights(*shape):
    return torch.randn(*shape) * 0.02  # typical trained weights, far below fp8's sweet spot


def test_fp8_scaled_writes_comfyui_layout():
    sd = {
        "blocks.0.proj.weight": small_weights(64, 32),
        "blocks.0.proj.bias": torch.randn(64),
        "blocks.0.norm.weight": torch.ones(32),
        "conv.weight": torch.randn(8, 4, 3, 3),
        "model.embed_tokens.weight": torch.randn(100, 32),
        "pos_ids": torch.arange(10),
    }
    out, report = convert_state_dict(sd, "fp8_scaled")
    assert out["model.embed_tokens.weight"].dtype == torch.float32  # embeddings stay precise, like Comfy-Org's files
    assert out["blocks.0.proj.weight"].dtype == torch.float8_e4m3fn
    assert out["blocks.0.proj.weight_scale"].dtype == torch.float32 and out["blocks.0.proj.weight_scale"].dim() == 0
    assert json.loads(bytes(out["blocks.0.proj.comfy_quant"].tolist())) == {"format": "float8_e4m3fn"}
    assert out["blocks.0.proj.comfy_quant"].dtype == torch.uint8
    # Not Linear/Embedding weights: left alone (default fallback keeps their dtype).
    assert out["blocks.0.proj.bias"].dtype == torch.float32
    assert out["blocks.0.norm.weight"].dtype == torch.float32
    assert out["conv.weight"].dtype == torch.float32
    assert out["pos_ids"].dtype == torch.int64
    assert report.scaled == 1


def test_fp8_scaled_fallback_precision_for_other_tensors():
    out, _ = convert_state_dict({"a.weight": small_weights(16, 16), "a.bias": torch.randn(16)}, "fp8_scaled", fallback="bf16")
    assert out["a.bias"].dtype == torch.bfloat16


def test_scaling_keeps_far_more_precision_than_a_plain_fp8_cast():
    # Weights this small sit below e4m3fn's smallest normal number (2^-6) when cast directly.
    sd = {"layer.weight": torch.randn(256, 256) * 0.002}
    _, plain = convert_state_dict(sd, "fp8_e4m3fn", measure=True)
    _, scaled = convert_state_dict(sd, "fp8_scaled", measure=True)
    assert scaled.errors["layer.weight"] < 0.03
    assert plain.errors["layer.weight"] > 4 * scaled.errors["layer.weight"]


def test_dequantize_comfyui_scaled_fp8_round_trip():
    original = {"m.weight": small_weights(32, 64)}
    quantized, _ = convert_state_dict(original, "fp8_scaled")
    restored, report = convert_state_dict(quantized, "bf16")
    assert set(restored) == {"m.weight"} and restored["m.weight"].dtype == torch.bfloat16
    error = (restored["m.weight"].float() - original["m.weight"]).norm() / original["m.weight"].norm()
    assert error < 0.05 and report.dequantized == 1


def test_dequantize_older_comfyui_scaled_fp8():
    w = small_weights(8, 8)
    q, scale = quantize_scaled(w)
    sd = {"m.weight": q, "m.scale_weight": scale, "m.scale_input": torch.tensor(1.0),
          "scaled_fp8": torch.empty(0, dtype=torch.float8_e4m3fn)}
    out, _ = convert_state_dict(sd, "fp16")
    assert set(out) == {"m.weight"}
    assert torch.allclose(out["m.weight"].float(), q.float() * scale, rtol=1e-3)


def test_dequantize_block_scaled_fp8_like_deepseek():
    # 200x300 is not a multiple of 128, like many real layers: blocks at the edges are partial.
    w = small_weights(200, 300)
    blocks = torch.rand(2, 3) + 0.5
    expanded = blocks.repeat_interleave(128, 0)[:200].repeat_interleave(128, 1)[:, :300]
    q = (w / expanded).to(torch.float8_e4m3fn)
    out, report = convert_state_dict({"m.weight": q, "m.weight_scale_inv": blocks}, "bf16")
    assert report.dequantized == 1
    assert torch.allclose(out["m.weight"].float(), q.float() * expanded, rtol=1e-2, atol=1e-4)


def test_dequantize_per_row_scales_like_compressed_tensors():
    q = torch.randn(4, 8).to(torch.float8_e4m3fn)
    scale = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    out, _ = convert_state_dict({"m.weight": q, "m.weight_scale": scale, "m.input_scale": torch.tensor(1.0)}, "fp32")
    assert set(out) == {"m.weight"}
    assert torch.equal(out["m.weight"], q.float() * scale)


def test_dequantize_fp8_weights_stored_as_bytes():
    w = small_weights(4, 4)
    q, scale = quantize_scaled(w)
    sd = {"m.weight": q.view(torch.uint8), "m.weight_scale": scale,
          "m.comfy_quant": torch.tensor(list(b'{"format": "float8_e4m3fn"}'), dtype=torch.uint8)}
    out, _ = convert_state_dict(sd, "fp32")
    assert torch.allclose(out["m.weight"], q.float() * scale)


def test_unsupported_comfyui_formats_are_reported():
    sd = {"m.weight": torch.zeros(4, 2, dtype=torch.uint8), "m.weight_scale": torch.ones(1),
          "m.comfy_quant": torch.tensor(list(b'{"format": "nvfp4"}'), dtype=torch.uint8)}
    with pytest.raises(ValueError, match="nvfp4"):
        convert_state_dict(sd, "bf16")


def test_keep_leaves_scaled_fp8_untouched():
    q, scale = quantize_scaled(small_weights(4, 4))
    sd = {"m.weight": q, "m.weight_scale": scale}
    out, _ = convert_state_dict(sd, "keep")
    assert out["m.weight"] is q and "m.weight_scale" in out


def test_find_scaled_weights_lists_extras():
    q, scale = quantize_scaled(small_weights(4, 4))
    found = find_scaled_weights({"m.weight": q, "m.weight_scale": scale, "m.input_scale": scale,
                                 "m.comfy_quant": torch.tensor(list(b'{"format": "float8_e4m3fn"}'), dtype=torch.uint8)})
    assert found.scales == {"m.weight": "m.weight_scale"}
    assert found.extra == {"m.weight_scale", "m.input_scale", "m.comfy_quant"}


def test_dequantize_rejects_mismatched_scales():
    with pytest.raises(ValueError, match="scale"):
        dequantize(torch.zeros(4, 4, dtype=torch.float8_e4m3fn), torch.ones(3, 5))


def test_old_quantization_metadata_is_dropped_after_conversion():
    assert clean_metadata({"_quantization_metadata": "{}", "x": "1"}, "bf16") == {"x": "1"}
    assert clean_metadata({"_quantization_metadata": "{}"}, "keep") == {"_quantization_metadata": "{}"}


# Automatic mixed precision ---------------------------------------------------------------------


def test_max_error_keeps_sensitive_tensors_at_fallback():
    sd = {
        "smooth.weight": torch.full((64, 64), 0.5),  # exactly representable: zero error
        "noisy.weight": torch.randn(64, 64) * 1e-3,  # plain fp8 flushes most of this to subnormals
    }
    out, report = convert_state_dict(sd, "fp8_e4m3fn", max_error=0.05, fallback="bf16")
    assert out["smooth.weight"].dtype == torch.float8_e4m3fn
    assert out["noisy.weight"].dtype == torch.bfloat16
    assert report.fallback == ["noisy.weight"]
    assert report.errors["smooth.weight"] == 0.0


def test_max_error_with_scaled_fp8_and_keep_fallback():
    sd = {"a.weight": small_weights(32, 32)}
    out, report = convert_state_dict(sd, "fp8_scaled", max_error=1e-6)
    assert out == {"a.weight": sd["a.weight"]} and report.fallback == ["a.weight"]


def test_output_parses_the_way_comfyui_loads_it(tmp_path):
    """Mirrors ComfyUI's loader (comfy/ops.py): json.loads(comfy_quant.numpy().tobytes()), then
    pop_scale("weight_scale") for float8 formats, then weight * scale when dequantizing."""
    from safetensors.torch import load_file as load

    w = small_weights(48, 32)
    out, _ = convert_state_dict({"diffusion_model.blocks.0.attn.qkv.weight": w}, "fp8_scaled")
    write(tmp_path / "m.safetensors", out)
    sd = load(tmp_path / "m.safetensors")
    prefix = "diffusion_model.blocks.0.attn.qkv."
    layer_conf = json.loads(sd.pop(prefix + "comfy_quant").numpy().tobytes())
    assert layer_conf["format"] in ("float8_e4m3fn", "float8_e5m2")
    scale = sd.pop(prefix + "weight_scale")
    weight = sd.pop(prefix + "weight")
    assert weight.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    restored = weight.to(torch.bfloat16) * scale.to(torch.bfloat16)
    assert ((restored.float() - w).norm() / w.norm()) < 0.05
    assert sd == {}
