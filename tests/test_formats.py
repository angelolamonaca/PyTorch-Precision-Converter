import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from precision_converter import convert_state_dict, int8, open_checkpoint
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


def test_quantize_scaled_to_416_like_comfy_org():
    w = small_weights(32, 16)
    q, scale = quantize_scaled(w, fp8_max=416)
    assert q.float().abs().max().item() == 416.0
    assert scale.item() == pytest.approx(w.abs().max().item() / 416)
    with pytest.raises(ValueError, match="fp8_max"):
        quantize_scaled(w, fp8_max=500)


# ComfyUI int8 and int8_convrot -----------------------------------------------------------------


def test_convrot_hadamard_is_its_own_inverse():
    h = int8.hadamard(16)
    assert torch.equal(h, h.T)
    assert torch.allclose(h @ h, torch.eye(16), atol=1e-6)
    block = torch.tensor([[1.0, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]])
    assert torch.equal(h, torch.kron(block, block) / 4)  # Kronecker powers of ConvRot's 4x4 block
    for size in (2, 8, 32):
        with pytest.raises(ValueError, match="power of 4"):
            int8.hadamard(size)


def reference_convrot(w, group_size=256):
    """comfy-kitchen's eager reference, written out: rotate in FP32, per-row scale amax/127, round half to even."""
    h = int8.hadamard(group_size)
    rotated = (w.float().reshape(w.shape[0], -1, group_size) @ h.T).reshape(w.shape)
    scale = (rotated.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-30)
    return (rotated / scale).round().clamp(-128, 127).to(torch.int8), scale


def test_int8_convrot_matches_the_reference_in_chunks(monkeypatch):
    w = small_weights(300, 1024).bfloat16()
    q_ref, s_ref = reference_convrot(w)
    monkeypatch.setattr(int8, "CHUNK", 7 * 1024)  # 7 rows per chunk, so the last chunk is partial
    q, s = int8.quantize_convrot(w)
    assert torch.equal(q, q_ref) and torch.equal(s, s_ref)
    assert q.dtype == torch.int8 and s.dtype == torch.float32 and list(s.shape) == [300, 1]


def test_int8_convrot_writes_comfyui_layout_and_round_trips():
    sd = {"a.weight": small_weights(32, 512), "b.weight": small_weights(16, 100), "a.bias": torch.randn(32)}
    out, report = convert_state_dict(sd, "int8_convrot", measure=True)
    assert out["a.weight"].dtype == torch.int8 and list(out["a.weight_scale"].shape) == [32, 1]
    assert bytes(out["a.comfy_quant"].tolist()) == b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'
    assert out["b.weight"].dtype == torch.float32 and report.not_quantized == ["b.weight"]  # 100 columns: no whole group
    assert out["a.bias"].dtype == torch.float32
    assert report.scaled == 1 and report.errors["a.weight"] < 0.02
    back, back_report = convert_state_dict(out, "fp32")
    assert back_report.dequantized == back_report.dequantized_int8 == 1
    assert set(back) == {"a.weight", "a.bias", "b.weight"}
    error = (back["a.weight"] - sd["a.weight"]).norm() / sd["a.weight"].norm()
    assert error < 0.02


def reference_rowwise(w):
    """comfy-kitchen's quantize_int8_rowwise on the FP32 weight: per-row scale amax/127, round half to even."""
    w = w.float()
    scale = (w.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-30)
    return (w / scale).round().clamp(-128, 127).to(torch.int8), scale


def test_int8_matches_the_reference_and_leaves_the_input_alone(monkeypatch):
    w = small_weights(300, 100)  # without ConvRot any column count works
    original = w.clone()
    monkeypatch.setattr(int8, "CHUNK", 7 * 100)  # 7 rows per chunk, so the last chunk is partial
    for weight in (w, w.bfloat16()):
        q_ref, s_ref = reference_rowwise(weight)
        q, s = int8.quantize(weight)
        assert torch.equal(q, q_ref) and torch.equal(s, s_ref)
        assert q.dtype == torch.int8 and s.dtype == torch.float32 and list(s.shape) == [300, 1]
    assert torch.equal(w, original)  # an FP32 weight is copied before dividing, not divided in place
    with pytest.raises(ValueError, match="2-D"):
        int8.quantize(torch.ones(2, 3, 4))


def test_int8_writes_comfyui_layout_and_round_trips():
    sd = {"a.weight": small_weights(32, 512), "b.weight": small_weights(16, 100), "a.bias": torch.randn(32),
          "zero.weight": torch.zeros(8, 16)}
    out, report = convert_state_dict(sd, "int8", measure=True)
    assert all(out[k + ".weight"].dtype == torch.int8 for k in ("a", "b", "zero"))
    assert list(out["b.weight_scale"].shape) == [16, 1] and report.not_quantized == []
    assert bytes(out["a.comfy_quant"].tolist()) == b'{"format": "int8_tensorwise"}'
    assert out["a.bias"].dtype == torch.float32
    assert report.scaled == 3 and max(report.errors["a.weight"], report.errors["b.weight"]) < 0.02
    back, back_report = convert_state_dict(out, "fp32")
    assert back_report.dequantized == back_report.dequantized_int8 == 3
    assert set(back) == set(sd) and torch.equal(back["zero.weight"], sd["zero.weight"])
    for key in ("a.weight", "b.weight"):
        assert (back[key] - sd[key]).norm() / sd[key].norm() < 0.02


def test_int8_without_convrot_and_unknown_formats():
    q = torch.randint(-127, 128, (4, 8), dtype=torch.int8)
    plain = {"x.weight": q, "x.weight_scale": torch.tensor(0.5),
             "x.comfy_quant": torch.tensor(list(b'{"format": "int8_tensorwise"}'), dtype=torch.uint8)}
    out, _ = convert_state_dict(plain, "fp32")
    assert torch.equal(out["x.weight"], q.float() * 0.5)
    plain["x.comfy_quant"] = torch.tensor(list(b'{"format": "nvfp4"}'), dtype=torch.uint8)
    with pytest.raises(ValueError, match="nvfp4"):
        convert_state_dict(plain, "fp32")


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


# Copying a reference file's layer choice -------------------------------------------------------------


def test_reference_layers_from_each_layout(tmp_path):
    from precision_converter.fp8 import reference_layers
    from precision_converter.io import read_header

    q, scale = quantize_scaled(small_weights(4, 4))
    comfy = {"a.weight": q, "a.weight_scale": scale,
             "a.comfy_quant": torch.tensor(list(b'{"format": "int8_tensorwise"}'), dtype=torch.uint8),
             "b.weight": torch.randn(4, 4).bfloat16()}
    legacy = {"c.weight": q, "c.scale_weight": scale, "scaled_fp8": torch.empty(0, dtype=torch.float8_e4m3fn)}
    plain = {"d.weight": q, "e.weight": torch.randn(4, 4)}
    for name, sd, expected in (("comfy", comfy, {"a.weight"}), ("legacy", legacy, {"c.weight"}), ("plain", plain, {"d.weight"})):
        write(tmp_path / f"{name}.safetensors", sd)
        assert reference_layers(read_header(tmp_path / f"{name}.safetensors")) == expected


def test_quantize_only_follows_the_reference():
    sd = {"embed_tokens.weight": small_weights(10, 8), "q.weight": small_weights(8, 8), "k.weight": small_weights(8, 8)}
    out, report = convert_state_dict(sd, "fp8_scaled", quantize_only={"embed_tokens.weight", "q.weight"})
    assert out["embed_tokens.weight"].dtype == torch.float8_e4m3fn  # listed, so quantized despite the name
    assert out["q.weight"].dtype == torch.float8_e4m3fn
    assert out["k.weight"].dtype == torch.float32 and "k.weight_scale" not in out
    assert report.scaled == 2


def test_reference_fp8_max_finds_comfy_orgs_416(tmp_path):
    from precision_converter.fp8 import COMFY_ORG_FP8_MAX, reference_fp8_max
    from precision_converter.io import read_header, read_tensor

    def fp8_max_of(path, sd):
        write(path, sd)
        return reference_fp8_max(read_header(path), lambda key: read_tensor(path, key))

    def reference(fp8_max, scale_suffix=".weight_scale"):
        sd = {}
        for name in ("a", "b", "c"):
            q, scale = quantize_scaled(small_weights(8, 8), fp8_max=fp8_max)
            sd.update({f"{name}.weight": q, f"{name}{scale_suffix}": scale})
        return sd

    assert fp8_max_of(tmp_path / "te.safetensors", reference(COMFY_ORG_FP8_MAX)) == COMFY_ORG_FP8_MAX
    assert fp8_max_of(tmp_path / "legacy.safetensors", reference(COMFY_ORG_FP8_MAX, ".scale_weight")) == COMFY_ORG_FP8_MAX
    assert fp8_max_of(tmp_path / "dit.safetensors", reference(None)) is None  # 448, the default
    int8 = {"a.weight": torch.ones(8, 8, dtype=torch.int8), "a.weight_scale": torch.ones(8, 1)}
    assert fp8_max_of(tmp_path / "int8.safetensors", int8) is None


@pytest.fixture
def http_files(monkeypatch):
    """Serve bytes at fake URLs through urllib, answering range requests. Returns the files to serve
    (url -> bytes) and the list of response sizes."""
    import io as stdlib_io
    import urllib.request

    files, sent = {}, []

    class Response(stdlib_io.BytesIO):
        status = 206

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        start, end = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        sent.append(end - start + 1)
        return Response(files[request.full_url][start : end + 1])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return files, sent


def test_read_header_over_http(tmp_path, http_files):
    """Only the header is fetched, with range requests, even for a large remote file."""
    from precision_converter.io import read_header

    files, sent = http_files
    write(tmp_path / "remote.safetensors", {"w": torch.zeros(1000, 1000)})
    files["https://example.com/remote.safetensors"] = (tmp_path / "remote.safetensors").read_bytes()
    assert read_header("https://example.com/remote.safetensors")["w"]["shape"] == [1000, 1000]
    assert sum(sent) < 1000  # never the 4 MB of tensor data


def test_read_tensor_over_http(tmp_path, http_files):
    """One small tensor is fetched without the large ones."""
    from precision_converter.io import read_tensor

    files, sent = http_files
    small = torch.randn(4, 8).to(torch.float8_e4m3fn)
    write(tmp_path / "remote.safetensors", {"big": torch.zeros(1000, 1000), "small": small})
    files["https://example.com/remote.safetensors"] = (tmp_path / "remote.safetensors").read_bytes()
    got = read_tensor("https://example.com/remote.safetensors", "small")
    assert got.dtype == torch.float8_e4m3fn and torch.equal(got.view(torch.uint8), small.view(torch.uint8))
    assert sum(sent) < 1000


def test_read_tensors_over_http_groups_close_tensors(tmp_path, http_files):
    """The header is read once; tensors less than 1 MB apart share a request; a large one between them is skipped."""
    from precision_converter.io import read_tensors

    files, sent = http_files
    configs = {f"layers.{i}.comfy_attention.config": torch.tensor(list(b'{"attention": "x"}'), dtype=torch.uint8)
               for i in range(3)}
    write(tmp_path / "remote.safetensors", {"big": torch.zeros(1000, 1000), "empty": torch.zeros(0), **configs})
    files["https://example.com/remote.safetensors"] = (tmp_path / "remote.safetensors").read_bytes()
    got = read_tensors("https://example.com/remote.safetensors", [*configs, "empty"])
    assert set(got) == {*configs, "empty"} and got["empty"].shape == (0,)
    assert all(torch.equal(got[k], v) for k, v in configs.items())
    assert len(sent) == 3 and sum(sent) < 2000  # 8-byte size, header, one range for the configs


def test_renamed_mapping_and_inference():
    from precision_converter.io import RenamedMapping, infer_renames

    hf = {"model.language_model.layers.0.mlp.weight": torch.zeros(4, 2), "model.visual.proj.weight": torch.zeros(3, 3),
          "lm_head.weight": torch.zeros(5, 2)}
    reference = {"model.layers.0.mlp.weight": {"shape": [4, 2]}, "model.visual.proj.weight": {"shape": [3, 3]},
                 "lm_head.weight": {"shape": [5, 2]}}
    rules = infer_renames(hf, reference)
    assert rules == {"model.language_model.": "model."}
    renamed = RenamedMapping(hf, rules)
    assert set(renamed) == set(reference)
    assert renamed.shape("model.layers.0.mlp.weight") == [4, 2]
    with pytest.raises(ValueError, match="both"):
        RenamedMapping({"a.x": torch.zeros(1), "b.x": torch.zeros(1)}, {"a.": "c.", "b.": "c."})
