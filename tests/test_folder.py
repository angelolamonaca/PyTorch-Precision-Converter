import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from precision_converter import convert_folder
from precision_converter.fp8 import quantize_scaled


@pytest.fixture
def fp8_model(tmp_path):
    """A Hugging Face style FP8 model folder with block scales, sharded across two files.

    The scale of layer 1 lives in the other shard, as happens in real DeepSeek checkpoints.
    """
    root = tmp_path / "Model-FP8"
    root.mkdir()
    blocks = torch.full((1, 1), 0.5)
    w0 = torch.randn(64, 64).to(torch.float8_e4m3fn)
    w1 = torch.randn(64, 64).to(torch.float8_e4m3fn)
    save_file({"layers.0.weight": w0, "layers.0.weight_scale_inv": blocks, "layers.1.weight": w1,
               "embed.weight": torch.randn(10, 64).bfloat16()},
              root / "model-00001-of-00002.safetensors", metadata={"format": "pt"})
    save_file({"layers.1.weight_scale_inv": blocks * 2, "norm.weight": torch.ones(64).bfloat16()},
              root / "model-00002-of-00002.safetensors", metadata={"format": "pt"})
    weight_map = {
        "layers.0.weight": "model-00001-of-00002.safetensors",
        "layers.0.weight_scale_inv": "model-00001-of-00002.safetensors",
        "layers.1.weight": "model-00001-of-00002.safetensors",
        "embed.weight": "model-00001-of-00002.safetensors",
        "layers.1.weight_scale_inv": "model-00002-of-00002.safetensors",
        "norm.weight": "model-00002-of-00002.safetensors",
    }
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}))
    config = {"architectures": ["X"], "torch_dtype": "bfloat16",
              "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}}
    (root / "config.json").write_text(json.dumps(config))
    (root / "tokenizer.json").write_text('{"vocab": {}}')
    (root / ".cache").mkdir()
    (root / ".cache" / "junk").write_text("x")
    return root, w0, w1


def test_fp8_folder_to_bf16(fp8_model, tmp_path):
    root, w0, w1 = fp8_model
    out = tmp_path / "Model-BF16"
    result = convert_folder(root, out, precision="bf16")

    assert result.report.dequantized == 2
    assert sorted(result.converted_files) == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    shard1 = load_file(out / "model-00001-of-00002.safetensors")
    shard2 = load_file(out / "model-00002-of-00002.safetensors")
    assert set(shard1) == {"layers.0.weight", "layers.1.weight", "embed.weight"}
    assert set(shard2) == {"norm.weight"}
    assert torch.allclose(shard1["layers.0.weight"].float(), w0.float() * 0.5, rtol=1e-2)
    assert torch.allclose(shard1["layers.1.weight"].float(), w1.float() * 1.0, rtol=1e-2)  # scale from the other shard
    assert shard1["layers.0.weight"].dtype == torch.bfloat16

    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {"layers.0.weight", "layers.1.weight", "embed.weight", "norm.weight"}
    expected_total = sum(t.numel() * t.element_size() for shard in (shard1, shard2) for t in shard.values())
    assert index["metadata"]["total_size"] == expected_total

    config = json.loads((out / "config.json").read_text())
    assert "quantization_config" not in config and config["torch_dtype"] == "bfloat16"
    assert result.updated_configs == ["config.json"]
    assert (out / "tokenizer.json").read_text() == '{"vocab": {}}'
    assert not (out / ".cache").exists()


def test_folder_fp16_updates_dtype_and_keeps_plain_bin_files(tmp_path):
    root = tmp_path / "pipe"
    (root / "text_encoder").mkdir(parents=True)
    (root / "vae").mkdir()
    torch.save({"w": torch.randn(4, 4)}, root / "text_encoder" / "pytorch_model.bin")
    (root / "text_encoder" / "config.json").write_text(json.dumps({"torch_dtype": "float32"}))
    save_file({"decoder.weight": torch.randn(4, 4)}, root / "vae" / "diffusion_pytorch_model.safetensors")
    (root / "vae" / "config.json").write_text(json.dumps({"_class_name": "AutoencoderKL"}))
    (root / "model_index.json").write_text("{}")

    out = tmp_path / "pipe-fp16"
    result = convert_folder(root, out, precision="fp16", skip_files=["vae/*"])

    bin_data = torch.load(out / "text_encoder" / "pytorch_model.bin")
    assert set(bin_data) == {"w"} and bin_data["w"].dtype == torch.float16  # still a plain state dict
    assert json.loads((out / "text_encoder" / "config.json").read_text())["torch_dtype"] == "float16"
    with safe_open(out / "vae" / "diffusion_pytorch_model.safetensors", framework="pt") as f:
        assert f.get_tensor("decoder.weight").dtype == torch.float32  # skipped: copied as-is
    assert json.loads((out / "vae" / "config.json").read_text()) == {"_class_name": "AutoencoderKL"}
    assert (out / "model_index.json").exists()
    assert result.converted_files == ["text_encoder/pytorch_model.bin"]


def test_folder_to_comfyui_fp8_scaled(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    save_file({"proj.weight": torch.randn(32, 32) * 0.01, "proj.bias": torch.randn(32)}, root / "model.safetensors")
    convert_folder(root, tmp_path / "m-fp8", precision="fp8_scaled")
    out = load_file(tmp_path / "m-fp8" / "model.safetensors")
    assert set(out) == {"proj.weight", "proj.weight_scale", "proj.comfy_quant", "proj.bias"}


def test_output_inside_input_is_refused(tmp_path):
    (tmp_path / "m").mkdir()
    with pytest.raises(ValueError, match="inside"):
        convert_folder(tmp_path / "m", tmp_path / "m" / "out", precision="fp16")


def test_scaled_round_trip_through_folder(tmp_path):
    root = tmp_path / "comfy"
    root.mkdir()
    q, scale = quantize_scaled(torch.randn(16, 16) * 0.02)
    save_file({"a.weight": q, "a.weight_scale": scale,
               "a.comfy_quant": torch.tensor(list(b'{"format": "float8_e4m3fn"}'), dtype=torch.uint8)},
              root / "unet.safetensors", metadata={"_quantization_metadata": "{}", "modelspec.title": "x"})
    convert_folder(root, tmp_path / "comfy-bf16", precision="bf16")
    with safe_open(tmp_path / "comfy-bf16" / "unet.safetensors", framework="pt") as f:
        assert list(f.keys()) == ["a.weight"]
        assert f.metadata() == {"format": "pt", "modelspec.title": "x"}
