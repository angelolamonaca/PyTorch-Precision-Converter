import argparse

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from precision_converter import (
    UnsafeCheckpointError,
    convert_state_dict,
    describe,
    load_checkpoint,
    save_checkpoint,
)


def ldm_state_dict():
    """A tiny state dict laid out like a Stable Diffusion training checkpoint."""
    return {
        "model.diffusion_model.input_blocks.0.0.weight": torch.zeros(4),
        "model.diffusion_model.input_blocks.0.0.bias": torch.zeros(2),
        "model_ema.diffusion_modelinput_blocks00weight": torch.ones(4),
        "model_ema.diffusion_modelinput_blocks00bias": torch.ones(2),
        "model_ema.decay": torch.tensor(0.9999),
        "model_ema.num_updates": torch.tensor(1234, dtype=torch.int32),
        "cond_stage_model.embeddings.position_ids": torch.arange(77).unsqueeze(0),
        "first_stage_model.decoder.weight": torch.randn(3, 3),
    }


def test_fp16_casts_only_floating_point_tensors():
    out, report = convert_state_dict(ldm_state_dict(), "fp16")
    assert out["model.diffusion_model.input_blocks.0.0.weight"].dtype == torch.float16
    assert out["cond_stage_model.embeddings.position_ids"].dtype == torch.int64
    assert out["model_ema.num_updates"].dtype == torch.int32
    assert report.converted == 6 and report.unchanged == 2


def test_keep_changes_nothing():
    sd = ldm_state_dict()
    out, report = convert_state_dict(sd, "keep")
    assert all(out[k] is sd[k] for k in sd)
    assert report.converted == 0


def test_fp32_upcasts_half_tensors():
    out, _ = convert_state_dict({"w": torch.ones(2, dtype=torch.float16)}, "fp32")
    assert out["w"].dtype == torch.float32


def test_ema_only_uses_ema_weights_and_drops_ema_keys():
    out, report = convert_state_dict(ldm_state_dict(), "keep", "ema-only")
    assert torch.equal(out["model.diffusion_model.input_blocks.0.0.weight"], torch.ones(4))
    assert torch.equal(out["model.diffusion_model.input_blocks.0.0.bias"], torch.ones(2))
    assert not any(k.startswith("model_ema.") for k in out)
    assert "first_stage_model.decoder.weight" in out
    assert report.ema_swapped == 2 and report.dropped == 4


def test_ema_only_with_dotted_ema_keys():
    sd = {"model.layer.weight": torch.zeros(2), "model_ema.layer.weight": torch.ones(2)}
    out, report = convert_state_dict(sd, "keep", "ema-only")
    assert list(out) == ["model.layer.weight"]
    assert torch.equal(out["model.layer.weight"], torch.ones(2))
    assert report.ema_swapped == 1


def test_ema_only_without_ema_keeps_regular_weights():
    sd = {"model.layer.weight": torch.zeros(2)}
    out, report = convert_state_dict(sd, "keep", "ema-only")
    assert torch.equal(out["model.layer.weight"], torch.zeros(2))
    assert report.ema_swapped == 0


def test_no_ema_drops_ema_and_keeps_regular_weights():
    out, report = convert_state_dict(ldm_state_dict(), "bf16", "no-ema")
    assert not any(k.startswith("model_ema.") for k in out)
    assert torch.equal(out["model.diffusion_model.input_blocks.0.0.weight"].float(), torch.zeros(4))
    assert report.dropped == 4


def test_fp16_clamps_overflow_but_keeps_inf_and_nan():
    t = torch.tensor([70000.0, -1e6, 1.5, float("inf"), float("-inf"), float("nan")])
    out, report = convert_state_dict({"w": t}, "fp16")
    w = out["w"].float()
    assert w[:3].tolist() == [65504.0, -65504.0, 1.5]
    assert w[3] == float("inf") and w[4] == float("-inf") and torch.isnan(w[5])
    assert report.clamped == {"w": 2}


@pytest.mark.parametrize("precision, limit", [("fp8_e4m3fn", 448.0), ("fp8_e5m2", 57344.0)])
def test_fp8_stays_finite_for_out_of_range_values(precision, limit):
    out, report = convert_state_dict({"w": torch.tensor([1e5, -1e5, 0.5])}, precision)
    assert out["w"].float().tolist() == [limit, -limit, 0.5]
    assert report.clamped == {"w": 2}


def test_fp8_e4m3fn_clamps_inf():
    out, _ = convert_state_dict({"w": torch.tensor([float("inf")])}, "fp8_e4m3fn")
    assert out["w"].float().item() == 448.0


def test_keep_original_patterns():
    sd = ldm_state_dict()
    out, _ = convert_state_dict(sd, "fp16", keep_original=["first_stage_model.*"])
    assert out["first_stage_model.decoder.weight"].dtype == torch.float32
    assert out["model.diffusion_model.input_blocks.0.0.weight"].dtype == torch.float16


def test_rejects_unknown_options():
    with pytest.raises(ValueError):
        convert_state_dict({}, "fp4")
    with pytest.raises(ValueError):
        convert_state_dict({}, "fp16", "ema")


def test_safetensors_roundtrip_keeps_metadata(tmp_path):
    src = tmp_path / "lora.safetensors"
    save_file({"w": torch.randn(4)}, src, metadata={"ss_network_dim": "16"})
    sd, metadata = load_checkpoint(src)
    out, _ = convert_state_dict(sd, "fp16")
    dst = tmp_path / "lora-fp16.safetensors"
    save_checkpoint(out, dst, metadata=metadata)
    with safe_open(dst, framework="pt") as f:
        assert f.metadata() == {"format": "pt", "ss_network_dim": "16"}
        assert f.get_tensor("w").dtype == torch.float16
    assert not list(tmp_path.glob("*.tmp"))


def test_safetensors_save_handles_shared_noncontiguous_and_non_tensor_values(tmp_path):
    w = torch.randn(2, 3)
    skipped = save_checkpoint({"a": w, "b": w, "c": w.t(), "step": 5}, tmp_path / "m.safetensors")
    assert skipped == ["step"]
    sd, _ = load_checkpoint(tmp_path / "m.safetensors")
    assert torch.equal(sd["a"], w) and torch.equal(sd["b"], w) and torch.equal(sd["c"], w.t())


def test_torch_checkpoint_roundtrip(tmp_path):
    torch.save({"state_dict": ldm_state_dict(), "global_step": 3}, tmp_path / "m.ckpt")
    sd, metadata = load_checkpoint(tmp_path / "m.ckpt")
    assert metadata is None and set(sd) == set(ldm_state_dict())
    save_checkpoint(sd, tmp_path / "out.ckpt")
    assert set(torch.load(tmp_path / "out.ckpt")["state_dict"]) == set(sd)


def test_plain_state_dict_file(tmp_path):
    torch.save({"w": torch.ones(2)}, tmp_path / "pytorch_model.bin")
    sd, _ = load_checkpoint(tmp_path / "pytorch_model.bin")
    assert list(sd) == ["w"]


def test_pickled_objects_need_unsafe_load(tmp_path):
    path = tmp_path / "m.ckpt"
    torch.save({"state_dict": {"w": torch.ones(2)}, "hparams": argparse.Namespace(lr=0.1)}, path)
    with pytest.raises(UnsafeCheckpointError, match="--unsafe-load"):
        load_checkpoint(path)
    sd, _ = load_checkpoint(path, unsafe=True)
    assert list(sd) == ["w"]


def test_describe():
    info = describe({**ldm_state_dict(), "step": 1})
    assert info.tensors == 8 and info.ema_tensors == 4
    assert info.dtypes["int64"] == (1, 77 * 8)
    assert info.other_keys == ["step"]
