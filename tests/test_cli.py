import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from precision_converter.cli import main

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def ckpt(tmp_path):
    path = tmp_path / "model.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.layer.weight": torch.zeros(8),
                "model_ema.layerweight": torch.ones(8),
                "model_ema.num_updates": torch.tensor(5, dtype=torch.int32),
            }
        },
        path,
    )
    return path


def test_default_output_name_matches_1x(ckpt, capsys):
    assert main([str(ckpt), "-p", "fp16"]) == 0
    out = torch.load(ckpt.with_name("model-full-fp16.ckpt"))["state_dict"]
    assert out["model.layer.weight"].dtype == torch.float16
    assert "Converted 2 tensors to fp16" in capsys.readouterr().out


def test_ema_only_to_safetensors(ckpt):
    assert main(["-f", str(ckpt), "-p", "bf16", "-t", "ema-only", "-st"]) == 0
    with safe_open(ckpt.with_name("model-ema-only-bf16.safetensors"), framework="pt") as f:
        assert list(f.keys()) == ["model.layer.weight"]
        assert torch.equal(f.get_tensor("model.layer.weight").float(), torch.ones(8))


def test_output_extension_picks_format(ckpt, tmp_path):
    dst = tmp_path / "custom.safetensors"
    assert main([str(ckpt), "-p", "fp16", "-o", str(dst)]) == 0
    with safe_open(dst, framework="pt") as f:
        assert f.get_tensor("model.layer.weight").dtype == torch.float16


def test_keep_precision_omits_it_from_the_name(ckpt):
    assert main([str(ckpt), "-t", "no-ema"]) == 0
    assert ckpt.with_name("model-no-ema.ckpt").exists()


def test_refuses_to_overwrite(ckpt, tmp_path):
    dst = tmp_path / "out.ckpt"
    dst.write_text("keep me")
    with pytest.raises(SystemExit):
        main([str(ckpt), "-o", str(dst)])
    assert dst.read_text() == "keep me"
    assert main([str(ckpt), "-o", str(dst), "--overwrite"]) == 0


def test_refuses_to_replace_input(ckpt):
    with pytest.raises(SystemExit):
        main([str(ckpt), "-o", str(ckpt), "--overwrite"])


def test_safe_tensors_flag_conflicts_with_other_extension(ckpt, tmp_path):
    with pytest.raises(SystemExit):
        main([str(ckpt), "-st", "-o", str(tmp_path / "out.ckpt")])


def test_missing_input(tmp_path):
    with pytest.raises(SystemExit):
        main([str(tmp_path / "nope.ckpt")])


def test_info(tmp_path, capsys):
    src = tmp_path / "lora.safetensors"
    save_file({"a": torch.zeros(10), "b": torch.zeros(5, dtype=torch.float16)}, src, metadata={"ss_dim": "4"})
    assert main([str(src), "--info"]) == 0
    out = capsys.readouterr().out
    assert "2 tensors, 15 values" in out and "float16" in out and "ss_dim" in out
    assert list(tmp_path.iterdir()) == [src]


def test_clamping_warning(tmp_path, capsys):
    src = tmp_path / "m.safetensors"
    save_file({"big": torch.tensor([1e6, 1.0])}, src)
    assert main([str(src), "-p", "fp16", "-st", "-v"]) == 0
    err = capsys.readouterr().err
    assert "outside the fp16 range" in err and "big: 1 values" in err


def test_unsafe_checkpoint_error_message(tmp_path, capsys):
    import argparse

    src = tmp_path / "m.ckpt"
    torch.save({"state_dict": {"w": torch.ones(1)}, "args": argparse.Namespace()}, src)
    assert main([str(src), "-p", "fp16"]) == 1
    assert "--unsafe-load" in capsys.readouterr().err
    assert main([str(src), "-p", "fp16", "--unsafe-load"]) == 0


def test_convert_py_shim_still_works(ckpt):
    result = subprocess.run(
        [sys.executable, str(REPO / "convert.py"), "-f", str(ckpt), "-p", "fp16", "-st"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stderr
    assert ckpt.with_name("model-full-fp16.safetensors").exists()
