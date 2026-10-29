import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

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


def run_json(argv, capsys):
    assert main(argv + ["--json"]) == 0
    captured = capsys.readouterr()
    assert "⭐" not in captured.err  # machine output stays clean
    return json.loads(captured.out)


def test_safetensors_input_defaults_to_safetensors_output(tmp_path):
    src = tmp_path / "m.safetensors"
    save_file({"w": torch.randn(4)}, src)
    assert main([str(src), "-p", "bf16"]) == 0
    assert (tmp_path / "m-full-bf16.safetensors").exists()


def test_json_output(tmp_path, capsys):
    src = tmp_path / "m.safetensors"
    save_file({"a.weight": torch.randn(8, 8) * 0.01, "n.weight": torch.ones(8)}, src)
    data = run_json([str(src), "-p", "fp8_scaled", "--report"], capsys)
    assert data["ok"] and data["scaled"] == 1 and data["output"].endswith("m-full-fp8_scaled.safetensors")
    assert data["errors"]["largest"][0]["key"] == "a.weight"
    assert data["output_bytes"] > 0


def test_info_json(tmp_path, capsys):
    q = torch.randn(4, 4).to(torch.float8_e4m3fn)
    save_file({"a.weight": q, "a.weight_scale_inv": torch.ones(1, 1)}, tmp_path / "m.safetensors")
    assert main([str(tmp_path / "m.safetensors"), "--info", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["tensors"] == 2 and "block-scaled FP8 (DeepSeek / Qwen style)" in data["formats"]


def test_star_line_and_quiet(tmp_path, capsys):
    src = tmp_path / "m.safetensors"
    save_file({"w": torch.randn(4)}, src)
    assert main([str(src), "-p", "fp16"]) == 0
    captured = capsys.readouterr()
    assert "github.com/angelolamonaca/PyTorch-Precision-Converter" in captured.err
    assert main([str(src), "-p", "bf16", "-q"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_merge_shards_from_index(tmp_path, capsys):
    save_file({"a": torch.ones(2)}, tmp_path / "model-00001-of-00002.safetensors")
    save_file({"b": torch.zeros(3)}, tmp_path / "model-00002-of-00002.safetensors")
    weight_map = {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    data = run_json([str(tmp_path / "model.safetensors.index.json"), "-o", str(tmp_path / "merged.safetensors")], capsys)
    assert data["files"] == [str(tmp_path / "merged.safetensors")]
    merged = load_file(tmp_path / "merged.safetensors")
    assert set(merged) == {"a", "b"}


def test_split_into_shards(tmp_path, capsys):
    src = tmp_path / "big.safetensors"
    save_file({f"t{i}": torch.randn(256) for i in range(4)}, src)
    out = tmp_path / "out" / "model.safetensors"
    out.parent.mkdir()
    data = run_json([str(src), "-o", str(out), "--max-shard-size", "2KiB"], capsys)
    assert len(data["files"]) == 3  # 2 shards + index
    assert (out.parent / "model.safetensors.index.json").exists()


def test_folder_input(tmp_path, capsys):
    root = tmp_path / "model"
    root.mkdir()
    save_file({"w": torch.randn(4, 4)}, root / "model.safetensors")
    (root / "config.json").write_text(json.dumps({"torch_dtype": "float32"}))
    data = run_json([str(root), "-p", "bf16"], capsys)
    assert data["output"] == str(tmp_path / "model-bf16")
    assert data["updated_configs"] == ["config.json"]
    with pytest.raises(SystemExit):
        main([str(root), "-p", "bf16"])  # output exists now


def test_nested_key_message(tmp_path, capsys):
    torch.save({"model": {"w": torch.ones(2)}, "epoch": 1}, tmp_path / "train.pt")
    assert main([str(tmp_path / "train.pt"), "-p", "fp16", "-o", str(tmp_path / "w.pt")]) == 0
    assert "Using the weights under the 'model' key" in capsys.readouterr().out
    assert list(torch.load(tmp_path / "w.pt")) == ["w"]  # .pt output stays a plain state dict


def test_error_as_json(tmp_path, capsys):
    src = tmp_path / "m.ckpt"
    torch.save({"state_dict": {"w": torch.ones(1)}, "args": argparse.Namespace()}, src)
    assert main([str(src), "-p", "fp16", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_version_mentions_torch(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "torch" in capsys.readouterr().out
