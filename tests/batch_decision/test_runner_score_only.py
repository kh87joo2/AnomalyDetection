from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from batch_decision.runner import main
from datasets.transforms.fdc_normalization import ChannelScaler
from models.patchtst.patchtst_ssl import PatchTSTSSL


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_runner_score_only_patchtst_executes_with_temp_artifacts(
    torch_module,
    patchtst_smoke_config,
    tmp_path: Path,
    capsys,
) -> None:
    preprocess_cfg = copy.deepcopy(patchtst_smoke_config)
    preprocess_cfg["device"]["prefer_cuda"] = False
    preprocess_cfg["device"]["amp"] = False
    preprocess_cfg["data"].update(
        {
            "source": "csv",
            "path": str(Path(__file__).resolve().parents[1] / "smoke" / "data" / "fdc_dummy.csv"),
            "timestamp_col": "timestamp",
            "seq_len": 16,
            "seq_stride": 8,
            "normalization": "robust",
        }
    )
    preprocess_cfg["dqvl"] = {
        "enabled": True,
        "allow_sort_fix": False,
        "report_dir": str(tmp_path / "dqvl_fdc"),
        "hard_fail": {
            "require_timestamp": True,
            "invalid_timestamp": True,
            "max_missing_ratio": 0.8,
        },
        "warn": {
            "missing_ratio": 0.2,
            "stuck_std": 1.0e-8,
            "jump_ratio": 0.9,
        },
    }
    preprocess_cfg["training"]["batch_size"] = 2

    preprocess_cfg_path = tmp_path / "patchtst_runtime.yaml"
    _write_yaml(preprocess_cfg_path, preprocess_cfg)

    scaler = ChannelScaler(method="robust")
    scaler.fit(
        torch_module.randn(6, 16, 3, dtype=torch_module.float32).numpy()
    )
    scaler_path = tmp_path / "artifacts" / "scaler_fdc.json"
    scaler.save(scaler_path)

    model = PatchTSTSSL(
        seq_len=int(preprocess_cfg["data"]["seq_len"]),
        patch_len=int(preprocess_cfg["model"]["patch_len"]),
        patch_stride=int(preprocess_cfg["model"]["patch_stride"]),
        d_model=int(preprocess_cfg["model"]["d_model"]),
        nhead=int(preprocess_cfg["model"]["nhead"]),
        num_layers=int(preprocess_cfg["model"]["num_layers"]),
        ff_dim=int(preprocess_cfg["model"]["ff_dim"]),
        dropout=float(preprocess_cfg["model"]["dropout"]),
        mask_ratio=float(preprocess_cfg["model"]["mask_ratio"]),
    )
    checkpoint_path = tmp_path / "checkpoints" / "patchtst_ssl.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    thresholds_path = tmp_path / "artifacts" / "thresholds.json"
    _write_json(
        thresholds_path,
        {"patchtst": {"warn": 0.5, "anomaly": 0.7}},
    )

    runtime_cfg = {
        "run": {
            "run_id": "score-only-test",
            "stream": "patchtst",
            "input_paths": {
                "patchtst": str(Path(__file__).resolve().parents[1] / "smoke" / "data" / "fdc_dummy.csv"),
                "swinmae": None,
            },
            "artifact_paths": {
                "thresholds": str(thresholds_path),
                "patchtst_checkpoint": str(checkpoint_path),
                "swinmae_checkpoint": None,
                "scaler": str(scaler_path),
            },
            "output_dir": str(tmp_path / "artifacts" / "batch_decision"),
        },
        "preprocess": {
            "patchtst_config": str(preprocess_cfg_path),
            "swinmae_config": "unused.yaml",
        },
    }
    runtime_cfg_path = tmp_path / "batch_runtime.yaml"
    _write_yaml(runtime_cfg_path, runtime_cfg)

    rc = main(["--config", str(runtime_cfg_path), "--score-only"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "score-only run completed" in captured.out
    assert "patchtst_windows=" in captured.out
