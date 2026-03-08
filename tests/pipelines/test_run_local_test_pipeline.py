from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from batch_decision.contracts import WindowScore
from batch_decision.decision_engine import DecisionThresholds, decide_records
from batch_decision.preprocess import prepare_patchtst_batch
from batch_decision.reporting import export_report
from datasets.transforms.fdc_normalization import ChannelScaler
from models.patchtst.patchtst_ssl import PatchTSTSSL
from pipelines.run_local_test_pipeline import main


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_decide_records_assigns_expected_states() -> None:
    thresholds = DecisionThresholds(stream="patchtst", warn=0.5, anomaly=0.8)
    records = [
        WindowScore(
            event_id="patchtst:000000",
            stream="patchtst",
            file_id="a.csv",
            timestamp="2026-01-01T00:00:00",
            window_index=0,
            score=0.2,
            aux={},
        ),
        WindowScore(
            event_id="patchtst:000001",
            stream="patchtst",
            file_id="a.csv",
            timestamp="2026-01-01T00:00:01",
            window_index=1,
            score=0.6,
            aux={},
        ),
        WindowScore(
            event_id="patchtst:000002",
            stream="patchtst",
            file_id="a.csv",
            timestamp="2026-01-01T00:00:02",
            window_index=2,
            score=0.9,
            aux={},
        ),
    ]

    payload = decide_records(records, thresholds=thresholds)
    decisions = [item["decision"] for item in payload["records"]]
    assert decisions == ["normal", "warn", "anomaly"]
    assert payload["summary"]["decision_counts"] == {
        "normal": 1,
        "warn": 1,
        "anomaly": 1,
    }


def test_export_report_writes_summary_and_stream_files(tmp_path: Path) -> None:
    payload = {
        "run_id": "unit-run",
        "stream": "patchtst",
        "thresholds_path": "artifacts/thresholds.json",
        "streams": {
            "patchtst": {
                "summary": {"window_count": 1},
                "records": [
                    {
                        "event_id": "patchtst:000000",
                        "stream": "patchtst",
                        "file_id": "a.csv",
                        "timestamp": None,
                        "window_index": 0,
                        "score": 0.4,
                        "decision": "normal",
                        "reason": "score=0.400000 < warn(0.500000)",
                        "warn_threshold": 0.5,
                        "anomaly_threshold": 0.8,
                    }
                ],
            }
        },
    }

    manifest = export_report(payload, output_dir=str(tmp_path / "reports"))
    assert Path(manifest["summary_json"]).exists()
    assert Path(manifest["streams"]["patchtst"]["json"]).exists()
    assert Path(manifest["streams"]["patchtst"]["csv"]).exists()


def test_run_local_test_pipeline_patchtst_executes_with_temp_artifacts(
    torch_module,
    patchtst_smoke_config,
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    cfg = copy.deepcopy(patchtst_smoke_config)
    cfg["device"]["prefer_cuda"] = False
    cfg["device"]["amp"] = False
    cfg["data"].update(
        {
            "source": "csv",
            "path": str(Path(__file__).resolve().parents[1] / "smoke" / "data" / "fdc_dummy.csv"),
            "timestamp_col": "timestamp",
            "seq_len": 16,
            "seq_stride": 8,
            "normalization": "robust",
        }
    )
    cfg["dqvl"] = {
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
    cfg["training"]["batch_size"] = 2

    patch_cfg_path = tmp_path / "configs" / "patchtst_ssl_local.yaml"
    _write_yaml(patch_cfg_path, cfg)
    _write_yaml(
        tmp_path / "configs" / "swinmae_ssl_local.yaml",
        {"seed": 42, "data": {"source": "csv", "path": "unused.csv"}},
    )

    prepared = prepare_patchtst_batch(cfg)
    scaler = ChannelScaler(method=str(cfg["data"].get("normalization", "robust")))
    scaler.fit(prepared.windows)
    scaler_path = tmp_path / "artifacts" / "scaler_fdc.json"
    scaler.save(scaler_path)

    model = PatchTSTSSL(
        seq_len=int(cfg["data"]["seq_len"]),
        patch_len=int(cfg["model"]["patch_len"]),
        patch_stride=int(cfg["model"]["patch_stride"]),
        d_model=int(cfg["model"]["d_model"]),
        nhead=int(cfg["model"]["nhead"]),
        num_layers=int(cfg["model"]["num_layers"]),
        ff_dim=int(cfg["model"]["ff_dim"]),
        dropout=float(cfg["model"]["dropout"]),
        mask_ratio=float(cfg["model"]["mask_ratio"]),
    )
    checkpoint_path = tmp_path / "checkpoints" / "patchtst_ssl.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    thresholds_path = tmp_path / "artifacts" / "thresholds" / "batch_decision_thresholds.json"
    _write_json(
        thresholds_path,
        {
            "patchtst": {"warn": 0.5, "anomaly": 0.7},
            "swinmae": {"warn": 0.5, "anomaly": 0.7},
            "dual": {"warn": 0.6, "anomaly": 0.8},
        },
    )

    runtime_cfg = {
        "run": {
            "run_id": "local-test",
            "stream": "patchtst",
            "input_paths": {
                "patchtst": "data/local/test/fdc/*.csv",
                "swinmae": None,
            },
            "artifact_paths": {
                "thresholds": "artifacts/thresholds/batch_decision_thresholds.json",
                "patchtst_checkpoint": "checkpoints/patchtst_ssl.pt",
                "swinmae_checkpoint": None,
                "scaler": "artifacts/scaler_fdc.json",
            },
            "output_dir": "artifacts/batch_decision/local_gpu",
        },
        "preprocess": {
            "patchtst_config": "configs/patchtst_ssl_local.yaml",
            "swinmae_config": "configs/swinmae_ssl_local.yaml",
        },
    }
    _write_yaml(tmp_path / "configs" / "batch_decision_runtime_local_gpu.yaml", runtime_cfg)

    test_data_dir = tmp_path / "data" / "local" / "test" / "fdc"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_data_dir / "fdc_dummy.csv"
    source_file = Path(__file__).resolve().parents[1] / "smoke" / "data" / "fdc_dummy.csv"
    test_file.write_text(source_file.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.chdir(tmp_path.parent)

    rc = main(
        [
            "--repo-root",
            str(tmp_path),
            "--runtime-config",
            "configs/batch_decision_runtime_local_gpu.yaml",
            "--stream",
            "patchtst",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "[report] summary_json=" in captured.out
    assert "[report] patchtst_decisions=" in captured.out

    summary_path = (
        tmp_path
        / "artifacts"
        / "batch_decision"
        / "local_gpu"
        / "local-test"
        / "summary.json"
    )
    effective_runtime_cfg = (
        tmp_path / "artifacts" / "runtime_configs" / "batch_decision_runtime_local_gpu.yaml"
    )
    assert summary_path.exists()
    assert effective_runtime_cfg.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runtime_payload = yaml.safe_load(effective_runtime_cfg.read_text(encoding="utf-8"))
    assert "patchtst" in summary["streams"]
    assert summary["streams"]["patchtst"]["summary"]["window_count"] > 0
    assert runtime_payload["run"]["input_paths"]["patchtst"] == str(test_data_dir / "*.csv")
    assert runtime_payload["run"]["artifact_paths"]["thresholds"] == str(thresholds_path)
    assert runtime_payload["run"]["artifact_paths"]["patchtst_checkpoint"] == str(checkpoint_path)
    assert runtime_payload["run"]["artifact_paths"]["scaler"] == str(scaler_path)
    assert runtime_payload["preprocess"]["patchtst_config"] == str(patch_cfg_path)


def test_run_local_test_pipeline_dry_run_prints_output_dir(tmp_path: Path, capsys) -> None:
    runtime_cfg = {
        "run": {
            "run_id": "dry-run-check",
            "stream": "patchtst",
            "input_paths": {
                "patchtst": "data/local/test/fdc/*.csv",
                "swinmae": None,
            },
            "artifact_paths": {
                "thresholds": "artifacts/thresholds/batch_decision_thresholds.json",
                "patchtst_checkpoint": "checkpoints/patchtst_ssl.pt",
                "swinmae_checkpoint": None,
                "scaler": "artifacts/scaler_fdc.json",
            },
            "output_dir": "artifacts/batch_decision/local_gpu",
        },
        "preprocess": {
            "patchtst_config": "configs/patchtst_ssl_local.yaml",
            "swinmae_config": "configs/swinmae_ssl_local.yaml",
        },
    }

    _write_json(
        tmp_path / "artifacts" / "thresholds" / "batch_decision_thresholds.json",
        {
            "patchtst": {"warn": 0.5, "anomaly": 0.7},
            "swinmae": {"warn": 0.5, "anomaly": 0.7},
            "dual": {"warn": 0.6, "anomaly": 0.8},
        },
    )
    _write_yaml(
        tmp_path / "configs" / "patchtst_ssl_local.yaml",
        {"seed": 42, "data": {"source": "csv", "path": "unused.csv"}},
    )
    _write_yaml(
        tmp_path / "configs" / "swinmae_ssl_local.yaml",
        {"seed": 42, "data": {"source": "csv", "path": "unused.csv"}},
    )
    _write_yaml(tmp_path / "configs" / "batch_decision_runtime_local_gpu.yaml", runtime_cfg)

    sample_file = tmp_path / "data" / "local" / "test" / "fdc" / "dummy.csv"
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sample_file.write_text("timestamp,a,b\n1,0.1,0.2\n", encoding="utf-8")

    rc = main(
        [
            "--repo-root",
            str(tmp_path),
            "--runtime-config",
            "configs/batch_decision_runtime_local_gpu.yaml",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "[config] output_dir=" in captured.out
    assert "dry-run-check" in captured.out
