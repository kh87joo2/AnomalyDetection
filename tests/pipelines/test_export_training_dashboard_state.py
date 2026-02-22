from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.export_training_dashboard_state import (
    export_dashboard_state,
    validate_dashboard_state_schema,
)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _prepare_repo(root: Path) -> None:
    _write_bytes(root / "checkpoints/patchtst_ssl.pt", b"patch-ckpt")
    _write_bytes(root / "checkpoints/swinmae_ssl.pt", b"swin-ckpt")
    _write_text(
        root / "artifacts/scaler_fdc.json",
        json.dumps(
            {
                "method": "robust",
                "center": [0.0],
                "scale": [1.0],
                "eps": 1e-6,
            }
        ),
    )
    _write_bytes(root / "runs/patchtst_ssl/events.out.tfevents.test", b"evt")
    _write_bytes(root / "runs/swinmae_ssl/events.out.tfevents.test", b"evt")
    _write_text(
        root / "configs/patchtst_ssl_real.yaml",
        "\n".join(
            [
                "data:",
                "  source: csv",
                "  path: data/fdc.csv",
                "training:",
                "  lr: 0.0001",
                "  epochs: 10",
                "model:",
                "  mask_ratio: 0.4",
                "device:",
                "  amp: false",
                "",
            ]
        ),
    )
    _write_text(
        root / "configs/swinmae_ssl_real.yaml",
        "\n".join(
            [
                "data:",
                "  source: csv",
                "  path: data/vib.csv",
                "  fs: 2000",
                "training:",
                "  lr: 0.0001",
                "  epochs: 10",
                "model:",
                "  mask_ratio: 0.3",
                "device:",
                "  amp: false",
                "",
            ]
        ),
    )
    _write_text(
        root / "docs/calibration_split_policy.md",
        "train/calibration(normal) split policy for dashboard export tests.",
    )
    _write_text(
        root / "training_dashboard/data/dashboard-layout.json",
        json.dumps(
            {
                "meta": {"title": "Test"},
                "views": [
                    {
                        "id": "v1",
                        "name": "Training Flow",
                        "nodes": [
                            {"id": "orchestrator"},
                            {"id": "patchtst"},
                            {"id": "swinmae"},
                            {"id": "validation-gate"},
                        ],
                        "connections": [],
                    }
                ],
            }
        ),
    )
    _write_text(
        root / "artifacts/loss/patchtst_loss_history.csv",
        "epoch,train_loss,val_loss\n1,1.0,1.2\n2,0.8,1.0\n",
    )
    _write_text(
        root / "artifacts/loss/swinmae_loss_history.csv",
        "epoch,train_loss,val_loss\n1,0.9,1.1\n2,0.7,0.95\n",
    )


def test_export_dashboard_state_generates_required_contract(tmp_path: Path) -> None:
    _prepare_repo(tmp_path)
    out_path = Path("training_dashboard/data/dashboard-state.json")

    payload = export_dashboard_state(
        repo_root=tmp_path,
        out_path=out_path,
        run_id="test-run",
        run_smoke=False,
    )

    output_file = tmp_path / out_path
    assert output_file.exists()

    loaded = json.loads(output_file.read_text(encoding="utf-8"))
    assert loaded["meta"]["run_id"] == "test-run"
    assert set(loaded.keys()) >= {"meta", "nodes", "checklist", "metrics", "artifacts"}
    assert len(loaded["checklist"]) == 7
    assert loaded["metrics"]["patchtst"]["loss"][0]["epoch"] == 1
    assert loaded["metrics"]["swinmae"]["config"]["fs"] == 2000
    assert loaded["nodes"]["orchestrator"]["status"] in {"done", "fail", "idle", "running"}
    assert loaded["artifacts"]["logs"]["patchtst"]["event_files"] == 1
    assert payload == loaded


def test_validate_dashboard_state_schema_rejects_missing_keys() -> None:
    with pytest.raises(ValueError, match="missing top-level keys"):
        validate_dashboard_state_schema({"meta": {}})
