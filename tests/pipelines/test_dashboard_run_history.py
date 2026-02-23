from __future__ import annotations

import json
from pathlib import Path

from pipelines.export_training_dashboard_state import export_dashboard_state


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
        "train/calibration(normal) split policy for run history tests.",
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


def test_export_persists_run_snapshot_with_normalized_run_id(tmp_path: Path) -> None:
    _prepare_repo(tmp_path)

    payload = export_dashboard_state(
        repo_root=tmp_path,
        out_path=Path("training_dashboard/data/dashboard-state.json"),
        run_id="demo run/01",
        run_smoke=False,
        persist_run_history=True,
    )

    assert payload["meta"]["run_id"] == "demo_run_01"
    run_history_meta = payload["meta"].get("run_history", {})
    assert run_history_meta.get("snapshot_path") == "training_dashboard/data/runs/demo_run_01.json"
    assert run_history_meta.get("index_path") == "training_dashboard/data/runs/index.json"

    snapshot_file = tmp_path / "training_dashboard/data/runs/demo_run_01.json"
    index_file = tmp_path / "training_dashboard/data/runs/index.json"
    assert snapshot_file.exists()
    assert index_file.exists()

    index_payload = json.loads(index_file.read_text(encoding="utf-8"))
    assert isinstance(index_payload.get("runs"), list)
    assert index_payload["runs"][0]["run_id"] == "demo_run_01"
    assert index_payload["runs"][0]["file"] == "demo_run_01.json"


def test_run_history_limit_prunes_old_snapshots(tmp_path: Path) -> None:
    _prepare_repo(tmp_path)

    export_dashboard_state(
        repo_root=tmp_path,
        out_path=Path("training_dashboard/data/dashboard-state.json"),
        run_id="run-a",
        run_smoke=False,
        persist_run_history=True,
        run_history_limit=2,
    )
    export_dashboard_state(
        repo_root=tmp_path,
        out_path=Path("training_dashboard/data/dashboard-state.json"),
        run_id="run-b",
        run_smoke=False,
        persist_run_history=True,
        run_history_limit=2,
    )
    export_dashboard_state(
        repo_root=tmp_path,
        out_path=Path("training_dashboard/data/dashboard-state.json"),
        run_id="run-c",
        run_smoke=False,
        persist_run_history=True,
        run_history_limit=2,
    )

    run_dir = tmp_path / "training_dashboard/data/runs"
    files = sorted(p.name for p in run_dir.glob("*.json"))
    assert "index.json" in files
    assert "run-c.json" in files
    assert "run-b.json" in files
    assert "run-a.json" not in files

    index_payload = json.loads((run_dir / "index.json").read_text(encoding="utf-8"))
    run_ids = [item["run_id"] for item in index_payload["runs"]]
    assert run_ids == ["run-c", "run-b"]
