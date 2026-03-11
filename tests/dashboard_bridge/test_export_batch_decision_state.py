from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard_bridge.export_batch_decision_state import (
    export_batch_decision_state,
    validate_batch_decision_state_schema,
)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _prepare_repo(root: Path) -> None:
    _write_text(
        root / "training_dashboard/data/dashboard-layout.json",
        json.dumps(
            {
                "meta": {"title": "Dashboard"},
                "views": [
                    {
                        "id": "training-flow",
                        "name": "Training Flow",
                        "nodes": [{"id": "orchestrator"}],
                        "connections": [],
                    },
                    {
                        "id": "batch-decision",
                        "name": "Batch Decision",
                        "nodes": [
                            {"id": "batch-orchestrator"},
                            {"id": "batch-import"},
                            {"id": "batch-dqvl"},
                            {"id": "batch-patchtst"},
                            {"id": "batch-swinmae"},
                            {"id": "batch-decision"},
                            {"id": "batch-report"},
                            {"id": "batch-bridge"},
                        ],
                        "connections": [],
                    },
                ],
            }
        ),
    )
    report_dir = root / "artifacts/batch_decision/demo"
    _write_text(
        report_dir / "decision_report.json",
        json.dumps(
            {
                "run_id": "batch_demo",
                "stream": "dual",
                "summary": {
                    "stream": "dual",
                    "total_events": 3,
                    "decision_counts": {"normal": 1, "warn": 1, "anomaly": 1},
                    "max_fused_score": 0.91,
                    "mean_fused_score": 0.61,
                },
                "metadata": {
                    "patchtst": {"scored_windows": 3, "dqvl_reports": 1},
                    "swinmae": {"scored_windows": 3, "dqvl_reports": 1},
                },
                "events": [
                    {
                        "event_id": "dual:000000",
                        "decision": "normal",
                        "timestamp": "2026-03-11T00:00:00",
                        "fused_score": 0.2,
                        "reason": "demo",
                    },
                    {
                        "event_id": "dual:000001",
                        "decision": "warn",
                        "timestamp": "2026-03-11T00:01:00",
                        "fused_score": 0.7,
                        "reason": "demo",
                    },
                ],
                "chart_payload": {
                    "index": [0, 1, 2],
                    "timestamp": ["a", "b", "c"],
                    "decision": ["normal", "warn", "anomaly"],
                    "fused_score": [0.2, 0.7, 0.91],
                    "stream_scores": {
                        "patchtst": [0.1, 0.65, 0.4],
                        "swinmae": [0.2, 0.55, 0.91],
                    },
                    "thresholds": {"warn": 0.68, "anomaly": 0.88},
                },
            }
        ),
    )
    _write_text(report_dir / "decision_events.csv", "event_id,decision\nx,normal\n")
    _write_text(report_dir / "chart_payload.json", json.dumps({"ok": True}))


def test_export_batch_decision_state_generates_required_contract(tmp_path: Path) -> None:
    _prepare_repo(tmp_path)
    out_path = Path("training_dashboard/data/batch-decision-state.json")

    payload = export_batch_decision_state(
        repo_root=tmp_path,
        out_path=out_path,
    )

    output_file = tmp_path / out_path
    assert output_file.exists()

    loaded = json.loads(output_file.read_text(encoding="utf-8"))
    assert loaded["meta"]["run_id"] == "batch_demo"
    assert loaded["meta"]["dashboard_state_path"].endswith("training_dashboard/data/batch-decision-state.json")
    assert set(loaded.keys()) >= {"meta", "nodes", "summary", "chart", "artifacts"}
    assert loaded["summary"]["decision_counts"]["anomaly"] == 1
    assert loaded["chart"]["thresholds"]["warn"] == 0.68
    assert loaded["nodes"]["batch-bridge"]["status"] == "done"
    assert payload == loaded


def test_validate_batch_decision_state_schema_rejects_missing_keys() -> None:
    with pytest.raises(ValueError, match="missing top-level keys"):
        validate_batch_decision_state_schema({"meta": {}})
