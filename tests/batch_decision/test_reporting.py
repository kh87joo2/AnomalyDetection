from __future__ import annotations

import csv
import json
from pathlib import Path

from batch_decision.contracts import BatchDecisionResult, DecisionEvent, ThresholdSpec
from batch_decision.reporting import export_report


def test_export_report_writes_json_csv_and_chart_payload(tmp_path: Path) -> None:
    result = BatchDecisionResult(
        run_id="report-run",
        stream="patchtst",
        events=[
            DecisionEvent(
                event_id="patchtst:000000",
                stream="patchtst",
                timestamp="2026-03-11T00:00:00",
                window_index=0,
                decision="warn",
                reason="patchtst score=0.700000 >= warn(0.650000) and < anomaly(0.850000)",
                thresholds=ThresholdSpec(warn=0.65, anomaly=0.85),
                fused_score=0.70,
                stream_scores={"patchtst": 0.70},
                file_ids={"patchtst": "fdc_a"},
                aux={"per_channel_error": [0.1, 0.2]},
            )
        ],
        summary={
            "stream": "patchtst",
            "total_events": 1,
            "decision_counts": {"normal": 0, "warn": 1, "anomaly": 0},
            "max_fused_score": 0.70,
            "mean_fused_score": 0.70,
            "per_stream": {"patchtst": {"count": 1, "max": 0.70, "mean": 0.70}},
        },
        chart_payload={
            "index": [0],
            "timestamp": ["2026-03-11T00:00:00"],
            "decision": ["warn"],
            "fused_score": [0.70],
            "stream_scores": {"patchtst": [0.70], "swinmae": [None]},
            "thresholds": {"warn": 0.65, "anomaly": 0.85},
        },
        metadata={"source": "unit-test"},
    )

    artifacts = export_report(result, output_dir=str(tmp_path / "report"))

    assert artifacts.report_json_path.exists()
    assert artifacts.events_csv_path.exists()
    assert artifacts.chart_json_path.exists()

    report_json = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert report_json["run_id"] == "report-run"
    assert report_json["summary"]["decision_counts"]["warn"] == 1

    with artifacts.events_csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["decision"] == "warn"
    assert rows[0]["patchtst_file_id"] == "fdc_a"
