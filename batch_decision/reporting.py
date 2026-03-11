from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from batch_decision.contracts import BatchDecisionResult, ReportArtifacts


def _ensure_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _write_events_csv(path: Path, result: BatchDecisionResult) -> None:
    fieldnames = [
        "event_id",
        "stream",
        "timestamp",
        "window_index",
        "decision",
        "reason",
        "fused_score",
        "warn_threshold",
        "anomaly_threshold",
        "patchtst_score",
        "swinmae_score",
        "patchtst_file_id",
        "swinmae_file_id",
        "aux_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in result.events:
            writer.writerow(
                {
                    "event_id": event.event_id,
                    "stream": event.stream,
                    "timestamp": event.timestamp,
                    "window_index": event.window_index,
                    "decision": event.decision,
                    "reason": event.reason,
                    "fused_score": f"{event.fused_score:.10f}",
                    "warn_threshold": f"{event.thresholds.warn:.10f}",
                    "anomaly_threshold": f"{event.thresholds.anomaly:.10f}",
                    "patchtst_score": (
                        f"{event.stream_scores['patchtst']:.10f}"
                        if "patchtst" in event.stream_scores
                        else ""
                    ),
                    "swinmae_score": (
                        f"{event.stream_scores['swinmae']:.10f}"
                        if "swinmae" in event.stream_scores
                        else ""
                    ),
                    "patchtst_file_id": event.file_ids.get("patchtst", ""),
                    "swinmae_file_id": event.file_ids.get("swinmae", ""),
                    "aux_json": json.dumps(event.aux, ensure_ascii=True, sort_keys=True),
                }
            )


def export_report(result_payload: BatchDecisionResult, *, output_dir: str) -> ReportArtifacts:
    out_dir = _ensure_output_dir(output_dir)
    report_json_path = out_dir / "decision_report.json"
    events_csv_path = out_dir / "decision_events.csv"
    chart_json_path = out_dir / "chart_payload.json"

    report_payload = {
        "run_id": result_payload.run_id,
        "stream": result_payload.stream,
        "summary": result_payload.summary,
        "metadata": result_payload.metadata,
        "events": [asdict(event) for event in result_payload.events],
        "chart_payload": result_payload.chart_payload,
    }
    _write_json(report_json_path, report_payload)
    _write_json(chart_json_path, result_payload.chart_payload)
    _write_events_csv(events_csv_path, result_payload)

    return ReportArtifacts(
        output_dir=out_dir,
        report_json_path=report_json_path,
        events_csv_path=events_csv_path,
        chart_json_path=chart_json_path,
    )
