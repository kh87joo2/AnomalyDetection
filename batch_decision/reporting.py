from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_stream_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event_id",
        "stream",
        "file_id",
        "timestamp",
        "window_index",
        "score",
        "decision",
        "reason",
        "warn_threshold",
        "anomaly_threshold",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name) for name in fieldnames})


def export_report(result_payload: object, *, output_dir: str) -> object:
    if not isinstance(result_payload, dict):
        raise TypeError("result_payload must be a mapping")

    streams = result_payload.get("streams")
    if not isinstance(streams, dict):
        raise ValueError("result_payload.streams must be a mapping")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    _write_json(summary_path, result_payload)

    manifest: dict[str, Any] = {
        "summary_json": str(summary_path),
        "streams": {},
    }
    for stream_name, payload in streams.items():
        if not isinstance(stream_name, str) or not isinstance(payload, dict):
            continue
        records = payload.get("records", [])
        if not isinstance(records, list):
            records = []

        stream_json = out_dir / f"{stream_name}_events.json"
        stream_csv = out_dir / f"{stream_name}_events.csv"
        _write_json(stream_json, payload)
        _write_stream_csv(stream_csv, records)

        manifest["streams"][stream_name] = {
            "json": str(stream_json),
            "csv": str(stream_csv),
            "record_count": len(records),
        }

    return manifest
