from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipelines.export_training_dashboard_state import (
    VALID_NODE_STATUS,
    _extract_layout_node_ids,
    _resolve,
    normalize_run_id,
)


def _pick_latest_decision_report(repo_root: Path) -> Path:
    candidates = sorted(
        repo_root.glob("artifacts/batch_decision/**/decision_report.json"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No decision_report.json found under artifacts/batch_decision"
        )
    return candidates[0]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be mapping: {path}")
    return payload


def _preview_events(events: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for item in events[:limit]:
        if not isinstance(item, dict):
            continue
        preview.append(
            {
                "event_id": item.get("event_id"),
                "decision": item.get("decision"),
                "timestamp": item.get("timestamp"),
                "fused_score": item.get("fused_score"),
                "reason": item.get("reason"),
            }
        )
    return preview


def _build_node_statuses(
    *,
    node_ids: set[str],
    stream: str,
    summary: dict[str, Any],
    metadata: dict[str, Any],
    artifacts: dict[str, Any],
    timestamp: str,
) -> dict[str, dict[str, str]]:
    defaults = {
        "batch-orchestrator",
        "batch-import",
        "batch-dqvl",
        "batch-patchtst",
        "batch-swinmae",
        "batch-decision",
        "batch-report",
        "batch-bridge",
    }
    all_node_ids = set(node_ids) | defaults
    total_events = int(summary.get("total_events", 0) or 0)
    counts = summary.get("decision_counts", {}) if isinstance(summary.get("decision_counts"), dict) else {}
    anomaly_count = int(counts.get("anomaly", 0) or 0)
    warn_count = int(counts.get("warn", 0) or 0)

    node_status: dict[str, dict[str, str]] = {}

    def set_status(node_id: str, status: str, message: str) -> None:
        node_status[node_id] = {
            "status": status,
            "message": message,
            "updated_at": timestamp,
        }

    patch_meta = metadata.get("patchtst", {}) if isinstance(metadata.get("patchtst"), dict) else {}
    swin_meta = metadata.get("swinmae", {}) if isinstance(metadata.get("swinmae"), dict) else {}
    patch_present = stream in {"patchtst", "dual"}
    swin_present = stream in {"swinmae", "dual"}
    patch_scored = int(patch_meta.get("scored_windows", 0) or 0)
    swin_scored = int(swin_meta.get("scored_windows", 0) or 0)
    dqvl_reports = int(patch_meta.get("dqvl_reports", 0) or 0) + int(swin_meta.get("dqvl_reports", 0) or 0)

    set_status(
        "batch-orchestrator",
        "done" if total_events > 0 else "fail",
        f"Batch decision run produced {total_events} event(s).",
    )
    set_status(
        "batch-import",
        "done" if total_events > 0 else "fail",
        f"Imported data for stream={stream}.",
    )
    set_status(
        "batch-dqvl",
        "done" if dqvl_reports > 0 else "idle",
        f"DQVL reports: {dqvl_reports}",
    )
    set_status(
        "batch-patchtst",
        "done" if patch_present and patch_scored > 0 else "idle",
        f"PatchTST scored_windows={patch_scored}",
    )
    set_status(
        "batch-swinmae",
        "done" if swin_present and swin_scored > 0 else "idle",
        f"SwinMAE scored_windows={swin_scored}",
    )
    set_status(
        "batch-decision",
        "done" if total_events > 0 else "fail",
        f"Counts normal={counts.get('normal', 0)}, warn={warn_count}, anomaly={anomaly_count}",
    )
    report_json = artifacts["reports"]["report_json"]
    events_csv = artifacts["reports"]["events_csv"]
    chart_json = artifacts["reports"]["chart_json"]
    reports_ready = all(
        bool(item.get("exists"))
        for item in [report_json, events_csv, chart_json]
        if isinstance(item, dict)
    )
    set_status(
        "batch-report",
        "done" if reports_ready else "fail",
        "Decision exports written." if reports_ready else "Decision exports missing.",
    )
    set_status(
        "batch-bridge",
        "done" if total_events > 0 and reports_ready else "fail",
        "Batch decision state ready for dashboard." if total_events > 0 else "No batch decision data yet.",
    )

    for node_id in all_node_ids:
        if node_id not in node_status:
            node_status[node_id] = {
                "status": "idle",
                "message": "No mapped runtime status yet.",
                "updated_at": timestamp,
            }
    return node_status


def validate_batch_decision_state_schema(payload: dict[str, Any]) -> None:
    required_top = {"meta", "nodes", "summary", "chart", "artifacts"}
    missing_top = sorted(required_top - set(payload.keys()))
    if missing_top:
        raise ValueError(f"batch decision state missing top-level keys: {missing_top}")

    meta = payload["meta"]
    if not isinstance(meta, dict):
        raise ValueError("meta must be a mapping")
    for key in ["run_id", "timestamp", "repo_root", "source_report_path"]:
        if key not in meta:
            raise ValueError(f"meta missing key: {key}")

    nodes = payload["nodes"]
    if not isinstance(nodes, dict):
        raise ValueError("nodes must be a mapping")
    for node_id, node in nodes.items():
        if not isinstance(node_id, str):
            raise ValueError("node id must be string")
        if not isinstance(node, dict):
            raise ValueError(f"node status must be mapping: {node_id}")
        for key in ["status", "message", "updated_at"]:
            if key not in node:
                raise ValueError(f"node {node_id} missing key: {key}")
        if node["status"] not in VALID_NODE_STATUS:
            raise ValueError(f"node {node_id} has invalid status: {node['status']}")

    if not isinstance(payload["summary"], dict):
        raise ValueError("summary must be a mapping")
    if not isinstance(payload["chart"], dict):
        raise ValueError("chart must be a mapping")
    if not isinstance(payload["artifacts"], dict):
        raise ValueError("artifacts must be a mapping")


def export_batch_decision_state(
    *,
    repo_root: Path,
    out_path: Path,
    report_json_path: Path | None = None,
    layout_path: Path = Path("training_dashboard/data/dashboard-layout.json"),
) -> dict[str, Any]:
    root = repo_root.resolve()
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    layout_file = _resolve(root, layout_path)
    output = _resolve(root, out_path)

    report_path = (
        _resolve(root, report_json_path)
        if report_json_path is not None
        else _pick_latest_decision_report(root)
    )
    report_payload = _read_json(report_path)

    run_id = normalize_run_id(str(report_payload.get("run_id", ""))) or datetime.now(
        timezone.utc
    ).strftime("batch-%Y%m%d-%H%M%S")
    stream = str(report_payload.get("stream", "dual"))
    summary = report_payload.get("summary", {})
    if not isinstance(summary, dict):
        raise ValueError("decision report summary must be a mapping")
    chart = report_payload.get("chart_payload", {})
    if not isinstance(chart, dict):
        raise ValueError("decision report chart_payload must be a mapping")
    metadata = report_payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    events = report_payload.get("events", [])
    if not isinstance(events, list):
        events = []

    report_dir = report_path.parent
    events_csv_path = report_dir / "decision_events.csv"
    chart_json_path = report_dir / "chart_payload.json"

    artifacts = {
        "reports": {
            "report_json": {
                "path": str(report_path),
                "exists": report_path.exists(),
            },
            "events_csv": {
                "path": str(events_csv_path),
                "exists": events_csv_path.exists(),
            },
            "chart_json": {
                "path": str(chart_json_path),
                "exists": chart_json_path.exists(),
            },
        }
    }

    node_ids = _extract_layout_node_ids(layout_file)
    payload: dict[str, Any] = {
        "meta": {
            "run_id": run_id,
            "timestamp": timestamp,
            "repo_root": str(root),
            "source_report_path": str(report_path),
            "layout_path": str(layout_file),
            "dashboard_state_path": str(output),
        },
        "nodes": _build_node_statuses(
            node_ids=node_ids,
            stream=stream,
            summary=summary,
            metadata=metadata,
            artifacts=artifacts,
            timestamp=timestamp,
        ),
        "summary": summary,
        "chart": chart,
        "artifacts": artifacts,
        "events_preview": _preview_events(events),
    }

    validate_batch_decision_state_schema(payload)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export dashboard runtime state JSON for batch decision view.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("training_dashboard/data/batch-decision-state.json"),
    )
    parser.add_argument("--report-json-path", type=Path, default=None)
    parser.add_argument(
        "--layout-path",
        type=Path,
        default=Path("training_dashboard/data/dashboard-layout.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = export_batch_decision_state(
        repo_root=args.repo_root,
        out_path=args.out,
        report_json_path=args.report_json_path,
        layout_path=args.layout_path,
    )
    print(f"exported: {args.out}")
    print(f"run_id: {payload['meta']['run_id']}")
    print(f"events: {payload['summary'].get('total_events', 0)}")


if __name__ == "__main__":
    main()
