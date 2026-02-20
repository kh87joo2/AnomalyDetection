from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, TypedDict

SCHEMA_VERSION = "0.1"


class DQVLReport(TypedDict):
    schema_version: str
    run_id: str
    file_id: str
    decision: str
    hard_fails: list[str]
    warnings: list[str]
    metrics: dict[str, Any]


def new_run_id() -> str:
    return uuid.uuid4().hex


def build_report(
    *,
    run_id: str,
    file_id: str,
    hard_fails: list[str],
    warnings: list[str],
    metrics: dict[str, Any],
) -> DQVLReport:
    decision = "drop" if hard_fails else "keep"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "file_id": file_id,
        "decision": decision,
        "hard_fails": hard_fails,
        "warnings": warnings,
        "metrics": metrics,
    }


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


def save_report(report: DQVLReport, output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{_safe_name(report['run_id'])}__{_safe_name(report['file_id'])}.json"
    out_path = out_dir / filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return out_path
