from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from batch_decision.contracts import WindowScore

DecisionState = Literal["normal", "warn", "anomaly"]


@dataclass(frozen=True)
class DecisionThresholds:
    stream: str
    warn: float
    anomaly: float


def load_thresholds(path: str | Path, *, stream: str) -> DecisionThresholds:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Threshold file root must be a mapping: {path}")

    stream_payload = payload.get(stream)
    if not isinstance(stream_payload, dict):
        raise ValueError(f"Threshold file missing '{stream}' mapping: {path}")

    warn = stream_payload.get("warn")
    anomaly = stream_payload.get("anomaly")
    if not isinstance(warn, (int, float)) or not isinstance(anomaly, (int, float)):
        raise ValueError(f"Thresholds for '{stream}' must include numeric warn/anomaly: {path}")
    if float(warn) >= float(anomaly):
        raise ValueError(f"Threshold rule invalid for '{stream}': warn must be < anomaly")

    return DecisionThresholds(stream=stream, warn=float(warn), anomaly=float(anomaly))


def decide_score(score: float, *, thresholds: DecisionThresholds) -> tuple[DecisionState, str]:
    if score >= thresholds.anomaly:
        return "anomaly", f"score={score:.6f} >= anomaly({thresholds.anomaly:.6f})"
    if score >= thresholds.warn:
        return "warn", f"score={score:.6f} >= warn({thresholds.warn:.6f})"
    return "normal", f"score={score:.6f} < warn({thresholds.warn:.6f})"


def decide_records(
    scores: list[WindowScore],
    *,
    thresholds: DecisionThresholds,
) -> dict[str, Any]:
    decision_counts = {"normal": 0, "warn": 0, "anomaly": 0}
    records: list[dict[str, Any]] = []
    score_values: list[float] = []

    for record in scores:
        state, reason = decide_score(float(record.score), thresholds=thresholds)
        decision_counts[state] += 1
        score_values.append(float(record.score))
        records.append(
            {
                "event_id": record.event_id,
                "stream": record.stream,
                "file_id": record.file_id,
                "timestamp": record.timestamp,
                "window_index": int(record.window_index),
                "score": float(record.score),
                "decision": state,
                "reason": reason,
                "warn_threshold": float(thresholds.warn),
                "anomaly_threshold": float(thresholds.anomaly),
                "aux": dict(record.aux),
            }
        )

    if score_values:
        mean_score = sum(score_values) / len(score_values)
        score_stats = {
            "min": min(score_values),
            "max": max(score_values),
            "mean": mean_score,
        }
    else:
        score_stats = {
            "min": None,
            "max": None,
            "mean": None,
        }

    return {
        "stream": thresholds.stream,
        "thresholds": {
            "warn": float(thresholds.warn),
            "anomaly": float(thresholds.anomaly),
        },
        "summary": {
            "window_count": len(records),
            "decision_counts": decision_counts,
            "score_stats": score_stats,
        },
        "records": records,
    }
