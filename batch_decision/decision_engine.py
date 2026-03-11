from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from batch_decision.contracts import (
    BatchDecisionResult,
    BatchScorePayload,
    DecisionEvent,
    DecisionLabel,
    StreamName,
    ThresholdSpec,
)


class BatchDecisionError(ValueError):
    pass


def _load_threshold_payload(
    thresholds: dict[str, Any] | None,
    thresholds_path: str | None,
) -> dict[str, Any]:
    if thresholds is not None:
        return thresholds
    if not thresholds_path:
        raise BatchDecisionError("Decision engine requires thresholds mapping or thresholds_path")
    path = Path(thresholds_path)
    if not path.exists():
        raise FileNotFoundError(f"Threshold file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise BatchDecisionError("Threshold file root must be a mapping")
    return payload


def _threshold_spec(payload: dict[str, Any], stream: StreamName) -> ThresholdSpec:
    section = payload.get(stream)
    if not isinstance(section, dict):
        raise BatchDecisionError(f"Threshold section missing for stream={stream}")
    warn = section.get("warn")
    anomaly = section.get("anomaly")
    if not isinstance(warn, (int, float)) or not isinstance(anomaly, (int, float)):
        raise BatchDecisionError(f"Threshold section for stream={stream} requires numeric warn/anomaly")
    if float(warn) >= float(anomaly):
        raise BatchDecisionError(f"Invalid thresholds for stream={stream}: warn must be < anomaly")
    return ThresholdSpec(warn=float(warn), anomaly=float(anomaly))


def _classify(score: float, spec: ThresholdSpec) -> DecisionLabel:
    if score >= spec.anomaly:
        return "anomaly"
    if score >= spec.warn:
        return "warn"
    return "normal"


def _single_reason(stream: str, score: float, spec: ThresholdSpec, decision: DecisionLabel) -> str:
    if decision == "anomaly":
        return f"{stream} score={score:.6f} >= anomaly({spec.anomaly:.6f})"
    if decision == "warn":
        return (
            f"{stream} score={score:.6f} >= warn({spec.warn:.6f}) "
            f"and < anomaly({spec.anomaly:.6f})"
        )
    return f"{stream} score={score:.6f} < warn({spec.warn:.6f})"


def _dual_reason(
    patch_score: float,
    swin_score: float,
    fused_score: float,
    spec: ThresholdSpec,
    decision: DecisionLabel,
) -> str:
    dominant_stream = "patchtst" if patch_score >= swin_score else "swinmae"
    if decision == "anomaly":
        boundary = f"anomaly({spec.anomaly:.6f})"
    elif decision == "warn":
        boundary = f"warn({spec.warn:.6f})"
    else:
        boundary = f"warn({spec.warn:.6f})"
    comparator = ">=" if decision in {"warn", "anomaly"} else "<"
    return (
        f"dual fused_score={fused_score:.6f} {comparator} {boundary}; "
        f"patchtst={patch_score:.6f}, swinmae={swin_score:.6f}, dominant={dominant_stream}"
    )


def _build_summary(events: list[DecisionEvent], *, stream: StreamName) -> dict[str, Any]:
    counts = {"normal": 0, "warn": 0, "anomaly": 0}
    for event in events:
        counts[event.decision] += 1

    fused_scores = [event.fused_score for event in events]
    summary: dict[str, Any] = {
        "stream": stream,
        "total_events": len(events),
        "decision_counts": counts,
        "max_fused_score": max(fused_scores) if fused_scores else None,
        "mean_fused_score": (sum(fused_scores) / len(fused_scores)) if fused_scores else None,
    }

    stream_names = sorted({key for event in events for key in event.stream_scores})
    per_stream: dict[str, Any] = {}
    for name in stream_names:
        values = [event.stream_scores[name] for event in events if name in event.stream_scores]
        per_stream[name] = {
            "count": len(values),
            "max": max(values) if values else None,
            "mean": (sum(values) / len(values)) if values else None,
        }
    summary["per_stream"] = per_stream
    return summary


def _build_chart_payload(events: list[DecisionEvent], spec: ThresholdSpec) -> dict[str, Any]:
    return {
        "index": [event.window_index for event in events],
        "timestamp": [event.timestamp for event in events],
        "decision": [event.decision for event in events],
        "fused_score": [event.fused_score for event in events],
        "stream_scores": {
            "patchtst": [event.stream_scores.get("patchtst") for event in events],
            "swinmae": [event.stream_scores.get("swinmae") for event in events],
        },
        "thresholds": {
            "warn": spec.warn,
            "anomaly": spec.anomaly,
        },
    }


def _single_stream_events(
    *,
    stream: StreamName,
    scores: BatchScorePayload,
    spec: ThresholdSpec,
) -> list[DecisionEvent]:
    source = scores.patchtst_records if stream == "patchtst" else scores.swinmae_records
    events: list[DecisionEvent] = []
    for record in source:
        decision = _classify(record.score, spec)
        events.append(
            DecisionEvent(
                event_id=record.event_id,
                stream=stream,
                timestamp=record.timestamp,
                window_index=record.window_index,
                decision=decision,
                reason=_single_reason(stream, record.score, spec, decision),
                thresholds=spec,
                fused_score=record.score,
                stream_scores={stream: record.score},
                file_ids={stream: record.file_id},
                aux=dict(record.aux),
            )
        )
    return events


def _dual_events(scores: BatchScorePayload, spec: ThresholdSpec) -> tuple[list[DecisionEvent], dict[str, Any]]:
    patch_records = scores.patchtst_records
    swin_records = scores.swinmae_records
    pair_count = min(len(patch_records), len(swin_records))
    events: list[DecisionEvent] = []
    for idx in range(pair_count):
        patch = patch_records[idx]
        swin = swin_records[idx]
        fused_score = max(patch.score, swin.score)
        decision = _classify(fused_score, spec)
        events.append(
            DecisionEvent(
                event_id=f"dual:{idx:06d}",
                stream="dual",
                timestamp=patch.timestamp or swin.timestamp,
                window_index=idx,
                decision=decision,
                reason=_dual_reason(patch.score, swin.score, fused_score, spec, decision),
                thresholds=spec,
                fused_score=fused_score,
                stream_scores={
                    "patchtst": patch.score,
                    "swinmae": swin.score,
                },
                file_ids={
                    "patchtst": patch.file_id,
                    "swinmae": swin.file_id,
                },
                aux={
                    "patchtst": dict(patch.aux),
                    "swinmae": dict(swin.aux),
                },
            )
        )
    return events, {
        "paired_events": pair_count,
        "unpaired_patchtst": len(patch_records) - pair_count,
        "unpaired_swinmae": len(swin_records) - pair_count,
        "fusion_rule": "max",
    }


def decide(
    scores: BatchScorePayload,
    *,
    thresholds: dict[str, Any] | None = None,
    thresholds_path: str | None = None,
) -> BatchDecisionResult:
    payload = _load_threshold_payload(thresholds, thresholds_path)
    spec = _threshold_spec(payload, scores.stream)

    metadata = dict(scores.metadata)
    if scores.stream == "dual":
        events, dual_meta = _dual_events(scores, spec)
        metadata["dual_alignment"] = dual_meta
    else:
        events = _single_stream_events(stream=scores.stream, scores=scores, spec=spec)

    return BatchDecisionResult(
        run_id=scores.run_id,
        stream=scores.stream,
        events=events,
        summary=_build_summary(events, stream=scores.stream),
        chart_payload=_build_chart_payload(events, spec),
        metadata=metadata,
    )
