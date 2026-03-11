from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

StreamName = Literal["patchtst", "swinmae", "dual"]
DecisionLabel = Literal["normal", "warn", "anomaly"]


@dataclass(frozen=True)
class InputPaths:
    patchtst: str | None = None
    swinmae: str | None = None


@dataclass(frozen=True)
class ArtifactPaths:
    thresholds_path: str
    patchtst_checkpoint: str | None = None
    swinmae_checkpoint: str | None = None
    scaler_path: str | None = None


@dataclass(frozen=True)
class BatchRunRequest:
    run_id: str
    stream: StreamName
    input_paths: InputPaths
    artifacts: ArtifactPaths
    output_dir: str | None = None


@dataclass(frozen=True)
class ImportedFile:
    file_id: str
    path: Path
    source_type: str
    row_count: int
    timestamps_present: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DQVLRecord:
    file_id: str
    decision: Literal["keep", "drop"]
    report_path: Path | None
    metrics: dict[str, Any]
    hard_fails: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class PreparedBatch:
    stream: Literal["patchtst", "swinmae"]
    windows: np.ndarray
    window_file_ids: list[str]
    window_anchor_timestamps: list[str | None]
    imported_files: list[ImportedFile]
    dqvl_records: list[DQVLRecord]
    skipped_files: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WindowScore:
    event_id: str
    stream: Literal["patchtst", "swinmae"]
    file_id: str
    timestamp: str | None
    window_index: int
    score: float
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StreamScorePayload:
    stream: Literal["patchtst", "swinmae"]
    records: list[WindowScore]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchScorePayload:
    run_id: str
    stream: StreamName
    patchtst_records: list[WindowScore] = field(default_factory=list)
    swinmae_records: list[WindowScore] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ThresholdSpec:
    warn: float
    anomaly: float


@dataclass(frozen=True)
class DecisionEvent:
    event_id: str
    stream: StreamName
    timestamp: str | None
    window_index: int
    decision: DecisionLabel
    reason: str
    thresholds: ThresholdSpec
    fused_score: float
    stream_scores: dict[str, float]
    file_ids: dict[str, str]
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchDecisionResult:
    run_id: str
    stream: StreamName
    events: list[DecisionEvent]
    summary: dict[str, Any]
    chart_payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReportArtifacts:
    output_dir: Path
    report_json_path: Path
    events_csv_path: Path
    chart_json_path: Path
