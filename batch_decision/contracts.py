from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

StreamName = Literal["patchtst", "swinmae", "dual"]


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
