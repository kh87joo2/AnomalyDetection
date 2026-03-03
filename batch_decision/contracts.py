from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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

