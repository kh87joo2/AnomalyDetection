from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets.fdc_dataset import _resolve_paths as _resolve_fdc_paths
from datasets.readers.fdc_reader import FDCReadResult, read_fdc_file
from datasets.readers.vib_reader import VibReadResult, read_vibration_file
from datasets.vib_dataset import _resolve_paths as _resolve_vibration_paths

from batch_decision.contracts import ImportedFile


class BatchImportError(ValueError):
    pass


def _build_imported_file(
    *,
    file_id: str,
    path: Path,
    source_type: str,
    row_count: int,
    timestamps_present: bool,
    metadata: dict[str, Any] | None = None,
) -> ImportedFile:
    return ImportedFile(
        file_id=file_id,
        path=path,
        source_type=source_type,
        row_count=row_count,
        timestamps_present=timestamps_present,
        metadata=metadata or {},
    )


def load_patchtst_inputs(
    path_cfg: str | list[str],
    *,
    timestamp_col: str | None = None,
) -> tuple[list[FDCReadResult], list[ImportedFile]]:
    file_paths = _resolve_fdc_paths(path_cfg)
    if not file_paths:
        raise FileNotFoundError(f"No files matched patchtst input path={path_cfg}")

    samples: list[FDCReadResult] = []
    imported: list[ImportedFile] = []
    for file_path in file_paths:
        sample = read_fdc_file(file_path, timestamp_col=timestamp_col)
        samples.append(sample)
        imported.append(
            _build_imported_file(
                file_id=sample.file_id,
                path=sample.path,
                source_type=sample.path.suffix.lower().lstrip("."),
                row_count=int(sample.values.shape[0]),
                timestamps_present=sample.timestamps is not None,
                metadata={"feature_columns": list(sample.feature_columns)},
            )
        )
    return samples, imported


def load_swinmae_inputs(
    path_cfg: str | list[str],
    *,
    fs: float | None,
    timestamp_col: str | None = None,
) -> tuple[list[VibReadResult], list[ImportedFile]]:
    file_paths = _resolve_vibration_paths(path_cfg)
    if not file_paths:
        raise FileNotFoundError(f"No files matched swinmae input path={path_cfg}")

    samples: list[VibReadResult] = []
    imported: list[ImportedFile] = []
    for file_path in file_paths:
        sample = read_vibration_file(file_path, fs=fs, timestamp_col=timestamp_col)
        samples.append(sample)
        imported.append(
            _build_imported_file(
                file_id=sample.file_id,
                path=sample.path,
                source_type=sample.source_type,
                row_count=int(sample.values.shape[0]) if sample.values.ndim == 2 else 0,
                timestamps_present=sample.timestamps is not None,
                metadata={
                    "raw_shape": list(sample.raw_shape),
                    "missing_axes": list(sample.missing_axes),
                    "fs": sample.fs,
                },
            )
        )
    return samples, imported

