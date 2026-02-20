from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

_TIMESTAMP_CANDIDATES = ("timestamp", "time")
_AXIS_NAMES = ("x", "y", "z")


@dataclass(frozen=True)
class VibReadResult:
    file_id: str
    path: Path
    source_type: str
    timestamps: np.ndarray | None
    values: np.ndarray  # expected (T, 3), float32
    missing_axes: list[str]
    raw_shape: tuple[int, ...]
    fs: float | None


def _resolve_timestamp_column(columns: Sequence[str], timestamp_col: str | None) -> str | None:
    if timestamp_col:
        for col in columns:
            if col == timestamp_col:
                return col
        return None

    col_map = {col.lower(): col for col in columns}
    for candidate in _TIMESTAMP_CANDIDATES:
        if candidate in col_map:
            return col_map[candidate]
    return None


def _read_csv(path: Path, timestamp_col: str | None, fs: float | None) -> VibReadResult:
    frame = pd.read_csv(path)
    col_map = {col.lower(): col for col in frame.columns}

    axes: list[np.ndarray] = []
    missing_axes: list[str] = []
    for axis in _AXIS_NAMES:
        if axis in col_map:
            series = pd.to_numeric(frame[col_map[axis]], errors="coerce")
            axes.append(series.to_numpy(dtype=np.float32))
        else:
            missing_axes.append(axis)
            axes.append(np.full((len(frame),), np.nan, dtype=np.float32))

    values = np.stack(axes, axis=-1) if axes else np.empty((0, 3), dtype=np.float32)

    ts_col = _resolve_timestamp_column(list(frame.columns), timestamp_col)
    timestamps = frame[ts_col].to_numpy() if ts_col is not None else None

    return VibReadResult(
        file_id=path.name,
        path=path,
        source_type="csv",
        timestamps=timestamps,
        values=values,
        missing_axes=missing_axes,
        raw_shape=tuple(values.shape),
        fs=fs,
    )


def _read_npy(path: Path, fs: float | None) -> VibReadResult:
    arr = np.load(path)
    raw_shape = tuple(arr.shape)
    missing_axes: list[str] = []

    if arr.ndim == 2 and arr.shape[1] == 3:
        values = arr.astype(np.float32, copy=False)
    else:
        values = np.empty((0, 3), dtype=np.float32)

    return VibReadResult(
        file_id=path.name,
        path=path,
        source_type="npy",
        timestamps=None,
        values=values,
        missing_axes=missing_axes,
        raw_shape=raw_shape,
        fs=fs,
    )


def read_vibration_file(
    path: str | Path,
    fs: float | None = None,
    timestamp_col: str | None = None,
) -> VibReadResult:
    """Read vibration file as-is without row-order mutation."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Vibration file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return _read_csv(file_path, timestamp_col=timestamp_col, fs=fs)
    if suffix == ".npy":
        return _read_npy(file_path, fs=fs)
    raise ValueError(f"Unsupported vibration file type: {file_path}")
