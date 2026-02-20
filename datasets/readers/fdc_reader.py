from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

_TIMESTAMP_CANDIDATES = ("timestamp", "time")


@dataclass(frozen=True)
class FDCReadResult:
    file_id: str
    path: Path
    timestamp_col: str | None
    feature_columns: list[str]
    timestamps: np.ndarray | None
    values: np.ndarray  # (T, C), float32


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


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported FDC file type: {path}")


def read_fdc_file(path: str | Path, timestamp_col: str | None = None) -> FDCReadResult:
    """Read an FDC file as-is without row-order mutation."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"FDC file not found: {file_path}")

    frame = _load_table(file_path)
    ts_col = _resolve_timestamp_column(list(frame.columns), timestamp_col)

    feature_columns = [col for col in frame.columns if col != ts_col]
    if feature_columns:
        numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=np.float32)
    else:
        values = np.empty((len(frame), 0), dtype=np.float32)

    timestamps = frame[ts_col].to_numpy() if ts_col is not None else None

    return FDCReadResult(
        file_id=file_path.name,
        path=file_path,
        timestamp_col=ts_col,
        feature_columns=feature_columns,
        timestamps=timestamps,
        values=values,
    )
