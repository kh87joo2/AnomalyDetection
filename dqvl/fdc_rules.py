from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from datasets.readers.fdc_reader import FDCReadResult
from dqvl.report import DQVLReport, build_report


def _as_numeric_timestamps(timestamps: np.ndarray) -> tuple[np.ndarray, int]:
    series = pd.Series(timestamps)
    dt = pd.to_datetime(series, errors="coerce")

    if int(dt.notna().sum()) > 0:
        invalid = int(dt.isna().sum())
        arr = dt.astype("int64").to_numpy(dtype=np.float64)
        if invalid > 0:
            arr[dt.isna().to_numpy()] = np.nan
        return arr, invalid

    num = pd.to_numeric(series, errors="coerce")
    arr = num.to_numpy(dtype=np.float64)
    invalid = int(np.isnan(arr).sum())
    return arr, invalid


def evaluate_fdc_quality(
    sample: FDCReadResult,
    dqvl_cfg: dict[str, Any],
    run_id: str,
) -> DQVLReport:
    hard_cfg = dqvl_cfg.get("hard_fail", {}) if isinstance(dqvl_cfg, dict) else {}
    warn_cfg = dqvl_cfg.get("warn", {}) if isinstance(dqvl_cfg, dict) else {}

    hard_fails: list[str] = []
    warnings: list[str] = []

    values = sample.values
    rows = int(values.shape[0])
    channels = int(values.shape[1]) if values.ndim == 2 else 0

    if values.ndim != 2:
        hard_fails.append(f"invalid_value_shape={values.shape}")

    if channels == 0:
        hard_fails.append("no_feature_columns")

    require_timestamp = bool(hard_cfg.get("require_timestamp", True))
    allow_sort_fix = bool(dqvl_cfg.get("allow_sort_fix", False))

    invalid_timestamp_count = 0
    duplicate_timestamp_count = 0
    out_of_order_count = 0

    if sample.timestamps is None:
        if require_timestamp:
            hard_fails.append("missing_timestamp_column")
    else:
        ts_numeric, invalid_timestamp_count = _as_numeric_timestamps(sample.timestamps)
        if invalid_timestamp_count > 0 and bool(hard_cfg.get("invalid_timestamp", True)):
            hard_fails.append(f"invalid_timestamp_count={invalid_timestamp_count}")

        if ts_numeric.size >= 2:
            diffs = np.diff(ts_numeric)
            valid_diffs = diffs[~np.isnan(diffs)]
            duplicate_timestamp_count = int(np.sum(valid_diffs == 0))
            out_of_order_count = int(np.sum(valid_diffs < 0))
            non_increasing = duplicate_timestamp_count + out_of_order_count
            if non_increasing > 0:
                msg = f"timestamp_non_increasing={non_increasing}"
                if allow_sort_fix:
                    warnings.append(msg)
                else:
                    hard_fails.append(msg)

    missing_ratio = float(np.isnan(values).mean()) if values.size > 0 else 1.0
    hard_missing_ratio = float(hard_cfg.get("max_missing_ratio", 0.50))
    warn_missing_ratio = float(warn_cfg.get("missing_ratio", 0.05))

    if missing_ratio > hard_missing_ratio:
        hard_fails.append(f"missing_ratio={missing_ratio:.6f}>hard({hard_missing_ratio:.6f})")
    elif missing_ratio > warn_missing_ratio:
        warnings.append(f"missing_ratio={missing_ratio:.6f}>warn({warn_missing_ratio:.6f})")

    stuck_channels_count = 0
    jump_ratio = 0.0

    if values.size > 0 and rows > 0:
        with np.errstate(invalid="ignore"):
            std_per_channel = np.nanstd(values, axis=0)
        stuck_std = float(warn_cfg.get("stuck_std", 1e-8))
        stuck_channels_count = int(np.sum(std_per_channel <= stuck_std))
        if stuck_channels_count > 0:
            warnings.append(f"stuck_channels={stuck_channels_count}")

    if values.size > 0 and rows > 1:
        deltas = np.abs(np.diff(values, axis=0))
        with np.errstate(invalid="ignore"):
            p99 = np.nanpercentile(deltas, 99, axis=0)
        p99 = np.nan_to_num(p99, nan=np.inf)
        high_jump = deltas > p99.reshape(1, -1)
        jump_ratio = float(np.mean(high_jump))

        warn_jump_ratio = float(warn_cfg.get("jump_ratio", 0.20))
        if jump_ratio > warn_jump_ratio:
            warnings.append(f"jump_ratio={jump_ratio:.6f}>warn({warn_jump_ratio:.6f})")

    metrics: dict[str, Any] = {
        "row_count": rows,
        "channel_count": channels,
        "missing_ratio": missing_ratio,
        "invalid_timestamp_count": invalid_timestamp_count,
        "duplicate_timestamp_count": duplicate_timestamp_count,
        "out_of_order_count": out_of_order_count,
        "stuck_channels_count": stuck_channels_count,
        "jump_ratio": jump_ratio,
    }

    return build_report(
        run_id=run_id,
        file_id=sample.file_id,
        hard_fails=hard_fails,
        warnings=warnings,
        metrics=metrics,
    )
