from __future__ import annotations

from typing import Any

import numpy as np

from datasets.readers.vib_reader import VibReadResult
from dqvl.report import DQVLReport, build_report


def evaluate_vibration_quality(
    sample: VibReadResult,
    dqvl_cfg: dict[str, Any],
    run_id: str,
    expected_fs: float | None,
    resample_enabled: bool,
) -> DQVLReport:
    hard_cfg = dqvl_cfg.get("hard_fail", {}) if isinstance(dqvl_cfg, dict) else {}
    warn_cfg = dqvl_cfg.get("warn", {}) if isinstance(dqvl_cfg, dict) else {}

    hard_fails: list[str] = []
    warnings: list[str] = []

    values = sample.values
    rows = int(values.shape[0]) if values.ndim == 2 else 0

    if sample.missing_axes:
        hard_fails.append(f"missing_axes={','.join(sample.missing_axes)}")

    if values.ndim != 2 or values.shape[1] != 3:
        hard_fails.append(f"invalid_value_shape={sample.raw_shape}")

    missing_ratio = float(np.isnan(values).mean()) if values.size > 0 else 1.0
    hard_missing_ratio = float(hard_cfg.get("max_missing_ratio", 0.50))
    warn_missing_ratio = float(warn_cfg.get("missing_ratio", 0.05))

    if missing_ratio > hard_missing_ratio:
        hard_fails.append(f"missing_ratio={missing_ratio:.6f}>hard({hard_missing_ratio:.6f})")
    elif missing_ratio > warn_missing_ratio:
        warnings.append(f"missing_ratio={missing_ratio:.6f}>warn({warn_missing_ratio:.6f})")

    actual_fs = sample.fs
    if expected_fs is not None:
        if actual_fs is None:
            if bool(hard_cfg.get("missing_fs", False)):
                hard_fails.append("missing_actual_fs")
            else:
                warnings.append("missing_actual_fs")
        else:
            fs_tol = float(hard_cfg.get("fs_tol", 1e-6))
            if abs(float(actual_fs) - float(expected_fs)) > fs_tol:
                msg = f"fs_mismatch=actual({actual_fs})_expected({expected_fs})"
                if resample_enabled:
                    warnings.append(msg)
                else:
                    hard_fails.append(msg)

    clipping_ratio = 0.0
    flat_ratio = 0.0
    rms_per_axis: list[float] = [0.0, 0.0, 0.0]

    if values.size > 0 and rows > 0:
        axis_min = np.nanmin(values, axis=0)
        axis_max = np.nanmax(values, axis=0)
        at_min = np.isclose(values, axis_min.reshape(1, -1), atol=1e-7, rtol=0.0)
        at_max = np.isclose(values, axis_max.reshape(1, -1), atol=1e-7, rtol=0.0)
        clipping_ratio = float(np.mean(np.logical_or(at_min, at_max)))

        warn_clipping = float(warn_cfg.get("clipping_ratio", 0.05))
        if clipping_ratio > warn_clipping:
            warnings.append(f"clipping_ratio={clipping_ratio:.6f}>warn({warn_clipping:.6f})")

        with np.errstate(invalid="ignore"):
            rms = np.sqrt(np.nanmean(values**2, axis=0))
        rms = np.nan_to_num(rms, nan=0.0)
        rms_per_axis = [float(x) for x in rms.tolist()]

        rms_min = float(warn_cfg.get("rms_min", 1e-6))
        rms_max = float(warn_cfg.get("rms_max", 1e6))
        if bool(np.any(rms < rms_min)):
            warnings.append(f"rms_below_min={rms_per_axis}")
        if bool(np.any(rms > rms_max)):
            warnings.append(f"rms_above_max={rms_per_axis}")

    if values.size > 0 and rows > 1:
        diff = np.abs(np.diff(values, axis=0))
        flat_eps = float(warn_cfg.get("flat_eps", 1e-6))
        flat_ratio = float(np.mean(diff <= flat_eps))

        warn_flat_ratio = float(warn_cfg.get("flat_ratio", 0.30))
        if flat_ratio > warn_flat_ratio:
            warnings.append(f"flat_ratio={flat_ratio:.6f}>warn({warn_flat_ratio:.6f})")

    metrics: dict[str, Any] = {
        "row_count": rows,
        "missing_ratio": missing_ratio,
        "clipping_ratio": clipping_ratio,
        "flat_ratio": flat_ratio,
        "rms_per_axis": rms_per_axis,
        "expected_fs": expected_fs,
        "actual_fs": actual_fs,
        "raw_shape": list(sample.raw_shape),
    }

    return build_report(
        run_id=run_id,
        file_id=sample.file_id,
        hard_fails=hard_fails,
        warnings=warnings,
        metrics=metrics,
    )
