from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from batch_decision.contracts import DQVLRecord, PreparedBatch
from batch_decision.importers import load_patchtst_inputs, load_swinmae_inputs
from datasets.fdc_dataset import (
    _safe_sliding_windows as _fdc_safe_sliding_windows,
    _to_numeric_timestamps,
)
from datasets.readers.vib_reader import VibReadResult
from datasets.vib_dataset import (
    _actual_fs_for_file,
    _resample_linear,
    _safe_sliding_windows as _vib_safe_sliding_windows,
)
from dqvl.fdc_rules import evaluate_fdc_quality
from dqvl.report import save_report
from dqvl.report import new_run_id
from dqvl.vib_rules import evaluate_vibration_quality


class BatchPreprocessError(ValueError):
    pass


def _build_dqvl_record(report: dict[str, Any], report_path: Path | None) -> DQVLRecord:
    return DQVLRecord(
        file_id=str(report["file_id"]),
        decision=str(report["decision"]),  # type: ignore[arg-type]
        report_path=report_path,
        metrics=dict(report.get("metrics", {})),
        hard_fails=list(report.get("hard_fails", [])),
        warnings=list(report.get("warnings", [])),
    )


def _sanitize_fdc_values(values: np.ndarray) -> np.ndarray:
    x = np.where(np.isfinite(values), values, np.nan).astype(np.float32)
    with np.errstate(invalid="ignore"):
        fill = np.nanmedian(x, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0).astype(np.float32)
    x = np.where(np.isfinite(x), x, fill.reshape(1, -1))
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _sort_fdc_values_and_timestamps(
    values: np.ndarray,
    timestamps: np.ndarray | None,
    *,
    allow_sort_fix: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    if not allow_sort_fix or timestamps is None or values.shape[0] <= 1:
        return values, timestamps

    ts_num = _to_numeric_timestamps(timestamps)
    valid = np.isfinite(ts_num)
    if int(valid.sum()) < 2:
        return values, timestamps

    values_valid = values[valid]
    timestamps_valid = timestamps[valid]
    ts_valid = ts_num[valid]

    order = np.argsort(ts_valid, kind="mergesort")
    values_sorted = values_valid[order]
    timestamps_sorted = timestamps_valid[order]
    ts_sorted = ts_valid[order]

    keep = np.ones(ts_sorted.shape[0], dtype=bool)
    keep[1:] = ts_sorted[1:] != ts_sorted[:-1]
    return values_sorted[keep], timestamps_sorted[keep]


def _build_anchor_timestamps(
    timestamps: np.ndarray | None,
    *,
    window: int,
    stride: int,
    expected_count: int,
) -> list[str | None]:
    if timestamps is None:
        return [None] * expected_count
    anchors: list[str | None] = []
    for start in range(0, len(timestamps) - window + 1, stride):
        anchors.append(str(timestamps[start + window - 1]))
    if len(anchors) != expected_count:
        raise BatchPreprocessError(
            f"Anchor timestamp count mismatch: got={len(anchors)} expected={expected_count}"
        )
    return anchors


def prepare_patchtst_batch(config: dict[str, Any]) -> PreparedBatch:
    data_cfg = config["data"]
    dqvl_cfg = config.get("dqvl", {})

    samples, imported_files = load_patchtst_inputs(
        data_cfg["path"], timestamp_col=data_cfg.get("timestamp_col")
    )

    seq_len = int(data_cfg["seq_len"])
    seq_stride = int(data_cfg["seq_stride"])
    allow_sort_fix = bool(dqvl_cfg.get("allow_sort_fix", False))
    dqvl_enabled = bool(dqvl_cfg.get("enabled", True))
    report_dir = Path(str(dqvl_cfg.get("report_dir", "artifacts/dqvl/fdc")))
    run_id = new_run_id()

    windows_chunks: list[np.ndarray] = []
    window_file_ids: list[str] = []
    window_anchor_timestamps: list[str | None] = []
    dqvl_records: list[DQVLRecord] = []
    skipped_files: list[str] = []
    expected_feature_columns: list[str] | None = None

    for sample in samples:
        if dqvl_enabled:
            report = evaluate_fdc_quality(sample, dqvl_cfg=dqvl_cfg, run_id=run_id)
            report_path = save_report(report, report_dir)
            dqvl_records.append(_build_dqvl_record(report, report_path))
            if report["decision"] == "drop":
                skipped_files.append(sample.file_id)
                continue

        if sample.values.ndim != 2 or sample.values.shape[1] == 0:
            skipped_files.append(sample.file_id)
            continue

        if expected_feature_columns is None:
            expected_feature_columns = list(sample.feature_columns)
        elif list(sample.feature_columns) != expected_feature_columns:
            skipped_files.append(sample.file_id)
            continue

        values, timestamps = _sort_fdc_values_and_timestamps(
            sample.values.astype(np.float32),
            sample.timestamps,
            allow_sort_fix=allow_sort_fix,
        )
        if values.shape[0] <= 1:
            skipped_files.append(sample.file_id)
            continue

        sanitized = _sanitize_fdc_values(values)
        windows = _fdc_safe_sliding_windows(sanitized, window=seq_len, stride=seq_stride)
        if windows.shape[0] == 0:
            skipped_files.append(sample.file_id)
            continue

        anchors = _build_anchor_timestamps(
            timestamps,
            window=seq_len,
            stride=seq_stride,
            expected_count=windows.shape[0],
        )
        windows_chunks.append(windows)
        window_file_ids.extend([sample.file_id] * windows.shape[0])
        window_anchor_timestamps.extend(anchors)

    if not windows_chunks:
        raise BatchPreprocessError("No valid patchtst windows produced from input files")

    all_windows = np.concatenate(windows_chunks, axis=0).astype(np.float32)
    return PreparedBatch(
        stream="patchtst",
        windows=all_windows,
        window_file_ids=window_file_ids,
        window_anchor_timestamps=window_anchor_timestamps,
        imported_files=imported_files,
        dqvl_records=dqvl_records,
        skipped_files=skipped_files,
        metadata={
            "feature_columns": expected_feature_columns or [],
            "seq_len": seq_len,
            "seq_stride": seq_stride,
            "normalization": str(data_cfg.get("normalization", "robust")),
        },
    )


def _with_actual_fs(sample: VibReadResult, actual_fs: float | None) -> VibReadResult:
    return VibReadResult(
        file_id=sample.file_id,
        path=sample.path,
        source_type=sample.source_type,
        timestamps=sample.timestamps,
        values=sample.values,
        missing_axes=sample.missing_axes,
        raw_shape=sample.raw_shape,
        fs=actual_fs,
    )


def prepare_swinmae_batch(config: dict[str, Any]) -> PreparedBatch:
    data_cfg = config["data"]
    dqvl_cfg = config.get("dqvl", {})

    samples, imported_files = load_swinmae_inputs(
        data_cfg["path"],
        fs=None,
        timestamp_col=data_cfg.get("timestamp_col"),
    )

    expected_fs = float(data_cfg["fs"])
    win_len = int(float(data_cfg["win_sec"]) * expected_fs)
    win_stride_sec = float(data_cfg.get("win_stride_sec", data_cfg.get("stride_sec")))
    win_stride = int(win_stride_sec * expected_fs)

    dqvl_enabled = bool(dqvl_cfg.get("enabled", True))
    report_dir = Path(str(dqvl_cfg.get("report_dir", "artifacts/dqvl/vib")))
    run_id = new_run_id()

    resample_cfg = data_cfg.get("resample", {})
    resample_enabled = bool(resample_cfg.get("enabled", False))
    resample_method = str(resample_cfg.get("method", "linear"))

    windows_chunks: list[np.ndarray] = []
    window_file_ids: list[str] = []
    window_anchor_timestamps: list[str | None] = []
    dqvl_records: list[DQVLRecord] = []
    skipped_files: list[str] = []

    for sample in samples:
        actual_fs = _actual_fs_for_file(
            data_cfg,
            file_path=sample.path,
            expected_fs=expected_fs,
        )
        sample = _with_actual_fs(sample, actual_fs)

        if dqvl_enabled:
            report = evaluate_vibration_quality(
                sample,
                dqvl_cfg=dqvl_cfg,
                run_id=run_id,
                expected_fs=expected_fs,
                resample_enabled=resample_enabled,
            )
            report_path = save_report(report, report_dir)
            dqvl_records.append(_build_dqvl_record(report, report_path))
            if report["decision"] == "drop":
                skipped_files.append(sample.file_id)
                continue

        values = sample.values.astype(np.float32, copy=False)
        if values.ndim != 2 or values.shape[1] != 3:
            skipped_files.append(sample.file_id)
            continue

        timestamps = sample.timestamps
        if (
            actual_fs is not None
            and abs(float(actual_fs) - expected_fs) > 1e-12
            and resample_enabled
        ):
            if resample_method != "linear":
                raise BatchPreprocessError(f"Unsupported resample method: {resample_method}")
            values = _resample_linear(values, src_fs=float(actual_fs), dst_fs=expected_fs)
            timestamps = None

        windows = _vib_safe_sliding_windows(values, window=win_len, stride=win_stride)
        if windows.shape[0] == 0:
            skipped_files.append(sample.file_id)
            continue

        anchors = _build_anchor_timestamps(
            timestamps,
            window=win_len,
            stride=win_stride,
            expected_count=windows.shape[0],
        )
        windows_chunks.append(windows)
        window_file_ids.extend([sample.file_id] * windows.shape[0])
        window_anchor_timestamps.extend(anchors)

    if not windows_chunks:
        raise BatchPreprocessError("No valid swinmae windows produced from input files")

    all_windows = np.concatenate(windows_chunks, axis=0).astype(np.float32)
    return PreparedBatch(
        stream="swinmae",
        windows=all_windows,
        window_file_ids=window_file_ids,
        window_anchor_timestamps=window_anchor_timestamps,
        imported_files=imported_files,
        dqvl_records=dqvl_records,
        skipped_files=skipped_files,
        metadata={
            "expected_fs": expected_fs,
            "win_len": win_len,
            "win_stride": win_stride,
            "cwt": dict(config["cwt"]),
            "image": dict(config["image"]),
        },
    )
