from __future__ import annotations

import glob
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from datasets.fdc_synthetic import generate_synthetic_fdc
from datasets.readers.fdc_reader import read_fdc_file
from datasets.transforms.fdc_normalization import ChannelScaler
from datasets.transforms.windowing import sliding_windows
from dqvl.fdc_rules import evaluate_fdc_quality
from dqvl.report import new_run_id, save_report


class FDCDataset(Dataset):
    def __init__(self, windows: np.ndarray):
        if windows.ndim != 3:
            raise ValueError(f"Expected windows shape (N, T, C), got {windows.shape}")
        self.windows = windows.astype(np.float32)

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.windows[idx])


@dataclass
class FDCDatasets:
    train: FDCDataset
    val: FDCDataset
    scaler: ChannelScaler


def _safe_sliding_windows(series: np.ndarray, window: int, stride: int) -> np.ndarray:
    if series.ndim != 2:
        raise ValueError(f"Expected 2D input series, got {series.shape}")
    if series.shape[0] < window:
        return np.empty((0, window, series.shape[1]), dtype=np.float32)
    return sliding_windows(series, window=window, stride=stride).astype(np.float32)


def _to_numeric_timestamps(timestamps: np.ndarray) -> np.ndarray:
    series = pd.Series(timestamps)

    num = pd.to_numeric(series, errors="coerce")
    if int(num.notna().sum()) > 0:
        return num.to_numpy(dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        dt_default = pd.to_datetime(series, errors="coerce")
        dt_dayfirst = pd.to_datetime(series, errors="coerce", dayfirst=True)

    dt = dt_dayfirst if int(dt_dayfirst.notna().sum()) > int(dt_default.notna().sum()) else dt_default
    arr = dt.astype("int64").to_numpy(dtype=np.float64)
    arr[dt.isna().to_numpy()] = np.nan
    return arr


def _sort_by_timestamp_if_enabled(
    values: np.ndarray,
    timestamps: np.ndarray | None,
    allow_sort_fix: bool,
) -> np.ndarray:
    if not allow_sort_fix or timestamps is None or values.shape[0] <= 1:
        return values

    ts_num = _to_numeric_timestamps(timestamps)
    valid = np.isfinite(ts_num)
    if int(valid.sum()) < 2:
        return values

    values_valid = values[valid]
    ts_valid = ts_num[valid]
    order = np.argsort(ts_valid, kind="mergesort")
    values_sorted = values_valid[order]
    ts_sorted = ts_valid[order]

    # Keep only first row per duplicated timestamp to ensure strict ordering.
    keep = np.ones(ts_sorted.shape[0], dtype=bool)
    keep[1:] = ts_sorted[1:] != ts_sorted[:-1]
    return values_sorted[keep]


def _impute_non_finite_with_train_stats(
    train_raw: np.ndarray,
    val_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train = np.where(np.isfinite(train_raw), train_raw, np.nan).astype(np.float32)
    val = np.where(np.isfinite(val_raw), val_raw, np.nan).astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fill = np.nanmedian(train, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0).astype(np.float32)

    train = np.where(np.isfinite(train), train, fill.reshape(1, -1))
    val = np.where(np.isfinite(val), val, fill.reshape(1, -1))

    train = np.nan_to_num(train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    val = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return train, val


def _resolve_paths(path_cfg: Any) -> list[Path]:
    if isinstance(path_cfg, str):
        patterns = [path_cfg]
    elif isinstance(path_cfg, list):
        patterns = [str(p) for p in path_cfg]
    else:
        raise ValueError("data.path must be a glob string or list of glob strings")

    out: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in sorted(glob.glob(pattern)):
            key = str(Path(match).resolve())
            if key not in seen:
                seen.add(key)
                out.append(Path(match))
    return out


def _build_synthetic(config: dict) -> FDCDatasets:
    data_cfg = config["data"]
    total_steps = int(data_cfg["total_steps"])
    channels = int(data_cfg["channels"])
    train_ratio = float(data_cfg["train_ratio"])
    seq_len = int(data_cfg["seq_len"])
    seq_stride = int(data_cfg["seq_stride"])
    norm_method = str(data_cfg.get("normalization", "robust"))

    raw = generate_synthetic_fdc(
        total_steps=total_steps,
        channels=channels,
        seed=int(config["seed"]),
    )

    split_idx = int(total_steps * train_ratio)
    train_raw = raw[:split_idx]
    val_raw = raw[split_idx:]

    train_windows = _safe_sliding_windows(train_raw, window=seq_len, stride=seq_stride)
    val_windows = _safe_sliding_windows(val_raw, window=seq_len, stride=seq_stride)

    if train_windows.shape[0] == 0:
        raise ValueError("No train windows generated from synthetic FDC data")

    scaler = ChannelScaler(method=norm_method)
    train_windows = scaler.fit_transform(train_windows)
    val_windows = scaler.transform(val_windows) if val_windows.shape[0] > 0 else val_windows

    return FDCDatasets(train=FDCDataset(train_windows), val=FDCDataset(val_windows), scaler=scaler)


def _build_real(config: dict) -> FDCDatasets:
    data_cfg = config["data"]
    dqvl_cfg = config.get("dqvl", {})

    path_cfg = data_cfg.get("path")
    file_paths = _resolve_paths(path_cfg)
    if not file_paths:
        raise FileNotFoundError(f"No files matched data.path={path_cfg}")

    train_ratio = float(data_cfg["train_ratio"])
    seq_len = int(data_cfg["seq_len"])
    seq_stride = int(data_cfg["seq_stride"])
    norm_method = str(data_cfg.get("normalization", "robust"))
    timestamp_col = data_cfg.get("timestamp_col")

    dqvl_enabled = bool(dqvl_cfg.get("enabled", True))
    allow_sort_fix = bool(dqvl_cfg.get("allow_sort_fix", False))
    report_dir = str(dqvl_cfg.get("report_dir", "artifacts/dqvl/fdc"))
    run_id = new_run_id()

    train_chunks: list[np.ndarray] = []
    val_chunks: list[np.ndarray] = []

    for file_path in file_paths:
        sample = read_fdc_file(file_path, timestamp_col=timestamp_col)

        if dqvl_enabled:
            report = evaluate_fdc_quality(sample, dqvl_cfg=dqvl_cfg, run_id=run_id)
            save_report(report, report_dir)
            if report["decision"] == "drop":
                continue

        values = sample.values.astype(np.float32)
        if values.ndim != 2 or values.shape[1] == 0:
            continue

        values = _sort_by_timestamp_if_enabled(values, sample.timestamps, allow_sort_fix)
        if values.shape[0] <= 1:
            continue

        split_idx = int(values.shape[0] * train_ratio)
        if split_idx <= 0 or split_idx >= values.shape[0]:
            continue

        train_raw, val_raw = _impute_non_finite_with_train_stats(values[:split_idx], values[split_idx:])

        train_w = _safe_sliding_windows(train_raw, window=seq_len, stride=seq_stride)
        val_w = _safe_sliding_windows(val_raw, window=seq_len, stride=seq_stride)

        if train_w.shape[0] > 0:
            train_chunks.append(train_w)
        if val_w.shape[0] > 0:
            val_chunks.append(val_w)

    if not train_chunks:
        raise ValueError("No valid train windows produced from real FDC input files")

    train_windows = np.concatenate(train_chunks, axis=0).astype(np.float32)
    if val_chunks:
        val_windows = np.concatenate(val_chunks, axis=0).astype(np.float32)
    else:
        val_windows = np.empty((0, seq_len, train_windows.shape[-1]), dtype=np.float32)

    scaler = ChannelScaler(method=norm_method)
    train_windows = scaler.fit_transform(train_windows)
    val_windows = scaler.transform(val_windows) if val_windows.shape[0] > 0 else val_windows

    train_windows = np.nan_to_num(train_windows, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if val_windows.shape[0] > 0:
        val_windows = np.nan_to_num(val_windows, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return FDCDatasets(train=FDCDataset(train_windows), val=FDCDataset(val_windows), scaler=scaler)


def build_fdc_datasets(config: dict) -> FDCDatasets:
    data_cfg = config["data"]
    source = str(data_cfg.get("source", "synthetic")).lower()

    if source == "synthetic":
        return _build_synthetic(config)
    if source in {"csv", "parquet"}:
        return _build_real(config)
    raise ValueError(f"Unsupported data.source for FDC: {source}")
