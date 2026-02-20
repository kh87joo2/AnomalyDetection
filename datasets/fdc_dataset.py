from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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

        values = sample.values
        if values.ndim != 2 or values.shape[1] == 0:
            continue

        split_idx = int(values.shape[0] * train_ratio)
        if split_idx <= 0 or split_idx >= values.shape[0]:
            continue

        train_raw = values[:split_idx]
        val_raw = values[split_idx:]

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

    return FDCDatasets(train=FDCDataset(train_windows), val=FDCDataset(val_windows), scaler=scaler)


def build_fdc_datasets(config: dict) -> FDCDatasets:
    data_cfg = config["data"]
    source = str(data_cfg.get("source", "synthetic")).lower()

    if source == "synthetic":
        return _build_synthetic(config)
    if source in {"csv", "parquet"}:
        return _build_real(config)
    raise ValueError(f"Unsupported data.source for FDC: {source}")
