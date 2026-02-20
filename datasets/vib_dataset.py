from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.readers.vib_reader import read_vibration_file
from datasets.transforms.cwt import vibration_window_to_image
from datasets.transforms.windowing import sliding_windows
from datasets.vib_synthetic import generate_synthetic_vibration
from dqvl.report import new_run_id, save_report
from dqvl.vib_rules import evaluate_vibration_quality


class VibrationImageDataset(Dataset):
    def __init__(self, windows: np.ndarray, cwt_cfg: dict, image_cfg: dict, data_cfg: dict):
        self.windows = windows.astype(np.float32)
        self.cwt_cfg = cwt_cfg
        self.image_cfg = image_cfg
        self.data_cfg = data_cfg

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        window = self.windows[idx]
        img = vibration_window_to_image(
            window=window,
            fs=int(self.data_cfg["fs"]),
            freq_min=float(self.cwt_cfg["freq_min"]),
            freq_max=float(self.cwt_cfg["freq_max"]),
            n_freqs=int(self.cwt_cfg["n_freqs"]),
            image_size=int(self.image_cfg["size"]),
            wavelet=str(self.cwt_cfg.get("wavelet", "morl")),
            log_mag=bool(self.cwt_cfg.get("log_mag", True)),
            normalize=str(self.cwt_cfg.get("normalize", "robust")),
        )
        return torch.from_numpy(img)


@dataclass
class VibDatasets:
    train: VibrationImageDataset
    val: VibrationImageDataset


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


def _resample_linear(values: np.ndarray, src_fs: float, dst_fs: float) -> np.ndarray:
    if abs(float(src_fs) - float(dst_fs)) < 1e-12:
        return values.astype(np.float32, copy=False)

    if src_fs <= 0 or dst_fs <= 0:
        raise ValueError("src_fs and dst_fs must be positive for resampling")

    src_len = values.shape[0]
    if src_len < 2:
        return values.astype(np.float32, copy=False)

    duration = (src_len - 1) / float(src_fs)
    dst_len = int(round(duration * float(dst_fs))) + 1

    src_t = np.arange(src_len, dtype=np.float64) / float(src_fs)
    dst_t = np.arange(dst_len, dtype=np.float64) / float(dst_fs)

    out = np.empty((dst_len, values.shape[1]), dtype=np.float32)
    for axis in range(values.shape[1]):
        out[:, axis] = np.interp(dst_t, src_t, values[:, axis]).astype(np.float32)
    return out


def _actual_fs_for_file(
    data_cfg: dict[str, Any],
    file_path: Path,
    expected_fs: float,
) -> float | None:
    fs_by_file = data_cfg.get("actual_fs_by_file", {})
    if isinstance(fs_by_file, dict):
        if file_path.name in fs_by_file:
            return float(fs_by_file[file_path.name])
        if str(file_path) in fs_by_file:
            return float(fs_by_file[str(file_path)])

    if "actual_fs" in data_cfg and data_cfg["actual_fs"] is not None:
        return float(data_cfg["actual_fs"])

    if bool(data_cfg.get("assume_actual_fs_equals_config", True)):
        return expected_fs
    return None


def _build_synthetic(config: dict) -> VibDatasets:
    data_cfg = config["data"]
    cwt_cfg = config["cwt"]
    image_cfg = config["image"]

    total_steps = int(data_cfg["total_steps"])
    fs = int(data_cfg["fs"])
    train_ratio = float(data_cfg["train_ratio"])

    win_len = int(float(data_cfg["win_sec"]) * fs)
    win_stride_sec = float(data_cfg.get("win_stride_sec", data_cfg.get("stride_sec")))
    win_stride = int(win_stride_sec * fs)

    raw = generate_synthetic_vibration(total_steps=total_steps, fs=fs, seed=int(config["seed"]))

    split_idx = int(total_steps * train_ratio)
    train_raw = raw[:split_idx]
    val_raw = raw[split_idx:]

    train_windows = _safe_sliding_windows(train_raw, window=win_len, stride=win_stride)
    val_windows = _safe_sliding_windows(val_raw, window=win_len, stride=win_stride)

    if train_windows.shape[0] == 0:
        raise ValueError("No train windows generated from synthetic vibration data")

    train_ds = VibrationImageDataset(
        train_windows,
        cwt_cfg=cwt_cfg,
        image_cfg=image_cfg,
        data_cfg=data_cfg,
    )
    val_ds = VibrationImageDataset(
        val_windows,
        cwt_cfg=cwt_cfg,
        image_cfg=image_cfg,
        data_cfg=data_cfg,
    )
    return VibDatasets(train=train_ds, val=val_ds)


def _build_real(config: dict) -> VibDatasets:
    data_cfg = config["data"]
    cwt_cfg = config["cwt"]
    image_cfg = config["image"]
    dqvl_cfg = config.get("dqvl", {})

    expected_fs = float(data_cfg["fs"])
    train_ratio = float(data_cfg["train_ratio"])

    win_len = int(float(data_cfg["win_sec"]) * expected_fs)
    win_stride_sec = float(data_cfg.get("win_stride_sec", data_cfg.get("stride_sec")))
    win_stride = int(win_stride_sec * expected_fs)

    path_cfg = data_cfg.get("path")
    file_paths = _resolve_paths(path_cfg)
    if not file_paths:
        raise FileNotFoundError(f"No files matched data.path={path_cfg}")

    source = str(data_cfg.get("source", "csv")).lower()
    allowed_suffixes = {".csv"} if source == "csv" else {".npy"}

    dqvl_enabled = bool(dqvl_cfg.get("enabled", True))
    report_dir = str(dqvl_cfg.get("report_dir", "artifacts/dqvl/vib"))
    run_id = new_run_id()

    resample_cfg = data_cfg.get("resample", {})
    resample_enabled = bool(resample_cfg.get("enabled", False))
    resample_method = str(resample_cfg.get("method", "linear"))

    train_chunks: list[np.ndarray] = []
    val_chunks: list[np.ndarray] = []

    for file_path in file_paths:
        if file_path.suffix.lower() not in allowed_suffixes:
            continue

        actual_fs = _actual_fs_for_file(data_cfg, file_path=file_path, expected_fs=expected_fs)
        sample = read_vibration_file(
            file_path,
            fs=actual_fs,
            timestamp_col=data_cfg.get("timestamp_col"),
        )

        if dqvl_enabled:
            report = evaluate_vibration_quality(
                sample,
                dqvl_cfg=dqvl_cfg,
                run_id=run_id,
                expected_fs=expected_fs,
                resample_enabled=resample_enabled,
            )
            save_report(report, report_dir)
            if report["decision"] == "drop":
                continue

        values = sample.values
        if values.ndim != 2 or values.shape[1] != 3:
            continue

        if (
            actual_fs is not None
            and abs(float(actual_fs) - expected_fs) > 1e-12
            and resample_enabled
        ):
            if resample_method != "linear":
                raise ValueError(f"Unsupported resample method: {resample_method}")
            values = _resample_linear(values, src_fs=float(actual_fs), dst_fs=expected_fs)

        split_idx = int(values.shape[0] * train_ratio)
        if split_idx <= 0 or split_idx >= values.shape[0]:
            continue

        train_raw = values[:split_idx]
        val_raw = values[split_idx:]

        train_w = _safe_sliding_windows(train_raw, window=win_len, stride=win_stride)
        val_w = _safe_sliding_windows(val_raw, window=win_len, stride=win_stride)

        if train_w.shape[0] > 0:
            train_chunks.append(train_w)
        if val_w.shape[0] > 0:
            val_chunks.append(val_w)

    if not train_chunks:
        raise ValueError("No valid train windows produced from real vibration input files")

    train_windows = np.concatenate(train_chunks, axis=0).astype(np.float32)
    if val_chunks:
        val_windows = np.concatenate(val_chunks, axis=0).astype(np.float32)
    else:
        val_windows = np.empty((0, win_len, 3), dtype=np.float32)

    train_ds = VibrationImageDataset(
        train_windows,
        cwt_cfg=cwt_cfg,
        image_cfg=image_cfg,
        data_cfg=data_cfg,
    )
    val_ds = VibrationImageDataset(
        val_windows,
        cwt_cfg=cwt_cfg,
        image_cfg=image_cfg,
        data_cfg=data_cfg,
    )
    return VibDatasets(train=train_ds, val=val_ds)


def build_vibration_datasets(config: dict) -> VibDatasets:
    data_cfg = config["data"]
    source = str(data_cfg.get("source", "synthetic")).lower()

    if source == "synthetic":
        return _build_synthetic(config)
    if source in {"csv", "npy"}:
        return _build_real(config)
    raise ValueError(f"Unsupported data.source for vibration: {source}")
