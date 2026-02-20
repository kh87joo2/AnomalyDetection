from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.fdc_synthetic import generate_synthetic_fdc
from datasets.transforms.fdc_normalization import ChannelScaler
from datasets.transforms.windowing import sliding_windows


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


def build_fdc_datasets(config: dict) -> FDCDatasets:
    data_cfg = config["data"]
    total_steps = int(data_cfg["total_steps"])
    channels = int(data_cfg["channels"])
    train_ratio = float(data_cfg["train_ratio"])
    seq_len = int(data_cfg["seq_len"])
    seq_stride = int(data_cfg["seq_stride"])
    norm_method = str(data_cfg.get("normalization", "robust"))

    raw = generate_synthetic_fdc(total_steps=total_steps, channels=channels, seed=int(config["seed"]))

    split_idx = int(total_steps * train_ratio)
    train_raw = raw[:split_idx]
    val_raw = raw[split_idx:]

    train_windows = sliding_windows(train_raw, window=seq_len, stride=seq_stride)
    val_windows = sliding_windows(val_raw, window=seq_len, stride=seq_stride)

    scaler = ChannelScaler(method=norm_method)
    train_windows = scaler.fit_transform(train_windows)
    val_windows = scaler.transform(val_windows)

    return FDCDatasets(train=FDCDataset(train_windows), val=FDCDataset(val_windows), scaler=scaler)
