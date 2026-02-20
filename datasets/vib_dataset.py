from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.transforms.cwt import vibration_window_to_image
from datasets.transforms.windowing import sliding_windows
from datasets.vib_synthetic import generate_synthetic_vibration


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


def build_vibration_datasets(config: dict) -> VibDatasets:
    data_cfg = config["data"]
    cwt_cfg = config["cwt"]
    image_cfg = config["image"]

    total_steps = int(data_cfg["total_steps"])
    fs = int(data_cfg["fs"])
    train_ratio = float(data_cfg["train_ratio"])

    win_len = int(float(data_cfg["win_sec"]) * fs)
    win_stride = int(float(data_cfg["win_stride_sec"]) * fs)

    raw = generate_synthetic_vibration(total_steps=total_steps, fs=fs, seed=int(config["seed"]))

    split_idx = int(total_steps * train_ratio)
    train_raw = raw[:split_idx]
    val_raw = raw[split_idx:]

    train_windows = sliding_windows(train_raw, window=win_len, stride=win_stride)
    val_windows = sliding_windows(val_raw, window=win_len, stride=win_stride)

    train_ds = VibrationImageDataset(train_windows, cwt_cfg=cwt_cfg, image_cfg=image_cfg, data_cfg=data_cfg)
    val_ds = VibrationImageDataset(val_windows, cwt_cfg=cwt_cfg, image_cfg=image_cfg, data_cfg=data_cfg)
    return VibDatasets(train=train_ds, val=val_ds)
