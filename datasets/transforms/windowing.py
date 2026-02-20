from __future__ import annotations

import numpy as np


def sliding_windows(series: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Create sliding windows from a 2D time series (T, C) -> (N, window, C)."""
    if series.ndim != 2:
        raise ValueError(f"Expected 2D series (T, C), got shape={series.shape}")
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")

    t, c = series.shape
    if t < window:
        raise ValueError(f"Series length {t} is smaller than window {window}")

    starts = np.arange(0, t - window + 1, stride)
    out = np.empty((len(starts), window, c), dtype=series.dtype)
    for i, s in enumerate(starts):
        out[i] = series[s : s + window]
    return out
