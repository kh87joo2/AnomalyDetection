from __future__ import annotations

import numpy as np
import pywt
import torch
import torch.nn.functional as F


def _normalize_2d(x: np.ndarray, mode: str = "robust", eps: float = 1e-6) -> np.ndarray:
    if mode == "robust":
        med = np.median(x)
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        scale = max(q3 - q1, eps)
        return (x - med) / scale
    if mode == "zscore":
        mean = np.mean(x)
        std = max(float(np.std(x)), eps)
        return (x - mean) / std
    if mode == "minmax":
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        denom = max(x_max - x_min, eps)
        return (x - x_min) / denom
    raise ValueError(f"Unsupported normalize mode: {mode}")


def cwt_scalogram(
    signal_1d: np.ndarray,
    fs: int,
    freq_min: float,
    freq_max: float,
    n_freqs: int,
    wavelet: str = "morl",
    log_mag: bool = True,
    normalize: str = "robust",
) -> np.ndarray:
    """Compute a CWT scalogram with pywt, output shape (F, T)."""
    if freq_min <= 0.0:
        raise ValueError("freq_min must be positive for CWT scale conversion")
    if freq_max <= freq_min:
        raise ValueError("freq_max must be larger than freq_min")

    freqs = np.linspace(freq_min, freq_max, n_freqs, dtype=np.float32)
    central = pywt.central_frequency(wavelet)
    scales = (central * fs) / freqs

    coeffs, _ = pywt.cwt(signal_1d, scales=scales, wavelet=wavelet, sampling_period=1.0 / fs)
    mag = np.abs(coeffs)

    if log_mag:
        mag = np.log1p(mag)

    mag = _normalize_2d(mag, mode=normalize)
    return mag.astype(np.float32)


def vibration_window_to_image(
    window: np.ndarray,
    fs: int,
    freq_min: float,
    freq_max: float,
    n_freqs: int,
    image_size: int,
    wavelet: str = "morl",
    log_mag: bool = True,
    normalize: str = "robust",
) -> np.ndarray:
    """(win_len, 3) -> (3, H, W) using axis-wise CWT scalograms."""
    if window.ndim != 2 or window.shape[1] != 3:
        raise ValueError(f"Expected window shape (win_len, 3), got {window.shape}")

    channels = []
    for axis in range(3):
        s = cwt_scalogram(
            window[:, axis],
            fs=fs,
            freq_min=freq_min,
            freq_max=freq_max,
            n_freqs=n_freqs,
            wavelet=wavelet,
            log_mag=log_mag,
            normalize=normalize,
        )
        channels.append(s)

    img = np.stack(channels, axis=0)  # (3, F, T)
    tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 3, F, T)
    resized = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return resized.squeeze(0).numpy().astype(np.float32)
