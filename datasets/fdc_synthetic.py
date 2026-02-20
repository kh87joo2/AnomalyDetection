from __future__ import annotations

import numpy as np


def generate_synthetic_fdc(total_steps: int, channels: int, seed: int = 42) -> np.ndarray:
    """Generate synthetic multivariate FDC stream: shape (T, C)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 20.0, total_steps, dtype=np.float32)

    series = []
    for ch in range(channels):
        base_freq = 0.1 + 0.03 * (ch % 7)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        trend = 0.001 * (ch + 1) * t
        signal = (
            0.7 * np.sin(2.0 * np.pi * base_freq * t + phase)
            + 0.2 * np.sin(2.0 * np.pi * (base_freq * 0.5) * t)
            + trend
        )
        noise = 0.08 * rng.normal(size=total_steps)
        series.append(signal + noise)

    arr = np.stack(series, axis=-1).astype(np.float32)
    return arr
