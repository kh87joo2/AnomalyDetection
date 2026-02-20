from __future__ import annotations

import numpy as np


def generate_synthetic_vibration(total_steps: int, fs: int, seed: int = 42) -> np.ndarray:
    """Generate synthetic vibration stream (T, 3)."""
    rng = np.random.default_rng(seed)
    t = np.arange(total_steps, dtype=np.float32) / float(fs)

    axes = []
    base_freqs = [25.0, 45.0, 80.0]
    for axis in range(3):
        f1 = base_freqs[axis]
        f2 = f1 * 2.7
        amp_mod = 1.0 + 0.2 * np.sin(2.0 * np.pi * 0.15 * t + axis)
        signal = amp_mod * (
            0.8 * np.sin(2.0 * np.pi * f1 * t)
            + 0.35 * np.sin(2.0 * np.pi * f2 * t + 0.5 * axis)
        )
        noise = 0.08 * rng.normal(size=total_steps)
        axes.append(signal + noise)

    return np.stack(axes, axis=-1).astype(np.float32)
