from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Literal

import numpy as np

NormMethod = Literal["robust", "zscore"]


class ChannelScaler:
    """Channel-wise scaler for (N, T, C) or (T, C) data."""

    def __init__(self, method: NormMethod = "robust", eps: float = 1e-6):
        if method not in {"robust", "zscore"}:
            raise ValueError(f"Unsupported method: {method}")
        self.method = method
        self.eps = eps
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def _to_2d(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            return x
        if x.ndim == 3:
            n, t, c = x.shape
            return x.reshape(n * t, c)
        raise ValueError(f"Expected 2D or 3D input, got shape={x.shape}")

    def fit(self, x: np.ndarray) -> "ChannelScaler":
        x2 = self._to_2d(x).astype(np.float32)
        x2 = np.where(np.isfinite(x2), x2, np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if self.method == "robust":
                center = np.nanmedian(x2, axis=0)
                q1 = np.nanpercentile(x2, 25, axis=0)
                q3 = np.nanpercentile(x2, 75, axis=0)
                scale = q3 - q1
            else:
                center = np.nanmean(x2, axis=0)
                scale = np.nanstd(x2, axis=0)

        center = np.where(np.isfinite(center), center, 0.0)
        scale = np.where(np.isfinite(scale) & (np.abs(scale) >= self.eps), scale, 1.0)
        self.center_ = center.astype(np.float32)
        self.scale_ = scale.astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted")
        z = (x.astype(np.float32) - self.center_) / self.scale_
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def to_dict(self) -> dict:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted")
        return {
            "method": self.method,
            "eps": self.eps,
            "center": self.center_.tolist(),
            "scale": self.scale_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelScaler":
        scaler = cls(method=d["method"], eps=float(d.get("eps", 1e-6)))
        scaler.center_ = np.asarray(d["center"], dtype=np.float32)
        scaler.scale_ = np.asarray(d["scale"], dtype=np.float32)
        return scaler

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str | Path) -> "ChannelScaler":
        with Path(path).open("r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)
