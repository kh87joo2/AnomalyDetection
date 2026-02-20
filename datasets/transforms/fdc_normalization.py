from __future__ import annotations

import json
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
        x2 = self._to_2d(x)
        if self.method == "robust":
            center = np.median(x2, axis=0)
            q1 = np.percentile(x2, 25, axis=0)
            q3 = np.percentile(x2, 75, axis=0)
            scale = q3 - q1
        else:
            center = np.mean(x2, axis=0)
            scale = np.std(x2, axis=0)

        scale = np.where(scale < self.eps, 1.0, scale)
        self.center_ = center.astype(np.float32)
        self.scale_ = scale.astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted")
        return ((x - self.center_) / self.scale_).astype(np.float32)

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
