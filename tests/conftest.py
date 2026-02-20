from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from core.config import load_yaml_config

_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config_with_overrides(
    config_name: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_yaml_config(_CONFIGS_DIR / config_name)
    if not overrides:
        return config
    return _deep_update(copy.deepcopy(config), overrides)


@pytest.fixture(scope="session")
def torch_module():
    return pytest.importorskip("torch", reason="torch is required for smoke tests")


@pytest.fixture(scope="session")
def pywt_module():
    return pytest.importorskip("pywt", reason="pywt is required for vibration CWT tests")


@pytest.fixture(scope="session")
def timm_module():
    return pytest.importorskip("timm", reason="timm is required when use_timm_swin is enabled")


@pytest.fixture(scope="session")
def patchtst_smoke_config() -> dict[str, Any]:
    return load_config_with_overrides(
        "patchtst_ssl.yaml",
        {
            "seed": 7,
            "deterministic": True,
            "data": {
                "channels": 4,
                "total_steps": 64,
                "train_ratio": 0.5,
                "seq_len": 32,
                "seq_stride": 32,
            },
            "model": {
                "patch_len": 8,
                "patch_stride": 8,
                "mask_ratio": 0.5,
                "d_model": 16,
                "nhead": 4,
                "num_layers": 1,
                "ff_dim": 32,
                "dropout": 0.0,
            },
            "training": {"batch_size": 2},
        },
    )


@pytest.fixture(scope="session")
def swinmae_smoke_config() -> dict[str, Any]:
    return load_config_with_overrides(
        "swinmae_ssl.yaml",
        {
            "seed": 7,
            "deterministic": True,
            "data": {
                "fs": 64,
                "total_steps": 128,
                "train_ratio": 0.5,
                "win_sec": 0.5,
                "win_stride_sec": 0.25,
            },
            "cwt": {
                "freq_min": 1.0,
                "freq_max": 24.0,
                "n_freqs": 16,
            },
            "image": {"size": 32},
            "model": {
                "use_timm_swin": False,
                "mask_ratio": 0.5,
                "patch_size": 8,
                "decoder_dim": 64,
            },
            "training": {"batch_size": 1},
        },
    )
