from __future__ import annotations

import copy
from pathlib import Path

import pytest

from batch_decision.preprocess import (
    BatchPreprocessError,
    prepare_patchtst_batch,
    prepare_swinmae_batch,
)


def test_prepare_patchtst_batch_builds_windows_from_csv(
    patchtst_smoke_config,
    tmp_path: Path,
) -> None:
    cfg = copy.deepcopy(patchtst_smoke_config)
    cfg["data"].update(
        {
            "source": "csv",
            "path": str(Path(__file__).resolve().parents[1] / "smoke" / "data" / "fdc_dummy.csv"),
            "timestamp_col": "timestamp",
            "seq_len": 16,
            "seq_stride": 8,
            "normalization": "robust",
        }
    )
    cfg["dqvl"] = {
        "enabled": True,
        "allow_sort_fix": False,
        "report_dir": str(tmp_path / "dqvl_fdc"),
        "hard_fail": {
            "require_timestamp": True,
            "invalid_timestamp": True,
            "max_missing_ratio": 0.8,
        },
        "warn": {
            "missing_ratio": 0.2,
            "stuck_std": 1.0e-8,
            "jump_ratio": 0.9,
        },
    }

    batch = prepare_patchtst_batch(cfg)
    assert batch.stream == "patchtst"
    assert batch.windows.ndim == 3
    assert batch.windows.shape[0] > 0
    assert len(batch.window_file_ids) == batch.windows.shape[0]
    assert len(batch.window_anchor_timestamps) == batch.windows.shape[0]
    assert batch.metadata["feature_columns"] == ["param_a", "param_b", "param_c"]


def test_prepare_patchtst_batch_rejects_missing_timestamp(
    patchtst_smoke_config,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "fdc_missing_timestamp.csv"
    csv_path.write_text(
        "\n".join(
            [
                "param_a,param_b",
                "1.0,2.0",
                "1.1,2.1",
                "1.2,2.2",
                "1.3,2.3",
                "1.4,2.4",
                "1.5,2.5",
                "1.6,2.6",
                "1.7,2.7",
                "1.8,2.8",
                "1.9,2.9",
                "2.0,3.0",
                "2.1,3.1",
                "2.2,3.2",
                "2.3,3.3",
                "2.4,3.4",
                "2.5,3.5",
            ]
        ),
        encoding="utf-8",
    )

    cfg = copy.deepcopy(patchtst_smoke_config)
    cfg["data"].update(
        {
            "source": "csv",
            "path": str(csv_path),
            "timestamp_col": "timestamp",
            "seq_len": 8,
            "seq_stride": 4,
        }
    )
    cfg["dqvl"] = {
        "enabled": True,
        "allow_sort_fix": False,
        "report_dir": str(tmp_path / "dqvl_fdc_bad"),
        "hard_fail": {
            "require_timestamp": True,
            "invalid_timestamp": True,
            "max_missing_ratio": 0.8,
        },
        "warn": {"missing_ratio": 0.2, "stuck_std": 1.0e-8, "jump_ratio": 0.9},
    }

    with pytest.raises(BatchPreprocessError, match="No valid patchtst windows"):
        prepare_patchtst_batch(cfg)


def test_prepare_swinmae_batch_builds_windows_from_csv(
    swinmae_smoke_config,
    tmp_path: Path,
) -> None:
    cfg = copy.deepcopy(swinmae_smoke_config)
    cfg["data"].update(
        {
            "source": "csv",
            "path": str(Path(__file__).resolve().parents[1] / "smoke" / "data" / "vib_dummy.csv"),
            "timestamp_col": "timestamp",
            "fs": 64,
            "train_ratio": 0.5,
            "win_sec": 0.5,
            "win_stride_sec": 0.25,
            "assume_actual_fs_equals_config": True,
            "resample": {"enabled": False, "method": "linear"},
        }
    )
    cfg["dqvl"] = {
        "enabled": True,
        "allow_sort_fix": False,
        "report_dir": str(tmp_path / "dqvl_vib"),
        "hard_fail": {
            "max_missing_ratio": 0.8,
            "fs_tol": 1.0e-6,
            "missing_fs": False,
        },
        "warn": {
            "missing_ratio": 0.2,
            "clipping_ratio": 1.0,
            "flat_eps": 1.0e-8,
            "flat_ratio": 1.0,
            "rms_min": 1.0e-8,
            "rms_max": 1000.0,
        },
    }

    batch = prepare_swinmae_batch(cfg)
    assert batch.stream == "swinmae"
    assert batch.windows.ndim == 3
    assert batch.windows.shape[0] > 0
    assert batch.windows.shape[-1] == 3
    assert len(batch.window_file_ids) == batch.windows.shape[0]
    assert batch.metadata["expected_fs"] == 64.0


def test_prepare_swinmae_batch_rejects_missing_axis(
    swinmae_smoke_config,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "vib_missing_axis.csv"
    csv_path.write_text(
        "\n".join(
            [
                "timestamp,x,y",
                "2026-01-01T00:00:00.000,0.1,0.2",
                "2026-01-01T00:00:00.001,0.2,0.3",
                "2026-01-01T00:00:00.002,0.3,0.4",
                "2026-01-01T00:00:00.003,0.4,0.5",
                "2026-01-01T00:00:00.004,0.5,0.6",
                "2026-01-01T00:00:00.005,0.6,0.7",
                "2026-01-01T00:00:00.006,0.7,0.8",
                "2026-01-01T00:00:00.007,0.8,0.9",
                "2026-01-01T00:00:00.008,0.9,1.0",
                "2026-01-01T00:00:00.009,1.0,1.1",
                "2026-01-01T00:00:00.010,1.1,1.2",
                "2026-01-01T00:00:00.011,1.2,1.3",
                "2026-01-01T00:00:00.012,1.3,1.4",
                "2026-01-01T00:00:00.013,1.4,1.5",
                "2026-01-01T00:00:00.014,1.5,1.6",
                "2026-01-01T00:00:00.015,1.6,1.7",
                "2026-01-01T00:00:00.016,1.7,1.8",
                "2026-01-01T00:00:00.017,1.8,1.9",
                "2026-01-01T00:00:00.018,1.9,2.0",
                "2026-01-01T00:00:00.019,2.0,2.1",
                "2026-01-01T00:00:00.020,2.1,2.2",
                "2026-01-01T00:00:00.021,2.2,2.3",
                "2026-01-01T00:00:00.022,2.3,2.4",
                "2026-01-01T00:00:00.023,2.4,2.5",
                "2026-01-01T00:00:00.024,2.5,2.6",
                "2026-01-01T00:00:00.025,2.6,2.7",
                "2026-01-01T00:00:00.026,2.7,2.8",
                "2026-01-01T00:00:00.027,2.8,2.9",
                "2026-01-01T00:00:00.028,2.9,3.0",
                "2026-01-01T00:00:00.029,3.0,3.1",
                "2026-01-01T00:00:00.030,3.1,3.2",
                "2026-01-01T00:00:00.031,3.2,3.3",
            ]
        ),
        encoding="utf-8",
    )

    cfg = copy.deepcopy(swinmae_smoke_config)
    cfg["data"].update(
        {
            "source": "csv",
            "path": str(csv_path),
            "timestamp_col": "timestamp",
            "fs": 64,
            "win_sec": 0.5,
            "win_stride_sec": 0.25,
            "assume_actual_fs_equals_config": True,
            "resample": {"enabled": False, "method": "linear"},
        }
    )
    cfg["dqvl"] = {
        "enabled": True,
        "allow_sort_fix": False,
        "report_dir": str(tmp_path / "dqvl_vib_bad"),
        "hard_fail": {
            "max_missing_ratio": 0.8,
            "fs_tol": 1.0e-6,
            "missing_fs": False,
        },
        "warn": {
            "missing_ratio": 0.2,
            "clipping_ratio": 1.0,
            "flat_eps": 1.0e-8,
            "flat_ratio": 1.0,
            "rms_min": 1.0e-8,
            "rms_max": 1000.0,
        },
    }

    with pytest.raises(BatchPreprocessError, match="No valid swinmae windows"):
        prepare_swinmae_batch(cfg)

