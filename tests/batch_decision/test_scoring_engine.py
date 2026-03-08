from __future__ import annotations

import copy
from pathlib import Path

from batch_decision.contracts import ArtifactPaths
from batch_decision.preprocess import prepare_patchtst_batch, prepare_swinmae_batch
from batch_decision.scoring_engine import score_windows
from datasets.transforms.fdc_normalization import ChannelScaler
from models.patchtst.patchtst_ssl import PatchTSTSSL
from models.swinmae.swinmae_ssl import SwinMAESSL


def _patchtst_csv_config(base_config: dict, tmp_path: Path) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg["device"]["prefer_cuda"] = False
    cfg["device"]["amp"] = False
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
    cfg["training"]["batch_size"] = 2
    return cfg


def _swinmae_csv_config(base_config: dict, tmp_path: Path) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg["device"]["prefer_cuda"] = False
    cfg["device"]["amp"] = False
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
    cfg["model"]["use_timm_swin"] = False
    cfg["training"]["batch_size"] = 1
    return cfg


def _save_patchtst_checkpoint(torch_module, cfg: dict, path: Path) -> None:
    model = PatchTSTSSL(
        seq_len=int(cfg["data"]["seq_len"]),
        patch_len=int(cfg["model"]["patch_len"]),
        patch_stride=int(cfg["model"]["patch_stride"]),
        d_model=int(cfg["model"]["d_model"]),
        nhead=int(cfg["model"]["nhead"]),
        num_layers=int(cfg["model"]["num_layers"]),
        ff_dim=int(cfg["model"]["ff_dim"]),
        dropout=float(cfg["model"]["dropout"]),
        mask_ratio=float(cfg["model"]["mask_ratio"]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save({"model_state_dict": model.state_dict()}, path)


def _save_swinmae_checkpoint(torch_module, cfg: dict, path: Path) -> None:
    model = SwinMAESSL(
        mask_ratio=float(cfg["model"]["mask_ratio"]),
        patch_size=int(cfg["model"]["patch_size"]),
        use_timm_swin=bool(cfg["model"].get("use_timm_swin", True)),
        timm_name=str(cfg["model"].get("timm_name", "swin_tiny_patch4_window7_224")),
        decoder_dim=int(cfg["model"].get("decoder_dim", 256)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save({"model_state_dict": model.state_dict()}, path)


def test_score_patchtst_windows_with_saved_scaler_and_checkpoint(
    torch_module,
    patchtst_smoke_config,
    tmp_path: Path,
) -> None:
    cfg = _patchtst_csv_config(patchtst_smoke_config, tmp_path)
    prepared = prepare_patchtst_batch(cfg)

    scaler = ChannelScaler(method=str(cfg["data"].get("normalization", "robust")))
    scaler.fit(prepared.windows)
    scaler_path = tmp_path / "artifacts" / "scaler_fdc.json"
    scaler.save(scaler_path)

    checkpoint_path = tmp_path / "checkpoints" / "patchtst_ssl.pt"
    _save_patchtst_checkpoint(torch_module, cfg, checkpoint_path)

    payload = score_windows(
        prepared,
        stream="patchtst",
        config=cfg,
        artifacts=ArtifactPaths(
            thresholds_path="unused",
            patchtst_checkpoint=str(checkpoint_path),
            scaler_path=str(scaler_path),
        ),
    )

    assert payload.stream == "patchtst"
    assert len(payload.records) == prepared.windows.shape[0]
    assert payload.metadata["window_count"] == prepared.windows.shape[0]
    assert payload.records[0].stream == "patchtst"
    assert "per_channel_error" in payload.records[0].aux
    assert torch_module.isfinite(
        torch_module.tensor([record.score for record in payload.records], dtype=torch_module.float32)
    ).all().item()


def test_score_swinmae_windows_with_saved_checkpoint(
    torch_module,
    pywt_module,
    swinmae_smoke_config,
    tmp_path: Path,
) -> None:
    _ = pywt_module
    cfg = _swinmae_csv_config(swinmae_smoke_config, tmp_path)
    prepared = prepare_swinmae_batch(cfg)

    checkpoint_path = tmp_path / "checkpoints" / "swinmae_ssl.pt"
    _save_swinmae_checkpoint(torch_module, cfg, checkpoint_path)

    payload = score_windows(
        prepared,
        stream="swinmae",
        config=cfg,
        artifacts=ArtifactPaths(
            thresholds_path="unused",
            swinmae_checkpoint=str(checkpoint_path),
        ),
    )

    assert payload.stream == "swinmae"
    assert len(payload.records) == prepared.windows.shape[0]
    assert payload.metadata["window_count"] == prepared.windows.shape[0]
    assert payload.records[0].stream == "swinmae"
    assert "per_axis_error" in payload.records[0].aux
    assert torch_module.isfinite(
        torch_module.tensor([record.score for record in payload.records], dtype=torch_module.float32)
    ).all().item()
