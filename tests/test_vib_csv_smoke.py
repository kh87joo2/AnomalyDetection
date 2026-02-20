from __future__ import annotations

import copy
from pathlib import Path


def test_swinmae_csv_one_batch_forward(
    torch_module,
    pywt_module,
    swinmae_smoke_config,
    tmp_path,
) -> None:
    from torch.utils.data import DataLoader

    from datasets.vib_dataset import build_vibration_datasets
    from models.swinmae.swinmae_ssl import SwinMAESSL

    _ = pywt_module

    cfg = copy.deepcopy(swinmae_smoke_config)
    csv_path = Path(__file__).resolve().parent / "smoke" / "data" / "vib_dummy.csv"

    cfg["data"].update(
        {
            "source": "csv",
            "path": str(csv_path),
            "timestamp_col": "timestamp",
            "fs": 64,
            "train_ratio": 0.5,
            "win_sec": 0.5,
            "win_stride_sec": 0.25,
            "assume_actual_fs_equals_config": True,
            "resample": {
                "enabled": False,
                "method": "linear",
            },
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

    datasets = build_vibration_datasets(cfg)
    assert len(datasets.train) > 0

    batch = next(iter(DataLoader(datasets.train, batch_size=1, shuffle=False, num_workers=0)))

    model = SwinMAESSL(
        mask_ratio=float(cfg["model"]["mask_ratio"]),
        patch_size=int(cfg["model"]["patch_size"]),
        use_timm_swin=False,
        decoder_dim=int(cfg["model"]["decoder_dim"]),
    )

    model.eval()
    with torch_module.no_grad():
        output = model(batch)
        loss = model.masked_mse(output)

    assert output.recon.shape == output.target.shape
    assert output.pixel_mask.shape == (
        output.recon.shape[0],
        1,
        output.recon.shape[2],
        output.recon.shape[3],
    )
    assert torch_module.isfinite(loss).item()
