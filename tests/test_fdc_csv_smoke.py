from __future__ import annotations

import copy
from pathlib import Path


def test_patchtst_csv_one_batch_forward(torch_module, patchtst_smoke_config, tmp_path) -> None:
    from torch.utils.data import DataLoader

    from datasets.fdc_dataset import build_fdc_datasets
    from models.patchtst.patchtst_ssl import PatchTSTSSL

    cfg = copy.deepcopy(patchtst_smoke_config)
    csv_path = Path(__file__).resolve().parent / "smoke" / "data" / "fdc_dummy.csv"

    cfg["data"].update(
        {
            "source": "csv",
            "path": str(csv_path),
            "timestamp_col": "timestamp",
            "train_ratio": 0.5,
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

    datasets = build_fdc_datasets(cfg)
    assert len(datasets.train) > 0

    batch = next(iter(DataLoader(datasets.train, batch_size=2, shuffle=False, num_workers=0)))

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

    model.eval()
    with torch_module.no_grad():
        output = model(batch)
        loss = model.masked_mse(output)

    assert output.recon.shape == output.target.shape
    assert output.mask.shape == output.recon.shape[:-1]
    assert torch_module.isfinite(loss).item()
