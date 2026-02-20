from __future__ import annotations


def test_patchtst_one_batch_forward(torch_module, patchtst_smoke_config) -> None:
    from torch.utils.data import DataLoader

    from datasets.fdc_dataset import build_fdc_datasets
    from models.patchtst.patchtst_ssl import PatchTSTSSL

    torch_module.manual_seed(0)
    datasets = build_fdc_datasets(patchtst_smoke_config)
    batch = next(iter(DataLoader(datasets.train, batch_size=2, shuffle=False, num_workers=0)))

    cfg = patchtst_smoke_config
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
