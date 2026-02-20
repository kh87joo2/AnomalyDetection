from __future__ import annotations


def test_swinmae_one_batch_forward(
    request,
    torch_module,
    pywt_module,
    swinmae_smoke_config,
) -> None:
    from torch.utils.data import DataLoader

    cfg = swinmae_smoke_config
    if bool(cfg["model"].get("use_timm_swin", True)):
        request.getfixturevalue("timm_module")

    from datasets.vib_dataset import build_vibration_datasets
    from models.swinmae.swinmae_ssl import SwinMAESSL

    _ = pywt_module
    torch_module.manual_seed(0)
    datasets = build_vibration_datasets(cfg)
    batch = next(iter(DataLoader(datasets.train, batch_size=1, shuffle=False, num_workers=0)))

    model = SwinMAESSL(
        mask_ratio=float(cfg["model"]["mask_ratio"]),
        patch_size=int(cfg["model"]["patch_size"]),
        use_timm_swin=bool(cfg["model"].get("use_timm_swin", True)),
        timm_name=str(cfg["model"].get("timm_name", "swin_tiny_patch4_window7_224")),
        decoder_dim=int(cfg["model"].get("decoder_dim", 256)),
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
