from __future__ import annotations

import pytest


def _build_patchtst_case(cfg: dict):
    from torch.utils.data import DataLoader

    from datasets.fdc_dataset import build_fdc_datasets
    from models.patchtst.patchtst_ssl import PatchTSTSSL

    datasets = build_fdc_datasets(cfg)
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
    return batch, model


def _build_swinmae_case(cfg: dict):
    from torch.utils.data import DataLoader

    from datasets.vib_dataset import build_vibration_datasets
    from models.swinmae.swinmae_ssl import SwinMAESSL

    datasets = build_vibration_datasets(cfg)
    batch = next(iter(DataLoader(datasets.train, batch_size=1, shuffle=False, num_workers=0)))
    model = SwinMAESSL(
        mask_ratio=float(cfg["model"]["mask_ratio"]),
        patch_size=int(cfg["model"]["patch_size"]),
        use_timm_swin=bool(cfg["model"].get("use_timm_swin", True)),
        timm_name=str(cfg["model"].get("timm_name", "swin_tiny_patch4_window7_224")),
        decoder_dim=int(cfg["model"].get("decoder_dim", 256)),
    )
    return batch, model


@pytest.mark.parametrize("stream", ["patchtst", "swinmae"])
def test_infer_score_smoke(
    stream: str,
    request,
    torch_module,
    patchtst_smoke_config,
    swinmae_smoke_config,
) -> None:
    from inference.scoring import infer_score

    torch_module.manual_seed(0)

    if stream == "patchtst":
        batch, model = _build_patchtst_case(patchtst_smoke_config)
    else:
        request.getfixturevalue("pywt_module")
        if bool(swinmae_smoke_config["model"].get("use_timm_swin", True)):
            request.getfixturevalue("timm_module")
        batch, model = _build_swinmae_case(swinmae_smoke_config)

    model.eval()
    output = infer_score(batch=batch, model=model, stream=stream)

    assert output["score"].shape[0] == batch.shape[0]
    assert torch_module.isfinite(output["score"]).all().item()
    assert isinstance(output["aux"], dict)
