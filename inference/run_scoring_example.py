from __future__ import annotations

import argparse
from typing import Any, Literal

import torch

from core.config import load_yaml_config
from datasets.fdc_dataset import build_fdc_datasets
from datasets.vib_dataset import build_vibration_datasets
from inference.checkpoint_io import load_checkpoint
from inference.scoring import infer_score
from models.patchtst.patchtst_ssl import PatchTSTSSL
from models.swinmae.swinmae_ssl import SwinMAESSL

StreamName = Literal["patchtst", "swinmae"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-batch scoring example for a selected stream."
    )
    parser.add_argument(
        "--stream", type=str, choices=["patchtst", "swinmae"], required=True
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def _select_device(config: dict[str, Any]) -> torch.device:
    prefer_cuda = bool(config.get("device", {}).get("prefer_cuda", True))
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model(
    stream: StreamName, config: dict[str, Any], device: torch.device
) -> torch.nn.Module:
    model_cfg = config["model"]
    if stream == "patchtst":
        data_cfg = config["data"]
        model = PatchTSTSSL(
            seq_len=int(data_cfg["seq_len"]),
            patch_len=int(model_cfg["patch_len"]),
            patch_stride=int(model_cfg["patch_stride"]),
            d_model=int(model_cfg["d_model"]),
            nhead=int(model_cfg["nhead"]),
            num_layers=int(model_cfg["num_layers"]),
            ff_dim=int(model_cfg["ff_dim"]),
            dropout=float(model_cfg["dropout"]),
            mask_ratio=float(model_cfg["mask_ratio"]),
        )
    else:
        model = SwinMAESSL(
            mask_ratio=float(model_cfg["mask_ratio"]),
            patch_size=int(model_cfg["patch_size"]),
            use_timm_swin=bool(model_cfg.get("use_timm_swin", True)),
            timm_name=str(
                model_cfg.get("timm_name", "swin_tiny_patch4_window7_224")
            ),
            decoder_dim=int(model_cfg.get("decoder_dim", 256)),
        )
    return model.to(device)


def _build_synthetic_batch(
    stream: StreamName, config: dict[str, Any]
) -> torch.Tensor:
    if stream == "patchtst":
        datasets = build_fdc_datasets(config)
        dataset = datasets.val if len(datasets.val) > 0 else datasets.train
    else:
        datasets = build_vibration_datasets(config)
        dataset = datasets.val if len(datasets.val) > 0 else datasets.train

    if len(dataset) == 0:
        raise ValueError("Dataset builder returned an empty dataset.")
    return dataset[0].unsqueeze(0)


def _extract_model_state_dict(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a mapping.")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid model_state_dict.")
    return state_dict


def main() -> None:
    args = parse_args()
    stream: StreamName = args.stream

    config = load_yaml_config(args.config)
    device = _select_device(config)

    model = _build_model(stream, config, device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    state_dict = _extract_model_state_dict(checkpoint)

    incompatible = model.load_state_dict(state_dict, strict=False)
    model.eval()

    batch = _build_synthetic_batch(stream, config).to(device)
    output = infer_score(batch=batch, model=model, stream=stream)

    score = output["score"].detach().cpu()
    aux = output["aux"]

    print(f"loaded checkpoint: {args.checkpoint}")
    print(
        "load_state_dict(strict=False): "
        f"missing={len(incompatible.missing_keys)}, "
        f"unexpected={len(incompatible.unexpected_keys)}"
    )
    print(f"score shape: {tuple(score.shape)}")
    print(f"score sample: {score.flatten()[:5].tolist()}")
    print(f"aux keys: {sorted(aux.keys())}")


if __name__ == "__main__":
    main()
