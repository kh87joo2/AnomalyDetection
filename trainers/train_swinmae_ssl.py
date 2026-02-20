from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import load_yaml_config, validate_required_keys
from datasets.vib_dataset import build_vibration_datasets
from models.swinmae.swinmae_ssl import SwinMAESSL
from trainers.utils import get_device, make_writer, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SwinMAE SSL")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def train_one_epoch(
    model: SwinMAESSL,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    max_batches: int,
) -> float:
    model.train()
    scaler = torch.amp.GradScaler(enabled=use_amp)

    total_loss = 0.0
    n_steps = 0
    for batch_idx, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        if batch_idx >= max_batches:
            break
        batch = batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(batch)
            loss = model.masked_mse(output)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().cpu())
        n_steps += 1

    return total_loss / max(n_steps, 1)


@torch.no_grad()
def validate(
    model: SwinMAESSL,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    model.eval()
    total_loss = 0.0
    n_steps = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="val", leave=False)):
        if batch_idx >= max_batches:
            break
        batch = batch.to(device)
        output = model(batch)
        loss = model.masked_mse(output)

        total_loss += float(loss.detach().cpu())
        n_steps += 1

    return total_loss / max(n_steps, 1)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    data_source = str(config.get("data", {}).get("source", "synthetic")).lower()
    required_keys = [
        "seed",
        "device.prefer_cuda",
        "device.amp",
        "data.fs",
        "data.train_ratio",
        "data.win_sec",
        "cwt.backend",
        "cwt.freq_min",
        "cwt.freq_max",
        "cwt.n_freqs",
        "image.size",
        "model.mask_ratio",
        "model.patch_size",
        "training.epochs",
        "training.batch_size",
        "training.lr",
        "logging.checkpoint_path",
    ]
    # Keep backward compatibility with existing key while allowing the alias.
    if "win_stride_sec" in config.get("data", {}):
        required_keys.append("data.win_stride_sec")
    else:
        required_keys.append("data.stride_sec")

    if data_source == "synthetic":
        required_keys.append("data.total_steps")
    else:
        required_keys.append("data.path")

    validate_required_keys(config, required_keys)

    if str(config["cwt"].get("backend", "pywt")) != "pywt":
        raise ValueError("Phase 1 decision fixed backend to pywt")

    set_seed(int(config["seed"]), deterministic=bool(config.get("deterministic", False)))
    device = get_device(prefer_cuda=bool(config["device"]["prefer_cuda"]))
    use_amp = bool(config["device"].get("amp", False)) and device.type == "cuda"

    datasets = build_vibration_datasets(config)
    train_loader = DataLoader(
        datasets.train,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 0)),
    )
    val_loader = DataLoader(
        datasets.val,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
    )

    model = SwinMAESSL(
        mask_ratio=float(config["model"]["mask_ratio"]),
        patch_size=int(config["model"]["patch_size"]),
        use_timm_swin=bool(config["model"].get("use_timm_swin", True)),
        timm_name=str(config["model"].get("timm_name", "swin_tiny_patch4_window7_224")),
        decoder_dim=int(config["model"].get("decoder_dim", 256)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    writer = make_writer(config["logging"]["log_dir"])

    epochs = int(config["training"]["epochs"])
    max_train_batches = int(config["training"].get("max_train_batches", 10**9))
    max_val_batches = int(config["training"].get("max_val_batches", 10**9))

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, use_amp, max_train_batches)
        val_loss = validate(model, val_loader, device, max_val_batches)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)

        print(f"[SwinMAE][Epoch {epoch}] train={train_loss:.6f} val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                config["logging"]["checkpoint_path"],
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "val_loss": val_loss,
                },
            )

    writer.close()


if __name__ == "__main__":
    main()
