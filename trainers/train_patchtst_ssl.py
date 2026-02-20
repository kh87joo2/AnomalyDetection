from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import load_yaml_config, validate_required_keys
from datasets.fdc_dataset import build_fdc_datasets
from models.patchtst.patchtst_ssl import PatchTSTSSL
from trainers.utils import get_device, make_writer, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PatchTST SSL")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def train_one_epoch(
    model: PatchTSTSSL,
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
    model: PatchTSTSSL,
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
        "data.seq_len",
        "data.seq_stride",
        "data.train_ratio",
        "model.patch_len",
        "model.patch_stride",
        "model.mask_ratio",
        "training.epochs",
        "training.batch_size",
        "training.lr",
        "logging.checkpoint_path",
    ]
    if data_source == "synthetic":
        required_keys.extend(["data.channels", "data.total_steps"])
    else:
        required_keys.append("data.path")
    validate_required_keys(config, required_keys)

    set_seed(int(config["seed"]), deterministic=bool(config.get("deterministic", False)))
    device = get_device(prefer_cuda=bool(config["device"]["prefer_cuda"]))
    use_amp = bool(config["device"].get("amp", False)) and device.type == "cuda"

    datasets = build_fdc_datasets(config)
    scaler_path = Path("artifacts/scaler_fdc.json")
    datasets.scaler.save(scaler_path)

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

    model = PatchTSTSSL(
        seq_len=int(config["data"]["seq_len"]),
        patch_len=int(config["model"]["patch_len"]),
        patch_stride=int(config["model"]["patch_stride"]),
        d_model=int(config["model"]["d_model"]),
        nhead=int(config["model"]["nhead"]),
        num_layers=int(config["model"]["num_layers"]),
        ff_dim=int(config["model"]["ff_dim"]),
        dropout=float(config["model"]["dropout"]),
        mask_ratio=float(config["model"]["mask_ratio"]),
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

        print(f"[PatchTST][Epoch {epoch}] train={train_loss:.6f} val={val_loss:.6f}")

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
                    "scaler_path": str(scaler_path),
                },
            )

    writer.close()


if __name__ == "__main__":
    main()
