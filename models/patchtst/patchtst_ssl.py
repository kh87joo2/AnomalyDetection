from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.patchtst.patch_ops import patchify, random_patch_mask


@dataclass
class PatchTSTOutput:
    recon: torch.Tensor  # (B, C, N, P)
    target: torch.Tensor  # (B, C, N, P)
    mask: torch.Tensor  # (B, C, N) bool


class PatchTSTSSL(nn.Module):
    def __init__(
        self,
        seq_len: int,
        patch_len: int,
        patch_stride: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        mask_ratio: float,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.mask_ratio = mask_ratio

        n_patches = 1 + (seq_len - patch_len) // patch_stride
        self.input_proj = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_patches, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, patch_len)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> PatchTSTOutput:
        # x: (B, T, C)
        patches = patchify(x, patch_len=self.patch_len, patch_stride=self.patch_stride)
        b, c, n, p = patches.shape

        tokens = patches.reshape(b * c, n, p)
        mask = random_patch_mask((b * c, n), self.mask_ratio, x.device)

        masked_tokens = tokens.clone()
        masked_tokens[mask] = 0.0

        z = self.input_proj(masked_tokens) + self.pos_emb[:, :n, :]
        h = self.encoder(z)
        recon = self.head(h)

        recon = recon.reshape(b, c, n, p)
        target = patches
        mask = mask.reshape(b, c, n)

        return PatchTSTOutput(recon=recon, target=target, mask=mask)

    @staticmethod
    def masked_mse(output: PatchTSTOutput) -> torch.Tensor:
        diff2 = (output.recon - output.target) ** 2  # (B,C,N,P)
        mask = output.mask.unsqueeze(-1).float()  # (B,C,N,1)
        denom = mask.sum().clamp_min(1.0) * output.recon.shape[-1]
        loss = (diff2 * mask).sum() / denom
        return loss

    @staticmethod
    def score_from_output(output: PatchTSTOutput) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Per-sample anomaly-like score from masked reconstruction error."""
        diff2 = (output.recon - output.target) ** 2
        mask = output.mask.unsqueeze(-1).float()

        per_sample = (diff2 * mask).sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)).clamp_min(1.0) * output.recon.shape[-1])
        per_channel = (diff2 * mask).sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)).clamp_min(1.0) * output.recon.shape[-1])
        return per_sample, {"per_channel_error": per_channel, "mask_ratio_effective": output.mask.float().mean(dim=(1, 2))}
