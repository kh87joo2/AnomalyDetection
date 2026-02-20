from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.swinmae.mask_ops import random_image_patch_mask

try:
    import timm
except Exception:  # pragma: no cover
    timm = None


@dataclass
class SwinMAEOutput:
    recon: torch.Tensor  # (B, 3, H, W)
    target: torch.Tensor  # (B, 3, H, W)
    pixel_mask: torch.Tensor  # (B,1,H,W)


class ConvFallbackEncoder(nn.Module):
    def __init__(self, in_chans: int = 3, out_chans: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(256, out_chans, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LightDecoder(nn.Module):
    def __init__(self, in_chans: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, 3, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        y = self.net(x)
        return F.interpolate(y, size=out_size, mode="bilinear", align_corners=False)


class SwinMAESSL(nn.Module):
    def __init__(
        self,
        mask_ratio: float,
        patch_size: int,
        use_timm_swin: bool = True,
        timm_name: str = "swin_tiny_patch4_window7_224",
        decoder_dim: int = 256,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

        self._uses_timm = bool(use_timm_swin and timm is not None)
        if self._uses_timm:
            self.encoder = timm.create_model(timm_name, pretrained=False, num_classes=0)
            enc_chans = getattr(self.encoder, "num_features", 768)
        else:
            self.encoder = ConvFallbackEncoder(in_chans=3, out_chans=384)
            enc_chans = 384

        self.decoder = LightDecoder(in_chans=enc_chans, hidden=decoder_dim)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self._uses_timm:
            feat = self.encoder.forward_features(x)
            # timm Swin often returns (B, H, W, C)
            if feat.ndim == 4 and feat.shape[-1] > feat.shape[1]:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            elif feat.ndim == 3:
                # fallback if returned as tokens (B, N, C)
                b, n, c = feat.shape
                s = int(n**0.5)
                feat = feat.transpose(1, 2).reshape(b, c, s, s)
            return feat
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> SwinMAEOutput:
        masked_x, pixel_mask, _ = random_image_patch_mask(
            x, patch_size=self.patch_size, mask_ratio=self.mask_ratio
        )
        feat = self._forward_features(masked_x)
        recon = self.decoder(feat, out_size=(x.shape[-2], x.shape[-1]))
        return SwinMAEOutput(recon=recon, target=x, pixel_mask=pixel_mask)

    @staticmethod
    def masked_mse(output: SwinMAEOutput) -> torch.Tensor:
        diff2 = (output.recon - output.target) ** 2
        mask = output.pixel_mask
        denom = mask.sum().clamp_min(1.0) * output.recon.shape[1]
        loss = (diff2 * mask).sum() / denom
        return loss

    @staticmethod
    def score_from_output(output: SwinMAEOutput) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        diff2 = (output.recon - output.target) ** 2
        mask = output.pixel_mask
        per_sample = (diff2 * mask).sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)).clamp_min(1.0) * output.recon.shape[1])

        per_axis = (diff2 * mask).sum(dim=(2, 3)) / mask.sum(dim=(2, 3)).clamp_min(1.0)
        return per_sample, {"per_axis_error": per_axis, "mask_ratio_effective": mask.mean(dim=(1, 2, 3))}
