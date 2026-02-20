from __future__ import annotations

import torch


def patchify(x: torch.Tensor, patch_len: int, patch_stride: int) -> torch.Tensor:
    """Patchify sequence x: (B, T, C) -> (B, C, N, patch_len)."""
    if x.ndim != 3:
        raise ValueError(f"Expected x shape (B, T, C), got {x.shape}")
    b, t, c = x.shape
    if t < patch_len:
        raise ValueError(f"T={t} is smaller than patch_len={patch_len}")

    # Unfold along time dimension per channel.
    x_bc_t = x.permute(0, 2, 1)  # (B, C, T)
    patches = x_bc_t.unfold(dimension=-1, size=patch_len, step=patch_stride)
    # (B, C, N, patch_len)
    return patches.contiguous()


def random_patch_mask(shape: tuple[int, int], mask_ratio: float, device: torch.device) -> torch.Tensor:
    """Return boolean mask with shape (B*C, N), True means masked."""
    bc, n = shape
    mask = torch.rand((bc, n), device=device) < mask_ratio
    return mask
