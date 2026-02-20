from __future__ import annotations

import torch


def random_image_patch_mask(
    x: torch.Tensor,
    patch_size: int,
    mask_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask image in patch units.

    Returns:
        masked_x: (B, C, H, W)
        pixel_mask: (B, 1, H, W), 1 for masked pixels
        patch_mask: (B, Npatch), True for masked patches
    """
    b, c, h, w = x.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"H,W must be divisible by patch_size. got {(h, w)} / {patch_size}")

    nh = h // patch_size
    nw = w // patch_size
    n_patches = nh * nw

    patch_mask = (torch.rand((b, n_patches), device=x.device) < mask_ratio)

    pixel_mask = patch_mask.view(b, nh, nw)
    pixel_mask = pixel_mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    pixel_mask = pixel_mask.unsqueeze(1).float()

    masked_x = x * (1.0 - pixel_mask)
    return masked_x, pixel_mask, patch_mask
