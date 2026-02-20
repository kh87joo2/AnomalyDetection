from __future__ import annotations

import torch

from models.swinmae.swinmae_ssl import SwinMAESSL


@torch.no_grad()
def infer_swinmae_score(model: SwinMAESSL, batch: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output = model(batch)
    return model.score_from_output(output)
