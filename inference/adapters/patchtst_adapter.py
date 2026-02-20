from __future__ import annotations

import torch

from models.patchtst.patchtst_ssl import PatchTSTSSL


@torch.no_grad()
def infer_patchtst_score(model: PatchTSTSSL, batch: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output = model(batch)
    return model.score_from_output(output)
