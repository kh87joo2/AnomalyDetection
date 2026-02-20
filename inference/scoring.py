from __future__ import annotations

from typing import Literal

import torch

from core.types import ScoreOutput
from inference.adapters.patchtst_adapter import infer_patchtst_score
from inference.adapters.swinmae_adapter import infer_swinmae_score


def infer_score(
    batch: torch.Tensor,
    model: torch.nn.Module,
    stream: Literal["patchtst", "swinmae"],
) -> ScoreOutput:
    if stream == "patchtst":
        score, aux = infer_patchtst_score(model, batch)
        return {"score": score, "aux": aux}
    if stream == "swinmae":
        score, aux = infer_swinmae_score(model, batch)
        return {"score": score, "aux": aux}
    raise ValueError(f"Unsupported stream: {stream}")
