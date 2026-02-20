from __future__ import annotations

from typing import Any, TypedDict

import torch


class ScoreOutput(TypedDict):
    score: torch.Tensor
    aux: dict[str, torch.Tensor | Any]
