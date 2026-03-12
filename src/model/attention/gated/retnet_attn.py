from typing import Optional

import torch
import torch.nn as nn

from ..retnet import MultiScaleRetention


class RetNetAttention(nn.Module):
    """RetNet retention attention (Sun et al., 2023; arXiv:2307.08621).

    Wraps the existing multi-scale retention implementation to expose an explicit
    `retnet_attn` block name in the new gated-attention package.
    """

    def __init__(self, config):
        super().__init__()
        self.inner = MultiScaleRetention(config)

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        return self.inner(x)
