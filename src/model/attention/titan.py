from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BitLinear
from .hope import HoPE
from .rope import RoPE


class TitanAttention(nn.Module):
    """Real multi-head Titan attention using BitLinear projections and HoPE/RoPE on q/k."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for TitanAttention")

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)

        positional_encoding = getattr(config, "positional_encoding", None)
        if positional_encoding is None:
            positional_encoding = "hope" if bool(getattr(config, "use_hope", True)) else "rope"
        positional_encoding = str(positional_encoding).lower()

        if positional_encoding == "hope":
            self.pos_encoder = HoPE(self.head_dim, base=config.hope_base, damping=config.hope_damping)
        elif positional_encoding == "rope":
            self.pos_encoder = RoPE(
                self.head_dim,
                base=getattr(config, "rope_base", 10_000.0),
                scaling=getattr(config, "rope_scaling", 1.0),
            )
        else:
            raise ValueError(
                "positional_encoding must be one of {'hope', 'rope'} for TitanAttention"
            )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape
        logical_layer_idx = logical_layer_idx or 0

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.pos_encoder(q, logical_layer_idx=logical_layer_idx)
        k = self.pos_encoder(k, logical_layer_idx=logical_layer_idx)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
