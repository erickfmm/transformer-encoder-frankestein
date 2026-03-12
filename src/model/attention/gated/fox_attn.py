from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class ForgettingAttention(nn.Module):
    """Forgetting Transformer attention (Lin et al., 2025; arXiv:2503.02130).

    Injects a learned forget gate into softmax logits via cumulative log-bias terms to
    control recency while keeping full softmax attention expressiveness.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.total_dim = self.head_dim * self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for ForgettingAttention")

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.f_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        self.out_proj = proj_cls(self.total_dim, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        f = torch.sigmoid(self.f_proj(x)).permute(0, 2, 1)
        log_f = torch.log(f + 1e-6)
        cum_log_f = torch.cumsum(log_f, dim=-1)

        bias = cum_log_f.unsqueeze(-1) - cum_log_f.unsqueeze(-2)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        bias = bias.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = (q @ k.transpose(-2, -1)) * self.scale + bias
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(bsz, seq_len, self.total_dim)
        return self.out_proj(out)
