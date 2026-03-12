from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class DeltaNetAttention(nn.Module):
    """DeltaNet attention (Yang et al., 2024; arXiv:2406.06484).

    Applies a delta-rule state update to recurrent linear attention, performing targeted
    error-correcting writes instead of purely additive memory updates.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.total_dim = self.head_dim * self.num_heads

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for DeltaNetAttention")

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.beta_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.g_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.out_proj = proj_cls(self.total_dim, self.hidden_size, bias=False)
        self.norm = nn.LayerNorm(self.total_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = F.normalize(self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim), dim=-1)
        k = F.normalize(self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim), dim=-1)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        beta = torch.sigmoid(self.beta_proj(x))

        state = torch.zeros(
            bsz,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            device=x.device,
            dtype=x.dtype,
        )

        outputs = []
        for t in range(seq_len):
            b_t = beta[:, t, :, None, None]
            k_t = k[:, t]
            v_t = v[:, t]
            kk = k_t.unsqueeze(-1) * k_t.unsqueeze(-2)
            state = state * (1 - b_t * kk) + b_t * (v_t.unsqueeze(-1) * k_t.unsqueeze(-2))
            out_t = (state * q[:, t].unsqueeze(-2)).sum(-1)
            outputs.append(out_t)

        out = torch.stack(outputs, dim=1).reshape(bsz, seq_len, self.total_dim)
        out = self.norm(out)
        out = out * F.silu(self.g_proj(x))
        out = self.dropout(out)
        return self.out_proj(out)
