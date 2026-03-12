from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class GatedLinearAttention(nn.Module):
    """Gated Linear Attention (Yang et al., 2023; arXiv:2312.06635).

    Uses data-dependent gates in a recurrent linear-attention state update to control
    memory retention and reduce overload from purely additive accumulation.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.total_dim = self.head_dim * self.num_heads

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for GatedLinearAttention")

        gate_low_rank = max(1, int(getattr(config, "gla_gate_low_rank", 16)))

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.g_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.gk_proj = nn.Sequential(
            nn.Linear(self.hidden_size, gate_low_rank, bias=False),
            nn.Linear(gate_low_rank, self.total_dim, bias=True),
        )
        self.out_proj = proj_cls(self.total_dim, self.hidden_size, bias=False)
        self.norm = nn.LayerNorm(self.total_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        gk = F.logsigmoid(self.gk_proj(x)).view(bsz, seq_len, self.num_heads, self.head_dim) / 16.0

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
            gate = torch.exp(gk[:, t])
            state = state * gate.unsqueeze(-1) + v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(-2)
            out_t = (state * q[:, t].unsqueeze(-2)).sum(-1)
            outputs.append(out_t)

        out = torch.stack(outputs, dim=1).reshape(bsz, seq_len, self.total_dim)
        out = self.norm(out)
        out = out * F.silu(self.g_proj(x))
        out = self.dropout(out)
        return self.out_proj(out)
