from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class FASAAttention(nn.Module):
    """Frequency-Aware Sparse Attention (Wang et al., 2026; arXiv:2602.03152).

    A training-free sparse heuristic that estimates token importance from dominant frequency
    chunks and runs full attention only on selected key/value subsets.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for FASAAttention")

        self.n_tip = max(1, int(getattr(config, "fasa_n_tip", 16)))
        self.n_fac = max(1, int(getattr(config, "fasa_n_fac", 256)))

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        pair_dim = max(1, self.head_dim // 2)
        dominant = torch.arange(self.n_tip, dtype=torch.long) % pair_dim
        self.register_buffer("dominant_fcs", dominant.unsqueeze(0).repeat(self.num_heads, 1), persistent=False)

    def _dominant_dim_indices(self, head_idx: int, device: torch.device) -> torch.Tensor:
        fc_idx = self.dominant_fcs[head_idx].to(device)
        dim_idx = torch.stack((fc_idx * 2, fc_idx * 2 + 1), dim=-1).flatten()
        return dim_idx.clamp(max=self.head_dim - 1)

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        out = torch.zeros_like(q)
        n_select = min(self.n_fac, seq_len)

        for h in range(self.num_heads):
            dim_idx = self._dominant_dim_indices(h, x.device)
            q_sub = q[:, h, :, dim_idx]
            k_sub = k[:, h, :, dim_idx]
            importance = torch.matmul(q_sub, k_sub.transpose(-2, -1))
            top_idx = importance.topk(n_select, dim=-1).indices

            for b in range(bsz):
                k_head = k[b, h]
                v_head = v[b, h]
                k_sel = k_head[top_idx[b]]
                v_sel = v_head[top_idx[b]]
                q_head = q[b, h].unsqueeze(1)
                scores = (q_head * k_sel).sum(dim=-1) / math.sqrt(self.head_dim)
                attn = F.softmax(scores, dim=-1)
                out[b, h] = torch.sum(attn.unsqueeze(-1) * v_sel, dim=1)

        out = self.dropout(out)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
