from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class SparseTransformerAttention(nn.Module):
    """Sparse Transformer attention (Child et al., 2019; arXiv:1904.10509).

    Implements a factorized sparse pattern that mixes strided and fixed blocks to reduce
    dense attention cost while preserving long-range information flow.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for SparseTransformerAttention")

        self.stride = int(getattr(config, "sparse_transformer_stride", 0))
        self.summary_cols = max(1, int(getattr(config, "sparse_transformer_summary_cols", 1)))

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def _resolved_stride(self, seq_len: int) -> int:
        if self.stride > 0:
            return self.stride
        return max(1, int(math.sqrt(max(1, seq_len))))

    def _strided_mask(self, seq_len: int, stride: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        for i in range(seq_len):
            indices = list(range(max(0, i - stride + 1), i + 1))
            indices += list(range(i % stride, i + 1, stride))
            if indices:
                mask[i, torch.tensor(sorted(set(indices)), device=device)] = True
        return mask

    def _fixed_mask(self, seq_len: int, stride: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        for i in range(seq_len):
            block_start = (i // stride) * stride
            block_end = min(i + 1, block_start + stride)
            indices = list(range(block_start, block_end))
            indices += [
                j
                for j in range(i + 1)
                if (j % stride) >= max(0, stride - self.summary_cols)
            ]
            if indices:
                mask[i, torch.tensor(sorted(set(indices)), device=device)] = True
        return mask

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape
        stride = self._resolved_stride(seq_len)

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale

        strided_mask = self._strided_mask(seq_len, stride, x.device)
        fixed_mask = self._fixed_mask(seq_len, stride, x.device)

        half = max(1, self.num_heads // 2)
        scores[:, :half] = scores[:, :half].masked_fill(~strided_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        scores[:, half:] = scores[:, half:].masked_fill(~fixed_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
