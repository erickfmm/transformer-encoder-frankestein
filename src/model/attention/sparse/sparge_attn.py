from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class SpargeAttention(nn.Module):
    """SpargeAttn (Zhang et al., 2025; arXiv:2502.18137).

    A training-free two-stage sparse filter that predicts low-value attention blocks and then
    applies a softmax-aware pruning pass to reduce block computations.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for SpargeAttention")

        self.block_size = max(1, int(getattr(config, "sparge_block_size", 64)))
        self.threshold = float(getattr(config, "sparge_threshold", 0.01))

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        bs = self.block_size
        n_blocks = (seq_len + bs - 1) // bs
        padded_len = n_blocks * bs
        pad_len = padded_len - seq_len

        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))

        q_blocks = q.view(bsz, self.num_heads, n_blocks, bs, self.head_dim)
        k_blocks = k.view(bsz, self.num_heads, n_blocks, bs, self.head_dim)

        q_mean = q_blocks.mean(dim=3)
        k_mean = k_blocks.mean(dim=3)
        block_scores = torch.matmul(q_mean, k_mean.transpose(-2, -1)) / math.sqrt(self.head_dim)
        block_mask = block_scores > self.threshold

        full_mask = block_mask.unsqueeze(3).unsqueeze(5).expand(-1, -1, -1, bs, -1, bs)
        full_mask = full_mask.reshape(bsz, self.num_heads, padded_len, padded_len)

        eye = torch.eye(padded_len, device=x.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        full_mask = full_mask | eye

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~full_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        attn_blocks = attn.view(bsz, self.num_heads, n_blocks, bs, n_blocks, bs)
        block_max = attn_blocks.amax(dim=(3, 5))
        softmax_mask = block_max > self.threshold
        softmax_full = softmax_mask.unsqueeze(3).unsqueeze(5).expand(-1, -1, -1, bs, -1, bs)
        softmax_full = softmax_full.reshape(bsz, self.num_heads, padded_len, padded_len)
        softmax_full = softmax_full | eye

        attn = attn.masked_fill(~softmax_full, 0.0)

        out = torch.matmul(attn, v)
        out = out[:, :, :seq_len, :]
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
