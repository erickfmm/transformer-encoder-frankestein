from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class SparseKOperator(torch.autograd.Function):
    """Differentiable SparseK projection used by SparseK attention."""

    @staticmethod
    def forward(ctx, scores: torch.Tensor, k: int):
        sorted_scores, _ = scores.sort(dim=-1, descending=True)
        cumsum = sorted_scores.cumsum(dim=-1)
        arange = torch.arange(1, scores.shape[-1] + 1, device=scores.device, dtype=scores.dtype)
        threshold = (cumsum - float(k)) / arange
        support = sorted_scores > threshold
        rho = support.sum(dim=-1, keepdim=True).clamp(min=1)
        tau = (cumsum.gather(-1, rho - 1) - float(k)) / rho.to(scores.dtype)
        out = (scores - tau).clamp(min=0)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (out,) = ctx.saved_tensors
        support = (out > 0).to(grad_output.dtype)
        n_support = support.sum(dim=-1, keepdim=True).clamp(min=1)
        grad = support * (grad_output - (grad_output * support).sum(dim=-1, keepdim=True) / n_support)
        return grad, None


class SparseKAttention(nn.Module):
    """SparseK attention (Lou et al., 2024; arXiv:2406.16747).

    Applies a differentiable top-k style selection over key/value candidates before
    running scaled dot-product attention on the selected subset.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for SparseKAttention")

        self.k = max(1, int(getattr(config, "sparsek_k", 128)))

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)

        self.score_net = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 1),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_scores = self.score_net(k).squeeze(-1)
        selection = SparseKOperator.apply(kv_scores, self.k)

        k_actual = min(self.k, seq_len)
        _, top_idx = selection.topk(k_actual, dim=-1)
        top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)

        k_sel = k.gather(2, top_idx_exp)
        v_sel = v.gather(2, top_idx_exp)

        scores = torch.matmul(q, k_sel.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v_sel).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
