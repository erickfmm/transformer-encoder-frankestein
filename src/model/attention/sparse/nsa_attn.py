from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class NSAAttention(nn.Module):
    """Native Sparse Attention (Yuan et al., 2025; arXiv:2502.11089).

    Uses three sparse branches (compress, select, and local window) and blends them with
    learned gates to approximate dense attention with trainable sparse structure.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for NSAAttention")

        self.block_size = max(1, int(getattr(config, "nsa_block_size", 32)))
        self.stride = max(1, int(getattr(config, "nsa_stride", 16)))
        self.select_block_size = max(1, int(getattr(config, "nsa_select_block_size", 64)))
        self.n_select = max(1, int(getattr(config, "nsa_n_select", 16)))
        self.window_size = max(1, int(getattr(config, "nsa_window_size", 512)))

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)

        self.compress_k = nn.Linear(self.block_size * self.head_dim, self.head_dim)
        self.compress_v = nn.Linear(self.block_size * self.head_dim, self.head_dim)
        self.gate = nn.Sequential(nn.Linear(self.head_dim, 3), nn.Sigmoid())
        self.dropout = nn.Dropout(config.dropout)

    def _compress(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, heads, seq_len, dim = k.shape

        if seq_len <= self.block_size:
            pad_len = self.block_size - seq_len
            if pad_len > 0:
                k_pad = F.pad(k, (0, 0, 0, pad_len))
                v_pad = F.pad(v, (0, 0, 0, pad_len))
            else:
                k_pad = k
                v_pad = v

            comp_k = self.compress_k(k_pad.reshape(bsz, heads, -1)).unsqueeze(2)
            comp_v = self.compress_v(v_pad.reshape(bsz, heads, -1)).unsqueeze(2)
            return comp_k, comp_v

        n_blocks = (seq_len - self.block_size) // self.stride + 1
        comp_k_list = []
        comp_v_list = []
        for i in range(n_blocks):
            start = i * self.stride
            end = start + self.block_size
            k_block = k[:, :, start:end, :].reshape(bsz, heads, -1)
            v_block = v[:, :, start:end, :].reshape(bsz, heads, -1)
            comp_k_list.append(self.compress_k(k_block))
            comp_v_list.append(self.compress_v(v_block))

        comp_k = torch.stack(comp_k_list, dim=2)
        comp_v = torch.stack(comp_v_list, dim=2)
        return comp_k, comp_v

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        comp_k, comp_v = self._compress(k, v)
        comp_scores = torch.matmul(q, comp_k.transpose(-2, -1)) * self.scale
        comp_attn = F.softmax(comp_scores, dim=-1)
        out_comp = torch.matmul(comp_attn, comp_v)

        block_importance = comp_attn.mean(dim=1).mean(dim=1)
        n_sel = min(self.n_select, block_importance.shape[-1])
        _, top_blocks = block_importance.topk(n_sel, dim=-1)

        sel_indices = []
        for b_idx in top_blocks.unbind(-1):
            start = b_idx * self.stride
            idx = torch.arange(self.select_block_size, device=x.device).unsqueeze(0) + start.unsqueeze(-1)
            sel_indices.append(idx.clamp(max=seq_len - 1))

        sel_idx = torch.cat(sel_indices, dim=-1)
        sel_idx = sel_idx.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, self.head_dim)
        k_sel = k.gather(2, sel_idx)
        v_sel = v.gather(2, sel_idx)

        sel_scores = torch.matmul(q, k_sel.transpose(-2, -1)) * self.scale
        sel_attn = F.softmax(sel_scores, dim=-1)
        out_sel = torch.matmul(sel_attn, v_sel)

        win_size = min(self.window_size, seq_len)
        k_win = k[:, :, -win_size:, :]
        v_win = v[:, :, -win_size:, :]
        win_scores = torch.matmul(q, k_win.transpose(-2, -1)) * self.scale
        win_attn = F.softmax(win_scores, dim=-1)
        out_win = torch.matmul(win_attn, v_win)

        gates = self.gate(q.mean(dim=2))
        g_comp = gates[..., 0:1].unsqueeze(2)
        g_sel = gates[..., 1:2].unsqueeze(2)
        g_win = gates[..., 2:3].unsqueeze(2)

        out = g_comp * out_comp + g_sel * out_sel + g_win * out_win
        out = self.dropout(out)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
