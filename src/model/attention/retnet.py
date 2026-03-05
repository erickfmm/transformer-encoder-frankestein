import math

import torch
import torch.nn as nn

from .common import BitLinear, get_norm


class MultiScaleRetention(nn.Module):
    """
    Implementacion paralela de RetNet con BitLinear.
    Referencia: 'Retentive Network: A Successor to Transformer...'
    """

    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size
        self.heads = config.retention_heads
        self.head_dim = self.dim // self.heads
        self.scale = self.head_dim ** -0.5

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.dim, self.dim, bias=False)
        self.k_proj = proj_cls(self.dim, self.dim, bias=False)
        self.v_proj = proj_cls(self.dim, self.dim, bias=False)
        self.g_proj = proj_cls(self.dim, self.dim, bias=False)
        self.out_proj = proj_cls(self.dim, self.dim, bias=False)
        self.swish = nn.SiLU()
        self.norm = get_norm(config)

        self.register_buffer("decay_mask", self._build_decay_mask(config.hidden_size, 2048))

    def _build_decay_mask(self, dim, max_len=2048):
        gammas = 1 - torch.exp(torch.linspace(math.log(1 / 32), math.log(1 / 512), self.heads))
        return gammas.view(self.heads, 1, 1)

    def forward(self, x):
        bsz, seq_len, dim = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        gammas = 1 - torch.exp(torch.linspace(math.log(1 / 32), math.log(1 / 512), self.heads, device=x.device))
        n = torch.arange(seq_len, device=x.device).unsqueeze(1)
        m = torch.arange(seq_len, device=x.device).unsqueeze(0)
        dist = n - m

        decay_matrix = gammas.view(self.heads, 1, 1) ** dist.abs().unsqueeze(0)
        causal_mask = (dist >= 0).float().unsqueeze(0)

        retention_scores = attn * decay_matrix * causal_mask

        y = retention_scores @ v
        y = y.transpose(1, 2).reshape(bsz, seq_len, dim)
        y = self.norm(y)

        g = self.swish(self.g_proj(x))
        out = y * g

        return self.out_proj(out)
