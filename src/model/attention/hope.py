import math

import torch
import torch.nn as nn


class HoPE(nn.Module):
    """Hyperbolic positional encoding over dim pairs with monotonic exponential damping."""

    def __init__(self, head_dim: int, base: float = 10_000.0, damping: float = 0.01):
        super().__init__()
        self.head_dim = head_dim
        self.pair_dim = head_dim // 2
        self.base = base
        self.damping = damping

    def forward(self, x: torch.Tensor, logical_layer_idx: int = 0) -> torch.Tensor:
        # x: [B, H, N, Dh]
        if self.pair_dim == 0:
            return x

        _, _, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        pos = torch.arange(seq_len, device=device, dtype=dtype)
        if self.pair_dim > 1:
            idx = torch.arange(self.pair_dim, device=device, dtype=dtype)
            inv_freq = torch.exp(-math.log(self.base) * idx / (self.pair_dim - 1))
        else:
            inv_freq = torch.ones(1, device=device, dtype=dtype)

        layer_scale = 1.0 + 0.05 * float(logical_layer_idx)
        angles = (pos[:, None] * inv_freq[None, :] * layer_scale).clamp(-12.0, 12.0)

        damping = torch.exp(-(self.damping * layer_scale) * pos).unsqueeze(-1)
        cosh_term = torch.cosh(angles) * damping
        sinh_term = torch.sinh(angles) * damping

        x_even = x[..., : self.pair_dim * 2 : 2]
        x_odd = x[..., 1 : self.pair_dim * 2 : 2]

        cosh_term = cosh_term.unsqueeze(0).unsqueeze(0)
        sinh_term = sinh_term.unsqueeze(0).unsqueeze(0)

        y_even = x_even * cosh_term + x_odd * sinh_term
        y_odd = x_even * sinh_term + x_odd * cosh_term

        y = x.clone()
        y[..., : self.pair_dim * 2 : 2] = y_even
        y[..., 1 : self.pair_dim * 2 : 2] = y_odd
        return y
