import torch
import torch.nn as nn


class RoPE(nn.Module):
    """Rotary positional encoding over dim pairs."""

    def __init__(self, head_dim: int, base: float = 10_000.0, scaling: float = 1.0):
        super().__init__()
        self.head_dim = head_dim
        self.pair_dim = head_dim // 2
        self.base = base
        self.scaling = scaling

    def forward(self, x: torch.Tensor, logical_layer_idx: int = 0) -> torch.Tensor:
        # x: [B, H, N, Dh]
        if self.pair_dim == 0:
            return x

        _, _, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        pos = torch.arange(seq_len, device=device, dtype=dtype) * self.scaling
        if self.pair_dim > 1:
            idx = torch.arange(self.pair_dim, device=device, dtype=dtype)
            inv_freq = self.base ** (-idx / (self.pair_dim - 1))
        else:
            inv_freq = torch.ones(1, device=device, dtype=dtype)

        angles = pos[:, None] * inv_freq[None, :]
        sin_term = torch.sin(angles).unsqueeze(0).unsqueeze(0)
        cos_term = torch.cos(angles).unsqueeze(0).unsqueeze(0)

        x_even = x[..., : self.pair_dim * 2 : 2]
        x_odd = x[..., 1 : self.pair_dim * 2 : 2]

        y_even = x_even * cos_term - x_odd * sin_term
        y_odd = x_even * sin_term + x_odd * cos_term

        y = x.clone()
        y[..., : self.pair_dim * 2 : 2] = y_even
        y[..., 1 : self.pair_dim * 2 : 2] = y_odd
        return y
