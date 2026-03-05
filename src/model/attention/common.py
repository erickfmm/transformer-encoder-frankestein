import torch
import torch.nn as nn
import torch.nn.functional as F


def activation_quant(x):
    """Cuantizacion de activaciones a 8-bit (rango -128 a 127 escalado)."""
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    y = (x * scale).round().clamp(-128, 127) / scale
    return x + (y - x).detach()  # STE


def weight_quant(w):
    """Cuantizacion ternaria de pesos {-1, 0, 1}."""
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    w_quant = (w * scale).round().clamp(-1, 1) / scale
    return w + (w_quant - w).detach()  # STE


class BitLinear(nn.Linear):
    """
    Capa Lineal BitNet b1.58.
    Reemplazo 'drop-in' para nn.Linear que reduce VRAM en 3x-4x.
    """

    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias)

    def forward(self, x):
        # 1. Norm antes de quant (crucial para estabilidad)
        x_norm = F.layer_norm(x, x.shape[1:])

        # 2. Quantizacion
        w_q = weight_quant(self.weight)
        x_q = activation_quant(x_norm)

        # 3. Operacion Lineal
        output = F.linear(x_q, w_q, self.bias)
        return output


class DynamicTanhNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True)
        x_norm = (x - mu) / (sigma + self.eps)
        return torch.tanh(x_norm * self.alpha + self.beta)


class Derf(nn.Module):
    """Derf normalization: y = gamma * erf(alpha * x + s) + beta."""

    def __init__(self, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.s = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.erf(self.alpha * x + self.s) + self.beta


def get_norm(config):
    if config.norm_type == "dynamic_tanh":
        return DynamicTanhNorm(config.hidden_size)
    if config.norm_type == "derf":
        return Derf(config.hidden_size)
    return nn.LayerNorm(config.hidden_size)
