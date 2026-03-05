import torch
import torch.nn as nn

from .common import BitLinear, get_norm


class ODEFunc(nn.Module):
    """
    La funcion derivada dx/dt modelada como Self-Attention.
    Usa BitLinear internamente.
    """

    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.qkv = proj_cls(self.dim, self.dim * 3, bias=False)
        self.out_proj = proj_cls(self.dim, self.dim, bias=False)
        self.norm = get_norm(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, t, x):
        bsz, seq_len, dim = x.shape

        h = self.norm(x)
        qkv = self.qkv(h).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(bsz, seq_len, dim)
        out = self.out_proj(out)
        return out


class ODEAttentionBlock(nn.Module):
    """
    Resuelve: z(1) = z(0) + integral_0^1 f(z(t), t) dt
    Usa un solver RK4 manual para evitar dependencias externas y overhead.
    """

    def __init__(self, config):
        super().__init__()
        self.ode_func = ODEFunc(config)
        self.solver = config.ode_solver
        self.steps = config.ode_steps

    def forward(self, x):
        dt = 1.0 / self.steps
        t = 0.0
        z = x

        for _ in range(self.steps):
            if self.solver == "euler":
                dz = self.ode_func(t, z)
                z = z + dz * dt
            elif self.solver == "rk4":
                k1 = self.ode_func(t, z)
                k2 = self.ode_func(t + dt / 2, z + k1 * dt / 2)
                k3 = self.ode_func(t + dt / 2, z + k2 * dt / 2)
                k4 = self.ode_func(t + dt, z + k3 * dt)
                z = z + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6.0)
            t += dt

        return z
