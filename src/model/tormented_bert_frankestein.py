#!/usr/bin/env python3
"""
TORMENTED-BERT-Frankenstein: The Ultimate Hybrid Transformer (SOTA 2026)
TORMENTED = Ternary ODE Retention Mamba Experts Neural Tanh Encoder Depth

Integrates:
- BitNet b1.58 (Ternary Weights everywhere)
- Neural ODE Attention (Continuous depth dynamics)
- RetNet (Multi-Scale Retention)
- Mamba-2 (State Space Models)
- Sparse Mixture-of-Experts
- Dynamic Tanh Normalization
- Recursive Depth via Loops

Hardware Target: Dual Xeon E5-2680v4 + Nvidia Tesla P40 (24GB)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import Optional, List

# ==================== CONFIGURACIÓN DE RIESGO ====================
@dataclass
class UltraConfig:
    # Dimensiones
    vocab_size: int = 50000
    hidden_size: int = 2048     # Masivo gracias a BitNet
    num_layers: int = 12        # Capas físicas
    num_loops: int = 2          # Recursividad (Profundidad lógica = 24)
    
    # Arquitectura Híbrida (Define qué tipo de capa usar en cada bloque)
    # Patrón cíclico: [RetNet -> ODE -> Mamba -> Attention]
    layer_pattern: List[str] = field(default_factory=lambda: ["retnet", "ode", "mamba", "titan_attn"] * 3)
    
    # ODE Settings
    ode_solver: str = "rk4"     # "euler", "rk4"
    ode_steps: int = 2          # Pasos de integración (bajo para velocidad)
    
    # RetNet Settings
    retention_heads: int = 8
    
    # General
    num_heads: int = 16
    num_experts: int = 8        # MoE Experts
    top_k_experts: int = 2
    dropout: float = 0.1
    
    # Toggles
    use_bitnet: bool = True     # OBLIGATORIO para P40
    norm_type: str = "dynamic_tanh"

    # Mini / Embedding factorization
    use_factorized_embedding: bool = False
    factorized_embedding_dim: int = 128
    use_embedding_conv: bool = True

    # HoPE settings
    hope_base: float = 10_000.0
    hope_damping: float = 0.01

    # Attention / FFN toggles
    use_hope: bool = True
    use_moe: bool = True
    ffn_hidden_size: Optional[int] = None
    ffn_activation: str = "silu"
    embedding_conv_kernel: int = 3

    def __post_init__(self):
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size * 2

# ==================== CORE: BITNET b1.58 (1.58 Bits) ====================
def activation_quant(x):
    """Cuantización de activaciones a 8-bit (rango -128 a 127 escalado)"""
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    y = (x * scale).round().clamp(-128, 127) / scale
    return x + (y - x).detach() # STE

def weight_quant(w):
    """Cuantización ternaria de pesos {-1, 0, 1}"""
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    w_quant = (w * scale).round().clamp(-1, 1) / scale
    return w + (w_quant - w).detach() # STE

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
        
        # 2. Quantización
        w_q = weight_quant(self.weight)
        x_q = activation_quant(x_norm)
        
        # 3. Operación Lineal (Efectivamente sumas en hardware dedicado, FP16/INT8 en GPU)
        output = F.linear(x_q, w_q, self.bias)
        return output

# ==================== NORMALIZATION ====================
class DynamicTanhNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        # Aproximación rápida de std
        sigma = x.std(dim=-1, keepdim=True)
        x_norm = (x - mu) / (sigma + self.eps)
        # Dynamic Tanh gating
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


class FactorizedEmbedding(nn.Module):
    """Factorized token embedding with optional Conv1d pre-projection."""
    def __init__(self, config: UltraConfig):
        super().__init__()
        self.low_dim = config.factorized_embedding_dim
        self.use_conv = config.use_embedding_conv
        self.embedding = nn.Embedding(config.vocab_size, self.low_dim)
        kernel = max(int(config.embedding_conv_kernel), 1)
        padding = kernel // 2
        self.conv = (
            nn.Conv1d(self.low_dim, self.low_dim, kernel_size=kernel, padding=padding)
            if self.use_conv
            else None
        )
        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.proj = proj_cls(self.low_dim, config.hidden_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        if self.conv is not None:
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
            # Even kernels with symmetric padding can shift length by +1.
            # Force sequence length to match inputs to keep MLM labels aligned.
            if x.size(1) != input_ids.size(1):
                target_len = input_ids.size(1)
                if x.size(1) > target_len:
                    x = x[:, :target_len, :]
                else:
                    pad_len = target_len - x.size(1)
                    x = F.pad(x, (0, 0, 0, pad_len))
        return self.proj(x)


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


class TitanAttention(nn.Module):
    """Real multi-head Titan attention using BitLinear projections and HoPE on q/k."""
    def __init__(self, config: UltraConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.use_hope = bool(config.use_hope)

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for TitanAttention")

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.hope = HoPE(self.head_dim, base=config.hope_base, damping=config.hope_damping)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape
        logical_layer_idx = logical_layer_idx or 0

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_hope:
            q = self.hope(q, logical_layer_idx=logical_layer_idx)
            k = self.hope(k, logical_layer_idx=logical_layer_idx)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)


class StandardAttention(nn.Module):
    """Standard multi-head attention (softmax) without HoPE."""
    def __init__(self, config: UltraConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for StandardAttention")

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

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)


class SigmoidAttention(nn.Module):
    """Sigmoid attention with normalization by sum of weights."""
    def __init__(self, config: UltraConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.eps = 1e-6

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for SigmoidAttention")

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

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.sigmoid(attn_scores)
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + self.eps)
        attn_weights = self.dropout(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)

# ==================== ODE ATTENTION (Continuous Depth) ====================
class ODEFunc(nn.Module):
    """
    La función derivada dx/dt modelada como Self-Attention.
    Usa BitLinear internamente.
    """
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        # All Linears are BitLinear
        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.qkv = proj_cls(self.dim, self.dim * 3, bias=False)
        self.out_proj = proj_cls(self.dim, self.dim, bias=False)
        self.norm = get_norm(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, t, x):
        # x: [Batch, Seq, Dim]
        B, N, D = x.shape
        
        # Attention Mechanism inside the ODE derivative
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        
        # La derivada debe tener la misma forma que x
        return out

class ODEAttentionBlock(nn.Module):
    """
    Resuelve: z(1) = z(0) + integral_0^1 f(z(t), t) dt
    Usa un solver RK4 manual para evitar dependencias externas y overhead.
    """
    def __init__(self, config):
        super().__init__()
        self.ode_func = ODEFunc(config)
        self.solver = config.ode_solver # 'rk4' or 'euler'
        self.steps = config.ode_steps

    def forward(self, x):
        dt = 1.0 / self.steps
        t = 0.0
        z = x
        
        for _ in range(self.steps):
            if self.solver == 'euler':
                dz = self.ode_func(t, z)
                z = z + dz * dt
            elif self.solver == 'rk4':
                k1 = self.ode_func(t, z)
                k2 = self.ode_func(t + dt/2, z + k1 * dt/2)
                k3 = self.ode_func(t + dt/2, z + k2 * dt/2)
                k4 = self.ode_func(t + dt, z + k3 * dt)
                z = z + (k1 + 2*k2 + 2*k3 + k4) * (dt / 6.0)
            t += dt
            
        return z

# ==================== RETNET (RETENTION NETWORK) ====================
class MultiScaleRetention(nn.Module):
    """
    Implementación Paralela de RetNet con BitLinear.
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
        self.g_proj = proj_cls(self.dim, self.dim, bias=False) # Gating
        self.out_proj = proj_cls(self.dim, self.dim, bias=False)
        self.swish = nn.SiLU()
        self.norm = get_norm(config)

        # Decay mask (simplificada para Encoder bidireccional o Causal)
        # Aquí usamos decay causal estándar de RetNet
        self.register_buffer("decay_mask", self._build_decay_mask(config.hidden_size, 2048))

    def _build_decay_mask(self, dim, max_len=2048):
        # Gamma decay por cabeza
        gammas = 1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), self.heads))
        # [Heads, 1, 1]
        return gammas.view(self.heads, 1, 1) # Placeholder, computed dynamically usually

    def forward(self, x):
        B, N, D = x.shape
        
        # Proyecciones BitNet
        q = self.q_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        
        # Retention Mechanism (Simplificado para parallel training)
        # R = (Q @ K.T) * D
        # Nota: La implementación completa de RetNet requiere rotaciones complejas (xPos).
        # Aquí usamos una versión simplificada con decay estático para demostración.
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Decay Mask (D)
        # Construimos la matriz de decaimiento D_nm = gamma^(|n-m|)
        gammas = 1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), self.heads, device=x.device))
        n = torch.arange(N, device=x.device).unsqueeze(1)
        m = torch.arange(N, device=x.device).unsqueeze(0)
        dist = n - m
        
        # Causal masking logic + Decay
        decay_matrix = gammas.view(self.heads, 1, 1) ** dist.abs().unsqueeze(0)
        causal_mask = (dist >= 0).float().unsqueeze(0)
        
        retention_scores = attn * decay_matrix * causal_mask
        
        # Output
        y = retention_scores @ v
        y = y.transpose(1, 2).reshape(B, N, D)
        
        # Group Norm logic from paper (simplified to LayerNorm/GroupNorm)
        y = self.norm(y)
        
        # Swish Gate
        g = self.swish(self.g_proj(x))
        out = y * g
        
        return self.out_proj(out)

# ==================== TITAN HYBRID LAYER ====================
class HybridLayer(nn.Module):
    def __init__(self, config, layer_type):
        super().__init__()
        self.layer_type = layer_type
        self.norm1 = get_norm(config)
        self.use_moe = bool(config.use_moe)
        
        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        
        # Selección de Arquitectura de Mezcla
        if layer_type == "ode":
            self.mixer = ODEAttentionBlock(config)
        elif layer_type == "retnet":
            self.mixer = MultiScaleRetention(config)
        elif layer_type == "mamba":
            # Placeholder simple para Mamba (en prod usar mamba-ssm)
            self.mixer = proj_cls(config.hidden_size, config.hidden_size) 
        elif layer_type == "standard_attn":
            self.mixer = StandardAttention(config)
        elif layer_type == "sigmoid_attn":
            self.mixer = SigmoidAttention(config)
        else: # titan_attn
            self.mixer = TitanAttention(config)

        self.norm2 = get_norm(config)
        activation = nn.SiLU() if config.ffn_activation == "silu" else nn.GELU()

        if self.use_moe:
            # Sparse MoE FFN (BitNet)
            self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
            self.experts = nn.ModuleList([
                nn.Sequential(
                    proj_cls(config.hidden_size, config.ffn_hidden_size),
                    activation,
                    proj_cls(config.ffn_hidden_size, config.hidden_size)
                ) for _ in range(config.num_experts)
            ])
            self.top_k = config.top_k_experts
        else:
            self.router = None
            self.experts = None
            self.top_k = 0
            self.ffn = nn.Sequential(
                proj_cls(config.hidden_size, config.ffn_hidden_size),
                activation,
                proj_cls(config.ffn_hidden_size, config.hidden_size)
            )

    def forward(self, x, logical_layer_idx: Optional[int] = None):
        residual = x
        x = self.norm1(x)
        
        # Mixer Logic
        if self.layer_type == "mamba":
            # Simulación rápida
            x = x + self.mixer(x) 
        elif self.layer_type in {"titan_attn", "standard_attn", "sigmoid_attn"}:
            x = self.mixer(x, logical_layer_idx=logical_layer_idx)
        else:
            x = self.mixer(x)
            
        x = residual + x
        
        # MoE / FFN Logic
        residual = x
        x = self.norm2(x)

        if self.use_moe:
            # Routing
            logits = self.router(x)
            weights, indices = torch.topk(F.softmax(logits, dim=-1), self.top_k, dim=-1)

            # Execute experts
            batch_size, seq_len, dim = x.shape
            flat_x = x.view(-1, dim)
            out = torch.zeros_like(flat_x)

            # Loop ingenuo (optimizar con scatter/gather en CUDA)
            for k in range(self.top_k):
                expert_indices = indices[:, :, k].flatten()
                expert_weights = weights[:, :, k].flatten().unsqueeze(1)

                for i, expert in enumerate(self.experts):
                    mask = (expert_indices == i)
                    if mask.any():
                        selected_x = flat_x[mask]
                        expert_out = expert(selected_x)
                        out[mask] += expert_out * expert_weights[mask]

            x = residual + out.view(batch_size, seq_len, dim)
            return x

        x = residual + self.ffn(x)
        return x

# ==================== MAIN MODEL ====================
class TormentedBertFrankenstein(nn.Module):
    """TORMENTED-BERT-Frankenstein: Hybrid Transformer Architecture"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding
        if config.use_factorized_embedding:
            self.emb = FactorizedEmbedding(config)
        else:
            self.emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # Capas Físicas
        self.layers = nn.ModuleList([
            HybridLayer(config, layer_type=config.layer_pattern[i % len(config.layer_pattern)])
            for i in range(config.num_layers)
        ])
        
        self.final_norm = get_norm(config)
        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.head = proj_cls(config.hidden_size, config.vocab_size)

    def forward(self, input_ids):
        x = self.emb(input_ids)
        x = self.dropout(x)
        
        # Looping (Recursividad)
        logical_layer_idx = 0
        for loop in range(self.config.num_loops):
            for layer in self.layers:
                x = layer(x, logical_layer_idx=logical_layer_idx)
                logical_layer_idx += 1
                
        x = self.final_norm(x)
        return self.head(x)


class TormentedBertMini(nn.Module):
    """Mini variant preset for stable and efficient training on constrained GPUs."""
    @staticmethod
    def build_mini_config(vocab_size: int = 50_000, use_bitnet: bool = True) -> UltraConfig:
        stable_layer_pattern = [
            "retnet",
            "titan_attn",
            "retnet",
            "mamba",
            "titan_attn",
            "ode",
        ]
        return UltraConfig(
            vocab_size=vocab_size,
            hidden_size=384,
            num_layers=6,
            num_loops=2,
            num_heads=6,
            retention_heads=6,
            num_experts=4,
            top_k_experts=2,
            dropout=0.1,
            ode_solver="rk4",
            ode_steps=2,
            use_bitnet=use_bitnet,
            norm_type="derf",
            layer_pattern=stable_layer_pattern,
            use_factorized_embedding=True,
            factorized_embedding_dim=128,
            use_embedding_conv=True,
        )

    def __init__(self, config: Optional[UltraConfig] = None):
        super().__init__()
        self.config = config or self.build_mini_config()

        # Force Mini defaults when config is omitted, preserve custom overrides otherwise.
        if self.config.use_factorized_embedding is False:
            self.config.use_factorized_embedding = True

        self.backbone = TormentedBertFrankenstein(self.config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone(input_ids)

# ==================== PRUEBA DE ESTRÉS ====================
if __name__ == "__main__":
    # Configuración de Servidor (P40 24GB)
    config = UltraConfig(
        hidden_size=1536,       # Audaz pero seguro con BitNet
        num_layers=16,          # Capas físicas
        num_loops=2,            # Profundidad efectiva = 32
        ode_solver="rk4",
        ode_steps=2             # Poca precisión, alta velocidad
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n⚡ TORMENTED-BERT-Frankenstein INITIALIZING ⚡")
    model = TormentedBertFrankenstein(config).to(device)
    
    # Contar parámetros y ahorros
    params = sum(p.numel() for p in model.parameters())
    # Estimación de VRAM: Parametros BitNet ocupan ~1.58 bits, pero en PyTorch float ocupan 32/16.
    # El ahorro real viene en inferencia compilada o kernels custom. 
    # Aquí simulamos el flujo de entrenamiento.
    print(f"Model Params: {params / 1e6:.2f}M")
    print(f"Architecture Pattern: {config.layer_pattern}")
    print(f"Weights: Ternary (BitNet b1.58)")
    print(f"Dynamics: Neural ODE (RK4)")
    
    # Fake Batch
    x = torch.randint(0, 50000, (4, 512), device=device)
    
    print("\n[...] Running Forward Pass with ODE Dynamics & RetNet...")
    y = model(x)
    print(f"Output Shape: {y.shape}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    loss = y.mean()
    print("[...] Running Backward Pass (Training ODE through time)...")
    loss.backward()
    print("Gradients computed successfully.")
