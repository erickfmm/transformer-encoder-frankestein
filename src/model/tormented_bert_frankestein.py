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

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention.common import BitLinear, get_norm
from .attention.ode import ODEAttentionBlock
from .attention.retnet import MultiScaleRetention
from .attention.sigmoid import SigmoidAttention
from .attention.standard import StandardAttention
from .attention.titan import TitanAttention


# ==================== CONFIGURACION DE RIESGO ====================
@dataclass
class UltraConfig:
    # Dimensiones
    vocab_size: int = 50000
    hidden_size: int = 2048
    num_layers: int = 12
    num_loops: int = 2

    # Arquitectura Hibrida
    layer_pattern: List[str] = field(default_factory=lambda: ["retnet", "ode", "mamba", "titan_attn"] * 3)

    # ODE Settings
    ode_solver: str = "rk4"
    ode_steps: int = 2

    # RetNet Settings
    retention_heads: int = 8

    # General
    num_heads: int = 16
    num_experts: int = 8
    top_k_experts: int = 2
    dropout: float = 0.1

    # Toggles
    use_bitnet: bool = True
    norm_type: str = "dynamic_tanh"

    # Mini / Embedding factorization
    use_factorized_embedding: bool = False
    factorized_embedding_dim: int = 128
    use_embedding_conv: bool = True

    # HoPE / RoPE settings
    hope_base: float = 10_000.0
    hope_damping: float = 0.01
    rope_base: float = 10_000.0
    rope_scaling: float = 1.0

    # Attention / FFN toggles
    use_hope: bool = True
    positional_encoding: Optional[str] = None
    use_moe: bool = True
    ffn_hidden_size: Optional[int] = None
    ffn_activation: str = "silu"
    embedding_conv_kernel: int = 3

    def __post_init__(self):
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size * 2

        if self.positional_encoding is None:
            self.positional_encoding = "hope" if bool(self.use_hope) else "rope"
        else:
            self.positional_encoding = str(self.positional_encoding).lower()
            if self.positional_encoding not in {"hope", "rope"}:
                raise ValueError("positional_encoding must be one of {'hope', 'rope'}")

        # Keep legacy flag semantically aligned with the selected encoder.
        self.use_hope = self.positional_encoding == "hope"


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


# ==================== TITAN HYBRID LAYER ====================
class HybridLayer(nn.Module):
    def __init__(self, config, layer_type):
        super().__init__()
        self.layer_type = layer_type
        self.norm1 = get_norm(config)
        self.use_moe = bool(config.use_moe)

        proj_cls = BitLinear if config.use_bitnet else nn.Linear

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
        else:  # titan_attn
            self.mixer = TitanAttention(config)

        self.norm2 = get_norm(config)
        activation = nn.SiLU() if config.ffn_activation == "silu" else nn.GELU()

        if self.use_moe:
            self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
            self.experts = nn.ModuleList(
                [
                    nn.Sequential(
                        proj_cls(config.hidden_size, config.ffn_hidden_size),
                        activation,
                        proj_cls(config.ffn_hidden_size, config.hidden_size),
                    )
                    for _ in range(config.num_experts)
                ]
            )
            self.top_k = config.top_k_experts
        else:
            self.router = None
            self.experts = None
            self.top_k = 0
            self.ffn = nn.Sequential(
                proj_cls(config.hidden_size, config.ffn_hidden_size),
                activation,
                proj_cls(config.ffn_hidden_size, config.hidden_size),
            )

    def forward(self, x, logical_layer_idx: Optional[int] = None):
        residual = x
        x = self.norm1(x)

        if self.layer_type == "mamba":
            x = x + self.mixer(x)
        elif self.layer_type in {"titan_attn", "standard_attn", "sigmoid_attn"}:
            x = self.mixer(x, logical_layer_idx=logical_layer_idx)
        else:
            x = self.mixer(x)

        x = residual + x

        residual = x
        x = self.norm2(x)

        if self.use_moe:
            logits = self.router(x)
            weights, indices = torch.topk(F.softmax(logits, dim=-1), self.top_k, dim=-1)

            batch_size, seq_len, dim = x.shape
            flat_x = x.view(-1, dim)
            out = torch.zeros_like(flat_x)

            for k in range(self.top_k):
                expert_indices = indices[:, :, k].flatten()
                expert_weights = weights[:, :, k].flatten().unsqueeze(1)

                for i, expert in enumerate(self.experts):
                    mask = expert_indices == i
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

        if config.use_factorized_embedding:
            self.emb = FactorizedEmbedding(config)
        else:
            self.emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [
                HybridLayer(config, layer_type=config.layer_pattern[i % len(config.layer_pattern)])
                for i in range(config.num_layers)
            ]
        )

        self.final_norm = get_norm(config)
        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.head = proj_cls(config.hidden_size, config.vocab_size)

    def forward(self, input_ids):
        x = self.emb(input_ids)
        x = self.dropout(x)

        logical_layer_idx = 0
        for _ in range(self.config.num_loops):
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

        if self.config.use_factorized_embedding is False:
            self.config.use_factorized_embedding = True

        self.backbone = TormentedBertFrankenstein(self.config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone(input_ids)


# ==================== PRUEBA DE ESTRES ====================
if __name__ == "__main__":
    config = UltraConfig(
        hidden_size=1536,
        num_layers=16,
        num_loops=2,
        ode_solver="rk4",
        ode_steps=2,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n⚡ TORMENTED-BERT-Frankenstein INITIALIZING ⚡")
    model = TormentedBertFrankenstein(config).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model Params: {params / 1e6:.2f}M")
    print(f"Architecture Pattern: {config.layer_pattern}")
    print("Weights: Ternary (BitNet b1.58)")
    print("Dynamics: Neural ODE (RK4)")

    x = torch.randint(0, 50000, (4, 512), device=device)

    print("\n[...] Running Forward Pass with ODE Dynamics & RetNet...")
    y = model(x)
    print(f"Output Shape: {y.shape}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    loss = y.mean()
    print("[...] Running Backward Pass (Training ODE through time)...")
    loss.backward()
    print("Gradients computed successfully.")
