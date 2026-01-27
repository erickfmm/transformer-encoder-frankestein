import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
import logging

from tokenizer.spm_spa_redpajama35 import SpanishSPMTokenizer
from model.v1.configuration import ModelConfig

# ==================== MODEL COMPONENTS ====================
def dynamic_tanh_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Dynamic Tanh Normalization (Yann LeCun)"""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    x_normalized = (x - mean) / (std + eps)
    
    # Learnable parameters
    if not hasattr(dynamic_tanh_norm, 'params_initialized'):
        dynamic_tanh_norm.alpha = nn.Parameter(torch.ones(1, 1, x.size(-1)))
        dynamic_tanh_norm.beta = nn.Parameter(torch.zeros(1, 1, x.size(-1)))
        dynamic_tanh_norm.params_initialized = True
    
    x_scaled = x_normalized * dynamic_tanh_norm.alpha + dynamic_tanh_norm.beta
    return torch.tanh(x_scaled)

class RotaryPositionEmbedding(nn.Module):
    """RoPE (Rotary Position Embedding)"""
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        
    def forward(self, x: torch.Tensor, seq_len: int):
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        dim_indices = torch.arange(self.dim // 2, dtype=torch.float, device=device)
        freqs = 1.0 / (self.theta ** (2 * dim_indices / self.dim))
        angles = position * freqs
        
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1)
        return rotated_x.flatten(-2)

class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(dim, dim * 2)
        self.v = nn.Linear(dim, dim)
        self.activation = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w(x)
        x, gate = x.chunk(2, dim=-1)
        return self.v(x * self.activation(gate))

class MixedAttention(nn.Module):
    """Attention with GQA, Latent, and Normal types"""
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.layer_idx = layer_idx
        
        # Determine attention type for this layer
        if layer_idx % 3 == 0:
            self.attention_type = "gqa"
            self.num_kv_heads = config.num_key_value_heads
        elif layer_idx % 3 == 1:
            self.attention_type = "latent"
        else:
            self.attention_type = "normal"
        
        # Query projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Key-Value projections
        if self.attention_type == "gqa":
            self.kv_proj = nn.Linear(config.hidden_size, 
                                   2 * self.num_kv_heads * self.head_dim)
        else:
            self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
            self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.rope = RotaryPositionEmbedding(self.head_dim, config.rope_theta)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project queries
        queries = self.q_proj(hidden_states)
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)
        queries = self.rope(queries, seq_len)
        
        # Project keys and values
        if self.attention_type == "gqa":
            kv = self.kv_proj(hidden_states)
            kv = kv.view(batch_size, seq_len, 2, self.num_kv_heads, self.head_dim)
            keys, values = kv.unbind(2)
            keys = keys.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
            values = values.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        else:
            keys = self.k_proj(hidden_states)
            values = self.v_proj(hidden_states)
            keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim)
            values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        keys = self.rope(keys, seq_len)
        
        # Attention computation
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Apply window mask for some layers
        if self.layer_idx % 4 < 2:
            window_mask = torch.ones(seq_len, seq_len, device=scores.device)
            for i in range(seq_len):
                start = max(0, i - 64)
                end = min(seq_len, i + 64)
                window_mask[i, :start] = 0
                window_mask[i, end:] = 0
            scores = scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        return self.o_proj(attn_output)

class SparseMoE(nn.Module):
    """Sparse Mixture of Experts layer"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.hidden_size = config.hidden_size
        self.expert_size = config.moe_intermediate_size
        
        # Router
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
        # Experts (mix of linear and standard attention)
        self.experts = nn.ModuleList([
            self._create_expert(i, config) for i in range(config.num_experts)
        ])
        
    def _create_expert(self, expert_idx: int, config: ModelConfig) -> nn.Module:
        """Create expert with different attention types"""
        if expert_idx % 4 == 0:
            # Linear attention expert
            return LinearAttentionExpert(config)
        else:
            # Standard attention expert
            return StandardAttentionExpert(config)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Routing logic
        router_logits = self.router(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Initialize output
        final_output = torch.zeros_like(hidden_states)
        
        # Process through experts
        for k in range(self.top_k):
            expert_mask = torch.zeros(batch_size, seq_len, self.num_experts, 
                                    device=hidden_states.device)
            expert_mask.scatter_(-1, top_k_indices[..., k:k+1], 1)
            
            expert_outputs = []
            for exp_idx in range(self.num_experts):
                mask = expert_mask[..., exp_idx].bool()
                if mask.any():
                    expert_input = hidden_states[mask].view(-1, self.hidden_size)
                    expert_out = self.experts[exp_idx](expert_input)
                    expert_outputs.append((expert_out, mask))
            
            # Combine expert outputs
            combined = torch.zeros(batch_size * seq_len, self.hidden_size, 
                                 device=hidden_states.device)
            for expert_out, mask in expert_outputs:
                combined[mask.flatten()] = expert_out
            
            final_output += (combined.view_as(hidden_states) * 
                          top_k_weights[..., k:k+1])
        
        return final_output

class LinearAttentionExpert(nn.Module):
    """Expert with linear attention for efficiency"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.expert_size = config.moe_intermediate_size
        
        self.qkv = nn.Linear(config.hidden_size, 3 * config.moe_intermediate_size)
        self.projection = nn.Linear(config.moe_intermediate_size, config.hidden_size)
        self.swiglu = SwiGLU(config.moe_intermediate_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Linear attention approximation
        k_t = k.transpose(-2, -1)
        kv = torch.matmul(k_t, v)
        z = 1.0 / (torch.einsum('...nd,...nd->...n', q, k.sum(dim=1).unsqueeze(1)) + 1e-6)
        output = torch.matmul(q, kv) * z.unsqueeze(-1)
        
        output = self.swiglu(output)
        return self.projection(output)

class StandardAttentionExpert(nn.Module):
    """Expert with standard self-attention"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.expert_size = config.moe_intermediate_size
        self.num_heads = 4
        self.head_dim = config.moe_intermediate_size // self.num_heads
        
        self.qkv = nn.Linear(config.hidden_size, 3 * config.moe_intermediate_size)
        self.projection = nn.Linear(config.moe_intermediate_size, config.hidden_size)
        self.swiglu = SwiGLU(config.moe_intermediate_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        
        output = self.swiglu(attn_output)
        return self.projection(output)

class TransformerLayer(nn.Module):
    """Single transformer layer with MoE"""
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.attention = MixedAttention(config, layer_idx)
        self.moe = SparseMoE(config)
        
        # Dynamic Tanh normalization
        self.input_norm = lambda x: dynamic_tanh_norm(x)
        self.post_attention_norm = lambda x: dynamic_tanh_norm(x)
        self.post_moe_norm = lambda x: dynamic_tanh_norm(x)
        
        self.dropout = nn.Dropout(config.dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        attn_output = self.attention(hidden_states, attention_mask)
        hidden_states = residual + self.dropout(attn_output)
        
        # MoE layer
        residual = hidden_states
        hidden_states = self.post_attention_norm(hidden_states)
        moe_output = self.moe(hidden_states)
        hidden_states = residual + self.dropout(moe_output)
        hidden_states = self.post_moe_norm(hidden_states)
        
        return hidden_states

class SpanishMoEBERT(nn.Module):
    """Complete Spanish MoE-BERT model"""
    
    def __init__(self, config: ModelConfig, tokenizer: SpanishSPMTokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        # Embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_seq_length, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Output layers
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        
        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size, config.vocab_size)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logging.info(f"Model initialized with ~{config.total_params:,} parameters")
        logging.info(f"Storage estimate: {self._estimate_storage():.2f}GB")
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.layer_norm_eps)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.layer_norm_eps)
    
    def _estimate_storage(self) -> float:
        """Estimate model storage in GB"""
        param_size = sum(p.numel() for p in self.parameters())
        buffer_size = sum(b.numel() for b in self.buffers())
        total_size = param_size + buffer_size
        return total_size * 4 / 1024**3  # FP32 = 4 bytes per parameter
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, 
                                  device=input_ids.device).unsqueeze(0)
        
        # Get embeddings
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = word_embeddings + position_embeddings
        
        # Apply attention mask
        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        
        # Pass through layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Pooled output for sentence embeddings
        pooled_output = self.pooler(hidden_states[:, 0])
        
        # MLM predictions
        mlm_logits = self.mlm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(mlm_logits.view(-1, self.config.vocab_size), 
                          labels.view(-1))
        
        return {
            "loss": loss,
            "logits": mlm_logits,
            "hidden_states": hidden_states,
            "pooled_output": pooled_output,
        }
