from dataclasses import dataclass

# ==================== CONFIGURATION ====================
@dataclass
class ModelConfig:
    """Configuration matching BERT-Large parameters"""
    vocab_size: int = 50000
    max_seq_length: int = 512
    hidden_size: int = 1024  # BERT-Large: 1024
    num_hidden_layers: int = 24  # BERT-Large: 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4  # For GQA
    intermediate_size: int = 4096
    num_experts: int = 32  # Reduced for memory (original: 64)
    top_k_experts: int = 4
    moe_intermediate_size: int = 1024
    attention_type: str = "mixed"
    window_size: int = 128
    rope_theta: float = 10000.0
    dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    use_kv_cache: bool = True
    
    @property
    def total_params(self) -> int:
        """Approximate parameter count (~340M like BERT-Large)"""
        # Embeddings
        emb_params = self.vocab_size * self.hidden_size
        
        # Transformer layers
        layer_params = 0
        layer_params += 4 * self.hidden_size * self.hidden_size  # Attention
        layer_params += 2 * self.hidden_size * self.moe_intermediate_size * self.num_experts  # MoE
        layer_params += self.num_experts * self.moe_intermediate_size * self.hidden_size
        
        total = emb_params + (layer_params * self.num_hidden_layers)
        return total
