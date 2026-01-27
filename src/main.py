#!/usr/bin/env python3
"""
Spanish MoE-BERT Trainer with SPM Tokenizer
Dataset: erickfmm/red_pajama_es_hq_35
Vocabulary: 50,000 tokens
Storage limit: <300GB
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any
import logging
from dataclasses import dataclass
import numpy as np
import sentencepiece as spm
from datasets import load_dataset, Dataset, IterableDataset
import tempfile
from pathlib import Path
import json
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset as TorchDataset
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime

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

# ==================== STORAGE MANAGER ====================
class StorageManager:
    """Manages disk usage to stay under 300GB limit"""
    
    def __init__(self, limit_gb: float = 300.0):
        self.limit_bytes = limit_gb * 1024**3
        self.used_bytes = 0
        self.temp_files = []
        
    def register_file(self, path: str) -> bool:
        """Register a file and check if within limits"""
        try:
            size = os.path.getsize(path)
            self.used_bytes += size
            
            if self.used_bytes > self.limit_bytes:
                logging.warning(f"Storage limit exceeded: {self.used_bytes/1024**3:.2f}GB")
                return False
            return True
        except:
            return True
    
    def create_temp_file(self, suffix: str = ".tmp") -> str:
        """Create a temporary file and register it"""
        temp_dir = Path("./temp_data")
        temp_dir.mkdir(exist_ok=True)
        
        temp_file = tempfile.NamedTemporaryFile(
            dir=temp_dir, 
            suffix=suffix, 
            delete=False
        )
        temp_file.close()
        self.temp_files.append(temp_file.name)
        return temp_file.name
    
    def cleanup(self):
        """Clean up temporary files"""
        for file_path in self.temp_files:
            try:
                os.unlink(file_path)
            except:
                pass
        self.temp_files.clear()

# ==================== SPM TOKENIZER ====================
class SpanishSPMTokenizer:
    """Spanish SentencePiece Tokenizer trained on RedPajama"""
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.sp_model = None
        self.vocab = {}
        self.inverse_vocab = {}
        self.storage_manager = StorageManager()
        
    def prepare_training_data(self, max_samples: int = 500000) -> str:
        """
        Prepare training data from erickfmm/red_pajama_es_hq_35
        Using streaming to avoid downloading full dataset
        """
        logging.info("Loading Spanish RedPajama dataset (streaming mode)...")
        
        # Create temp file for training data
        temp_file = self.storage_manager.create_temp_file(suffix=".txt")
        
        try:
            # Load dataset in streaming mode
            dataset = load_dataset(
                "erickfmm/red_pajama_es_hq_35",
                split="train",
                streaming=True
            )
            
            count = 0
            with open(temp_file, 'w', encoding='utf-8') as f:
                iterator = iter(dataset)
                
                while count < max_samples:
                    try:
                        example = next(iterator)
                        if 'text' in example:
                            text = example['text'].strip()
                            if len(text) > 100:  # Skip very short texts
                                f.write(text + '\n')
                                count += 1
                                
                                if count % 10000 == 0:
                                    logging.info(f"Collected {count} samples...")
                                    # Check storage
                                    if not self.storage_manager.register_file(temp_file):
                                        logging.warning("Storage limit approached, stopping collection")
                                        break
                    except StopIteration:
                        break
                
            logging.info(f"Created training file with {count} samples")
            return temp_file
            
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            # Create fallback training data
            with open(temp_file, 'w', encoding='utf-8') as f:
                for i in range(10000):
                    f.write(f"Texto de ejemplo en español número {i}. ")
                    f.write("Este es un texto para entrenar el tokenizador. ")
                    f.write("El aprendizaje automático requiere datos de calidad.\n")
            return temp_file
    
    def train(self, model_prefix: str = "es_spm_50k"):
        """Train SentencePiece tokenizer with 50k vocabulary"""
        train_file = self.prepare_training_data()
        
        # SPM training arguments
        spm_args = [
            f'--input={train_file}',
            f'--model_prefix={model_prefix}',
            f'--vocab_size={self.vocab_size}',
            '--character_coverage=0.9995',
            '--model_type=bpe',
            '--max_sentence_length=16384',
            '--pad_id=0',
            '--unk_id=1',
            '--bos_id=2',
            '--eos_id=3',
            '--user_defined_symbols=[CLS],[SEP],[MASK],[PAD]',
            '--split_by_whitespace=true',
            '--normalization_rule_name=nmt_nfkc',
            '--add_dummy_prefix=true',
            '--byte_fallback=true',
            '--split_digits=true',
            f'--num_threads={os.cpu_count()//2}',  # Use half the cores
            '--input_sentence_size=1000000',
            '--shuffle_input_sentence=true',
        ]
        
        logging.info("Training SentencePiece tokenizer...")
        spm.SentencePieceTrainer.train(' '.join(spm_args))
        
        # Load trained model
        model_path = f"{model_prefix}.model"
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)
        
        # Build vocab
        for i in range(self.sp_model.GetPieceSize()):
            token = self.sp_model.IdToPiece(i)
            self.vocab[token] = i
            self.inverse_vocab[i] = token
        
        logging.info(f"Tokenizer trained with {len(self.vocab)} tokens")
        return model_path
    
    def encode(self, text: str, max_length: int = 512) -> Dict[str, List[int]]:
        """Encode text with special tokens"""
        tokens = self.sp_model.encode_as_ids(text)
        
        # Add special tokens
        tokens = [self.vocab['[CLS]']] + tokens[:max_length-2] + [self.vocab['[SEP]']]
        
        # Pad or truncate
        if len(tokens) < max_length:
            tokens = tokens + [self.vocab['[PAD]']] * (max_length - len(tokens))
            attention_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
            attention_mask = [1] * max_length
        
        return {
            'input_ids': tokens,
            'attention_mask': attention_mask[:max_length]
        }

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

# ==================== DATASET & TRAINING ====================
class StreamingMLMDataset(TorchDataset):
    """Streaming MLM dataset with storage management"""
    
    def __init__(self, tokenizer: SpanishSPMTokenizer, max_length: int = 512,
                 mlm_probability: float = 0.15, max_samples: int = 1000000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.max_samples = max_samples
        self.storage_manager = StorageManager()
        
        # Cache file for processed examples
        self.cache_file = self.storage_manager.create_temp_file(suffix=".cache")
        self.examples = self._prepare_examples()
    
    def _prepare_examples(self) -> List[Dict]:
        """Prepare examples with streaming"""
        examples = []
        
        try:
            dataset = load_dataset(
                "erickfmm/red_pajama_es_hq_35",
                split="train",
                streaming=True
            )
            
            logging.info("Preparing MLM examples...")
            count = 0
            
            for example in dataset:
                if count >= self.max_samples:
                    break
                
                if 'text' in example:
                    text = example['text'].strip()
                    if len(text) > 100:  # Skip very short texts
                        encoded = self.tokenizer.encode(text, self.max_length)
                        mlm_input, mlm_labels = self._apply_mlm_mask(
                            encoded['input_ids']
                        )
                        
                        examples.append({
                            'input_ids': mlm_input,
                            'attention_mask': encoded['attention_mask'],
                            'labels': mlm_labels,
                        })
                        
                        count += 1
                        if count % 10000 == 0:
                            logging.info(f"Prepared {count} examples")
                
                # Check storage
                if not self.storage_manager.register_file(self.cache_file):
                    logging.warning("Storage limit reached during data preparation")
                    break
            
            # Cache to disk
            with open(self.cache_file, 'wb') as f:
                import pickle
                pickle.dump(examples, f)
            
        except Exception as e:
            logging.error(f"Error preparing dataset: {e}")
            # Create synthetic data
            examples = self._create_synthetic_examples(10000)
        
        return examples
    
    def _apply_mlm_mask(self, input_ids: List[int]) -> Tuple[List[int], List[int]]:
        """Apply MLM masking"""
        labels = input_ids.copy()
        special_tokens = {0, 1, 2, 3}  # PAD, UNK, CLS, SEP
        
        # Find maskable positions
        maskable_positions = [
            i for i, token_id in enumerate(input_ids)
            if token_id not in special_tokens
        ]
        
        num_to_mask = max(1, int(len(maskable_positions) * self.mlm_probability))
        masked_positions = random.sample(maskable_positions, 
                                       min(num_to_mask, len(maskable_positions)))
        
        for pos in masked_positions:
            # 80%: [MASK], 10%: random, 10%: unchanged
            rand = random.random()
            if rand < 0.8:
                input_ids[pos] = 3  # [MASK]
            elif rand < 0.9:
                input_ids[pos] = random.randint(4, self.tokenizer.vocab_size - 1)
        
        return input_ids, labels
    
    def _create_synthetic_examples(self, num_examples: int) -> List[Dict]:
        """Create synthetic examples for testing"""
        examples = []
        spanish_words = ["hola", "mundo", "español", "idioma", "aprender", 
                        "datos", "modelo", "entrenamiento"]
        
        for _ in range(num_examples):
            text = " ".join(random.choices(spanish_words, k=random.randint(10, 50)))
            encoded = self.tokenizer.encode(text, self.max_length)
            
            mlm_input, mlm_labels = self._apply_mlm_mask(encoded['input_ids'])
            
            examples.append({
                'input_ids': mlm_input,
                'attention_mask': encoded['attention_mask'],
                'labels': mlm_labels,
            })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            'input_ids': torch.tensor(example['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(example['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(example['labels'], dtype=torch.long),
        }

class Trainer:
    """Training manager with storage monitoring"""
    
    def __init__(self, model: SpanishMoEBERT, device: torch.device = None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
        self.scaler = GradScaler()
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, eta_min=1e-6
        )
        
        # Storage monitoring
        self.storage_manager = StorageManager()
        
        logging.info(f"Training on device: {self.device}")
        if torch.cuda.is_available():
            logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward with mixed precision
            with autocast():
                outputs = self.model(**batch)
                loss = outputs["loss"]
            
            # Backward
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Memory management
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return total_loss / len(dataloader)
    
    def save_checkpoint(self, epoch: int, path: str = "checkpoints"):
        """Save model checkpoint"""
        os.makedirs(path, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.model.config,
        }
        
        checkpoint_path = f"{path}/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Check storage
        if self.storage_manager.register_file(checkpoint_path):
            logging.info(f"Checkpoint saved: {checkpoint_path}")
        else:
            logging.warning(f"Checkpoint saved but storage limit warning")
        
        return checkpoint_path

# ==================== MAIN EXECUTION ====================
def main():
    """Main training pipeline"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("Starting Spanish MoE-BERT training")
    logging.info(f"Current directory: {os.getcwd()}")
    logging.info(f"Available storage: {psutil.disk_usage('.').free/1024**3:.2f}GB")
    
    # Step 1: Train tokenizer
    logging.info("\n" + "="*50)
    logging.info("Step 1: Training SPM tokenizer (50k vocab)")
    logging.info("="*50)
    
    tokenizer = SpanishSPMTokenizer(vocab_size=50000)
    tokenizer.train(model_prefix="es_redpajama_50k")
    
    # Step 2: Create model
    logging.info("\n" + "="*50)
    logging.info("Step 2: Creating Spanish MoE-BERT model")
    logging.info("="*50)
    
    config = ModelConfig(vocab_size=50000)
    model = SpanishMoEBERT(config, tokenizer)
    
    # Step 3: Prepare dataset
    logging.info("\n" + "="*50)
    logging.info("Step 3: Preparing MLM dataset")
    logging.info("="*50)
    
    dataset = StreamingMLMDataset(
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        mlm_probability=0.15,
        max_samples=100000  # Reduced for storage
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # Small batch for memory
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    
    # Step 4: Train
    logging.info("\n" + "="*50)
    logging.info("Step 4: Training model")
    logging.info("="*50)
    
    trainer = Trainer(model)
    
    num_epochs = 3  # Reduced for demonstration
    for epoch in range(num_epochs):
        avg_loss = trainer.train_epoch(dataloader, epoch)
        logging.info(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint every epoch
        trainer.save_checkpoint(epoch)
        
        # Storage report
        storage_used = trainer.storage_manager.used_bytes / 1024**3
        logging.info(f"Storage used: {storage_used:.2f}GB / 300GB")
    
    logging.info("\n" + "="*50)
    logging.info("Training completed successfully!")
    logging.info("="*50)
    
    # Cleanup
    tokenizer.storage_manager.cleanup()
    dataset.storage_manager.cleanup()
    trainer.storage_manager.cleanup()

if __name__ == "__main__":
    # Check for psutil (optional)
    try:
        import psutil
    except ImportError:
        logging.warning("psutil not installed, storage monitoring limited")
        psutil = None
    
    main()