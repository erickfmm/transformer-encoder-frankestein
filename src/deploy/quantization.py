#!/usr/bin/env python3
"""
BitNet Quantization and Compression Utilities for TORMENTED-BERT
Provides additional compression beyond base BitNet b1.58 implementation.

Features:
- Ternary weight packing (1.58 bits per weight)
- INT8 activation quantization
- Model size reduction (3-4x smaller checkpoints)
- Fast serialization/deserialization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BitNetQuantizer:
    """
    Quantization manager for BitNet models.
    Handles packing/unpacking of ternary weights and activation quantization.
    """
    
    def __init__(self):
        self.quantization_config = {
            'weight_bits': 1.58,  # Ternary: {-1, 0, 1}
            'activation_bits': 8,   # INT8 for activations
        }
    
    @staticmethod
    def quantize_ternary_weights(weight: torch.Tensor) -> Tuple[np.ndarray, float]:
        """
        Quantize weights to ternary {-1, 0, 1} and pack for storage.
        
        Args:
            weight: Float tensor to quantize
            
        Returns:
            packed_weights: Packed ternary weights (2 bits per weight, stored efficiently)
            scale: Scaling factor for dequantization
        """
        # Calculate scale
        scale = weight.abs().mean().item()
        if scale < 1e-5:
            scale = 1.0
            
        # Quantize to {-1, 0, 1}
        w_scaled = weight / scale
        w_ternary = torch.round(w_scaled).clamp(-1, 1).cpu().numpy()
        
        # Pack ternary values: -1 -> 0b00, 0 -> 0b01, 1 -> 0b10
        # This allows 4 weights per byte (2 bits each)
        packed = BitNetQuantizer._pack_ternary(w_ternary.flatten())
        
        return packed, scale
    
    @staticmethod
    def _pack_ternary(ternary_array: np.ndarray) -> np.ndarray:
        """
        Pack ternary values into 2 bits each (4 values per byte).
        Mapping: -1 -> 0, 0 -> 1, 1 -> 2
        """
        # Convert -1, 0, 1 to 0, 1, 2
        packed_values = (ternary_array + 1).astype(np.uint8)
        
        # Pack 4 values into each byte
        n = len(packed_values)
        n_bytes = (n + 3) // 4  # Round up
        
        packed = np.zeros(n_bytes, dtype=np.uint8)
        for i in range(n):
            byte_idx = i // 4
            bit_offset = (i % 4) * 2
            packed[byte_idx] |= (packed_values[i] & 0b11) << bit_offset
            
        return packed
    
    @staticmethod
    def _unpack_ternary(packed: np.ndarray, original_size: int) -> np.ndarray:
        """
        Unpack ternary values from 2-bit packed format.
        """
        unpacked = np.zeros(original_size, dtype=np.int8)
        
        for i in range(original_size):
            byte_idx = i // 4
            bit_offset = (i % 4) * 2
            value = (packed[byte_idx] >> bit_offset) & 0b11
            unpacked[i] = value - 1  # Convert back to -1, 0, 1
            
        return unpacked
    
    @staticmethod
    def dequantize_ternary_weights(
        packed: np.ndarray,
        scale: float,
        original_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Dequantize packed ternary weights back to float tensor.
        
        Args:
            packed: Packed ternary weights
            scale: Scaling factor
            original_shape: Original tensor shape
            
        Returns:
            Dequantized float tensor
        """
        n_elements = np.prod(original_shape)
        ternary = BitNetQuantizer._unpack_ternary(packed, n_elements)
        weights = torch.from_numpy(ternary).float() * scale
        return weights.reshape(original_shape)
    
    def quantize_model_weights(self, model: nn.Module) -> Dict[str, Any]:
        """
        Quantize all BitLinear weights in the model to ternary format.
        
        Args:
            model: PyTorch model to quantize
            
        Returns:
            Dictionary with quantized weights and metadata
        """
        quantized_state = {
            'weights': {},
            'scales': {},
            'shapes': {},
            'config': self.quantization_config
        }
        
        total_original_size = 0
        total_compressed_size = 0
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Quantize weight
                packed, scale = self.quantize_ternary_weights(param.data)
                
                quantized_state['weights'][name] = packed
                quantized_state['scales'][name] = scale
                quantized_state['shapes'][name] = tuple(param.shape)
                
                # Track compression
                original_size = param.numel() * 4  # FP32 = 4 bytes
                compressed_size = len(packed)
                
                total_original_size += original_size
                total_compressed_size += compressed_size
                
                logger.debug(f"{name}: {original_size / 1024:.2f}KB -> {compressed_size / 1024:.2f}KB")
            else:
                # Keep biases and other parameters as-is (if any)
                quantized_state['weights'][name] = param.data.cpu().numpy()
        
        compression_ratio = total_original_size / total_compressed_size
        logger.info(f"Model compressed: {total_original_size / (1024**2):.2f}MB -> "
                   f"{total_compressed_size / (1024**2):.2f}MB "
                   f"(Compression ratio: {compression_ratio:.2f}x)")
        
        return quantized_state
    
    def dequantize_model_weights(
        self,
        quantized_state: Dict[str, Any],
        model: nn.Module
    ) -> None:
        """
        Load quantized weights back into model.
        
        Args:
            quantized_state: Dictionary with quantized weights
            model: PyTorch model to load weights into
        """
        state_dict = {}
        
        for name in quantized_state['weights'].keys():
            if name in quantized_state['scales']:
                # Dequantize ternary weight
                packed = quantized_state['weights'][name]
                scale = quantized_state['scales'][name]
                shape = quantized_state['shapes'][name]
                
                weight = self.dequantize_ternary_weights(packed, scale, shape)
                state_dict[name] = weight
            else:
                # Load non-quantized parameter
                state_dict[name] = torch.from_numpy(quantized_state['weights'][name])
        
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded {len(state_dict)} quantized parameters into model")


class ActivationQuantizer:
    """
    Runtime activation quantization for inference optimization.
    """
    
    @staticmethod
    def quantize_activation_int8(x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Quantize activations to INT8 for efficient computation.
        
        Args:
            x: Activation tensor
            
        Returns:
            Quantized tensor and scale factor
        """
        scale = 127.0 / x.abs().max().clamp(min=1e-5)
        x_q = (x * scale).round().clamp(-128, 127)
        return x_q, scale.item()
    
    @staticmethod
    def dequantize_activation_int8(
        x_q: torch.Tensor,
        scale: float
    ) -> torch.Tensor:
        """
        Dequantize INT8 activations back to float.
        """
        return x_q.float() / scale


def save_quantized_checkpoint(
    model: nn.Module,
    save_path: str,
    additional_data: Dict[str, Any] = None
) -> None:
    """
    Save model as quantized checkpoint.
    
    Args:
        model: PyTorch model to save
        save_path: Path to save checkpoint
        additional_data: Additional metadata to save (config, tokenizer info, etc.)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Quantizing and saving model to {save_path}")
    
    quantizer = BitNetQuantizer()
    quantized_state = quantizer.quantize_model_weights(model)
    
    # Add metadata
    checkpoint = {
        'quantized_weights': quantized_state,
        'model_class': model.__class__.__name__,
    }
    
    if additional_data:
        checkpoint.update(additional_data)
    
    # Save with compression
    torch.save(checkpoint, save_path, pickle_protocol=4)
    
    file_size_mb = save_path.stat().st_size / (1024**2)
    logger.info(f"Checkpoint saved: {file_size_mb:.2f}MB")


def load_quantized_checkpoint(
    load_path: str,
    model: nn.Module
) -> Dict[str, Any]:
    """
    Load quantized checkpoint into model.
    
    Args:
        load_path: Path to quantized checkpoint
        model: Model to load weights into
        
    Returns:
        Dictionary with additional metadata from checkpoint
    """
    logger.info(f"Loading quantized checkpoint from {load_path}")
    
    checkpoint = torch.load(load_path, map_location='cpu')
    quantized_state = checkpoint['quantized_weights']
    
    quantizer = BitNetQuantizer()
    quantizer.dequantize_model_weights(quantized_state, model)
    
    logger.info("Model weights loaded successfully")
    
    # Return metadata
    metadata = {k: v for k, v in checkpoint.items() if k != 'quantized_weights'}
    return metadata


def estimate_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Estimate model size in different formats.
    
    Returns:
        Dictionary with size estimates in MB
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    sizes = {
        'fp32_mb': total_params * 4 / (1024**2),
        'fp16_mb': total_params * 2 / (1024**2),
        'bitnet_158_mb': total_params * 1.58 / 8 / (1024**2),  # 1.58 bits per param
    }
    
    return sizes


if __name__ == "__main__":
    # Test quantization
    logger.info("Testing BitNet quantization...")
    
    # Create dummy weight tensor
    test_weight = torch.randn(1024, 1024)
    logger.info(f"Original weight size: {test_weight.numel() * 4 / 1024:.2f}KB")
    
    # Quantize
    quantizer = BitNetQuantizer()
    packed, scale = quantizer.quantize_ternary_weights(test_weight)
    logger.info(f"Compressed size: {len(packed) / 1024:.2f}KB")
    
    # Dequantize
    reconstructed = quantizer.dequantize_ternary_weights(
        packed, scale, test_weight.shape
    )
    
    # Check reconstruction
    error = (test_weight - reconstructed).abs().mean()
    logger.info(f"Reconstruction error: {error:.6f}")
    logger.info(f"Compression ratio: {test_weight.numel() * 4 / len(packed):.2f}x")
