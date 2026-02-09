"""
TORMENTED-BERT v2 Deployment Package
BitNet quantization and inference pipeline for production deployment.
"""

from .quantization import (
    BitNetQuantizer,
    ActivationQuantizer,
    save_quantized_checkpoint,
    load_quantized_checkpoint,
    estimate_model_size
)

from .deploy import ModelDeployer

from .inference import TormentedBertInference

__version__ = "2.0.0"
__all__ = [
    'BitNetQuantizer',
    'ActivationQuantizer',
    'save_quantized_checkpoint',
    'load_quantized_checkpoint',
    'estimate_model_size',
    'ModelDeployer',
    'TormentedBertInference',
]
