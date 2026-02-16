#!/usr/bin/env python3
"""
Model Deployment Pipeline for TORMENTED-BERT-Frankenstein
Converts trained checkpoints to optimized, quantized deployable format.

Usage:
    python deploy.py --checkpoint path/to/checkpoint.pt --output deployed_model/
"""

import torch
import argparse
import logging
from pathlib import Path
import json
from typing import Optional
import sys

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from model.tormented_bert_frankestein import TormentedBertFrankenstein, UltraConfig
from deploy.quantization import (
    save_quantized_checkpoint,
    load_quantized_checkpoint,
    estimate_model_size,
    BitNetQuantizer
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelDeployer:
    """
    Handles conversion of trained models to deployment-ready format.
    """
    
    def __init__(self, config: UltraConfig):
        self.config = config
        self.model = None
        
    def load_training_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model from training checkpoint.
        
        Args:
            checkpoint_path: Path to training checkpoint
        """
        logger.info(f"Loading training checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract config if saved with checkpoint
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            # Update config with saved values
            for key, value in saved_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            logger.info("Loaded config from checkpoint")
        
        # Initialize model
        self.model = TormentedBertFrankenstein(self.config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume checkpoint is state dict
            self.model.load_state_dict(checkpoint)
        
        logger.info("Model loaded successfully")
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {total_params / 1e6:.2f}M")
        
    def convert_to_deployment(
        self,
        output_dir: str,
        save_format: str = 'quantized'
    ) -> None:
        """
        Convert model to deployment format.
        
        Args:
            output_dir: Directory to save deployment artifacts
            save_format: 'quantized' or 'standard'
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Converting model to deployment format: {save_format}")
        
        # Set model to eval mode
        self.model.eval()
        
        # Save config
        config_path = output_path / "config.json"
        config_dict = self.config.__dict__.copy()
        # Convert non-serializable types
        if 'layer_pattern' in config_dict:
            config_dict['layer_pattern'] = list(config_dict['layer_pattern'])
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Config saved to {config_path}")
        
        # Save model
        if save_format == 'quantized':
            model_path = output_path / "model_quantized.pt"
            
            # Prepare additional data
            additional_data = {
                'config': config_dict,
                'vocab_size': self.config.vocab_size,
                'hidden_size': self.config.hidden_size,
            }
            
            save_quantized_checkpoint(
                self.model,
                str(model_path),
                additional_data=additional_data
            )
            
            # Print size comparison
            sizes = estimate_model_size(self.model)
            logger.info("Model size estimates:")
            for format_name, size_mb in sizes.items():
                logger.info(f"  {format_name}: {size_mb:.2f} MB")
                
        else:  # standard format
            model_path = output_path / "model.pt"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': config_dict,
            }, model_path)
            logger.info(f"Standard model saved to {model_path}")
        
        # Save deployment info
        info_path = output_path / "deployment_info.json"
        info = {
            'format': save_format,
            'model_class': 'TormentedBertFrankenstein',
            'config_file': 'config.json',
            'model_file': model_path.name,
            'quantization': 'BitNet b1.58 (Ternary weights)',
            'parameters': sum(p.numel() for p in self.model.parameters()),
        }
        
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"✅ Deployment artifacts saved to {output_path}")
        logger.info(f"   - config.json")
        logger.info(f"   - {model_path.name}")
        logger.info(f"   - deployment_info.json")
    
    def validate_deployment(self, deployment_dir: str) -> bool:
        """
        Validate that deployed model can be loaded and used.
        
        Args:
            deployment_dir: Directory containing deployment artifacts
            
        Returns:
            True if validation successful
        """
        logger.info(f"Validating deployment: {deployment_dir}")
        
        deployment_path = Path(deployment_dir)
        
        try:
            # Load config
            config_path = deployment_path / "config.json"
            with open(config_path) as f:
                config_dict = json.load(f)
            
            # Create config object
            config = UltraConfig(**config_dict)
            
            # Initialize model
            model = TormentedBertFrankenstein(config)
            
            # Load weights
            model_files = list(deployment_path.glob("model*.pt"))
            if not model_files:
                logger.error("No model file found")
                return False
            
            model_path = model_files[0]
            
            if 'quantized' in model_path.name:
                load_quantized_checkpoint(str(model_path), model)
            else:
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
            
            # Test forward pass
            logger.info("Testing forward pass...")
            model.eval()
            with torch.no_grad():
                test_input = torch.randint(0, config.vocab_size, (2, 64))
                output = model(test_input)
                logger.info(f"Output shape: {output.shape}")
            
            logger.info("✅ Validation successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Deploy TORMENTED-BERT model for production'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to training checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for deployment artifacts'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['quantized', 'standard'],
        default='quantized',
        help='Deployment format (default: quantized)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate deployment after conversion'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Optional: Path to config JSON (if not in checkpoint)'
    )
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        config = UltraConfig(**config_dict)
    else:
        # Use default config
        config = UltraConfig()
    
    # Create deployer
    deployer = ModelDeployer(config)
    
    # Load training checkpoint
    deployer.load_training_checkpoint(args.checkpoint)
    
    # Convert to deployment format
    deployer.convert_to_deployment(args.output, save_format=args.format)
    
    # Validate if requested
    if args.validate:
        success = deployer.validate_deployment(args.output)
        if not success:
            logger.error("Deployment validation failed")
            return 1
    
    logger.info("🎉 Deployment complete!")
    return 0


if __name__ == "__main__":
    exit(main())
