#!/usr/bin/env python3
"""
SBERT Fine-tuning for TORMENTED-BERT-Frankenstein
Fine-tunes the v2 model on Spanish sentence similarity using STS dataset.
Dataset: erickfmm/agentlans__multilingual-sentences__paired_10_sts
"""

import os
import sys
import torch
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    models,
    losses,
    evaluation,
    InputExample
)
from torch.utils.data import DataLoader
import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from model.v2.tormented_bert_frankestein import TormentedBertFrankenstein, UltraConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TormentedBertSentenceTransformer:
    """Wrapper to adapt TormentedBert for Sentence-BERT training"""
    
    def __init__(
        self,
        model_config: Optional[UltraConfig] = None,
        pretrained_path: Optional[str] = None,
        max_seq_length: int = 512,
        pooling_mode: str = "mean"
    ):
        """
        Initialize SBERT model with TormentedBert base.
        
        Args:
            model_config: UltraConfig for the model (if training from scratch)
            pretrained_path: Path to pretrained checkpoint
            max_seq_length: Maximum sequence length
            pooling_mode: Pooling strategy ("mean", "cls", "max")
        """
        self.max_seq_length = max_seq_length
        self.pooling_mode = pooling_mode
        
        # Initialize or load base model
        if pretrained_path and os.path.exists(pretrained_path):
            logger.info(f"Loading pretrained model from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Extract config if available
            if 'config' in checkpoint:
                config = checkpoint['config']
            else:
                config = model_config or self._get_default_config()
            
            self.base_model = TormentedBertFrankenstein(config)
            
            # Load weights (handle potential key mismatches)
            if 'model_state_dict' in checkpoint:
                self.base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif 'state_dict' in checkpoint:
                self.base_model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.base_model.load_state_dict(checkpoint, strict=False)
        else:
            logger.info("Initializing model from scratch")
            config = model_config or self._get_default_config()
            self.base_model = TormentedBertFrankenstein(config)
        
        self.config = config
        self.model = self._build_sentence_transformer()
    
    def _get_default_config(self) -> UltraConfig:
        """Get default config optimized for P40 24GB"""
        return UltraConfig(
            vocab_size=50000,
            hidden_size=768,        # Reduced for SBERT training
            num_layers=12,
            num_loops=1,            # Less recursion for faster training
            num_heads=12,
            ode_steps=1,            # Minimal ODE steps for speed
            dropout=0.1,
            use_bitnet=True
        )
    
    def _build_sentence_transformer(self) -> SentenceTransformer:
        """Build SentenceTransformer model with custom base"""
        
        # Create a custom transformer model wrapper
        class TormentedBertWrapper(torch.nn.Module):
            def __init__(self, tormented_model, hidden_size):
                super().__init__()
                self.tormented_model = tormented_model
                self.config_keys = ['max_seq_length']
                self.max_seq_length = 512
                
                # Remove language modeling head if exists
                if hasattr(self.tormented_model, 'head'):
                    self.tormented_model.head = torch.nn.Identity()
            
            def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
                """Forward pass compatible with sentence-transformers"""
                input_ids = features['input_ids']
                attention_mask = features.get('attention_mask', None)
                
                # Get embeddings from base model
                # TormentedBert returns [batch, seq, hidden]
                output = self.tormented_model(input_ids)
                
                # Apply attention mask if provided
                if attention_mask is not None:
                    output = output * attention_mask.unsqueeze(-1)
                
                features['token_embeddings'] = output
                features['attention_mask'] = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
                
                return features
        
        # Create wrapper
        word_embedding_model = TormentedBertWrapper(
            self.base_model,
            self.config.hidden_size
        )
        
        # Add pooling layer
        pooling_model = models.Pooling(
            self.config.hidden_size,
            pooling_mode_mean_tokens=(self.pooling_mode == "mean"),
            pooling_mode_cls_token=(self.pooling_mode == "cls"),
            pooling_mode_max_tokens=(self.pooling_mode == "max")
        )
        
        # Add normalization layer (important for cosine similarity)
        normalize = models.Normalize()
        
        # Build sentence transformer
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model, normalize]
        )
        
        return model
    
    def get_model(self) -> SentenceTransformer:
        """Get the SentenceTransformer model"""
        return self.model


class SBERTTrainer:
    """Trainer for SBERT fine-tuning on STS task"""
    
    def __init__(
        self,
        model: SentenceTransformer,
        output_dir: str = "./output/sbert_tormented_v2",
        batch_size: int = 16,
        epochs: int = 4,
        warmup_steps: int = 1000,
        evaluation_steps: int = 5000,
        learning_rate: float = 2e-5,
        use_amp: bool = True
    ):
        """
        Initialize SBERT Trainer.
        
        Args:
            model: SentenceTransformer model
            output_dir: Directory to save checkpoints
            batch_size: Training batch size
            epochs: Number of training epochs
            warmup_steps: Warmup steps for learning rate
            evaluation_steps: Steps between evaluations
            learning_rate: Learning rate
            use_amp: Use automatic mixed precision
        """
        self.model = model
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.evaluation_steps = evaluation_steps
        self.learning_rate = learning_rate
        self.use_amp = use_amp
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training metadata
        self.training_metadata = {
            'start_time': datetime.now().isoformat(),
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate
        }
    
    def load_dataset(
        self,
        dataset_name: str = "erickfmm/agentlans__multilingual-sentences__paired_10_sts",
        max_train_samples: Optional[int] = None,
        max_eval_samples: Optional[int] = 10000,
        resample_balanced: bool = True,
        target_std: float = 0.3
    ):
        """
        Load and prepare the STS dataset.
        
        Scores are kept in [-1, 1] range to preserve sign:
        - Negative scores remain negative (dissimilar sentences)
        - Zero remains zero (neutral similarity)
        - Positive scores remain positive (similar sentences)
        
        Args:
            dataset_name: HuggingFace dataset name
            max_train_samples: Maximum training samples
            max_eval_samples: Maximum evaluation samples
            resample_balanced: Resample to get normal distribution centered at 0
            target_std: Target standard deviation for normal distribution (default: 0.3)
        """
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Load dataset
        dataset = load_dataset(dataset_name, split="train")
        
        logger.info(f"Dataset loaded: {len(dataset)} total samples")
        
        # Resample to balance distribution (avoid imbalance as recommended)
        if resample_balanced:
            dataset = self._resample_balanced(dataset, target_std=target_std)
            logger.info(f"After resampling: {len(dataset)} samples")
        
        # Shuffle and split
        dataset = dataset.shuffle(seed=42)
        
        if max_train_samples:
            split_point = max_train_samples
        else:
            split_point = int(len(dataset) * 0.95)  # 95% train, 5% eval
        
        train_dataset = dataset.select(range(split_point))
        eval_dataset = dataset.select(range(split_point, len(dataset)))
        
        if max_eval_samples and len(eval_dataset) > max_eval_samples:
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        
        logger.info(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
        
        # Convert to InputExample format (keeping scores in [-1, 1] to preserve sign)
        self.train_examples = self._dataset_to_examples(train_dataset)
        self.eval_examples = self._dataset_to_examples(eval_dataset)
        
        # Create evaluator
        self.evaluator = self._create_evaluator()
        
        return self.train_examples, self.eval_examples
    
    def _resample_balanced(self, dataset, bins: int = 20, target_std: float = 0.3):
        """
        Resample dataset to create a normal distribution centered at 0.
        
        This undersamples to avoid imbalanced data as recommended by the dataset authors.
        The final distribution will approximate N(0, target_std) in the [-1, 1] range.
        
        Args:
            dataset: The dataset to resample
            bins: Number of bins for score distribution (more bins = finer control)
            target_std: Target standard deviation for the normal distribution
        """
        from scipy import stats
        
        scores = np.array(dataset['score'])
        
        logger.info(f"Original distribution: mean={scores.mean():.4f}, std={scores.std():.4f}")
        logger.info(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # Create bins
        hist, bin_edges = np.histogram(scores, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Target: Normal distribution centered at 0 with specified std
        # Calculate expected count per bin based on normal distribution
        total_samples = len(scores)
        norm_dist = stats.norm(loc=0, scale=target_std)
        
        # Get probability for each bin
        bin_probs = []
        for i in range(bins):
            prob = norm_dist.cdf(bin_edges[i + 1]) - norm_dist.cdf(bin_edges[i])
            bin_probs.append(prob)
        
        bin_probs = np.array(bin_probs)
        bin_probs = bin_probs / bin_probs.sum()  # Normalize
        
        # Calculate target samples per bin
        # Use undersampling: limit by the most restrictive bin
        samples_if_all_filled = hist / bin_probs
        samples_if_all_filled[bin_probs < 1e-6] = np.inf  # Ignore near-zero probability bins
        
        # Total samples is limited by the bin with lowest samples/probability ratio
        total_target = int(np.min(samples_if_all_filled[np.isfinite(samples_if_all_filled)]))
        
        # Now calculate per-bin targets
        target_per_bin = (bin_probs * total_target).astype(int)
        
        logger.info(f"Undersampling from {len(scores)} to ~{total_target} samples")
        
        indices_to_keep = []
        for i in range(bins):
            # Get indices in this bin
            if i < bins - 1:
                in_bin = np.where((scores >= bin_edges[i]) & (scores < bin_edges[i + 1]))[0]
            else:
                in_bin = np.where((scores >= bin_edges[i]) & (scores <= bin_edges[i + 1]))[0]
            
            target = target_per_bin[i]
            
            # Sample from bin (undersample if needed)
            if len(in_bin) > target and target > 0:
                sampled = np.random.choice(in_bin, target, replace=False)
            elif target > 0:
                sampled = in_bin
            else:
                continue
            
            indices_to_keep.extend(sampled.tolist())
        
        resampled = dataset.select(indices_to_keep)
        resampled_scores = np.array([resampled[i]['score'] for i in range(len(resampled))])
        
        logger.info(f"Resampled distribution: mean={resampled_scores.mean():.4f}, std={resampled_scores.std():.4f}")
        logger.info(f"Final sample count: {len(resampled)}")
        
        return resampled
    
    def _dataset_to_examples(self, dataset):
        """
        Convert dataset to InputExample list.
        
        Scores are kept in [-1, 1] range (cosine similarity) to preserve:
        - Negative values: dissimilar sentences
        - Zero: neutral similarity  
        - Positive values: similar sentences
        """
        examples = []
        
        for item in dataset:
            # Keep score in [-1, 1] range to preserve sign
            score = float(item['score'])
            
            example = InputExample(
                texts=[item['sentence1'], item['sentence2']],
                label=score
            )
            examples.append(example)
        
        return examples
    
    def _create_evaluator(self):
        """Create evaluator for validation"""
        # Extract sentences and scores
        sentences1 = [ex.texts[0] for ex in self.eval_examples]
        sentences2 = [ex.texts[1] for ex in self.eval_examples]
        scores = [ex.label for ex in self.eval_examples]
        
        evaluator = evaluation.EmbeddingSimilarityEvaluator(
            sentences1,
            sentences2,
            scores,
            name='sts_eval'
        )
        
        return evaluator
    
    def train(self):
        """Run training loop"""
        logger.info("Starting SBERT training...")
        logger.info(f"Device: {self.model.device}")
        logger.info(f"Training samples: {len(self.train_examples)}")
        logger.info(f"Batch size: {self.batch_size}, Epochs: {self.epochs}")
        
        # Create data loader
        train_dataloader = DataLoader(
            self.train_examples,
            shuffle=True,
            batch_size=self.batch_size
        )
        
        # Define loss function (CosineSimilarityLoss for regression)
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Calculate total steps
        num_train_steps = len(train_dataloader) * self.epochs
        logger.info(f"Total training steps: {num_train_steps}")
        
        # Train
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=self.evaluator,
            epochs=self.epochs,
            warmup_steps=self.warmup_steps,
            evaluation_steps=self.evaluation_steps,
            output_path=self.output_dir,
            save_best_model=True,
            optimizer_params={'lr': self.learning_rate},
            use_amp=self.use_amp,
            show_progress_bar=True
        )
        
        logger.info(f"Training completed! Model saved to {self.output_dir}")
        
        # Final evaluation
        logger.info("Running final evaluation...")
        final_score = self.evaluator(self.model)
        logger.info(f"Final Spearman correlation: {final_score:.4f}")
        
        self.training_metadata['end_time'] = datetime.now().isoformat()
        self.training_metadata['final_score'] = final_score
        
        return final_score


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SBERT on TormentedBert")
    parser.add_argument("--pretrained", type=str, help="Path to pretrained checkpoint")
    parser.add_argument("--output_dir", type=str, default="./output/sbert_tormented_v2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=10000)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--pooling_mode", type=str, default="mean", choices=["mean", "cls", "max"])
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--no_resample", action="store_true", 
                       help="Don't resample dataset (default: undersample to normal distribution centered at 0)")
    parser.add_argument("--resample_std", type=float, default=0.3,
                       help="Target std for resampled normal distribution (default: 0.3)")
    
    args = parser.parse_args()
    
    # Setup
    logger.info("=" * 80)
    logger.info("SBERT Training on TORMENTED-BERT-Frankenstein v2")
    logger.info("=" * 80)
    
    # Create model
    if args.pretrained:
        model_wrapper = TormentedBertSentenceTransformer(
            pretrained_path=args.pretrained,
            pooling_mode=args.pooling_mode
        )
    else:
        config = UltraConfig(
            vocab_size=50000,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_loops=1,
            ode_steps=1,
            use_bitnet=True
        )
        model_wrapper = TormentedBertSentenceTransformer(
            model_config=config,
            pooling_mode=args.pooling_mode
        )
    
    model = model_wrapper.get_model()
    logger.info(f"Model initialized with pooling mode: {args.pooling_mode}")
    
    # Create trainer
    trainer = SBERTTrainer(
        model=model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        use_amp=not args.no_amp
    )
    
    # Load dataset
    trainer.load_dataset(
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        resample_balanced=not args.no_resample,
        target_std=args.resample_std
    )
    
    # Train
    final_score = trainer.train()
    
    logger.info("=" * 80)
    logger.info(f"Training complete! Final score: {final_score:.4f}")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
