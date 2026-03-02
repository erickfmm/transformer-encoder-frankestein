#!/usr/bin/env python3
"""
SBERT Fine-tuning for TORMENTED-BERT-Frankenstein
Fine-tunes the v2 model on Spanish sentence similarity using STS dataset.
Dataset: erickfmm/agentlans__multilingual-sentences__paired_10_sts
"""

import os
import torch
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

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

try:
    from ..model.tormented_bert_frankestein import TormentedBertFrankenstein, UltraConfig
    from ..utils.device import SUPPORTED_DEVICE_CHOICES, resolve_torch_device
    from ..utils.gpu_temp_guard import GPUTemperatureGuard
except ImportError:
    from model.tormented_bert_frankestein import TormentedBertFrankenstein, UltraConfig
    from utils.device import SUPPORTED_DEVICE_CHOICES, resolve_torch_device
    from utils.gpu_temp_guard import GPUTemperatureGuard

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SUPPORTED_SBERT_DATASET_TYPES = ("paired_similarity", "triplets", "qa")

DEFAULT_SBERT_COLUMNS = {
    "paired_similarity": {
        "sentence1": "sentence1",
        "sentence2": "sentence2",
        "similarity": "score",
    },
    "triplets": {
        "query": "Q",
        "positive": "POS",
        "negatives": "NEGs",
    },
    "qa": {
        "question": "Question",
        "answer": "Answer",
    },
}


class TormentedBertSentenceTransformer:
    """Wrapper to adapt TormentedBert for Sentence-BERT training"""
    
    def __init__(
        self,
        model_config: Optional[UltraConfig] = None,
        pretrained_path: Optional[str] = None,
        base_model_name_or_path: Optional[str] = None,
        max_seq_length: int = 512,
        pooling_mode: str = "mean",
        trust_remote_code: bool = False,
        device: str = "auto"
    ):
        """
        Initialize SBERT model with TormentedBert base.
        
        Args:
            model_config: UltraConfig for the model (if training from scratch)
            pretrained_path: Path to pretrained checkpoint
            base_model_name_or_path: Any HF/local model path compatible with sentence-transformers
            max_seq_length: Maximum sequence length
            pooling_mode: Pooling strategy ("mean", "cls", "max")
        """
        self.max_seq_length = max_seq_length
        self.pooling_mode = pooling_mode
        self.trust_remote_code = bool(trust_remote_code)
        self.device = resolve_torch_device(device)

        if base_model_name_or_path:
            logger.info(f"Loading base model for SBERT from {base_model_name_or_path}")
            self.base_model = None
            self.config = None
            self.model = self._build_hf_sentence_transformer(base_model_name_or_path)
            return
        
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

        self.base_model.to(self.device)
        
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
            modules=[word_embedding_model, pooling_model, normalize],
            device=self.device,
        )
        
        return model

    def _build_hf_sentence_transformer(self, base_model_name_or_path: str) -> SentenceTransformer:
        """Build SentenceTransformer from a base HF model identifier/path."""
        transformer_kwargs = {
            "max_seq_length": self.max_seq_length,
        }

        # sentence-transformers changed this constructor across versions.
        try:
            word_embedding_model = models.Transformer(
                base_model_name_or_path,
                max_seq_length=self.max_seq_length,
                model_args={"trust_remote_code": self.trust_remote_code},
                tokenizer_args={"trust_remote_code": self.trust_remote_code},
            )
        except TypeError:
            word_embedding_model = models.Transformer(
                base_model_name_or_path,
                **transformer_kwargs,
            )

        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=(self.pooling_mode == "mean"),
            pooling_mode_cls_token=(self.pooling_mode == "cls"),
            pooling_mode_max_tokens=(self.pooling_mode == "max")
        )

        normalize = models.Normalize()
        return SentenceTransformer(
            modules=[word_embedding_model, pooling_model, normalize],
            device=self.device,
        )
    
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
        use_amp: bool = True,
        device: str = "auto",
        gpu_temp_guard_enabled: bool = True,
        gpu_temp_pause_threshold_c: float = 90.0,
        gpu_temp_resume_threshold_c: float = 80.0,
        gpu_temp_critical_threshold_c: Optional[float] = None,
        gpu_temp_poll_interval_seconds: float = 30.0,
        nvml_device_index: int = 0,
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
        self.device = resolve_torch_device(device)
        self.gpu_temp_guard = GPUTemperatureGuard(
            enabled=bool(gpu_temp_guard_enabled),
            device=self.device,
            nvml_device_index=int(nvml_device_index),
            pause_threshold_c=float(gpu_temp_pause_threshold_c),
            resume_threshold_c=float(gpu_temp_resume_threshold_c),
            critical_threshold_c=gpu_temp_critical_threshold_c,
            poll_interval_seconds=float(gpu_temp_poll_interval_seconds),
        )
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training metadata
        self.training_metadata = {
            'start_time': datetime.now().isoformat(),
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate
        }
        if self.gpu_temp_guard.is_active:
            logger.info(
                "GPU thermal guard enabled for SBERT (pause>%.1fC, resume<=%.1fC, poll=%.1fs, critical=%s)",
                float(gpu_temp_pause_threshold_c),
                float(gpu_temp_resume_threshold_c),
                float(gpu_temp_poll_interval_seconds),
                (
                    f"{float(gpu_temp_critical_threshold_c):.1f}C"
                    if gpu_temp_critical_threshold_c is not None
                    else "disabled"
                ),
            )
        else:
            logger.info("GPU thermal guard disabled for SBERT")
    
    def load_dataset(
        self,
        dataset_name: str = "erickfmm/agentlans__multilingual-sentences__paired_10_sts",
        dataset_type: str = "paired_similarity",
        columns: Optional[Dict[str, str]] = None,
        query_prefix: str = "",
        document_prefix: str = "",
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
            dataset_type: Dataset format type (paired_similarity, triplets, qa)
            columns: Optional dataset column mapping
            query_prefix: Optional prefix prepended to query/question text
            document_prefix: Optional prefix prepended to answer/positive/negative text
            max_train_samples: Maximum training samples
            max_eval_samples: Maximum evaluation samples
            resample_balanced: Resample to get normal distribution centered at 0
            target_std: Target standard deviation for normal distribution (default: 0.3)
        """
        dataset_type = str(dataset_type).strip().lower()
        if dataset_type not in SUPPORTED_SBERT_DATASET_TYPES:
            raise ValueError(
                f"Unsupported dataset_type='{dataset_type}'. "
                f"Supported values: {', '.join(SUPPORTED_SBERT_DATASET_TYPES)}"
            )
        self.dataset_type = dataset_type
        self.query_prefix = str(query_prefix or "")
        self.document_prefix = str(document_prefix or "")
        self.dataset_columns = self._resolve_dataset_columns(dataset_type, columns or {})

        logger.info(f"Loading dataset: {dataset_name}")
        logger.info(
            "SBERT dataset configuration: type=%s, columns=%s, query_prefix=%r, document_prefix=%r",
            self.dataset_type,
            self.dataset_columns,
            self.query_prefix,
            self.document_prefix,
        )
        
        # Load dataset
        dataset = load_dataset(dataset_name, split="train")
        
        logger.info(f"Dataset loaded: {len(dataset)} total samples")
        self._validate_required_columns(dataset)
        
        # Resample to balance distribution (avoid imbalance as recommended)
        if resample_balanced and self.dataset_type != "paired_similarity":
            logger.warning(
                "resample_balanced is only supported for paired_similarity datasets; disabling for dataset_type='%s'",
                self.dataset_type,
            )
        elif resample_balanced:
            similarity_column = self.dataset_columns["similarity"]
            dataset = self._resample_balanced(
                dataset,
                score_column=similarity_column,
                target_std=target_std,
            )
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
        
        # Convert to InputExample format according to dataset type
        self.train_examples = self._dataset_to_examples(train_dataset, split_name="train")
        self.eval_examples = self._dataset_to_examples(eval_dataset, split_name="eval")
        
        # Create evaluator for paired similarity only
        self.evaluator = self._create_evaluator() if self.dataset_type == "paired_similarity" else None
        
        return self.train_examples, self.eval_examples
    
    def _resolve_dataset_columns(self, dataset_type: str, columns: Dict[str, Any]) -> Dict[str, str]:
        defaults = dict(DEFAULT_SBERT_COLUMNS[dataset_type])
        resolved = dict(defaults)
        for key in defaults:
            value = columns.get(key)
            if value is None:
                continue
            text_value = str(value).strip()
            if not text_value:
                raise ValueError(
                    f"Column mapping for key '{key}' cannot be empty for dataset_type='{dataset_type}'"
                )
            resolved[key] = text_value
        return resolved

    def _validate_required_columns(self, dataset):
        available = set(dataset.column_names)
        required = set(self.dataset_columns.values())
        missing = sorted(required - available)
        if missing:
            raise ValueError(
                f"Missing required dataset columns for dataset_type='{self.dataset_type}': {missing}. "
                f"Available columns: {sorted(available)}"
            )

    def _with_prefix(self, text: Any, prefix: str) -> str:
        base = str(text)
        return f"{prefix}{base}" if prefix else base

    def _resample_balanced(
        self,
        dataset,
        score_column: str,
        bins: int = 20,
        target_std: float = 0.3,
    ):
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
        
        scores = np.array(dataset[score_column], dtype=np.float32)
        
        logger.info(f"Original distribution: mean={scores.mean():.4f}, std={scores.std():.4f}")
        logger.info(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # Create bins
        hist, bin_edges = np.histogram(scores, bins=bins)
        # Target: Normal distribution centered at 0 with specified std
        # Calculate expected count per bin based on normal distribution
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
        resampled_scores = np.array(
            [float(resampled[i][score_column]) for i in range(len(resampled))],
            dtype=np.float32,
        )
        
        logger.info(f"Resampled distribution: mean={resampled_scores.mean():.4f}, std={resampled_scores.std():.4f}")
        logger.info(f"Final sample count: {len(resampled)}")
        
        return resampled
    
    def _dataset_to_examples(self, dataset, split_name: str):
        """
        Convert dataset split to InputExample list according to dataset_type.
        """
        if self.dataset_type == "paired_similarity":
            return self._dataset_to_examples_paired_similarity(dataset)
        if self.dataset_type == "triplets":
            return self._dataset_to_examples_triplets(dataset, split_name=split_name)
        if self.dataset_type == "qa":
            return self._dataset_to_examples_qa(dataset)
        raise ValueError(f"Unsupported dataset_type='{self.dataset_type}'")

    def _dataset_to_examples_paired_similarity(self, dataset):
        examples = []
        similarity_column = self.dataset_columns["similarity"]
        sentence1_column = self.dataset_columns["sentence1"]
        sentence2_column = self.dataset_columns["sentence2"]
        
        for item in dataset:
            # Keep score in [-1, 1] range to preserve sign
            score = float(item[similarity_column])
            
            example = InputExample(
                texts=[
                    self._with_prefix(item[sentence1_column], self.query_prefix),
                    self._with_prefix(item[sentence2_column], self.document_prefix),
                ],
                label=score
            )
            examples.append(example)
        
        return examples

    def _dataset_to_examples_triplets(self, dataset, split_name: str):
        examples: List[InputExample] = []
        skipped_rows = 0
        query_column = self.dataset_columns["query"]
        positive_column = self.dataset_columns["positive"]
        negatives_column = self.dataset_columns["negatives"]

        for item in dataset:
            negatives = item[negatives_column]
            if not isinstance(negatives, list):
                skipped_rows += 1
                continue

            query_text = self._with_prefix(item[query_column], self.query_prefix)
            positive_text = self._with_prefix(item[positive_column], self.document_prefix)

            valid_negative_count = 0
            for negative in negatives:
                if not isinstance(negative, str) or not negative.strip():
                    continue
                negative_text = self._with_prefix(negative, self.document_prefix)
                examples.append(InputExample(texts=[query_text, positive_text, negative_text]))
                valid_negative_count += 1
            if valid_negative_count == 0:
                skipped_rows += 1

        if skipped_rows > 0:
            logger.warning(
                "Skipped %d rows in %s split for triplets due to malformed or empty negatives list",
                skipped_rows,
                split_name,
            )
        return examples

    def _dataset_to_examples_qa(self, dataset):
        examples: List[InputExample] = []
        question_column = self.dataset_columns["question"]
        answer_column = self.dataset_columns["answer"]
        for item in dataset:
            examples.append(
                InputExample(
                    texts=[
                        self._with_prefix(item[question_column], self.query_prefix),
                        self._with_prefix(item[answer_column], self.document_prefix),
                    ]
                )
            )
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
        logger.info(f"Dataset type: {self.dataset_type}")
        
        # Create data loader
        train_dataloader = DataLoader(
            self.train_examples,
            shuffle=True,
            batch_size=self.batch_size
        )
        train_dataloader = _ThermalGuardedDataLoader(
            dataloader=train_dataloader,
            guard=self.gpu_temp_guard,
            context_prefix="sbert.fit",
        )
        
        # Define loss function according to dataset type.
        if self.dataset_type == "paired_similarity":
            train_loss = losses.CosineSimilarityLoss(self.model)
        else:
            train_loss = losses.MultipleNegativesRankingLoss(self.model)
        
        # Calculate total steps
        num_train_steps = len(train_dataloader) * self.epochs
        logger.info(f"Total training steps: {num_train_steps}")
        
        # Train
        fit_kwargs = dict(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.epochs,
            warmup_steps=self.warmup_steps,
            evaluation_steps=self.evaluation_steps,
            output_path=self.output_dir,
            save_best_model=True,
            optimizer_params={'lr': self.learning_rate},
            use_amp=self.use_amp,
            show_progress_bar=True
        )
        if self.evaluator is not None:
            fit_kwargs["evaluator"] = self.evaluator
        self.model.fit(**fit_kwargs)
        
        logger.info(f"Training completed! Model saved to {self.output_dir}")
        
        final_score = None
        if self.evaluator is not None:
            # Final evaluation
            logger.info("Running final evaluation...")
            final_score = self.evaluator(self.model)
            logger.info(f"Final Spearman correlation: {final_score:.4f}")
        else:
            logger.info("Skipping final evaluator for dataset type '%s'", self.dataset_type)
        
        self.training_metadata['end_time'] = datetime.now().isoformat()
        self.training_metadata['final_score'] = final_score
        
        return final_score


class _ThermalGuardedDataLoader:
    """Dataloader wrapper that blocks when the GPU is too hot."""

    def __init__(self, dataloader: DataLoader, guard: GPUTemperatureGuard, context_prefix: str):
        self._dataloader = dataloader
        self._guard = guard
        self._context_prefix = context_prefix

    def __len__(self):
        return len(self._dataloader)

    def __iter__(self):
        for batch_idx, batch in enumerate(self._dataloader):
            result = self._guard.wait_until_safe(
                context=f"{self._context_prefix} batch={batch_idx + 1}"
            )
            if result.paused:
                logger.warning(
                    "[ThermalGuard][SBERT] pause event: %s (temp=%.1fC)",
                    result.repair_action,
                    result.temp_c,
                )
            yield batch


def main(argv=None):
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SBERT on TormentedBert")
    parser.add_argument("--base-model", type=str, default=None, help="HF model id/path for base-model SBERT finetuning")
    parser.add_argument("--pretrained", type=str, help="Path to pretrained checkpoint")
    parser.add_argument("--output_dir", type=str, default="./output/sbert_tormented_v2")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="erickfmm/agentlans__multilingual-sentences__paired_10_sts",
        help="Hugging Face dataset name for SBERT training",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="paired_similarity",
        choices=SUPPORTED_SBERT_DATASET_TYPES,
        help="Dataset layout used for SBERT training",
    )
    parser.add_argument("--col_sentence1", type=str, default=None, help="Column name for sentence1 in paired_similarity datasets")
    parser.add_argument("--col_sentence2", type=str, default=None, help="Column name for sentence2 in paired_similarity datasets")
    parser.add_argument("--col_similarity", type=str, default=None, help="Column name for similarity score in paired_similarity datasets")
    parser.add_argument("--col_query", type=str, default=None, help="Column name for query in triplets datasets")
    parser.add_argument("--col_positive", type=str, default=None, help="Column name for positive text in triplets datasets")
    parser.add_argument("--col_negatives", type=str, default=None, help="Column name for negatives list in triplets datasets")
    parser.add_argument("--col_question", type=str, default=None, help="Column name for question in qa datasets")
    parser.add_argument("--col_answer", type=str, default=None, help="Column name for answer in qa datasets")
    parser.add_argument("--query_prefix", type=str, default="", help="Optional prefix for query/question text")
    parser.add_argument("--document_prefix", type=str, default="", help="Optional prefix for document/answer/positive/negative text")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--evaluation_steps", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=10000)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--pooling_mode", type=str, default="mean", choices=["mean", "cls", "max"])
    parser.add_argument("--trust_remote_code", action="store_true", help="Allow remote code for HF models/tokenizers")
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--no_resample", action="store_true", 
                       help="Don't resample dataset (default: undersample to normal distribution centered at 0)")
    parser.add_argument("--resample_std", type=float, default=0.3,
                       help="Target std for resampled normal distribution (default: 0.3)")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=SUPPORTED_DEVICE_CHOICES,
        help="Device to run SBERT training on"
    )
    gpu_temp_group = parser.add_mutually_exclusive_group()
    gpu_temp_group.add_argument(
        "--gpu-temp-guard",
        dest="gpu_temp_guard",
        action="store_true",
        help="Enable GPU thermal guard for CUDA training.",
    )
    gpu_temp_group.add_argument(
        "--no-gpu-temp-guard",
        dest="gpu_temp_guard",
        action="store_false",
        help="Disable GPU thermal guard.",
    )
    parser.set_defaults(gpu_temp_guard=None)
    parser.add_argument("--gpu-temp-pause-threshold-c", type=float, default=90.0)
    parser.add_argument("--gpu-temp-resume-threshold-c", type=float, default=80.0)
    parser.add_argument("--gpu-temp-critical-threshold-c", type=float, default=None)
    parser.add_argument("--gpu-temp-poll-interval-seconds", type=float, default=30.0)
    parser.add_argument("--nvml-device-index", type=int, default=0)
    
    args = parser.parse_args(argv)
    resolved_device = resolve_torch_device(args.device)
    logger.info(f"SBERT train device requested='{args.device}', resolved='{resolved_device}'")
    gpu_temp_guard_enabled = True if args.gpu_temp_guard is None else bool(args.gpu_temp_guard)
    if not resolved_device.startswith("cuda"):
        gpu_temp_guard_enabled = False
    if args.gpu_temp_pause_threshold_c <= 0:
        raise ValueError("gpu_temp_pause_threshold_c must be > 0")
    if args.gpu_temp_resume_threshold_c <= 0:
        raise ValueError("gpu_temp_resume_threshold_c must be > 0")
    if args.gpu_temp_resume_threshold_c >= args.gpu_temp_pause_threshold_c:
        raise ValueError("gpu_temp_resume_threshold_c must be < gpu_temp_pause_threshold_c")
    if args.gpu_temp_poll_interval_seconds <= 0:
        raise ValueError("gpu_temp_poll_interval_seconds must be > 0")
    if (
        args.gpu_temp_critical_threshold_c is not None
        and args.gpu_temp_critical_threshold_c <= 0
    ):
        raise ValueError("gpu_temp_critical_threshold_c must be > 0 when provided")
    
    # Setup
    logger.info("=" * 80)
    logger.info("SBERT Training on TORMENTED-BERT-Frankenstein v2")
    logger.info("=" * 80)

    if args.base_model and args.pretrained:
        raise ValueError("--base-model and --pretrained are mutually exclusive")
    
    # Create model
    if args.base_model:
        model_wrapper = TormentedBertSentenceTransformer(
            base_model_name_or_path=args.base_model,
            max_seq_length=args.max_seq_length,
            pooling_mode=args.pooling_mode,
            trust_remote_code=args.trust_remote_code,
            device=resolved_device,
        )
    elif args.pretrained:
        model_wrapper = TormentedBertSentenceTransformer(
            pretrained_path=args.pretrained,
            max_seq_length=args.max_seq_length,
            pooling_mode=args.pooling_mode,
            device=resolved_device,
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
            max_seq_length=args.max_seq_length,
            pooling_mode=args.pooling_mode,
            device=resolved_device,
        )
    
    model = model_wrapper.get_model()
    logger.info(f"Model initialized with pooling mode: {args.pooling_mode}")
    
    # Create trainer
    trainer = SBERTTrainer(
        model=model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        evaluation_steps=args.evaluation_steps,
        learning_rate=args.learning_rate,
        use_amp=not args.no_amp,
        device=resolved_device,
        gpu_temp_guard_enabled=gpu_temp_guard_enabled,
        gpu_temp_pause_threshold_c=args.gpu_temp_pause_threshold_c,
        gpu_temp_resume_threshold_c=args.gpu_temp_resume_threshold_c,
        gpu_temp_critical_threshold_c=args.gpu_temp_critical_threshold_c,
        gpu_temp_poll_interval_seconds=args.gpu_temp_poll_interval_seconds,
        nvml_device_index=args.nvml_device_index,
    )
    
    # Load dataset
    column_overrides = {
        "sentence1": args.col_sentence1,
        "sentence2": args.col_sentence2,
        "similarity": args.col_similarity,
        "query": args.col_query,
        "positive": args.col_positive,
        "negatives": args.col_negatives,
        "question": args.col_question,
        "answer": args.col_answer,
    }
    column_overrides = {
        key: value for key, value in column_overrides.items()
        if isinstance(value, str) and value.strip()
    }
    trainer.load_dataset(
        dataset_name=args.dataset_name,
        dataset_type=args.dataset_type,
        columns=column_overrides,
        query_prefix=args.query_prefix,
        document_prefix=args.document_prefix,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        resample_balanced=not args.no_resample,
        target_std=args.resample_std
    )
    
    # Train
    final_score = trainer.train()
    
    logger.info("=" * 80)
    if final_score is not None:
        logger.info(f"Training complete! Final score: {final_score:.4f}")
    else:
        logger.info("Training complete! Final score: n/a (no evaluator configured for this dataset type)")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
