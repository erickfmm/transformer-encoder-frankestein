#!/usr/bin/env python3
"""
TORMENTED-BERT-Frankenstein Training Pipeline

TORMENTED = Ternary ODE Retention Mamba Experts Neural Tanh Encoder Depth

Spanish Transformer with:
- BitNet b1.58 (Ternary Weights)
- Neural ODE Attention
- RetNet Multi-Scale Retention
- Mamba State Space Models
- Sparse Mixture-of-Experts
- Dynamic Tanh Normalization
- Recursive Loop Architecture

Dataset: erickfmm/red_pajama_es_hq_35
Vocabulary: 50,000 tokens
Storage limit: <300GB

STABLE CONFIGURATION NOTES (based on literature research):
- Layer pattern prioritizes RetNet for stability (good with recurrent patterns)
- Titan attention anchors the pattern (proven stable)
- ODE layers need lower learning rates (continuous dynamics)
- Mamba layers work best interspersed (not consecutive)
- 6-layer pattern provides balance: 2x RetNet, 2x Titan, 1x ODE, 1x Mamba
"""

import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader

from tokenizer.spm_spa_redpajama35 import SpanishSPMTokenizer
from training.streaming_mlm_dataset import StreamingMLMDataset
from training.trainer import TitanTrainer
from training.config_loader import load_training_config, list_config_paths
from model.tormented_bert_frankestein import TormentedBertFrankenstein, TormentedBertMini, UltraConfig

# ==================== MAIN EXECUTION ====================
def main():
    """Main training pipeline for TORMENTED-BERT-Frankenstein"""
    parser = argparse.ArgumentParser(description="Train models from YAML configs")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=os.environ.get("CONFIG_NAME", "mini"),
        help="Config name under src/training/configs (without extension)"
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configs and exit"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from YAML"
    )
    parser.add_argument(
        "--model-mode",
        choices=["frankenstein", "mini"],
        default=None,
        help="Deprecated: use --config-name instead"
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("⚡ Starting TORMENTED-BERT-Frankenstein training ⚡")
    logging.info(f"Current directory: {os.getcwd()}")

    try:
        import psutil
        logging.info(f"Available storage: {psutil.disk_usage('.').free/1024**3:.2f}GB")
    except ImportError:
        logging.warning("psutil not installed, storage monitoring limited")

    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    available_configs = list_config_paths(config_dir)
    if args.list_configs:
        logging.info("Available configs:")
        for name, path in available_configs.items():
            logging.info(f"  - {name}: {path}")
        return

    config_path = None
    if args.config:
        config_path = args.config
    else:
        config_name = args.config_name
        if args.model_mode and args.model_mode not in ("", None):
            config_name = args.model_mode
        config_path = available_configs.get(config_name)

    if not config_path or not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    model_class, config, training_config, training_runtime = load_training_config(config_path)
    logging.info(f"Using config: {config_path}")
    
    # Step 1: Train/Load tokenizer
    logging.info("\n" + "="*60)
    logging.info(f"Step 1: Training/Loading SPM tokenizer ({config.vocab_size} vocab)")
    logging.info("="*60)
    
    # Check if tokenizer already exists
    vocab_size = config.vocab_size
    model_prefix = "es_redpajama_50k" if vocab_size == 50_000 else f"es_redpajama_{vocab_size}"
    model_path = f"{model_prefix}.model"
    if os.path.exists(model_path):
        logging.info("Loading existing tokenizer...")
        tokenizer = SpanishSPMTokenizer(vocab_size=vocab_size, model_path=model_path)
    else:
        logging.info("Training new tokenizer with maximum data (100GB RAM target)...")
        tokenizer = SpanishSPMTokenizer(vocab_size=vocab_size)
        tokenizer.train(
            model_prefix=model_prefix,
            max_training_samples=50_000_000,  # Up to 50M samples
            target_ram_gb=100.0  # Use up to 100GB RAM for quality tokenizer
        )
    
    logging.info(f"Tokenizer loaded with {len(tokenizer.vocab)} tokens")
    logging.info(f"Tokenizer model path: {tokenizer.model_path}")
    
    # Step 2: Create TORMENTED model
    logging.info("\n" + "="*60)
    logging.info("Step 2: Creating TORMENTED-BERT-Frankenstein model")
    logging.info("="*60)
    
    stable_layer_pattern = [
        "retnet",
        "titan_attn",
        "retnet",
        "mamba",
        "titan_attn",
        "ode",
    ]

    if not config.layer_pattern:
        config.layer_pattern = stable_layer_pattern

    if model_class == "mini":
        model = TormentedBertMini(config)
    else:
        model = TormentedBertFrankenstein(config)
    
    logging.info(f"Model Config:")
    logging.info(f"  - Model Class: {model_class}")
    logging.info(f"  - Hidden Size: {config.hidden_size}")
    logging.info(f"  - Layers: {config.num_layers} x {config.num_loops} = {config.num_layers * config.num_loops} logical")
    logging.info(f"  - Layer Pattern: {config.layer_pattern}")
    logging.info(f"  - BitNet: {config.use_bitnet}")
    logging.info(f"  - ODE Solver: {config.ode_solver} ({config.ode_steps} steps)")
    logging.info(f"  - Norm Type: {config.norm_type}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total Parameters: {total_params/1e6:.2f}M")
    logging.info(f"Trainable Parameters: {trainable_params/1e6:.2f}M")
    
    # Step 3: Prepare dataset with fault-tolerant parallel processing
    logging.info("\n" + "="*60)
    logging.info("Step 3: Preparing MLM dataset with resilient caching")
    logging.info("="*60)
    
    max_length = int(training_runtime.get("max_length", 512))
    mlm_probability = float(training_runtime.get("mlm_probability", 0.15))
    max_samples = int(training_runtime.get("max_samples", 20_000_000))
    dataset_batch_size = int(training_runtime.get("dataset_batch_size", 25_000))
    dataset_num_workers = int(training_runtime.get("num_workers", 8))
    cache_dir = training_runtime.get("cache_dir", "./temp_data/v2_dataset_cache")
    local_parquet_dir = training_runtime.get(
        "local_parquet_dir",
        "/home/erickfmm/.cache/huggingface/hub/"
        "datasets--erickfmm--red_pajama_es_hq_35/"
        "snapshots/bd7286c289a95dc3803c375bc36aaaeb138b1eab/"
        "train/",
    )
    prefer_local_cache = bool(training_runtime.get("prefer_local_cache", True))
    stream_local_parquet = bool(training_runtime.get("stream_local_parquet", True))

    dataset = StreamingMLMDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        mlm_probability=mlm_probability,
        max_samples=max_samples,
        batch_size=dataset_batch_size,
        num_workers=dataset_num_workers,
        cache_dir=cache_dir,
        local_parquet_dir=local_parquet_dir,
        prefer_local_cache=prefer_local_cache,
        stream_local_parquet=stream_local_parquet,
    )
    
    # Show dataset statistics
    stats = dataset.get_stats()
    logging.info(f"Dataset Statistics:")
    logging.info(f"  - Total examples: {stats['total_examples']}")
    logging.info(f"  - Completed batches: {stats['completed_batches']}")
    logging.info(f"  - Samples processed: {stats['total_samples_processed']}")
    logging.info(f"  - Parallel workers: {stats['num_workers']}")
    logging.info(f"  - Cache directory: {stats['cache_dir']}")
    logging.info(f"  - Recovery enabled: Cache will be reused on restart")
    
    # Batch size policy:
    # - Use YAML default if present.
    # - CLI override takes priority.
    batch_size = training_runtime.get("batch_size", None)
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be > 0")
        batch_size = args.batch_size
    if batch_size is None:
        batch_size = 1

    dataloader_workers = int(training_runtime.get("dataloader_workers", 2))
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataloader_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    logging.info(f"Dataset size: {len(dataset)} examples")
    logging.info(f"Batch size: {batch_size}")
    if args.batch_size is None:
        logging.info("Batch size selected automatically based on model mode and device")
    logging.info(f"Steps per epoch: {len(dataloader)}")
    
    # Step 4: Train
    logging.info("\n" + "="*60)
    logging.info(f"Step 4: Training TORMENTED-BERT ({model_class})")
    logging.info("="*60)
    
    # Configure training behavior from YAML
    trainer = TitanTrainer(model, config, training_config=training_config)
    
    num_epochs = 5  # Increased for better convergence
    nan_detected = False
    
    try:
        for epoch in range(num_epochs):
            logging.info(f"\n🚀 Starting Epoch {epoch+1}/{num_epochs}")
            
            try:
                avg_loss, should_stop = trainer.train_epoch(dataloader, epoch)
                
                # Check if NaN was detected
                if should_stop:
                    logging.error(f"❌ Training stopped due to NaN at epoch {epoch+1}")
                    nan_detected = True
                    break
                
                logging.info(f"✅ Epoch {epoch+1} completed - Average Loss: {avg_loss:.4f}")
                
                # Save epoch checkpoint (in addition to rolling checkpoints)
                checkpoint_path = trainer.save_checkpoint(epoch, suffix="_epoch_end")
                logging.info(f"💾 Epoch checkpoint saved: {checkpoint_path}")
                
                # Memory and storage report
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_cached = torch.cuda.memory_reserved() / 1024**3
                    logging.info(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Cached: {memory_cached:.2f}GB")
                
                storage_used = trainer.storage_manager.used_bytes / 1024**3
                logging.info(f"Storage used: {storage_used:.2f}GB / 300GB")
                
                # Early stopping on memory issues
                if storage_used > 250:
                    logging.warning("Approaching storage limit, stopping training")
                    break
                    
            except Exception as e:
                logging.error(f"Error in epoch {epoch+1}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Try to save emergency checkpoint
                try:
                    emergency_path = trainer.save_checkpoint(epoch, suffix="_emergency")
                    logging.info(f"Emergency checkpoint saved: {emergency_path}")
                except:
                    logging.error("Failed to save emergency checkpoint")
                
                raise
    finally:
        # Always close the CSV logger
        trainer.close()
    
    if nan_detected:
        logging.error("\n" + "="*60)
        logging.error("🚨 TRAINING TERMINATED DUE TO NaN")
        logging.error("Check training_metrics.csv for the progression leading to NaN")
        logging.error("Review the debug logs above for root cause analysis")
        logging.error("="*60)
    else:
        logging.info("\n" + "="*60)
        logging.info("🎉 TORMENTED-BERT-Frankenstein training completed successfully!")
        logging.info("="*60)
        
        # Final model evaluation
        model.eval()
        with torch.no_grad():
            # Test forward pass
            test_input = torch.randint(0, 50000, (1, 512))
            if torch.cuda.is_available():
                test_input = test_input.to("cuda")
            
            logging.info("🔍 Testing final model...")
            test_output = model(test_input)
            logging.info(f"✅ Model output shape: {test_output.shape}")
            logging.info(f"Output range: [{test_output.min().item():.3f}, {test_output.max().item():.3f}]")
    
    # Cleanup
    logging.info("\n🧹 Cleaning up temporary files...")
    tokenizer.storage_manager.cleanup()
    # Note: NOT cleaning dataset cache to allow recovery on restart
    # To clean manually: dataset.cleanup_cache()
    trainer.storage_manager.cleanup()
    
    logging.info("💡 Dataset cache preserved for fault recovery")
    logging.info(f"   Location: {stats['cache_dir']}")
    
    # Log summary of saved checkpoints
    logging.info("\n📁 Checkpoint Summary:")
    logging.info(f"  Rolling checkpoints kept: {len(trainer.rolling_checkpoints)}")
    for cp in trainer.rolling_checkpoints:
        logging.info(f"    - {cp}")
    logging.info(f"  Best model checkpoints: {len(trainer.best_checkpoints)}")
    for neg_loss, cp in sorted(trainer.best_checkpoints, reverse=True):
        logging.info(f"    - {cp} (loss={-neg_loss:.6f})")
    
    logging.info(f"\n📊 Training metrics saved to: {training_config.csv_log_path}")
    logging.info("✨ Training pipeline completed!")

if __name__ == "__main__":
    main()
