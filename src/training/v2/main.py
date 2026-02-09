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
import logging
import torch
from torch.utils.data import DataLoader

from tokenizer.spm_spa_redpajama35 import SpanishSPMTokenizer
from training.v1.streaming_mlm_dataset import StreamingMLMDataset
from training.v2.trainer import TitanTrainer, TrainingConfig
from src.model.v2.tormented_bert_frankestein import TormentedBertFrankenstein, UltraConfig

# ==================== MAIN EXECUTION ====================
def main():
    """Main training pipeline for TORMENTED-BERT-Frankenstein"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("⚡ Starting TORMENTED-BERT-Frankenstein training ⚡")
    logging.info(f"Current directory: {os.getcwd()}")
    
    try:
        import psutil
        logging.info(f"Available storage: {psutil.disk_usage('.').free/1024**3:.2f}GB")
    except ImportError:
        logging.warning("psutil not installed, storage monitoring limited")
    
    # Step 1: Train/Load tokenizer
    logging.info("\n" + "="*60)
    logging.info("Step 1: Training/Loading SPM tokenizer (50k vocab)")
    logging.info("="*60)
    
    # Check if tokenizer already exists
    model_path = "es_redpajama_50k.model"
    if os.path.exists(model_path):
        logging.info("Loading existing tokenizer...")
        tokenizer = SpanishSPMTokenizer(vocab_size=50000, model_path=model_path)
    else:
        logging.info("Training new tokenizer with maximum data (100GB RAM target)...")
        tokenizer = SpanishSPMTokenizer(vocab_size=50000)
        tokenizer.train(
            model_prefix="es_redpajama_50k",
            max_training_samples=50_000_000,  # Up to 50M samples
            target_ram_gb=100.0  # Use up to 100GB RAM for quality tokenizer
        )
    
    logging.info(f"Tokenizer loaded with {len(tokenizer.vocab)} tokens")
    logging.info(f"Tokenizer model path: {tokenizer.model_path}")
    
    # Step 2: Create TORMENTED model
    logging.info("\n" + "="*60)
    logging.info("Step 2: Creating TORMENTED-BERT-Frankenstein model")
    logging.info("="*60)
    
    # =========================================================================
    # STABLE LAYER PATTERN CONFIGURATION
    # =========================================================================
    # Based on research from hybrid architectures (TransMamba, SST, etc.):
    # 
    # Key insights for stability:
    # 1. RetNet: Most stable for long-range dependencies, use as anchors
    # 2. Titan Attention: Standard attention, proven stable, good for local patterns
    # 3. ODE: Continuous dynamics can be unstable, use sparingly, lower LR
    # 4. Mamba: Good efficiency but can struggle with certain patterns
    #
    # STABLE 6-LAYER PATTERN (repeated to fill num_layers):
    # [retnet -> titan_attn -> retnet -> mamba -> titan_attn -> ode]
    #
    # Rationale:
    # - Start with RetNet for stable gradient flow initialization
    # - Alternate RetNet/Titan for stability anchoring
    # - Place ODE at the end where gradients are more stable
    # - Mamba in middle position, sandwiched by stable layers
    # - 2x RetNet, 2x Titan, 1x Mamba, 1x ODE per cycle
    # =========================================================================
    
    stable_layer_pattern = [
        "retnet",       # 1. Stable anchor, good gradient flow
        "titan_attn",   # 2. Proven attention mechanism
        "retnet",       # 3. Another stable anchor
        "mamba",        # 4. Efficient SSM, sandwiched by stable layers
        "titan_attn",   # 5. Attention for local patterns
        "ode"           # 6. ODE at end, more stable gradients from above
    ]
    
    # Configure for available hardware (P40 24GB target)
    config = UltraConfig(
        vocab_size=50000,
        hidden_size=1024,           # Reduced from 2048 for stability
        num_layers=12,              # Physical layers (uses 2 cycles of 6-pattern)
        num_loops=2,                # Logical depth = 24
        num_heads=16,
        retention_heads=8,
        num_experts=4,              # Reduced MoE experts
        top_k_experts=2,
        dropout=0.1,
        ode_solver="rk4",
        ode_steps=4,                # Low steps for speed
        use_bitnet=True,            # Essential for memory efficiency
        norm_type="dynamic_tanh",
        layer_pattern=stable_layer_pattern  # Use stable pattern
    )
    
    logging.info(f"Model Config:")
    logging.info(f"  - Hidden Size: {config.hidden_size}")
    logging.info(f"  - Layers: {config.num_layers} x {config.num_loops} = {config.num_layers * config.num_loops} logical")
    logging.info(f"  - Layer Pattern: {config.layer_pattern}")
    logging.info(f"  - BitNet: {config.use_bitnet}")
    logging.info(f"  - ODE Solver: {config.ode_solver} ({config.ode_steps} steps)")
    
    model = TormentedBertFrankenstein(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total Parameters: {total_params/1e6:.2f}M")
    logging.info(f"Trainable Parameters: {trainable_params/1e6:.2f}M")
    
    # Step 3: Prepare dataset with fault-tolerant parallel processing
    logging.info("\n" + "="*60)
    logging.info("Step 3: Preparing MLM dataset with resilient caching")
    logging.info("="*60)
    
    dataset = StreamingMLMDataset(
        tokenizer=tokenizer,
        max_length=512,             # Standard sequence length
        mlm_probability=0.15,
        max_samples=20_000_000,         # Increased for v2 training
        batch_size=25000,            # Process 25000 examples per batch
        num_workers=56,             # Use all 56 cores for parallel processing
        cache_dir="./temp_data/v2_dataset_cache"  # Separate cache for v2
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
    
    # Smaller batch size due to model complexity
    batch_size = 2 if torch.cuda.is_available() else 1
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    logging.info(f"Dataset size: {len(dataset)} examples")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Steps per epoch: {len(dataloader)}")
    
    # Step 4: Train
    logging.info("\n" + "="*60)
    logging.info("Step 4: Training TORMENTED-BERT-Frankenstein")
    logging.info("="*60)
    
    # Configure training behavior
    training_config = TrainingConfig(
        csv_log_path="training_metrics.csv",      # CSV file for all metrics
        checkpoint_every_n_steps=500,             # Rolling checkpoint frequency
        max_rolling_checkpoints=3,                # Keep only 3 rolling checkpoints
        num_best_checkpoints=2,                   # Keep top 2 best models
        nan_check_interval=10,                    # Check NaN every 10 steps
        log_gradient_stats=True,
        gradient_log_interval=10
    )
    
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
                test_input = test_input.cuda()
            
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