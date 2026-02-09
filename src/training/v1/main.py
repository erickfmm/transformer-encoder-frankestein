#!/usr/bin/env python3
"""
Spanish MoE-BERT Trainer with SPM Tokenizer
Dataset: erickfmm/red_pajama_es_hq_35
Vocabulary: 50,000 tokens
Storage limit: <300GB
"""

import os
import logging
from torch.utils.data import DataLoader

from tokenizer.spm_spa_redpajama35 import SpanishSPMTokenizer
from training.v1.streaming_mlm_dataset import StreamingMLMDataset
from training.v1.trainer import Trainer
from model.v1.model import SpanishMoEBERT, ModelConfig

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
    
    # Step 1: Train/Load tokenizer
    logging.info("\n" + "="*50)
    logging.info("Step 1: Training/Loading SPM tokenizer (50k vocab)")
    logging.info("="*50)
    
    # Check if tokenizer already exists to avoid retraining
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
            target_ram_gb=100.0  # Use up to 100GB RAM
        )
    
    logging.info(f"Tokenizer loaded with {len(tokenizer.vocab)} tokens")
    
    # Step 2: Create model
    logging.info("\n" + "="*50)
    logging.info("Step 2: Creating Spanish MoE-BERT model")
    logging.info("="*50)
    
    config = ModelConfig(vocab_size=50000)
    model = SpanishMoEBERT(config, tokenizer)
    
    # Step 3: Prepare dataset
    logging.info("\n" + "="*50)
    logging.info("Step 3: Preparing MLM dataset with fault-tolerant caching")
    logging.info("="*50)
    
    dataset = StreamingMLMDataset(
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        mlm_probability=0.15,
        max_samples=100000,  # Reduced for storage
        batch_size=5000,  # Process 5000 examples per batch
        num_workers=56,  # Use all 56 cores for parallel processing
        cache_dir="./temp_data/dataset_cache"  # Persistent cache directory
    )
    
    # Show dataset statistics
    stats = dataset.get_stats()
    logging.info(f"Dataset stats: {stats}")
    logging.info(f"  - Total examples: {stats['total_examples']}")
    logging.info(f"  - Completed batches: {stats['completed_batches']}")
    logging.info(f"  - Samples processed: {stats['total_samples_processed']}")
    logging.info(f"  - Workers used: {stats['num_workers']}")
    logging.info(f"  - Cache directory: {stats['cache_dir']}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # Small batch for memory
        shuffle=True,
        num_workers=2,  # DataLoader workers (different from dataset workers)
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
    
    # Cleanup (preserving dataset cache for future runs)
    tokenizer.storage_manager.cleanup()
    # Note: NOT cleaning dataset cache to allow recovery on next run
    # To clean dataset cache manually if needed: dataset.cleanup_cache()
    trainer.storage_manager.cleanup()
    
    logging.info("Tip: Dataset cache preserved at ./temp_data/dataset_cache")
    logging.info("     If you restart training, it will resume from existing cache")

if __name__ == "__main__":
    # Check for psutil (optional)
    try:
        import psutil
    except ImportError:
        logging.warning("psutil not installed, storage monitoring limited")
        psutil = None
    
    main()