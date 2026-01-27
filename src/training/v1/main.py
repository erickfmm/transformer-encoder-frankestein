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