#!/usr/bin/env python3
"""
Test script to demonstrate fault-tolerant dataset recovery
Shows how the StreamingMLMDataset can recover from interruptions
"""

import logging
import sys
from pathlib import Path

from tokenizer.spm_spa_redpajama35 import SpanishSPMTokenizer
from training.v1.streaming_mlm_dataset import StreamingMLMDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_recovery():
    """Test recovery from cache"""
    
    logging.info("="*60)
    logging.info("FAULT-TOLERANT DATASET RECOVERY TEST")
    logging.info("="*60)
    
    # Check if we have a trained tokenizer model
    model_path = "es_redpajama_50k.model"
    
    if not Path(model_path).exists():
        logging.info("\n1. Training tokenizer (first time only)...")
        logging.info("   Using up to 100GB RAM for maximum data quality...")
        tokenizer = SpanishSPMTokenizer(vocab_size=50000)
        tokenizer.train(
            model_prefix="es_redpajama_50k",
            max_training_samples=50_000_000,  # Up to 50M samples
            target_ram_gb=100.0  # Use up to 100GB RAM
        )
    else:
        logging.info(f"\n1. Loading existing tokenizer from {model_path}...")
        tokenizer = SpanishSPMTokenizer(model_path=model_path)
    
    # Create dataset with small batch size for testing
    logging.info("\n2. Creating dataset with fault-tolerant caching...")
    logging.info("   - This will resume from cache if available")
    logging.info("   - Try interrupting (Ctrl+C) and running again")
    
    cache_dir = "./temp_data/test_recovery_cache"
    
    dataset = StreamingMLMDataset(
        tokenizer=tokenizer,
        max_length=512,
        mlm_probability=0.15,
        max_samples=50000,  # Smaller for testing
        batch_size=2000,  # 2000 examples per batch = ~25 batches total
        num_workers=8,  # Use fewer workers for testing
        cache_dir=cache_dir
    )
    
    # Display statistics
    logging.info("\n3. Dataset Statistics:")
    stats = dataset.get_stats()
    for key, value in stats.items():
        logging.info(f"   {key}: {value}")
    
    # Check cache status
    cache_path = Path(cache_dir)
    if cache_path.exists():
        metadata_file = cache_path / "metadata.json"
        batch_files = list(cache_path.glob("batch_*.pkl"))
        
        logging.info(f"\n4. Cache Status:")
        logging.info(f"   Cache directory: {cache_dir}")
        logging.info(f"   Metadata exists: {metadata_file.exists()}")
        logging.info(f"   Batch files: {len(batch_files)}")
        
        if batch_files:
            logging.info(f"   Cached batches:")
            for batch_file in sorted(batch_files)[:10]:  # Show first 10
                size_mb = batch_file.stat().st_size / (1024**2)
                logging.info(f"      - {batch_file.name} ({size_mb:.2f} MB)")
            if len(batch_files) > 10:
                logging.info(f"      ... and {len(batch_files) - 10} more")
    
    # Sample some data
    logging.info(f"\n5. Sample Data:")
    logging.info(f"   Total examples: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        logging.info(f"   First example keys: {sample.keys()}")
        logging.info(f"   Input shape: {sample['input_ids'].shape}")
        logging.info(f"   Attention mask shape: {sample['attention_mask'].shape}")
        logging.info(f"   Labels shape: {sample['labels'].shape}")
    
    logging.info("\n" + "="*60)
    logging.info("RECOVERY TEST COMPLETE")
    logging.info("="*60)
    logging.info("\nHow to test fault tolerance:")
    logging.info("1. Run this script")
    logging.info("2. Interrupt it with Ctrl+C during processing")
    logging.info("3. Run it again - it will resume from cache!")
    logging.info("\nTo clean cache and start fresh:")
    logging.info("   dataset.cleanup_cache()")
    logging.info(f"   Or manually delete: {cache_dir}")

if __name__ == "__main__":
    try:
        test_recovery()
    except KeyboardInterrupt:
        logging.info("\n\n" + "="*60)
        logging.info("INTERRUPTED! Run again to resume from cache.")
        logging.info("="*60)
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
