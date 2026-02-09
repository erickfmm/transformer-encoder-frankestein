
from typing import Tuple, List, Dict, Optional
import logging
from datasets import load_dataset
import random
from torch.utils.data import Dataset as TorchDataset
import torch
import json
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

from utils.storage_manager import StorageManager
from tokenizer.spm_spa_redpajama35 import SpanishSPMTokenizer


# ==================== PARALLEL PROCESSING FUNCTIONS ====================
def _process_single_example(args):
    """Process a single example (tokenize + MLM). Used for parallel processing."""
    text, tokenizer_model_path, max_length, mlm_probability, vocab_size = args
    
    # Recreate tokenizer in worker process
    tokenizer = SpanishSPMTokenizer(model_path=tokenizer_model_path)
    
    try:
        encoded = tokenizer.encode(text, max_length)
        mlm_input, mlm_labels = _apply_mlm_mask_standalone(
            encoded['input_ids'], mlm_probability, vocab_size
        )
        
        return {
            'input_ids': mlm_input,
            'attention_mask': encoded['attention_mask'],
            'labels': mlm_labels,
        }
    except Exception as e:
        logging.warning(f"Error processing example: {e}")
        return None


def _apply_mlm_mask_standalone(input_ids: List[int], mlm_probability: float, 
                                vocab_size: int) -> Tuple[List[int], List[int]]:
    """Apply MLM masking (standalone function for multiprocessing)"""
    labels = input_ids.copy()
    special_tokens = {0, 1, 2, 3}  # PAD, UNK, CLS, SEP
    
    # Find maskable positions
    maskable_positions = [
        i for i, token_id in enumerate(input_ids)
        if token_id not in special_tokens
    ]
    
    if not maskable_positions:
        return input_ids, labels
    
    num_to_mask = max(1, int(len(maskable_positions) * mlm_probability))
    masked_positions = random.sample(maskable_positions, 
                                   min(num_to_mask, len(maskable_positions)))
    
    for pos in masked_positions:
        # 80%: [MASK], 10%: random, 10%: unchanged
        rand = random.random()
        if rand < 0.8:
            input_ids[pos] = 3  # [MASK]
        elif rand < 0.9:
            input_ids[pos] = random.randint(4, vocab_size - 1)
    
    return input_ids, labels


# ==================== DATASET & TRAINING ====================
class StreamingMLMDataset(TorchDataset):
    """Streaming MLM dataset with fault-tolerant parallel processing"""
    
    def __init__(self, tokenizer: SpanishSPMTokenizer, max_length: int = 512,
                 mlm_probability: float = 0.15, max_samples: int = 1000000,
                 batch_size: int = 5000, num_workers: Optional[int] = None,
                 cache_dir: Optional[str] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.num_workers = num_workers or min(56, mp.cpu_count() - 2)  # Leave 2 cores free
        self.storage_manager = StorageManager()
        
        # Setup cache directory structure
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path("./temp_data/dataset_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.cache_dir / "metadata.json"
        self.progress_file = self.cache_dir / "progress.json"
        
        # Load or initialize metadata
        self.metadata = self._load_metadata()
        
        # Prepare or load examples
        self.examples = self._prepare_examples()
    
    def _load_metadata(self) -> Dict:
        """Load or initialize metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                logging.info(f"Loaded metadata: {len(metadata.get('completed_batches', []))} batches completed")
                return metadata
            except Exception as e:
                logging.warning(f"Error loading metadata: {e}, creating new")
        
        return {
            'version': '2.0',
            'total_samples_target': self.max_samples,
            'total_samples_processed': 0,
            'completed_batches': [],
            'batch_size': self.batch_size,
            'last_batch_id': -1,
        }
    
    def _save_metadata(self):
        """Save metadata to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving metadata: {e}")
    
    def _get_batch_file(self, batch_id: int) -> Path:
        """Get path for a batch file"""
        return self.cache_dir / f"batch_{batch_id:05d}.pkl"
    
    def _load_batch(self, batch_id: int) -> List[Dict]:
        """Load a batch from disk"""
        batch_file = self._get_batch_file(batch_id)
        if not batch_file.exists():
            return []
        
        try:
            with open(batch_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading batch {batch_id}: {e}")
            return []
    
    def _save_batch(self, batch_id: int, examples: List[Dict]):
        """Save a batch to disk"""
        batch_file = self._get_batch_file(batch_id)
        try:
            with open(batch_file, 'wb') as f:
                pickle.dump(examples, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except Exception as e:
            logging.error(f"Error saving batch {batch_id}: {e}")
            return False
    
    def _load_existing_batches(self) -> List[Dict]:
        """Load all existing completed batches"""
        examples = []
        completed_batches = sorted(self.metadata.get('completed_batches', []))
        
        logging.info(f"Loading {len(completed_batches)} existing batches...")
        for batch_id in completed_batches:
            batch_examples = self._load_batch(batch_id)
            examples.extend(batch_examples)
            
            if len(examples) % 50000 == 0:
                logging.info(f"Loaded {len(examples)} examples so far...")
        
        logging.info(f"Loaded {len(examples)} examples from {len(completed_batches)} batches")
        return examples
    
    def _prepare_examples(self) -> List[Dict]:
        """Prepare examples with fault-tolerant parallel processing"""
        # Load existing batches
        examples = self._load_existing_batches()
        samples_processed = len(examples)
        
        # Check if we already have enough samples
        if samples_processed >= self.max_samples:
            logging.info(f"Already have {samples_processed} samples, skipping data preparation")
            return examples[:self.max_samples]
        
        logging.info(f"Starting data preparation from {samples_processed} samples...")
        logging.info(f"Using {self.num_workers} parallel workers")
        
        try:
            dataset = load_dataset(
                "erickfmm/red_pajama_es_hq_35",
                split="train",
                streaming=True
            )
            
            # Skip already processed examples
            dataset_iter = iter(dataset)
            if samples_processed > 0:
                logging.info(f"Skipping first {samples_processed} examples...")
                for _ in range(samples_processed):
                    try:
                        next(dataset_iter)
                    except StopIteration:
                        logging.warning("Dataset exhausted during skip")
                        return examples
            
            # Process in batches with parallel workers
            current_batch_id = self.metadata.get('last_batch_id', -1) + 1
            batch_texts = []
            
            while samples_processed < self.max_samples:
                # Collect texts for batch
                try:
                    example = next(dataset_iter)
                    if 'text' in example:
                        text = example['text'].strip()
                        if len(text) > 10:
                            batch_texts.append(text)
                            
                            # Process batch when full
                            if len(batch_texts) >= self.batch_size:
                                new_examples = self._process_batch_parallel(
                                    batch_texts, current_batch_id
                                )
                                
                                if new_examples:
                                    examples.extend(new_examples)
                                    samples_processed += len(new_examples)
                                    
                                    # Save batch and update metadata
                                    if self._save_batch(current_batch_id, new_examples):
                                        self.metadata['completed_batches'].append(current_batch_id)
                                        self.metadata['last_batch_id'] = current_batch_id
                                        self.metadata['total_samples_processed'] = samples_processed
                                        self._save_metadata()
                                        
                                        logging.info(
                                            f"Batch {current_batch_id} completed: "
                                            f"{len(new_examples)} examples, "
                                            f"total: {samples_processed}/{self.max_samples}"
                                        )
                                    
                                    current_batch_id += 1
                                
                                batch_texts = []
                                
                                # Check storage
                                if not self.storage_manager.register_file(str(self.cache_dir)):
                                    logging.warning("Storage limit reached")
                                    break
                                
                except StopIteration:
                    logging.info("Dataset exhausted")
                    break
                except Exception as e:
                    logging.error(f"Error reading from dataset: {e}")
                    break
            
            # Process remaining texts in last batch if any
            if batch_texts and samples_processed < self.max_samples:
                new_examples = self._process_batch_parallel(batch_texts, current_batch_id)
                if new_examples:
                    examples.extend(new_examples)
                    samples_processed += len(new_examples)
                    
                    if self._save_batch(current_batch_id, new_examples):
                        self.metadata['completed_batches'].append(current_batch_id)
                        self.metadata['last_batch_id'] = current_batch_id
                        self.metadata['total_samples_processed'] = samples_processed
                        self._save_metadata()
                        
                        logging.info(
                            f"Final batch {current_batch_id} completed: "
                            f"{len(new_examples)} examples, total: {samples_processed}"
                        )
            
            logging.info(f"Data preparation completed: {samples_processed} total examples")
            
        except Exception as e:
            logging.error("="*60)
            logging.error("🚨 CRITICAL ERROR: Dataset preparation failed")
            logging.error("="*60)
            logging.error(f"Error type: {type(e).__name__}")
            logging.error(f"Error message: {e}")
            logging.error(f"Dataset: erickfmm/red_pajama_es_hq_35")
            logging.error(f"Samples processed before failure: {samples_processed}")
            logging.error(f"Examples collected: {len(examples)}")
            logging.error(f"\nAttempted configuration:")
            logging.error(f"  - max_samples: {self.max_samples:,}")
            logging.error(f"  - batch_size: {self.batch_size:,}")
            logging.error(f"  - num_workers: {self.num_workers}")
            logging.error(f"  - cache_dir: {self.cache_dir}")
            
            # If we have some examples from cache, that's acceptable
            if examples:
                logging.error(f"\n⚠️  {len(examples)} examples available from cache")
                logging.error("Continuing with cached data only (no synthetic fallback)")
                logging.error("="*60)
                return examples[:self.max_samples]
            else:
                logging.error("\n⛔ SYSTEM HALTED - No cached data and no synthetic fallback allowed")
                logging.error("="*60)
                import traceback
                logging.error("\nFull traceback:")
                logging.error(traceback.format_exc())
                raise RuntimeError(
                    f"Failed to prepare dataset and no cached data available. "
                    f"Error: {e}. Check logs for details."
                ) from e
        
        return examples[:self.max_samples]
    
    def _process_batch_parallel(self, texts: List[str], batch_id: int) -> List[Dict]:
        """Process a batch of texts in parallel"""
        if not texts:
            return []
        
        # Prepare arguments for parallel processing
        args_list = [
            (text, self.tokenizer.model_path, self.max_length, 
             self.mlm_probability, self.tokenizer.vocab_size)
            for text in texts
        ]
        
        results = []
        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(_process_single_example, args): idx 
                    for idx, args in enumerate(args_list)
                }
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        logging.warning(f"Error processing example in batch {batch_id}: {e}")
        
        except Exception as e:
            logging.error(f"Error in parallel processing for batch {batch_id}: {e}")
            # Fallback to sequential processing
            logging.info("Falling back to sequential processing...")
            for args in args_list:
                try:
                    result = _process_single_example(args)
                    if result is not None:
                        results.append(result)
                except Exception as ex:
                    logging.warning(f"Error in sequential fallback: {ex}")
        
        return results
    
    def _apply_mlm_mask(self, input_ids: List[int]) -> Tuple[List[int], List[int]]:
        """Apply MLM masking (kept for compatibility)"""
        return _apply_mlm_mask_standalone(input_ids, self.mlm_probability, 
                                         self.tokenizer.vocab_size)
    
    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        return {
            'total_examples': len(self.examples),
            'completed_batches': len(self.metadata.get('completed_batches', [])),
            'total_samples_processed': self.metadata.get('total_samples_processed', 0),
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'cache_dir': str(self.cache_dir),
        }
    
    def cleanup_cache(self):
        """Clean up all cache files (use with caution!)"""
        import shutil
        if self.cache_dir.exists():
            try:
                shutil.rmtree(self.cache_dir)
                logging.info(f"Cleaned up cache directory: {self.cache_dir}")
            except Exception as e:
                logging.error(f"Error cleaning up cache: {e}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            'input_ids': torch.tensor(example['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(example['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(example['labels'], dtype=torch.long),
        }
