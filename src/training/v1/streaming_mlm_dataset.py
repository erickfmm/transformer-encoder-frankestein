
from typing import Tuple, List, Dict
import logging
from datasets import load_dataset
import random
from torch.utils.data import Dataset as TorchDataset
import torch

from utils.storage_manager import StorageManager
from tokenizer.spm_spa_redpajama35 import SpanishSPMTokenizer

# ==================== DATASET & TRAINING ====================
class StreamingMLMDataset(TorchDataset):
    """Streaming MLM dataset with storage management"""
    
    def __init__(self, tokenizer: SpanishSPMTokenizer, max_length: int = 512,
                 mlm_probability: float = 0.15, max_samples: int = 1000000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.max_samples = max_samples
        self.storage_manager = StorageManager()
        
        # Cache file for processed examples
        self.cache_file = self.storage_manager.create_temp_file(suffix=".cache")
        self.examples = self._prepare_examples()
    
    def _prepare_examples(self) -> List[Dict]:
        """Prepare examples with streaming"""
        examples = []
        
        try:
            dataset = load_dataset(
                "erickfmm/red_pajama_es_hq_35",
                split="train",
                streaming=False
            )
            
            logging.info("Preparing MLM examples...")
            count = 0
            
            for example in dataset:
                if count >= self.max_samples:
                    break
                
                if 'text' in example:
                    text = example['text'].strip()
                    if len(text) > 10:  # Skip very short texts
                        encoded = self.tokenizer.encode(text, self.max_length)
                        mlm_input, mlm_labels = self._apply_mlm_mask(
                            encoded['input_ids']
                        )
                        
                        examples.append({
                            'input_ids': mlm_input,
                            'attention_mask': encoded['attention_mask'],
                            'labels': mlm_labels,
                        })
                        
                        count += 1
                        if count % 10000 == 0:
                            logging.info(f"Prepared {count} examples")
                
                # Check storage
                if not self.storage_manager.register_file(self.cache_file):
                    logging.warning("Storage limit reached during data preparation")
                    break
            
            # Cache to disk
            with open(self.cache_file, 'wb') as f:
                import pickle
                pickle.dump(examples, f)
            
        except Exception as e:
            logging.error(f"Error preparing dataset: {e}")
            # Create synthetic data
            examples = self._create_synthetic_examples(10000)
        
        return examples
    
    def _apply_mlm_mask(self, input_ids: List[int]) -> Tuple[List[int], List[int]]:
        """Apply MLM masking"""
        labels = input_ids.copy()
        special_tokens = {0, 1, 2, 3}  # PAD, UNK, CLS, SEP
        
        # Find maskable positions
        maskable_positions = [
            i for i, token_id in enumerate(input_ids)
            if token_id not in special_tokens
        ]
        
        num_to_mask = max(1, int(len(maskable_positions) * self.mlm_probability))
        masked_positions = random.sample(maskable_positions, 
                                       min(num_to_mask, len(maskable_positions)))
        
        for pos in masked_positions:
            # 80%: [MASK], 10%: random, 10%: unchanged
            rand = random.random()
            if rand < 0.8:
                input_ids[pos] = 3  # [MASK]
            elif rand < 0.9:
                input_ids[pos] = random.randint(4, self.tokenizer.vocab_size - 1)
        
        return input_ids, labels
    
    def _create_synthetic_examples(self, num_examples: int) -> List[Dict]:
        """Create synthetic examples for testing"""
        examples = []
        spanish_words = ["hola", "mundo", "español", "idioma", "aprender", 
                        "datos", "modelo", "entrenamiento"]
        
        for _ in range(num_examples):
            text = " ".join(random.choices(spanish_words, k=random.randint(10, 50)))
            encoded = self.tokenizer.encode(text, self.max_length)
            
            mlm_input, mlm_labels = self._apply_mlm_mask(encoded['input_ids'])
            
            examples.append({
                'input_ids': mlm_input,
                'attention_mask': encoded['attention_mask'],
                'labels': mlm_labels,
            })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            'input_ids': torch.tensor(example['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(example['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(example['labels'], dtype=torch.long),
        }
