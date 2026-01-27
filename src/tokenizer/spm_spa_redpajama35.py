import os
from typing import List, Dict
import logging
import sentencepiece as spm
from datasets import load_dataset

from utils.storage_manager import StorageManager

# ==================== SPM TOKENIZER ====================
class SpanishSPMTokenizer:
    """Spanish SentencePiece Tokenizer trained on RedPajama"""
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.sp_model = None
        self.vocab = {}
        self.inverse_vocab = {}
        self.storage_manager = StorageManager()
        
    def prepare_training_data(self, max_samples: int = 500000) -> str:
        """
        Prepare training data from erickfmm/red_pajama_es_hq_35
        Using streaming to avoid downloading full dataset
        """
        logging.info("Loading Spanish RedPajama dataset (streaming mode)...")
        
        # Create temp file for training data
        temp_file = self.storage_manager.create_temp_file(suffix=".txt")
        
        try:
            # Load dataset in streaming mode
            dataset = load_dataset(
                "erickfmm/red_pajama_es_hq_35",
                split="train",
                streaming=True
            )
            
            count = 0
            with open(temp_file, 'w', encoding='utf-8') as f:
                iterator = iter(dataset)
                
                while count < max_samples:
                    try:
                        example = next(iterator)
                        if 'text' in example:
                            text = example['text'].strip()
                            if len(text) > 100:  # Skip very short texts
                                f.write(text + '\n')
                                count += 1
                                
                                if count % 10000 == 0:
                                    logging.info(f"Collected {count} samples...")
                                    # Check storage
                                    if not self.storage_manager.register_file(temp_file):
                                        logging.warning("Storage limit approached, stopping collection")
                                        break
                    except StopIteration:
                        break
                
            logging.info(f"Created training file with {count} samples")
            return temp_file
            
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            # Create fallback training data
            with open(temp_file, 'w', encoding='utf-8') as f:
                for i in range(10000):
                    f.write(f"Texto de ejemplo en español número {i}. ")
                    f.write("Este es un texto para entrenar el tokenizador. ")
                    f.write("El aprendizaje automático requiere datos de calidad.\n")
            return temp_file
    
    def train(self, model_prefix: str = "es_spm_50k"):
        """Train SentencePiece tokenizer with 50k vocabulary"""
        train_file = self.prepare_training_data()
        
        # SPM training arguments
        spm_args = [
            f'--input={train_file}',
            f'--model_prefix={model_prefix}',
            f'--vocab_size={self.vocab_size}',
            '--character_coverage=0.9995',
            '--model_type=bpe',
            '--max_sentence_length=16384',
            '--pad_id=0',
            '--unk_id=1',
            '--bos_id=2',
            '--eos_id=3',
            '--user_defined_symbols=[CLS],[SEP],[MASK],[PAD]',
            '--split_by_whitespace=true',
            '--normalization_rule_name=nmt_nfkc',
            '--add_dummy_prefix=true',
            '--byte_fallback=true',
            '--split_digits=true',
            f'--num_threads={os.cpu_count()//2}',  # Use half the cores
            '--input_sentence_size=1000000',
            '--shuffle_input_sentence=true',
        ]
        
        logging.info("Training SentencePiece tokenizer...")
        spm.SentencePieceTrainer.train(' '.join(spm_args))
        
        # Load trained model
        model_path = f"{model_prefix}.model"
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)
        
        # Build vocab
        for i in range(self.sp_model.GetPieceSize()):
            token = self.sp_model.IdToPiece(i)
            self.vocab[token] = i
            self.inverse_vocab[i] = token
        
        logging.info(f"Tokenizer trained with {len(self.vocab)} tokens")
        return model_path
    
    def encode(self, text: str, max_length: int = 512) -> Dict[str, List[int]]:
        """Encode text with special tokens"""
        tokens = self.sp_model.encode_as_ids(text)
        
        # Add special tokens
        tokens = [self.vocab['[CLS]']] + tokens[:max_length-2] + [self.vocab['[SEP]']]
        
        # Pad or truncate
        if len(tokens) < max_length:
            tokens = tokens + [self.vocab['[PAD]']] * (max_length - len(tokens))
            attention_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
            attention_mask = [1] * max_length
        
        return {
            'input_ids': tokens,
            'attention_mask': attention_mask[:max_length]
        }
