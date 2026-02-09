"""
SBERT Module for TORMENTED-BERT-Frankenstein

This module provides Sentence-BERT fine-tuning and inference
for different versions of the TORMENTED-BERT model.
"""

# v2 is the current active version
from .v2 import (
    SBERTInference,
    SimilarityResult,
    TormentedBertSentenceTransformer,
    SBERTTrainer
)

__all__ = [
    'SBERTInference',
    'SimilarityResult',
    'TormentedBertSentenceTransformer',
    'SBERTTrainer'
]

__version__ = '2.0.0'
