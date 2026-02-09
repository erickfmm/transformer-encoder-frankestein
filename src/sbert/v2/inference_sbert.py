#!/usr/bin/env python3
"""
SBERT Inference for TORMENTED-BERT-Frankenstein
Compute sentence embeddings and similarity scores using fine-tuned SBERT model.
"""

import os
import sys
import torch
import logging
import numpy as np
from pathlib import Path
from typing import List, Union, Optional, Tuple
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """Result of similarity computation"""
    sentence1: str
    sentence2: str
    similarity: float
    distance: float
    
    def to_dict(self):
        return {
            'sentence1': self.sentence1,
            'sentence2': self.sentence2,
            'similarity': self.similarity,
            'distance': self.distance
        }


class SBERTInference:
    """Inference engine for SBERT models"""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Initialize SBERT inference engine.
        
        Args:
            model_path: Path to fine-tuned SBERT model
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for encoding
        """
        self.model_path = model_path
        self.batch_size = batch_size
        
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model
        logger.info(f"Loading SBERT model from {model_path}")
        self.model = SentenceTransformer(model_path, device=self.device)
        logger.info(f"Model loaded on device: {self.device}")
        
        # Get model info
        self.max_seq_length = self.model.max_seq_length
        logger.info(f"Max sequence length: {self.max_seq_length}")
    
    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode sentences into embeddings.
        
        Args:
            sentences: Single sentence or list of sentences
            batch_size: Batch size (uses default if None)
            show_progress: Show progress bar
            normalize: Normalize embeddings to unit length
        
        Returns:
            Embeddings array of shape (n_sentences, embedding_dim)
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        
        batch_size = batch_size or self.batch_size
        
        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        
        return embeddings
    
    def compute_similarity(
        self,
        sentence1: Union[str, List[str]],
        sentence2: Union[str, List[str]],
        metric: str = "cosine"
    ) -> Union[float, np.ndarray]:
        """
        Compute similarity between sentence(s).
        
        Args:
            sentence1: First sentence(s)
            sentence2: Second sentence(s)
            metric: Similarity metric ('cosine' or 'dot')
        
        Returns:
            Similarity score(s) in range [0, 1] (or [-1, 1] for unnormalized)
        """
        # Encode sentences
        emb1 = self.encode(sentence1, normalize=(metric == "cosine"))
        emb2 = self.encode(sentence2, normalize=(metric == "cosine"))
        
        # Compute similarity
        if metric == "cosine":
            # For normalized embeddings, dot product = cosine similarity
            similarity = np.sum(emb1 * emb2, axis=1) if emb1.ndim > 1 else np.dot(emb1, emb2)
        elif metric == "dot":
            similarity = np.sum(emb1 * emb2, axis=1) if emb1.ndim > 1 else np.dot(emb1, emb2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return similarity
    
    def find_most_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find most similar sentences to query from candidates.
        
        Args:
            query: Query sentence
            candidates: List of candidate sentences
            top_k: Number of top results to return
        
        Returns:
            List of (sentence, similarity_score) tuples, sorted by similarity
        """
        # Encode query and candidates
        query_emb = self.encode(query)
        candidate_embs = self.encode(candidates, show_progress=True)
        
        # Compute similarities
        similarities = np.dot(candidate_embs, query_emb)
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (candidates[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def batch_compare(
        self,
        sentences1: List[str],
        sentences2: List[str]
    ) -> List[SimilarityResult]:
        """
        Batch comparison of sentence pairs.
        
        Args:
            sentences1: First sentences
            sentences2: Second sentences (must match length of sentences1)
        
        Returns:
            List of SimilarityResult objects
        """
        if len(sentences1) != len(sentences2):
            raise ValueError("sentences1 and sentences2 must have same length")
        
        # Encode both sets
        emb1 = self.encode(sentences1, show_progress=True)
        emb2 = self.encode(sentences2, show_progress=True)
        
        # Compute pairwise similarities
        similarities = np.sum(emb1 * emb2, axis=1)
        distances = 1 - similarities
        
        # Create results
        results = [
            SimilarityResult(
                sentence1=s1,
                sentence2=s2,
                similarity=float(sim),
                distance=float(dist)
            )
            for s1, s2, sim, dist in zip(sentences1, sentences2, similarities, distances)
        ]
        
        return results
    
    def semantic_search(
        self,
        queries: Union[str, List[str]],
        corpus: List[str],
        top_k: int = 5
    ) -> List[List[Tuple[int, float]]]:
        """
        Semantic search: find most similar corpus sentences for each query.
        
        Args:
            queries: Query sentence(s)
            corpus: Corpus of sentences to search
            top_k: Number of results per query
        
        Returns:
            List of results for each query, each result is (corpus_idx, score)
        """
        if isinstance(queries, str):
            queries = [queries]
        
        # Encode
        logger.info(f"Encoding {len(queries)} queries...")
        query_embs = self.encode(queries, show_progress=False)
        
        logger.info(f"Encoding corpus of {len(corpus)} sentences...")
        corpus_embs = self.encode(corpus, show_progress=True)
        
        # Compute similarities (queries x corpus)
        logger.info("Computing similarities...")
        similarities = np.dot(query_embs, corpus_embs.T)
        
        # Get top-k for each query
        results = []
        for query_sims in similarities:
            top_indices = np.argsort(query_sims)[::-1][:top_k]
            query_results = [
                (int(idx), float(query_sims[idx]))
                for idx in top_indices
            ]
            results.append(query_results)
        
        return results
    
    def cluster_sentences(
        self,
        sentences: List[str],
        n_clusters: int = 5,
        method: str = "kmeans"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster sentences by semantic similarity.
        
        Args:
            sentences: List of sentences to cluster
            n_clusters: Number of clusters
            method: Clustering method ('kmeans' or 'agglomerative')
        
        Returns:
            (cluster_labels, embeddings)
        """
        from sklearn.cluster import KMeans, AgglomerativeClustering
        
        # Encode sentences
        logger.info(f"Encoding {len(sentences)} sentences for clustering...")
        embeddings = self.encode(sentences, show_progress=True)
        
        # Cluster
        logger.info(f"Clustering into {n_clusters} groups using {method}...")
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == "agglomerative":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        labels = clusterer.fit_predict(embeddings)
        
        return labels, embeddings
    
    def save_embeddings(
        self,
        sentences: List[str],
        output_path: str,
        metadata: Optional[dict] = None
    ):
        """
        Encode sentences and save embeddings to file.
        
        Args:
            sentences: List of sentences
            output_path: Output file path (.npz format)
            metadata: Optional metadata to save
        """
        logger.info(f"Encoding {len(sentences)} sentences...")
        embeddings = self.encode(sentences, show_progress=True)
        
        # Prepare data
        data = {
            'embeddings': embeddings,
            'sentences': np.array(sentences, dtype=object)
        }
        
        if metadata:
            data['metadata'] = json.dumps(metadata)
        
        # Save
        np.savez_compressed(output_path, **data)
        logger.info(f"Embeddings saved to {output_path}")
    
    def load_embeddings(self, input_path: str) -> Tuple[np.ndarray, List[str], Optional[dict]]:
        """
        Load precomputed embeddings.
        
        Args:
            input_path: Path to .npz file
        
        Returns:
            (embeddings, sentences, metadata)
        """
        data = np.load(input_path, allow_pickle=True)
        
        embeddings = data['embeddings']
        sentences = data['sentences'].tolist()
        metadata = json.loads(data['metadata'].item()) if 'metadata' in data else None
        
        logger.info(f"Loaded {len(sentences)} embeddings from {input_path}")
        
        return embeddings, sentences, metadata


def main():
    """Main inference script with CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SBERT Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--mode", type=str, required=True,
                       choices=["similarity", "search", "cluster", "encode"],
                       help="Inference mode")
    
    # Similarity mode
    parser.add_argument("--sentence1", type=str, help="First sentence (similarity mode)")
    parser.add_argument("--sentence2", type=str, help="Second sentence (similarity mode)")
    
    # Search mode
    parser.add_argument("--query", type=str, help="Query sentence (search mode)")
    parser.add_argument("--corpus_file", type=str, help="File with corpus sentences (one per line)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results (search mode)")
    
    # Cluster mode
    parser.add_argument("--sentences_file", type=str, help="File with sentences to cluster")
    parser.add_argument("--n_clusters", type=int, default=5, help="Number of clusters")
    
    # Encode mode
    parser.add_argument("--input_file", type=str, help="Input sentences file")
    parser.add_argument("--output_file", type=str, help="Output embeddings file (.npz)")
    
    # General
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = SBERTInference(
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Execute based on mode
    if args.mode == "similarity":
        if not args.sentence1 or not args.sentence2:
            parser.error("--sentence1 and --sentence2 required for similarity mode")
        
        similarity = inference.compute_similarity(args.sentence1, args.sentence2)
        
        print(f"\nSentence 1: {args.sentence1}")
        print(f"Sentence 2: {args.sentence2}")
        print(f"Similarity: {similarity:.4f}")
    
    elif args.mode == "search":
        if not args.query or not args.corpus_file:
            parser.error("--query and --corpus_file required for search mode")
        
        # Load corpus
        with open(args.corpus_file, 'r', encoding='utf-8') as f:
            corpus = [line.strip() for line in f if line.strip()]
        
        print(f"\nSearching corpus of {len(corpus)} sentences...")
        results = inference.find_most_similar(args.query, corpus, top_k=args.top_k)
        
        print(f"\nQuery: {args.query}")
        print(f"\nTop {args.top_k} results:")
        for i, (sentence, score) in enumerate(results, 1):
            print(f"{i}. [{score:.4f}] {sentence}")
    
    elif args.mode == "cluster":
        if not args.sentences_file:
            parser.error("--sentences_file required for cluster mode")
        
        # Load sentences
        with open(args.sentences_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        
        labels, embeddings = inference.cluster_sentences(sentences, n_clusters=args.n_clusters)
        
        print(f"\nClustered {len(sentences)} sentences into {args.n_clusters} groups:")
        for cluster_id in range(args.n_clusters):
            cluster_sentences = [s for s, l in zip(sentences, labels) if l == cluster_id]
            print(f"\nCluster {cluster_id + 1} ({len(cluster_sentences)} sentences):")
            for sent in cluster_sentences[:5]:  # Show first 5
                print(f"  - {sent}")
            if len(cluster_sentences) > 5:
                print(f"  ... and {len(cluster_sentences) - 5} more")
    
    elif args.mode == "encode":
        if not args.input_file or not args.output_file:
            parser.error("--input_file and --output_file required for encode mode")
        
        # Load sentences
        with open(args.input_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        
        # Save embeddings
        inference.save_embeddings(sentences, args.output_file)


if __name__ == "__main__":
    main()
