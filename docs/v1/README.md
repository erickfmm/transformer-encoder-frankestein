# Spanish MoE-BERT: An Advanced Transformer for Spanish NLP

A state-of-the-art BERT encoder model, built with a Sparse Mixture of Experts (MoE) architecture, specifically pre-trained on high-quality Spanish text. This model features advanced attention mechanisms, Rotary Position Embeddings (RoPE), and a custom SentencePiece tokenizer, optimized for similarity tasks and compatible with the Sentence Transformers library.

**Paper:** [Spanish MoE-BERT: A Mixture-of-Experts BERT Model for Efficient Spanish Language Understanding](papers.example.com) | **Model:** [Hugging Face Hub Link](#) | **Dataset:** [erickfmm/red_pajama_es_hq_35](https://huggingface.co/datasets/erickfmm/red_pajama_es_hq_35)

## 📋 Table of Contents
- [✨ Model Highlights](#-model-highlights)
- [🏗️ Model Architecture](#️-model-architecture)
- [📊 Dataset](#-dataset)
- [🚀 Quick Start](#-quick-start)
- [⚙️ Training](#️-training)
- [📈 Performance & Usage](#-performance--usage)
- [📝 Citation](#-citation)
- [📄 License](#-license)

---

## ✨ Model Highlights

This project implements a modern BERT-like encoder with several key innovations designed for efficiency and performance on Spanish language tasks:

*   **Parameter-Efficient Scaling (~340M params)**: Achieves a scale similar to BERT-Large through a **Sparse Mixture-of-Experts (MoE)** layer, where a router dynamically selects a subset of expert networks (e.g., top-4 out of 32) for each input token. This allows for a large model capacity without a proportional increase in computation.
*   **Optimized for Spanish**: Pre-trained from scratch on the `erickfmm/red_pajama_es_hq_35` dataset, a curated Spanish subset of the massive RedPajama-V2 corpus, using a custom **50k vocabulary SentencePiece tokenizer**.
*   **Advanced Architectural Features**:
    *   **Mixed Attention Types**: Combines Grouped Query Attention (GQA), standard multi-head attention, and latent attention across different layers.
    *   **RoPE Embeddings**: Uses Rotary Position Embeddings for better generalization to sequence length.
    *   **SwiGLU Activation**: Replaces GELU with the SwiGLU activation function in feed-forward networks.
    *   **Dynamic Tanh Normalization**: An experimental alternative to LayerNorm inspired by Yann LeCun's work.
*   **Production & Resource Aware**: The training script includes a `StorageManager` to stay under a 300GB disk budget and is structured for clarity and extensibility.

## 🏗️ Model Architecture

The model is a Transformer encoder. Below is a summary of its core components and configuration:

| Component | Implementation Details |
| :--- | :--- |
| **Architecture** | Transformer Encoder (BERT-like) with Sparse MoE |
| **Hidden Size** | 1024 |
| **Layers** | 24 |
| **Attention Heads** | 16 |
| **Total Parameters** | ~340 Million |
| **Vocabulary Size** | 50,000 (SentencePiece BPE) |
| **Max Sequence Length** | 512 |
| **Positional Encoding** | Rotary Position Embeddings (RoPE) |
| **MoE Experts** | 32 experts (top-4 activated) |
| **Activation** | SwiGLU |
| **Normalization** | Dynamic Tanh Normalization |

## 📊 Dataset

The model is pre-trained using the **Masked Language Modeling (MLM)** objective on a Spanish corpus.

*   **Primary Dataset**: `erickfmm/red_pajama_es_hq_35`
*   **Source**: A high-quality Spanish subset filtered from the RedPajama-Data-V2 project.
*   **RedPajama-V2 Scale**: The full dataset contains over 100 billion text documents and 30 trillion tokens across 5 languages, with extensive quality annotations.
*   **Content**: The dataset consists of diverse web text in Spanish, processed and filtered for quality.
*   **Training Data Management**: The provided `StreamingMLMDataset` class loads data in a memory-efficient streaming mode and includes a storage-aware caching system to respect disk space limits.

## 🚀 Quick Start

### Installation

1.  **Create and activate a Python environment** (Python 3.8+ is recommended).
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
2.  **Install dependencies using the provided `pyproject.toml`.** We recommend using `uv` for fast, reproducible installs.
    ```bash
    # Install uv (if you don't have it)
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Install the project and its core dependencies
    uv pip install -e .

    # For training, install the optional dependencies
    uv pip install -e ".[train]"
    ```
    *Core dependencies include PyTorch, SentencePiece, Hugging Face `datasets`, and `transformers*.

### Using the Pre-trained Model (Example)

Load the model and tokenizer for inference:

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load the tokenizer and model
tokenizer = SpanishSPMTokenizer.from_pretrained("./models/es_redpajama_50k")
model = SpanishMoEBERT.from_pretrained("./models/spanish_moe_bert")

# Or, if uploaded to the Hugging Face Hub:
# tokenizer = AutoTokenizer.from_pretrained("your-username/spanish-moe-bert-50k")
# model = AutoModel.from_pretrained("your-username/spanish-moe-bert-50k")

# Prepare input
text = "El aprendizaje automático es fascinante."
inputs = tokenizer.encode(text, return_tensors="pt") # Returns dict with 'input_ids', 'attention_mask'

# Get model outputs
with torch.no_grad():
    outputs = model(**inputs)
    sentence_embedding = outputs.pooler_output  # [CLS] token representation for similarity
    # hidden_states = outputs.last_hidden_state  # All token representations
```

### For Sentence Similarity with Sentence Transformers

The model outputs a `pooler_output` suitable for cosine similarity. You can wrap it for use with the Sentence Transformers library:

```python
from sentence_transformers import SentenceTransformer, util

# Assume you have a wrapper or the model is compatible
model = SentenceTransformer('path_to_your_model_or_wrapper')
sentences = ["Texto de ejemplo uno.", "Un segundo texto de ejemplo."]

embeddings = model.encode(sentences)
similarity = util.cos_sim(embeddings[0], embeddings[1])
print(f"Similarity: {similarity.item():.4f}")
```

## ⚙️ Training

### 1. Train the SentencePiece Tokenizer
This step creates a custom 50k vocabulary tokenizer from the Spanish corpus.
```bash
python -c "
from spanish_moe_tokenizer import SpanishSPMTokenizer
tokenizer = SpanishSPMTokenizer(vocab_size=50000)
tokenizer.train(model_prefix='models/es_redpajama_50k')
"
```

### 2. Run the Pre-training Script
The main script handles dataset streaming, MLM, and checkpointing.
```bash
python train_spanish_moe.py
```
**Key Training Features**:
*   **Storage Management**: Script monitors and warns if disk usage approaches the 300GB limit.
*   **Mixed Precision**: Uses `torch.cuda.amp` for faster training and reduced memory usage on compatible GPUs.
*   **Gradient Accumulation**: Allows for effective larger batch sizes on GPUs with limited VRAM (like the Tesla P40).
*   **Checkpointing**: Saves model and optimizer states periodically for resilience.

## 📈 Performance & Usage

### Intended Use
*   **Semantic Textual Similarity (STS)**: The primary use case, leveraging the `pooler_output`.
*   **Information Retrieval**: As a dense retriever encoder.
*   **Feature Extraction**: Using the `last_hidden_state` for downstream tasks like classification or NER (requires fine-tuning).
*   **Fill-Mask**: The model includes an MLM head for masked token prediction.

### Hardware Requirements
*   **Training**: The code is optimized for a setup similar to:
    *   **GPU**: 1x NVIDIA Tesla P40 (20GB VRAM, FP32). Using FP16/BF16 is recommended for other GPUs.
    *   **CPU**: Multi-core (e.g., 2x E5-2680 v4, 56 threads) for fast data loading.
    *   **RAM**: 120GB+.
    *   **Disk**: <300GB of free space for datasets, tokenizers, and checkpoints.
*   **Inference**: Requires significantly less resources. Can run on a modern CPU or a much smaller GPU.

### Limitations
*   **Language**: Specialized for Spanish. Multilingual ability has not been evaluated.
*   **Bias**: As with all models trained on web data, it may reflect societal biases present in the source corpus.
*   **Domain**: Best performance is expected on general web and textual domain data similar to its training corpus.

## 📝 Citation

If you use this model, code, or dataset in your research, please cite the relevant sources:

```bibtex
@software{SpanishMoEBERT2025,
  author = {Your Name},
  title = {Spanish MoE-BERT: A Mixture-of-Experts BERT Model for Efficient Spanish Language Understanding},
  year = {2025},
  url = {https://github.com/your-username/spanish-moe-bert}
}

@dataset{redpajama_es_35,
  author = {erickfmm},
  title = {red\_pajama\_es\_hq\_35},
  year = {2024},
  url = {https://huggingface.co/datasets/erickfmm/red_pajama_es_hq_35}
}

@article{together2023redpajamav2,
  title={RedPajama-Data-V2: An Open Dataset with 30 Trillion Tokens for Training Large Language Models},
  author={Together Computer},
  journal={Together AI Blog},
  year={2023},
  url = {https://www.together.ai/blog/redpajama-data-v2}
}
```

## 📄 License

The model and code are intended for research purposes and are released under the [Apache License 2.0](LICENSE).