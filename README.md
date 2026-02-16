# 🧬 Transformer Encoder Frankenstein

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)](https://huggingface.co/)
[![arXiv](https://img.shields.io/badge/arXiv-Research-b31b1b.svg)](https://arxiv.org/)

> **⚡ Experimental Transformer Architectures for Constrained Hardware**

A research playground for building **unconventional, "Frankenstein" Transformer encoders** that combine cutting-edge techniques like **Mixture of Experts (MoE)**, **BitNet 1.58-bit quantization**, **Neural ODEs**, **RetNet retention mechanisms**, and **Titan neural memory**—all optimized for training on **consumer-grade or legacy GPUs** like the Nvidia Tesla P40.

---

## 📋 Table of Contents

- [✨ Overview](#-overview)
- [🚀 Quick Tour](#-quick-tour)
- [🏗️ Model Versions](#️-model-versions)
- [⚙️ Installation](#️-installation)
- [🎯 Usage](#-usage)
- [📊 Dataset](#-dataset)
- [🔧 Hardware Requirements](#-hardware-requirements)
- [📚 References & Citations](#-references--citations)
- [📜 Terms of Service](#-terms-of-service)
- [📄 License](#-license)
- [🤝 Contributing](#-contributing)

---

## ✨ Overview

**Transformer Encoder Frankenstein** is an experimental encoder-only Transformer model designed to push the boundaries of what's possible on constrained hardware. Instead of following conventional architectures, we combine multiple cutting-edge research ideas into a hybrid "Frankenstein" model.

### Why "Frankenstein"?

Like Dr. Frankenstein's creation, this model is assembled from various "parts":

| Component | Source | Purpose |
|-----------|--------|---------|
| **BitNet b1.58** | Microsoft Research | 3.5x memory reduction via ternary weights |
| **Neural ODEs** | NeurIPS 2018 | Continuous depth modeling |
| **RetNet** | Microsoft Research | RNN-efficient inference with Transformer training |
| **Mixture of Experts** | Google/OpenAI | Parameter-efficient scaling |
| **Titan Memory** | Google DeepMind | Dynamic context extension |
| **RoPE/HOPE** | Meta/Research | Better positional extrapolation |

### Key Features

- 🇪🇸 **Spanish-first**: Pre-trained on high-quality Spanish web corpora
- 💾 **Memory-efficient**: Designed for GPUs with 20-24GB VRAM
- 🧪 **Research-oriented**: Easy to experiment with architectural variants
- 📦 **Self-contained**: Custom tokenizer, model, and training pipeline

---

## 🚀 Quick Tour

```
transformer-encoder-frankestein/
├── 📄 pyproject.toml          # Project configuration & dependencies
├── 📖 README.md               # You are here!
│
├── 📁 docs/
│   └── 📁 v2/                 # Titan-BERT-Ultra documentation
│       ├── README.md          # Full v2 architecture guide
│       ├── diagram.mermaid    # Architecture diagram
│       └── paper.tex          # Research paper draft
│
├── 📁 src/
│   ├── 📁 model/
│   │   └── tormented_bert_frankestein.py  # Tormented-BERT-Frankenstein implementation
│   │
│   ├── 📁 tokenizer/
│   │   └── spm_spa_redpajama35.py  # SentencePiece tokenizer
│   │
│   ├── 📁 training/
│   │   ├── main.py            # Main training script
│   │   ├── trainer.py         # Trainer implementation
│   │   └── streaming_mlm_dataset.py  # Streaming MLM dataset
│   │
│   └── 📁 utils/
│       └── storage_manager.py # Disk budget management
```

---

## 🏗️ Model Versions

### [📗 Tormented-BERT-Frankenstein](docs/v2/README.md)

An **audacious 1.58-bit** Transformer combining Neural ODEs, RetNet, and Titan Memory for extreme efficiency.

| Feature | Specification |
|---------|---------------|
| **Architecture** | Hybrid ODE + RetNet + SSM |
| **Hidden Size** | 2048 |
| **Physical Layers** | 12 |
| **Logical Depth** | 24+ (via recursive looping) |
| **Quantization** | BitNet b1.58 (ternary weights) |
| **ODE Solver** | RK4 (Runge-Kutta 4th order) |
| **Memory Module** | Titan Fast Weights |

**Key Innovations:**
- 🔢 **BitNet b1.58**: Weights constrained to {-1, 0, +1}
- 🌊 **Neural ODE Attention**: Continuous dynamics instead of discrete layers
- 🔄 **Multi-Scale Retention**: RetNet-style decay mechanisms
- 🧠 **Titan Memory**: Dynamic context storage without massive KV caches
- ♻️ **Recursive Looping**: Physical layers reused for deeper logical models

**Training Infrastructure (v2.1):**
- 📈 **CSV Metrics Logging**: Per-step loss, accuracy, LR, and GPU memory tracking
- 🛡️ **NaN Detection**: Automatic halt with debug logs and emergency checkpoints
- 💾 **Smart Checkpointing**: Rolling checkpoints + top-K best model tracking
- 🧬 **Stable Layer Pattern**: Research-backed `[retnet, titan_attn, retnet, mamba, titan_attn, ode]`

👉 **[Read the full v2 documentation →](docs/v2/README.md)**

---

## ⚙️ Installation

### Prerequisites

- **Python**: 3.9 or higher
- **CUDA**: 11.8+ (for GPU training)
- **uv**: Fast Python package manager ([install uv](https://docs.astral.sh/uv/))

### Quick Install with uv

```bash
# Clone the repository
git clone https://github.com/your-username/transformer-encoder-frankestein.git
cd transformer-encoder-frankestein

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the project (core dependencies)
uv pip install -e .

# Install with training dependencies (wandb, tensorboard)
uv pip install -e ".[train]"
```

### Alternative: pip install

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[train]"
```

### Verify Installation

```bash
python -c "from src.model.tormented_bert_frankestein import TormentedBertFrankenstein; print('✅ Model loaded')"
```

---

## 🎯 Usage

### Training

```bash
# Run the trainer
python src/training/main.py
```

### Inference Example

```python
import torch
from src.model.tormented_bert_frankestein import TormentedBertFrankenstein, UltraConfig
from src.tokenizer.spm_spa_redpajama35 import SpanishSPMTokenizer

# Load model and tokenizer
config = UltraConfig()
model = TormentedBertFrankenstein(config)
tokenizer = SpanishSPMTokenizer.from_pretrained("./models/es_redpajama_50k")

# Encode text
text = "El aprendizaje automático es fascinante."
inputs = tokenizer.encode(text, return_tensors="pt")

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)
    sentence_embedding = outputs.pooler_output  # [CLS] representation
```

---

## 📊 Dataset

Both models are pre-trained on Spanish text using the **Masked Language Modeling (MLM)** objective.

| Dataset | Description |
|---------|-------------|
| **[erickfmm/red_pajama_es_hq_35](https://huggingface.co/datasets/erickfmm/red_pajama_es_hq_35)** | High-quality Spanish subset from RedPajama-V2 |

The training pipeline includes:
- **Streaming mode**: Memory-efficient data loading
- **Storage management**: Respects disk budget limits (default: 300GB)
- **Dynamic caching**: Optimized for systems with large RAM

---

## 🔧 Hardware Requirements

These models are specifically optimized for:

| Component | Recommended | Minimum |
|-----------|-------------|---------|
| **GPU** | Nvidia Tesla P40 (24GB) | Any GPU with 16GB+ VRAM |
| **CPU** | Dual Xeon (56+ cores) | 8+ cores |
| **RAM** | 128GB DDR4 | 32GB |
| **Storage** | NVMe SSD (300GB free) | 100GB free |

### Optimization Tips

1. **For P40 users**: BitNet quantization bypasses the lack of FP16 Tensor Cores
2. **RAM disk for speed**: `sudo mount -t tmpfs -o size=64G tmpfs /mnt/ramdisk`
3. **CPU prefetching**: Leverage multi-core CPUs for data preprocessing

---

## 📚 References & Citations

This project builds upon the following research:

### Core Papers

```bibtex
@article{wang2024bitnet,
  title={The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits},
  author={Wang, Shuming and Ma, Hongyu and Dong, Li and Wang, Xingxing and Wang, Shaohan and Peng, Furu and Wei, Furu},
  journal={arXiv preprint arXiv:2402.17764},
  year={2024}
}

@inproceedings{chen2018neural,
  title={Neural Ordinary Differential Equations},
  author={Chen, Ricky TQ and Rubanova, Yulia and Bettencourt, Jesse and Duvenaud, David K},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6571--6583},
  year={2018}
}

@article{sun2023retentive,
  title={Retentive Network: A Successor to Transformer for Large Language Models},
  author={Sun, Yutao and Dong, Li and Huang, Shaohan and Ma, Shuming and Xia, Yuqing and Xue, Jilong and Wang, Jianyong and Wei, Furu},
  journal={arXiv preprint arXiv:2307.08621},
  year={2023}
}

@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}

@article{behrouz2025titan,
  title={Titan: Memory as Context for Large Language Models},
  author={Behrouz, Ali and others},
  journal={Google DeepMind Technical Report},
  year={2025}
}
```

### Additional References

- **RedPajama-V2**: [Together AI Blog](https://www.together.ai/blog/redpajama-data-v2) — 30T token multilingual dataset
- **RoPE**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- **SwiGLU**: Shazeer, "GLU Variants Improve Transformer"
- **Mixture of Experts**: Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models"

---

## 📜 Terms of Service

By using this software, you agree to the following terms:

### 1. Research Use

This software is provided **primarily for research and educational purposes**. Commercial use is permitted under the Apache 2.0 license, but users should be aware of the experimental nature of the code.

### 2. No Warranty

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED. The authors make no guarantees regarding:
- Model accuracy or fitness for any particular purpose
- Stability of training (especially for v2 ODE + BitNet combinations)
- Reproducibility across different hardware configurations

### 3. Responsible AI Use

Users agree to:
- Not use this software to generate harmful, misleading, or illegal content
- Acknowledge potential biases inherited from web-scraped training data
- Properly cite this work and underlying research in publications

### 4. Data & Privacy

- This repository does not collect user data
- Pre-training datasets are sourced from publicly available corpora
- Users are responsible for compliance with local data protection laws when deploying models

### 5. Limitation of Liability

In no event shall the authors or copyright holders be liable for any claim, damages, or other liability arising from the use of this software.

### 6. Model Outputs

The authors are not responsible for outputs generated by models trained with this code. Users deploying these models in production should implement appropriate content moderation.

---

## 📄 License

This project is licensed under the **Apache License 2.0**.

```
Copyright 2025 Erick Merino

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
uv pip install -e ".[train]"

# Run tests (if available)
pytest tests/

# Format code
black src/
```

---

## 🙏 Acknowledgments

- **Microsoft Research** for BitNet and RetNet research
- **Together AI** for the RedPajama dataset
- **Google DeepMind** for Titan memory concepts
- The open-source NLP community for inspiration

---

<p align="center">
  <b>Built with ❤️ for the Spanish NLP community</b><br>
  <sub>If you find this useful, consider giving it a ⭐!</sub>
</p>
