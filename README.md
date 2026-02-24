# Transformer Encoder Frankenstein

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

Config-driven training library and CLI for end-to-end NLP workflows:

- Training with strict YAML schema validation
- Deployment artifact generation and quantization
- Batch/interactive inference
- SBERT training and inference workflows

## What This Project Is

This repository is an installable Python package centered on a single CLI:

```bash
frankestein-transformer
```

The project is organized around:

- `src/training/configs/schema.yaml`: authoritative training configuration schema
- `src/cli.py`: CLI entrypoint and subcommands
- `src/deploy/*`: deployment, quantization, inference runtime
- `src/sbert/*`: sentence-embedding fine-tuning and inference tools

## Installation

### With `uv` (recommended)

```bash
git clone https://github.com/your-username/transformer-encoder-frankestein.git
cd transformer-encoder-frankestein

uv venv
source .venv/bin/activate
uv pip install -e "."
```

Optional training extras:

```bash
uv pip install -e ".[train]"
```

### With `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train]"
```

Verify:

```bash
frankestein-transformer --help
```

## CLI Overview

```bash
frankestein-transformer train ...
frankestein-transformer deploy ...
frankestein-transformer quantize ...
frankestein-transformer infer ...
frankestein-transformer sbert-train ...
frankestein-transformer sbert-infer ...
```

All model-executing commands support:

```bash
--device auto|cpu|cuda|mps
```

## Training (Schema-Driven)

Training is fully controlled by YAML files validated against:

`src/training/configs/schema.yaml`

### Run Training

```bash
# list available named configs
frankestein-transformer train --list-configs

# run a named preset
frankestein-transformer train --config-name mini --device auto

# run a custom config file
frankestein-transformer train --config src/training/configs/standard.yaml --device auto
```

### Schema Feature Surface

Top-level sections:

- `model_class`: `frankenstein` or `mini`
- `model`: model hyperparameters and feature toggles
- `training`: data loading, optimizer, scheduler, checkpointing, telemetry, stability controls

Model options include:

- Core sizing: `vocab_size`, `hidden_size`, `num_layers`, `num_loops`, `num_heads`, `retention_heads`
- Routing/FFN: `num_experts`, `top_k_experts`, `use_moe`, `ffn_hidden_size`, `ffn_activation`
- Mixer selection: `layer_pattern` with `retnet | mamba | ode | titan_attn | standard_attn | sigmoid_attn`
- ODE controls: `ode_solver` (`rk4|euler`), `ode_steps`
- Quantization/normalization: `use_bitnet`, `norm_type` (`layer_norm|dynamic_tanh|derf`)
- Embedding options: `use_factorized_embedding`, `factorized_embedding_dim`, `use_embedding_conv`, `embedding_conv_kernel`
- Positional controls: `use_hope`, `hope_base`, `hope_damping`

Training options include:

- Data pipeline: `batch_size`, `dataloader_workers`, `max_length`, `mlm_probability`, `max_samples`, `dataset_batch_size`, `num_workers`, `cache_dir`
- Local dataset toggles: `local_parquet_dir`, `prefer_local_cache`, `stream_local_parquet`
- Precision/accumulation: `use_amp`, `gradient_accumulation_steps`
- Optimizer block: `optimizer.optimizer_class` + prefixed parameter map
- Scheduler: `scheduler_total_steps`, `scheduler_warmup_ratio`, `scheduler_type`
- Stability/recovery: `grad_clip_max_norm`, `inf_post_clip_threshold`, `max_nan_retries`, `nan_check_interval`
- Checkpoint policy: `checkpoint_every_n_steps`, `max_rolling_checkpoints`, `num_best_checkpoints`
- Logging/telemetry: `log_gradient_stats`, `gradient_log_interval`, `csv_log_path`, `csv_rotate_on_schema_change`, `gpu_metrics_backend`, `nvml_device_index`, `enable_block_grad_norms`, `telemetry_log_interval`
- GaLore controls: `use_galore`, `galore_rank`, `galore_update_interval`, `galore_scale`, `galore_max_dim`

Supported optimizer classes:

- `sgd_momentum`, `adamw`, `adafactor`, `galore_adamw`, `prodigy`, `lion`, `sophia`, `muon`, `turbo_muon`, `radam`, `adan`, `adopt`, `ademamix`, `mars_adamw`, `cautious_adamw`, `lamb`, `schedulefree_adamw`, `shampoo`, `soap`

## Deployment and Inference

### Deploy checkpoint to artifact directory

```bash
frankestein-transformer deploy \
  --checkpoint path/to/checkpoint.pt \
  --output deployed_model \
  --format quantized \
  --validate \
  --device auto
```

### Quantization shortcut

```bash
frankestein-transformer quantize \
  --checkpoint path/to/checkpoint.pt \
  --output deployed_model_quantized \
  --validate \
  --device auto
```

### Inference modes

```bash
# single text
frankestein-transformer infer --model deployed_model --text "Texto de ejemplo" --device auto

# file input -> output
frankestein-transformer infer --model deployed_model --input texts.txt --output preds.pt --batch-size 8 --device auto

# benchmark
frankestein-transformer infer --model deployed_model --benchmark --device auto
```

Deployment artifacts include:

- `config.json`
- `model_quantized.pt` or `model.pt`
- `deployment_info.json`

## SBERT Workflows

### Train SBERT

```bash
frankestein-transformer sbert-train \
  --output_dir ./output/sbert_model \
  --batch_size 16 \
  --epochs 4 \
  --pooling_mode mean \
  --device auto
```

### SBERT Inference Modes

```bash
# pairwise similarity
frankestein-transformer sbert-infer \
  --model_path ./output/sbert_model \
  --mode similarity \
  --sentence1 "El gato está en la casa" \
  --sentence2 "Un felino está en el hogar" \
  --device auto

# semantic search
frankestein-transformer sbert-infer \
  --model_path ./output/sbert_model \
  --mode search \
  --query "aprendizaje automático" \
  --corpus_file corpus.txt \
  --top_k 5

# clustering
frankestein-transformer sbert-infer \
  --model_path ./output/sbert_model \
  --mode cluster \
  --sentences_file sentences.txt \
  --n_clusters 5

# encode to file
frankestein-transformer sbert-infer \
  --model_path ./output/sbert_model \
  --mode encode \
  --input_file sentences.txt \
  --output_file embeddings.npz
```

`--mode` options:

- `similarity`
- `search`
- `cluster`
- `encode`

## Documentation Map

- `README.md`: overview and quick usage
- `docs/README.md`: detailed schema and CLI reference
- `docs/paper.tex`: technical report focused on toolkit architecture and workflow
- `src/training/configs/README.md`: schema walkthrough and preset details
- `src/deploy/README.md`: deployment runtime details
- `src/sbert/README.md`: SBERT-specific training/inference details

## License

Apache License 2.0.
