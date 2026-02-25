# Toolkit Reference

This document describes the project as a configurable library + CLI toolchain.

## 1. CLI Command Surface

The entrypoint is:

```bash
frankestein-transformer
```

Subcommands:

- `train`
- `deploy`
- `quantize`
- `infer`
- `sbert-train`
- `sbert-infer`

Common execution device choices:

```bash
--device auto|cpu|cuda|mps
```

## 2. Train Command

Usage pattern:

```bash
frankestein-transformer train --config-name mini --device auto
frankestein-transformer train --config path/to/config.yaml --device auto
frankestein-transformer train --list-configs
```

Key train flags:

- `--config`
- `--config-name`
- `--list-configs`
- `--batch-size`
- `--model-mode` (`frankenstein|mini`)
- `--device`
- `--gpu-temp-guard` / `--no-gpu-temp-guard`
- `--gpu-temp-pause-threshold-c`
- `--gpu-temp-resume-threshold-c`
- `--gpu-temp-critical-threshold-c`
- `--gpu-temp-poll-interval-seconds`

## 3. Configuration Schema (`src/training/configs/schema.yaml`)

Top-level required keys:

- `model_class`
- `model`
- `training`

### 3.1 `model_class`

Allowed values:

- `frankenstein`
- `mini`

### 3.2 `model` section

Required fields:

- `vocab_size`
- `hidden_size`
- `num_layers`
- `num_loops`
- `num_heads`
- `retention_heads`
- `num_experts`
- `top_k_experts`
- `dropout`
- `layer_pattern`
- `ode_solver`
- `ode_steps`
- `use_bitnet`
- `norm_type`
- `use_factorized_embedding`
- `factorized_embedding_dim`
- `use_embedding_conv`
- `embedding_conv_kernel`
- `use_hope`
- `use_moe`
- `ffn_hidden_size`
- `ffn_activation`

Model enums/toggles:

- `layer_pattern` items: `retnet`, `mamba`, `ode`, `titan_attn`, `standard_attn`, `sigmoid_attn`
- `ode_solver`: `rk4`, `euler`
- `norm_type`: `layer_norm`, `dynamic_tanh`, `derf`
- `ffn_activation`: `silu`, `gelu`

Optional model keys:

- `hope_base`
- `hope_damping`

### 3.3 `training` section

Required core fields:

- `batch_size`, `dataloader_workers`, `max_length`, `mlm_probability`
- `max_samples`, `dataset_batch_size`, `num_workers`, `cache_dir`
- `use_amp`, `gradient_accumulation_steps`
- `optimizer`
- `scheduler_total_steps`, `scheduler_warmup_ratio`, `scheduler_type`
- `grad_clip_max_norm`, `inf_post_clip_threshold`, `max_nan_retries`
- `checkpoint_every_n_steps`, `max_rolling_checkpoints`, `num_best_checkpoints`
- `nan_check_interval`, `log_gradient_stats`, `gradient_log_interval`
- `csv_log_path`, `csv_rotate_on_schema_change`
- `gpu_metrics_backend`, `nvml_device_index`, `enable_block_grad_norms`, `telemetry_log_interval`
- `gpu_temp_guard_enabled`, `gpu_temp_pause_threshold_c`, `gpu_temp_resume_threshold_c`, `gpu_temp_critical_threshold_c`, `gpu_temp_poll_interval_seconds`
- `use_galore`, `galore_rank`, `galore_update_interval`, `galore_scale`, `galore_max_dim`

Optional dataset locality fields:

- `local_parquet_dir`
- `prefer_local_cache`
- `stream_local_parquet`

Scheduler enum:

- `cosine`
- `constant`
- `linear_warmup_then_constant`

GPU metrics backend enum:

- `nvml`
- `none`

### 3.4 Optimizer configuration

`training.optimizer` requires:

- `optimizer_class`
- `parameters`

Allowed `optimizer_class` values:

- `sgd_momentum`
- `adamw`
- `adafactor`
- `galore_adamw`
- `prodigy`
- `lion`
- `sophia`
- `muon`
- `turbo_muon`
- `radam`
- `adan`
- `adopt`
- `ademamix`
- `mars_adamw`
- `cautious_adamw`
- `lamb`
- `schedulefree_adamw`
- `shampoo`
- `soap`

`parameters` keys must use the prefix for the selected optimizer class (enforced by schema `allOf` conditions).

## 4. Deployment Commands

Deploy checkpoint to artifacts:

```bash
frankestein-transformer deploy \
  --checkpoint path/to/checkpoint.pt \
  --output deployed_model \
  --format quantized \
  --validate \
  --device auto
```

Deploy flags:

- `--checkpoint` (required)
- `--output` (required)
- `--format` (`quantized|standard`)
- `--validate`
- `--config` (optional JSON)
- `--device`

Quantize shortcut:

```bash
frankestein-transformer quantize --checkpoint ckpt.pt --output deployed_model_quantized --validate
```

## 5. Inference Command

```bash
frankestein-transformer infer --model deployed_model --text "hola" --device auto
```

Infer flags:

- `--model` (required)
- `--text`
- `--input`
- `--output`
- `--device`
- `--fp16`
- `--batch-size`
- `--benchmark`

## 6. SBERT Commands

### 6.1 `sbert-train`

```bash
frankestein-transformer sbert-train --output_dir ./output/sbert_model --batch_size 16 --epochs 4 --device auto
```

Flags:

- `--pretrained`
- `--output_dir`
- `--batch_size`
- `--epochs`
- `--learning_rate`
- `--max_train_samples`
- `--max_eval_samples`
- `--hidden_size`
- `--num_layers`
- `--pooling_mode` (`mean|cls|max`)
- `--no_amp`
- `--no_resample`
- `--resample_std`
- `--device`

### 6.2 `sbert-infer`

```bash
frankestein-transformer sbert-infer --model_path ./output/sbert_model --mode similarity --sentence1 "a" --sentence2 "b"
```

Flags:

- `--model_path` (required)
- `--mode` (`similarity|search|cluster|encode`) (required)
- `--sentence1`, `--sentence2`
- `--query`, `--corpus_file`, `--top_k`
- `--sentences_file`, `--n_clusters`
- `--input_file`, `--output_file`
- `--batch_size`
- `--device`

## 7. Recommended Workflow

1. Select or create YAML config that validates against `schema.yaml`.
2. Run `train`.
3. Export with `deploy` or `quantize`.
4. Run `infer` for runtime validation and benchmark.
5. Train/evaluate sentence embeddings via `sbert-train` and `sbert-infer`.
