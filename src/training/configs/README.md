# Training Configurations (YAML)

This directory contains YAML presets. Each file describes the model (`model`) and training/dataset parameters (`training`).

## Files

- `frankenstein.yaml`
- `mini.yaml`
- `standard.yaml`
- `standard_hope.yaml`
- `tinybert.yaml`
- `embbert.yaml`
- `schema.yaml` (JSON Schema in YAML form)

## General Structure

```yaml
model_class: frankenstein|mini
model:
  # UltraConfig
  vocab_size: 50000
  hidden_size: 768
  num_layers: 12
  num_loops: 1
  num_heads: 12
  retention_heads: 12
  num_experts: 4
  top_k_experts: 2
  dropout: 0.1
  layer_pattern: [standard_attn]
  ode_solver: rk4
  ode_steps: 2
  use_bitnet: false
  norm_type: layer_norm|dynamic_tanh|derf
  use_factorized_embedding: false
  factorized_embedding_dim: 128
  use_embedding_conv: false
  embedding_conv_kernel: 3
  hope_base: 10000.0
  hope_damping: 0.01
  use_hope: true
  use_moe: true
  ffn_hidden_size: 3072
  ffn_activation: silu|gelu
training:
  # TrainingConfig + runtime
  batch_size: 4
  dataloader_workers: 2
  max_length: 512
  mlm_probability: 0.15
  max_samples: 20000000
  dataset_batch_size: 25000
  num_workers: 8
  cache_dir: "./temp_data/v2_dataset_cache"
  local_parquet_dir: "/path/to/parquet"  # optional
  prefer_local_cache: true
  stream_local_parquet: true
  use_amp: false
  gradient_accumulation_steps: 4
  optimizer:
    optimizer_class: adamw
    parameters:
      adamw-lr_embeddings: 1e-6
      adamw-lr_norms: 5e-6
      adamw-lr_ode: 1e-7
      adamw-lr_retnet: 5e-6
      adamw-lr_mamba: 2e-6
      adamw-lr_attention: 3e-6
      adamw-lr_other: 2e-6
      adamw-wd_embeddings: 0.01
      adamw-wd_norms: 0.001
      adamw-wd_ode: 0.01
      adamw-wd_retnet: 0.01
      adamw-wd_mamba: 0.01
      adamw-wd_attention: 0.01
      adamw-wd_other: 0.01
      adamw-betas_embeddings: [0.9, 0.95]
      adamw-betas_norms: [0.9, 0.95]
      adamw-betas_ode: [0.9, 0.95]
      adamw-betas_retnet: [0.9, 0.95]
      adamw-betas_mamba: [0.9, 0.95]
      adamw-betas_attention: [0.9, 0.95]
      adamw-betas_other: [0.9, 0.95]
      adamw-eps_embeddings: 1e-8
      adamw-eps_norms: 1e-8
      adamw-eps_ode: 1e-8
      adamw-eps_retnet: 1e-8
      adamw-eps_mamba: 1e-8
      adamw-eps_attention: 1e-8
      adamw-eps_other: 1e-8
  scheduler_total_steps: 10000
  scheduler_warmup_ratio: 0.1
  scheduler_type: cosine
  grad_clip_max_norm: 5.0
  inf_post_clip_threshold: 100.0
  max_nan_retries: 3
  checkpoint_every_n_steps: 500
  max_rolling_checkpoints: 3
  num_best_checkpoints: 2
  nan_check_interval: 10
  log_gradient_stats: true
  gradient_log_interval: 10
  use_galore: false
  galore_rank: 64
  galore_update_interval: 1
  galore_scale: 1.0
  galore_max_dim: 4096
```

## Optimizer Chooser

`training.optimizer.optimizer_class` supported values:
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

`training.optimizer.parameters` must use the selected optimizer prefix. Example: when `optimizer_class: adamw`, all keys must start with `adamw-`.

### Optimizer Parameters Reference

All optimizers support these shared per-group parameter suffixes (use with optimizer prefix, e.g. `adamw-lr_embeddings`):

- `lr_embeddings`, `lr_norms`, `lr_ode`, `lr_retnet`, `lr_mamba`, `lr_attention`, `lr_other`
- `wd_embeddings`, `wd_norms`, `wd_ode`, `wd_retnet`, `wd_mamba`, `wd_attention`, `wd_other`
- `betas_embeddings`, `betas_norms`, `betas_ode`, `betas_retnet`, `betas_mamba`, `betas_attention`, `betas_other`
- `eps_embeddings`, `eps_norms`, `eps_ode`, `eps_retnet`, `eps_mamba`, `eps_attention`, `eps_other`

Optimizer-specific global parameter suffixes (also prefixed):

- `sgd_momentum`: `momentum`, `nesterov`
- `adamw`: none
- `adafactor`: `beta2_decay`, `clip_threshold`, `eps1`, `eps2`
- `galore_adamw`: `rank`, `update_proj_gap`
- `prodigy`: `d_coef`
- `lion`: none
- `sophia`: `rho`, `update_k`
- `muon`: `momentum`, `nesterov`, `ns_steps`, `ns_eps`
- `turbo_muon`: `momentum`, `nesterov`, `ns_steps`, `ns_eps`
- `radam`: none
- `adan`: none
- `adopt`: none
- `ademamix`: none
- `mars_adamw`: none
- `cautious_adamw`: `cautious_clip`
- `lamb`: none
- `schedulefree_adamw`: none
- `shampoo`: none
- `soap`: none

Examples:
- `adafactor-beta2_decay: 0.8`
- `galore_adamw-rank: 128`
- `sophia-update_k: 10`
- `muon-ns_steps: 5`

## Available Fields (Detailed)

### model_class
- `frankenstein` or `mini`.

### model (UltraConfig)
- `vocab_size`: vocabulary size.
- `hidden_size`: hidden dimension.
- `num_layers`: physical layers.
- `num_loops`: logical loops.
- `num_heads`: attention heads.
- `retention_heads`: retention heads.
- `num_experts`: MoE experts.
- `top_k_experts`: top-k routing in MoE.
- `dropout`: global dropout.
- `layer_pattern`: list of blocks (`retnet`, `mamba`, `ode`, `titan_attn`, `standard_attn`, `sigmoid_attn`).
- `ode_solver`: `rk4` or `euler`.
- `ode_steps`: integration steps.
- `use_bitnet`: BitLinear on/off.
- `norm_type`: `dynamic_tanh`, `derf`, or `layer_norm`.
- `use_factorized_embedding`: enable factorized embeddings.
- `factorized_embedding_dim`: reduced embedding dimension.
- `use_embedding_conv`: optional Conv1d on embeddings.
- `embedding_conv_kernel`: Conv1d kernel size.
- `hope_base`: HoPE base.
- `hope_damping`: HoPE damping.
- `use_hope`: apply HoPE in `titan_attn`.
- `use_moe`: use MoE in FFN.
- `ffn_hidden_size`: FFN intermediate dimension.
- `ffn_activation`: `silu` or `gelu`.

### training (TrainingConfig + runtime)
- `batch_size`: dataloader batch size.
- `dataloader_workers`: dataloader workers.
- `max_length`: max sequence length.
- `mlm_probability`: MLM mask probability.
- `max_samples`: sample limit.
- `dataset_batch_size`: internal streaming dataset batch size.
- `num_workers`: internal streaming dataset workers.
- `cache_dir`: cache folder.
- `local_parquet_dir`: local parquet path (optional).
- `prefer_local_cache`: prefer local cache.
- `stream_local_parquet`: stream from local parquet.
- `use_amp`: mixed precision.
- `gradient_accumulation_steps`: gradient accumulation steps.
- `optimizer`: optimizer chooser object with `optimizer_class` and prefixed `parameters`.
- `scheduler_total_steps`: total steps for the scheduler.
- `scheduler_warmup_ratio`: warmup fraction of total steps.
- `scheduler_type`: `cosine`, `constant`, or `linear_warmup_then_constant`.
- `grad_clip_max_norm`: gradient clipping max norm.
- `inf_post_clip_threshold`: threshold for exploding gradients after clipping.
- `max_nan_retries`: max retries before stopping on NaN.
- `checkpoint_every_n_steps`: checkpoint frequency.
- `max_rolling_checkpoints`: rolling checkpoints to keep.
- `num_best_checkpoints`: best checkpoints to keep.
- `nan_check_interval`: NaN/Inf check interval.
- `log_gradient_stats`: enable gradient logging.
- `gradient_log_interval`: gradient logging interval.
- `use_galore`: enable GaLore.
- `galore_rank`: low-rank projection rank.
- `galore_update_interval`: projection update frequency.
- `galore_scale`: projected gradient scale.
- `galore_max_dim`: max tensor size for GaLore.

## Migration Note

This is a hard migration. Legacy top-level optimizer keys (`lr_*`, `wd_*`, `betas_*`, `eps_*`) are no longer accepted in `training`.

## Notes

- `ffn_hidden_size` is required to match BERT/TinyBERT/EmbBERT.
- `embedding_conv_kernel` allows replicating EmbBERT with kernel 32.
- `standard_attn` and `sigmoid_attn` are compatible with `use_hope=false`.
