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
  lr_embeddings: 1e-6
  lr_norms: 5e-6
  lr_ode: 1e-7
  lr_retnet: 5e-6
  lr_mamba: 2e-6
  lr_attention: 3e-6
  lr_other: 2e-6
  wd_embeddings: 0.01
  wd_norms: 0.001
  wd_ode: 0.01
  wd_retnet: 0.01
  wd_mamba: 0.01
  wd_attention: 0.01
  wd_other: 0.01
  betas_embeddings: [0.9, 0.95]
  betas_norms: [0.9, 0.95]
  betas_ode: [0.9, 0.95]
  betas_retnet: [0.9, 0.95]
  betas_mamba: [0.9, 0.95]
  betas_attention: [0.9, 0.95]
  betas_other: [0.9, 0.95]
  eps_embeddings: 1e-8
  eps_norms: 1e-8
  eps_ode: 1e-8
  eps_retnet: 1e-8
  eps_mamba: 1e-8
  eps_attention: 1e-8
  eps_other: 1e-8
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
- `lr_embeddings`: LR for embedding parameters.
- `lr_norms`: LR for normalization parameters.
- `lr_ode`: LR for ODE parameters.
- `lr_retnet`: LR for RetNet parameters.
- `lr_mamba`: LR for Mamba parameters.
- `lr_attention`: LR for attention parameters.
- `lr_other`: LR for all other parameters.
- `wd_embeddings`: weight decay for embedding parameters.
- `wd_norms`: weight decay for normalization parameters.
- `wd_ode`: weight decay for ODE parameters.
- `wd_retnet`: weight decay for RetNet parameters.
- `wd_mamba`: weight decay for Mamba parameters.
- `wd_attention`: weight decay for attention parameters.
- `wd_other`: weight decay for all other parameters.
- `betas_embeddings`: Adam betas for embedding parameters (list like `[0.9, 0.95]`).
- `betas_norms`: Adam betas for normalization parameters.
- `betas_ode`: Adam betas for ODE parameters.
- `betas_retnet`: Adam betas for RetNet parameters.
- `betas_mamba`: Adam betas for Mamba parameters.
- `betas_attention`: Adam betas for attention parameters.
- `betas_other`: Adam betas for all other parameters.
- `eps_embeddings`: Adam eps for embedding parameters.
- `eps_norms`: Adam eps for normalization parameters.
- `eps_ode`: Adam eps for ODE parameters.
- `eps_retnet`: Adam eps for RetNet parameters.
- `eps_mamba`: Adam eps for Mamba parameters.
- `eps_attention`: Adam eps for attention parameters.
- `eps_other`: Adam eps for all other parameters.
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

## Notes

- `ffn_hidden_size` is required to match BERT/TinyBERT/EmbBERT.
- `embedding_conv_kernel` allows replicating EmbBERT with kernel 32.
- `standard_attn` and `sigmoid_attn` are compatible with `use_hope=false`.
