# Configuraciones de entrenamiento (YAML)

Este directorio contiene presets en YAML. Cada archivo describe el modelo (`model`) y parámetros de entrenamiento/dataset (`training`).

## Archivos

- `frankenstein.yaml`
- `mini.yaml`
- `standard.yaml`
- `standard_hope.yaml`
- `tinybert.yaml`
- `embbert.yaml`

## Estructura general

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
  local_parquet_dir: "/path/to/parquet"  # opcional
  prefer_local_cache: true
  stream_local_parquet: true
  use_amp: false
  gradient_accumulation_steps: 4
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

## Campos disponibles (detallado)

### model_class
- `frankenstein` o `mini`.

### model (UltraConfig)
- `vocab_size`: tamaño del vocabulario.
- `hidden_size`: dimensión oculta.
- `num_layers`: capas físicas.
- `num_loops`: loops lógicos.
- `num_heads`: heads de atención.
- `retention_heads`: heads de retención.
- `num_experts`: expertos MoE.
- `top_k_experts`: top-k en routing MoE.
- `dropout`: dropout global.
- `layer_pattern`: lista de bloques (`retnet`, `mamba`, `ode`, `titan_attn`, `standard_attn`, `sigmoid_attn`).
- `ode_solver`: `rk4` o `euler`.
- `ode_steps`: pasos de integración.
- `use_bitnet`: BitLinear on/off.
- `norm_type`: `dynamic_tanh`, `derf`, o `layer_norm`.
- `use_factorized_embedding`: activa embeddings factorizados.
- `factorized_embedding_dim`: dimensión reducida del embedding.
- `use_embedding_conv`: Conv1d opcional en embeddings.
- `embedding_conv_kernel`: kernel de Conv1d.
- `hope_base`: base de HoPE.
- `hope_damping`: damping de HoPE.
- `use_hope`: aplica HoPE en `titan_attn`.
- `use_moe`: usa MoE en FFN.
- `ffn_hidden_size`: dimensión intermedia del FFN.
- `ffn_activation`: `silu` o `gelu`.

### training (TrainingConfig + runtime)
- `batch_size`: batch del dataloader.
- `dataloader_workers`: workers del dataloader.
- `max_length`: longitud máxima de secuencia.
- `mlm_probability`: probabilidad de máscara MLM.
- `max_samples`: límite de muestras.
- `dataset_batch_size`: batch interno del dataset streaming.
- `num_workers`: workers internos del dataset streaming.
- `cache_dir`: carpeta de cache.
- `local_parquet_dir`: ruta a parquet local (opcional).
- `prefer_local_cache`: preferir cache local.
- `stream_local_parquet`: stream desde parquet local.
- `use_amp`: mixed precision.
- `gradient_accumulation_steps`: acumulación de gradientes.
- `checkpoint_every_n_steps`: frecuencia de checkpoint.
- `max_rolling_checkpoints`: checkpoints rolling.
- `num_best_checkpoints`: mejores checkpoints a guardar.
- `nan_check_interval`: intervalo de check NaN/Inf.
- `log_gradient_stats`: logging de gradientes.
- `gradient_log_interval`: intervalo de log de gradientes.
- `use_galore`: activar GaLore.
- `galore_rank`: rango low-rank.
- `galore_update_interval`: frecuencia de proyección.
- `galore_scale`: escala de gradientes proyectados.
- `galore_max_dim`: máximo tamaño de tensor para GaLore.

## Notas

- `ffn_hidden_size` es requerido para igualar BERT/TinyBERT/EmbBERT.
- `embedding_conv_kernel` permite replicar EmbBERT con kernel 32.
- `standard_attn` y `sigmoid_attn` son compatibles con `use_hope=false`.
