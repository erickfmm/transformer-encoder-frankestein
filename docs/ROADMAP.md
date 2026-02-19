# ROADMAP EJECUTABLE PARA IA: YAML + Arquitecturas Estándar

Este roadmap refleja SOLO lo pendiente. Lo ya implementado fue retirado.

## Objetivo

- Introducir configuraciones YAML por archivo en `src/training/configs/`.
- Leer configs desde YAML en `src/training/main.py`.
- Implementar `StandardAttention` y `SigmoidAttention` como tipos de bloque.
- Hacer HoPE opcional y parametrizable en la atención.
- Agregar GaLore opcional con proyección low-rank de gradientes.
- Definir presets BERT-Base, TinyBERT6 y EmbBERT (adaptado a vocab 50k).

---

## Alcance por archivos

### 1) `src/model/tormented_bert_frankestein.py`

- Agregar campos nuevos a `UltraConfig`:
  - `use_hope`, `use_moe`, `ffn_hidden_size`, `ffn_activation`, `embedding_conv_kernel`
- Implementar `StandardAttention` y `SigmoidAttention`.
- Hacer `TitanAttention` con HoPE opcional (`use_hope`).
- Permitir FFN estándar cuando `use_moe=False`.
- Soportar `layer_pattern` con `standard_attn` y `sigmoid_attn`.
- Usar `embedding_conv_kernel` en `FactorizedEmbedding`.

### 2) `src/training/configs/*.yaml`

- Crear presets YAML:
  - `frankenstein.yaml`, `mini.yaml`
  - `standard.yaml` (BERT-Base)
  - `standard_hope.yaml`
  - `tinybert.yaml` (TinyBERT6)
  - `embbert.yaml` (EmbBERT adaptado)

### 3) `src/training/config_loader.py`

- Cargar YAML y construir `UltraConfig` + `TrainingConfig`.
- Exponer listado de configs.

### 4) `src/training/main.py`

- Añadir flags `--config`, `--config-name`, `--list-configs`, `--batch-size`.
- Resolver config desde YAML y construir modelo a partir del `model_class`.
- Mover parámetros de dataset y trainer a YAML.

### 5) `src/training/trainer.py`

- Agregar campos GaLore en `TrainingConfig`.
- Implementar proyección low-rank de gradientes y aplicarla antes del clipping.

---

## Fases de ejecución

### Fase 1 — Modelo

- `StandardAttention` y `SigmoidAttention`.
- HoPE opcional en `TitanAttention`.
- FFN estándar si `use_moe=False`.
- `embedding_conv_kernel` en `FactorizedEmbedding`.

**Criterio de aceptación**
- Los nuevos tipos de bloque funcionan en un forward simple.

### Fase 2 — YAML Configs

- Crear carpeta `src/training/configs/`.
- Agregar presets solicitados.
- Implementar loader y listado de configs.

**Criterio de aceptación**
- `--list-configs` lista correctamente los YAML.
- `--config-name standard` construye un modelo sin error.

### Fase 3 — Entrenamiento desde YAML

- `main.py` usa los valores YAML para dataset, dataloader y trainer.
- `--batch-size` sobreescribe YAML.

**Criterio de aceptación**
- Entrenamiento arranca con config YAML sin tocar el código.

### Fase 4 — GaLore

- Proyección low-rank opcional en gradientes.

**Criterio de aceptación**
- Activando `use_galore=true` no rompe el step.

---

## Definición de terminado

- YAMLs de presets creados y utilizables.
- `main.py` carga YAML y entrena con ellos.
- `StandardAttention` y `SigmoidAttention` disponibles en `layer_pattern`.
- HoPE parametrizable y GaLore opcional implementados.
