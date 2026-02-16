# ROADMAP EJECUTABLE PARA IA: Tormented-BERT-Mini

Este documento está rediseñado para que una IA de desarrollo pueda modificar el codebase **sin ambigüedad**.

## Objetivo

Convertir `TormentedBertFrankenstein` en una variante **Mini** estable y eficiente para P40 con:

- `hidden_size=384`
- `num_layers=6` (físicas)
- `num_loops=2` (12 lógicas)
- Sustituir `DynamicTanh` por `Derf`
- Implementar **factorización de embeddings** (entrada 128d -> proyección a 384d)
- Reemplazar placeholders de atención Titan por implementación real con `HoPE`
- Mantener `BitLinear` en proyecciones lineales
- Dejar base para memoria tipo Titans/Nested Learning (fase incremental)

---

## Reglas de implementación para la IA

1. **No romper API existente**: mantener `UltraConfig` y `TormentedBertFrankenstein` funcionando.
2. **Agregar, no destruir**: introducir nueva clase `TormentedBertMini` en archivo nuevo o en el módulo actual, sin eliminar Frankenstein.
3. **Cambios pequeños y verificables**: un bloque funcional por commit lógico.
4. **Compatibilidad CPU/GPU**: ningún `.cuda()` hardcodeado dentro del modelo.
5. **Evitar deuda técnica**: documentar cada módulo nuevo con docstring breve.

---

## Alcance por archivos

### 1) `src/model/tormented_bert_frankestein.py`

Implementar o modificar:

- `class Derf(nn.Module)`
  - Fórmula: $y = \gamma \cdot \mathrm{erf}(\alpha x + s) + \beta$
  - `alpha`, `s` escalares aprendibles; `gamma`, `beta` por canal.

- `get_norm(config)`
  - Soportar `norm_type="derf"`.
  - Mantener `dynamic_tanh` y fallback a `LayerNorm`.

- `class FactorizedEmbedding(nn.Module)`
  - `nn.Embedding(vocab, 128)` + proyección a `hidden_size` (384 en mini).
  - Variante preferida: `Embedding -> Conv1d(kernel=3, padding=1) -> Proyección`.
  - Si `use_bitnet=True`, usar `BitLinear`; si no, `nn.Linear`.
  - Debe quedar activada por defecto en `TormentedBertMini`.

- `class HoPE(nn.Module)`
  - Aplicar transformación hiperbólica por pares de dimensiones.
  - Incluir damping exponencial para decaimiento monotónico.

- `class TitanAttention(nn.Module)`
  - Reemplazar placeholder `BitLinear(hidden, hidden)`.
  - Q/K/V con `BitLinear`, multi-head attention y uso de `HoPE` sobre `q,k`.
  - Salida con `out_proj`.

- `class HybridLayer`
  - Para `layer_type == "titan_attn"`, usar `TitanAttention` real.
  - Permitir recibir `logical_layer_idx` para futuras variantes de decaimiento.

- `class TormentedBertMini(nn.Module)`
  - Config preset mini (`hidden_size=384`, `num_layers=6`, `num_loops=2`, `num_heads=6`).
  - Usar `FactorizedEmbedding`.
  - Forward con loop lógico y paso de `logical_layer_idx` a cada capa.

### 2) `src/training/main.py`

Modificar:

- Agregar flag/config para elegir modelo: `frankenstein` o `mini`.
- Para modo mini:
  - `hidden_size=384`, `num_layers=6`, `num_loops=2`, `num_heads=6`
  - `norm_type="derf"`
  - patrón estable: `['retnet','titan_attn','retnet','mamba','titan_attn','ode']`

### 3) `src/training/trainer.py`

Modificar para robustez numérica post-backward:

- Insertar chequeos explícitos tras `loss.backward()` para detectar:
  - gradientes `NaN`
  - gradientes `Inf`
  - gradientes totalmente cero por parámetro
  - norma global inválida o extremadamente baja
- Aplicar estrategia de reparación por tipo de falla (ver Fase 3).
- Añadir early stop por condiciones no recuperables.
- Loggear métricas de estabilidad por step (`grad_norm`, `has_nan`, `has_inf`, `has_zero`, `repair_action`).

### 4) `docs/README.md` (opcional pero recomendado)

Actualizar sección de uso con ejemplo de entrenamiento del modo mini.

---

## Fases de ejecución (orden obligatorio)

### Fase 0 — Baseline

1. Cargar proyecto sin cambios.
2. Ejecutar import de modelo y un forward pequeño en CPU.
3. Guardar resultado como baseline (shape de salida y número de parámetros).

**Criterio de aceptación**: baseline corre sin excepción.

### Fase 1 — Derf + FactorizedEmbedding (obligatorio)

1. Implementar `Derf`.
2. Integrar `norm_type='derf'`.
3. Implementar `FactorizedEmbedding` (`Embedding 128d -> Proyección 384d`).
4. Añadir variante con `Conv1d(k=3)` antes de la proyección (configurable).
5. Integrar embedding factorizado en `TormentedBertMini`.

**Criterio de aceptación**:
- Forward de `TormentedBertMini` funciona con batch pequeño.
- Parámetros de embedding reducidos vs embedding denso de 384.
- Se puede activar/desactivar `Conv1d` con flag sin romper compatibilidad.

### Fase 2 — Titan Attention real con HoPE

1. Implementar `HoPE`.
2. Implementar `TitanAttention` multi-head con Q/K/V + `HoPE`.
3. Reemplazar placeholder en `HybridLayer`.

**Criterio de aceptación**:
- `layer_type='titan_attn'` deja de usar placeholder.
- Forward completo sin NaN en prueba corta.

### Fase 3 — Verificación, reparación y early stop de gradientes

Implementar en el loop de entrenamiento, inmediatamente después de `loss.backward()`:

```python
has_nan = False
has_inf = False
has_zero = False
total_norm = 0.0

for param in model.parameters():
  if param.grad is not None:
    if torch.isnan(param.grad).any():
      has_nan = True
    if torch.isinf(param.grad).any():
      has_inf = True
    if (param.grad == 0).all():
      has_zero = True
    total_norm += param.grad.norm().item()

if not torch.isfinite(torch.tensor(total_norm, device=next(model.parameters()).device)):
  has_inf = True
```

#### Política de reparación por tipo de error

1. **NaN en gradientes**
   - No aplicar `optimizer.step()` en ese batch.
   - `optimizer.zero_grad(set_to_none=True)` y registrar incidente.
   - Con AMP: usar `GradScaler`; hacer `scaler.unscale_(optimizer)` antes de clipping.
   - Aplicar `torch.nn.utils.clip_grad_norm_(..., max_norm=1.0)`.
   - Reintentar batch con `lr = lr/2` hasta 3 intentos.
   - Si falla 3 veces seguidas: guardar checkpoint y detener entrenamiento.

2. **Inf en gradientes (exploding)**
   - Clipping obligatorio `clip_grad_norm_(..., 1.0)`.
   - Si norma post-clip > 10 o no finita, reducir LR dinámicamente.
   - Si persiste después de reparación, saltar step.
   - Early stop solo si ocurre por >5 épocas consecutivas.

3. **Gradientes en cero (vanishing/dead)**
   - No reiniciar batch automáticamente.
   - Registrar alerta si `total_norm < 1e-10`.
   - Recomendar mitigaciones estructurales: init Xavier/He, activaciones no saturantes, residuales, ajuste de ODE step size.
   - Early stop si condición persistente + `val_loss` en plateau (patience=5).

#### Reglas AMP / clipping

- Si hay AMP:
  - `scaler.scale(loss).backward()`
  - `scaler.unscale_(optimizer)`
  - `clip_grad_norm_`
  - `scaler.step(optimizer)`
  - `scaler.update()`
- Si no hay AMP:
  - `loss.backward()`
  - `clip_grad_norm_`
  - `optimizer.step()`

#### Criterio de aceptación

- Cada step loggea estado de gradientes y norma global.
- Ante NaN/Inf, el step se salta de forma segura sin corromper el estado.
- El entrenamiento se detiene automáticamente en condiciones no recuperables.

### Fase 4 — Loop lógico explícito

1. Pasar `logical_layer_idx` en cada iteración del loop.
2. Ajustar firma de capas/mixers para aceptar índice lógico (aunque inicialmente no lo usen todos).

**Criterio de aceptación**:
- Se verifica (log/debug) que cada capa recibe índice lógico correcto `0..(num_layers*num_loops-1)`.

### Fase 5 — Integración en entrenamiento

1. Agregar selección de modo mini en `src/training/main.py`.
2. Ajustar hiperparámetros por defecto para mini.
3. Validar una época corta o smoke-train.

**Criterio de aceptación**:
- Script de entrenamiento inicia en modo mini y hace al menos varios steps sin crash.

### Fase 6 — Extensión Titans/Nested Learning (incremental)

1. Crear módulo de memoria opcional con regla tipo Delta:

$$M_t = M_{t-1}(\alpha_t I - \eta_t k_t k_t^\top) + \eta_t v_t k_t^\top$$

2. Activarlo con flag (por ejemplo `use_titan_memory=True`).
3. Mantener fallback al `TitanAttention` estándar si el módulo no está activo.

**Criterio de aceptación**:
- Feature flag funciona ON/OFF.
- No degrada ejecución base cuando está OFF.

---

## Definición de terminado (DoD)

La tarea se considera completa solo si:

1. Existe `TormentedBertMini` funcional.
2. `Derf`, `FactorizedEmbedding` y `HoPE` están integrados y usados.
3. `titan_attn` ya no es placeholder.
4. `main.py` permite elegir modo mini.
5. Existe verificación post-backward para NaN/Inf/zero-grad + política de reparación.
6. Hay early stop para no-recoverable NaN/Inf y vanishing persistente.
7. Hay prueba de smoke test (forward y backward simples).
8. Documentación mínima actualizada.

---

## Prompt listo para pegar en otra IA

> Refactoriza este proyecto PyTorch para introducir `TormentedBertMini` sin romper `TormentedBertFrankenstein`.
>
> Objetivos obligatorios:
> 1) Implementa `Derf` y soporta `norm_type='derf'`.
> 2) Implementa `FactorizedEmbedding` obligatorio (128 -> 384) con opción `Conv1d(k=3)` antes de proyección.
> 3) Implementa `HoPE` y úsalo en una `TitanAttention` real (QKV multi-head), reemplazando el placeholder actual de `titan_attn`.
> 4) Crea `TormentedBertMini` con `hidden_size=384`, `num_layers=6`, `num_loops=2`, `num_heads=6`, patrón estable `[retnet, titan_attn, retnet, mamba, titan_attn, ode]`.
> 5) Pasa `logical_layer_idx` en el loop lógico para futuras estrategias de decaimiento.
> 6) Ajusta `src/training/main.py` para seleccionar modo `mini` o `frankenstein` por configuración.
> 7) En `src/training/trainer.py`, agrega detección post-backward de NaN/Inf/zero-grad, reparación (skip-step, zero_grad, clipping, lr/2 retry hasta 3) y early stop robusto.
>
> Restricciones:
> - Mantén `BitLinear` en proyecciones lineales críticas.
> - Compatibilidad CPU/GPU sin hardcode de `.cuda()`.
> - En AMP, hacer `unscale_` antes de `clip_grad_norm_`.
> - Cambios incrementales y verificables.
>
> Entrega requerida:
> - Lista de archivos modificados
> - Resumen técnico por cambio
> - Resultado de smoke test (forward/backward)

---

## Riesgos conocidos y mitigación

- **Inestabilidad numérica en HoPE**: limitar magnitud de argumentos de `cosh/sinh` o usar escalado seguro.
- **NaN en ODE + BitNet**: mantener `ode_steps` bajos y gradient clipping.
- **Costo en MoE**: conservar top-k bajo (`top_k_experts=2`) para no penalizar throughput.

---

## Siguiente iteración sugerida

Cuando Mini esté estable, continuar con:

1. Titans memory auto-referencial con regla Delta (feature flag).
2. Benchmarks de memoria/velocidad P40 vs Frankenstein.
3. Ablation: `dynamic_tanh` vs `derf`, y HoPE ON/OFF.