import os
import csv
import torch
import logging
import math
import heapq
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from utils.storage_manager import StorageManager
from model.tormented_bert_frankestein import TormentedBertFrankenstein, UltraConfig


# ==================== TRAINING CONFIGURATION ====================
@dataclass
class TrainingConfig:
    """Configuration for training behavior and checkpointing"""
    # CSV Logging
    csv_log_path: str = "training_metrics.csv"
    
    # Rolling Checkpoints
    checkpoint_every_n_steps: int = 500  # Save every N steps
    max_rolling_checkpoints: int = 3     # Keep only last N rolling checkpoints
    
    # Best Model Checkpoints
    num_best_checkpoints: int = 2        # Keep top K best models
    
    # NaN Detection
    nan_check_interval: int = 10         # Check for NaN every N steps
    
    # Gradient monitoring
    log_gradient_stats: bool = True
    gradient_log_interval: int = 100
    gradient_accumulation_steps: int = 4

    # Optimizer learning rates by parameter group
    lr_embeddings: float = 1e-6
    lr_norms: float = 5e-6
    lr_ode: float = 1e-7
    lr_retnet: float = 5e-6
    lr_mamba: float = 2e-6
    lr_attention: float = 3e-6
    lr_other: float = 2e-6

    # Optimizer weight decay by parameter group
    wd_embeddings: float = 0.01
    wd_norms: float = 0.001
    wd_ode: float = 0.01
    wd_retnet: float = 0.01
    wd_mamba: float = 0.01
    wd_attention: float = 0.01
    wd_other: float = 0.01

    # Optimizer betas by parameter group
    betas_embeddings: Tuple[float, float] = (0.9, 0.95)
    betas_norms: Tuple[float, float] = (0.9, 0.95)
    betas_ode: Tuple[float, float] = (0.9, 0.95)
    betas_retnet: Tuple[float, float] = (0.9, 0.95)
    betas_mamba: Tuple[float, float] = (0.9, 0.95)
    betas_attention: Tuple[float, float] = (0.9, 0.95)
    betas_other: Tuple[float, float] = (0.9, 0.95)

    # Optimizer eps by parameter group
    eps_embeddings: float = 1e-8
    eps_norms: float = 1e-8
    eps_ode: float = 1e-8
    eps_retnet: float = 1e-8
    eps_mamba: float = 1e-8
    eps_attention: float = 1e-8
    eps_other: float = 1e-8

    # Scheduler configuration
    scheduler_total_steps: int = 10000
    scheduler_warmup_ratio: float = 0.1
    scheduler_type: str = "cosine"

    # Post-backward stability
    max_nan_retries: int = 3
    grad_clip_max_norm: float = 5.0  # Increased from 1.0 - too aggressive clipping was causing issues
    inf_post_clip_threshold: float = 100.0  # Increased from 10.0 - more conservative threshold
    min_grad_norm_for_signal: float = 1e-10
    inf_epoch_patience: int = 5
    zero_grad_plateau_patience: int = 5

    # GaLore (low-rank gradient projection)
    use_galore: bool = False
    galore_rank: int = 64
    galore_update_interval: int = 1
    galore_scale: float = 1.0
    galore_max_dim: int = 4096

    # Precision control
    # NOTE: Default False because hybrid ODE/SSM blocks can become unstable in fp16 on older GPUs (e.g., Tesla P40).
    use_amp: bool = False

# ==================== TITAN TRAINER ====================
class TitanTrainer:
    """Advanced trainer for TITAN-BERT-ULTRA with specialized optimizations"""
    
    def __init__(self, model: torch.nn.Module, config: UltraConfig,
                 training_config: TrainingConfig = None,
                 device: str = None): #torch.device = None):
        self.model = model
        self.config = config
        self.training_config = training_config or TrainingConfig()
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu" # device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = (
            bool(self.training_config.use_amp)
            and torch.cuda.is_available()
            and str(self.device).startswith("cuda")
        )
        self.amp_dtype = torch.float16
        if self.use_amp and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        self.model.to(self.device)
        
        # Specialized optimizer for hybrid architecture
        self.optimizer = self._setup_optimizer()
        
        # Mixed precision with aggressive scaling for BitNet
        self.scaler = GradScaler(
            self.device,
            init_scale=2.**10,  # Lower initial scale for BitNet stability
            growth_factor=1.5,  # Conservative growth
            backoff_factor=0.8,
            growth_interval=100
        )
        
        # Learning rate scheduler with warmup for ODE dynamics
        self.scheduler = self._setup_scheduler()
        
        # Storage monitoring
        self.storage_manager = StorageManager()
        
        # Training state tracking
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Gradient accumulation for effective larger batches
        self.gradient_accumulation_steps = max(int(self.training_config.gradient_accumulation_steps), 1)
        
        # === NEW: CSV Logger ===
        self.csv_file = None
        self.csv_writer = None
        self._init_csv_logger()
        
        # === NEW: Rolling checkpoint tracking ===
        self.rolling_checkpoints: List[str] = []  # FIFO queue of checkpoint paths
        
        # === NEW: Best checkpoints tracking (min-heap by negative loss for max extraction) ===
        # Format: [(-loss, checkpoint_path), ...]
        self.best_checkpoints: List[Tuple[float, str]] = []
        
        # === NEW: NaN detection state ===
        self.nan_detected = False
        self.last_valid_state = None  # For debugging

        # Stability / early-stop state
        self.current_epoch_had_inf = False
        self.consecutive_inf_epochs = 0
        self.consecutive_zero_grad_plateau_epochs = 0
        self.non_improving_epochs = 0
        
        self._log_setup()
    
    def _init_csv_logger(self):
        """Initialize CSV file for metrics logging"""
        csv_path = self.training_config.csv_log_path
        file_exists = os.path.exists(csv_path)
        
        self.csv_file = open(csv_path, 'a', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write header if new file
        if not file_exists:
            self.csv_writer.writerow([
                'timestamp', 'epoch', 'step', 'global_step', 
                'loss', 'accuracy', 'learning_rate', 'grad_norm',
                'scaler_scale', 'gpu_memory_gb', 'gpu_cached_gb',
                'has_nan', 'has_inf', 'has_zero', 'repair_action'
            ])
            self.csv_file.flush()
    
    def _log_step_to_csv(self, epoch: int, step: int, loss: float, accuracy: float,
                         lr: float, grad_norm: float = 0.0,
                         has_nan: bool = False, has_inf: bool = False,
                         has_zero: bool = False, repair_action: str = "none"):
        """Log a single training step to CSV"""
        gpu_memory = 0.0
        gpu_cached = 0.0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_cached = torch.cuda.memory_reserved() / 1024**3
        
        self.csv_writer.writerow([
            datetime.now().isoformat(),
            epoch,
            step,
            self.global_step,
            f'{loss:.6f}',
            f'{accuracy:.6f}',
            f'{lr:.8e}',
            f'{grad_norm:.6f}',
            f'{self.scaler.get_scale():.2f}',
            f'{gpu_memory:.4f}',
            f'{gpu_cached:.4f}',
            int(has_nan),
            int(has_inf),
            int(has_zero),
            repair_action
        ])
        self.csv_file.flush()
    
    def _check_for_nan(self, loss: torch.Tensor, step: int, batch: dict) -> bool:
        """Check if loss is NaN and log debug information if detected"""
        if not torch.isnan(loss) and not torch.isinf(loss):
            # Save last valid state for debugging
            if step % 50 == 0:  # Don't save every step for memory efficiency
                self.last_valid_state = {
                    'step': step,
                    'loss': loss.item(),
                    'scaler_scale': self.scaler.get_scale()
                }
            return False
        
        # NaN detected! Log comprehensive debug information
        self.nan_detected = True
        logging.error("=" * 80)
        logging.error("🚨 NaN/Inf DETECTED IN LOSS - STOPPING TRAINING 🚨")
        logging.error("=" * 80)
        
        logging.error(f"Step: {self.global_step}, Batch index: {step}")
        logging.error(f"Loss value: {loss.item()}")
        logging.error(f"Scaler scale: {self.scaler.get_scale()}")
        
        # Log last valid state
        if self.last_valid_state:
            logging.error(f"Last valid state: step={self.last_valid_state['step']}, "
                         f"loss={self.last_valid_state['loss']:.6f}, "
                         f"scale={self.last_valid_state['scaler_scale']}")
        
        # Log learning rates for each param group
        logging.error("Learning rates by parameter group:")
        for i, group in enumerate(self.optimizer.param_groups):
            name = group.get('name', f'group_{i}')
            logging.error(f"  {name}: lr={group['lr']:.8e}")
        
        # Check gradients
        logging.error("Gradient statistics by parameter:")
        nan_params = []
        inf_params = []
        large_grad_params = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                if torch.isnan(grad).any():
                    nan_params.append(name)
                elif torch.isinf(grad).any():
                    inf_params.append(name)
                else:
                    grad_max = grad.abs().max().item()
                    if grad_max > 1000:
                        large_grad_params.append((name, grad_max))
        
        if nan_params:
            logging.error(f"Parameters with NaN gradients ({len(nan_params)}):")
            for p in nan_params[:10]:  # Limit output
                logging.error(f"  - {p}")
        
        if inf_params:
            logging.error(f"Parameters with Inf gradients ({len(inf_params)}):")
            for p in inf_params[:10]:
                logging.error(f"  - {p}")
        
        if large_grad_params:
            logging.error(f"Parameters with large gradients (>1000) ({len(large_grad_params)}):")
            for name, val in sorted(large_grad_params, key=lambda x: -x[1])[:10]:
                logging.error(f"  - {name}: {val:.2f}")
        
        # Check model weights
        logging.error("Model weight statistics:")
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                logging.error(f"  NaN weights in: {name}")
            elif torch.isinf(param).any():
                logging.error(f"  Inf weights in: {name}")
        
        # Log input batch statistics
        logging.error("Input batch statistics:")
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                logging.error(f"  {key}: shape={val.shape}, dtype={val.dtype}, "
                            f"min={val.min().item()}, max={val.max().item()}")
        
        # GPU memory state
        if torch.cuda.is_available():
            logging.error(f"GPU Memory - Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB, "
                         f"Cached: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
        
        # Log config that might affect stability
        logging.error("Model configuration (stability-relevant):")
        logging.error(f"  - BitNet enabled: {self.config.use_bitnet}")
        logging.error(f"  - ODE solver: {self.config.ode_solver}, steps: {self.config.ode_steps}")
        logging.error(f"  - Layer pattern: {self.config.layer_pattern}")
        logging.error(f"  - Num loops: {self.config.num_loops}")
        logging.error(f"  - Dropout: {self.config.dropout}")
        
        logging.error("=" * 80)
        
        return True

    def _inspect_gradients(self) -> Dict[str, float]:
        """Inspect gradients right after backward for NaN/Inf/zero and global norm validity."""
        has_nan = False
        has_inf = False
        has_zero = False
        total_norm = 0.0

        for param in self.model.parameters():
            if param.grad is None:
                continue

            grad = param.grad
            if torch.isnan(grad).any():
                has_nan = True
            if torch.isinf(grad).any():
                has_inf = True
            if (grad == 0).all():
                has_zero = True

            grad_norm = grad.norm().item()
            total_norm += grad_norm

        if not torch.isfinite(torch.tensor(total_norm, device=self.device)):
            has_inf = True

        if total_norm < self.training_config.min_grad_norm_for_signal:
            has_zero = True

        return {
            "has_nan": has_nan,
            "has_inf": has_inf,
            "has_zero": has_zero,
            "total_norm": float(total_norm),
        }

    def _apply_galore_projection(self):
        """Apply low-rank projection to gradients (GaLore-style) if enabled."""
        if not self.training_config.use_galore:
            return
        if self.global_step % max(int(self.training_config.galore_update_interval), 1) != 0:
            return

        rank = max(int(self.training_config.galore_rank), 1)
        max_dim = int(self.training_config.galore_max_dim)
        scale = float(self.training_config.galore_scale)

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if param.grad.dim() != 2:
                continue
            if "bias" in name.lower() or "embed" in name.lower():
                continue

            grad = param.grad
            rows, cols = grad.shape
            if min(rows, cols) < rank:
                continue
            if max(rows, cols) > max_dim:
                continue

            try:
                u, s, vh = torch.linalg.svd(grad.float(), full_matrices=False)
            except RuntimeError:
                continue

            u = u[:, :rank]
            s = s[:rank]
            vh = vh[:rank, :]
            approx = (u * s) @ vh
            grad.copy_(approx.to(dtype=grad.dtype) * scale)

    def _scale_learning_rate(self, factor: float):
        """Scale all optimizer param-group learning rates by a multiplicative factor."""
        for group in self.optimizer.param_groups:
            group['lr'] = max(group['lr'] * factor, 1e-9)  # Lower floor to allow more aggressive reduction
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with parameter groups for different components"""
        def _as_betas(value):
            if isinstance(value, (list, tuple)) and len(value) == 2:
                return (float(value[0]), float(value[1]))
            return (0.9, 0.95)

        # Separate parameters by component type
        ode_params = []
        retnet_params = []
        mamba_params = []
        titan_attn_params = []
        norm_params = []
        embed_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'ode' in name.lower():
                ode_params.append(param)
            elif 'retention' in name.lower() or 'retnet' in name.lower():
                retnet_params.append(param)
            elif 'mamba' in name.lower():
                mamba_params.append(param)
            elif 'titan_attn' in name.lower() or 'attention' in name.lower():
                titan_attn_params.append(param)
            elif 'norm' in name.lower() or 'layer_norm' in name.lower():
                norm_params.append(param)
            elif 'embed' in name.lower():
                embed_params.append(param)
            else:
                other_params.append(param)
        
        # Parameter groups with DRASTICALLY REDUCED learning rates for stability
        # Previous LRs were causing immediate gradient explosion (see metrics: grad_norm 2→42889)
        param_groups = [
            {
                'params': embed_params,
                'lr': self.training_config.lr_embeddings,
                'weight_decay': self.training_config.wd_embeddings,
                'betas': _as_betas(self.training_config.betas_embeddings),
                'eps': self.training_config.eps_embeddings,
                'name': 'embeddings'
            },  # Reduced 100x
            {
                'params': norm_params,
                'lr': self.training_config.lr_norms,
                'weight_decay': self.training_config.wd_norms,
                'betas': _as_betas(self.training_config.betas_norms),
                'eps': self.training_config.eps_norms,
                'name': 'norms'
            },  # Reduced 40x
            {
                'params': ode_params,
                'lr': self.training_config.lr_ode,
                'weight_decay': self.training_config.wd_ode,
                'betas': _as_betas(self.training_config.betas_ode),
                'eps': self.training_config.eps_ode,
                'name': 'ode'
            },  # Reduced 500x - ODE extremely unstable
            {
                'params': retnet_params,
                'lr': self.training_config.lr_retnet,
                'weight_decay': self.training_config.wd_retnet,
                'betas': _as_betas(self.training_config.betas_retnet),
                'eps': self.training_config.eps_retnet,
                'name': 'retnet'
            },  # Reduced 30x
            {
                'params': mamba_params,
                'lr': self.training_config.lr_mamba,
                'weight_decay': self.training_config.wd_mamba,
                'betas': _as_betas(self.training_config.betas_mamba),
                'eps': self.training_config.eps_mamba,
                'name': 'mamba'
            },  # Reduced 50x
            {
                'params': titan_attn_params,
                'lr': self.training_config.lr_attention,
                'weight_decay': self.training_config.wd_attention,
                'betas': _as_betas(self.training_config.betas_attention),
                'eps': self.training_config.eps_attention,
                'name': 'attention'
            },  # Reduced 33x
            {
                'params': other_params,
                'lr': self.training_config.lr_other,
                'weight_decay': self.training_config.wd_other,
                'betas': _as_betas(self.training_config.betas_other),
                'eps': self.training_config.eps_other,
                'name': 'other'
            }  # Reduced 50x
        ]
        
        # Filter out empty groups
        param_groups = [group for group in param_groups if group['params']]
        
        # Log parameter distribution
        for group in param_groups:
            param_count = sum(p.numel() for p in group['params'])
            logging.info(f"Parameter group '{group['name']}': {param_count:,} parameters, lr={group['lr']}")
        
        return optim.AdamW(param_groups)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup"""
        total_steps = int(self.training_config.scheduler_total_steps)
        if total_steps <= 0:
            total_steps = 10000  # Estimated total steps
        warmup_ratio = float(self.training_config.scheduler_warmup_ratio)
        warmup_ratio = min(max(warmup_ratio, 0.0), 1.0)
        warmup_steps = int(warmup_ratio * total_steps)
        
        scheduler_type = str(self.training_config.scheduler_type).strip().lower()

        def lr_lambda(step):
            if scheduler_type == "constant":
                return 1.0
            if scheduler_type == "linear_warmup_then_constant":
                if warmup_steps == 0:
                    return 1.0
                if step < warmup_steps:
                    return (step + 1) / warmup_steps
                return 1.0
            # cosine (default)
            if warmup_steps == 0:
                progress = step / max(total_steps, 1)
                progress = min(max(progress, 0.0), 1.0)
                return 0.5 * (1 + math.cos(math.pi * progress))
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _log_setup(self):
        """Log training setup information"""
        logging.info(f"🎯 Training on device: {self.device}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logging.info(f"🔥 GPU: {gpu_name}")
            logging.info(f"💾 GPU Memory: {gpu_memory:.2f}GB")
            
            # Enable optimizations for P40
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Model-specific logging
        logging.info(f"🧠 Architecture: Hybrid TITAN-BERT-ULTRA")
        logging.info(f"⚡ BitNet Quantization: {self.config.use_bitnet}")
        logging.info(f"🌊 ODE Solver: {self.config.ode_solver} ({self.config.ode_steps} steps)")
        logging.info(f"🔄 Recursion Loops: {self.config.num_loops}")
        logging.info(f"📐 Gradient Accumulation: {self.gradient_accumulation_steps} steps")
    
    def compute_mlm_loss(self, input_ids, attention_mask, labels):
        """Compute MLM loss with proper masking and shape safety checks"""
        # Forward pass
        logits = self.model(input_ids)

        # Safety: avoid propagating non-finite logits into CE (would yield NaN loss)
        if not torch.isfinite(logits).all():
            bad = (~torch.isfinite(logits)).sum().item()
            raise RuntimeError(f"Non-finite logits detected in forward pass: {bad} elements")
        
        # Reshape for loss computation
        batch_size, seq_len, vocab_size = logits.shape
        
        # CRITICAL: Ensure all tensors have matching shapes before flattening
        # Variable sequence lengths can cause shape mismatches
        if labels.shape != input_ids.shape:
            raise RuntimeError(f"Shape mismatch: labels {labels.shape} != input_ids {input_ids.shape}")
        if attention_mask.shape != input_ids.shape:
            raise RuntimeError(f"Shape mismatch: attention_mask {attention_mask.shape} != input_ids {input_ids.shape}")
        
        logits = logits.view(-1, vocab_size)
        labels = labels.view(-1)
        input_flat = input_ids.view(-1)
        attn_flat = attention_mask.view(-1)
        
        # Support both label conventions:
        # 1) labels already contain -100 on unmasked tokens (HF-style)
        # 2) labels contain full original sequence; infer masked positions by input!=label
        if (labels == -100).any():
            mlm_labels = labels.clone()
        else:
            mlm_labels = labels.clone()
            mlm_labels[labels == input_flat] = -100

        valid_mask = (mlm_labels != -100)

        # Rare fallback: if no valid MLM targets exist, force one valid supervised token
        # from attended positions to keep CE finite and training alive.
        if valid_mask.sum().item() == 0:
            candidate_positions = torch.nonzero(attn_flat > 0, as_tuple=False)
            if candidate_positions.numel() > 0:
                first_idx = int(candidate_positions[0].item())
                mlm_labels[first_idx] = labels[first_idx]
                valid_mask = (mlm_labels != -100)

        # Compute CE in float32 for numerical stability
        if valid_mask.sum().item() > 0:
            loss = F.cross_entropy(logits[valid_mask].float(), mlm_labels[valid_mask])
        else:
            # Defensive fallback (should almost never happen after forced target above)
            loss = logits.new_zeros((), requires_grad=True)
        
        # Compute accuracy on masked tokens
        with torch.no_grad():
            mask = valid_mask
            if mask.any():
                predictions = logits.argmax(dim=-1)
                correct = (predictions == mlm_labels) & mask
                accuracy = correct.sum().float() / mask.sum().float()
            else:
                accuracy = torch.tensor(0.0, device=logits.device)
        
        return loss, accuracy
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Tuple[float, bool]:
        """
        Train for one epoch with advanced monitoring.
        
        Returns:
            Tuple[float, bool]: (average_loss, should_stop)
            should_stop is True if NaN was detected
        """
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        grad_norm = 0.0
        self.current_epoch_had_inf = False
        current_epoch_had_zero = False
        
        # Progress bar with detailed metrics
        progress_bar = tqdm(
            dataloader, 
            desc=f"🚀 Epoch {epoch+1}",
            leave=True,
            ncols=120
        )
        
        self.optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move to device
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                repair_action = "none"
                retry_count = 0
                batch_done = False

                while retry_count <= self.training_config.max_nan_retries and not batch_done:
                    # Forward pass (AMP if enabled)
                    with autocast(self.device, enabled=self.use_amp, dtype=self.amp_dtype):
                        loss, accuracy = self.compute_mlm_loss(
                            batch['input_ids'],
                            batch['attention_mask'],
                            batch['labels']
                        )

                        if self._check_for_nan(loss * self.gradient_accumulation_steps, batch_idx, batch):
                            repair_action = f"nan_loss_retry_{retry_count + 1}"
                            self.optimizer.zero_grad(set_to_none=True)
                            self._scale_learning_rate(0.5)
                            retry_count += 1

                            if retry_count > self.training_config.max_nan_retries:
                                try:
                                    emergency_path = self.save_checkpoint(epoch, suffix="_nan_emergency")
                                    logging.error(f"Emergency checkpoint saved: {emergency_path}")
                                except Exception as e:
                                    logging.error(f"Failed to save emergency checkpoint: {e}")
                                return total_loss / max(num_batches, 1), True
                            continue

                        raw_loss = loss.item()
                        scaled_loss = loss / self.gradient_accumulation_steps

                    # Backward pass
                    if self.use_amp:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    grad_stats = self._inspect_gradients()
                    current_epoch_had_zero = current_epoch_had_zero or grad_stats["has_zero"]

                    # Repair policy for NaN gradients: skip step, zero grad, LR*0.75 retry up to N times
                    if grad_stats["has_nan"]:
                        repair_action = f"nan_grad_retry_{retry_count + 1}"
                        self.optimizer.zero_grad(set_to_none=True)
                        self._scale_learning_rate(0.75)  # Less aggressive reduction (was 0.5)
                        retry_count += 1

                        if retry_count > self.training_config.max_nan_retries:
                            logging.error(
                                "NaN gradients persisted after retries. Saving checkpoint and stopping training."
                            )
                            try:
                                emergency_path = self.save_checkpoint(epoch, suffix="_nan_grad_emergency")
                                logging.error(f"Emergency checkpoint saved: {emergency_path}")
                            except Exception as e:
                                logging.error(f"Failed to save emergency checkpoint: {e}")
                            return total_loss / max(num_batches, 1), True
                        continue

                    # Update every gradient_accumulation_steps
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        if self.use_amp:
                            self.scaler.unscale_(self.optimizer)

                        self._apply_galore_projection()

                        grad_norm = clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.training_config.grad_clip_max_norm
                        )
                        grad_norm_value = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

                        post_clip_bad = (not math.isfinite(grad_norm_value)) or (
                            grad_norm_value > self.training_config.inf_post_clip_threshold
                        )

                        # Repair policy for Inf/exploding gradients
                        if grad_stats["has_inf"] or post_clip_bad:
                            self.current_epoch_had_inf = True
                            repair_action = "inf_skip_step_lr_half"
                            self._scale_learning_rate(0.5)
                            self.optimizer.zero_grad(set_to_none=True)

                            self.global_step += 1
                            current_lr = self.scheduler.get_last_lr()[0]
                            self._log_step_to_csv(
                                epoch=epoch,
                                step=batch_idx,
                                loss=raw_loss,
                                accuracy=accuracy.item(),
                                lr=current_lr,
                                grad_norm=grad_norm_value,
                                has_nan=grad_stats["has_nan"],
                                has_inf=True,
                                has_zero=grad_stats["has_zero"],
                                repair_action=repair_action,
                            )
                            batch_done = True
                            continue

                        # Normal optimizer step
                        if self.use_amp:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()

                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)

                        self.global_step += 1
                        current_lr = self.scheduler.get_last_lr()[0]
                        self._log_step_to_csv(
                            epoch=epoch,
                            step=batch_idx,
                            loss=raw_loss,
                            accuracy=accuracy.item(),
                            lr=current_lr,
                            grad_norm=grad_norm_value,
                            has_nan=grad_stats["has_nan"],
                            has_inf=grad_stats["has_inf"],
                            has_zero=grad_stats["has_zero"],
                            repair_action=repair_action,
                        )

                        if self.global_step % self.training_config.checkpoint_every_n_steps == 0:
                            self._save_rolling_checkpoint(epoch)

                        self._maybe_save_best_checkpoint(epoch, raw_loss)

                    # Accumulate metrics
                    total_loss += raw_loss
                    total_accuracy += accuracy.item()
                    num_batches += 1
                    batch_done = True

                if not batch_done:
                    return total_loss / max(num_batches, 1), True
                
                # Update progress bar
                current_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{(total_loss / max(num_batches, 1)):.4f}',
                    'acc': f'{accuracy.item():.3f}',
                    'lr': f'{current_lr:.2e}',
                    'step': self.global_step
                })
                
                # Memory management every 50 steps
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Detailed logging every 100 steps
                if batch_idx % 100 == 0 and batch_idx > 0:
                    avg_loss = total_loss / num_batches
                    avg_acc = total_accuracy / num_batches
                    
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated() / 1024**3
                        memory_cached = torch.cuda.memory_reserved() / 1024**3
                        logging.info(
                            f"Step {self.global_step}: Loss={avg_loss:.4f}, "
                            f"Acc={avg_acc:.3f}, GPU={memory_used:.2f}GB"
                        )
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error(f"OOM at batch {batch_idx}, clearing cache and continuing...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                else:
                    raise e
        
        # Final epoch metrics
        avg_loss = total_loss / max(num_batches, 1)
        avg_accuracy = total_accuracy / max(num_batches, 1)
        
        # Update best loss / plateau tracking
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.non_improving_epochs = 0
            logging.info(f"🎉 New best loss: {self.best_loss:.4f}")
        else:
            self.non_improving_epochs += 1

        if self.current_epoch_had_inf:
            self.consecutive_inf_epochs += 1
        else:
            self.consecutive_inf_epochs = 0

        if current_epoch_had_zero and self.non_improving_epochs >= self.training_config.zero_grad_plateau_patience:
            self.consecutive_zero_grad_plateau_epochs += 1
        else:
            self.consecutive_zero_grad_plateau_epochs = 0
        
        logging.info(
            f"📊 Epoch {epoch+1} Summary: "
            f"Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.3f}, "
            f"Best Loss={self.best_loss:.4f}, "
            f"InfEpochs={self.consecutive_inf_epochs}, "
            f"ZeroGradPlateauEpochs={self.consecutive_zero_grad_plateau_epochs}"
        )

        # Early stop policy: unrecoverable exploding gradients across epochs
        if self.consecutive_inf_epochs > self.training_config.inf_epoch_patience:
            logging.error(
                "Early stop: Inf/exploding gradients persisted for too many consecutive epochs."
            )
            return avg_loss, True

        # Early stop policy: vanishing gradients + plateau persistence
        if self.consecutive_zero_grad_plateau_epochs >= 1:
            logging.error(
                "Early stop: persistent near-zero gradients with plateaued loss detected."
            )
            return avg_loss, True
        
        return avg_loss, False  # No NaN detected
    
    def _save_rolling_checkpoint(self, epoch: int, path: str = "checkpoints"):
        """Save a rolling checkpoint and manage the queue"""
        os.makedirs(path, exist_ok=True)
        
        filename = f"titan_rolling_step_{self.global_step}.pt"
        checkpoint_path = os.path.join(path, filename)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss,
            'model_class': self.model.__class__.__name__
        }
        
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"💾 Rolling checkpoint saved: {checkpoint_path}")
        
        # Add to queue
        self.rolling_checkpoints.append(checkpoint_path)
        
        # Remove old checkpoints if exceeding max
        while len(self.rolling_checkpoints) > self.training_config.max_rolling_checkpoints:
            old_checkpoint = self.rolling_checkpoints.pop(0)
            try:
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
                    logging.info(f"🗑️ Removed old rolling checkpoint: {old_checkpoint}")
            except Exception as e:
                logging.warning(f"Failed to remove old checkpoint {old_checkpoint}: {e}")
    
    def _maybe_save_best_checkpoint(self, epoch: int, current_loss: float, path: str = "checkpoints"):
        """Save checkpoint if it's among the top K best models"""
        os.makedirs(path, exist_ok=True)
        
        k = self.training_config.num_best_checkpoints
        
        # Use negative loss because heapq is a min-heap
        # We want to keep the K smallest losses (best models)
        should_save = False
        
        if len(self.best_checkpoints) < k:
            should_save = True
        elif current_loss < -self.best_checkpoints[0][0]:  # Compare with worst of the best
            should_save = True
        
        if should_save:
            filename = f"titan_best_loss_{current_loss:.6f}_step_{self.global_step}.pt"
            checkpoint_path = os.path.join(path, filename)
            
            checkpoint = {
                'epoch': epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'config': self.config,
                'best_loss': current_loss,
                'model_class': self.model.__class__.__name__
            }
            
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"⭐ Best model checkpoint saved: {checkpoint_path} (loss={current_loss:.6f})")
            
            # Add to heap (using negative loss for min-heap to act as max-heap)
            heapq.heappush(self.best_checkpoints, (-current_loss, checkpoint_path))
            
            # Remove worst of the best if exceeding K
            if len(self.best_checkpoints) > k:
                _, removed_path = heapq.heappop(self.best_checkpoints)
                try:
                    if os.path.exists(removed_path):
                        os.remove(removed_path)
                        logging.info(f"🗑️ Removed replaced best checkpoint: {removed_path}")
                except Exception as e:
                    logging.warning(f"Failed to remove old best checkpoint {removed_path}: {e}")
    
    def close(self):
        """Cleanup resources"""
        if self.csv_file:
            self.csv_file.close()
            logging.info(f"📊 CSV log closed: {self.training_config.csv_log_path}")
    
    def save_checkpoint(self, epoch: int, suffix: str = "", path: str = "checkpoints") -> str:
        """Save comprehensive checkpoint"""
        os.makedirs(path, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss,
            'model_class': self.model.__class__.__name__
        }
        
        filename = f"titan_checkpoint_epoch_{epoch}{suffix}.pt"
        checkpoint_path = os.path.join(path, filename)
        
        torch.save(checkpoint, checkpoint_path)
        
        # Register with storage manager
        if self.storage_manager.register_file(checkpoint_path):
            logging.info(f"💾 Checkpoint saved: {checkpoint_path}")
        else:
            logging.warning(f"⚠️ Checkpoint saved but storage limit warning")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training"""
        logging.info(f"📂 Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        logging.info(f"✅ Checkpoint loaded - Global Step: {self.global_step}, Best Loss: {self.best_loss:.4f}")
        
        return checkpoint['epoch']
