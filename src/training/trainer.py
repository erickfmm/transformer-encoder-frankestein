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

    # Post-backward stability
    max_nan_retries: int = 3
    grad_clip_max_norm: float = 1.0
    inf_post_clip_threshold: float = 10.0
    min_grad_norm_for_signal: float = 1e-10
    inf_epoch_patience: int = 5
    zero_grad_plateau_patience: int = 5

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
        self.use_amp = torch.cuda.is_available() and str(self.device).startswith("cuda")
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
        self.gradient_accumulation_steps = 4
        
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

    def _scale_learning_rate(self, factor: float):
        """Scale all optimizer param-group learning rates by a multiplicative factor."""
        for group in self.optimizer.param_groups:
            group['lr'] = max(group['lr'] * factor, 1e-8)
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with parameter groups for different components"""
        
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
        
        # Parameter groups with different learning rates
        param_groups = [
            {'params': embed_params, 'lr': 1e-4, 'weight_decay': 0.01, 'name': 'embeddings'},
            {'params': norm_params, 'lr': 2e-4, 'weight_decay': 0.001, 'name': 'norms'},
            {'params': ode_params, 'lr': 5e-5, 'weight_decay': 0.01, 'name': 'ode'},  # Lower LR for ODE stability
            {'params': retnet_params, 'lr': 1.5e-4, 'weight_decay': 0.01, 'name': 'retnet'},
            {'params': mamba_params, 'lr': 1e-4, 'weight_decay': 0.01, 'name': 'mamba'},
            {'params': titan_attn_params, 'lr': 1e-4, 'weight_decay': 0.01, 'name': 'attention'},
            {'params': other_params, 'lr': 1e-4, 'weight_decay': 0.01, 'name': 'other'}
        ]
        
        # Filter out empty groups
        param_groups = [group for group in param_groups if group['params']]
        
        # Log parameter distribution
        for group in param_groups:
            param_count = sum(p.numel() for p in group['params'])
            logging.info(f"Parameter group '{group['name']}': {param_count:,} parameters, lr={group['lr']}")
        
        return optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup"""
        total_steps = 10000  # Estimated total steps
        warmup_steps = int(0.1 * total_steps)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
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
        """Compute MLM loss with proper masking"""
        # Forward pass
        logits = self.model(input_ids)
        
        # Reshape for loss computation
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)
        labels = labels.view(-1)
        
        # Only compute loss on masked tokens (labels != -100)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
        # Update labels: keep original for masked positions, set -100 for others
        mlm_labels = labels.clone()
        mlm_labels[labels == input_ids.view(-1)] = -100  # Don't predict non-masked tokens
        
        loss = loss_fct(logits, mlm_labels)
        
        # Compute accuracy on masked tokens
        with torch.no_grad():
            mask = (mlm_labels != -100)
            if mask.sum() > 0:
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
                    with autocast(self.device, enabled=self.use_amp):
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

                    # Repair policy for NaN gradients: skip step, zero grad, LR/2 retry up to N times
                    if grad_stats["has_nan"]:
                        repair_action = f"nan_grad_retry_{retry_count + 1}"
                        self.optimizer.zero_grad(set_to_none=True)
                        self._scale_learning_rate(0.5)
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