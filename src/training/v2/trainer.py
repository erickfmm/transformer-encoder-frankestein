import os
import torch
import logging
import math
from tqdm import tqdm
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from utils.storage_manager import StorageManager
from model.v2.titan_bert_ultra import TitanBertUltra, UltraConfig

# ==================== TITAN TRAINER ====================
class TitanTrainer:
    """Advanced trainer for TITAN-BERT-ULTRA with specialized optimizations"""
    
    def __init__(self, model: TitanBertUltra, config: UltraConfig, device: torch.device = None):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Specialized optimizer for hybrid architecture
        self.optimizer = self._setup_optimizer()
        
        # Mixed precision with aggressive scaling for BitNet
        self.scaler = GradScaler(
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
        
        self._log_setup()
    
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
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch with advanced monitoring"""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        # Progress bar with detailed metrics
        progress_bar = tqdm(
            dataloader, 
            desc=f"🚀 Epoch {epoch+1}",
            leave=True,
            ncols=120
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move to device
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with autocast():
                    loss, accuracy = self.compute_mlm_loss(
                        batch['input_ids'],
                        batch['attention_mask'], 
                        batch['labels']
                    )
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Update every gradient_accumulation_steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping before scaling
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                
                # Accumulate metrics
                total_loss += loss.item() * self.gradient_accumulation_steps
                total_accuracy += accuracy.item()
                num_batches += 1
                
                # Update progress bar
                current_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
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
                    self.optimizer.zero_grad()
                    continue
                else:
                    raise e
        
        # Final epoch metrics
        avg_loss = total_loss / max(num_batches, 1)
        avg_accuracy = total_accuracy / max(num_batches, 1)
        
        # Update best loss
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            logging.info(f"🎉 New best loss: {self.best_loss:.4f}")
        
        logging.info(
            f"📊 Epoch {epoch+1} Summary: "
            f"Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.3f}, "
            f"Best Loss={self.best_loss:.4f}"
        )
        
        return avg_loss
    
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
            'model_class': 'TitanBertUltra'
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