import os
import torch
import logging
from tqdm import tqdm
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from utils.storage_manager import StorageManager
from model.v1.model import SpanishMoEBERT

# ==================== DATASET & TRAINING ====================

class Trainer:
    """Training manager with storage monitoring"""
    
    def __init__(self, model: SpanishMoEBERT, device: torch.device = None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
        self.scaler = GradScaler()
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, eta_min=1e-6
        )
        
        # Storage monitoring
        self.storage_manager = StorageManager()
        
        logging.info(f"Training on device: {self.device}")
        if torch.cuda.is_available():
            logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward with mixed precision
            with autocast():
                outputs = self.model(**batch)
                loss = outputs["loss"]
            
            # Backward
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Memory management
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return total_loss / len(dataloader)
    
    def save_checkpoint(self, epoch: int, path: str = "checkpoints"):
        """Save model checkpoint"""
        os.makedirs(path, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.model.config,
        }
        
        checkpoint_path = f"{path}/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Check storage
        if self.storage_manager.register_file(checkpoint_path):
            logging.info(f"Checkpoint saved: {checkpoint_path}")
        else:
            logging.warning(f"Checkpoint saved but storage limit warning")
        
        return checkpoint_path
