"""
Meta-learning trainer for ProtoNet.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Optional
import time

from models.protonet import ProtoNet
from data.sampler import EpisodicSampler

logger = logging.getLogger(__name__)


class ProtoNetTrainer:
    """
    Trainer for Prototypical Networks.
    
    Handles meta-training loop with episodic sampling.
    """
    
    def __init__(
        self,
        model: ProtoNet,
        train_sampler: EpisodicSampler,
        val_sampler: EpisodicSampler,
        config: dict,
        device: str = 'cuda'
    ):
        """
        Args:
            model: ProtoNet model
            train_sampler: Training episode sampler
            val_sampler: Validation episode sampler
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.writer = None
        if config.get('tensorboard', True):
            tensorboard_dir = config.get('tensorboard_dir', 'runs')
            self.writer = SummaryWriter(tensorboard_dir)
        
        # Training state
        self.current_episode = 0
        self.best_val_acc = 0.0
        
        # Early stopping
        self.patience = config.get('patience', 2000)
        self.min_delta = config.get('min_delta', 0.001)
        self.patience_counter = 0
        
        logger.info("Trainer initialized")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        optimizer_type = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.0005)
        
        if optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_type = self.config.get('lr_scheduler', 'step').lower()
        
        if scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('lr_step_size', 2000),
                gamma=self.config.get('lr_gamma', 0.5)
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('num_episodes', 10000)
            )
        elif scheduler_type is None or scheduler_type == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    def train_episode(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Train on one episode.
        
        Args:
            support_images: (N*K, C, H, W)
            support_labels: (N*K,)
            query_images: (N*Q, C, H, W)
            query_labels: (N*Q,)
            
        Returns:
            Dictionary with loss and accuracy
        """
        self.model.train()
        
        # Move to device
        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        query_labels = query_labels.to(self.device)
        
        # Forward pass
        output = self.model(support_images, support_labels, query_images, query_labels)
        
        loss = output['loss']
        logits = output['logits']
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Compute accuracy
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == query_labels).float().mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation episodes.
        
        Returns:
            Dictionary with average loss and accuracy
        """
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        n_episodes = len(self.val_sampler)
        
        with torch.no_grad():
            for support_images, support_labels, query_images, query_labels in self.val_sampler:
                # Move to device
                support_images = support_images.to(self.device)
                support_labels = support_labels.to(self.device)
                query_images = query_images.to(self.device)
                query_labels = query_labels.to(self.device)
                
                # Forward pass
                output = self.model(support_images, support_labels, query_images, query_labels)
                
                loss = output['loss']
                logits = output['logits']
                
                # Compute accuracy
                predictions = logits.argmax(dim=1)
                accuracy = (predictions == query_labels).float().mean().item()
                
                total_loss += loss.item()
                total_accuracy += accuracy
        
        avg_loss = total_loss / n_episodes
        avg_accuracy = total_accuracy / n_episodes
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': avg_accuracy
        }
    
    def train(self):
        """
        Main training loop.
        """
        num_episodes = self.config.get('num_episodes', 10000)
        val_interval = self.config.get('validation_interval', 100)
        save_interval = self.config.get('save_interval', 500)
        
        logger.info(f"Starting training for {num_episodes} episodes")
        
        start_time = time.time()
        pbar = tqdm(total=num_episodes, desc="Training")
        
        for episode_idx, (support_images, support_labels, query_images, query_labels) in enumerate(self.train_sampler):
            self.current_episode = episode_idx
            
            # Train episode
            metrics = self.train_episode(support_images, support_labels, query_images, query_labels)
            
            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar('train/loss', metrics['loss'], episode_idx)
                self.writer.add_scalar('train/accuracy', metrics['accuracy'], episode_idx)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], episode_idx)
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']:.4f}"
            })
            
            # Validation
            if (episode_idx + 1) % val_interval == 0:
                val_metrics = self.validate()
                
                logger.info(
                    f"Episode {episode_idx + 1}/{num_episodes} - "
                    f"Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"Val Acc: {val_metrics['val_accuracy']:.4f}"
                )
                
                if self.writer:
                    self.writer.add_scalar('val/loss', val_metrics['val_loss'], episode_idx)
                    self.writer.add_scalar('val/accuracy', val_metrics['val_accuracy'], episode_idx)
                
                # Check for improvement
                if val_metrics['val_accuracy'] > self.best_val_acc + self.min_delta:
                    self.best_val_acc = val_metrics['val_accuracy']
                    self.patience_counter = 0
                    
                    # Save best model
                    self.save_checkpoint('best_model.pt')
                    logger.info(f"New best validation accuracy: {self.best_val_acc:.4f}")
                else:
                    self.patience_counter += val_interval
                
                # Early stopping
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered after {episode_idx + 1} episodes")
                    break
            
            # Periodic checkpointing
            if (episode_idx + 1) % save_interval == 0:
                self.save_checkpoint(f'checkpoint_episode_{episode_idx + 1}.pt')
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
        
        pbar.close()
        
        # Training complete
        elapsed_time = time.time() - start_time
        logger.info(f"Training complete in {elapsed_time:.2f}s")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        
        if self.writer:
            self.writer.close()
    
    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'episode': self.current_episode,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_episode = checkpoint['episode']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resuming from episode {self.current_episode}")
