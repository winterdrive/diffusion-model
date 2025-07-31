"""
Main trainer class for diffusion model training and evaluation.
"""

import os
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ddpm import DDPM
from models.unet import UNet
from .config import TrainingConfig
from .dataset import create_data_loaders, get_object_names
from .utils import (
    set_seed, get_device, create_run_dir, save_checkpoint, 
    load_checkpoint, setup_logging, save_config, count_parameters,
    format_time, get_memory_usage
)


class DiffusionTrainer:
    """
    Main trainer class for diffusion model training and evaluation.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        # Set random seed
        set_seed(config.seed)
        
        # Setup device
        self.device = get_device(config.device)
        
        # Create run directory
        self.run_dir = create_run_dir(config.save_dir, config.run_name)
        
        # Setup logging
        self.logger = setup_logging(self.run_dir)
        self.logger.info(f"Starting training run: {config.run_name}")
        
        # Save configuration
        save_config(vars(config), self.run_dir)
        
        # Initialize models
        self._setup_models()
        
        # Setup data loaders
        self._setup_data()
        
        # Setup training components
        self._setup_training()
        
        # Initialize wandb if requested
        if config.use_wandb:
            self._setup_wandb()
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'epoch_times': [],
            'learning_rates': []
        }
    
    def _setup_models(self):
        """Setup UNet and DDPM models."""
        self.logger.info("Setting up models...")
        
        # Get object names for num_classes
        object_names = get_object_names(self.config.data_dir)
        num_classes = len(object_names)
        
        # Create UNet model
        self.unet = UNet(
            img_size=self.config.img_size,
            in_channels=3,
            out_channels=3,
            base_channels=self.config.base_channels,
            channel_multipliers=self.config.channel_multipliers,
            num_res_blocks=self.config.num_res_blocks,
            attention_resolutions=self.config.attention_resolutions,
            num_classes=num_classes,
            dropout=self.config.dropout,
            use_attention=self.config.use_attention,
        ).to(self.device)
        
        # Create DDPM model
        self.ddpm = DDPM(
            model=self.unet,
            timesteps=self.config.timesteps,
            beta_schedule=self.config.beta_schedule,
            loss_type=self.config.loss_type,
            use_classifier_guidance=self.config.use_classifier_guidance,
            classifier_guidance_scale=self.config.classifier_guidance_scale,
            classifier_guidance_start_step=self.config.classifier_guidance_start_step,
        ).to(self.device)
        
        # Log model information
        total_params, trainable_params = count_parameters(self.unet)
        self.logger.info(f"UNet parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Load pretrained model if specified
        if self.config.model_ckpt:
            self._load_pretrained_model()
    
    def _setup_data(self):
        """Setup data loaders."""
        self.logger.info("Setting up data loaders...")
        
        self.data_loaders = create_data_loaders(
            data_dir=self.config.data_dir,
            batch_size=self.config.batch_size,
            img_size=(self.config.img_size, self.config.img_size),
            num_workers=self.config.num_workers,
            use_preset_size=self.config.use_preset_size
        )
        
        # Log dataset information
        if 'train' in self.data_loaders:
            train_size = len(self.data_loaders['train'].dataset)
            self.logger.info(f"Training dataset size: {train_size}")
        
        if 'test' in self.data_loaders:
            test_size = len(self.data_loaders['test'].dataset)
            self.logger.info(f"Test dataset size: {test_size}")
    
    def _setup_training(self):
        """Setup optimizer, scheduler, and other training components."""
        self.logger.info("Setting up training components...")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.unet.parameters(),
            lr=self.config.lr,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.lr * self.config.min_lr_factor
        )
        
        # Load checkpoint if resuming training
        if self.config.resume_training and self.config.model_ckpt:
            self._resume_training()
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        try:
            import wandb
            
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_name or self.config.run_name,
                config=vars(self.config),
                dir=self.run_dir
            )
            
            self.wandb = wandb
            self.logger.info("Weights & Biases logging initialized")
        except ImportError:
            self.logger.warning("wandb not installed, skipping W&B logging")
            self.wandb = None
    
    def _load_pretrained_model(self):
        """Load pretrained model."""
        self.logger.info(f"Loading pretrained model from: {self.config.model_ckpt}")
        
        try:
            load_checkpoint(
                self.config.model_ckpt,
                self.unet,
                device=self.device,
                strict=False
            )
            self.logger.info("Pretrained model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading pretrained model: {e}")
            raise
    
    def _resume_training(self):
        """Resume training from checkpoint."""
        self.logger.info(f"Resuming training from: {self.config.model_ckpt}")
        
        try:
            metadata = load_checkpoint(
                self.config.model_ckpt,
                self.unet,
                self.optimizer if not self.config.reset_optimizer else None,
                self.scheduler if not self.config.reset_optimizer else None,
                device=self.device
            )
            
            self.current_epoch = metadata['epoch']
            self.best_loss = metadata['loss']
            
            self.logger.info(f"Resumed from epoch {self.current_epoch}")
        except Exception as e:
            self.logger.error(f"Error resuming training: {e}")
            raise
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            float: Average training loss for the epoch
        """
        self.unet.train()
        total_loss = 0.0
        num_batches = len(self.data_loaders['train'])
        
        progress_bar = tqdm(
            self.data_loaders['train'],
            desc=f"Epoch {self.current_epoch}",
            leave=False
        )
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            loss = self.ddpm(images, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.unet.parameters(), 
                        self.config.grad_clip
                    )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Update statistics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.config.gradient_accumulation_steps:.6f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log to wandb
            if self.wandb and batch_idx % 100 == 0:
                self.wandb.log({
                    'batch_loss': loss.item() * self.config.gradient_accumulation_steps,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': self.current_epoch,
                    'batch': batch_idx
                })
        
        # Handle any remaining gradients
        if num_batches % self.config.gradient_accumulation_steps != 0:
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.unet.parameters(), 
                    self.config.grad_clip
                )
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def generate_samples(self, num_samples: int = 16, save_path: Optional[str] = None) -> torch.Tensor:
        """
        Generate sample images.
        
        Args:
            num_samples: Number of samples to generate
            save_path: Path to save sample images (optional)
        
        Returns:
            torch.Tensor: Generated samples
        """
        self.unet.eval()
        
        with torch.no_grad():
            # Use test data for conditional generation if available
            if 'test' in self.data_loaders:
                test_loader = self.data_loaders['test']
                test_batch = next(iter(test_loader))
                _, test_labels = test_batch
                
                # Take first num_samples labels
                labels = test_labels[:num_samples].to(self.device)
            else:
                labels = None
            
            # Generate samples
            samples = self.ddpm.sample(
                batch_size=num_samples,
                class_labels=labels
            )
            
            # Denormalize samples from [-1, 1] to [0, 1]
            samples = (samples + 1) / 2
            samples = torch.clamp(samples, 0, 1)
            
            # Save samples if path provided
            if save_path:
                self._save_sample_grid(samples, save_path)
        
        return samples
    
    def _save_sample_grid(self, samples: torch.Tensor, save_path: str):
        """Save sample images as a grid."""
        import torchvision.utils as vutils
        
        # Create grid
        grid = vutils.make_grid(
            samples.cpu(),
            nrow=int(np.sqrt(samples.shape[0])),
            padding=2,
            normalize=False
        )
        
        # Convert to PIL and save
        grid_np = grid.permute(1, 2, 0).numpy()
        grid_pil = Image.fromarray((grid_np * 255).astype(np.uint8))
        grid_pil.save(save_path)
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train for one epoch
            avg_loss = self.train_epoch()
            
            # Update learning rate
            self.scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Update training history
            self.training_history['train_loss'].append(avg_loss)
            self.training_history['epoch_times'].append(epoch_time)
            self.training_history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Log epoch results
            self.logger.info(
                f"Epoch {epoch}: loss={avg_loss:.6f}, "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}, "
                f"time={format_time(epoch_time)}"
            )
            
            # Log to wandb
            if self.wandb:
                self.wandb.log({
                    'epoch': epoch,
                    'train_loss': avg_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch_time': epoch_time
                })
            
            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint_freq == 0:
                checkpoint_path = save_checkpoint(
                    self.unet,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    avg_loss,
                    self.run_dir,
                    config=vars(self.config)
                )
                self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Generate samples
            if (epoch + 1) % self.config.sample_freq == 0:
                sample_path = os.path.join(
                    self.run_dir, 'samples', f'samples_epoch_{epoch}.png'
                )
                samples = self.generate_samples(save_path=sample_path)
                self.logger.info(f"Samples saved: {sample_path}")
                
                # Log samples to wandb
                if self.wandb:
                    self.wandb.log({
                        'samples': self.wandb.Image(sample_path),
                        'epoch': epoch
                    })
            
            # Update best loss
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                best_checkpoint_path = save_checkpoint(
                    self.unet,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    avg_loss,
                    self.run_dir,
                    filename='best_model.pth',
                    config=vars(self.config)
                )
                self.logger.info(f"Best model saved: {best_checkpoint_path}")
        
        # Training completed
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {format_time(total_time)}")
        
        # Save final model
        final_checkpoint_path = save_checkpoint(
            self.unet,
            self.optimizer,
            self.scheduler,
            self.config.epochs - 1,
            avg_loss,
            self.run_dir,
            filename='final_model.pth',
            config=vars(self.config)
        )
        self.logger.info(f"Final model saved: {final_checkpoint_path}")
        
        # Generate final samples
        final_sample_path = os.path.join(
            self.run_dir, 'samples', 'final_samples.png'
        )
        self.generate_samples(save_path=final_sample_path)
        self.logger.info(f"Final samples saved: {final_sample_path}")
        
        # Plot training curves
        self._plot_training_curves()
        
        if self.wandb:
            self.wandb.finish()
    
    def _plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Training loss
        axes[0, 0].plot(self.training_history['train_loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Learning rate
        axes[0, 1].plot(self.training_history['learning_rates'])
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        
        # Epoch times
        axes[1, 0].plot(self.training_history['epoch_times'])
        axes[1, 0].set_title('Epoch Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].grid(True)
        
        # Memory usage (if available)
        memory_info = get_memory_usage()
        if memory_info:
            axes[1, 1].bar(memory_info.keys(), memory_info.values())
            axes[1, 1].set_title('Memory Usage (GB)')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'Memory info\nnot available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Memory Usage')
        
        plt.tight_layout()
        plot_path = os.path.join(self.run_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training curves saved: {plot_path}")
