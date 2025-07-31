"""
Configuration settings for diffusion model training.
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TrainingConfig:
    """
    Configuration class for diffusion model training.
    """
    # Execution mode
    mode: str = 'train'  # 'train', 'inference', 'sample'
    
    # Data and paths
    data_dir: str = './data/iclevr'
    save_dir: str = './results'
    run_name: Optional[str] = None
    model_ckpt: Optional[str] = None
    resume_training: bool = False
    evaluator_ckpt: Optional[str] = None
    use_preset_size: bool = False
    
    # Checkpoint and sampling frequencies
    checkpoint_freq: int = 10
    sample_freq: int = 5
    
    # Training parameters
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-4
    min_lr_factor: float = 0.05
    img_size: int = 64
    grad_clip: float = 1.0
    reset_optimizer: bool = False
    gradient_accumulation_steps: int = 1
    
    # Diffusion model parameters
    timesteps: int = 400
    sampling_timesteps: int = 50
    beta_schedule: str = 'cosine'  # 'linear' or 'cosine'
    loss_type: str = 'l1'  # 'l1', 'l2', or 'huber'
    
    # Classifier guidance parameters
    use_classifier_guidance: bool = False
    classifier_guidance_scale: float = 1.0
    classifier_guidance_start_step: int = 0
    
    # Model architecture parameters
    base_channels: int = 128
    channel_multipliers: List[int] = None
    num_res_blocks: int = 2
    attention_resolutions: List[int] = None
    dropout: float = 0.1
    use_attention: bool = True
    
    # Memory optimization
    reduce_memory: bool = False
    
    # Logging and monitoring
    use_wandb: bool = False
    wandb_project: str = 'diffusion-model'
    wandb_name: Optional[str] = None
    
    # System settings
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'mps'
    seed: int = 42
    num_workers: int = 4
    
    # Data preprocessing
    preprocess_method: str = 'direct_resize'  # 'direct_resize', 'pad_resize'
    
    def __post_init__(self):
        """Post-initialization to set default values."""
        if self.channel_multipliers is None:
            self.channel_multipliers = [1, 2, 4, 8]
        
        if self.attention_resolutions is None:
            self.attention_resolutions = [16, 8]
        
        if self.run_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"diffusion_model_{timestamp}"


def get_args():
    """
    Parse command line arguments and return training configuration.
    
    Returns:
        TrainingConfig: Parsed configuration object
    """
    parser = argparse.ArgumentParser(description='Conditional DDPM Training and Testing')
    
    # Execution mode
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'inference', 'sample'],
                        help='Execution mode: train, inference, or sample')
    
    # Data and paths
    parser.add_argument('--data_dir', type=str, default='./data/iclevr', 
                        help='Dataset directory path')
    parser.add_argument('--save_dir', type=str, default='./results', 
                        help='Results save directory path')
    parser.add_argument('--run_name', type=str, default=None, 
                        help='Custom run folder name (required)')
    parser.add_argument('--model_ckpt', type=str, default=None, 
                        help='Model checkpoint path (for loading pretrained model)')
    parser.add_argument('--resume_training', action='store_true', 
                        help='Resume training (loads optimizer and training state)')
    parser.add_argument('--evaluator_ckpt', type=str, default=None, 
                        help='Evaluator checkpoint absolute path')
    parser.add_argument('--use_preset_size', action='store_true', 
                        help='Use preset resized images, skip resize operation')
    parser.add_argument('--checkpoint_freq', type=int, default=10, 
                        help='Checkpoint save frequency (every N epochs)')
    parser.add_argument('--sample_freq', type=int, default=5, 
                        help='Sample generation frequency (every N epochs)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--min_lr_factor', type=float, default=0.05, 
                        help='Minimum learning rate factor for cosine annealing')
    parser.add_argument('--img_size', type=int, default=64, 
                        help='Generated image size (must be 64 for evaluation compatibility)')
    parser.add_argument('--grad_clip', type=float, default=1.0, 
                        help='Gradient clipping threshold (<=0 means no clipping)')
    parser.add_argument('--reset_optimizer', action='store_true', 
                        help='Reset optimizer state even when loading checkpoint')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                        help='Gradient accumulation steps')
    
    # Diffusion model parameters
    parser.add_argument('--timesteps', type=int, default=400, 
                        help='Number of diffusion steps (originally 200, recommended 400)')
    parser.add_argument('--sampling_timesteps', type=int, default=50, 
                        help='Number of sampling steps (originally 100, DDIM acceleration to 50)')
    parser.add_argument('--beta_schedule', type=str, default='cosine', 
                        choices=['linear', 'cosine'], 
                        help='Beta schedule method (originally linear, recommended cosine)')
    parser.add_argument('--loss_type', type=str, default='l1', 
                        choices=['l1', 'l2', 'huber'], help='Loss function type')
    
    # Classifier guidance parameters
    parser.add_argument('--use_classifier_guidance', action='store_true', 
                        help='Use classifier guidance sampling')
    parser.add_argument('--classifier_guidance_scale', type=float, default=1.0, 
                        help='Classifier guidance strength (0.0=no guidance, 2.0-5.0 enhances label accuracy)')
    parser.add_argument('--classifier_guidance_start_step', type=int, default=0, 
                        help='Step to start classifier guidance (0=from beginning)')
    
    # Model architecture parameters
    parser.add_argument('--base_channels', type=int, default=128, 
                        help='Base number of channels in UNet')
    parser.add_argument('--num_res_blocks', type=int, default=2, 
                        help='Number of residual blocks per level')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--use_attention', action='store_true', 
                        help='Use attention layers in UNet')
    
    # Memory optimization
    parser.add_argument('--reduce_memory', action='store_true', 
                        help='Reduce memory usage (may affect performance)')
    
    # Logging and monitoring
    parser.add_argument('--use_wandb', action='store_true', 
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='diffusion-model', 
                        help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default=None, 
                        help='W&B run name')
    
    # System settings
    parser.add_argument('--device', type=str, default='auto', 
                        help='Device to use: auto, cpu, cuda, mps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of data loader workers')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert to TrainingConfig
    config = TrainingConfig()
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config
