"""
Utility functions for training diffusion models.
"""

import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from typing import Optional, Dict, Any
import json
import logging
from datetime import datetime


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    cudnn.deterministic = True
    cudnn.benchmark = False


def get_device(device_str: str = 'auto') -> torch.device:
    """
    Get appropriate device for training.
    
    Args:
        device_str (str): Device specification ('auto', 'cpu', 'cuda', 'mps')
    
    Returns:
        torch.device: Selected device
    """
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_str)
    
    print(f"Using device: {device}")
    return device


def create_run_dir(save_dir: str, run_name: str) -> str:
    """
    Create run directory for saving results.
    
    Args:
        save_dir (str): Base save directory
        run_name (str): Name of the run
    
    Returns:
        str: Created run directory path
    """
    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'logs'), exist_ok=True)
    return run_dir


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    loss: float,
    run_dir: str,
    filename: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        scheduler: Learning rate scheduler state to save
        epoch: Current epoch
        loss: Current loss value
        run_dir: Run directory
        filename: Custom filename (optional)
        config: Training configuration (optional)
    
    Returns:
        str: Path to saved checkpoint
    """
    if filename is None:
        filename = f'checkpoint_epoch_{epoch}.pth'
    
    checkpoint_path = os.path.join(run_dir, 'checkpoints', filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if config is not None:
        checkpoint['config'] = config
    
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest checkpoint
    latest_path = os.path.join(run_dir, 'checkpoints', 'latest.pth')
    torch.save(checkpoint, latest_path)
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load tensors to (optional)
        strict: Whether to strictly enforce state dict keys match
    
    Returns:
        dict: Checkpoint metadata (epoch, loss, etc.)
    """
    if device is None:
        device = next(model.parameters()).device
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'timestamp': checkpoint.get('timestamp', 'unknown'),
        'config': checkpoint.get('config', {}),
    }
    
    print(f"Loaded checkpoint from epoch {metadata['epoch']} with loss {metadata['loss']:.6f}")
    
    return metadata


def setup_logging(run_dir: str, log_level: str = 'INFO') -> logging.Logger:
    """
    Setup logging for training.
    
    Args:
        run_dir: Run directory for log files
        log_level: Logging level
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger('diffusion_training')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = os.path.join(run_dir, 'logs', 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def save_config(config: Dict[str, Any], run_dir: str):
    """
    Save training configuration to file.
    
    Args:
        config: Configuration dictionary
        run_dir: Run directory
    """
    config_path = os.path.join(run_dir, 'config.json')
    
    # Convert non-serializable objects to strings
    serializable_config = {}
    for key, value in config.items():
        try:
            json.dumps(value)
            serializable_config[key] = value
        except (TypeError, ValueError):
            serializable_config[key] = str(value)
    
    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)


def count_parameters(model: torch.nn.Module) -> tuple:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.
    
    Returns:
        dict: Memory usage statistics
    """
    memory_info = {}
    
    if torch.cuda.is_available():
        memory_info['cuda_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_info['cuda_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
        memory_info['cuda_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    return memory_info
