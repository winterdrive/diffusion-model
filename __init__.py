"""
Conditional Diffusion Model for i-CLEVR Dataset Generation.

This package implements a conditional Denoising Diffusion Probabilistic Model (DDPM)
for generating i-CLEVR dataset images based on object specifications.
"""

__version__ = "1.0.0"
__author__ = "Diffusion Model Project"
__email__ = "contact@diffusion-model.com"

from .src.models import DDPM, UNet
from .src.training import DiffusionTrainer, TrainingConfig
from .src.evaluation import DiffusionEvaluator

__all__ = [
    'DDPM',
    'UNet', 
    'DiffusionTrainer',
    'TrainingConfig',
    'DiffusionEvaluator'
]
