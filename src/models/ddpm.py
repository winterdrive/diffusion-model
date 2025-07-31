"""
Denoising Diffusion Probabilistic Model (DDPM) implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def extract(tensor, t, x_shape):
    """
    Extract values from tensor at given timesteps t for broadcast.
    
    Args:
        tensor (torch.Tensor): Tensor to extract from
        t (torch.Tensor): Timesteps
        x_shape (tuple): Shape for broadcasting
    
    Returns:
        torch.Tensor: Extracted and reshaped values
    """
    batch_size = t.shape[0]
    out = tensor.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model implementation.
    
    This model implements the forward and reverse diffusion processes
    for generating images from noise.
    """
    
    def __init__(
        self,
        model,
        timesteps=1000,
        beta_schedule='linear',
        loss_type='l2',
        use_classifier_guidance=False,
        classifier_guidance_scale=1.0,
        classifier_guidance_start_step=0,
    ):
        """
        Initialize DDPM model.
        
        Args:
            model: Neural network model for noise prediction
            timesteps (int): Number of diffusion timesteps
            beta_schedule (str): Beta schedule type ('linear' or 'cosine')
            loss_type (str): Loss function type ('l1', 'l2', or 'huber')
            use_classifier_guidance (bool): Whether to use classifier guidance
            classifier_guidance_scale (float): Classifier guidance scale
            classifier_guidance_start_step (int): Step to start classifier guidance
        """
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.use_classifier_guidance = use_classifier_guidance
        self.classifier_guidance_scale = classifier_guidance_scale
        self.classifier_guidance_start_step = classifier_guidance_start_step
        self.evaluator = None  # Will be set when needed
        
        # Create beta schedule
        if beta_schedule == 'linear':
            betas = self._linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-compute diffusion constants
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Compute constants for diffusion q(x_t | x_{t-1})
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        # Compute posterior variance q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
    
    def _linear_beta_schedule(self, timesteps):
        """Linear beta schedule."""
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine beta schedule."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: sample x_t given x_0 and t.
        
        Args:
            x_start (torch.Tensor): Clean images x_0
            t (torch.Tensor): Timesteps
            noise (torch.Tensor, optional): Custom noise, generated if None
        
        Returns:
            torch.Tensor: Noisy images x_t at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute posterior mean and variance q(x_{t-1} | x_t, x_0).
        
        Args:
            x_start (torch.Tensor): Clean images x_0
            x_t (torch.Tensor): Noisy images x_t
            t (torch.Tensor): Timesteps
        
        Returns:
            tuple: (posterior_mean, posterior_variance, posterior_log_variance)
        """
        posterior_mean_coef1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2 = extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        posterior_mean = posterior_mean_coef1 * x_start + posterior_mean_coef2 * x_t
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from x_t and predicted noise.
        
        Args:
            x_t (torch.Tensor): Noisy images at timestep t
            t (torch.Tensor): Timesteps
            noise (torch.Tensor): Predicted noise
        
        Returns:
            torch.Tensor: Predicted clean images x_0
        """
        sqrt_recip_alphas_cumprod_t = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
    
    def p_mean_variance(self, x_t, t, class_labels=None):
        """
        Compute mean and variance for reverse process p(x_{t-1} | x_t).
        
        Args:
            x_t (torch.Tensor): Noisy images at timestep t
            t (torch.Tensor): Timesteps
            class_labels (torch.Tensor, optional): Class labels for conditioning
        
        Returns:
            tuple: (model_mean, posterior_variance, posterior_log_variance, pred_x_start)
        """
        # Predict noise
        predicted_noise = self.model(x_t, t, class_labels)
        
        # Predict x_0
        pred_x_start = self.predict_start_from_noise(x_t, t, predicted_noise)
        pred_x_start = torch.clamp(pred_x_start, -1., 1.)
        
        # Compute posterior mean and variance
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(
            pred_x_start, x_t, t
        )
        
        return model_mean, posterior_variance, posterior_log_variance, pred_x_start
    
    def p_sample(self, x_t, t, class_labels=None):
        """
        Sample x_{t-1} from x_t using reverse process.
        
        Args:
            x_t (torch.Tensor): Noisy images at timestep t
            t (torch.Tensor): Timesteps
            class_labels (torch.Tensor, optional): Class labels for conditioning
        
        Returns:
            tuple: (x_{t-1}, pred_x_start)
        """
        model_mean, _, model_log_variance, pred_x_start = self.p_mean_variance(x_t, t, class_labels)
        
        noise = torch.randn_like(x_t)
        # No noise when t == 0
        nonzero_mask = (t != 0).float().reshape(x_t.shape[0], *((1,) * (len(x_t.shape) - 1)))
        
        x_prev = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        
        return x_prev, pred_x_start
    
    def p_sample_loop(self, shape, class_labels=None, return_intermediates=False):
        """
        Generate samples using reverse diffusion process.
        
        Args:
            shape (tuple): Shape of samples to generate
            class_labels (torch.Tensor, optional): Class labels for conditioning
            return_intermediates (bool): Whether to return intermediate steps
        
        Returns:
            torch.Tensor or tuple: Generated samples (and intermediates if requested)
        """
        device = next(self.parameters()).device
        batch_size = shape[0]
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        imgs = [img] if return_intermediates else None
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling', total=self.timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img, _ = self.p_sample(img, t, class_labels)
            
            if return_intermediates:
                imgs.append(img)
        
        if return_intermediates:
            return img, imgs
        return img
    
    def sample(self, batch_size, class_labels=None, return_intermediates=False):
        """
        Generate samples with specified batch size and class labels.
        
        Args:
            batch_size (int): Number of samples to generate
            class_labels (torch.Tensor, optional): Class labels for conditioning
            return_intermediates (bool): Whether to return intermediate steps
        
        Returns:
            torch.Tensor or tuple: Generated samples (and intermediates if requested)
        """
        image_size = getattr(self.model, 'img_size', 64)
        in_channels = getattr(self.model, 'in_channels', 3)
        shape = (batch_size, in_channels, image_size, image_size)
        
        return self.p_sample_loop(shape, class_labels, return_intermediates)
    
    def training_loss(self, x_start, t, class_labels=None, noise=None):
        """
        Compute training loss for DDPM.
        
        Args:
            x_start (torch.Tensor): Clean images
            t (torch.Tensor): Timesteps
            class_labels (torch.Tensor, optional): Class labels for conditioning
            noise (torch.Tensor, optional): Custom noise, generated if None
        
        Returns:
            torch.Tensor: Training loss
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise
        predicted_noise = self.model(x_noisy, t, class_labels)
        
        # Compute loss
        if self.loss_type == 'l1':
            loss = F.l1_loss(predicted_noise, noise)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(predicted_noise, noise)
        elif self.loss_type == 'huber':
            loss = F.huber_loss(predicted_noise, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    def forward(self, x_start, class_labels=None):
        """
        Forward pass for training.
        
        Args:
            x_start (torch.Tensor): Clean images
            class_labels (torch.Tensor, optional): Class labels for conditioning
        
        Returns:
            torch.Tensor: Training loss
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        
        return self.training_loss(x_start, t, class_labels)
