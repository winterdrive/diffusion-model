"""
Basic neural network layers for diffusion models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResBlock(nn.Module):
    """
    Residual block with time embedding support.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim=None, up=False, down=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels) if time_emb_dim else None
        
        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # Up/down sampling flags
        self.up = up
        self.down = down
        
    def forward(self, x, time_emb=None):
        residual = self.residual(x)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        # Add time embedding
        if self.time_mlp is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            time_emb = time_emb[:, :, None, None]
            x = x + time_emb
            
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        
        # Up/down sampling
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            residual = F.interpolate(residual, scale_factor=2, mode='nearest')
        elif self.down:
            x = F.avg_pool2d(x, 2)
            residual = F.avg_pool2d(residual, 2)
            
        return x + residual


class SelfAttention(nn.Module):
    """
    Self-attention module for capturing long-range dependencies.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert self.head_dim * num_heads == channels, f"Channels {channels} must be divisible by num_heads {num_heads}"
        
        # Projection layers (query, key, value)
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        
        # Output projection layer
        self.out_proj = nn.Conv1d(channels, channels, 1)
        
        # Scaling factor for attention computation
        self.scale = self.head_dim ** -0.5
        
        # Layer normalization
        self.norm = nn.GroupNorm(8, channels)
    
    def forward(self, x):
        """
        Forward pass for self-attention.
        
        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W)
        
        Returns:
            torch.Tensor: Output feature map of same shape
        """
        residual = x
        x = self.norm(x)
        
        batch_size, channels, height, width = x.shape
        # Reshape to (B, C, H*W)
        x = x.reshape(batch_size, channels, height * width)
        
        # Compute query, key, value
        q = self.query(x)  # (B, C, H*W)
        k = self.key(x)    # (B, C, H*W)
        v = self.value(x)  # (B, C, H*W)
        
        # Reshape for multi-head attention: (B, num_heads, head_dim, H*W)
        q = q.reshape(batch_size, self.num_heads, self.head_dim, height * width)
        k = k.reshape(batch_size, self.num_heads, self.head_dim, height * width)
        v = v.reshape(batch_size, self.num_heads, self.head_dim, height * width)
        
        # Compute attention scores
        # (B, num_heads, H*W, H*W)
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        # (B, num_heads, head_dim, H*W)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        
        # Reshape back to (B, C, H*W)
        out = out.reshape(batch_size, channels, height * width)
        
        # Apply output projection
        out = self.out_proj(out)
        
        # Reshape back to (B, C, H, W)
        out = out.reshape(batch_size, channels, height, width)
        
        return out + residual


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for diffusion timesteps.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings
