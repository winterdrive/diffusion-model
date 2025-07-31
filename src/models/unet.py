"""
UNet architecture for diffusion models with conditional generation support.
"""

import torch
import torch.nn as nn
from .layers import ResBlock, SelfAttention, TimeEmbedding


class UNet(nn.Module):
    """
    UNet model for denoising diffusion with conditional generation support.
    
    This model predicts noise added to images and supports conditional generation
    through class embeddings.
    """
    
    def __init__(
        self,
        img_size=64,
        in_channels=3,
        out_channels=3,
        time_emb_dim=128,
        num_classes=24,
        class_emb_dim=64,
        base_channels=64,
        channel_multipliers=[1, 2, 4, 8],
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        dropout=0.1,
        use_attention=True,
    ):
        """
        Initialize UNet model.
        
        Args:
            img_size (int): Input image size
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            time_emb_dim (int): Time embedding dimension
            num_classes (int): Number of conditional classes
            class_emb_dim (int): Class embedding dimension
            base_channels (int): Base number of channels
            channel_multipliers (list): Channel multipliers for each level
            num_res_blocks (int): Number of residual blocks per level
            attention_resolutions (list): Resolutions to apply attention at
            dropout (float): Dropout rate
            use_attention (bool): Whether to use attention layers
        """
        super().__init__()
        
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.num_classes = num_classes
        self.class_emb_dim = class_emb_dim
        self.use_attention = use_attention
        
        # Time embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        
        # Class embedding for conditional generation
        self.class_embed = nn.Embedding(num_classes, class_emb_dim)
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Calculate channel dimensions
        self.channels = [base_channels * mult for mult in channel_multipliers]
        
        # Encoder (downsampling)
        self.down_blocks = nn.ModuleList()
        prev_channels = base_channels
        
        for i, channels in enumerate(self.channels):
            is_last = i == len(self.channels) - 1
            
            # ResNet blocks
            layers = []
            for j in range(num_res_blocks):
                layers.append(ResBlock(
                    prev_channels if j == 0 else channels,
                    channels,
                    time_emb_dim + class_emb_dim,
                    down=False
                ))
                prev_channels = channels
                
                # Add attention if specified
                current_res = img_size // (2 ** i)
                if use_attention and current_res in attention_resolutions:
                    layers.append(SelfAttention(channels))
            
            # Downsampling (except for last block)
            if not is_last:
                layers.append(ResBlock(
                    channels,
                    channels,
                    time_emb_dim + class_emb_dim,
                    down=True
                ))
            
            self.down_blocks.append(nn.ModuleList(layers))
        
        # Middle blocks
        mid_channels = self.channels[-1]
        self.mid_blocks = nn.ModuleList([
            ResBlock(mid_channels, mid_channels, time_emb_dim + class_emb_dim),
            SelfAttention(mid_channels) if use_attention else nn.Identity(),
            ResBlock(mid_channels, mid_channels, time_emb_dim + class_emb_dim),
        ])
        
        # Decoder (upsampling)
        self.up_blocks = nn.ModuleList()
        
        for i, channels in enumerate(reversed(self.channels)):
            is_last = i == len(self.channels) - 1
            
            # ResNet blocks
            layers = []
            for j in range(num_res_blocks + 1):  # +1 for skip connection
                input_channels = mid_channels if i == 0 and j == 0 else channels
                if j == 0 and i > 0:
                    # Skip connection from encoder
                    input_channels = channels + self.channels[-(i+1)]
                
                layers.append(ResBlock(
                    input_channels,
                    channels,
                    time_emb_dim + class_emb_dim,
                    up=False
                ))
                
                # Add attention if specified
                current_res = img_size // (2 ** (len(self.channels) - 1 - i))
                if use_attention and current_res in attention_resolutions:
                    layers.append(SelfAttention(channels))
            
            # Upsampling (except for last block)
            if not is_last:
                layers.append(ResBlock(
                    channels,
                    channels,
                    time_emb_dim + class_emb_dim,
                    up=True
                ))
            
            self.up_blocks.append(nn.ModuleList(layers))
            mid_channels = channels
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x, timesteps, class_labels=None):
        """
        Forward pass of UNet model.
        
        Args:
            x (torch.Tensor): Noisy input images [B, C, H, W]
            timesteps (torch.Tensor): Diffusion timesteps [B]
            class_labels (torch.Tensor, optional): Class labels [B, num_objects]
        
        Returns:
            torch.Tensor: Predicted noise [B, C, H, W]
        """
        # Time embedding
        time_emb = self.time_embed(timesteps)  # [B, time_emb_dim]
        
        # Class embedding
        if class_labels is not None:
            # Handle multi-hot encoding for multiple objects
            if class_labels.dim() == 2:  # Multi-hot encoding [B, num_classes]
                # Sum embeddings for active classes
                class_emb = torch.matmul(class_labels.float(), self.class_embed.weight)  # [B, class_emb_dim]
            else:  # Single class labels [B]
                class_emb = self.class_embed(class_labels)  # [B, class_emb_dim]
            
            # Combine time and class embeddings
            emb = torch.cat([time_emb, class_emb], dim=-1)  # [B, time_emb_dim + class_emb_dim]
        else:
            # Use zero class embedding if no labels provided
            class_emb = torch.zeros(time_emb.shape[0], self.class_emb_dim, device=time_emb.device)
            emb = torch.cat([time_emb, class_emb], dim=-1)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder with skip connections
        skip_connections = []
        
        for i, block_layers in enumerate(self.down_blocks):
            for j, layer in enumerate(block_layers):
                if isinstance(layer, ResBlock):
                    x = layer(x, emb)
                else:  # Attention layer
                    x = layer(x)
            
            # Save skip connection before downsampling
            if i < len(self.down_blocks) - 1:  # Not the last block
                skip_connections.append(x)
        
        # Middle blocks
        for layer in self.mid_blocks:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            else:  # Attention layer or Identity
                x = layer(x)
        
        # Decoder with skip connections
        skip_connections = skip_connections[::-1]  # Reverse order
        
        for i, block_layers in enumerate(self.up_blocks):
            # Add skip connection (except for first block)
            if i > 0:
                skip = skip_connections[i - 1]
                x = torch.cat([x, skip], dim=1)
            
            for layer in block_layers:
                if isinstance(layer, ResBlock):
                    x = layer(x, emb)
                else:  # Attention layer
                    x = layer(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x
