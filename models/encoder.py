"""
Feature encoder architectures for drawing embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .base import EncoderInterface


class Conv4Encoder(EncoderInterface, nn.Module):
    """
    4-layer convolutional encoder (default for ProtoNet).
    
    Lightweight and effective for drawing recognition.
    Architecture: [Conv-BN-ReLU-MaxPool] x 4
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        channels: List[int] = [64, 128, 256, 512],
        embedding_dim: int = 512,
        use_batchnorm: bool = True,
        dropout: float = 0.1
    ):
        """
        Args:
            input_channels: Number of input channels (1 for grayscale)
            channels: Channel sizes for each conv block
            embedding_dim: Output embedding dimension
            use_batchnorm: Whether to use batch normalization
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.use_batchnorm = use_batchnorm
        
        # Build convolutional blocks
        layers = []
        in_channels = input_channels
        
        for out_channels in channels:
            layers.append(self._make_conv_block(in_channels, out_channels, dropout))
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*layers)
        
        # Global average pooling (adaptive to any input size)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Final projection to embedding dimension (if different from last conv)
        if channels[-1] != embedding_dim:
            self.projection = nn.Linear(channels[-1], embedding_dim)
        else:
            self.projection = nn.Identity()
    
    def _make_conv_block(self, in_channels: int, out_channels: int, dropout: float) -> nn.Module:
        """Create a convolutional block."""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        ]
        
        if self.use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.extend([
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout)
        ])
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Embeddings (B, embedding_dim)
        """
        # Convolutional feature extraction
        features = self.encoder(x)  # (B, channels[-1], H', W')
        
        # Global average pooling
        features = self.gap(features)  # (B, channels[-1], 1, 1)
        features = features.view(features.size(0), -1)  # (B, channels[-1])
        
        # Project to embedding dimension
        embeddings = self.projection(features)  # (B, embedding_dim)
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim
    
    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


def get_encoder(
    encoder_type: str,
    config: dict
) -> EncoderInterface:
    """
    Factory function to create encoders.
    
    Args:
        encoder_type: Type of encoder ("conv4", "resnet18", "efficientnet")
        config: Configuration dictionary
        
    Returns:
        Encoder instance
    """
    input_channels = config.get('num_channels', 1)
    embedding_dim = config.get('embedding_dim', 512)
    
    if encoder_type == "conv4":
        conv4_config = config.get('conv4', {})
        return Conv4Encoder(
            input_channels=input_channels,
            embedding_dim=embedding_dim,
            channels=conv4_config.get('channels', [64, 128, 256, 512]),
            use_batchnorm=conv4_config.get('use_batchnorm', True),
            dropout=conv4_config.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
