"""
Core Prototypical Network implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

from .base import EncoderInterface, DistanceMetric

logger = logging.getLogger(__name__)


class ProtoNet(nn.Module):
    """
    Prototypical Network for few-shot learning.
    
    Reference: Snell et al. "Prototypical Networks for Few-shot Learning" (NeurIPS 2017)
    
    Key idea:
    1. Compute prototype (mean embedding) for each class from support set
    2. Classify queries by finding nearest prototype
    """
    
    def __init__(
        self,
        encoder: EncoderInterface,
        distance_metric: DistanceMetric,
        embedding_dim: int = 512
    ):
        """
        Args:
            encoder: Feature encoder network
            distance_metric: Distance function for classification
            embedding_dim: Dimension of embeddings
        """
        super().__init__()
        
        self.encoder = encoder
        self.distance_metric = distance_metric
        self.embedding_dim = embedding_dim
        
        logger.info(f"ProtoNet initialized with {encoder.__class__.__name__} encoder")
        logger.info(f"Distance metric: {distance_metric.get_name()}")
    
    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for meta-learning.
        
        Args:
            support_images: (N*K, C, H, W) - support set images
            support_labels: (N*K,) - support set labels
            query_images: (N*Q, C, H, W) - query set images
            query_labels: (N*Q,) - query set labels (optional, for training)
            
        Returns:
            Dictionary with:
                - logits: (N*Q, N) - classification logits
                - loss: scalar - cross-entropy loss (if query_labels provided)
                - prototypes: (N, embedding_dim) - class prototypes
        """
        # Encode support and query sets
        support_embeddings = self.encoder(support_images)  # (N*K, embedding_dim)
        query_embeddings = self.encoder(query_images)  # (N*Q, embedding_dim)
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_embeddings, support_labels)
        
        # Classify queries
        logits = self.classify(query_embeddings, prototypes)
        
        # Compute loss if labels provided
        result = {
            'logits': logits,
            'prototypes': prototypes
        }
        
        if query_labels is not None:
            loss = F.cross_entropy(logits, query_labels)
            result['loss'] = loss
        
        return result
    
    def compute_prototypes(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute class prototypes as mean of support embeddings.
        
        Args:
            embeddings: (N*K, embedding_dim)
            labels: (N*K,)
            
        Returns:
            prototypes: (N, embedding_dim)
        """
        unique_labels = torch.unique(labels)
        n_classes = len(unique_labels)
        
        prototypes = torch.zeros(n_classes, self.embedding_dim, device=embeddings.device)
        
        for idx, label in enumerate(unique_labels):
            # Get all embeddings for this class
            class_mask = labels == label
            class_embeddings = embeddings[class_mask]
            
            # Compute mean (prototype)
            prototypes[idx] = class_embeddings.mean(dim=0)
        
        return prototypes
    
    def classify(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Classify queries using prototypes.
        
        Args:
            query_embeddings: (N*Q, embedding_dim)
            prototypes: (N, embedding_dim)
            
        Returns:
            logits: (N*Q, N) - negative distances (higher = more similar)
        """
        # Compute distances
        distances = self.distance_metric.compute(query_embeddings, prototypes)
        
        # Convert to logits (negative distances)
        logits = -distances
        
        return logits
    
    def predict(
        self,
        query_images: torch.Tensor,
        prototypes: torch.Tensor,
        class_names: List[str],
        top_k: int = 1
    ) -> List[Dict]:
        """
        Make predictions for query images.
        
        Args:
            query_images: (B, C, H, W)
            prototypes: (N, embedding_dim)
            class_names: List of class names (length N)
            top_k: Return top-k predictions
            
        Returns:
            List of prediction dicts with 'class', 'confidence', 'top_k'
        """
        self.eval()
        with torch.no_grad():
            # Encode queries
            query_embeddings = self.encoder(query_images)
            
            # Get logits
            logits = self.classify(query_embeddings, prototypes)
            
            # FIX: Ensure logits is 2D (B, N)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)  # (N,) -> (1, N)
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = probs.topk(top_k, dim=1)
            
            # Format results
            predictions = []
            for i in range(query_images.size(0)):
                pred = {
                    'class': class_names[top_indices[i, 0].item()],
                    'confidence': top_probs[i, 0].item(),
                    'top_k': [
                        {
                            'class': class_names[top_indices[i, k].item()],
                            'confidence': top_probs[i, k].item()
                        }
                        for k in range(top_k)
                    ]
                }
                predictions.append(pred)
            
            return predictions
    
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for images.
        
        Args:
            images: (B, C, H, W)
            
        Returns:
            embeddings: (B, embedding_dim)
        """
        self.eval()
        with torch.no_grad():
            return self.encoder(images)
