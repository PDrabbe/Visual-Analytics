"""
Distance metrics for prototype classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DistanceMetric


class EuclideanDistance(DistanceMetric):
    """
    Euclidean (L2) distance metric.
    
    Most common choice for ProtoNet.
    """
    
    def compute(self, query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Compute Euclidean distances.
        
        Args:
            query: (embedding_dim,) or (B, embedding_dim)
            prototypes: (N, embedding_dim)
            
        Returns:
            Distances (N,) or (B, N)
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)  # (1, embedding_dim)
        
        # Efficient computation: ||q - p||^2 = ||q||^2 + ||p||^2 - 2<q,p>
        query_norm = (query ** 2).sum(dim=1, keepdim=True)  # (B, 1)
        proto_norm = (prototypes ** 2).sum(dim=1, keepdim=True).t()  # (1, N)
        distances = query_norm + proto_norm - 2 * torch.mm(query, prototypes.t())
        
        # Numerical stability
        distances = torch.clamp(distances, min=0.0)
        
        return torch.sqrt(distances).squeeze(0) if query.size(0) == 1 else torch.sqrt(distances)
    
    def get_name(self) -> str:
        return "euclidean"


class CosineDistance(DistanceMetric):
    """
    Cosine distance metric (1 - cosine_similarity).
    
    Good for normalized embeddings.
    """
    
    def compute(self, query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine distances.
        
        Args:
            query: (embedding_dim,) or (B, embedding_dim)
            prototypes: (N, embedding_dim)
            
        Returns:
            Distances (N,) or (B, N)
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        # Normalize vectors
        query_norm = F.normalize(query, p=2, dim=1)
        proto_norm = F.normalize(prototypes, p=2, dim=1)
        
        # Cosine similarity
        similarity = torch.mm(query_norm, proto_norm.t())
        
        # Convert to distance
        distance = 1 - similarity
        
        return distance.squeeze(0) if query.size(0) == 1 else distance
    
    def get_name(self) -> str:
        return "cosine"


def get_distance_metric(
    metric_type: str,
    embedding_dim: int = None
) -> DistanceMetric:
    """
    Factory function to create distance metrics.
    
    Args:
        metric_type: "euclidean" or "cosine"
        embedding_dim: Embedding dimension (unused, kept for API compatibility)
        
    Returns:
        Distance metric instance
    """
    if metric_type == "euclidean":
        return EuclideanDistance()
    elif metric_type == "cosine":
        return CosineDistance()
    else:
        raise ValueError(f"Unknown distance metric: {metric_type}")
