"""
Abstract Base Classes for Prototype Storage

This module defines the interfaces for different storage backends.
New storage types can be added by implementing these interfaces.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import torch


class PrototypeStore(ABC):
    """
    Abstract base class for prototype storage.
    
    This interface allows easy switching between:
    - Static (pre-trained only)
    - Hybrid (static + session)
    - Database (persistent user storage)
    - Redis (distributed cache)
    """
    
    @abstractmethod
    def save_prototype(
        self, 
        class_name: str, 
        embedding: torch.Tensor,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Save a prototype for a class.
        
        Args:
            class_name: Name of the class
            embedding: Prototype embedding vector
            metadata: Optional metadata (user_id, timestamp, etc.)
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def load_prototype(self, class_name: str) -> Optional[torch.Tensor]:
        """
        Load a prototype for a class.
        
        Args:
            class_name: Name of the class
            
        Returns:
            Prototype embedding or None if not found
        """
        pass
    
    @abstractmethod
    def load_all_prototypes(self) -> Dict[str, torch.Tensor]:
        """
        Load all available prototypes.
        
        Returns:
            Dictionary mapping class names to embeddings
        """
        pass
    
    @abstractmethod
    def update_prototype(
        self,
        class_name: str,
        new_embeddings: List[torch.Tensor],
        strategy: str = "replace"
    ) -> bool:
        """
        Update an existing prototype.
        
        Args:
            class_name: Name of the class
            new_embeddings: New example embeddings
            strategy: Update strategy ("replace", "average", "weighted")
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def delete_prototype(self, class_name: str) -> bool:
        """Delete a prototype."""
        pass
    
    @abstractmethod
    def exists(self, class_name: str) -> bool:
        """Check if prototype exists."""
        pass
    
    @abstractmethod
    def list_classes(self) -> List[str]:
        """List all available classes."""
        pass
    
    @abstractmethod
    def get_metadata(self, class_name: str) -> Optional[Dict]:
        """Get metadata for a class."""
        pass
    
    def get_num_classes(self) -> int:
        """Get total number of classes."""
        return len(self.list_classes())


class EncoderInterface(ABC):
    """
    Abstract base class for feature encoders.
    
    This allows easy swapping of encoder architectures.
    """
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Feature embeddings (B, embedding_dim)
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of output embeddings."""
        pass


class DistanceMetric(ABC):
    """
    Abstract base class for distance metrics.
    
    Allows custom distance functions.
    """
    
    @abstractmethod
    def compute(
        self,
        query: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distances between query and prototypes.
        
        Args:
            query: Query embedding (embedding_dim,) or (B, embedding_dim)
            prototypes: Prototype embeddings (N, embedding_dim)
            
        Returns:
            Distances (N,) or (B, N)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get metric name."""
        pass

