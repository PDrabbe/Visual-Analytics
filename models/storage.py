"""
Concrete implementations of prototype storage backends.
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

from .base import PrototypeStore

logger = logging.getLogger(__name__)


class StaticPrototypeStore(PrototypeStore):
    """
    Static storage: Load pre-trained prototypes from disk (read-only).
    
    Use case: Production deployment with fixed classes.
    """
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: Path to saved model checkpoint
        """
        self.model_path = Path(model_path)
        self.prototypes: Dict[str, torch.Tensor] = {}
        self.metadata: Dict[str, Dict] = {}
        
        if self.model_path.exists():
            self._load_from_disk()
        else:
            logger.warning(f"Model path {model_path} not found. Starting with empty prototypes.")
    
    def _load_from_disk(self):
        """Load prototypes from checkpoint."""
        checkpoint = torch.load(self.model_path, map_location='cpu')
        self.prototypes = checkpoint.get('prototypes', {})
        self.metadata = checkpoint.get('metadata', {})
        logger.info(f"Loaded {len(self.prototypes)} base prototypes from {self.model_path}")
    
    def save_prototype(self, class_name: str, embedding: torch.Tensor, 
                      metadata: Optional[Dict] = None) -> bool:
        """Static mode: saving not allowed."""
        logger.warning("Cannot save prototypes in static mode")
        return False
    
    def load_prototype(self, class_name: str) -> Optional[torch.Tensor]:
        """Load a single prototype."""
        return self.prototypes.get(class_name)
    
    def load_all_prototypes(self) -> Dict[str, torch.Tensor]:
        """Load all prototypes."""
        return self.prototypes.copy()
    
    def update_prototype(self, class_name: str, new_embeddings: List[torch.Tensor],
                        strategy: str = "replace") -> bool:
        """Static mode: updating not allowed."""
        logger.warning("Cannot update prototypes in static mode")
        return False
    
    def delete_prototype(self, class_name: str) -> bool:
        """Static mode: deletion not allowed."""
        logger.warning("Cannot delete prototypes in static mode")
        return False
    
    def exists(self, class_name: str) -> bool:
        """Check if class exists."""
        return class_name in self.prototypes
    
    def list_classes(self) -> List[str]:
        """List all classes."""
        return list(self.prototypes.keys())
    
    def get_metadata(self, class_name: str) -> Optional[Dict]:
        """Get metadata."""
        return self.metadata.get(class_name)


class SessionPrototypeStore(PrototypeStore):
    """
    Session-based storage: In-memory prototypes that don't persist.
    
    Use case: User teaches custom categories during a session.
    """
    
    def __init__(self, base_store: Optional[PrototypeStore] = None):
        """
        Args:
            base_store: Optional base store to fall back to
        """
        self.base_store = base_store
        self.session_prototypes: Dict[str, torch.Tensor] = {}
        self.session_metadata: Dict[str, Dict] = {}
        logger.info("Session store initialized")
    
    def save_prototype(self, class_name: str, embedding: torch.Tensor,
                      metadata: Optional[Dict] = None) -> bool:
        """Save prototype in session memory."""
        self.session_prototypes[class_name] = embedding.detach().cpu()
        self.session_metadata[class_name] = metadata or {
            'created_at': datetime.now().isoformat(),
            'source': 'session'
        }
        logger.info(f"Saved session prototype for '{class_name}'")
        return True
    
    def load_prototype(self, class_name: str) -> Optional[torch.Tensor]:
        """Load from session first, fall back to base."""
        if class_name in self.session_prototypes:
            return self.session_prototypes[class_name]
        if self.base_store:
            return self.base_store.load_prototype(class_name)
        return None
    
    def load_all_prototypes(self) -> Dict[str, torch.Tensor]:
        """Merge base and session prototypes."""
        all_protos = {}
        if self.base_store:
            all_protos.update(self.base_store.load_all_prototypes())
        all_protos.update(self.session_prototypes)  # Session overrides base
        return all_protos
    
    def update_prototype(self, class_name: str, new_embeddings: List[torch.Tensor],
                        strategy: str = "replace") -> bool:
        """Update prototype using specified strategy."""
        if strategy == "replace":
            # Replace with mean of new embeddings
            new_proto = torch.stack(new_embeddings).mean(dim=0)
            return self.save_prototype(class_name, new_proto)
        
        elif strategy == "average":
            # Average with existing prototype
            existing = self.load_prototype(class_name)
            if existing is None:
                return self.save_prototype(class_name, torch.stack(new_embeddings).mean(0))
            
            new_mean = torch.stack(new_embeddings).mean(0)
            updated = (existing + new_mean) / 2
            return self.save_prototype(class_name, updated)
        
        elif strategy == "weighted":
            # Weighted average (favor new examples)
            existing = self.load_prototype(class_name)
            if existing is None:
                return self.save_prototype(class_name, torch.stack(new_embeddings).mean(0))
            
            new_mean = torch.stack(new_embeddings).mean(0)
            alpha = 0.7  # Weight for new examples
            updated = alpha * new_mean + (1 - alpha) * existing
            return self.save_prototype(class_name, updated)
        
        else:
            logger.error(f"Unknown update strategy: {strategy}")
            return False
    
    def delete_prototype(self, class_name: str) -> bool:
        """Delete from session (not base)."""
        if class_name in self.session_prototypes:
            del self.session_prototypes[class_name]
            del self.session_metadata[class_name]
            logger.info(f"Deleted session prototype '{class_name}'")
            return True
        return False
    
    def exists(self, class_name: str) -> bool:
        """Check session first, then base."""
        if class_name in self.session_prototypes:
            return True
        if self.base_store:
            return self.base_store.exists(class_name)
        return False
    
    def list_classes(self) -> List[str]:
        """List all classes (session + base)."""
        classes = set(self.session_prototypes.keys())
        if self.base_store:
            classes.update(self.base_store.list_classes())
        return list(classes)
    
    def get_metadata(self, class_name: str) -> Optional[Dict]:
        """Get metadata, preferring session."""
        if class_name in self.session_metadata:
            return self.session_metadata[class_name]
        if self.base_store:
            return self.base_store.get_metadata(class_name)
        return None
    
    def clear_session(self):
        """Clear all session prototypes."""
        self.session_prototypes.clear()
        self.session_metadata.clear()
        logger.info("Session cleared")
    
    def export_session(self, path: str):
        """Export session prototypes to file."""
        export_data = {
            'prototypes': self.session_prototypes,
            'metadata': self.session_metadata,
            'exported_at': datetime.now().isoformat()
        }
        torch.save(export_data, path)
        logger.info(f"Session exported to {path}")


class HybridPrototypeStore(PrototypeStore):
    """
    Hybrid storage: Static base + Session overlay.
    
    This is the recommended starting point - provides both
    stability (base classes) and flexibility (custom classes).
    """
    
    def __init__(self, base_model_path: str, max_custom_classes: int = 50):
        """
        Args:
            base_model_path: Path to pre-trained base model
            max_custom_classes: Maximum custom classes per session
        """
        self.static_store = StaticPrototypeStore(base_model_path)
        self.session_store = SessionPrototypeStore(base_store=self.static_store)
        self.max_custom_classes = max_custom_classes
        logger.info(f"Hybrid store initialized with {self.static_store.get_num_classes()} base classes")
    
    def save_prototype(self, class_name: str, embedding: torch.Tensor,
                      metadata: Optional[Dict] = None) -> bool:
        """Save to session layer."""
        if len(self.session_store.session_prototypes) >= self.max_custom_classes:
            logger.warning(f"Max custom classes ({self.max_custom_classes}) reached")
            return False
        return self.session_store.save_prototype(class_name, embedding, metadata)
    
    def load_prototype(self, class_name: str) -> Optional[torch.Tensor]:
        """Load from session first, fall back to base."""
        return self.session_store.load_prototype(class_name)
    
    def load_all_prototypes(self) -> Dict[str, torch.Tensor]:
        """Merge base and session."""
        return self.session_store.load_all_prototypes()
    
    def update_prototype(self, class_name: str, new_embeddings: List[torch.Tensor],
                        strategy: str = "replace") -> bool:
        """Update in session layer."""
        return self.session_store.update_prototype(class_name, new_embeddings, strategy)
    
    def delete_prototype(self, class_name: str) -> bool:
        """Delete from session only."""
        return self.session_store.delete_prototype(class_name)
    
    def exists(self, class_name: str) -> bool:
        """Check both layers."""
        return self.session_store.exists(class_name)
    
    def list_classes(self) -> List[str]:
        """List all classes."""
        return self.session_store.list_classes()
    
    def get_metadata(self, class_name: str) -> Optional[Dict]:
        """Get metadata."""
        return self.session_store.get_metadata(class_name)
    
    def get_base_classes(self) -> List[str]:
        """Get only base (pre-trained) classes."""
        return self.static_store.list_classes()
    
    def get_custom_classes(self) -> List[str]:
        """Get only custom (session) classes."""
        return list(self.session_store.session_prototypes.keys())
    
    def clear_custom_classes(self):
        """Clear only custom classes, keep base."""
        self.session_store.clear_session()
    
    def export_custom_classes(self, path: str):
        """Export custom classes for later reload."""
        self.session_store.export_session(path)

