"""
Inference API for production deployment.

This is the main interface for the visual analytics app.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional
import logging
import time

from models.protonet import ProtoNet
from models.encoder import get_encoder
from models.distance_metrics import get_distance_metric, CosineDistance
from models.storage import HybridPrototypeStore

logger = logging.getLogger(__name__)


class DrawingPredictor:
    """
    Production inference API for drawing classification.
    
    Features:
    - Hybrid mode: pre-trained base classes + custom session classes
    - Confidence scores and top-k predictions
    - Batch processing
    """
    
    def __init__(
        self,
        model_path: str,
        config: Optional[dict] = None,
        device: str = 'auto',
        max_custom_classes: int = 50
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to pre-trained model checkpoint
            config: Model configuration (auto-loaded if None)
            device: Device to run on ('auto', 'cuda', 'cpu', 'mps')
            max_custom_classes: Maximum custom classes per session
        """
        # Determine device
        if device == 'auto':
            from utils.helpers import get_device
            self.device = get_device('auto')
        else:
            self.device = device
        
        logger.info(f"Initializing predictor on {self.device}")
        
        # Load model
        self.model_path = Path(model_path)
        self.config = config or self._load_config()
        
        # Create model
        self.model = self._build_model()
        
        # Load weights
        self._load_weights()
        
        # Setup hybrid storage
        self.storage = HybridPrototypeStore(
            base_model_path=str(self.model_path),
            max_custom_classes=max_custom_classes
        )
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((self.config.get('image_size', 64), 
                             self.config.get('image_size', 64))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Inference settings
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.top_k = self.config.get('top_k', 3)
        
        # --- Inference-time improvements ---
        # Use cosine distance at inference for better stability across
        # base (200-sample) and custom (5-shot) prototypes.
        self.inference_distance = CosineDistance()
        # Temperature scaling: controls softmax sharpness.
        self.temperature = self.config.get('temperature', 5.0)
        # Whether to L2-normalize embeddings before prototype computation
        self.normalize_embeddings = True
        # N-way re-ranking: match inference to training conditions.
        # First find the top-N nearest prototypes, then softmax over only those.
        # Set to match training n_way for best results.
        self.n_way_inference = self.config.get('n_way_inference', 10)
        
        logger.info(f"Predictor ready with {len(self.storage.get_base_classes())} base classes")
        logger.info(f"Inference: cosine distance, temperature={self.temperature}, "
                     f"n_way_rerank={self.n_way_inference}, normalize=True")
    
    def _load_config(self) -> dict:
        """Load config from checkpoint."""
        checkpoint = torch.load(self.model_path, map_location='cpu')
        return checkpoint.get('config', {})
    
    def _build_model(self) -> ProtoNet:
        """Build model from config."""
        # Create encoder
        encoder = get_encoder(
            encoder_type=self.config.get('encoder', 'conv4'),
            config={
                'num_channels': self.config.get('num_channels', 1),
                'embedding_dim': self.config.get('embedding_dim', 512),
                **self.config.get('conv4', {})
            }
        )
        
        # Create distance metric
        distance_metric = get_distance_metric(
            metric_type=self.config.get('distance_metric', 'euclidean'),
            embedding_dim=self.config.get('embedding_dim', 512)
        )
        
        # Create ProtoNet
        model = ProtoNet(
            encoder=encoder,
            distance_metric=distance_metric,
            embedding_dim=self.config.get('embedding_dim', 512)
        )
        
        return model
    
    def _load_weights(self):
        """Load model weights."""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Load encoder weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'encoder_state_dict' in checkpoint:
            self.model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model weights loaded")
    
    def preprocess_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (path, PIL, numpy, or tensor)
            
        Returns:
            Preprocessed tensor (1, C, H, W)
        """
        # Handle different input types
        if isinstance(image, (str, Path)):
            # Load from file
            image = Image.open(image).convert('L')
        elif isinstance(image, np.ndarray):
            # Convert numpy to PIL
            image = Image.fromarray(image).convert('L')
        elif isinstance(image, torch.Tensor):
            # Already tensor
            if image.dim() == 2:
                image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            elif image.dim() == 3:
                image = image.unsqueeze(0)  # (1, C, H, W)
            return image.to(self.device)
        
        # Apply transforms
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor.to(self.device)
    
    def add_custom_class(
        self,
        class_name: str,
        examples: List[Union[str, Path, Image.Image, np.ndarray]],
        update_strategy: str = "replace"
    ) -> bool:
        """
        Add a custom class to the session.
        
        This is the key feature for hybrid mode - users can teach
        new classes with just a few examples!
        
        Args:
            class_name: Name of the new class
            examples: List of example drawings (3+ recommended)
            update_strategy: "replace", "average", or "weighted"
            
        Returns:
            Success status
        """
        if len(examples) < 1:
            logger.warning("Need at least 1 example to create a class")
            return False
        
        logger.info(f"Adding custom class '{class_name}' with {len(examples)} examples")
        
        # Embed all examples
        embeddings = []
        with torch.no_grad():
            for example in examples:
                # Preprocess
                tensor = self.preprocess_image(example)
                
                # Embed
                embedding = self.model.encoder(tensor)
                emb = embedding.squeeze(0).cpu()
                
                # L2-normalize for cosine-compatible prototypes
                if self.normalize_embeddings:
                    emb = F.normalize(emb, p=2, dim=0)
                
                embeddings.append(emb)
        
        # Update storage
        success = self.storage.update_prototype(class_name, embeddings, update_strategy)
        
        if success:
            logger.info(f"Custom class '{class_name}' added successfully")
        
        return success
    
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
        return_top_k: Optional[int] = None
    ) -> Dict:
        """
        Predict class for a single drawing.
        
        Args:
            image: Input drawing
            return_top_k: Number of top predictions (default: from config)
            
        Returns:
            Dictionary with:
                - class: predicted class name
                - confidence: confidence score (0-1)
                - top_k: list of top-k predictions
                - inference_time: time in milliseconds
        """
        start_time = time.time()
        
        # Preprocess
        tensor = self.preprocess_image(image)
        
        # Load prototypes
        prototypes_dict = self.storage.load_all_prototypes()
        class_names = list(prototypes_dict.keys())
        prototypes = torch.stack([prototypes_dict[name] for name in class_names]).to(self.device)
        
        # Normalize prototypes for cosine distance
        if self.normalize_embeddings:
            prototypes = F.normalize(prototypes, p=2, dim=1)
        
        # Predict using cosine distance + temperature scaling
        with torch.no_grad():
            predictions = self._predict_with_scaling(
                tensor,
                prototypes,
                class_names,
                top_k=return_top_k or self.top_k
            )
        
        # Add inference time
        predictions['inference_time_ms'] = (time.time() - start_time) * 1000
        
        # Check confidence threshold
        if predictions['confidence'] < self.confidence_threshold:
            predictions['warning'] = 'Low confidence prediction'
        
        return predictions
    
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Predict classes for multiple drawings (batched for efficiency).
        
        Args:
            images: List of input drawings
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        all_predictions = []
        
        # Load prototypes once
        prototypes_dict = self.storage.load_all_prototypes()
        class_names = list(prototypes_dict.keys())
        prototypes = torch.stack([prototypes_dict[name] for name in class_names]).to(self.device)
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            
            # Preprocess batch
            batch_tensors = torch.cat([
                self.preprocess_image(img) for img in batch_images
            ])
            
            # Predict
            with torch.no_grad():
                batch_predictions = self.model.predict(
                    batch_tensors,
                    prototypes,
                    class_names,
                    top_k=self.top_k
                )
            
            all_predictions.extend(batch_predictions)
        
        return all_predictions
    
    def _predict_with_scaling(
        self,
        query_images: torch.Tensor,
        prototypes: torch.Tensor,
        class_names: List[str],
        top_k: int = 3
    ) -> Dict:
        """
        Predict with N-way re-ranking + temperature-scaled softmax.
        
        Two-stage approach:
        1. Find the N nearest prototypes (raw distance, no softmax)
        2. Softmax over only those N candidates
        
        This matches the N-way training condition and produces
        well-calibrated confidences for visual analytics display.
        """
        # Encode query
        query_embeddings = self.model.encoder(query_images)
        
        # L2-normalize query embeddings
        if self.normalize_embeddings:
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        
        # Compute distances to ALL prototypes
        distances = self.inference_distance.compute(query_embeddings, prototypes)
        
        # Ensure 2D
        if distances.dim() == 1:
            distances = distances.unsqueeze(0)
        
        # Stage 1: Find top-N nearest prototypes (N = training n_way)
        n_candidates = min(self.n_way_inference, len(class_names))
        _, candidate_indices = distances.topk(n_candidates, dim=1, largest=False)  # smallest distances
        
        # Stage 2: Softmax over only the N candidates
        # Gather distances for candidates only
        candidate_distances = distances.gather(1, candidate_indices)  # (B, N)
        
        # Convert to similarities and scale
        candidate_similarities = (1.0 - candidate_distances) * self.temperature
        
        # Softmax over N candidates (not all 37!)
        candidate_probs = F.softmax(candidate_similarities, dim=1)
        
        # Get top-k from the N candidates
        actual_top_k = min(top_k, n_candidates)
        top_probs, top_within_candidates = candidate_probs.topk(actual_top_k, dim=1)
        
        # Map back to original class indices
        top_indices = candidate_indices.gather(1, top_within_candidates)
        
        # Format result (single image)
        i = 0
        prediction = {
            'class': class_names[top_indices[i, 0].item()],
            'confidence': top_probs[i, 0].item(),
            'top_k': [
                {
                    'class': class_names[top_indices[i, k].item()],
                    'confidence': top_probs[i, k].item()
                }
                for k in range(actual_top_k)
            ]
        }
        
        return prediction

    def get_available_classes(self) -> Dict[str, List[str]]:
        """
        Get all available classes.
        
        Returns:
            Dictionary with 'base' and 'custom' class lists
        """
        return {
            'base': self.storage.get_base_classes(),
            'custom': self.storage.get_custom_classes(),
            'total': len(self.storage.list_classes())
        }
    
    def remove_custom_class(self, class_name: str) -> bool:
        """Remove a custom class."""
        return self.storage.delete_prototype(class_name)
    
    def clear_custom_classes(self):
        """Clear all custom classes (keep base classes)."""
        self.storage.clear_custom_classes()
        logger.info("All custom classes cleared")
    
    def export_custom_classes(self, path: str):
        """
        Export custom classes for later reload.
        
        Args:
            path: Export file path
        """
        self.storage.export_custom_classes(path)
        logger.info(f"Custom classes exported to {path}")
    
    def get_embedding(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """
        Get embedding vector for an image.
        
        Useful for visualization (t-SNE, UMAP) in the analytics app.
        
        Args:
            image: Input drawing
            
        Returns:
            Embedding vector (embedding_dim,)
        """
        tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            embedding = self.model.encoder(tensor)
            if self.normalize_embeddings:
                embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding.squeeze(0).cpu().numpy()
    
    def get_prototypes(self) -> Dict[str, np.ndarray]:
        """
        Get all prototypes as numpy arrays.
        
        Returns:
            Dictionary mapping class names to prototype vectors
        """
        prototypes_dict = self.storage.load_all_prototypes()
        return {
            name: proto.cpu().numpy()
            for name, proto in prototypes_dict.items()
        }
