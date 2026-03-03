"""
Episodic sampler for meta-learning.

Samples N-way K-shot episodes from the dataset.
"""

import torch
from torch.utils.data import Sampler
import numpy as np
from typing import Iterator, List, Tuple
import logging

logger = logging.getLogger(__name__)


class EpisodicSampler:
    """
    Sample N-way K-shot episodes for meta-learning.
    
    Each episode contains:
    - Support set: N classes x K examples per class
    - Query set: N classes x Q examples per class
    """
    
    def __init__(
        self,
        dataset,
        n_way: int,
        n_support: int,
        n_query: int,
        n_episodes: int
    ):
        """
        Args:
            dataset: DrawingDataset instance
            n_way: Number of classes per episode
            n_support: Number of support examples per class
            n_query: Number of query examples per class
            n_episodes: Number of episodes to sample
        """
        self.dataset = dataset
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_episodes = n_episodes
        
        # Build class-to-samples mapping
        self.class_samples = {}
        for class_idx in range(len(dataset.class_to_idx)):
            self.class_samples[class_idx] = dataset.get_class_samples(class_idx)
        
        # Validate
        self._validate()
        
        logger.info(f"Episodic sampler: {n_way}-way {n_support}-shot, {n_episodes} episodes")
    
    def _validate(self):
        """Validate that we have enough samples per class."""
        min_samples_needed = self.n_support + self.n_query
        
        for class_idx, samples in self.class_samples.items():
            if len(samples) < min_samples_needed:
                class_name = self.dataset.idx_to_class[class_idx]
                raise ValueError(
                    f"Class '{class_name}' has only {len(samples)} samples, "
                    f"but need {min_samples_needed} ({self.n_support} support + {self.n_query} query)"
                )
    
    def sample_episode(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample one episode.
        
        Returns:
            support_images: (N*K, C, H, W)
            support_labels: (N*K,) 
            query_images: (N*Q, C, H, W)
            query_labels: (N*Q,)
        """
        # Sample N classes
        all_classes = list(self.class_samples.keys())
        episode_classes = np.random.choice(all_classes, self.n_way, replace=False)
        
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        
        for new_label, class_idx in enumerate(episode_classes):
            # Get all samples for this class
            class_sample_indices = self.class_samples[class_idx]
            
            # Sample K+Q examples
            selected_indices = np.random.choice(
                class_sample_indices,
                self.n_support + self.n_query,
                replace=False
            )
            
            # Split into support and query
            support_indices = selected_indices[:self.n_support]
            query_indices = selected_indices[self.n_support:]
            
            # Load support examples
            for idx in support_indices:
                image, _ = self.dataset[idx]
                support_images.append(image)
                support_labels.append(new_label)  # Use episode-local label
            
            # Load query examples
            for idx in query_indices:
                image, _ = self.dataset[idx]
                query_images.append(image)
                query_labels.append(new_label)
        
        # Stack into tensors
        support_images = torch.stack(support_images)
        support_labels = torch.tensor(support_labels, dtype=torch.long)
        query_images = torch.stack(query_images)
        query_labels = torch.tensor(query_labels, dtype=torch.long)
        
        return support_images, support_labels, query_images, query_labels
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Iterate over episodes."""
        for _ in range(self.n_episodes):
            yield self.sample_episode()
    
    def __len__(self) -> int:
        """Number of episodes."""
        return self.n_episodes


def create_episodic_dataloader(
    dataset,
    n_way: int,
    n_support: int,
    n_query: int,
    n_episodes: int
):
    """
    Create an EpisodicSampler for episodic training.
    
    Args:
        dataset: DrawingDataset
        n_way: Classes per episode
        n_support: Support examples per class
        n_query: Query examples per class
        n_episodes: Number of episodes
        
    Returns:
        EpisodicSampler instance
    """
    sampler = EpisodicSampler(
        dataset=dataset,
        n_way=n_way,
        n_support=n_support,
        n_query=n_query,
        n_episodes=n_episodes
    )
    
    return sampler
