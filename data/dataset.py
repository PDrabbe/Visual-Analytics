"""
Dataset loaders for drawing recognition.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import logging
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)


class DrawingDataset(Dataset):
    """
    Generic drawing dataset.
    
    Supports both image-based and stroke-based drawings.
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        image_size: int = 64,
        transform: Optional[transforms.Compose] = None,
        max_classes: Optional[int] = None,
        samples_per_class: Optional[int] = None
    ):
        """
        Args:
            data_path: Path to dataset
            split: 'train', 'val', or 'test'
            image_size: Target image size
            transform: Optional transforms
            max_classes: Limit number of classes
            samples_per_class: Limit samples per class
        """
        self.data_path = Path(data_path)
        self.split = split
        self.image_size = image_size
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
            ])
        else:
            self.transform = transform
        
        # Load dataset
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        self._load_data(max_classes, samples_per_class)
        
        logger.info(f"Loaded {len(self.samples)} samples from {len(self.class_to_idx)} classes ({split} split)")
    
    def _load_data(self, max_classes: Optional[int], samples_per_class: Optional[int]):
        """Load data from disk. Override for specific dataset formats."""
        raise NotImplementedError("Subclass must implement _load_data")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.
        
        Returns:
            image: (C, H, W) tensor
            label: class index
        """
        image_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(image_path).convert('L')  # Grayscale
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_samples(self, class_idx: int) -> List[int]:
        """Get all sample indices for a class."""
        return [i for i, (_, label) in enumerate(self.samples) if label == class_idx]


class QuickDrawDataset(DrawingDataset):
    """
    Google QuickDraw dataset loader.
    
    Dataset structure:
        data_path/
            train/
                class1/
                    img1.png
                    img2.png
                class2/
                    ...
            val/
                ...
            test/
                ...
    """
    
    def _load_data(self, max_classes: Optional[int], samples_per_class: Optional[int]):
        """Load QuickDraw data."""
        split_dir = self.data_path / self.split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        # Get class directories
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        if max_classes:
            class_dirs = class_dirs[:max_classes]
        
        # Build class mapping
        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
        
        # Load samples
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            
            # Get image files
            image_files = sorted(class_dir.glob('*.png'))
            
            if samples_per_class:
                image_files = image_files[:samples_per_class]
            
            for image_file in image_files:
                self.samples.append((image_file, class_idx))


class CustomDrawingDataset(DrawingDataset):
    """
    Custom drawing dataset from a directory structure.
    
    Similar to QuickDraw but more flexible.
    """
    
    def _load_data(self, max_classes: Optional[int], samples_per_class: Optional[int]):
        """Load custom drawing data."""
        split_dir = self.data_path / self.split
        
        if not split_dir.exists():
            # If no split structure, use data_path directly
            split_dir = self.data_path
        
        # Get class directories
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        if max_classes:
            class_dirs = class_dirs[:max_classes]
        
        # Build class mapping
        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
        
        # Load samples (support multiple image formats)
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            
            # Get image files
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                image_files.extend(class_dir.glob(ext))
            
            image_files = sorted(image_files)
            
            if samples_per_class:
                image_files = image_files[:samples_per_class]
            
            for image_file in image_files:
                self.samples.append((image_file, class_idx))


def get_augmentation_transform(config: dict, image_size: int) -> transforms.Compose:
    """
    Create augmentation pipeline for drawings.
    
    Args:
        config: Augmentation config
        image_size: Target image size
        
    Returns:
        Composed transforms
    """
    transform_list = [transforms.Resize((image_size, image_size))]
    
    if config.get('enabled', True):
        # Rotation
        if 'rotation_degrees' in config:
            transform_list.append(
                transforms.RandomRotation(config['rotation_degrees'])
            )
        
        # Random affine (scale)
        if 'scale_range' in config:
            scale_min, scale_max = config['scale_range']
            transform_list.append(
                transforms.RandomAffine(
                    degrees=0,
                    scale=(scale_min, scale_max)
                )
            )
        
        # Random horizontal flip (be careful with asymmetric drawings!)
        if config.get('horizontal_flip', False):
            transform_list.append(transforms.RandomHorizontalFlip())
    
    # Convert to tensor and normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return transforms.Compose(transform_list)


def create_dataloader(
    dataset_type: str,
    config: dict,
    split: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
) -> Tuple[DataLoader, DrawingDataset]:
    """
    Create a dataloader for drawings.
    
    Args:
        dataset_type: "quickdraw" or "custom"
        config: Dataset configuration
        split: "train", "val", or "test"
        batch_size: Batch size
        shuffle: Shuffle data
        num_workers: DataLoader workers
        
    Returns:
        dataloader, dataset
    """
    # Get augmentation transform
    transform = get_augmentation_transform(
        config.get('augmentation', {}),
        config['image_size']
    )
    
    # Create dataset
    if dataset_type == "quickdraw":
        dataset = QuickDrawDataset(
            data_path=config['data_path'],
            split=split,
            image_size=config['image_size'],
            transform=transform
        )
    elif dataset_type == "custom":
        dataset = CustomDrawingDataset(
            data_path=config['data_path'],
            split=split,
            image_size=config['image_size'],
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, dataset
