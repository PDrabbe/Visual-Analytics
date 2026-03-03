"""
Utility functions for configuration and logging.
"""

import yaml
import logging
import sys
import random
from pathlib import Path
from typing import Dict, Any
import torch
import numpy as np


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logging(
    log_file: str = None,
    level: str = 'INFO'
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        Logger instance
    """
    # Convert level string to logging constant
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    log_level = level_map.get(level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_device(device_str: str = 'auto') -> str:
    """
    Get appropriate torch device.
    
    Args:
        device_str: Device specification ('auto', 'cuda', 'cpu', 'mps')
        
    Returns:
        Device string
    """
    if device_str == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    else:
        return device_str


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directory_structure(base_dir: str):
    """
    Create standard directory structure for experiments.
    
    Args:
        base_dir: Base directory for experiment
    """
    base_path = Path(base_dir)
    
    directories = [
        'checkpoints',
        'logs',
        'plots',
        'runs',  # TensorBoard
        'data',
        'results'
    ]
    
    for dir_name in directories:
        (base_path / dir_name).mkdir(parents=True, exist_ok=True)
    
    return base_path
