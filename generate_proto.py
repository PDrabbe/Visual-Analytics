# generate_prototypes.py
"""Generate prototypes for the trained model."""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from models.encoder import get_encoder
from models.distance_metrics import get_distance_metric
from models.protonet import ProtoNet
from data.dataset import create_dataloader
from utils.helpers import load_config, get_device

def compute_prototypes(model, dataloader, device, num_samples_per_class=100):
    """
    Compute prototypes (mean embeddings) for each class.
    
    Args:
        model: ProtoNet model
        dataloader: DataLoader with training data
        device: Device to use
        num_samples_per_class: How many samples to use per class
        
    Returns:
        Dictionary mapping class names to prototype tensors
    """
    model.eval()
    
    # Collect embeddings for each class
    class_embeddings = {}
    dataset = dataloader.dataset
    
    print("Computing prototypes for each class...")
    
    for class_idx, class_name in dataset.idx_to_class.items():
        # Get sample indices for this class
        sample_indices = dataset.get_class_samples(class_idx)
        
        # Limit samples
        sample_indices = sample_indices[:num_samples_per_class]
        
        embeddings = []
        
        with torch.no_grad():
            for idx in tqdm(sample_indices, desc=f"Processing {class_name}", leave=False):
                image, label = dataset[idx]
                image = image.unsqueeze(0).to(device)  # Add batch dimension
                
                # Get embedding
                embedding = model.encoder(image)
                embeddings.append(embedding.squeeze(0).cpu())
        
        # Compute prototype (mean of embeddings)
        prototype = torch.stack(embeddings).mean(dim=0)
        class_embeddings[class_name] = prototype
        
        print(f"  {class_name}: prototype computed from {len(embeddings)} samples")
    
    return class_embeddings


def main():
    print("Generating prototypes for trained model...")
    
    config = load_config('config/config.yaml')
    device = get_device(config['system'].get('device', 'auto'))
    
    # Load checkpoint
    print("Loading model from checkpoint...")
    
    # Load checkpoint
    checkpoint_path = 'checkpoints/best_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Build model
    encoder = get_encoder(
        encoder_type=config['model']['encoder'],
        config={
            'num_channels': config['data']['num_channels'],
            'embedding_dim': config['model']['embedding_dim'],
            **config['model'].get(config['model']['encoder'], {})
        }
    )
    
    distance_metric = get_distance_metric(
        metric_type=config['model']['distance_metric'],
        embedding_dim=config['model']['embedding_dim']
    )
    
    model = ProtoNet(
        encoder=encoder,
        distance_metric=distance_metric,
        embedding_dim=config['model']['embedding_dim']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded.")
    
    # Load training data
    print("Loading training data...")
    
    train_loader, train_dataset = create_dataloader(
        dataset_type=config['data']['dataset'],
        config=config['data'],
        split='train',
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Loaded {len(train_dataset)} training samples, classes: {list(train_dataset.idx_to_class.values())}")
    
    # Compute prototypes
    print("Computing prototypes...")
    
    prototypes = compute_prototypes(
        model=model,
        dataloader=train_loader,
        device=device,
        num_samples_per_class=100  # Use 100 samples per class
    )
    
    print(f"Computed {len(prototypes)} prototypes.")
    
    # Save checkpoint with prototypes
    print("Saving checkpoint with prototypes...")
    
    checkpoint['prototypes'] = prototypes
    checkpoint['metadata'] = {
        class_name: {
            'created_at': 'training',
            'source': 'training_data',
            'num_samples': 100
        }
        for class_name in prototypes.keys()
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved to {checkpoint_path}")
    
    # Also save a backup
    new_checkpoint_path = 'checkpoints/best_model_with_prototypes.pt'
    torch.save(checkpoint, new_checkpoint_path)
    print(f"Backup saved to {new_checkpoint_path}")
    
    # Verify
    verify_checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'prototypes' in verify_checkpoint:
        print(f"Verified: prototypes for {list(verify_checkpoint['prototypes'].keys())}")
    else:
        print("ERROR: Prototypes not found in saved checkpoint!")
        return
    
    print("Done. Model is ready for inference.")


if __name__ == '__main__':
    main()