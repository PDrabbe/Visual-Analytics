"""
Main entry point for ProtoNet training and evaluation.
"""

import argparse
from pathlib import Path

from utils.helpers import load_config, setup_logging, get_device, set_seed, create_directory_structure
from data.dataset import create_dataloader
from data.sampler import create_episodic_dataloader
from models.encoder import get_encoder
from models.distance_metrics import get_distance_metric
from models.protonet import ProtoNet
from models.storage import HybridPrototypeStore
from training.trainer import ProtoNetTrainer
from inference.predictor import DrawingPredictor


def train(config_path: str):
    """
    Train ProtoNet model.
    
    Args:
        config_path: Path to configuration file
    """
    # Load config
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(
        log_file=config['logging'].get('log_file'),
        level=config['logging'].get('level', 'INFO')
    )
    logger.info("Starting ProtoNet training")
    
    # Set seed
    set_seed(config['system'].get('seed', 42))
    
    # Get device
    device = get_device(config['system'].get('device', 'auto'))
    logger.info(f"Using device: {device}")
    
    # Create directories
    create_directory_structure('.')
    
    # Load datasets
    logger.info("Loading datasets...")
    train_loader, train_dataset = create_dataloader(
        dataset_type=config['data']['dataset'],
        config=config['data'],
        split='train',
        num_workers=config['system'].get('num_workers', 4)
    )
    
    val_loader, val_dataset = create_dataloader(
        dataset_type=config['data']['dataset'],
        config=config['data'],
        split='val',
        num_workers=config['system'].get('num_workers', 4)
    )
    
    # Create episodic samplers
    logger.info("Creating episodic samplers...")
    train_sampler = create_episodic_dataloader(
        dataset=train_dataset,
        n_way=config['data']['n_way'],
        n_support=config['data']['n_support'],
        n_query=config['data']['n_query'],
        n_episodes=config['training']['num_episodes']
    )
    
    val_sampler = create_episodic_dataloader(
        dataset=val_dataset,
        n_way=config['evaluation']['n_way'],
        n_support=config['data']['n_support'],
        n_query=config['data']['n_query'],
        n_episodes=config['training']['validation_episodes']
    )
    
    # Create model
    logger.info("Building model...")
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
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = ProtoNetTrainer(
        model=model,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        config=config['training'],
        device=device
    )
    
    # Train
    trainer.train()
    
    logger.info("Training complete!")


def evaluate(config_path: str, model_path: str):
    """Evaluate trained model (not yet implemented)."""
    config = load_config(config_path)
    logger = setup_logging(level=config['logging'].get('level', 'INFO'))
    logger.info("Evaluation not yet implemented")


def inference_demo(model_path: str, image_path: str):
    """
    Run inference demo.
    
    Args:
        model_path: Path to trained model
        image_path: Path to test image
    """
    # Setup logging
    logger = setup_logging(level='INFO')
    logger.info("Running inference demo")
    
    # Create predictor
    predictor = DrawingPredictor(model_path)
    
    # Make prediction
    result = predictor.predict(image_path)
    
    logger.info(f"Prediction: {result['class']}")
    logger.info(f"Confidence: {result['confidence']:.2%}")
    logger.info(f"Top-3 predictions:")
    for pred in result['top_k']:
        logger.info(f"  - {pred['class']}: {pred['confidence']:.2%}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='ProtoNet for Drawing Recognition')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    eval_parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model'
    )
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model'
    )
    infer_parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to test image'
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args.config)
    elif args.command == 'evaluate':
        evaluate(args.config, args.model)
    elif args.command == 'infer':
        inference_demo(args.model, args.image)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
