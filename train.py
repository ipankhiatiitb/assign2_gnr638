#!/usr/bin/env python3
"""
Unified training script for satellite image classification using PyTorch
Trains models based on configuration and command-line arguments
"""

import os
import sys
import argparse
import numpy as np
import torch

from config import create_parser, get_config, print_config, get_model_config, get_available_models
from models import get_training_function


def set_seeds(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seeds set to {seed}")


def setup_gpu(config):
    """Configure GPU settings"""
    print(f"\nDevice Configuration:")
    print(f"  Device: {config['device']}")
    print(f"  CUDA Available: {config['cuda_available']}")
    if config['cuda_available']:
        print(f"  GPU Name: {config['gpu_name']}")
        print(f"  GPU Count: {config['gpu_count']}")
        print("  GPU memory management enabled")
    else:
        print("  No GPU detected. Training will use CPU.")


def validate_model_name(model_name, config):
    """
    Validate model name
    
    Args:
        model_name (str): Model name to validate
        config (dict): Configuration dictionary
        
    Returns:
        str: Validated model name (lowercase)
        
    Raises:
        ValueError: If model not found
    """
    available_models = get_available_models(config)
    if model_name.lower() not in available_models:
        raise ValueError(
            f"Model '{model_name}' not found.\n"
            f"Available models: {', '.join(available_models)}"
        )
    return model_name.lower()


def train_single_model(model_name, config):
    """
    Train a single model
    
    Args:
        model_name (str): Name of the model to train
        config (dict): Configuration dictionary
    """
    # Validate model
    model_name = validate_model_name(model_name, config)
    
    # Get model configuration
    model_config = get_model_config(config, model_name)
    
    print("\n" + "="*70)
    print(f"TRAINING {model_config['name']}")
    print(f"Training Mode: {config['selected_training_mode']}")
    print(f"Description: {config['training_modes'][config['selected_training_mode']]['description']}")
    print("="*70)
    
    # Get training function and execute (pass full config)
    training_func = get_training_function(model_name)
    training_func(model_config, training_mode=config['selected_training_mode'], config=config)
    
    print("\n" + "="*70)
    print(f"{model_config['name']} training completed successfully!")
    print("="*70)


def train_all_models(config):
    """Train all available models
    
    Args:
        config (dict): Configuration dictionary
    """
    available_models = get_available_models(config)
    
    print("\n" + "="*70)
    print(f"TRAINING ALL MODELS ({len(available_models)} models)")
    print(f"Training Mode: {config['selected_training_mode']}")
    print("="*70)
    
    results = {}
    for model_name in available_models:
        try:
            print(f"\n{'='*70}")
            print(f"Training {model_name.upper()}")
            print(f"{'='*70}")
            
            train_single_model(model_name, config)
            results[model_name] = "Success"
            
        except Exception as e:
            print(f"\nError training {model_name}: {e}")
            results[model_name] = f"Failed: {str(e)}"
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for model_name, status in results.items():
        print(f"{model_name:20s}: {status}")
    print("="*70)


def main():
    """Main function"""
    # Create parser with all config options
    base_parser = create_parser()
    
    # Add training-specific arguments
    base_parser.add_argument(
        '--model', '-m',
        type=str,
        help='Model to train (resnet50, efficientnetb0, inceptionv3)',
        default=None
    )
    
    base_parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Train all available models'
    )
    
    base_parser.add_argument(
        '--config', '-c',
        action='store_true',
        help='Show configuration and exit'
    )
    
    base_parser.add_argument(
        '--list-models', '-l',
        action='store_true',
        help='List available models and training modes, then exit'
    )
    
    args = base_parser.parse_args()
    
    # Get configuration from arguments
    config = get_config(args)
    
    # Setup
    print("="*70)
    print("SATELLITE IMAGE CLASSIFICATION - PYTORCH TRAINING")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    
    # Configure GPU
    setup_gpu(config)
    
    # Set random seeds
    set_seeds(config['random_seed'])
    
    # Handle different arguments
    if args.config:
        print_config(config)
        sys.exit(0)
    
    if args.list_models:
        print("\nAvailable models:")
        for model in get_available_models(config):
            model_info = get_model_config(config, model)
            print(f"  - {model:20s} (Input: {model_info['input_size']})")
        
        print("\nAvailable training modes:")
        for mode, mode_config in config['training_modes'].items():
            print(f"  - {mode:20s}: {mode_config['description']}")
        sys.exit(0)
    
    if args.all:
        train_all_models(config)
        sys.exit(0)
    
    if args.model:
        try:
            train_single_model(args.model, config)
            sys.exit(0)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    # No action specified - show help
    base_parser.print_help()
    print(f"\nNote: Use --config to see configuration")
    print(f"      Use --list-models to see available models and training modes")


if __name__ == "__main__":
    main()
