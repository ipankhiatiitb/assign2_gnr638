"""
Advanced Evaluation Module - Orchestrates Layer-Wise Probing and Corruption Robustness

Runs Assignment Scenarios 4.4 and 4.5:
- Scenario 4.4: Corruption Robustness Evaluation
- Scenario 4.5: Layer-Wise Feature Probing
"""

import os
import sys
import torch
import argparse
from data_utils import get_data_loaders
from layer_wise_probing import run_layer_wise_probing
from corruption_robustness import evaluate_corruption_robustness


def run_advanced_evaluations(model, model_name, config, device='cuda'):
    """
    Run advanced evaluations (Scenarios 4.4 and 4.5)
    
    Args:
        model: Trained model
        model_name (str): Model name
        config (dict): Configuration dictionary
        device: Computation device
    """
    print("\n" + "="*80)
    print(f"ADVANCED EVALUATIONS: {model_name.upper()}")
    print("="*80)
    
    # Get data loaders
    train_loader, val_loader, _ = get_data_loaders(config['models_config'][model_name.lower()]['input_size'])
    
    # Scenario 4.4: Corruption Robustness
    print("\n[Scenario 4.4: Corruption Robustness Evaluation]")
    print("Evaluating robustness under:")
    print("  - Gaussian noise (σ = 0.05, 0.10, 0.20)")
    print("  - Motion blur")
    print("  - Brightness shift")
    print("  (Corruptions applied ONLY at evaluation time)")
    
    try:
        corruption_results = evaluate_corruption_robustness(model, model_name, val_loader, device)
        print("✓ Corruption robustness evaluation completed")
    except Exception as e:
        print(f"✗ Error in corruption robustness evaluation: {e}")
        corruption_results = None
    
    # Scenario 4.5: Layer-Wise Feature Probing
    print("\n[Scenario 4.5: Layer-Wise Feature Probing]")
    print("Analyzing semantic abstraction across network depth:")
    print("  - Early layers")
    print("  - Middle layers")
    print("  - Final layers")
    
    try:
        layerwise_results = run_layer_wise_probing(model, model_name, train_loader, val_loader, device)
        print("✓ Layer-wise probing completed")
    except Exception as e:
        print(f"✗ Error in layer-wise probing: {e}")
        layerwise_results = None
    
    print("\n" + "="*80)
    print(f"Advanced evaluations for {model_name} completed!")
    print("="*80)
    
    return {
        'corruption': corruption_results,
        'layer_wise': layerwise_results,
    }


def main():
    """Main entry point for advanced evaluations"""
    parser = argparse.ArgumentParser(
        description='Run advanced evaluations (Corruption Robustness & Layer-Wise Probing)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_advanced.py --model resnet50 --model-path results/resnet50_final_model.pth
  python evaluate_advanced.py --model efficientnetb0 --model-path results/efficientnetb0_final_model.pth
  python evaluate_advanced.py --model inceptionv3 --model-path results/inceptionv3_final_model.pth --device cuda:0
        """
    )
    
    parser.add_argument('--model', type=str, required=True, 
                       choices=['resnet50', 'efficientnetb0', 'inceptionv3'],
                       help='Model to evaluate')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda, cpu, cuda:0, etc.)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation (default: 32)')
    
    args = parser.parse_args()
    
    # Import config
    from config import get_config
    config = get_config()
    
    # Device
    device = torch.device(args.device)
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = torch.load(args.model_path, map_location=device)
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded: {args.model}")
    
    # Run evaluations
    results = run_advanced_evaluations(model, args.model, config, device)
    
    print("\n✓ All advanced evaluations completed successfully!")


if __name__ == '__main__':
    main()
