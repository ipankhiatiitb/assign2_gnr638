#!/usr/bin/env python3
"""
Comprehensive visualization script for model results
Generates all visualizations for trained models
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from config import get_config, create_parser, get_model_config, get_available_models, CLASS_NAMES, RESULTS_DIR
from data_utils import get_test_loader
from models import ResNet50Trainer, EfficientNetB0Trainer, InceptionV3Trainer


def create_visualization_parser():
    """Create argument parser for visualization script"""
    parser = create_parser()
    
    # Add model selection arguments (these are specific to visualization/evaluation scripts)
    parser.add_argument('--model', '-m', 
                       choices=['resnet50', 'efficientnetb0', 'inceptionv3'],
                       help='Model to visualize results for')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Generate visualizations for all trained models')
    parser.add_argument('--list-models', '-l', action='store_true',
                       help='List available models')
    
    return parser


class VisualizationGenerator:
    """Generate comprehensive visualizations for trained models"""
    
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.class_names = config['class_names']
        self.num_classes = config['num_classes']
        self.results_dir = config['results_dir']
        self.trainers = {
            'resnet50': ResNet50Trainer,
            'efficientnetb0': EfficientNetB0Trainer,
            'inceptionv3': InceptionV3Trainer,
        }
    
    def generate_all_visualizations(self, model_name):
        """Generate all visualizations for a specific model"""
        print(f"\n{'='*70}")
        print(f"GENERATING VISUALIZATIONS FOR {model_name.upper()}")
        print(f"{'='*70}\n")
        
        model_config = get_model_config(self.config, model_name)
        trainer_class = self.trainers[model_name.lower()]
        trainer = trainer_class(model_config, model_name.replace('_', ' ').title())
        
        # Build model first
        trainer.build_model(training_mode='linear_probe')
        
        # Load best model
        best_model_path = model_config['best_model_path']
        if not os.path.exists(best_model_path):
            print(f"ERROR: Model not found at {best_model_path}")
            print(f"Please train the model first: python train.py --model {model_name}")
            return False
        
        trainer.load_model(best_model_path)
        print(f"✓ Loaded model from {best_model_path}")
        
        # 1. Training History
        print("\n[1/5] Generating training history plot...")
        history_plot = model_config['history_plot']
        if os.path.exists(history_plot):
            print(f"✓ Training history already saved: {history_plot}")
        else:
            print(f"⚠ Training history not found. Run training first.")
        
        # 2. Confusion Matrix
        print("[2/5] Generating confusion matrix...")
        self.plot_confusion_matrix(trainer, model_name, model_config)
        
        # 3. Per-Class Metrics
        print("[3/5] Generating per-class metrics visualization...")
        self.plot_per_class_metrics(trainer, model_name, model_config)
        
        # 4. Model Architecture Summary
        print("[4/5] Generating model architecture summary...")
        self.plot_model_summary(trainer, model_name, model_config)
        
        # 5. Sample Predictions
        print("[5/5] Generating sample predictions...")
        self.plot_sample_predictions(trainer, model_name, model_config)
        
        print(f"\n{'='*70}")
        print(f"✓ All visualizations generated successfully!")
        print(f"{'='*70}\n")
        return True
    
    def plot_confusion_matrix(self, trainer, model_name, model_config):
        """Generate and save confusion matrix"""
        # Get predictions on test set
        test_loader = get_test_loader(model_config['input_size'])
        true_labels = []
        predicted_labels = []
        
        trainer.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Computing predictions"):
                images = images.to(self.device)
                outputs = trainer.model(images)
                _, predictions = torch.max(outputs, 1)
                predicted_labels.extend(predictions.cpu().numpy())
                true_labels.extend(labels.numpy())
        
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Plot with annotations
        plt.figure(figsize=(20, 18))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title(f'{model_name} - Confusion Matrix\n(Test Set)', 
                 fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_path = model_config['confusion_matrix_path']
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix saved: {cm_path}")
        plt.close()
    
    def plot_per_class_metrics(self, trainer, model_name, model_config):
        """Generate per-class precision, recall, F1-score visualization"""
        # Get predictions
        test_loader = get_test_loader(model_config['input_size'])
        true_labels = []
        predicted_labels = []
        
        trainer.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Computing metrics"):
                images = images.to(self.device)
                outputs = trainer.model(images)
                _, predictions = torch.max(outputs, 1)
                predicted_labels.extend(predictions.cpu().numpy())
                true_labels.extend(labels.numpy())
        
        # Get classification report
        report = classification_report(true_labels, predicted_labels, 
                                      target_names=self.class_names, 
                                      output_dict=True, zero_division=0)
        
        # Extract metrics
        classes = list(self.class_names)
        precision = [report[cls]['precision'] for cls in classes]
        recall = [report[cls]['recall'] for cls in classes]
        f1 = [report[cls]['f1-score'] for cls in classes]
        
        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # Precision
        axes[0].barh(classes, precision, color='steelblue', alpha=0.8)
        axes[0].set_xlabel('Precision', fontweight='bold')
        axes[0].set_title(f'{model_name} - Per-Class Precision', fontsize=12, fontweight='bold')
        axes[0].set_xlim([0, 1])
        for i, v in enumerate(precision):
            axes[0].text(v + 0.02, i, f'{v:.2f}', va='center')
        
        # Recall
        axes[1].barh(classes, recall, color='lightcoral', alpha=0.8)
        axes[1].set_xlabel('Recall', fontweight='bold')
        axes[1].set_title(f'{model_name} - Per-Class Recall', fontsize=12, fontweight='bold')
        axes[1].set_xlim([0, 1])
        for i, v in enumerate(recall):
            axes[1].text(v + 0.02, i, f'{v:.2f}', va='center')
        
        # F1-Score
        axes[2].barh(classes, f1, color='mediumseagreen', alpha=0.8)
        axes[2].set_xlabel('F1-Score', fontweight='bold')
        axes[2].set_title(f'{model_name} - Per-Class F1-Score', fontsize=12, fontweight='bold')
        axes[2].set_xlim([0, 1])
        for i, v in enumerate(f1):
            axes[2].text(v + 0.02, i, f'{v:.2f}', va='center')
        
        plt.tight_layout()
        
        metrics_path = os.path.join(self.results_dir, f'{model_name.lower()}_per_class_metrics.png')
        plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
        print(f"✓ Per-class metrics saved: {metrics_path}")
        plt.close()
    
    def plot_model_summary(self, trainer, model_name, model_config):
        """Generate model architecture and parameters summary"""
        fig = plt.figure(figsize=(12, 8))
        
        # Get model info
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        
        # Calculate FLOPs if available
        try:
            from fvcore.nn import FlopCounterMode
            input_size = model_config['input_size']
            sample_input = torch.randn(1, 3, input_size[0], input_size[1]).to(self.device)
            with FlopCounterMode(trainer.model) as fmode:
                _ = trainer.model(sample_input)
            flops = fmode.flop_counts['Global']
            flops_str = f"{flops/1e9:.2f}G"
        except:
            flops_str = "N/A"
        
        # Create summary text
        summary_text = f"""
{model_name} - MODEL ARCHITECTURE SUMMARY

Architecture:
  Model Type: {model_config['name']}
  Input Size: {model_config['input_size']} pixels
  Output Classes: {self.num_classes}

Parameters:
  Total Parameters: {total_params:,}
  Trainable Parameters: {trainable_params:,}
  Non-trainable Parameters: {total_params - trainable_params:,}

Computational Complexity:
  FLOPs: {flops_str}
  Device: {self.device}

Configuration:
  Dropout Rate: {model_config['dropout_rate']}
  Batch Size: {self.config['batch_size']}
  Learning Rate: {self.config['initial_learning_rate']}
  Optimizer: Adam
  Loss Function: CrossEntropyLoss
  Activation: ReLU
        """
        
        plt.text(0.05, 0.95, summary_text, transform=fig.transFigure, 
                fontfamily='monospace', fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.axis('off')
        
        summary_path = os.path.join(self.results_dir, f'{model_name.lower()}_architecture_summary.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        print(f"✓ Architecture summary saved: {summary_path}")
        plt.close()
    
    def plot_sample_predictions(self, trainer, model_name, model_config):
        """Generate visualization of sample predictions"""
        from data_utils import SatelliteImageDataset
        import torchvision.transforms as transforms
        from PIL import Image
        
        # Create transforms without augmentation
        input_size = model_config['input_size']
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Get test dataset
        test_dataset = SatelliteImageDataset(
            self.config['test_dir'],
            self.config['class_names'],
            transform=transform
        )
        
        # Select random samples
        np.random.seed(42)
        indices = np.random.choice(len(test_dataset), min(9, len(test_dataset)), replace=False)
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        trainer.model.eval()
        with torch.no_grad():
            for idx, sample_idx in enumerate(indices):
                image, label = test_dataset[sample_idx]
                image_tensor = image.unsqueeze(0).to(self.device)
                
                output = trainer.model(image_tensor)
                probs = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, 1)
                
                # Denormalize image for display
                image_np = image.cpu().numpy().transpose(1, 2, 0)
                # Undo normalization: (x - mean) / std => x = (x * std) + mean
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_np = (image_np * std + mean)
                image_np = np.clip(image_np, 0, 1)
                
                axes[idx].imshow(image_np)
                true_label = self.config['class_names'][label]
                pred_label = self.config['class_names'][predicted.item()]
                conf = confidence.item() * 100
                
                color = 'green' if label == predicted.item() else 'red'
                axes[idx].set_title(
                    f'True: {true_label}\nPred: {pred_label} ({conf:.1f}%)',
                    fontsize=10, fontweight='bold', color=color
                )
                axes[idx].axis('off')
        
        plt.suptitle(f'{model_name} - Sample Predictions (Test Set)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        samples_path = os.path.join(self.results_dir, f'{model_name.lower()}_sample_predictions.png')
        plt.savefig(samples_path, dpi=150, bbox_inches='tight')
        print(f"✓ Sample predictions saved: {samples_path}")
        plt.close()


def main():
    parser = create_visualization_parser()
    args = parser.parse_args()
    
    # Get config with parsed arguments
    config = get_config(args)
    
    if args.list_models:
        print("\nAvailable models:")
        for model in get_available_models(config):
            print(f"  - {model}")
        print()
        return
    
    if not args.model and not args.all:
        parser.print_help()
        return
    
    visualizer = VisualizationGenerator(config)
    
    if args.all:
        for model in get_available_models(config):
            visualizer.generate_all_visualizations(model)
    else:
        visualizer.generate_all_visualizations(args.model)


if __name__ == "__main__":
    main()
