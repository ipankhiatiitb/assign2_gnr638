#!/usr/bin/env python3
"""
PyTorch model evaluation script for satellite image classification
Evaluates trained models on test set and generates reports
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm

from config import (
    get_model_config, get_available_models, CLASS_NAMES, 
    RESULTS_DIR, VERBOSE, NUM_CLASSES, DEVICE
)
from data_utils import get_test_loader
from models import ResNet50Trainer, EfficientNetB0Trainer, InceptionV3Trainer


DEVICE = DEVICE


class ModelEvaluator:
    """Evaluate trained PyTorch models on test set"""
    
    def __init__(self):
        self.results = {}
        self.device = DEVICE
        
    def load_model(self, model_path, trainer_class, model_config):
        """Load a trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create trainer and build model
        trainer = trainer_class(model_config, model_config['name'])
        trainer.build_model()
        trainer.load_model(model_path)
        
        return trainer.model
    
    def evaluate_model(self, model, test_loader, model_name):
        """
        Evaluate model on test set
        
        Args:
            model: Trained PyTorch model
            test_loader: Test data loader
            model_name: Name of the model
        """
        print("\n" + "="*70)
        print(f"EVALUATING {model_name.upper()}")
        print("="*70)
        
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc='Evaluating')
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
        # Store results
        self.results[model_name] = {
            'model': model,
            'predicted_labels': all_preds,
            'true_labels': all_labels,
            'accuracy': accuracy
        }
        
        return accuracy
    
    def plot_confusion_matrix(self, model_name, figsize=(20, 18)):
        """Plot confusion matrix"""
        if model_name not in self.results:
            print(f"Model {model_name} not found in results")
            return
        
        result = self.results[model_name]
        predicted_labels = result['predicted_labels']
        true_labels = result['true_labels']
        
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Plot
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save
        model_key = model_name.lower().replace(' ', '')
        config = get_model_config(model_key)
        plt.savefig(config['confusion_matrix_path'], dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {config['confusion_matrix_path']}")
    
    def print_classification_report(self, model_name):
        """Print and save classification report"""
        if model_name not in self.results:
            print(f"Model {model_name} not found in results")
            return
        
        result = self.results[model_name]
        predicted_labels = result['predicted_labels']
        true_labels = result['true_labels']
        
        print("\n" + "="*70)
        print(f"CLASSIFICATION REPORT - {model_name.upper()}")
        print("="*70)
        
        report = classification_report(true_labels, predicted_labels, 
                                      target_names=CLASS_NAMES, digits=4)
        print(report)
        
        # Save report
        model_key = model_name.lower().replace(' ', '')
        config = get_model_config(model_key)
        with open(config['classification_report_path'], 'w') as f:
            f.write(f"Classification Report - {model_name}\n")
            f.write("="*70 + "\n\n")
            f.write(report)
        
        print(f"Report saved to {config['classification_report_path']}")
    
    def plot_accuracy_comparison(self):
        """Plot accuracy comparison of all models"""
        if not self.results:
            print("No results to compare")
            return
        
        print("\n" + "="*70)
        print("PLOTTING ACCURACY COMPARISON")
        print("="*70)
        
        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.4f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Model', fontsize=12)
        plt.title('Test Set Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.ylim([0, 1])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(RESULTS_DIR, 'accuracy_comparison.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Accuracy comparison saved to {save_path}")
        
        # Print summary
        print("\nAccuracy Summary:")
        for model, acc in zip(models, accuracies):
            print(f"  {model:25s}: {acc:.4f}")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*70)
        print("GENERATING SUMMARY REPORT")
        print("="*70)
        
        report_text = "="*70 + "\n"
        report_text += "MODEL EVALUATION SUMMARY REPORT\n"
        report_text += "="*70 + "\n\n"
        
        if not self.results:
            report_text += "No models evaluated yet.\n"
        else:
            # Model accuracies
            report_text += "Test Set Accuracies:\n"
            report_text += "-"*70 + "\n"
            
            models = list(self.results.keys())
            accuracies = [self.results[m]['accuracy'] for m in models]
            
            for model, acc in zip(models, accuracies):
                report_text += f"{model:25s}: {acc:.4f}\n"
            
            best_model_idx = np.argmax(accuracies)
            best_model = models[best_model_idx]
            best_acc = accuracies[best_model_idx]
            
            report_text += "\n" + "-"*70 + "\n"
            report_text += f"Best Model: {best_model} (Accuracy: {best_acc:.4f})\n"
            report_text += "-"*70 + "\n\n"
            
            # Detailed results per class
            report_text += "PER-CLASS PERFORMANCE:\n"
            report_text += "="*70 + "\n\n"
            
            for model_name in models:
                result = self.results[model_name]
                predicted_labels = result['predicted_labels']
                true_labels = result['true_labels']
                
                cm = confusion_matrix(true_labels, predicted_labels)
                
                report_text += f"\n{model_name}:\n"
                report_text += "-"*70 + "\n"
                report_text += f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n"
                report_text += "-"*70 + "\n"
                
                for i, class_name in enumerate(CLASS_NAMES):
                    tp = cm[i, i]
                    fp = cm[:, i].sum() - tp
                    fn = cm[i, :].sum() - tp
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    report_text += f"{class_name:<25} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f}\n"
        
        # Save report
        save_path = os.path.join(RESULTS_DIR, 'evaluation_summary_report.txt')
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nReport saved to {save_path}")


def get_trainer_class(model_name):
    """Get trainer class for a model"""
    trainer_map = {
        'resnet50': ResNet50Trainer,
        'efficientnetb0': EfficientNetB0Trainer,
        'inceptionv3': InceptionV3Trainer,
    }
    return trainer_map.get(model_name.lower())


def evaluate_single_model(model_name, test_loader):
    """Evaluate a single model"""
    model_key = model_name.lower()
    config = get_model_config(model_key)
    model_name_display = config['name']
    
    evaluator = ModelEvaluator()
    
    print(f"\nLoading {model_name_display}...")
    trainer_class = get_trainer_class(model_name)
    model = evaluator.load_model(config['model_path'], trainer_class, config)
    
    # Evaluate
    evaluator.evaluate_model(model, test_loader, model_name_display)
    evaluator.print_classification_report(model_name_display)
    evaluator.plot_confusion_matrix(model_name_display)
    
    return evaluator


def evaluate_all_models(test_loader):
    """Evaluate all trained models"""
    evaluator = ModelEvaluator()
    available_models = get_available_models()
    
    for model_name in available_models:
        config = get_model_config(model_name)
        
        if not os.path.exists(config['model_path']):
            print(f"Model file not found for {model_name}, skipping...")
            continue
        
        print(f"\nLoading {config['name']}...")
        trainer_class = get_trainer_class(model_name)
        model = evaluator.load_model(config['model_path'], trainer_class, config)
        evaluator.evaluate_model(model, test_loader, config['name'])
    
    # Generate comparison and reports
    if evaluator.results:
        for model_name in list(evaluator.results.keys()):
            evaluator.print_classification_report(model_name)
            evaluator.plot_confusion_matrix(model_name)
        
        evaluator.plot_accuracy_comparison()
        evaluator.generate_summary_report()
    
    return evaluator


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Evaluate trained satellite image classification models using PyTorch"
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Model to evaluate (resnet50, efficientnetb0, inceptionv3)',
        default=None
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Evaluate all trained models'
    )
    
    parser.add_argument(
        '--list-models', '-l',
        action='store_true',
        help='List available models'
    )
    
    args = parser.parse_args()
    
    # Setup
    print("="*70)
    print("SATELLITE IMAGE CLASSIFICATION - PYTORCH EVALUATION")
    print("="*70)
    
    # List models
    if args.list_models:
        print("\nAvailable models:")
        for model in get_available_models():
            config = get_model_config(model)
            print(f"  - {model:20s} ({config['name']})")
        sys.exit(0)
    
    # Load test data - get image size from first model config
    if args.model:
        config = get_model_config(args.model.lower())
        img_size = config['input_size']
    elif args.all:
        img_size = (224, 224)  # Default size
    else:
        img_size = (224, 224)
    
    print("\nLoading test data...")
    test_loader = get_test_loader(img_size=img_size)
    
    # Evaluate
    if args.all:
        print("\nEvaluating all models...")
        evaluate_all_models(test_loader)
    elif args.model:
        try:
            model_name = args.model.lower()
            available_models = get_available_models()
            if model_name not in available_models:
                print(f"Error: Model '{args.model}' not found.")
                print(f"Available models: {', '.join(available_models)}")
                sys.exit(1)
            print(f"\nEvaluating {model_name}...")
            evaluate_single_model(model_name, test_loader)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
