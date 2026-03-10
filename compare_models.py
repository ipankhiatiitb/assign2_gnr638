#!/usr/bin/env python3
"""
Comparative visualization script
Compares all models side by side
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm

from config import get_config, get_model_config, get_available_models, CLASS_NAMES, RESULTS_DIR
from data_utils import get_test_loader
from models import ResNet50Trainer, EfficientNetB0Trainer, InceptionV3Trainer


class ComparisonVisualizer:
    """Generate comparative visualizations across all models"""
    
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
        self.model_results = {}
    
    def evaluate_all_models(self):
        """Evaluate all models on test set"""
        print("\n" + "="*70)
        print("EVALUATING ALL MODELS")
        print("="*70 + "\n")
        
        for model_name in get_available_models(self.config):
            print(f"Evaluating {model_name}...")
            model_config = get_model_config(model_name)
            best_model_path = model_config['best_model_path']
            
            if not os.path.exists(best_model_path):
                print(f"⚠ Model not found: {best_model_path}")
                continue
            
            trainer_class = self.trainers[model_name.lower()]
            trainer = trainer_class(model_config, model_name.replace('_', ' ').title())
            trainer.load_model(best_model_path)
            
            # Get predictions
            test_loader = get_test_loader(model_config['input_size'])
            true_labels = []
            predicted_labels = []
            
            trainer.model.eval()
            with torch.no_grad():
                for images, labels in tqdm(test_loader, desc=f"  {model_name}", leave=False):
                    images = images.to(self.device)
                    outputs = trainer.model(images)
                    _, predictions = torch.max(outputs, 1)
                    predicted_labels.extend(predictions.cpu().numpy())
                    true_labels.extend(labels.numpy())
            
            # Compute metrics
            accuracy = accuracy_score(true_labels, predicted_labels)
            report = classification_report(true_labels, predicted_labels, 
                                          target_names=self.class_names, 
                                          output_dict=True, zero_division=0)
            
            self.model_results[model_name] = {
                'accuracy': accuracy,
                'report': report,
                'true_labels': true_labels,
                'predicted_labels': predicted_labels,
            }
            print(f"✓ {model_name}: {accuracy*100:.2f}% accuracy\n")
    
    def plot_accuracy_comparison(self):
        """Plot accuracy comparison across all models"""
        print("Generating accuracy comparison...")
        
        model_names = list(self.model_results.keys())
        accuracies = [self.model_results[m]['accuracy']*100 for m in model_names]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(model_names, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        
        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax.text(acc + 1, i, f'{acc:.2f}%', va='center', fontweight='bold')
        
        ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Model Accuracy Comparison (Test Set)', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 110])
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(self.results_dir, 'model_accuracy_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"✓ Accuracy comparison saved: {path}")
        plt.close()
    
    def plot_per_class_comparison(self):
        """Compare per-class F1-scores across models"""
        print("Generating per-class metrics comparison...")
        
        model_names = list(self.model_results.keys())
        f1_scores = {}
        
        for model_name in model_names:
            report = self.model_results[model_name]['report']
            f1_scores[model_name] = [report[cls]['f1-score'] for cls in self.class_names]
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(18, 8))
        
        x = np.arange(len(self.class_names))
        width = 0.25
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for idx, model_name in enumerate(model_names):
            offset = (idx - 1) * width
            ax.bar(x + offset, f1_scores[model_name], width, 
                  label=model_name, alpha=0.8, color=colors[idx])
        
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class F1-Score Comparison Across Models', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(self.results_dir, 'per_class_f1_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"✓ Per-class comparison saved: {path}")
        plt.close()
    
    def plot_confusion_matrices(self):
        """Plot all confusion matrices side by side"""
        print("Generating confusion matrices comparison...")
        
        model_names = list(self.model_results.keys())
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        for idx, model_name in enumerate(model_names):
            true_labels = self.model_results[model_name]['true_labels']
            pred_labels = self.model_results[model_name]['predicted_labels']
            
            cm = confusion_matrix(true_labels, pred_labels)
            
            # Normalize for better visualization
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues', 
                       ax=axes[idx], cbar_kws={'label': 'Normalized Count'},
                       xticklabels=False, yticklabels=False)
            axes[idx].set_title(f'{model_name}\n(Accuracy: {self.model_results[model_name]["accuracy"]*100:.2f}%)',
                              fontsize=12, fontweight='bold')
        
        fig.suptitle('Confusion Matrices Comparison (Test Set)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(self.results_dir, 'confusion_matrices_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrices comparison saved: {path}")
        plt.close()
    
    def generate_comparison_report(self):
        """Generate text comparison report"""
        print("Generating comparison report...")
        
        report_path = os.path.join(self.results_dir, 'model_comparison_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MODEL COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Overall accuracy comparison
            f.write("OVERALL ACCURACY COMPARISON (TEST SET)\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Model':<20} {'Accuracy':<15} {'Rank':<10}\n")
            f.write("-"*80 + "\n")
            
            sorted_models = sorted(self.model_results.items(), 
                                  key=lambda x: x[1]['accuracy'], reverse=True)
            for rank, (model_name, results) in enumerate(sorted_models, 1):
                acc = results['accuracy'] * 100
                f.write(f"{model_name:<20} {acc:>6.2f}%{'':<7} Rank #{rank}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("PER-CLASS METRICS\n")
            f.write("="*80 + "\n\n")
            
            # Per-class metrics
            for model_name in sorted(self.model_results.keys()):
                report = self.model_results[model_name]['report']
                f.write(f"\n{model_name.upper()}\n")
                f.write("-"*80 + "\n")
                f.write(f"{'Class':<20} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}\n")
                f.write("-"*80 + "\n")
                
                for class_name in self.class_names:
                    p = report[class_name]['precision']
                    r = report[class_name]['recall']
                    f1 = report[class_name]['f1-score']
                    f.write(f"{class_name:<20} {p:>6.2f}{'':<8} {r:>6.2f}{'':<8} {f1:>6.2f}\n")
                
                # Weighted average
                f.write("-"*80 + "\n")
                p = report['weighted avg']['precision']
                r = report['weighted avg']['recall']
                f1 = report['weighted avg']['f1-score']
                f.write(f"{'Weighted Avg':<20} {p:>6.2f}{'':<8} {r:>6.2f}{'':<8} {f1:>6.2f}\n")
        
        print(f"✓ Comparison report saved: {report_path}")
    
    def generate_all_comparisons(self):
        """Generate all comparison visualizations"""
        self.evaluate_all_models()
        self.plot_accuracy_comparison()
        self.plot_per_class_comparison()
        self.plot_confusion_matrices()
        self.generate_comparison_report()
        
        print("\n" + "="*70)
        print("✓ ALL COMPARISON VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate comparative visualizations across all models'
    )
    args = parser.parse_args()
    
    config = get_config()
    visualizer = ComparisonVisualizer(config)
    visualizer.generate_all_comparisons()


if __name__ == "__main__":
    main()
