#!/usr/bin/env python3
"""
Feature Embeddings Visualization Script
Extracts and visualizes feature embeddings from trained models using PCA, t-SNE, and UMAP
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import create_parser, get_config, get_model_config, CLASS_NAMES
from data_utils import get_test_loader
from models import ResNet50Trainer, EfficientNetB0Trainer, InceptionV3Trainer


class FeatureExtractor:
    """Extract feature embeddings from trained models"""
    
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.class_names = CLASS_NAMES
        self.trainers = {
            'resnet50': ResNet50Trainer,
            'efficientnetb0': EfficientNetB0Trainer,
            'inceptionv3': InceptionV3Trainer,
        }
    
    def extract_features(self, model_name):
        """Extract feature embeddings from a trained model"""
        print(f"\n{'='*70}")
        print(f"EXTRACTING FEATURES FROM {model_name.upper()}")
        print(f"{'='*70}\n")
        
        model_config = get_model_config(self.config, model_name)
        trainer_class = self.trainers[model_name.lower()]
        trainer = trainer_class(model_config, model_name.replace('_', ' ').title())
        
        # Build model
        trainer.build_model(training_mode='linear_probe')
        
        # Load best model
        best_model_path = model_config['best_model_path']
        if not os.path.exists(best_model_path):
            print(f"ERROR: Model not found at {best_model_path}")
            return None, None
        
        trainer.load_model(best_model_path)
        print(f"✓ Loaded model from {best_model_path}\n")
        
        # Get test loader
        test_loader = get_test_loader(model_config['input_size'], self.config['batch_size'])
        
        # Extract features
        print("Extracting features from test set...")
        features_list = []
        labels_list = []
        
        # Create feature extraction model (backbone only)
        # Handle DataParallel wrapper
        if isinstance(trainer.model, nn.DataParallel):
            feature_model = trainer.model.module
        else:
            feature_model = trainer.model
        feature_model.eval()
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Feature Extraction'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Extract features (output of backbone)
                features = feature_model(images)
                features = features.view(features.size(0), -1)  # Flatten if needed
                
                features_list.append(features.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
        
        # Concatenate all features and labels
        all_features = np.vstack(features_list)
        all_labels = np.hstack(labels_list)
        
        print(f"✓ Extracted {len(all_features)} features of shape {all_features.shape}")
        print(f"✓ Feature dimension: {all_features.shape[1]}")
        
        # Sample features for faster t-SNE computation (max 500 samples)
        if len(all_features) > 500:
            print(f"\n⚠ Sampling {min(500, len(all_features))} features for faster t-SNE computation...")
            np.random.seed(42)
            sample_indices = np.random.choice(len(all_features), 500, replace=False)
            all_features = all_features[sample_indices]
            all_labels = all_labels[sample_indices]
            print(f"✓ Using {len(all_features)} samples for visualization")
        
        return all_features, all_labels
    
    def visualize_pca(self, features, labels, model_name, n_components=2):
        """Visualize features using PCA"""
        print("\nGenerating PCA visualization...")
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(features)
        
        # Print explained variance
        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"✓ PCA explained variance: {explained_var*100:.2f}%")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 10))
        
        scatter = ax.scatter(
            features_pca[:, 0], 
            features_pca[:, 1],
            c=labels,
            cmap='tab20c',
            alpha=0.9,
            s=30,
            edgecolors='k',
            linewidth=0.3
        )
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax.set_title(f'{model_name} - Feature Embeddings (PCA)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Class Index', fontsize=11)
        
        # Save
        output_path = os.path.join(
            self.config['results_dir'],
            f'{model_name.lower()}_embeddings_pca.png'
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved PCA visualization to {output_path}")
        plt.close()
        
        return features_pca, pca
    
    def visualize_tsne(self, features, labels, model_name, perplexity=30):
        """Visualize features using t-SNE"""
        print("\nGenerating t-SNE visualization (this may take a moment)...")
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=1000, random_state=42)
        features_tsne = tsne.fit_transform(features)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 10))
        
        scatter = ax.scatter(
            features_tsne[:, 0], 
            features_tsne[:, 1],
            c=labels,
            cmap='tab20c',
            alpha=0.9,
            s=30,
            edgecolors='k',
            linewidth=0.3
        )
        
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.set_title(f'{model_name} - Feature Embeddings (t-SNE)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Class Index', fontsize=11)
        
        # Save
        output_path = os.path.join(
            self.config['results_dir'],
            f'{model_name.lower()}_embeddings_tsne.png'
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved t-SNE visualization to {output_path}")
        plt.close()
        
        return features_tsne
    
    def visualize_umap(self, features, labels, model_name, n_neighbors=15, min_dist=0.1):
        """Visualize features using UMAP"""
        try:
            import umap
        except ImportError:
            print("⚠ UMAP not installed. Skipping UMAP visualization.")
            print("  Install it with: pip install umap-learn")
            return None
        
        print("\nGenerating UMAP visualization (this may take a moment)...")
        
        # Apply UMAP
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        features_umap = reducer.fit_transform(features)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 10))
        
        scatter = ax.scatter(
            features_umap[:, 0], 
            features_umap[:, 1],
            c=labels,
            cmap='tab20c',
            alpha=0.9,
            s=30,
            edgecolors='k',
            linewidth=0.3
        )
        
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title(f'{model_name} - Feature Embeddings (UMAP)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Class Index', fontsize=11)
        
        # Save
        output_path = os.path.join(
            self.config['results_dir'],
            f'{model_name.lower()}_embeddings_umap.png'
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved UMAP visualization to {output_path}")
        plt.close()
        
        return features_umap
    
    def analyze_separability(self, features_pca, labels, model_name):
        """Analyze feature separability"""
        print("\n" + "="*70)
        print(f"FEATURE SEPARABILITY ANALYSIS - {model_name.upper()}")
        print("="*70)
        
        # Calculate between-class and within-class distances
        unique_labels = np.unique(labels)
        within_class_dists = []
        between_class_dists = []
        
        # Within-class distances
        for label in unique_labels:
            class_features = features_pca[labels == label]
            if len(class_features) > 1:
                # Calculate pairwise distances
                diffs = class_features[:, np.newaxis, :] - class_features[np.newaxis, :, :]
                dists = np.linalg.norm(diffs, axis=2)
                within_class_dists.extend(dists[np.triu_indices_from(dists, k=1)])
        
        # Between-class distances (sample)
        for i, label1 in enumerate(unique_labels[:5]):  # Sample for efficiency
            for label2 in unique_labels[i+1:6]:
                class1_features = features_pca[labels == label1]
                class2_features = features_pca[labels == label2]
                
                # Sample distances
                diffs = class1_features[:5, np.newaxis, :] - class2_features[np.newaxis, :5, :]
                dists = np.linalg.norm(diffs, axis=2)
                between_class_dists.extend(dists.flatten())
        
        within_mean = np.mean(within_class_dists)
        between_mean = np.mean(between_class_dists)
        separability_ratio = between_mean / within_mean if within_mean > 0 else 0
        
        print(f"\n✓ Within-class distance (mean): {within_mean:.4f}")
        print(f"✓ Between-class distance (mean): {between_mean:.4f}")
        print(f"✓ Separability Ratio: {separability_ratio:.4f}")
        
        if separability_ratio > 2:
            print("  → Excellent class separability! Features are well-separated.")
        elif separability_ratio > 1.5:
            print("  → Good class separability. Features show reasonable separation.")
        else:
            print("  → Poor class separability. Classes may be overlapping.")
        
        return {
            'within_class_dist': within_mean,
            'between_class_dist': between_mean,
            'separability_ratio': separability_ratio
        }
    
    def generate_all_visualizations(self, model_name):
        """Generate all feature embedding visualizations for a model"""
        # Extract features
        features, labels = self.extract_features(model_name)
        if features is None:
            print(f"ERROR: Failed to extract features from {model_name}")
            return False
        
        # PCA visualization
        features_pca, pca = self.visualize_pca(features, labels, model_name)
        
        # Analyze separability
        self.analyze_separability(features_pca, labels, model_name)
        
        # t-SNE visualization
        self.visualize_tsne(features, labels, model_name)
        
        # UMAP visualization
        self.visualize_umap(features, labels, model_name)
        
        print(f"\n✓ All visualizations completed for {model_name}!")
        return True


def main():
    # Setup argument parser
    parser = create_parser()
    parser.add_argument('--model', '-m', default=None,
                       help='Specific model to visualize (resnet50, efficientnetb0, inceptionv3)')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Generate embeddings for all models')
    parser.add_argument('--list-models', '-l', action='store_true',
                       help='List available models')
    
    args = parser.parse_args()
    config = get_config(args)
    
    if args.list_models:
        print("\nAvailable models:")
        print("  - resnet50")
        print("  - efficientnetb0")
        print("  - inceptionv3")
        return
    
    extractor = FeatureExtractor(config)
    
    if args.all:
        for model_name in ['resnet50', 'efficientnetb0', 'inceptionv3']:
            extractor.generate_all_visualizations(model_name)
    elif args.model:
        extractor.generate_all_visualizations(args.model)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
