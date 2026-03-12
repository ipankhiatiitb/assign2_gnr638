"""
Layer-Wise Feature Probing Module (Assignment 4.5)

Examines semantic abstraction evolution across network depth by:
1. Extracting intermediate representations from early, middle, and final layers
2. Training separate linear classifiers on features from each depth
3. Analyzing how representation quality changes with layer depth
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import logging
from tqdm import tqdm

from config import NUM_CLASSES, RESULTS_DIR


class LayerExtractor(nn.Module):
    """Hook-based feature extractor for intermediate layers"""
    
    def __init__(self, model, layer_names):
        """
        Initialize layer extractor
        
        Args:
            model: PyTorch model
            layer_names (list): List of layer names to extract features from
        """
        super().__init__()
        self.model = model
        self.layer_names = layer_names
        self.features = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on specified layers"""
        def get_hook(name):
            def hook(module, input, output):
                # Flatten output if needed
                if isinstance(output, torch.Tensor):
                    if output.dim() > 2:
                        self.features[name] = output.detach().cpu()
                    else:
                        self.features[name] = output.detach().cpu()
            return hook
        
        for name, module in self.model.named_modules():
            if any(layer_name in name for layer_name in self.layer_names):
                hook = module.register_forward_hook(get_hook(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
    
    def forward(self, x):
        """Forward pass through model"""
        return self.model(x)
    
    def get_features(self):
        """Get extracted features"""
        return self.features


class LayerWiseProber:
    """Train linear classifiers on features from different layers"""
    
    def __init__(self, model, model_name, train_loader, val_loader, device='cuda'):
        """
        Initialize layer-wise prober
        
        Args:
            model: Pre-trained model
            model_name (str): Model name
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Computation device
        """
        self.model = model
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.results = {}
        
        # Create dynamic results directory: results_layer_wise_probing_{model_name}
        model_clean = model_name.lower().replace(" ", "")
        results_folder_name = f"results_layer_wise_probing_{model_clean}"
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(project_root, results_folder_name)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Setup logging
        self.log_file = os.path.join(self.results_dir, 'training_log.txt')
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger"""
        logger = logging.getLogger(f'{self.model_name}_LayerWiseProbing')
        logger.setLevel(logging.INFO)
        logger.handlers = []
        
        file_handler = logging.FileHandler(self.log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def select_layers(self):
        """
        Select early, middle, and final layers based on model architecture
        
        Returns:
            dict: Selected layers for each depth category
        """
        layers = dict(self.model.named_modules())
        all_layer_names = list(layers.keys())
        
        # Filter out non-relevant layers
        relevant_layers = [name for name in all_layer_names if any(
            keyword in name for keyword in ['layer', 'block', 'conv', 'stage']
        )]
        
        n_layers = len(relevant_layers)
        
        selected = {
            'early': relevant_layers[n_layers // 4] if n_layers > 0 else relevant_layers[0],
            'middle': relevant_layers[n_layers // 2] if n_layers > 0 else relevant_layers[0],
            'final': relevant_layers[-1] if n_layers > 0 else relevant_layers[0],
        }
        
        self.logger.info(f"\nSelected Layers:")
        for depth, layer_name in selected.items():
            self.logger.info(f"  {depth:8s}: {layer_name}")
        
        return selected
    
    def extract_layer_features(self, layer_names):
        """
        Extract features from specified layers
        
        Args:
            layer_names (list): List of layer names to extract
            
        Returns:
            dict: Features for each layer and dataset split
        """
        self.logger.info(f"\n[Extracting Features from {len(layer_names)} layers]")
        
        extractor = LayerExtractor(self.model, layer_names)
        self.model.eval()
        
        all_features = {'train': {}, 'val': {}}
        all_labels = {'train': [], 'val': []}
        
        with torch.no_grad():
            # Extract training features
            self.logger.info("\n  Extracting training features...")
            for batch_idx, (images, labels) in enumerate(tqdm(self.train_loader)):
                images = images.to(self.device)
                extractor(images)
                
                # Store first batch features
                if batch_idx == 0:
                    for layer_name, features in extractor.features.items():
                        all_features['train'][layer_name] = features
                all_labels['train'].extend(labels.cpu().numpy())
            
            # Extract validation features
            self.logger.info("  Extracting validation features...")
            for batch_idx, (images, labels) in enumerate(tqdm(self.val_loader)):
                images = images.to(self.device)
                extractor(images)
                
                # Store first batch features
                if batch_idx == 0:
                    for layer_name, features in extractor.features.items():
                        all_features['val'][layer_name] = features
                all_labels['val'].extend(labels.cpu().numpy())
        
        extractor.remove_hooks()
        
        return all_features, all_labels
    
    def probe_layer(self, features_train, labels_train, features_val, labels_val):
        """
        Train linear classifier on layer features
        
        Args:
            features_train: Training features
            labels_train: Training labels
            features_val: Validation features
            labels_val: Validation labels
            
        Returns:
            tuple: (train_accuracy, val_accuracy, feature_norms)
        """
        # Flatten features
        if features_train.dim() > 2:
            features_train = features_train.view(features_train.size(0), -1)
        if features_val.dim() > 2:
            features_val = features_val.view(features_val.size(0), -1)
        
        input_dim = features_train.shape[1]
        
        # Create linear classifier
        classifier = nn.Linear(input_dim, NUM_CLASSES).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
        
        # Train for 20 epochs
        for epoch in range(20):
            classifier.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for i in range(0, len(features_train), 32):
                batch_features = features_train[i:i+32].to(self.device)
                batch_labels = torch.tensor(labels_train[i:i+32]).to(self.device)
                
                optimizer.zero_grad()
                outputs = classifier(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_labels).sum().item()
                total += batch_labels.size(0)
        
        # Evaluate
        classifier.eval()
        with torch.no_grad():
            # Training accuracy
            outputs = classifier(features_train.to(self.device))
            _, predicted = torch.max(outputs, 1)
            train_acc = (predicted.cpu().numpy() == labels_train).mean()
            
            # Validation accuracy
            outputs = classifier(features_val.to(self.device))
            _, predicted = torch.max(outputs, 1)
            val_acc = (predicted.cpu().numpy() == labels_val).mean()
        
        # Calculate feature norms
        feature_norm = torch.norm(features_train).item() / features_train.shape[0]
        
        return train_acc, val_acc, feature_norm
    
    def probe_all_layers(self):
        """
        Probe all selected layers
        
        Returns:
            dict: Results for each layer depth
        """
        self.logger.info("\n" + "="*70)
        self.logger.info(f"LAYER-WISE PROBING: {self.model_name}")
        self.logger.info("="*70)
        
        # Select layers
        selected_layers = self.select_layers()
        layer_names = list(selected_layers.values())
        
        # Extract features
        features, labels = self.extract_layer_features(layer_names)
        
        # Probe each layer
        self.logger.info(f"\n[Probing Layers]")
        for depth, layer_name in selected_layers.items():
            self.logger.info(f"\n  Probing {depth:8s} layer: {layer_name}")
            
            train_acc, val_acc, feat_norm = self.probe_layer(
                features['train'][layer_name],
                labels['train'],
                features['val'][layer_name],
                labels['val']
            )
            
            self.results[depth] = {
                'layer': layer_name,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'feature_norm': feat_norm,
            }
            
            self.logger.info(f"    Train Accuracy: {train_acc:.4f}")
            self.logger.info(f"    Val Accuracy:   {val_acc:.4f}")
            self.logger.info(f"    Feature Norm:   {feat_norm:.4f}")
        
        return self.results
    
    def plot_results(self):
        """Plot layer-wise probing results"""
        if not self.results:
            return
        
        depths = list(self.results.keys())
        train_accs = [self.results[d]['train_acc'] for d in depths]
        val_accs = [self.results[d]['val_acc'] for d in depths]
        feat_norms = [self.results[d]['feature_norm'] for d in depths]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy vs depth
        x = np.arange(len(depths))
        axes[0].plot(x, train_accs, 'o-', label='Train', markersize=8)
        axes[0].plot(x, val_accs, 's-', label='Val', markersize=8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(depths)
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title(f'{self.model_name}: Accuracy vs Network Depth')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Feature norms vs depth
        axes[1].bar(x, feat_norms, color='steelblue', alpha=0.7)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(depths)
        axes[1].set_ylabel('Feature Norm (per sample)')
        axes[1].set_title(f'{self.model_name}: Feature Norms vs Depth')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"\n✓ Plot saved: {plot_path}")
        plt.close()


def run_layer_wise_probing(model, model_name, train_loader, val_loader, device='cuda'):
    """
    Run complete layer-wise probing analysis
    
    Args:
        model: Pre-trained model
        model_name (str): Model name
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Computation device
    """
    prober = LayerWiseProber(model, model_name, train_loader, val_loader, device)
    results = prober.probe_all_layers()
    prober.plot_results()
    return results
