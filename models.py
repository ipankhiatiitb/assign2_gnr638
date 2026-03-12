"""
Model training functions for satellite image classification using PyTorch
Defines separate training functions for each model architecture
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import logging
from datetime import datetime

from config import (
    BATCH_SIZE, EPOCHS, INITIAL_LEARNING_RATE, FINETUNE_LEARNING_RATE,
    EARLY_STOPPING_PATIENCE, DATA_AUGMENTATION_CONFIG,
    TRAIN_DIR, VAL_DIR, CLASS_NAMES, NUM_CLASSES, RANDOM_SEED,
    SELECTED_TRAINING_MODE, TRAINING_MODES, RESULTS_DIR, DEVICE
)
from data_utils import get_data_loaders


# Set device from config
DEVICE = DEVICE


class InceptionV3Wrapper(nn.Module):
    """Wrapper for InceptionV3 to handle its special output structure"""
    
    def __init__(self, inception_model):
        super(InceptionV3Wrapper, self).__init__()
        self.inception = inception_model
        
    def forward(self, x):
        # InceptionV3 returns InceptionOutputs named tuple during training
        # We only need the main output, not the auxiliary outputs
        output = self.inception(x)
        if isinstance(output, tuple):
            return output[0]  # Return only the main output
        return output


class BaseModelTrainer:
    """Base class for model training with PyTorch"""
    
    def __init__(self, model_config, model_name, full_config=None, training_mode=None):
        """
        Initialize trainer
        
        Args:
            model_config (dict): Model configuration from config.py
            model_name (str): Name of the model for display
            full_config (dict): Full configuration dictionary with epochs and training parameters
            training_mode (str): Training mode being used
        """
        self.model_config = model_config
        self.config = full_config if full_config else model_config  # Use full_config if provided, else fallback
        self.model_name = model_name
        self.training_mode = training_mode or 'default'
        self.model = None
        self.device = DEVICE
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_acc = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.epoch_times = []  # Track time per epoch
        
        # Create dynamic results directory based on training parameters
        few_shot_num = self.config.get('few_shot_percentage', 100)
        few_shot_suffix = f"_{few_shot_num}" if few_shot_num != 100 else ""
        model_clean = model_name.lower().replace(" ", "")
        
        # Create results folder: results_{training_mode}_{model_name}_{few_shot_percentage}
        results_folder_name = f"results_{training_mode}_{model_clean}{few_shot_suffix}" if training_mode else f"results_{model_clean}"
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(project_root, results_folder_name)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Setup logging with training_log.txt naming
        self.log_file = os.path.join(self.results_dir, 'training_log.txt')
        self.logger = self._setup_logger()
        
        # Update model paths to use the dynamic results directory
        if training_mode:
            self.model_config['model_path'] = os.path.join(self.results_dir, 'final_model.pth')
            self.model_config['best_model_path'] = os.path.join(self.results_dir, 'best_model.pth')
            self.model_config['history_plot'] = os.path.join(self.results_dir, 'training_history.png')
            self.model_config['confusion_matrix_path'] = os.path.join(self.results_dir, 'confusion_matrix.png')
            self.model_config['classification_report_path'] = os.path.join(self.results_dir, 'classification_report.txt')
        
    def _setup_logger(self):
        """Setup logger to print and save to file"""
        logger = logging.getLogger(self.model_name)
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        logger.handlers = []
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    def build_model(self, base_model_fn, training_mode=None):
        """
        Build model with custom top layers
        
        Args:
            base_model_fn: Function to create base model
            training_mode (str): Training mode from config. If None, uses SELECTED_TRAINING_MODE
            
        Returns:
            Model: PyTorch model
        """
        if training_mode is None:
            training_mode = SELECTED_TRAINING_MODE
            
        # Load pre-trained model
        base_model = base_model_fn(pretrained=True)
        
        # Get input features for the classification layer
        if isinstance(base_model, models.ResNet):
            num_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
        elif isinstance(base_model, models.EfficientNet):
            num_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Identity()
        elif isinstance(base_model, models.Inception3):
            num_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
            # Disable auxiliary outputs during training to avoid issues
            base_model.aux_logits = False
            # Wrap InceptionV3 to handle its special output structure
            base_model = InceptionV3Wrapper(base_model)
        
        # Create classifier head - simple linear layer for linear probing
        head = nn.Linear(num_features, NUM_CLASSES)
        
        # Combine base model and head
        self.model = nn.Sequential(base_model, head).to(self.device)
        
        # Apply training mode-specific freezing strategy (do this before DataParallel)
        self._apply_training_mode(training_mode)
        
        # Apply multi-GPU if available (after training mode is applied)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            print(f"✓ Multi-GPU enabled: {torch.cuda.device_count()} GPUs available")
        
        # Setup optimizer
        lr = self.config.get('initial_learning_rate', INITIAL_LEARNING_RATE)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr
        )
        
        early_stop_patience = self.config.get('early_stopping_patience', EARLY_STOPPING_PATIENCE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=early_stop_patience,
            min_lr=1e-7
        )
        
        return self.model
    
    def _apply_training_mode(self, training_mode):
        """
        Apply training mode specific layer freezing/unfreezing strategy
        
        Args:
            training_mode (str): One of 'linear_probe', 'last_block_finetune', 'partial_finetune', 
                                 'selective_20percent_last', 'selective_20percent_random', 'full_finetune', 'two_stage'
        """
        if training_mode not in TRAINING_MODES:
            print(f"Warning: Unknown training mode '{training_mode}'. Using 'two_stage'")
            training_mode = 'two_stage'
        
        mode_config = TRAINING_MODES[training_mode]
        print(f"\nApplying training mode: {training_mode}")
        print(f"Description: {mode_config['description']}")
        
        base_model = self.model[0]
        
        if training_mode == 'linear_probe':
            # Freeze entire backbone, only train top layers
            for param in base_model.parameters():
                param.requires_grad = False
            print("✓ Backbone frozen. Only top layers trainable (linear probing)")
            
        elif training_mode == 'last_block_finetune':
            # Unfreeze last 3-5 layers (last residual block)
            self._unfreeze_layers(base_model, mode_config['num_layers_to_unfreeze'])
            
        elif training_mode == 'partial_finetune':
            # Unfreeze last num_layers_to_unfreeze layers
            self._unfreeze_layers(base_model, mode_config['num_layers_to_unfreeze'])
            
        elif training_mode == 'selective_20percent_last':
            # Unfreeze approximately 20% of total parameters from the last layers
            self._unfreeze_layers_by_percentage(base_model, percentage=0.20, strategy='last')
            
        elif training_mode == 'selective_20percent_random':
            # Unfreeze approximately 20% of total parameters randomly
            self._unfreeze_layers_by_percentage(base_model, percentage=0.20, strategy='random')
            
        elif training_mode == 'full_finetune':
            # Unfreeze entire model
            for param in base_model.parameters():
                param.requires_grad = True
            print("✓ All layers unfrozen for full fine-tuning")
            
        elif training_mode == 'two_stage':
            # Initially freeze, will be unfrozen in stage 2
            for param in base_model.parameters():
                param.requires_grad = False
            print("✓ Backbone initially frozen for two-stage training")
            print(f"  Stage 1: Linear probing ({mode_config['stage_1']['epochs']} epochs)")
            print(f"  Stage 2: Partial fine-tuning ({mode_config['stage_2']['epochs']} epochs)")
    
    def _unfreeze_layers(self, model, num_layers):
        """
        Unfreeze the last num_layers layers of the model
        
        Args:
            model: PyTorch model
            num_layers (int): Number of layers to unfreeze from the end
        """
        # Count trainable layers
        layers = list(model.children())
        
        # Freeze all initially
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze last num_layers
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        print(f"✓ Unfroze last {num_layers} layers for partial fine-tuning")
    
    def _unfreeze_layers_by_percentage(self, model, percentage=0.20, strategy='last'):
        """
        Unfreeze approximately a percentage of total parameters in the model
        
        Args:
            model: PyTorch model
            percentage (float): Percentage of parameters to unfreeze (0.0-1.0)
            strategy (str): 'last' to unfreeze from the end, 'random' to unfreeze randomly
        
        Reasoning for 20% selective unfreezing:
        - Balances between fine-tuning expressiveness and avoiding overfitting
        - Typically includes last residual block + partial earlier blocks
        - Allows adaptation to new task while maintaining learned features
        - Reduces training time and memory requirements
        """
        import random
        
        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        target_params = int(total_params * percentage)
        
        # Freeze all initially
        for param in model.parameters():
            param.requires_grad = False
        
        if strategy == 'last':
            # Unfreeze from the end until we reach target percentage
            unfrozen_params = 0
            layers = list(model.children())
            
            for layer in reversed(layers):
                for param in layer.parameters():
                    param.requires_grad = True
                    unfrozen_params += param.numel()
                
                if unfrozen_params >= target_params:
                    break
            
            actual_percentage = (unfrozen_params / total_params) * 100
            print(f"✓ Unfroze {actual_percentage:.1f}% of parameters ({unfrozen_params:,}/{total_params:,}) from LAST layers")
            print(f"  Strategy: Unfroze last layers sequentially until reaching ~20% of total parameters")
            
        elif strategy == 'random':
            # Randomly unfreeze parameters until reaching target percentage
            all_params = [(name, param) for name, param in model.named_parameters()]
            random.shuffle(all_params)
            
            unfrozen_params = 0
            unfrozen_count = 0
            
            for name, param in all_params:
                param.requires_grad = True
                unfrozen_params += param.numel()
                unfrozen_count += 1
                
                if unfrozen_params >= target_params:
                    break
            
            actual_percentage = (unfrozen_params / total_params) * 100
            print(f"✓ Unfroze {actual_percentage:.1f}% of parameters ({unfrozen_params:,}/{total_params:,}) RANDOMLY")
            print(f"  Strategy: Randomly selected {unfrozen_count} parameter groups for unfreezing")

    
    def train_epoch(self, train_loader):
        """
        Train one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            tuple: (loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': total_loss / (total / len(labels))})
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader):
        """
        Validate one epoch
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            tuple: (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc='Validating')
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix({'loss': total_loss / (total / len(labels))})
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, stage='initial'):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            stage: 'initial' or 'finetune'
        """
        header = f"\n{'='*70}\nTRAINING {self.model_name.upper()} - {stage.upper()}\n{'='*70}"
        print(header)
        self.logger.info(header)
        
        patience_counter = 0
        total_start_time = time.time()
        self.epoch_times = []
        
        epochs = self.config.get('epochs', EPOCHS)
        early_stop_patience = self.config.get('early_stopping_patience', EARLY_STOPPING_PATIENCE)
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            epoch_info = f"\nEpoch {epoch+1}/{epochs}"
            print(epoch_info)
            self.logger.info(epoch_info)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            self.epoch_times.append(epoch_time)
            
            metrics = (f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n"
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n"
                      f"Epoch Time: {epoch_time:.2f}s")
            print(metrics)
            self.logger.info(metrics)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(self.model_config['best_model_path'])
                patience_counter = 0
                best_msg = f"✓ Best model saved! Val Acc: {val_acc:.4f}"
                print(best_msg)
                self.logger.info(best_msg)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stop_patience:
                early_stop_msg = f"Early stopping at epoch {epoch+1}"
                print(early_stop_msg)
                self.logger.info(early_stop_msg)
                break
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        avg_epoch_time = np.mean(self.epoch_times) if self.epoch_times else 0
        
        summary = (f"\n{self.model_name} {stage} training completed!\n"
                  f"Total Training Time: {total_time:.2f}s ({total_time/60:.2f} minutes)\n"
                  f"Average Time per Epoch: {avg_epoch_time:.2f}s\n"
                  f"Total Epochs Completed: {len(self.epoch_times)}")
        print(summary)
        self.logger.info(summary)
    
    
    def fine_tune(self, train_loader, val_loader):
        """
        Fine-tune the model by unfreezing some base layers
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print("\n" + "="*70)
        print(f"FINE-TUNING {self.model_name.upper()}")
        print("="*70)
        
        # Unfreeze last num_layers_to_unfreeze layers
        base_model = self.model[0]
        total_layers = len(list(base_model.parameters()))
        unfreeze_from = total_layers - self.config['num_layers_to_unfreeze']
        
        layer_count = 0
        for param in base_model.parameters():
            if layer_count >= unfreeze_from:
                param.requires_grad = True
            layer_count += 1
        
        # Update optimizer with lower learning rate
        finetune_lr = self.config.get('finetune_learning_rate', FINETUNE_LEARNING_RATE)
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=finetune_lr
        )
        
        early_stop_patience = self.config.get('early_stopping_patience', EARLY_STOPPING_PATIENCE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=early_stop_patience - 1,
            min_lr=1e-8,
            verbose=True
        )
        
        print(f"Unfroze last {self.config['num_layers_to_unfreeze']} layers")
        
        # Reset early stopping counter for fine-tuning
        patience_counter = 0
        epochs = self.config.get('epochs', EPOCHS)
        early_stop_patience = self.config.get('early_stopping_patience', EARLY_STOPPING_PATIENCE)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(self.model_config['best_model_path'])
                patience_counter = 0
                print(f"Best model saved! Val Acc: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"\n{self.model_name} fine-tuning completed!")
    
    def plot_training_history(self):
        """Plot training and validation metrics"""
        if not self.train_losses:
            print(f"No training history found for {self.model_name}")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        axes[0].plot(self.train_accs, label='Training Accuracy', linewidth=2)
        axes[0].plot(self.val_accs, label='Validation Accuracy', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Accuracy', fontsize=11)
        axes[0].set_title(f'{self.model_name} - Accuracy', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(self.train_losses, label='Training Loss', linewidth=2)
        axes[1].plot(self.val_losses, label='Validation Loss', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Loss', fontsize=11)
        axes[1].set_title(f'{self.model_name} - Loss', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.model_config['history_plot'], dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Training history saved to {self.model_config['history_plot']}")
    
    def save_model(self, filepath=None):
        """Save the trained model"""
        if self.model is None:
            print(f"No model to save for {self.model_name}")
            return
        
        if filepath is None:
            filepath = self.model_config['model_path']
        
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        """Load a trained model"""
        if filepath is None:
            filepath = self.config['model_path']
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {filepath}")
    
    def get_model_summary(self):
        """Print model summary with FLOPs and MACs"""
        if self.model is None:
            self.logger.info(f"No model found for {self.model_name}")
            return
        
        self.logger.info("\n" + "="*70)
        self.logger.info(f"{self.model_name} - MODEL SUMMARY")
        self.logger.info("="*70)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary_msg = (f"Total Parameters: {total_params:,}\n"
                      f"Trainable Parameters: {trainable_params:,}\n"
                      f"Input Size: {self.model_config['input_size']}\n"
                      f"Number of Classes: {NUM_CLASSES}\n"
                      f"Device: {self.device}")
        self.logger.info(summary_msg)
        
        # Calculate FLOPs and MACs
        flops, macs = self._calculate_flops_macs()
        if flops is not None and macs is not None:
            flops_msg = (f"FLOPs: {flops:,.0f} ({flops/1e9:.2f}G)\n"
                        f"MACs: {macs:,.0f} ({macs/1e9:.2f}G)")
            self.logger.info(flops_msg)
        
        self.logger.info("="*70)
    
    def _calculate_flops_macs(self):
        """Calculate FLOPs and MACs for the model"""
        try:
            from fvcore.nn import FlopCounterMode
            
            # Create a sample input
            input_size = self.model_config['input_size']
            sample_input = torch.randn(1, 3, input_size[0], input_size[1]).to(self.device)
            
            # Calculate FLOPs
            with FlopCounterMode(self.model) as fmode:
                _ = self.model(sample_input)
            
            flops = fmode.flop_counts['']  # Total FLOPs
            macs = flops / 2  # Approximate MACs as FLOPs/2
            
            return flops, macs
        except ImportError:
            # If fvcore not available, try simpler calculation
            try:
                total_params = sum(p.numel() for p in self.model.parameters())
                input_size = self.model_config['input_size']
                macs = total_params * input_size[0] * input_size[1]  # Rough estimation
                flops = macs * 2
                return flops, macs
            except:
                return None, None
    

class ResNet50Trainer(BaseModelTrainer):
    """ResNet50 specific trainer"""
    
    def build_model(self, training_mode=None):
        """Build ResNet50 model
        
        Args:
            training_mode (str): Training mode to apply
        """
        print("\n" + "="*70)
        print("BUILDING RESNET50 MODEL")
        print("="*70)
        
        return super().build_model(models.resnet50, training_mode=training_mode)


class EfficientNetB0Trainer(BaseModelTrainer):
    """EfficientNet B0 specific trainer"""
    
    def build_model(self, training_mode=None):
        """Build EfficientNet B0 model
        
        Args:
            training_mode (str): Training mode to apply
        """
        print("\n" + "="*70)
        print("BUILDING EFFICIENTNET B0 MODEL")
        print("="*70)
        
        return super().build_model(models.efficientnet_b0, training_mode=training_mode)


class InceptionV3Trainer(BaseModelTrainer):
    """InceptionV3 specific trainer"""
    
    def build_model(self, training_mode=None):
        """Build InceptionV3 model
        
        Args:
            training_mode (str): Training mode to apply
        """
        print("\n" + "="*70)
        print("BUILDING INCEPTIONV3 MODEL")
        print("="*70)
        
        # Note: Inception3 expects input size of 299x299
        return super().build_model(models.inception_v3, training_mode=training_mode)


# ============================================================================
# TRAINING FUNCTIONS FOR EACH MODEL
# ============================================================================

def train_resnet50(model_config, training_mode=None, config=None):
    """
    Train ResNet50 model
    
    Args:
        model_config (dict): Model configuration from config.py
        training_mode (str): Training mode strategy (linear_probe, partial_finetune, full_finetune, two_stage)
        config (dict): Full configuration dictionary with training parameters
    """
    if training_mode is None:
        training_mode = SELECTED_TRAINING_MODE
    
    # Use provided config or fallback to module-level defaults
    training_modes = config['training_modes'] if config else TRAINING_MODES
    
    trainer = ResNet50Trainer(model_config, 'ResNet50', full_config=config, training_mode=training_mode)
    trainer.build_model(training_mode=training_mode)
    trainer.get_model_summary()
    
    few_shot_pct = config.get('few_shot_percentage', 100) if config else 100
    train_loader, val_loader, data_loading_time = get_data_loaders(
        model_config['input_size'],
        few_shot_percentage=few_shot_pct
    )
    
    # Log data loading time
    data_msg = f"Data Loading Time: {data_loading_time:.2f}s"
    print(data_msg)
    trainer.logger.info(data_msg)
    
    # Execute training based on selected mode
    if training_mode == 'linear_probe':
        # Only train top layers
        mode_config = training_modes['linear_probe']
        trainer.train(train_loader, val_loader, stage='linear_probe')
        
    elif training_mode == 'last_block_finetune':
        # Last block fine-tuning
        mode_config = training_modes['last_block_finetune']
        trainer.train(train_loader, val_loader, stage='last_block_finetune')
        
    elif training_mode == 'partial_finetune':
        # Train with partial fine-tuning
        mode_config = training_modes['partial_finetune']
        trainer.train(train_loader, val_loader, stage='partial_finetune')
        
    elif training_mode == 'selective_20percent_last':
        # Selective 20% fine-tuning (last layers)
        mode_config = training_modes['selective_20percent_last']
        trainer.train(train_loader, val_loader, stage='selective_20percent_last')
        
    elif training_mode == 'selective_20percent_random':
        # Selective 20% fine-tuning (random)
        mode_config = training_modes['selective_20percent_random']
        trainer.train(train_loader, val_loader, stage='selective_20percent_random')
        
    elif training_mode == 'full_finetune':
        # Full fine-tuning
        mode_config = training_modes['full_finetune']
        trainer.train(train_loader, val_loader, stage='full_finetune')
        
    elif training_mode == 'two_stage':
        # Stage 1: Linear probing (frozen backbone)
        mode_config = training_modes['two_stage']
        trainer.train(train_loader, val_loader, stage='stage_1_linear_probe')
        
        # Stage 2: Partial fine-tuning
        trainer._apply_training_mode('partial_finetune')
        trainer.optimizer = optim.Adam(
            trainer.model.parameters(),
            lr=training_modes['two_stage']['stage_2']['learning_rate']
        )
        trainer.best_val_acc = 0  # Reset for stage 2
        trainer.train(train_loader, val_loader, stage='stage_2_finetune')
    
    # Save and visualize
    trainer.plot_training_history()
    trainer.save_model()


def train_efficientnetb0(model_config, training_mode=None, config=None):
    """
    Train EfficientNet B0 model
    
    Args:
        model_config (dict): Model configuration from config.py
        training_mode (str): Training mode strategy (linear_probe, partial_finetune, full_finetune, two_stage)
        config (dict): Full configuration dictionary with training parameters
    """
    if training_mode is None:
        training_mode = SELECTED_TRAINING_MODE
    
    # Use provided config or fallback to module-level defaults
    training_modes = config['training_modes'] if config else TRAINING_MODES
    
    trainer = EfficientNetB0Trainer(model_config, 'EfficientNet B0', full_config=config, training_mode=training_mode)
    trainer.build_model(training_mode=training_mode)
    trainer.get_model_summary()
    
    few_shot_pct = config.get('few_shot_percentage', 100) if config else 100
    train_loader, val_loader, data_loading_time = get_data_loaders(
        model_config['input_size'],
        few_shot_percentage=few_shot_pct
    )
    
    # Log data loading time
    data_msg = f"Data Loading Time: {data_loading_time:.2f}s"
    print(data_msg)
    trainer.logger.info(data_msg)
    
    # Execute training based on selected mode
    if training_mode == 'linear_probe':
        trainer.train(train_loader, val_loader, stage='linear_probe')
        
    elif training_mode == 'last_block_finetune':
        trainer.train(train_loader, val_loader, stage='last_block_finetune')
        
    elif training_mode == 'partial_finetune':
        trainer.train(train_loader, val_loader, stage='partial_finetune')
        
    elif training_mode == 'selective_20percent_last':
        trainer.train(train_loader, val_loader, stage='selective_20percent_last')
        
    elif training_mode == 'selective_20percent_random':
        trainer.train(train_loader, val_loader, stage='selective_20percent_random')
        
    elif training_mode == 'full_finetune':
        trainer.train(train_loader, val_loader, stage='full_finetune')
        
    elif training_mode == 'two_stage':
        # Stage 1: Linear probing
        trainer.train(train_loader, val_loader, stage='stage_1_linear_probe')
        
        # Stage 2: Partial fine-tuning
        trainer._apply_training_mode('partial_finetune')
        trainer.optimizer = optim.Adam(
            trainer.model.parameters(),
            lr=training_modes['two_stage']['stage_2']['learning_rate']
        )
        trainer.best_val_acc = 0
        trainer.train(train_loader, val_loader, stage='stage_2_finetune')
    
    # Save and visualize
    trainer.plot_training_history()
    trainer.save_model()


def train_inceptionv3(model_config, training_mode=None, config=None):
    """
    Train InceptionV3 model
    
    Args:
        model_config (dict): Model configuration from config.py
        training_mode (str): Training mode strategy (linear_probe, partial_finetune, full_finetune, two_stage)
        config (dict): Full configuration dictionary with training parameters
    """
    if training_mode is None:
        training_mode = SELECTED_TRAINING_MODE
    
    # Use provided config or fallback to module-level defaults
    training_modes = config['training_modes'] if config else TRAINING_MODES
    
    trainer = InceptionV3Trainer(model_config, 'InceptionV3', full_config=config, training_mode=training_mode)
    trainer.build_model(training_mode=training_mode)
    trainer.get_model_summary()
    
    few_shot_pct = config.get('few_shot_percentage', 100) if config else 100
    train_loader, val_loader, data_loading_time = get_data_loaders(
        model_config['input_size'],
        few_shot_percentage=few_shot_pct
    )
    
    # Log data loading time
    data_msg = f"Data Loading Time: {data_loading_time:.2f}s"
    print(data_msg)
    trainer.logger.info(data_msg)
    
    # Execute training based on selected mode
    if training_mode == 'linear_probe':
        trainer.train(train_loader, val_loader, stage='linear_probe')
        
    elif training_mode == 'last_block_finetune':
        trainer.train(train_loader, val_loader, stage='last_block_finetune')
        
    elif training_mode == 'partial_finetune':
        trainer.train(train_loader, val_loader, stage='partial_finetune')
        
    elif training_mode == 'selective_20percent_last':
        trainer.train(train_loader, val_loader, stage='selective_20percent_last')
        
    elif training_mode == 'selective_20percent_random':
        trainer.train(train_loader, val_loader, stage='selective_20percent_random')
        
    elif training_mode == 'full_finetune':
        trainer.train(train_loader, val_loader, stage='full_finetune')
        
    elif training_mode == 'two_stage':
        # Stage 1: Linear probing
        trainer.train(train_loader, val_loader, stage='stage_1_linear_probe')
        
        # Stage 2: Partial fine-tuning
        trainer._apply_training_mode('partial_finetune')
        trainer.optimizer = optim.Adam(
            trainer.model.parameters(),
            lr=training_modes['two_stage']['stage_2']['learning_rate']
        )
        trainer.best_val_acc = 0
        trainer.train(train_loader, val_loader, stage='stage_2_finetune')
    
    # Save and visualize
    trainer.plot_training_history()
    trainer.save_model()


# ============================================================================
# MODEL TRAINING REGISTRY
# ============================================================================

TRAINING_FUNCTIONS = {
    'resnet50': train_resnet50,
    'efficientnetb0': train_efficientnetb0,
    'inceptionv3': train_inceptionv3,
}


def get_training_function(model_name):
    """
    Get training function for a model
    
    Args:
        model_name (str): Model name
        
    Returns:
        function: Training function
        
    Raises:
        ValueError: If model not found
    """
    if model_name.lower() not in TRAINING_FUNCTIONS:
        raise ValueError(
            f"Training function for '{model_name}' not found. "
            f"Available models: {list(TRAINING_FUNCTIONS.keys())}"
        )
    return TRAINING_FUNCTIONS[model_name.lower()]


if __name__ == "__main__":
    print("Available training functions:")
    for model_name in TRAINING_FUNCTIONS.keys():
        print(f"  - {model_name}")
