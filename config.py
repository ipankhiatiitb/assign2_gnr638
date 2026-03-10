"""
Configuration file for Satellite Image Classification Assignment
Defines all parameters via command-line arguments with defaults
"""

import os
import argparse
import torch
from pathlib import Path

# ============================================================================
# ARGUMENT PARSER SETUP
# ============================================================================

def create_parser():
    """Create and return argument parser with all configuration options"""
    parser = argparse.ArgumentParser(
        description='Satellite Image Classification - Configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --model resnet50 --training-mode linear_probe
  python train.py --model efficientnetb0 --training-mode two_stage --epochs 50
  python train.py --all --batch-size 64 --learning-rate 0.0001
  python train.py --model resnet50 --device cuda:1
        """
    )
    
    # ========== Data Configuration ==========
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    
    # ========== Training Configuration ==========
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of epochs for training (default: 30)')
    parser.add_argument('--learning-rate', '--lr', type=float, default=0.001,
                       help='Initial learning rate (default: 0.001)')
    parser.add_argument('--finetune-lr', type=float, default=0.0001,
                       help='Fine-tuning learning rate (default: 0.0001)')
    parser.add_argument('--early-stopping-patience', type=int, default=5,
                       help='Early stopping patience (default: 5)')
    parser.add_argument('--reduce-lr-patience', type=int, default=3,
                       help='Learning rate reduction patience (default: 3)')
    parser.add_argument('--reduce-lr-factor', type=float, default=0.5,
                       help='Learning rate reduction factor (default: 0.5)')
    parser.add_argument('--min-lr', type=float, default=1e-7,
                       help='Minimum learning rate (default: 1e-7)')
    
    # ========== Model Configuration ==========
    parser.add_argument('--dropout-rate', type=float, default=0.3,
                       help='Dropout rate for all models (default: 0.3)')
    parser.add_argument('--num-layers-unfreeze', type=int, default=50,
                       help='Number of layers to unfreeze during fine-tuning (default: 50)')
    
    # ========== Training Mode Configuration ==========
    parser.add_argument('--training-mode', '-t', 
                       choices=['linear_probe', 'partial_finetune', 'full_finetune', 'two_stage'],
                       default='two_stage',
                       help='Training mode strategy (default: two_stage)')
    
    # ========== Few-Shot Learning ==========
    parser.add_argument('--few-shot', action='store_true',
                       help='Enable few-shot learning')
    parser.add_argument('--samples-per-class', type=int, default=10,
                       help='Number of samples per class for few-shot learning (default: 10)')
    
    # ========== Device Configuration ==========
    parser.add_argument('--device', '-d', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda, cpu, cuda:0, cuda:1, etc.) (default: cuda if available, else cpu)')
    
    # ========== Data Augmentation ==========
    parser.add_argument('--rotation-range', type=int, default=20,
                       help='Rotation range for augmentation (default: 20)')
    parser.add_argument('--zoom-range', type=float, default=0.2,
                       help='Zoom range for augmentation (default: 0.2)')
    parser.add_argument('--shift-range', type=float, default=0.2,
                       help='Width/height shift range for augmentation (default: 0.2)')
    parser.add_argument('--shear-range', type=float, default=0.2,
                       help='Shear range for augmentation (default: 0.2)')
    
    # ========== Display Options ==========
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (default: 1)')
    parser.add_argument('--print-interval', type=int, default=50,
                       help='Print interval for training logs (default: 50)')
    parser.add_argument('--plot-history', action='store_true', default=True,
                       help='Plot training history (default: True)')
    parser.add_argument('--no-plot', action='store_false', dest='plot_history',
                       help='Disable training history plot')
    parser.add_argument('--save-confusion-matrices', action='store_true', default=True,
                       help='Save confusion matrices (default: True)')
    parser.add_argument('--generate-reports', action='store_true', default=True,
                       help='Generate classification reports (default: True)')
    
    return parser


def get_config(args=None):
    """
    Parse arguments and return configuration dictionary
    
    Args:
        args: Parsed arguments (if None, parses from command line)
        
    Returns:
        dict: Configuration dictionary with all settings
    """
    if args is None:
        parser = create_parser()
        args = parser.parse_args()
    
    # ============================================================================
    # DEVICE SETUP
    # ============================================================================
    try:
        device = torch.device(args.device)
    except:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cuda_available = device.type == 'cuda'
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(device)
        gpu_count = torch.cuda.device_count()
    else:
        gpu_name = "None"
        gpu_count = 0
    
    # ============================================================================
    # PATHS SETUP
    # ============================================================================
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_root, 'data', 'split_data')
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')
    test_dir = os.path.join(data_path, 'test')
    
    # ============================================================================
    # CLASS NAMES
    # ============================================================================
    class_names = [
        'Airport', 'BareLand', 'BaseballField', 'Beach', 'Bridge', 'Center',
        'Church', 'Commercial', 'DenseResidential', 'Desert', 'Farmland',
        'Forest', 'Industrial', 'Meadow', 'MediumResidential', 'Mountain',
        'Park', 'Parking', 'Playground', 'Pond', 'Port', 'RailwayStation',
        'Resort', 'River', 'School', 'SparseResidential', 'Square', 'Stadium',
        'StorageTanks', 'Viaduct'
    ]
    num_classes = len(class_names)
    
    # ============================================================================
    # MODEL CONFIGURATIONS
    # ============================================================================
    models_config = {
        'resnet50': {
            'name': 'ResNet50',
            'input_size': (224, 224),
            'dropout_rate': args.dropout_rate,
            'hidden_units': [512, 256],
            'model_path': os.path.join(results_dir, 'resnet50_final_model.pth'),
            'best_model_path': os.path.join(results_dir, 'resnet50_best_model.pth'),
            'history_plot': os.path.join(results_dir, 'resnet50_training_history.png'),
            'confusion_matrix_path': os.path.join(results_dir, 'resnet50_confusion_matrix.png'),
            'classification_report_path': os.path.join(results_dir, 'resnet50_classification_report.txt'),
        },
        'efficientnetb0': {
            'name': 'EfficientNet B0',
            'input_size': (224, 224),
            'dropout_rate': args.dropout_rate,
            'hidden_units': [512, 256],
            'model_path': os.path.join(results_dir, 'efficientnetb0_final_model.pth'),
            'best_model_path': os.path.join(results_dir, 'efficientnetb0_best_model.pth'),
            'history_plot': os.path.join(results_dir, 'efficientnetb0_training_history.png'),
            'confusion_matrix_path': os.path.join(results_dir, 'efficientnetb0_confusion_matrix.png'),
            'classification_report_path': os.path.join(results_dir, 'efficientnetb0_classification_report.txt'),
        },
        'inceptionv3': {
            'name': 'InceptionV3',
            'input_size': (299, 299),
            'dropout_rate': args.dropout_rate,
            'hidden_units': [512, 256],
            'model_path': os.path.join(results_dir, 'inceptionv3_final_model.pth'),
            'best_model_path': os.path.join(results_dir, 'inceptionv3_best_model.pth'),
            'history_plot': os.path.join(results_dir, 'inceptionv3_training_history.png'),
            'confusion_matrix_path': os.path.join(results_dir, 'inceptionv3_confusion_matrix.png'),
            'classification_report_path': os.path.join(results_dir, 'inceptionv3_classification_report.txt'),
        }
    }
    
    # ============================================================================
    # TRAINING MODES CONFIGURATION
    # ============================================================================
    training_modes = {
        'linear_probe': {
            'description': 'Linear Probing - Train only top classification layer',
            'freeze_backbone': True,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
        },
        'partial_finetune': {
            'description': 'Partial Fine-tuning - Train last N layers + head',
            'freeze_backbone': False,
            'num_layers_to_unfreeze': args.num_layers_unfreeze,
            'learning_rate': args.finetune_lr,
            'epochs': args.epochs,
        },
        'full_finetune': {
            'description': 'Full Fine-tuning - Train all layers',
            'freeze_backbone': False,
            'num_layers_to_unfreeze': None,
            'learning_rate': args.finetune_lr,
            'epochs': args.epochs,
        },
        'two_stage': {
            'description': 'Two-Stage - Linear probe then fine-tune',
            'stage_1': {
                'freeze_backbone': True,
                'learning_rate': args.learning_rate,
                'epochs': args.epochs // 2 if args.epochs > 0 else 30,
            },
            'stage_2': {
                'freeze_backbone': False,
                'num_layers_to_unfreeze': args.num_layers_unfreeze,
                'learning_rate': args.finetune_lr,
                'epochs': args.epochs - (args.epochs // 2) if args.epochs > 0 else 30,
            }
        }
    }
    
    # ============================================================================
    # DATA AUGMENTATION CONFIGURATION
    # ============================================================================
    data_augmentation_config = {
        'rotation_range': args.rotation_range,
        'width_shift_range': args.shift_range,
        'height_shift_range': args.shift_range,
        'shear_range': args.shear_range,
        'zoom_range': args.zoom_range,
        'horizontal_flip': True,
        'vertical_flip': True,
        'fill_mode': 'nearest',
        'rescale': 1./255
    }
    
    # ============================================================================
    # FEW-SHOT LEARNING CONFIGURATION
    # ============================================================================
    few_shot_config = {
        'enabled': args.few_shot,
        'samples_per_class': args.samples_per_class,
    }
    
    # ============================================================================
    # RETURN CONFIGURATION DICTIONARY
    # ============================================================================
    config = {
        # Device
        'device': device,
        'cuda_available': cuda_available,
        'gpu_name': gpu_name,
        'gpu_count': gpu_count,
        
        # Paths
        'project_root': project_root,
        'data_path': data_path,
        'results_dir': results_dir,
        'train_dir': train_dir,
        'val_dir': val_dir,
        'test_dir': test_dir,
        
        # Data
        'batch_size': args.batch_size,
        'num_classes': num_classes,
        'class_names': class_names,
        'random_seed': args.random_seed,
        'num_workers': args.num_workers,
        
        # Training
        'epochs': args.epochs,
        'initial_learning_rate': args.learning_rate,
        'finetune_learning_rate': args.finetune_lr,
        'early_stopping_patience': args.early_stopping_patience,
        'reduce_lr_patience': args.reduce_lr_patience,
        'reduce_lr_factor': args.reduce_lr_factor,
        'min_learning_rate': args.min_lr,
        
        # Models
        'models_config': models_config,
        'dropout_rate': args.dropout_rate,
        'num_layers_to_unfreeze': args.num_layers_unfreeze,
        
        # Training modes
        'training_modes': training_modes,
        'selected_training_mode': args.training_mode,
        
        # Data augmentation
        'data_augmentation_config': data_augmentation_config,
        
        # Few-shot learning
        'few_shot_config': few_shot_config,
        
        # Display
        'verbose': args.verbose,
        'print_interval': args.print_interval,
        'plot_history': args.plot_history,
        'save_confusion_matrices': args.save_confusion_matrices,
        'generate_classification_reports': args.generate_reports,
    }
    
    return config


def print_config(config):
    """Print configuration summary"""
    print("=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"\nProject Root: {config['project_root']}")
    print(f"Data Path: {config['data_path']}")
    print(f"Results Directory: {config['results_dir']}")
    print(f"\nDevice Configuration:")
    print(f"  Device: {config['device']}")
    print(f"  CUDA Available: {config['cuda_available']}")
    if config['cuda_available']:
        print(f"  GPU Name: {config['gpu_name']}")
        print(f"  GPU Count: {config['gpu_count']}")
    print(f"\nTraining Configuration:")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Learning Rate: {config['initial_learning_rate']}")
    print(f"  Fine-tune LR: {config['finetune_learning_rate']}")
    print(f"  Random Seed: {config['random_seed']}")
    print(f"\nModel Configuration:")
    print(f"  Number of Classes: {config['num_classes']}")
    print(f"  Dropout Rate: {config['dropout_rate']}")
    print(f"  Layers to Unfreeze: {config['num_layers_to_unfreeze']}")
    print(f"\nSelected Training Mode: {config['selected_training_mode']}")
    print(f"  Description: {config['training_modes'][config['selected_training_mode']]['description']}")
    print(f"\nFew-Shot Learning: {'Enabled' if config['few_shot_config']['enabled'] else 'Disabled'}")
    if config['few_shot_config']['enabled']:
        print(f"  Samples per class: {config['few_shot_config']['samples_per_class']}")
    print(f"\nAvailable Models:")
    for model_name in config['models_config'].keys():
        model_info = config['models_config'][model_name]
        print(f"  - {model_name}: Input size {model_info['input_size']}")
    print("=" * 70)


def get_model_config(model_name_or_config, model_name=None):
    """
    Get configuration for a specific model
    
    Args:
        model_name_or_config: Either a model name string, or a config dict
        model_name (str): Model name (if first arg is config dict)
        
    Returns:
        dict: Model configuration
    """
    # Handle both old style (get_model_config('resnet50')) and new style (get_model_config(config, 'resnet50'))
    if isinstance(model_name_or_config, dict):
        # New style: first arg is config dict
        config = model_name_or_config
        if model_name is None:
            raise ValueError("model_name required when first argument is a config dict")
        models = config.get('models_config', MODELS_CONFIG)
    else:
        # Old style: first arg is model name
        model_name = model_name_or_config
        models = MODELS_CONFIG
    
    if model_name.lower() not in models:
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {list(models.keys())}"
        )
    return models[model_name.lower()]


def get_available_models(config=None):
    """
    Get list of available models
    
    Args:
        config (dict, optional): Configuration dict. If None, uses default MODELS_CONFIG
        
    Returns:
        list: List of available model names
    """
    if config is None:
        return list(MODELS_CONFIG.keys())
    return list(config.get('models_config', MODELS_CONFIG).keys())


# ============================================================================
# MODULE-LEVEL VARIABLES FOR BACKWARD COMPATIBILITY
# ============================================================================
# Create default config when module is imported
_default_parser = create_parser()
_default_args = _default_parser.parse_args([])
_default_config = get_config(_default_args)

# Export commonly used variables
DEVICE = _default_config['device']
CUDA_AVAILABLE = _default_config['cuda_available']
GPU_NAME = _default_config['gpu_name']
GPU_COUNT = _default_config['gpu_count']
PROJECT_ROOT = _default_config['project_root']
DATA_PATH = _default_config['data_path']
RESULTS_DIR = _default_config['results_dir']
TRAIN_DIR = _default_config['train_dir']
VAL_DIR = _default_config['val_dir']
TEST_DIR = _default_config['test_dir']
BATCH_SIZE = _default_config['batch_size']
NUM_CLASSES = _default_config['num_classes']
CLASS_NAMES = _default_config['class_names']
RANDOM_SEED = _default_config['random_seed']
EPOCHS = _default_config['epochs']
INITIAL_LEARNING_RATE = _default_config['initial_learning_rate']
FINETUNE_LEARNING_RATE = _default_config['finetune_learning_rate']
EARLY_STOPPING_PATIENCE = _default_config['early_stopping_patience']
MODELS_CONFIG = _default_config['models_config']
TRAINING_MODES = _default_config['training_modes']
SELECTED_TRAINING_MODE = _default_config['selected_training_mode']
DATA_AUGMENTATION_CONFIG = _default_config['data_augmentation_config']
FEW_SHOT_CONFIG = _default_config['few_shot_config']
VERBOSE = _default_config['verbose']
PRINT_INTERVAL = _default_config['print_interval']
PLOT_HISTORY = _default_config['plot_history']
SAVE_CONFUSION_MATRICES = _default_config['save_confusion_matrices']
GENERATE_CLASSIFICATION_REPORTS = _default_config['generate_classification_reports']


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    config = get_config(args)
    print_config(config)
