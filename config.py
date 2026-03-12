"""
Configuration file for GNR 638 Assignment 2: Transfer Learning & Robustness Analysis
Defines all parameters via command-line arguments with assignment constraints
Reference: GNR_638_Assignment_2.pdf

ASSIGNMENT CONSTRAINTS:
- Dataset: Aerial Images Dataset (AID) - 30 classes
- Models: ResNet50, InceptionV3, EfficientNet-B0 (all pre-trained on ImageNet)
- Max epochs: 30 for full-data, 20 for few-shot
- Max 6 hours per model per scenario
- Learning rates: 0.01 for linear_probe, 0.0001 for fine-tuning
- Reproducibility: Fixed random seed for all experiments
"""

import os
import argparse
import torch
from pathlib import Path

# ============================================================================
# ASSIGNMENT CONSTRAINT CONSTANTS
# ============================================================================
MAX_EPOCHS_FULL_DATA = 30  # Assignment constraint 6
MAX_EPOCHS_FEW_SHOT = 20   # Assignment constraint 6
MAX_HOURS_PER_SCENARIO = 6  # Assignment constraint 6
NUM_CLASSES = 30  # AID dataset
RANDOM_SEED = 42  # For reproducibility

# ============================================================================
# ARGUMENT PARSER SETUP
# ============================================================================

def create_parser():
    """Create and return argument parser with all configuration options"""
    parser = argparse.ArgumentParser(
        description='GNR 638 Assignment 2: Pre-trained CNN Transfer Learning & Robustness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --model resnet50 --training-mode linear_probe
  python train.py --model efficientnetb0 --training-mode selective_20percent_last
  python train.py --model resnet50 --training-mode two_stage --few-shot-percentage 5
  python train.py --model inceptionv3 --training-mode full_finetune --few-shot-percentage 100
        """
    )
    
    # ========== Data Configuration ==========
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--random-seed', type=int, default=RANDOM_SEED,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    
    # ========== Few-Shot Configuration (Assignment 4.3) ==========
    parser.add_argument('--few-shot-percentage', type=int, choices=[5, 20, 100], default=100,
                       help='Percentage of training data to use: 5%% (extreme), 20%% (low), 100%% (full). Default: 100')
    
    # ========== Training Configuration ==========
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS_FULL_DATA,
                       help=f'Number of epochs for training (max: {MAX_EPOCHS_FULL_DATA} for full-data, {MAX_EPOCHS_FEW_SHOT} for few-shot)')
    parser.add_argument('--learning-rate', '--lr', type=float, default=0.01,
                       help='Learning rate for linear_probe (default: 0.01, per assignment)')
    parser.add_argument('--finetune-lr', type=float, default=0.0001,
                       help='Learning rate for fine-tuning modes (default: 0.0001, per assignment)')
    parser.add_argument('--early-stopping-patience', type=int, default=1000,
                       help='Early stopping patience (default: 1000 - disabled)')
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
                       help='Number of layers to unfreeze during partial_finetune (default: 50)')
    
    # ========== Training Mode Configuration (Assignment 4.2) ==========
    parser.add_argument('--training-mode', '-t', 
                       choices=['linear_probe', 'last_block_finetune', 'partial_finetune', 
                               'selective_20percent_last', 'selective_20percent_random', 'full_finetune', 'two_stage'],
                       default='two_stage',
                       help='Training mode strategy: linear_probe, last_block_finetune, partial_finetune, selective_20percent_last, selective_20percent_random, full_finetune, two_stage (default: two_stage)')
    
    # ========== Few-Shot Learning (Assignment 4.3) ==========
    # Removed --few-shot flag, now use --few-shot-percentage instead
    # Options: 5% (extreme low-data), 20% (low-data), 100% (full data baseline)
    
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
    parser.add_argument('--report-efficiency', action='store_true', default=True,
                       help='Report model efficiency (parameters, MACs, FLOPs) per assignment (default: True)')
    
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
    # Parse device - support cuda, cpu, cuda:0, cuda:1, etc.
    device_str = args.device
    try:
        device = torch.device(device_str)
    except:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cuda_available = device.type == 'cuda'
    gpu_names = []
    gpu_indices = []
    
    if cuda_available:
        if device.index is not None:
            # Single GPU specified like cuda:0
            gpu_indices = [device.index]
            gpu_names = [torch.cuda.get_device_name(device.index)]
        else:
            # All GPUs available
            gpu_indices = list(range(torch.cuda.device_count()))
            gpu_names = [torch.cuda.get_device_name(i) for i in gpu_indices]
    
    gpu_count = len(gpu_indices)
    
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
    # TRAINING MODES CONFIGURATION (Assignment 4.2)
    # ============================================================================
    # Assignment Requirements:
    # - All modes must have fixed hyperparameters except data percentage (4.3)
    # - Linear probe: 0.01 learning rate (unfrozen head only)
    # - Fine-tuning modes: 0.0001 learning rate
    # - Selective 20%: Must explain reasoning (balance between expressiveness & overfitting)
    
    # Validate epochs based on few-shot setting
    effective_epochs = args.epochs
    if args.few_shot_percentage < 100:
        effective_epochs = min(args.epochs, MAX_EPOCHS_FEW_SHOT)
    else:
        effective_epochs = min(args.epochs, MAX_EPOCHS_FULL_DATA)
    
    training_modes = {
        'linear_probe': {
            'description': '[Assignment 4.1] Linear Probing - Frozen backbone, train only classification head',
            'freeze_backbone': True,
            'learning_rate': 0.01,  # Fixed per assignment
            'epochs': effective_epochs,
            'num_layers_to_unfreeze': None,
        },
        'last_block_finetune': {
            'description': '[Assignment 4.2] Last Block Fine-tuning - Unfreeze last 3 layers (~2% params)',
            'freeze_backbone': False,
            'num_layers_to_unfreeze': 3,
            'learning_rate': 0.0001,  # Fixed per assignment
            'epochs': effective_epochs,
        },
        'partial_finetune': {
            'description': '[Assignment 4.2] Partial Fine-tuning - Unfreeze last N layers (~30% default)',
            'freeze_backbone': False,
            'num_layers_to_unfreeze': args.num_layers_unfreeze,
            'learning_rate': 0.0001,  # Fixed per assignment
            'epochs': effective_epochs,
        },
        'selective_20percent_last': {
            'description': '[Assignment 4.2] Selective 20% (Last) - Unfreeze last 20% of params from end',
            'freeze_backbone': False,
            'num_layers_to_unfreeze': 'auto_20percent_last',
            'learning_rate': 0.0001,  # Fixed per assignment
            'epochs': effective_epochs,
            'strategy': 'last',
        },
        'selective_20percent_random': {
            'description': '[Assignment 4.2] Selective 20% (Random) - Randomly unfreeze 20% of params',
            'freeze_backbone': False,
            'num_layers_to_unfreeze': 'auto_20percent_random',
            'learning_rate': 0.0001,  # Fixed per assignment
            'epochs': effective_epochs,
            'strategy': 'random',
        },
        'full_finetune': {
            'description': '[Assignment 4.2] Full Fine-tuning - Unfreeze all backbone layers',
            'freeze_backbone': False,
            'num_layers_to_unfreeze': None,
            'learning_rate': 0.0001,  # Fixed per assignment
            'epochs': effective_epochs,
        },
        'two_stage': {
            'description': '[Assignment 4.2] Two-Stage - Linear probe then partial fine-tune',
            'stage_1': {
                'freeze_backbone': True,
                'learning_rate': 0.01,  # Fixed per assignment
                'epochs': effective_epochs // 2 if effective_epochs > 0 else 15,
            },
            'stage_2': {
                'freeze_backbone': False,
                'num_layers_to_unfreeze': args.num_layers_unfreeze,
                'learning_rate': 0.0001,  # Fixed per assignment
                'epochs': effective_epochs - (effective_epochs // 2) if effective_epochs > 0 else 15,
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
    # FEW-SHOT LEARNING CONFIGURATION (Assignment 4.3)
    # ============================================================================
    # Assignment Requirements:
    # - 100% training data (full baseline)
    # - 20% training data (low-data regime)
    # - 5% training data (extreme low-data regime)
    # - Max 20 epochs for few-shot (vs 30 for full-data)
    # - Must use random seed for subset generation
    
    few_shot_config = {
        'percentage': args.few_shot_percentage,  # 5, 20, or 100
        'use_subset': args.few_shot_percentage < 100,
        'max_epochs': MAX_EPOCHS_FEW_SHOT if args.few_shot_percentage < 100 else MAX_EPOCHS_FULL_DATA,
        'random_seed': args.random_seed,
        'description': f'Few-shot learning with {args.few_shot_percentage}% of training data'
    }
    
    # ============================================================================
    # RETURN CONFIGURATION DICTIONARY
    # ============================================================================
    config = {
        # Device
        'device': device,
        'gpu_indices': gpu_indices,
        'gpu_names': gpu_names,
        'cuda_available': cuda_available,
        'gpu_name': gpu_names[0] if gpu_names else "None",
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
        
        # Few-shot learning (Assignment 4.3)
        'few_shot_config': few_shot_config,
        'few_shot_percentage': args.few_shot_percentage,
        
        # Assignment constraints
        'assignment_constraints': {
            'max_epochs_full_data': MAX_EPOCHS_FULL_DATA,
            'max_epochs_few_shot': MAX_EPOCHS_FEW_SHOT,
            'max_hours_per_scenario': MAX_HOURS_PER_SCENARIO,
            'num_classes': NUM_CLASSES,
            'random_seed': RANDOM_SEED,
            'few_shot_percentages': [5, 20, 100],
            'linear_probe_lr': 0.01,
            'finetune_lr': 0.0001,
        },
        
        # Display
        'verbose': args.verbose,
        'print_interval': args.print_interval,
        'plot_history': args.plot_history,
        'save_confusion_matrices': args.save_confusion_matrices,
        'generate_classification_reports': args.generate_reports,
        'report_efficiency': args.report_efficiency,
    }
    
    return config


def print_config(config):
    """Print configuration summary with assignment constraints"""
    print("=" * 80)
    print(" GNR 638 ASSIGNMENT 2: TRANSFER LEARNING & ROBUSTNESS ANALYSIS")
    print("=" * 80)
    
    # Assignment Constraints
    print("\n[ASSIGNMENT CONSTRAINTS]")
    constraints = config['assignment_constraints']
    print(f"  Max Epochs (Full Data):     {constraints['max_epochs_full_data']}")
    print(f"  Max Epochs (Few-Shot):      {constraints['max_epochs_few_shot']}")
    print(f"  Max Hours per Scenario:     {constraints['max_hours_per_scenario']} hours")
    print(f"  Dataset:                    Aerial Images (AID) - {constraints['num_classes']} classes")
    print(f"  Linear Probe LR:            {constraints['linear_probe_lr']}")
    print(f"  Fine-tuning LR:             {constraints['finetune_lr']}")
    print(f"  Random Seed (Reproducibility): {constraints['random_seed']}")
    print(f"  Few-Shot Percentages:       {constraints['few_shot_percentages']}")
    
    # Paths
    print(f"\n[PATHS]")
    print(f"  Project Root:               {config['project_root']}")
    print(f"  Data Path:                  {config['data_path']}")
    print(f"  Results Directory:          {config['results_dir']}")
    
    # Device
    print(f"\n[DEVICE CONFIGURATION]")
    print(f"  Device:                     {config['device']}")
    print(f"  CUDA Available:             {config['cuda_available']}")
    if config['cuda_available']:
        print(f"  GPU Name(s):                {', '.join(config['gpu_names'])}")
        print(f"  GPU Count:                  {config['gpu_count']}")
    
    # Training
    print(f"\n[TRAINING CONFIGURATION]")
    print(f"  Batch Size:                 {config['batch_size']}")
    print(f"  Epochs:                     {config['epochs']}")
    print(f"  Few-Shot Percentage:        {config['few_shot_percentage']}%")
    print(f"  Effective Epochs:           {config['training_modes'][config['selected_training_mode']]['epochs']}")
    print(f"  Random Seed:                {config['random_seed']}")
    print(f"  Early Stopping Patience:    {config['early_stopping_patience']}")
    
    # Model
    print(f"\n[MODEL CONFIGURATION]")
    print(f"  Number of Classes:          {config['num_classes']}")
    print(f"  Dropout Rate:               {config['dropout_rate']}")
    print(f"  Layers to Unfreeze:         {config['num_layers_to_unfreeze']}")
    print(f"  Report Efficiency Metrics:  {config['report_efficiency']}")
    
    # Training Mode
    print(f"\n[TRAINING MODE (Assignment 4.2)]")
    mode_name = config['selected_training_mode']
    mode_config = config['training_modes'][mode_name]
    print(f"  Mode:                       {mode_name}")
    print(f"  Description:                {mode_config['description']}")
    print(f"  Learning Rate:              {mode_config['learning_rate']}")
    print(f"  Epochs:                     {mode_config['epochs']}")
    if 'strategy' in mode_config:
        print(f"  Strategy:                   {mode_config['strategy']}")
    
    # Few-Shot
    print(f"\n[FEW-SHOT CONFIGURATION (Assignment 4.3)]")
    print(f"  Percentage:                 {config['few_shot_percentage']}%")
    print(f"  Use Subset:                 {config['few_shot_config']['use_subset']}")
    print(f"  Max Epochs:                 {config['few_shot_config']['max_epochs']}")
    print(f"  Description:                {config['few_shot_config']['description']}")
    
    # Models
    print(f"\n[AVAILABLE MODELS]")
    for model_name in config['models_config'].keys():
        model_info = config['models_config'][model_name]
        print(f"  - {model_info['name']:20s} (Input: {model_info['input_size']})")
    
    print("=" * 80)



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
