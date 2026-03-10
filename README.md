# Satellite Image Classification - Assignment 2

A comprehensive **PyTorch-based** deep learning solution for satellite image classification using transfer learning with ResNet50, EfficientNet B0, and InceptionV3.

## Project Structure

```
├── config.py              # Centralized configuration (all parameters & paths)
├── models.py              # Model training classes and functions (PyTorch)
├── train.py               # Unified training script with CLI
├── evaluate.py            # Model evaluation script with CLI
├── data_utils.py          # PyTorch data loading and preprocessing
├── data/
│   └── split_data/
│       ├── train/         # Training data (33 classes)
│       ├── val/           # Validation data (33 classes)
│       └── test/          # Test data (33 classes)
└── results/               # Output directory (models, plots, reports)
```

## Configuration

All settings are centralized in `config.py`:

- **Data paths**: Training, validation, and test directories
- **Training parameters**: Epochs, batch size, learning rates
- **Model specifications**: Input sizes, hidden layers, dropout rates
- **Data augmentation**: Rotation, zoom, shifts, flips
- **Output paths**: Model save locations, plot paths

### Available Models

1. **ResNet50**: Input size 224×224, 50 layers
2. **EfficientNet B0**: Input size 224×224, efficient architecture
3. **InceptionV3**: Input size 299×299, multi-scale features

## Installation

Install required packages:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib scikit-learn pillow seaborn tqdm
```

### GPU Support

PyTorch will automatically detect and use CUDA if available. To force CPU usage:

```bash
export CUDA_VISIBLE_DEVICES=""
python train.py --model resnet50
```

## Usage

### Training Modes

The system supports four different training strategies to balance speed vs accuracy:

#### 1. Linear Probing
- **Description**: Train only the top classification layer while keeping the backbone frozen
- **Use case**: Quick baseline training with minimal computational cost
- **Best for**: Data-efficient scenarios, quick experimentation

```bash
python train.py --model resnet50 --training-mode linear_probe
```

#### 2. Partial Fine-tuning
- **Description**: Unfreeze the last 50 layers and train with lower learning rate
- **Use case**: Balance between speed and accuracy
- **Best for**: Good accuracy with moderate training time

```bash
python train.py --model resnet50 --training-mode partial_finetune
```

#### 3. Full Fine-tuning
- **Description**: Unfreeze all layers and train the entire network with very low learning rate
- **Use case**: Maximum accuracy when sufficient data and compute available
- **Best for**: Fully utilizing the model capacity

```bash
python train.py --model resnet50 --training-mode full_finetune
```

#### 4. Two-Stage Training (Default)
- **Description**: Combines linear probing (30 epochs) + partial fine-tuning (50 epochs)
- **Use case**: Best overall balance of speed and accuracy
- **Best for**: Most practical applications

```bash
python train.py --model resnet50 --training-mode two_stage
python train.py --all --training-mode two_stage
```

### Training Models

#### Train a specific model:
```bash
python train.py --model resnet50
python train.py --model efficientnetb0
python train.py --model inceptionv3
```

#### Train with specific training mode:
```bash
python train.py --model resnet50 --training-mode linear_probe
python train.py --model efficientnetb0 --training-mode two_stage
python train.py --model inceptionv3 --training-mode full_finetune
```

#### Train all models:
```bash
python train.py --all
```

#### Train all with specific mode:
```bash
python train.py --all --training-mode partial_finetune
```

#### Show configuration:
```bash
python train.py --config
```

#### List available models and training modes:
```bash
python train.py --list-models
```

### Evaluating Models

#### Evaluate a specific model:
```bash
python evaluate.py --model resnet50
python evaluate.py --model efficientnetb0
python evaluate.py --model inceptionv3
```

#### Evaluate all trained models:
```bash
python evaluate.py --all
```

#### List available models:
```bash
python evaluate.py --list-models
```

## Training Pipeline

### 1. Data Loading and Preprocessing

- Loads images from organized directory structure (33 classes)
- Applies data augmentation (rotation, zoom, shifts, flips)
- Rescales images to model-specific input sizes
- Uses PyTorch DataLoader for efficient batch processing

### 2. Model Building

- Loads ImageNet pre-trained weights from torchvision
- Freezes base model layers initially
- Adds custom top layers with dense and dropout layers
- Uses CrossEntropyLoss and Adam optimizer

### 3. Initial Training

- Trains with frozen base model layers
- Early stopping when validation loss plateaus
- Reduces learning rate on validation loss plateau
- Saves best model checkpoints
- Epochs: 50 (or until early stopping)

### 4. Fine-tuning

- Unfreezes last 50 base model layers
- Uses lower learning rate (0.0001)
- Trains again with same callbacks
- Further improves model performance

### 5. Evaluation

- Tests on test set
- Generates confusion matrices
- Creates classification reports
- Compares model accuracies

## Output Files

Models and results are saved in the `results/` directory:

```
results/
├── resnet50_final_model.pth
├── resnet50_best_model.pth
├── resnet50_training_history.png
├── resnet50_confusion_matrix.png
├── resnet50_classification_report.txt
├── efficientnetb0_final_model.pth
├── efficientnetb0_best_model.pth
├── efficientnetb0_training_history.png
├── efficientnetb0_confusion_matrix.png
├── efficientnetb0_classification_report.txt
├── inceptionv3_final_model.pth
├── inceptionv3_best_model.pth
├── inceptionv3_training_history.png
├── inceptionv3_confusion_matrix.png
├── inceptionv3_classification_report.txt
├── accuracy_comparison.png
└── evaluation_summary_report.txt
```

## Key Features

✅ **Pure PyTorch**: Uses PyTorch for all deep learning operations  
✅ **Transfer Learning**: Uses ImageNet pre-trained weights  
✅ **Data Augmentation**: Rotation, zoom, shifts, and flips  
✅ **Flexible Configuration**: Easily modify parameters in config.py  
✅ **Unified Training**: Single script handles all models  
✅ **Comprehensive Evaluation**: Confusion matrices, classification reports  
✅ **33 Classes**: Handles all satellite image categories  
✅ **Visualizations**: Training curves, confusion matrices, accuracy plots  
✅ **GPU Support**: Automatic GPU detection and utilization  
✅ **CLI Interface**: Easy command-line usage with argparse

## Model Performance

After training and evaluation, comparison metrics are generated including:
- Test set accuracy for each model
- Per-class precision, recall, and F1-scores
- Confusion matrices
- Training history plots
- Summary report with best model identification

## Customization

To modify training parameters, edit `config.py`:

```python
# Change number of epochs
EPOCHS = 50

# Change batch size
BATCH_SIZE = 32

# Change learning rate
INITIAL_LEARNING_RATE = 0.001

# Change dropout rate for all models
'dropout_rate': 0.3

# Change data augmentation
DATA_AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'zoom_range': 0.2,
    ...
}
```

## Troubleshooting

**GPU Out of Memory:**
```python
# Reduce batch size in config.py
BATCH_SIZE = 16  # Instead of 32
```

**Model Not Found:**
```bash
python train.py --list-models
# Check available models
```

**Data Loading Issues:**
```python
# Verify data paths in config.py
TRAIN_DIR = './data/split_data/train'
```

## PyTorch Implementation Details

### Data Loading
- Custom `SatelliteImageDataset` class extending `torch.utils.data.Dataset`
- PyTorch `DataLoader` with multiple workers for efficient loading
- Batch processing with automatic GPU transfer

### Model Architecture
- Base models from `torchvision.models` with pre-trained weights
- Custom sequential head with linear layers and dropout
- Proper feature extraction and classification layers

### Training
- CrossEntropyLoss for multi-class classification
- Adam optimizer with configurable learning rates
- Learning rate scheduler (ReduceLROnPlateau)
- Early stopping based on validation loss

### Evaluation
- Batch-wise predictions with no_grad context
- NumPy array conversions for scikit-learn compatibility
- GPU-efficient evaluation with proper memory management

## References

- ResNet: He, K., et al. (2015). Deep Residual Learning for Image Recognition
- EfficientNet: Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling
- Inception: Szegedy, C., et al. (2016). Rethinking the Inception Architecture
- PyTorch: https://pytorch.org/

## License

This project is part of the GNR 638 course assignment.
