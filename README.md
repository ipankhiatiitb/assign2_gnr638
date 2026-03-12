# Satellite Image Classification - GNR 638 Assignment 2

A comprehensive **PyTorch-based** deep learning solution for satellite image classification using transfer learning with **ResNet50**, **EfficientNet-B0**, and **InceptionV3**.

## 🚀 Quick Start

```bash
# Step 1: Create few-shot data splits
python create_splits.py

# Step 2: Train a model (example: Linear probe on 100% data)
python train.py --model resnet50 --training-mode linear_probe --few-shot-percentage 100 --epochs 30 --batch-size 32

# Step 3: Run advanced evaluations (corruption robustness + layer-wise probing)
python evaluate_advanced.py --model resnet50 --model-path results_linear_probe_resnet50/final_model.pth --batch-size 32
```

## 📋 Assignment Implementation

This project implements all **5 scenarios** from GNR 638 Assignment 2:

- **Scenario 4.1**: Linear probe (frozen backbone, head-only training)
- **Scenario 4.2**: Fine-tuning modes (last block, partial, selective, full)
- **Scenario 4.3**: Few-shot learning (100%, 20%, 5% data regimes)
- **Scenario 4.4**: Corruption robustness (Gaussian noise, motion blur, brightness)
- **Scenario 4.5**: Layer-wise feature probing (early, middle, final layers)

## 📁 Project Structure

```
├── config.py                     # Configuration & hyperparameters
├── models.py                     # Model training classes (PyTorch)
├── train.py                      # Training script with CLI
├── evaluate.py                   # Evaluation script
├── evaluate_advanced.py          # Corruption & layer-wise probing
├── data_utils.py                 # Data loading & preprocessing
├── create_splits.py              # Generate few-shot data splits
├── corruption_robustness.py      # Scenario 4.4 implementation
├── layer_wise_probing.py         # Scenario 4.5 implementation
├── feature_embeddings.py         # Feature extraction utilities
├── data/
│   ├── train_data/               # Original dataset (33 classes)
│   └── split_data_fewshot_*/     # Generated splits (100%, 20%, 5%)
└── results_*/                    # Results folders (one per experiment)
```

## ⚙️ Configuration

All settings centralized in `config.py`:

- **Fixed hyperparameters**:
  - Learning rates: 0.01 (linear probe), 0.0001 (fine-tuning)
  - Batch size: 32 (all experiments)
  - Epochs: 30 (full data), 20 (few-shot)
  - Random seed: 42 (reproducibility)

- **Data split**: 90% training, 10% validation (from original data)
- **Few-shot support**: Automatic 100%, 20%, 5% data regime handling
- **Models**: ResNet50 (224×224), EfficientNet-B0 (224×224), InceptionV3 (299×299)
- **Data augmentation**: Rotation, zoom, shifts, flips

## 📦 Installation

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib scikit-learn pillow seaborn tqdm
```

### GPU Support

GPU automatically detected and used. To force CPU:

```bash
export CUDA_VISIBLE_DEVICES=""
```

## 🎯 Training Modes

### 1. Linear Probe
- Train only classification head (frozen backbone)
- Fastest, least parameters updated
- Use case: Baseline, quick experimentation

```bash
python train.py --model resnet50 --training-mode linear_probe --few-shot-percentage 100
```

### 2. Last Block Fine-Tuning
- Unfreeze last residual block, train with LR=0.0001
- Balance between speed and accuracy
- Use case: Good accuracy with moderate training time

```bash
python train.py --model resnet50 --training-mode last_block_finetune --few-shot-percentage 100
```

### 3. Partial Fine-Tuning
- Unfreeze 50% of layers (from last), train with LR=0.0001
- More parameters updated than last block
- Use case: Better accuracy with longer training

```bash
python train.py --model resnet50 --training-mode partial_finetune --few-shot-percentage 100
```

### 4. Selective 20% - Last Layers
- Selectively unfreeze 20% of total parameters (last layers)
- Targeted fine-tuning strategy
- Use case: Limited computational budget

```bash
python train.py --model resnet50 --training-mode selective_20percent_last --few-shot-percentage 100
```

### 5. Selective 20% - Random Layers
- Selectively unfreeze 20% of random parameters
- Diverse layer updates across network
- Use case: Exploring alternative fine-tuning strategies

```bash
python train.py --model resnet50 --training-mode selective_20percent_random --few-shot-percentage 100
```

### 6. Full Fine-Tuning
- Unfreeze all layers, train entire network with LR=0.0001
- Maximum parameters updated, longest training
- Use case: Maximum accuracy when sufficient data/compute available

```bash
python train.py --model resnet50 --training-mode full_finetune --few-shot-percentage 100
```

## 🧪 Experiments

### Few-Shot Data Regimes

Run experiments with different amounts of training data:

```bash
# 100% data (9,000 training images, 30 epochs)
python train.py --model resnet50 --training-mode linear_probe --few-shot-percentage 100 --epochs 30

# 20% data (1,800 training images, 20 epochs)
python train.py --model resnet50 --training-mode linear_probe --few-shot-percentage 20 --epochs 20

# 5% data (450 training images, 20 epochs)
python train.py --model resnet50 --training-mode linear_probe --few-shot-percentage 5 --epochs 20
```

### Complete Experiment Suite

Train all models with all modes and all data regimes:

```bash
# Phase 0: Create splits
python create_splits.py

# Phase 1: Linear probe (9 configs: 3 models × 3 percentages)
for pct in 100 20 5; do
  for model in resnet50 efficientnetb0 inceptionv3; do
    epochs=$([[ $pct -eq 100 ]] && echo 30 || echo 20)
    python train.py --model $model --training-mode linear_probe --few-shot-percentage $pct --epochs $epochs
  done
done

# Phase 2: Fine-tuning (45 configs: 5 modes × 3 percentages × 3 models)
for pct in 100 20 5; do
  epochs=$([[ $pct -eq 100 ]] && echo 30 || echo 20)
  for mode in last_block_finetune partial_finetune selective_20percent_last selective_20percent_random full_finetune; do
    for model in resnet50 efficientnetb0 inceptionv3; do
      python train.py --model $model --training-mode $mode --few-shot-percentage $pct --epochs $epochs
    done
  done
done

# Phase 3: Advanced evaluations (corruption + layer-wise probing)
for model in resnet50 efficientnetb0 inceptionv3; do
  python evaluate_advanced.py --model $model --model-path results_linear_probe_${model}/final_model.pth
done
```

**Total**: 54+ training configurations + 6 advanced evaluations = 60+ experiments  
**Time**: ~22-24 hours (can run parallel on multiple GPUs)

## 📊 Output Structure

Each experiment creates a separate results folder:

```
results_linear_probe_resnet50/
├── training_log.txt              # Training metrics & logs
├── final_model.pth               # Final trained model
├── best_model.pth                # Best validation checkpoint
├── training_history.png          # Loss/accuracy curves
├── confusion_matrix.png          # Validation confusion matrix
└── classification_report.txt     # Per-class metrics

results_linear_probe_resnet50_20/
├── training_log.txt
├── final_model.pth
├── ... (same structure)

results_full_finetune_efficientnetb0_5/
├── training_log.txt
├── ... (same structure)

results_corruption_robustness_resnet50/
├── training_log.txt
└── training_history.png          # 4-panel robustness plot

results_layer_wise_probing_inceptionv3/
├── training_log.txt
└── training_history.png          # Layer-wise accuracy + feature norms
```

## 🔍 Key Features

✅ **Pure PyTorch**: All deep learning operations in PyTorch  
✅ **Transfer Learning**: ImageNet pre-trained models  
✅ **Data Augmentation**: Rotation, zoom, shifts, flips  
✅ **Flexible Config**: Centralized parameter management  
✅ **6 Training Modes**: Multiple fine-tuning strategies  
✅ **Few-Shot Support**: 100%, 20%, 5% data regimes  
✅ **33 Classes**: All satellite image categories  
✅ **Visualizations**: Training curves, confusion matrices, robustness plots  
✅ **GPU Optimized**: Automatic GPU detection and DataParallel  
✅ **Reproducible**: Fixed random seeds, deterministic behavior  
✅ **Advanced Evaluation**: Corruption robustness + layer-wise probing  
✅ **CLI Interface**: Easy command-line usage with argparse

## 📈 Advanced Evaluations

### Corruption Robustness (Scenario 4.4)

Evaluate model robustness to distribution shifts:

```bash
python evaluate_advanced.py --model resnet50 --model-path results_linear_probe_resnet50/final_model.pth
```

Tests with:
- **Gaussian noise** (σ = 0.05, 0.10, 0.20)
- **Motion blur** (kernel sizes)
- **Brightness shift** (factors 0.5, 1.5)

Metrics: Clean accuracy, corruption accuracy, robustness ratio

### Layer-Wise Probing (Scenario 4.5)

Analyze feature learning across layers:

```bash
python evaluate_advanced.py --model efficientnetb0 --model-path results_linear_probe_efficientnetb0/final_model.pth
```

Analyzes:
- **Early layers** (25% depth): Low-level features
- **Middle layers** (50% depth): Mid-level features
- **Final layers** (100% depth): High-level features

Metrics: Layer-wise accuracy, feature norms, abstraction progression

## 🔧 Customization

Modify `config.py` to change:

```python
# Data split (currently 90/10)
TRAIN_SIZE = 0.9

# Learning rates (fixed for reproducibility)
LINEAR_PROBE_LR = 0.01
FINETUNE_LR = 0.0001

# Batch size (fixed)
BATCH_SIZE = 32

# Max epochs (adjusts based on data regime)
MAX_EPOCHS_FULL_DATA = 30
MAX_EPOCHS_FEW_SHOT = 20

# Data augmentation
DATA_AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'zoom_range': 0.2,
    ...
}
```

## 📝 Constraints & Specifications

**Fixed Parameters** (enforced in code):
- Learning rates: 0.01 (linear), 0.0001 (fine-tuning)
- Batch size: 32
- Epochs: 30 (100% data), 20 (few-shot)
- Seed: 42
- Data split: 90% train / 10% validation
- Few-shot percentages: 100%, 20%, 5%

**Variable Parameters**:
- Model: ResNet50, EfficientNet-B0, InceptionV3
- Training mode: 6 different fine-tuning strategies
- Few-shot percentage: 100, 20, 5

## 🎯 Key Findings to Expect

1. **Few-shot degradation**: Model accuracy decreases with less training data
2. **Mode effectiveness**: Full FT > Partial > Last Block > Linear Probe (generally)
3. **Model robustness**: EfficientNet-B0 often best with limited data
4. **Corruption sensitivity**: All models degrade with Gaussian noise > brightness > motion blur
5. **Layer importance**: Final layers learn most important features for classification

## 🐛 Troubleshooting

**GPU Out of Memory:**
```python
# Reduce batch size in config.py
BATCH_SIZE = 16
```

**Model file not found:**
```bash
# Check folder name format: results_{mode}_{model}_{percentage}/final_model.pth
ls results_*/final_model.pth
```

**Data loading issues:**
```bash
# Verify split creation
python create_splits.py

# Check data directories
ls data/split_data_fewshot_*/
```

## 📚 References

- **ResNet**: He et al. (2015). Deep Residual Learning for Image Recognition
- **EfficientNet**: Tan & Le (2019). EfficientNet: Rethinking Model Scaling
- **InceptionV3**: Szegedy et al. (2016). Rethinking the Inception Architecture
- **PyTorch**: https://pytorch.org/
- **Assignment**: GNR 638 - Satellite Image Classification

## 📄 License

Part of GNR 638 Course Assignment
