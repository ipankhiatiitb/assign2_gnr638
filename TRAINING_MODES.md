# Training Modes Guide

This document explains the different training modes available in the satellite image classification system and how to use them.

## Overview

The system implements 4 distinct training strategies that allow you to balance:
- **Training speed** (time to completion)
- **Model accuracy** (final performance)
- **Computational resources** (GPU memory, computation)

## Training Modes

### 1. Linear Probing

**What it does:**
- Freezes the entire pre-trained backbone (feature extractor)
- Trains ONLY the top classification layers
- Quick training without gradient computation through backbone

**Configuration:**
```python3
TRAINING_MODES['linear_probe'] = {
    'description': 'Linear probing with frozen backbone',
    'epochs': 30,
    'initial_learning_rate': 0.001,
}
```

**When to use:**
- Quick baseline training
- Proof of concept
- Resource-constrained environments
- Testing data pipeline

**Expected performance:**
- Training time: ~5-10 minutes per model (depending on data size)
- Accuracy: Baseline accuracy (~60-75%)
- Memory: Low (frozen backbone)

**CLI Command:**
```bash
python3 train.py --model resnet50 --training-mode linear_probe
```

---

### 2. Partial Fine-tuning

**What it does:**
- Freezes most of the backbone
- Unfreezes the last 50 layers for training
- Lower learning rate (0.0001) to preserve pre-trained weights
- Adapts backbone to satellite imagery features

**Configuration:**
```python3
TRAINING_MODES['partial_finetune'] = {
    'description': 'Partial fine-tuning with last 50 layers unfrozen',
    'epochs': 50,
    'initial_learning_rate': 0.0001,
    'num_layers_to_unfreeze': 50,
}
```

**When to use:**
- Need better accuracy than linear probing
- Want to adapt pre-trained features to satellite domain
- Balance between training time and accuracy
- Production models with moderate computational resources

**Expected performance:**
- Training time: ~15-25 minutes per model
- Accuracy: Good (~80-85%)
- Memory: Medium (partial backward pass)

**CLI Command:**
```bash
python3 train.py --model efficientnetb0 --training-mode partial_finetune
```

---

### 3. Full Fine-tuning

**What it does:**
- Unfreezes ALL layers in the backbone
- Trains the entire network end-to-end
- Very low learning rate (0.00001) to stabilize training
- Maximum model adaptation but highest computational cost

**Configuration:**
```python3
TRAINING_MODES['full_finetune'] = {
    'description': 'Full fine-tuning with all layers unfrozen',
    'epochs': 100,
    'initial_learning_rate': 0.00001,
}
```

**When to use:**
- Need maximum accuracy
- Have large, high-quality dataset
- Sufficient computational resources (GPU with 8GB+ VRAM)
- Time is not a limiting factor

**Expected performance:**
- Training time: ~30-60 minutes per model
- Accuracy: Highest (~85-92%)
- Memory: High (full backward pass)

**CLI Command:**
```bash
python3 train.py --model inceptionv3 --training-mode full_finetune
```

---

### 4. Two-Stage Training (RECOMMENDED DEFAULT)

**What it does:**
- **Stage 1 (30 epochs):** Linear probing with frozen backbone
  - Quick initialization of top layers
  - Prevents garbage gradients in initial epochs
  
- **Stage 2 (50 epochs):** Partial fine-tuning
  - Unfreezes last 50 layers
  - Uses moderate learning rate
  - Fine-tunes pre-trained features

**Configuration:**
```python3
TRAINING_MODES['two_stage'] = {
    'description': 'Two-stage: linear probing + partial fine-tuning',
    'stage_1': {
        'epochs': 30,
        'initial_learning_rate': 0.001,
    },
    'stage_2': {
        'epochs': 50,
        'initial_learning_rate': 0.0001,
        'num_layers_to_unfreeze': 50,
    }
}
```

**When to use:**
- Best overall choice for most applications
- Want good accuracy without maximum computational cost
- Practical production scenarios
- Recommended for assignments and benchmarking

**Expected performance:**
- Training time: ~20-35 minutes per model
- Accuracy: Very good (~82-90%)
- Memory: Medium
- **Best accuracy-to-time ratio**

**CLI Command:**
```bash
python3 train.py --model resnet50 --training-mode two_stage
python3 train.py --all --training-mode two_stage  # All models with two-stage
```

---

## Comparison Table

| Aspect | Linear Probe | Partial Finetune | Full Finetune | Two-Stage |
|--------|--------------|------------------|---------------|-----------|
| **Speed** | ⚡⚡⚡ Fastest | ⚡⚡ Medium | ⚡ Slowest | ⚡⚡ Good |
| **Accuracy** | ⭐⭐ Poor | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Best | ⭐⭐⭐⭐ Very Good |
| **Memory** | 💾 Low | 💾💾 Medium | 💾💾💾 High | 💾💾 Medium |
| **Frozen Layers** | All | Most | None | Stages: All→Most |
| **Epochs** | 30 | 50 | 100 | 30+50 |
| **Best For** | Quick test | Production | Maximum accuracy | Default choice |

---

## How to Select Training Mode

### Method 1: Command Line (Recommended)
```bash
# Linear probing
python3 train.py --model resnet50 --training-mode linear_probe

# Partial fine-tuning
python3 train.py --model efficientnetb0 --training-mode partial_finetune

# Full fine-tuning
python3 train.py --model inceptionv3 --training-mode full_finetune

# Two-stage (default)
python3 train.py --model resnet50 --training-mode two_stage

# Train all models with specific mode
python3 train.py --all --training-mode partial_finetune
```

### Method 2: Modify Config File

Edit `config.py` and change the `SELECTED_TRAINING_MODE`:

```python3
# In config.py, line XX
SELECTED_TRAINING_MODE = 'two_stage'  # Change to desired mode
```

Then train without specifying mode:
```bash
python3 train.py --model resnet50
```

---

## Few-Shot Learning Integration

The system also supports few-shot learning mode (training with limited samples per class):

```python3
# In config.py
FEW_SHOT_CONFIG = {
    'enabled': False,  # Set to True to enable
    'samples_per_class': 10  # Number of samples per class
}
```

When enabled:
- Only uses specified number of samples per class
- Good for testing data efficiency
- Can be combined with any training mode
- Useful for simulating limited data scenarios

---

## Practical Recommendations

### For Rapid Prototyping
```bash
python3 train.py --model resnet50 --training-mode linear_probe
# 5-10 minutes, quick validation of code/data pipeline
```

### For Assignment Submission
```bash
python3 train.py --all --training-mode two_stage
# ~90-110 minutes total, excellent balance of accuracy and time
```

### For Production Model
```bash
python3 train.py --model resnet50 --training-mode full_finetune
python3 train.py --model efficientnetb0 --training-mode full_finetune
python3 train.py --model inceptionv3 --training-mode full_finetune
# 60-180 minutes total, maximum accuracy per model
```

### For Limited Resources
```bash
python3 train.py --model resnet50 --training-mode linear_probe
# or
python3 train.py --model resnet50 --training-mode partial_finetune
# Fast training on CPU or lower-end GPU
```

---

## Understanding the Training Process

### Layer Freezing Explanation

**What is "freezing"?**
- Frozen layers: Parameters not updated during training (requires_grad=False)
- Unfrozen layers: Parameters updated via backpropagation (requires_grad=True)

**Why freeze layers?**
1. **Preserve pre-trained knowledge** - ImageNet features still useful
2. **Reduce memory usage** - Don't store gradients for frozen layers
3. **Faster training** - Fewer computations per epoch
4. **Prevent overfitting** - Limited parameter updates with small datasets

---

## Troubleshooting

### My training is too slow
→ Use `linear_probe` or `partial_finetune` instead of `full_finetune`

### My model's accuracy is low
→ Try `two_stage` (default) or `full_finetune`

### I'm running out of GPU memory
→ Use `linear_probe` or reduce batch size in config.py

### I want quick results for testing
→ Use `linear_probe`, takes only 5-10 minutes per model

### I want the best accuracy
→ Use `full_finetune` (takes longer but highest accuracy)

---

## Summary

- **Linear Probing**: Fastest, baseline performance (~5 min/model)
- **Partial Fine-tuning**: Good balance (~15 min/model)
- **Full Fine-tuning**: Best accuracy, slowest (~45 min/model)
- **Two-Stage**: Recommended default, best tradeoff (~25 min/model)

Choose based on your priorities:
- Time-constrained? → Linear probing
- Want best results? → Two-stage or full fine-tuning
- Limited resources? → Linear probing
- Production model? → Full fine-tuning
