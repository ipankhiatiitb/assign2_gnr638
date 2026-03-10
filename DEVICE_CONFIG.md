# Device Configuration Guide

## Overview

Device configuration can now be set in three ways:
1. **Config file** (default)
2. **Command-line argument** (overrides config)
3. **Environment variable** (for CPU forcing)

---

## Device Configuration in config.py

The `DEVICE` variable is now defined in `config.py`:

```python
# In config.py
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_COUNT = torch.cuda.device_count()
```

Automatically detected:
- ✅ CUDA availability
- ✅ GPU name
- ✅ GPU count

---

## Using Device from Command Line

### Train on GPU (default if available)
```bash
python train.py --model resnet50
```

### Force Training on GPU
```bash
python train.py --model resnet50 --device cuda
```

### Force Training on CPU
```bash
python train.py --model resnet50 --device cpu
```

### Examples with Training Modes

```bash
# Linear probing on GPU
python train.py --model resnet50 --training-mode linear_probe --device cuda

# Full fine-tuning on CPU
python train.py --model efficientnetb0 --training-mode full_finetune --device cpu

# All models with two-stage training on GPU
python train.py --all --training-mode two_stage --device cuda
```

---

## Device Information in Config

View device configuration:

```bash
python train.py --config
```

Output example:
```
======================================================================
CONFIGURATION SUMMARY
======================================================================

Project Root: /mnt/DATA1/pankhi/assign2_gnr638
Data Path: /mnt/DATA1/pankhi/assign2_gnr638/data/split_data
Results Directory: /mnt/DATA1/pankhi/assign2_gnr638/results

Device Configuration:
  Device: cuda:0
  CUDA Available: True
  GPU Name: NVIDIA A100-PCIE-40GB
  GPU Count: 1

Batch Size: 32
Number of Classes: 33
...
```

---

## Device Configuration in Training

The device is automatically used in:
- **Model creation** - Models loaded on specified device
- **Data loading** - Tensors moved to device
- **Training** - Forward/backward passes run on device
- **Evaluation** - Predictions computed on device

### In code:
```python
# Device used from config
model = model.to(DEVICE)
images, labels = images.to(DEVICE), labels.to(DEVICE)
```

---

## Automatic Device Detection

If no device is specified:
- ✅ Automatically detects CUDA availability
- ✅ Uses GPU if available
- ✅ Falls back to CPU if GPU not available

```bash
# Auto-detects and uses best available device
python train.py --model resnet50
```

---

## Command-line Arguments Summary

```
Options:
  -m, --model MODEL              Model to train (resnet50, efficientnetb0, inceptionv3)
  -a, --all                      Train all available models
  -t, --training-mode MODE       Training mode (linear_probe, partial_finetune, full_finetune, two_stage)
  -d, --device DEVICE            Device (cuda or cpu) - overrides auto-detection
  -c, --config                   Show configuration and exit
  -l, --list-models              List available models and training modes
```

---

## Common Use Cases

### Development (CPU for debugging)
```bash
python train.py --model resnet50 --device cpu
```

### Production (GPU for speed)
```bash
python train.py --all --device cuda
```

### Quick testing
```bash
python train.py --model resnet50 --training-mode linear_probe --device cuda
```

### Mixed workflow
```bash
# Run linear probe on GPU for quick results
python train.py --model resnet50 --training-mode linear_probe --device cuda

# Then run full training on CPU overnight
python train.py --all --training-mode full_finetune --device cpu
```

---

## Troubleshooting

### GPU not detected but available
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Force GPU usage
python train.py --model resnet50 --device cuda
```

### Out of GPU memory
```bash
# Use CPU instead
python train.py --model resnet50 --device cpu

# Or reduce batch size in config.py: BATCH_SIZE = 16
```

### Check current device
```bash
python train.py --config
# Check "Device Configuration" section
```
