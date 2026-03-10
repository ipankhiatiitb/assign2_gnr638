# Training Logging and Metrics Documentation

## Overview

The training pipeline now logs comprehensive metrics to both console and text files, including:

1. **Data Loading Time**
2. **Model Architecture Metrics** (Parameters, FLOPs, MACs)
3. **Per-Epoch Training Metrics** (Loss, Accuracy, Time)
4. **Overall Training Summary** (Total time, Average epoch time)

## What Gets Logged

### 1. Data Loading Metrics
- **Data Loading Time**: Time taken to load and prepare training/validation datasets
- **Training Samples**: Total number of training examples
- **Validation Samples**: Total number of validation examples
- Logged at the start of training

```
Loading data with image size (224, 224)...
Training samples: 8580
Validation samples: 2145
Data Loading Time: 12.34s
```

### 2. Model Summary
- **Total Parameters**: All model parameters
- **Trainable Parameters**: Parameters that will be updated during training
- **Input Size**: Model's expected input dimensions
- **Number of Classes**: Classification targets (33 classes)
- **Device**: GPU or CPU being used
- **FLOPs**: Floating Point Operations (if fvcore installed)
- **MACs**: Multiply-Accumulate operations (if fvcore installed)

```
======================================================================
RESNET50 - MODEL SUMMARY
======================================================================
Total Parameters: 25,557,032
Trainable Parameters: 1,048,576
Input Size: (224, 224)
Number of Classes: 33
Device: cuda
FLOPs: 4,118,929,408 (4.12G)
MACs: 2,059,464,704 (2.06G)
======================================================================
```

### 3. Per-Epoch Metrics
For each epoch during training:
- **Epoch Number**: Current epoch (e.g., Epoch 5/30)
- **Training Loss**: Cross-entropy loss on training set
- **Training Accuracy**: Accuracy on training set
- **Validation Loss**: Cross-entropy loss on validation set
- **Validation Accuracy**: Accuracy on validation set
- **Epoch Time**: Time taken for one complete epoch (train + validate)

```
Epoch 5/30
Train Loss: 0.8417, Train Acc: 0.7327
Val Loss: 0.6810, Val Acc: 0.7903
Epoch Time: 15.05s
✓ Best model saved! Val Acc: 0.7903
```

### 4. Training Summary
After training completes:
- **Total Training Time**: Total time for the entire training phase (in seconds and minutes)
- **Average Time per Epoch**: Mean time across all epochs
- **Total Epochs Completed**: Number of epochs actually run (may be less than target due to early stopping)

```
ResNet50 linear_probe training completed!
Total Training Time: 445.67s (7.43 minutes)
Average Time per Epoch: 14.86s
Total Epochs Completed: 30
```

## Log File Locations

Training logs are automatically saved in the results directory:

```
results/
├── resnet50_training_log.txt
├── efficientnetb0_training_log.txt
└── inceptionv3_training_log.txt
```

**File path format:** `{RESULTS_DIR}/{model_name}_training_log.txt`

### Example:
- `/mnt/DATA1/pankhi/assign2_gnr638/results/resnet50_training_log.txt`

## Accessing the Logs

### View in Terminal
```bash
# View ResNet50 training log
cat /mnt/DATA1/pankhi/assign2_gnr638/results/resnet50_training_log.txt

# View with line numbers
cat -n /mnt/DATA1/pankhi/assign2_gnr638/results/resnet50_training_log.txt

# Search for specific metrics
grep "Epoch Time" /mnt/DATA1/pankhi/assign2_gnr638/results/resnet50_training_log.txt

# View last 20 lines
tail -20 /mnt/DATA1/pankhi/assign2_gnr638/results/resnet50_training_log.txt
```

### View in VS Code
- Open Explorer (Ctrl+Shift+E)
- Navigate to `results/` folder
- Double-click any `*_training_log.txt` file

## Metrics Explanation

### Training Time Per Epoch
- **Formula**: Time to run `train_epoch()` + Time to run `validate_epoch()`
- **Factors affecting time**:
  - Number of training samples
  - Batch size (32 default)
  - Model complexity (ResNet50 < EfficientNet B0 < InceptionV3)
  - GPU/CPU speed
  - Data loading pipeline

### FLOPs and MACs
- **FLOPs (Floating Point Operations)**: Total number of mathematical operations
- **MACs (Multiply-Accumulate operations)**: Core computational unit (~FLOPs/2)
- **Units**: 
  - G = Billion (10^9)
  - Used to estimate computational cost

**Expected Values:**
- ResNet50: ~4.1G FLOPs, 2.1G MACs
- EfficientNet B0: ~0.4G FLOPs, 0.2G MACs (more efficient)
- InceptionV3: ~5.7G FLOPs, 2.9G MACs

### Data Loading Time
- **Includes**: Reading images, applying transforms, creating batches
- **Typical values**: 5-15 seconds for full dataset
- **Note**: First time is slower (caching); subsequent runs are faster

## Training Modes and Logging

Each training mode is logged clearly:

```
======================================================================
TRAINING RESNET50 - LINEAR_PROBE
======================================================================
Applying training mode: linear_probe
Description: Linear Probing - Train only top classification layer
✓ Backbone frozen. Only top layers trainable (linear probing)
```

## Complete Example Log

```
======================================================================
TRAINING RESNET50 - LINEAR_PROBE
======================================================================

Loading data with image size (224, 224)...
Training samples: 8580
Validation samples: 2145
Data Loading Time: 12.34s

Data Loading Time: 12.34s

======================================================================
RESNET50 - MODEL SUMMARY
======================================================================
Total Parameters: 25,557,032
Trainable Parameters: 1,048,576
Input Size: (224, 224)
Number of Classes: 33
Device: cuda
FLOPs: 4,118,929,408 (4.12G)
MACs: 2,059,464,704 (2.06G)
======================================================================

Epoch 1/30
Train Loss: 2.5351, Train Acc: 0.2790
Val Loss: 1.4151, Val Acc: 0.5517
Epoch Time: 15.17s
✓ Best model saved! Val Acc: 0.5517

Epoch 2/30
Train Loss: 1.2946, Train Acc: 0.5991
Val Loss: 0.9534, Val Acc: 0.6802
Epoch Time: 14.26s
✓ Best model saved! Val Acc: 0.6802

... [more epochs] ...

ResNet50 linear_probe training completed!
Total Training Time: 445.67s (7.43 minutes)
Average Time per Epoch: 14.86s
Total Epochs Completed: 30
```

## Performance Analysis Using Logs

You can analyze logs to understand:

1. **Training efficiency**: Check if loss is decreasing
2. **Overfitting**: Compare training vs validation accuracy
3. **Bottlenecks**: Identify if data loading is slow
4. **Model complexity**: Use FLOPs/MACs to compare models
5. **Hardware utilization**: Epoch time indicates GPU usage
6. **Total project time**: Sum all training times across models

## Tips

- **Quick training test**: Check Data Loading Time + first epoch. If good, training will be fine.
- **Batch time estimation**: Divide "Epoch Time" by number of batches to see per-batch speed
- **Early stopping**: Look for "Early stopping" message to see if validation stopped improving
- **Model comparison**: Compare avg epoch times across ResNet50, EfficientNet B0, InceptionV3
