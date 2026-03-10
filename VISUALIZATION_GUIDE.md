# Visualization Guide

This guide explains how to generate visualizations for trained models.

## Overview

There are two main visualization scripts:

1. **`visualize_results.py`** - Individual model visualizations
2. **`compare_models.py`** - Comparative visualizations across all models

## 1. Individual Model Visualizations

### Usage

```bash
# Visualize a specific model
python visualize_results.py --model resnet50
python visualize_results.py --model efficientnetb0
python visualize_results.py --model inceptionv3

# Visualize all models
python visualize_results.py --all

# List available models
python visualize_results.py --list-models
```

### Generated Visualizations

For each model, the following visualizations are generated:

1. **Training History** (`{model}_training_history.png`)
   - Training vs Validation Loss
   - Training vs Validation Accuracy
   - Useful for detecting overfitting

2. **Confusion Matrix** (`{model}_confusion_matrix.png`)
   - Shows misclassifications for all 30 classes
   - Large annotated heatmap (20x18)
   - Helps identify which classes are confused with each other

3. **Per-Class Metrics** (`{model}_per_class_metrics.png`)
   - Precision for each class
   - Recall for each class
   - F1-Score for each class
   - Bar charts with values labeled

4. **Architecture Summary** (`{model}_architecture_summary.png`)
   - Model type and input size
   - Total and trainable parameters
   - FLOPs calculation
   - Training configuration

5. **Sample Predictions** (`{model}_sample_predictions.png`)
   - 9 random samples from test set
   - Shows true label, predicted label, and confidence
   - Correct predictions in green, incorrect in red

## 2. Comparative Model Visualizations

### Usage

```bash
# Generate all comparison visualizations
python compare_models.py
```

This script evaluates all models and generates:

### Generated Visualizations

1. **Accuracy Comparison** (`model_accuracy_comparison.png`)
   - Bar chart comparing accuracy of all 3 models
   - Ranked from highest to lowest

2. **Per-Class F1-Score Comparison** (`per_class_f1_comparison.png`)
   - Grouped bar chart comparing F1-scores
   - One set of bars per model
   - All 30 classes on x-axis

3. **Confusion Matrices Comparison** (`confusion_matrices_comparison.png`)
   - Three confusion matrices side by side
   - Normalized for easier comparison
   - Includes accuracy in title

4. **Comparison Report** (`model_comparison_report.txt`)
   - Text file with detailed metrics
   - Overall accuracy ranking
   - Per-class metrics for all models
   - Weighted averages

## Workflow Example

```bash
# 1. Train models (if not already trained)
python train.py --all --training-mode linear_probe --epochs 30

# 2. Generate individual visualizations
python visualize_results.py --model resnet50
python visualize_results.py --model efficientnetb0
python visualize_results.py --model inceptionv3

# Or all at once
python visualize_results.py --all

# 3. Generate comparison visualizations
python compare_models.py

# 4. Check results
ls results/
```

## Output Files Location

All visualization files are saved in the `results/` directory:

```
results/
├── resnet50_training_history.png
├── resnet50_confusion_matrix.png
├── resnet50_per_class_metrics.png
├── resnet50_architecture_summary.png
├── resnet50_sample_predictions.png
├── efficientnetb0_training_history.png
├── efficientnetb0_confusion_matrix.png
├── efficientnetb0_per_class_metrics.png
├── efficientnetb0_architecture_summary.png
├── efficientnetb0_sample_predictions.png
├── inceptionv3_training_history.png
├── inceptionv3_confusion_matrix.png
├── inceptionv3_per_class_metrics.png
├── inceptionv3_architecture_summary.png
├── inceptionv3_sample_predictions.png
├── model_accuracy_comparison.png
├── per_class_f1_comparison.png
├── confusion_matrices_comparison.png
└── model_comparison_report.txt
```

## Troubleshooting

### Issue: "Model not found" error

**Solution:** Train the model first
```bash
python train.py --model resnet50 --training-mode linear_probe
```

### Issue: Visualizations not appearing correctly

**Solution:** Ensure matplotlib backend is set correctly
```bash
# Add to beginning of visualization script
import matplotlib
matplotlib.use('Agg')
```

### Issue: Out of memory when generating visualizations

**Solution:** Reduce batch size in config
```bash
python train.py --model resnet50 --batch-size 16
```

## Tips

- Run visualizations after training completes
- Generate comparison visualizations after all individual models are trained
- Use `--all` flag to visualize all models at once
- Check the text report for detailed per-class metrics
- Compare confusion matrices to understand model-specific weaknesses

## See Also

- `train.py` - Training models
- `evaluate.py` - Model evaluation
- `config.py` - Configuration management
