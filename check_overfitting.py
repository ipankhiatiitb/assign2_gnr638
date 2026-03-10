import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(filename):
    """Parse training log file and extract metrics"""
    with open(filename, 'r') as f:
        content = f.read()
    
    # Extract epoch metrics
    epochs = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Pattern to match epoch lines
    pattern = r'Epoch (\d+)/\d+\nTrain Loss: ([\d.]+), Train Acc: ([\d.]+)\nVal Loss: ([\d.]+), Val Acc: ([\d.]+)'
    matches = re.finditer(pattern, content)
    
    for match in matches:
        epoch = int(match.group(1))
        train_loss = float(match.group(2))
        train_acc = float(match.group(3))
        val_loss = float(match.group(4))
        val_acc = float(match.group(5))
        
        epochs.append(epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    
    return epochs, train_losses, train_accs, val_losses, val_accs

def calculate_overfitting_metrics(train_losses, train_accs, val_losses, val_accs):
    """Calculate overfitting indicators"""
    
    # Loss gap: difference between train and validation loss
    final_loss_gap = val_losses[-1] - train_losses[-1]
    
    # Accuracy gap: difference between validation and train accuracy
    final_acc_gap = train_accs[-1] - val_accs[-1]
    
    # Check if val loss increases while train loss decreases (classic overfitting sign)
    train_loss_trend = train_losses[-1] - train_losses[5] if len(train_losses) > 5 else 0
    val_loss_trend = val_losses[-1] - val_losses[5] if len(val_losses) > 5 else 0
    
    return {
        'final_loss_gap': final_loss_gap,
        'final_acc_gap': final_acc_gap,
        'train_loss_trend': train_loss_trend,
        'val_loss_trend': val_loss_trend,
        'is_overfitting': final_loss_gap > 0.15 or final_acc_gap > 0.05
    }

def print_overfitting_analysis(model_name, epochs, train_losses, train_accs, val_losses, val_accs):
    """Print detailed overfitting analysis"""
    
    metrics = calculate_overfitting_metrics(train_losses, train_accs, val_losses, val_accs)
    
    print(f"\n{'='*70}")
    print(f"OVERFITTING ANALYSIS: {model_name.upper()}")
    print(f"{'='*70}")
    
    print(f"\nTotal Epochs: {len(epochs)}")
    print(f"\n{'Metric':<30} {'Train':<15} {'Val':<15} {'Gap':<15}")
    print("-" * 75)
    
    # Loss metrics
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    print(f"{'Final Loss':<30} {final_train_loss:<15.4f} {final_val_loss:<15.4f} {metrics['final_loss_gap']:<15.4f}")
    
    # Accuracy metrics
    final_train_acc = train_accs[-1]
    final_val_acc = val_accs[-1]
    print(f"{'Final Accuracy':<30} {final_train_acc:<15.4f} {final_val_acc:<15.4f} {metrics['final_acc_gap']:<15.4f}")
    
    print(f"\n{'Trend Analysis':<30} {'Train':<15} {'Val':<15}")
    print("-" * 60)
    print(f"{'Loss Trend (last 10 ep)':<30} {metrics['train_loss_trend']:<15.4f} {metrics['val_loss_trend']:<15.4f}")
    
    # Overfitting diagnosis
    print(f"\n{'OVERFITTING DIAGNOSIS':<70}")
    print("-" * 70)
    
    if metrics['final_loss_gap'] > 0.15:
        print(f"⚠️  HIGH LOSS GAP: {metrics['final_loss_gap']:.4f}")
        print("   → Validation loss significantly higher than training loss")
        print("   → Model may be overfitting")
    elif metrics['final_loss_gap'] > 0.05:
        print(f"⚡ MODERATE LOSS GAP: {metrics['final_loss_gap']:.4f}")
        print("   → Some overfitting detected, but acceptable")
    else:
        print(f"✓ LOW LOSS GAP: {metrics['final_loss_gap']:.4f}")
        print("   → Good generalization, minimal overfitting")
    
    print()
    
    if metrics['final_acc_gap'] > 0.10:
        print(f"⚠️  HIGH ACCURACY GAP: {metrics['final_acc_gap']:.4f}")
        print("   → Model performs significantly better on train than validation")
    elif metrics['final_acc_gap'] > 0.05:
        print(f"⚡ MODERATE ACCURACY GAP: {metrics['final_acc_gap']:.4f}")
        print("   → Some overfitting, model biased toward training data")
    else:
        print(f"✓ LOW ACCURACY GAP: {metrics['final_acc_gap']:.4f}")
        print("   → Good generalization")
    
    print()
    
    if metrics['val_loss_trend'] > 0.01 and metrics['train_loss_trend'] < -0.05:
        print("⚠️  INCREASING VALIDATION LOSS:")
        print("   → Classic overfitting pattern: train loss continues to decrease")
        print("   → while validation loss starts increasing")
    elif metrics['val_loss_trend'] < metrics['train_loss_trend']:
        print("✓ BOTH LOSSES DECREASING:")
        print("   → Good learning pattern, both train and validation improving")
    
    print("\n" + "="*70)
    
    return metrics

# Analyze all three models
models = [
    ('resnet50', 'results/resnet50_training_log.txt'),
    ('efficientnetb0', 'results/efficientnet_b0_training_log.txt'),
    ('inceptionv3', 'results/inceptionv3_training_log.txt')
]

all_metrics = {}

for model_name, log_file in models:
    try:
        epochs, train_losses, train_accs, val_losses, val_accs = parse_log_file(log_file)
        metrics = print_overfitting_analysis(model_name, epochs, train_losses, train_accs, val_losses, val_accs)
        all_metrics[model_name] = {
            'epochs': epochs,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'metrics': metrics
        }
    except Exception as e:
        print(f"Error processing {model_name}: {e}")

# Summary comparison
print(f"\n{'='*70}")
print("SUMMARY COMPARISON - OVERFITTING SCORES")
print(f"{'='*70}")
print(f"\n{'Model':<20} {'Loss Gap':<15} {'Acc Gap':<15} {'Status':<20}")
print("-" * 70)

for model_name, data in all_metrics.items():
    metrics = data['metrics']
    loss_gap = metrics['final_loss_gap']
    acc_gap = metrics['final_acc_gap']
    
    if loss_gap > 0.15 or acc_gap > 0.05:
        status = "⚠️ OVERFITTING"
    elif loss_gap > 0.05 or acc_gap > 0.02:
        status = "⚡ MODERATE"
    else:
        status = "✓ GOOD"
    
    print(f"{model_name:<20} {loss_gap:<15.4f} {acc_gap:<15.4f} {status:<20}")

print("\n" + "="*70)
