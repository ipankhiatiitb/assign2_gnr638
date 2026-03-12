"""
Create 90/10 train/validation splits with few-shot support
Supports 100%, 20%, and 5% data regimes
"""

import os
import shutil
import numpy as np
from pathlib import Path
from config import CLASS_NAMES, RANDOM_SEED

def create_train_val_split(source_dir='data/train_data', 
                           target_dir='data/split_data',
                           train_split=0.9,
                           few_shot_percentage=100):
    """
    Create train/val split from source directory with few-shot support
    
    Args:
        source_dir (str): Source directory with original data
        target_dir (str): Target directory for split data
        train_split (float): Proportion for training (0-1)
        few_shot_percentage (int): Percentage of training data to use (100, 20, or 5)
    """
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create base directories
    if few_shot_percentage == 100:
        split_name = 'split_data'
    else:
        split_name = f'split_data_fewshot_{few_shot_percentage}pct'
    
    target_path = Path(target_dir).parent / split_name
    train_dir = target_path / 'train'
    val_dir = target_path / 'val'
    
    print(f"\n{'='*70}")
    print(f"Creating {few_shot_percentage}% Few-Shot Split (90/10 train/val)")
    print(f"{'='*70}")
    print(f"Creating split directories...")
    
    for class_name in CLASS_NAMES:
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    # Split data with few-shot support
    total_files = 0
    train_files = 0
    val_files = 0
    
    for class_name in CLASS_NAMES:
        source_class_dir = source_path / class_name
        
        if not source_class_dir.exists():
            print(f"Warning: {source_class_dir} not found, skipping...")
            continue
        
        # Get all image files
        image_files = [f for f in os.listdir(source_class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        # Shuffle with fixed seed
        indices = np.arange(len(image_files))
        np.random.shuffle(indices)
        
        # First split: 90/10 for train/val
        split_idx = int(len(indices) * train_split)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # Second split: few-shot percentage on training data
        if few_shot_percentage < 100:
            few_shot_idx = int(len(train_indices) * (few_shot_percentage / 100.0))
            train_indices = train_indices[:few_shot_idx]
        
        # Copy train files
        for idx in train_indices:
            src_file = source_class_dir / image_files[idx]
            dst_file = train_dir / class_name / image_files[idx]
            shutil.copy2(src_file, dst_file)
            train_files += 1
        
        # Copy val files (always use full validation set)
        for idx in val_indices:
            src_file = source_class_dir / image_files[idx]
            dst_file = val_dir / class_name / image_files[idx]
            shutil.copy2(src_file, dst_file)
            val_files += 1
        
        total_files += len(image_files)
        print(f"  {class_name}: {len(image_files)} total → {len(train_indices)} train ({few_shot_percentage}%), {len(val_indices)} val")
    
    print("\n" + "=" * 70)
    print(f"SPLIT CREATION COMPLETE ({few_shot_percentage}%)")
    print("=" * 70)
    print(f"Total samples in source: {total_files}")
    print(f"Training samples ({few_shot_percentage}% of 90%): {train_files}")
    print(f"Validation samples (10%): {val_files}")
    print(f"\nTrain split directory: {train_dir}")
    print(f"Val split directory: {val_dir}")
    print("=" * 70)
    
    return train_dir, val_dir

if __name__ == "__main__":
    # Create splits for all few-shot percentages
    for pct in [100, 20, 5]:
        create_train_val_split(
            source_dir='data/train_data',
            target_dir='data/split_data',
            train_split=0.9,
            few_shot_percentage=pct
        )
