"""
PyTorch data loading and preprocessing utilities
Handles image loading, augmentation, and dataset creation
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import time
from pathlib import Path
from config import (
    BATCH_SIZE, DATA_AUGMENTATION_CONFIG, CLASS_NAMES, RANDOM_SEED
)


class CustomSplitDataset(Dataset):
    """Custom PyTorch Dataset for pre-split image lists"""
    
    def __init__(self, image_paths, labels, class_names, transform=None):
        """
        Args:
            image_paths (list): List of image file paths
            labels (list): List of class labels
            class_names (list): List of class names
            transform (callable, optional): Optional transform to be applied on images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a sample
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(img_size=(224, 224), augment=True):
    """
    Get data transformations
    
    Args:
        img_size (tuple): Image size (height, width)
        augment (bool): Whether to apply augmentation
        
    Returns:
        dict: Dictionary with 'train' and 'val' transforms
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(DATA_AUGMENTATION_CONFIG['rotation_range']),
            transforms.RandomAffine(
                degrees=0,
                translate=(DATA_AUGMENTATION_CONFIG['width_shift_range'], 
                          DATA_AUGMENTATION_CONFIG['height_shift_range']),
                shear=int(DATA_AUGMENTATION_CONFIG['shear_range'] * 30),
                scale=(1 - DATA_AUGMENTATION_CONFIG['zoom_range'], 
                      1 + DATA_AUGMENTATION_CONFIG['zoom_range'])
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return {'train': train_transform, 'val': val_transform}


def get_data_loaders(img_size=(224, 224), batch_size=BATCH_SIZE, train_split=0.9):
    """
    Create PyTorch data loaders with train/val split from original data
    
    Args:
        img_size (tuple): Image size (height, width)
        batch_size (int): Batch size for data loaders
        train_split (float): Proportion of data for training (0-1), rest for validation
        
    Returns:
        tuple: (train_loader, val_loader, data_loading_time)
    """
    start_time = time.time()
    print(f"\nLoading data with image size {img_size}...")
    print(f"Train/Val split: {train_split*100:.0f}% / {(1-train_split)*100:.0f}%")
    
    # Get transforms
    transforms_dict = get_transforms(img_size, augment=True)
    
    # Load all data from original train_data directory
    data_root = Path('data/train_data')
    
    # Collect all images with their paths and labels
    all_images = []
    all_labels = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_path = data_root / class_name
        if not class_path.exists():
            print(f"Warning: Class directory not found: {class_path}")
            continue
        
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = str(class_path / img_name)
                all_images.append(img_path)
                all_labels.append(class_idx)
    
    # Set random seed for reproducible split
    np.random.seed(RANDOM_SEED)
    indices = np.arange(len(all_images))
    np.random.shuffle(indices)
    
    # Split indices
    split_idx = int(len(indices) * train_split)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create train and validation datasets
    train_images = [all_images[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    
    val_images = [all_images[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]
    
    # Create custom datasets for train and val
    train_dataset = CustomSplitDataset(
        train_images, train_labels, CLASS_NAMES, 
        transform=transforms_dict['train']
    )
    
    val_dataset = CustomSplitDataset(
        val_images, val_labels, CLASS_NAMES,
        transform=transforms_dict['val']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    end_time = time.time()
    loading_time = end_time - start_time
    
    info_msg = (f"Training samples: {len(train_dataset)}\n"
                f"Validation samples: {len(val_dataset)}\n"
                f"Data Loading Time: {loading_time:.2f}s")
    print(info_msg)
    
    return train_loader, val_loader, loading_time



def get_dataset_info():
    """Print dataset information"""
    print("=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)
    
    data_root = Path('data/train_data')
    total_count = 0
    
    for class_name in CLASS_NAMES:
        class_path = data_root / class_name
        if class_path.exists():
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            total_count += count
    
    train_count = int(total_count * 0.9)
    val_count = total_count - train_count
    
    print(f"\nNumber of Classes: {len(CLASS_NAMES)}")
    print(f"Classes: {', '.join(CLASS_NAMES[:5])}... (showing first 5 of {len(CLASS_NAMES)})")
    print(f"\nTraining samples (90%): {train_count}")
    print(f"Validation samples (10%): {val_count}")
    print(f"Total samples: {total_count}")
    print("=" * 70)


if __name__ == "__main__":
    get_dataset_info()
    train_loader, val_loader, _ = get_data_loaders()
    print(f"\nLoaders created successfully!")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
