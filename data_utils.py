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
from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR, BATCH_SIZE, 
    DATA_AUGMENTATION_CONFIG, CLASS_NAMES
)


class SatelliteImageDataset(Dataset):
    """Custom PyTorch Dataset for satellite images"""
    
    def __init__(self, root_dir, class_names, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images organized by class
            class_names (list): List of class names
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.class_names = class_names
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        self.images = []
        self.labels = []
        
        # Load all image paths
        for class_name in class_names:
            class_path = os.path.join(root_dir, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Class directory not found: {class_path}")
                continue
                
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_path, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get a sample
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, label)
        """
        img_path = self.images[idx]
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


def get_data_loaders(img_size=(224, 224), batch_size=BATCH_SIZE):
    """
    Create PyTorch data loaders
    
    Args:
        img_size (tuple): Image size (height, width)
        batch_size (int): Batch size for data loaders
        
    Returns:
        tuple: (train_loader, val_loader, data_loading_time)
    """
    start_time = time.time()
    print(f"\nLoading data with image size {img_size}...")
    
    # Get transforms
    transforms_dict = get_transforms(img_size, augment=True)
    
    # Create datasets
    train_dataset = SatelliteImageDataset(
        TRAIN_DIR,
        CLASS_NAMES,
        transform=transforms_dict['train']
    )
    
    val_dataset = SatelliteImageDataset(
        VAL_DIR,
        CLASS_NAMES,
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


def get_test_loader(img_size=(224, 224), batch_size=BATCH_SIZE):
    """
    Create test data loader
    
    Args:
        img_size (tuple): Image size (height, width)
        batch_size (int): Batch size for data loader
        
    Returns:
        DataLoader: Test data loader
    """
    print(f"\nLoading test data with image size {img_size}...")
    
    # Get transforms (no augmentation for test)
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    test_dataset = SatelliteImageDataset(
        TEST_DIR,
        CLASS_NAMES,
        transform=val_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    return test_loader


def get_dataset_info():
    """Print dataset information"""
    print("=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)
    
    train_count = sum([len(os.listdir(os.path.join(TRAIN_DIR, c))) for c in CLASS_NAMES])
    val_count = sum([len(os.listdir(os.path.join(VAL_DIR, c))) for c in CLASS_NAMES])
    test_count = sum([len(os.listdir(os.path.join(TEST_DIR, c))) for c in CLASS_NAMES])
    
    print(f"\nNumber of Classes: {len(CLASS_NAMES)}")
    print(f"Classes: {', '.join(CLASS_NAMES[:5])}... (showing first 5 of {len(CLASS_NAMES)})")
    print(f"\nTraining samples: {train_count}")
    print(f"Validation samples: {val_count}")
    print(f"Test samples: {test_count}")
    print(f"Total samples: {train_count + val_count + test_count}")
    print("=" * 70)


if __name__ == "__main__":
    get_dataset_info()
    train_loader, val_loader = get_data_loaders()
    test_loader = get_test_loader()
    print(f"\nLoaders created successfully!")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
