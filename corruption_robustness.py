"""
Corruption Robustness Evaluation Module (Assignment 4.4)

Analyzes robustness under controlled distribution shifts by applying:
1. Pixel-level Gaussian noise (σ = 0.05, 0.1, 0.2)
2. Motion blur
3. Brightness shift (applied ONLY at evaluation time)
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
import logging
from tqdm import tqdm

from config import RESULTS_DIR


class CorruptionTransforms:
    """Corruption transformations for robustness evaluation"""
    
    @staticmethod
    def gaussian_noise(image, sigma):
        """
        Apply Gaussian noise to image
        
        Args:
            image: PIL Image or Tensor
            sigma (float): Standard deviation of Gaussian noise
            
        Returns:
            Tensor: Corrupted image
        """
        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        
        noise = torch.normal(0, sigma, image.shape)
        corrupted = torch.clamp(image + noise, 0, 1)
        return corrupted
    
    @staticmethod
    def motion_blur(image, kernel_size=15):
        """
        Apply motion blur to image
        
        Args:
            image: PIL Image or Tensor
            kernel_size (int): Kernel size for blur
            
        Returns:
            Tensor: Blurred image
        """
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL
            image = transforms.ToPILImage()(image)
        
        # Apply motion blur using PIL
        # Create motion blur kernel
        kernel = Image.new('L', (kernel_size, kernel_size))
        pixels = kernel.load()
        for i in range(kernel_size):
            pixels[i, kernel_size // 2] = 255 // kernel_size
        
        # Apply filter
        image = image.filter(ImageFilter.GaussianBlur(radius=3))
        
        return transforms.ToTensor()(image)
    
    @staticmethod
    def brightness_shift(image, factor=1.5):
        """
        Apply brightness shift to image
        
        Args:
            image: PIL Image or Tensor
            factor (float): Brightness factor (1.0 = original, <1.0 = darker, >1.0 = brighter)
            
        Returns:
            Tensor: Brightened/darkened image
        """
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        
        enhancer = ImageEnhance.Brightness(image)
        enhanced = enhancer.enhance(factor)
        
        return transforms.ToTensor()(enhanced)


class CorruptionEvaluator:
    """Evaluate model robustness under various corruptions"""
    
    def __init__(self, model, model_name, val_loader, device='cuda'):
        """
        Initialize corruption evaluator
        
        Args:
            model: Pre-trained model
            model_name (str): Model name
            val_loader: Validation data loader
            device: Computation device
        """
        self.model = model
        self.model_name = model_name
        self.val_loader = val_loader
        self.device = device
        self.results = {}
        
        # Create dynamic results directory: results_corruption_robustness_{model_name}
        model_clean = model_name.lower().replace(" ", "")
        results_folder_name = f"results_corruption_robustness_{model_clean}"
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(project_root, results_folder_name)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Setup logging
        self.log_file = os.path.join(self.results_dir, 'training_log.txt')
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger"""
        logger = logging.getLogger(f'{self.model_name}_CorruptionEval')
        logger.setLevel(logging.INFO)
        logger.handlers = []
        
        file_handler = logging.FileHandler(self.log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def evaluate_clean(self):
        """
        Evaluate model on clean (uncorrupted) images
        
        Returns:
            float: Accuracy on clean validation set
        """
        self.logger.info("\n[Evaluating on Clean Images]")
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Clean'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        clean_acc = correct / total if total > 0 else 0
        self.logger.info(f"  Clean Accuracy: {clean_acc:.4f}")
        
        return clean_acc
    
    def evaluate_gaussian_noise(self):
        """
        Evaluate robustness under Gaussian noise
        
        Returns:
            dict: Accuracy for each sigma value
        """
        self.logger.info("\n[Gaussian Noise Evaluation]")
        
        sigmas = [0.05, 0.10, 0.20]
        results = {}
        
        self.model.eval()
        
        for sigma in sigmas:
            correct = 0
            total = 0
            
            self.logger.info(f"  Evaluating σ={sigma}...")
            
            with torch.no_grad():
                for images, labels in tqdm(self.val_loader, desc=f'σ={sigma}', leave=False):
                    # Apply Gaussian noise
                    noisy_images = torch.zeros_like(images)
                    for i, img in enumerate(images):
                        noise = torch.normal(0, sigma, img.shape)
                        noisy_images[i] = torch.clamp(img + noise, 0, 1)
                    
                    noisy_images = noisy_images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(noisy_images)
                    _, predicted = torch.max(outputs, 1)
                    
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            
            acc = correct / total if total > 0 else 0
            results[sigma] = acc
            self.logger.info(f"    Accuracy: {acc:.4f}")
        
        return results
    
    def evaluate_motion_blur(self):
        """
        Evaluate robustness under motion blur
        
        Returns:
            float: Accuracy with motion blur
        """
        self.logger.info("\n[Motion Blur Evaluation]")
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Motion Blur'):
                # Apply motion blur
                blurred_images = torch.zeros_like(images)
                for i, img in enumerate(images):
                    # Convert to PIL, apply blur, convert back
                    pil_img = transforms.ToPILImage()(img.cpu())
                    blurred_pil = pil_img.filter(ImageFilter.GaussianBlur(radius=2))
                    blurred_images[i] = transforms.ToTensor()(blurred_pil)
                
                blurred_images = blurred_images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(blurred_images)
                _, predicted = torch.max(outputs, 1)
                
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        acc = correct / total if total > 0 else 0
        self.logger.info(f"  Accuracy: {acc:.4f}")
        
        return acc
    
    def evaluate_brightness_shift(self):
        """
        Evaluate robustness under brightness shifts
        
        Returns:
            dict: Accuracy for each brightness factor
        """
        self.logger.info("\n[Brightness Shift Evaluation]")
        
        factors = [0.5, 1.5]  # Darker and brighter
        results = {}
        
        self.model.eval()
        
        for factor in factors:
            correct = 0
            total = 0
            
            direction = 'darker' if factor < 1 else 'brighter'
            self.logger.info(f"  Evaluating brightness={factor} ({direction})...")
            
            with torch.no_grad():
                for images, labels in tqdm(self.val_loader, desc=f'brightness={factor}', leave=False):
                    # Apply brightness shift
                    bright_images = torch.zeros_like(images)
                    for i, img in enumerate(images):
                        pil_img = transforms.ToPILImage()(img.cpu())
                        enhancer = ImageEnhance.Brightness(pil_img)
                        enhanced = enhancer.enhance(factor)
                        bright_images[i] = transforms.ToTensor()(enhanced)
                    
                    bright_images = bright_images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(bright_images)
                    _, predicted = torch.max(outputs, 1)
                    
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            
            acc = correct / total if total > 0 else 0
            results[factor] = acc
            self.logger.info(f"    Accuracy: {acc:.4f}")
        
        return results
    
    def evaluate_all_corruptions(self):
        """
        Evaluate model under all corruption types
        
        Returns:
            dict: Results for all corruption types
        """
        self.logger.info("\n" + "="*70)
        self.logger.info(f"CORRUPTION ROBUSTNESS EVALUATION: {self.model_name}")
        self.logger.info("="*70)
        
        results = {
            'clean': self.evaluate_clean(),
            'gaussian_noise': self.evaluate_gaussian_noise(),
            'motion_blur': self.evaluate_motion_blur(),
            'brightness': self.evaluate_brightness_shift(),
        }
        
        self.results = results
        
        # Calculate and log corruption metrics
        self._log_metrics(results)
        
        return results
    
    def _log_metrics(self, results):
        """
        Log corruption error and relative robustness metrics
        
        Args:
            results (dict): Corruption evaluation results
        """
        clean_acc = results['clean']
        
        self.logger.info("\n" + "="*70)
        self.logger.info("CORRUPTION METRICS")
        self.logger.info("="*70)
        
        self.logger.info("\nGaussian Noise:")
        for sigma, acc in results['gaussian_noise'].items():
            corruption_error = 1 - acc
            relative_robustness = acc / clean_acc if clean_acc > 0 else 0
            self.logger.info(f"  σ={sigma}:")
            self.logger.info(f"    Accuracy:           {acc:.4f}")
            self.logger.info(f"    Corruption Error:   {corruption_error:.4f}")
            self.logger.info(f"    Relative Robustness: {relative_robustness:.4f}")
        
        self.logger.info("\nMotion Blur:")
        mb_acc = results['motion_blur']
        corruption_error = 1 - mb_acc
        relative_robustness = mb_acc / clean_acc if clean_acc > 0 else 0
        self.logger.info(f"  Accuracy:            {mb_acc:.4f}")
        self.logger.info(f"  Corruption Error:    {corruption_error:.4f}")
        self.logger.info(f"  Relative Robustness: {relative_robustness:.4f}")
        
        self.logger.info("\nBrightness Shift:")
        for factor, acc in results['brightness'].items():
            direction = 'Darker' if factor < 1 else 'Brighter'
            corruption_error = 1 - acc
            relative_robustness = acc / clean_acc if clean_acc > 0 else 0
            self.logger.info(f"  {direction} (factor={factor}):")
            self.logger.info(f"    Accuracy:            {acc:.4f}")
            self.logger.info(f"    Corruption Error:    {corruption_error:.4f}")
            self.logger.info(f"    Relative Robustness: {relative_robustness:.4f}")
    
    def plot_results(self):
        """Plot corruption robustness results"""
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Gaussian noise
        sigmas = sorted(self.results['gaussian_noise'].keys())
        noise_accs = [self.results['gaussian_noise'][s] for s in sigmas]
        axes[0, 0].plot(sigmas, noise_accs, 'o-', linewidth=2, markersize=8, color='steelblue')
        axes[0, 0].axhline(self.results['clean'], color='red', linestyle='--', label='Clean', linewidth=2)
        axes[0, 0].set_xlabel('Gaussian Noise (σ)')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title(f'{self.model_name}: Gaussian Noise Robustness')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Motion blur
        blur_acc = self.results['motion_blur']
        axes[0, 1].bar(['Clean', 'Motion Blur'], 
                      [self.results['clean'], blur_acc],
                      color=['green', 'orange'], alpha=0.7)
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title(f'{self.model_name}: Motion Blur Robustness')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Brightness shift
        factors = sorted(self.results['brightness'].keys())
        brightness_accs = [self.results['brightness'][f] for f in factors]
        labels = [f'Factor {f}' for f in factors]
        axes[1, 0].bar(labels, brightness_accs, color='purple', alpha=0.7)
        axes[1, 0].axhline(self.results['clean'], color='red', linestyle='--', label='Clean', linewidth=2)
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title(f'{self.model_name}: Brightness Shift Robustness')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Summary comparison
        all_accuracies = [self.results['clean']]
        all_labels = ['Clean']
        
        # Average noise accuracy
        avg_noise = np.mean(list(self.results['gaussian_noise'].values()))
        all_accuracies.append(avg_noise)
        all_labels.append('Avg Noise')
        
        # Blur
        all_accuracies.append(self.results['motion_blur'])
        all_labels.append('Motion Blur')
        
        # Average brightness
        avg_brightness = np.mean(list(self.results['brightness'].values()))
        all_accuracies.append(avg_brightness)
        all_labels.append('Avg Brightness')
        
        colors = ['green', 'orange', 'blue', 'purple']
        axes[1, 1].bar(all_labels, all_accuracies, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title(f'{self.model_name}: Robustness Summary')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"\n✓ Plot saved: {plot_path}")
        plt.close()


def evaluate_corruption_robustness(model, model_name, val_loader, device='cuda'):
    """
    Run complete corruption robustness evaluation
    
    Args:
        model: Pre-trained model
        model_name (str): Model name
        val_loader: Validation data loader
        device: Computation device
        
    Returns:
        dict: Robustness evaluation results
    """
    evaluator = CorruptionEvaluator(model, model_name, val_loader, device)
    results = evaluator.evaluate_all_corruptions()
    evaluator.plot_results()
    return results
