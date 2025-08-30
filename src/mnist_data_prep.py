"""
MNIST Data Preparation Module

This module provides a clean interface for loading and preprocessing MNIST data
using PyTorch and Hugging Face datasets.

Usage:
    from mnist_data_prep import MNISTDataLoader
    
    # Initialize data loader
    data_loader = MNISTDataLoader(batch_size=64, validation_split=0.2)
    
    # Get data loaders
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    
    # Get dataset info
    info = data_loader.get_dataset_info()
    print(f"Training samples: {info['train_size']}")
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from datasets import load_dataset
import numpy as np
from typing import Tuple, Dict, Optional


class HFMNISTDataset(torch.utils.data.Dataset):
    """Custom PyTorch Dataset for Hugging Face MNIST data."""
    
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image']
        label = item['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class MNISTDataLoader:
    """
    A comprehensive MNIST data loader with preprocessing and splitting capabilities.
    
    Args:
        batch_size (int): Batch size for data loaders
        validation_split (float): Fraction of training data to use for validation
        normalize (bool): Whether to normalize data with MNIST mean/std
        random_seed (int): Random seed for reproducible splits
        num_workers (int): Number of workers for data loading
    """
    
    def __init__(
        self, 
        batch_size: int = 64,
        validation_split: float = 0.2,
        normalize: bool = True,
        random_seed: int = 42,
        num_workers: int = 2
    ):
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.normalize = normalize
        self.random_seed = random_seed
        self.num_workers = num_workers
        
        # MNIST statistics
        self.mnist_mean = 0.1307
        self.mnist_std = 0.3081
        
        # Initialize datasets
        self._load_dataset()
        self._create_transforms()
        self._prepare_datasets()
    
    def _load_dataset(self):
        """Load MNIST dataset from Hugging Face."""
        print("Loading MNIST dataset from Hugging Face...")
        self.raw_dataset = load_dataset("mnist")
        print(f"Train samples: {len(self.raw_dataset['train']):,}")
        print(f"Test samples: {len(self.raw_dataset['test']):,}")
    
    def _create_transforms(self):
        """Create data transformation pipeline."""
        transform_list = [transforms.ToTensor()]
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize((self.mnist_mean,), (self.mnist_std,))
            )
        
        self.transform = transforms.Compose(transform_list)
    
    def _prepare_datasets(self):
        """Prepare PyTorch datasets with transforms."""
        self.train_dataset = HFMNISTDataset(
            self.raw_dataset['train'], 
            transform=self.transform
        )
        self.test_dataset = HFMNISTDataset(
            self.raw_dataset['test'], 
            transform=self.transform
        )
        
        # Split training data into train and validation
        train_size = int((1 - self.validation_split) * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size
        
        generator = torch.Generator().manual_seed(self.random_seed)
        self.train_subset, self.val_subset = random_split(
            self.train_dataset, [train_size, val_size], generator=generator
        )
        
        print(f"Dataset splits:")
        print(f"  Training: {len(self.train_subset):,}")
        print(f"  Validation: {len(self.val_subset):,}")
        print(f"  Test: {len(self.test_dataset):,}")
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get PyTorch DataLoaders for training, validation, and testing.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_loader = DataLoader(
            self.train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            self.val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, val_loader, test_loader
    
    def get_dataset_info(self) -> Dict:
        """
        Get information about the loaded datasets.
        
        Returns:
            Dictionary with dataset statistics
        """
        # Get a sample batch to determine shapes
        sample_loader = DataLoader(self.train_subset, batch_size=1)
        sample_batch = next(iter(sample_loader))
        image_shape = sample_batch[0].shape
        
        return {
            'train_size': len(self.train_subset),
            'val_size': len(self.val_subset),
            'test_size': len(self.test_dataset),
            'total_size': len(self.train_subset) + len(self.val_subset) + len(self.test_dataset),
            'num_classes': 10,
            'image_shape': image_shape,
            'batch_size': self.batch_size,
            'normalized': self.normalize
        }
    
    def get_class_distribution(self, loader_type: str = 'train') -> Dict[int, int]:
        """
        Get class distribution for a specific data loader.
        
        Args:
            loader_type: 'train', 'val', or 'test'
            
        Returns:
            Dictionary mapping class labels to counts
        """
        if loader_type == 'train':
            dataset = self.train_subset
        elif loader_type == 'val':
            dataset = self.val_subset
        elif loader_type == 'test':
            dataset = self.test_dataset
        else:
            raise ValueError("loader_type must be 'train', 'val', or 'test'")
        
        class_counts = {}
        loader = DataLoader(dataset, batch_size=100, shuffle=False)
        
        for _, labels in loader:
            for label in labels:
                label_item = label.item()
                class_counts[label_item] = class_counts.get(label_item, 0) + 1
        
        return dict(sorted(class_counts.items()))
    
    def visualize_samples(self, num_samples: int = 10, loader_type: str = 'train'):
        """
        Visualize sample images from the dataset.
        
        Args:
            num_samples: Number of samples to display
            loader_type: 'train', 'val', or 'test'
        """
        import matplotlib.pyplot as plt
        
        if loader_type == 'train':
            dataset = self.train_subset
        elif loader_type == 'val':
            dataset = self.val_subset
        elif loader_type == 'test':
            dataset = self.test_dataset
        else:
            raise ValueError("loader_type must be 'train', 'val', or 'test'")
        
        loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
        images, labels = next(iter(loader))
        
        # Denormalize if normalized
        if self.normalize:
            images = images * self.mnist_std + self.mnist_mean
        
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        for i in range(min(num_samples, 10)):
            row, col = i // 5, i % 5
            img = images[i].squeeze().numpy()
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(f'Label: {labels[i].item()}')
            axes[row, col].axis('off')
        
        plt.suptitle(f'Sample Images from {loader_type.title()} Set')
        plt.tight_layout()
        plt.show()


# Convenience functions for quick usage
def get_mnist_loaders(batch_size: int = 64, validation_split: float = 0.2) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Quick function to get MNIST data loaders with default settings.
    
    Args:
        batch_size: Batch size for data loaders
        validation_split: Fraction of training data for validation
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_loader = MNISTDataLoader(batch_size=batch_size, validation_split=validation_split)
    return data_loader.get_data_loaders()


def print_dataset_summary(data_loader: MNISTDataLoader):
    """Print a comprehensive summary of the dataset."""
    info = data_loader.get_dataset_info()
    
    print("=" * 50)
    print("MNIST Dataset Summary")
    print("=" * 50)
    print(f"Total samples: {info['total_size']:,}")
    print(f"Training samples: {info['train_size']:,}")
    print(f"Validation samples: {info['val_size']:,}")
    print(f"Test samples: {info['test_size']:,}")
    print(f"Number of classes: {info['num_classes']}")
    print(f"Image shape: {info['image_shape']}")
    print(f"Batch size: {info['batch_size']}")
    print(f"Normalized: {info['normalized']}")
    
    # Print class distributions
    for split in ['train', 'val', 'test']:
        dist = data_loader.get_class_distribution(split)
        print(f"\n{split.title()} class distribution:")
        for digit, count in dist.items():
            total = sum(dist.values())
            print(f"  Digit {digit}: {count:,} ({count/total*100:.1f}%)")


# Example usage
if __name__ == "__main__":
    # Method 1: Using the class
    print("Method 1: Using MNISTDataLoader class")
    data_loader = MNISTDataLoader(batch_size=64, validation_split=0.2)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    
    # Print summary
    print_dataset_summary(data_loader)
    
    # Visualize samples
    data_loader.visualize_samples(num_samples=10, loader_type='train')
    
    print("\n" + "="*50)
    
    # Method 2: Using convenience function
    print("Method 2: Using convenience function")
    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=32)
    
    # Test a batch
    sample_batch = next(iter(train_loader))
    images, labels = sample_batch
    print(f"Sample batch shape: {images.shape}")
    print(f"Sample labels: {labels[:5].tolist()}")