"""
Flexible MNIST Catastrophic Forgetting Framework

This module provides flexible tools for studying catastrophic forgetting
with complete control over digit subsets and training phases.

Usage:
    from mnist_catastrophic_forgetting import MNISTModel, FlexibleTrainer, DigitFilter
    
    # Create model
    model = MNISTModel()
    trainer = FlexibleTrainer(model, data_loader)
    
    # Phase 1: Train on custom subset
    phase1_results = trainer.train_on_digits([0,2,4,6,8], epochs=50)
    
    # Phase 2: Train on different subset and monitor both
    phase2_results = trainer.train_on_digits(
        [1,3,5,7,9], 
        epochs=30,
        monitor_digits=[0,2,4,6,8]  # Monitor forgetting
    )
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import copy

class MNISTModelNN(nn.Module):
    """
    Flexible feedforward neural network for MNIST digit classification.
    Automatically adapts to any architecture you specify.
    """
    
    def __init__(self, hidden_layers: List[int] = [2000, 1500, 1000, 500], dropout_rate: float = 0.3):
        """
        Initialize model with flexible architecture.
        
        Args:
            hidden_layers: List of hidden layer sizes (e.g., [2000, 1500, 1000, 500])
            dropout_rate: Dropout probability for regularization
        """
        super(MNISTModelNN, self).__init__()
        
        # Store architecture info
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.layer_sizes = [784] + hidden_layers + [10]  # Input + hidden + output
        
        # Create layers dynamically
        self.layers = nn.ModuleList()
        for i in range(len(self.layer_sizes) - 1):
            self.layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._init_weights()
        
        # Print architecture automatically
        self.print_architecture()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass through all layers dynamically."""
        # Flatten input if needed
        x = x.view(x.size(0), -1)
        
        # Pass through all layers except the last one
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final layer (no activation, no dropout)
        x = self.layers[-1](x)
        
        return x
    
    def get_layer_outputs(self, x):
        """Get outputs from all intermediate layers for analysis."""
        x = x.view(x.size(0), -1)
        
        outputs = {}
        outputs['input'] = x.clone()
        
        # Pass through all layers and store outputs
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
            outputs[f'layer{i+1}'] = x.clone()
        
        # Final layer
        x = self.layers[-1](x)
        outputs['output'] = x
        
        return outputs

    def get_model_summary(self):
        """Get architecture summary as dictionary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Dynamic architecture string
        arch_str = '-'.join(map(str, self.layer_sizes))
        
        # Dynamic layer info
        layer_info = [{'name': 'Input', 'size': 784, 'activation': None}]
        
        for i, size in enumerate(self.hidden_layers):
            layer_info.append({
                'name': f'Hidden{i+1}', 
                'size': size, 
                'activation': 'ReLU'
            })
        
        layer_info.append({'name': 'Output', 'size': 10, 'activation': None})
        
        return {
            'architecture': arch_str,
            'layers': layer_info,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024**2),
            'dropout_rate': self.dropout_rate,
            'num_hidden_layers': len(self.hidden_layers)
        }
    
    def print_architecture(self):
        """Print model architecture summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("\n" + "="*70)
        print("MODEL ARCHITECTURE")
        print("="*70)
        print(f"Input Layer:    784 (28x28 flattened)")
        
        for i, size in enumerate(self.hidden_layers):
            print(f"Hidden Layer {i+1}: {size:4d} neurons (ReLU + Dropout)")
        
        print(f"Output Layer:    10 neurons (no activation)")
        print("-"*70)
        print(f"Architecture: {'-'.join(map(str, self.layer_sizes))}")
        print(f"Total Layers: {len(self.layer_sizes)} ({len(self.hidden_layers)} hidden)")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: ~{total_params * 4 / (1024**2):.2f} MB")
        print(f"Dropout rate: {self.dropout_rate}")
        print("="*70)


class DigitFilter:
    """Utility class to filter datasets by digit groups."""
    
    @staticmethod
    def filter_by_digits(dataset, target_digits: List[int]) -> Subset:
        """
        Filter dataset to include only specified digits.
        
        Args:
            dataset: PyTorch dataset
            target_digits: List of digits to include (e.g., [0,2,4,6,8])
        
        Returns:
            Subset containing only samples with target digits
        """
        indices = []
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if label in target_digits:
                indices.append(idx)
        
        return Subset(dataset, indices)
    
    @staticmethod
    def get_even_digits() -> List[int]:
        """Get list of even digits."""
        return [0, 2, 4, 6, 8]
    
    @staticmethod
    def get_odd_digits() -> List[int]:
        """Get list of odd digits."""
        return [1, 3, 5, 7, 9]
    
    @staticmethod
    def get_all_digits() -> List[int]:
        """Get list of all digits."""
        return list(range(10))


class FlexibleTrainer:
    """
    Flexible trainer that allows training on custom digit subsets and monitoring 
    performance on multiple digit groups simultaneously.
    """
    
    def __init__(self, model: MNISTModelNN, data_loader, device: str = None):
        self.model = model
        self.data_loader = data_loader
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Trainer initialized on device: {self.device}")
    
    def create_digit_loaders(self, digits: List[int], batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test loaders for specific digits.
        
        Args:
            digits: List of digits to include
            batch_size: Batch size for data loaders
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Filter datasets
        train_filtered = DigitFilter.filter_by_digits(self.data_loader.train_subset, digits)
        val_filtered = DigitFilter.filter_by_digits(self.data_loader.val_subset, digits)
        test_filtered = DigitFilter.filter_by_digits(self.data_loader.test_dataset, digits)
        
        # Create loaders
        train_loader = DataLoader(train_filtered, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_filtered, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_filtered, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, val_loader, test_loader
    
    def evaluate_on_digits(self, digits: List[int], split: str = 'test') -> Dict[str, float]:
        """
        Evaluate model performance on specific digits.
        
        Args:
            digits: List of digits to evaluate on
            split: 'train', 'val', or 'test'
            
        Returns:
            Dictionary with loss and accuracy
        """
        if split == 'test':
            base_dataset = self.data_loader.test_dataset
        elif split == 'val':
            base_dataset = self.data_loader.val_subset
        else:
            base_dataset = self.data_loader.train_subset
        
        filtered_dataset = DigitFilter.filter_by_digits(base_dataset, digits)
        
        if len(filtered_dataset) == 0:
            return {'loss': float('inf'), 'accuracy': 0.0}
        
        loader = DataLoader(filtered_dataset, batch_size=64, shuffle=False)
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(loader)
        accuracy = 100 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train_epoch(self, train_loader: DataLoader, criterion, optimizer) -> Dict[str, float]:
        """Train for one epoch and return metrics."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train_on_digits(
        self,
        training_digits: List[int],
        epochs: int = 50,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        monitor_digits: Optional[List[int]] = None,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Train model on specific digits with optional monitoring of other digits.
        
        Args:
            training_digits: List of digits to train on
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            monitor_digits: Optional list of digits to monitor during training (for forgetting analysis)
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress
            
        Returns:
            Dictionary with complete training history
        """
        if verbose:
            print(f"\nTraining on digits: {training_digits}")
            if monitor_digits:
                print(f"Monitoring digits: {monitor_digits}")
        
        # Create data loaders for training digits
        train_loader, val_loader, _ = self.create_digit_loaders(training_digits, batch_size)
        
        if verbose:
            print(f"Training samples: {len(train_loader.dataset):,}")
            print(f"Validation samples: {len(val_loader.dataset):,}")
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        history = {
            'training_digits': training_digits,
            'monitor_digits': monitor_digits,
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
        }
        
        # Add monitoring history if specified
        if monitor_digits:
            history['monitor_train_loss'] = []
            history['monitor_train_accuracy'] = []
            history['monitor_val_loss'] = []
            history['monitor_val_accuracy'] = []
            history['monitor_test_loss'] = []
            history['monitor_test_accuracy'] = []
        
        # Early stopping variables
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None
        
        if verbose:
            print(f"\nStarting training for {epochs} epochs...")
            print("-" * 80)
        
        for epoch in range(epochs):
            # Train one epoch
            train_metrics = self.train_epoch(train_loader, criterion, optimizer)
            
            # Evaluate on validation set (same digits as training)
            val_metrics = self.evaluate_on_digits(training_digits, split='val')
            
            # Store training metrics
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Monitor other digits if specified
            if monitor_digits:
                monitor_train = self.evaluate_on_digits(monitor_digits, split='train')
                monitor_val = self.evaluate_on_digits(monitor_digits, split='val')
                monitor_test = self.evaluate_on_digits(monitor_digits, split='test')
                
                history['monitor_train_loss'].append(monitor_train['loss'])
                history['monitor_train_accuracy'].append(monitor_train['accuracy'])
                history['monitor_val_loss'].append(monitor_val['loss'])
                history['monitor_val_accuracy'].append(monitor_val['accuracy'])
                history['monitor_test_loss'].append(monitor_test['loss'])
                history['monitor_test_accuracy'].append(monitor_test['accuracy'])
            
            # Print progress
            if verbose:
                progress_str = f"Epoch {epoch+1:3d}: "
                progress_str += f"Train Acc: {train_metrics['accuracy']:6.2f}% | "
                progress_str += f"Val Acc: {val_metrics['accuracy']:6.2f}% | "
                progress_str += f"Train Loss: {train_metrics['loss']:.4f} | "
                progress_str += f"Val Loss: {val_metrics['loss']:.4f}"
                
                if monitor_digits:
                    monitor_acc = history['monitor_test_accuracy'][-1]
                    progress_str += f" | Monitor Acc: {monitor_acc:6.2f}%"
                
                print(progress_str)
            
            # Early stopping
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Add final metrics
        history['epochs_trained'] = len(history['train_loss'])
        history['best_val_accuracy'] = best_val_acc
        
        if verbose:
            print(f"\nTraining completed!")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            if monitor_digits:
                final_monitor_acc = history['monitor_test_accuracy'][-1]
                print(f"Final monitor accuracy: {final_monitor_acc:.2f}%")
        
        return history
    
    def get_performance_summary(self, digits_groups: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
        """
        Get performance summary on multiple digit groups.
        
        Args:
            digits_groups: Dictionary mapping group names to digit lists
                          e.g., {'even': [0,2,4,6,8], 'odd': [1,3,5,7,9]}
        
        Returns:
            Dictionary with performance metrics for each group
        """
        summary = {}
        
        for group_name, digits in digits_groups.items():
            train_perf = self.evaluate_on_digits(digits, split='train')
            val_perf = self.evaluate_on_digits(digits, split='val')
            test_perf = self.evaluate_on_digits(digits, split='test')
            
            summary[group_name] = {
                'train_accuracy': train_perf['accuracy'],
                'val_accuracy': val_perf['accuracy'],
                'test_accuracy': test_perf['accuracy'],
                'train_loss': train_perf['loss'],
                'val_loss': val_perf['loss'],
                'test_loss': test_perf['loss']
            }
        
        return summary
    
    def save_model(self, filepath: str):
        """Save model state."""
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model state."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Model loaded from {filepath}")


# Utility functions for common use cases
def print_performance_summary(summary: Dict[str, Dict[str, float]]):
    """Pretty print performance summary."""
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    for group_name, metrics in summary.items():
        print(f"\n{group_name.upper()} DIGITS:")
        print(f"  Train:      {metrics['train_accuracy']:6.2f}% (loss: {metrics['train_loss']:.4f})")
        print(f"  Validation: {metrics['val_accuracy']:6.2f}% (loss: {metrics['val_loss']:.4f})")
        print(f"  Test:       {metrics['test_accuracy']:6.2f}% (loss: {metrics['test_loss']:.4f})")


def calculate_forgetting(history_before: Dict, history_after: Dict, monitor_key: str = 'monitor_test_accuracy') -> float:
    """
    Calculate catastrophic forgetting between two training phases.
    
    Args:
        history_before: Training history from phase 1
        history_after: Training history from phase 2 (with monitoring)
        monitor_key: Key to use for forgetting calculation
        
    Returns:
        Forgetting amount (positive value indicates forgetting)
    """
    if monitor_key not in history_after:
        raise ValueError(f"Monitor key '{monitor_key}' not found in history_after")
    
    accuracy_before = history_before['best_val_accuracy']
    accuracy_after = history_after[monitor_key][-1]  # Final monitored accuracy
    
    return accuracy_before - accuracy_after


# Example usage
if __name__ == "__main__":
    model = MNISTModelNN([1024, 512], dropout_rate=0.5)


"""
    # Import the data loader from previous module
    from mnist_data_prep import MNISTDataLoader
    
    # Initialize
    data_loader = MNISTDataLoader(batch_size=64, validation_split=0.2)
    model = MNISTModel(dropout_rate=0.3)
    trainer = FlexibleTrainer(model, data_loader)
    
    # Phase 1: Train on even digits
    print("PHASE 1: Training on even digits")
    even_digits = [0, 2, 4, 6, 8]
    phase1_history = trainer.train_on_digits(
        training_digits=even_digits,
        epochs=30,
        learning_rate=0.001
    )
    
    # Check performance after phase 1
    summary_after_phase1 = trainer.get_performance_summary({
        'even': even_digits,
        'odd': [1, 3, 5, 7, 9],
        'all': list(range(10))
    })
    print_performance_summary(summary_after_phase1)
    
    # Phase 2: Train on odd digits while monitoring even digits
    print("\n\nPHASE 2: Training on odd digits (monitoring even digits)")
    odd_digits = [1, 3, 5, 7, 9]
    phase2_history = trainer.train_on_digits(
        training_digits=odd_digits,
        epochs=30,
        learning_rate=0.001,
        monitor_digits=even_digits  # Monitor forgetting
    )
    
    # Final performance summary
    summary_after_phase2 = trainer.get_performance_summary({
        'even': even_digits,
        'odd': odd_digits,
        'all': list(range(10))
    })
    print_performance_summary(summary_after_phase2)
    
    # Calculate forgetting
    forgetting = calculate_forgetting(phase1_history, phase2_history)
    print(f"\nCatastrophic Forgetting: {forgetting:.2f}%")
"""