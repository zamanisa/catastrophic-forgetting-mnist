#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import copy

from models.ff_nn import MNISTModelNN
from utils.digit_filter import DigitFilter  
    
 #%%   
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
        
        #print(f"Trainer initialized on device: {self.device}")
    
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
    
    def evaluate_on_digits(self, digits: List[int], 
                           split: str = 'test',
                           criterion: Optional[nn.Module] = None) -> Dict[str, float]:
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
        if criterion is None:
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
        verbose: bool = True,
        checkpointer: Optional[object] = None,
        criterion=None,
        optimizer=None,
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
            checkpointer: Optional checkpointer object
            criterion: The loss function. If None, defaults to nn.CrossEntropyLoss().
            optimizer: The optimizer. If None, defaults to optim.Adam.
            
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
        
        # Setup training: Use provided criterion/optimizer or create defaults
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        if optimizer is None:
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
        # Training loop
        for epoch in range(epochs):
            # Train one epoch
            train_metrics = self.train_epoch(train_loader, criterion, optimizer)
            
            # Evaluate on validation set (same digits as training)
            val_metrics = self.evaluate_on_digits(training_digits, split='val', criterion=criterion)
            
            # Store training metrics
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Monitor other digits if specified
            if monitor_digits:
                monitor_train = self.evaluate_on_digits(monitor_digits, split='train', criterion=criterion)
                monitor_val = self.evaluate_on_digits(monitor_digits, split='val', criterion=criterion)
                monitor_test = self.evaluate_on_digits(monitor_digits, split='test', criterion=criterion)

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
            if checkpointer:
                checkpointer.maybe_save_epoch(
                    current_epoch=epoch + 1,
                    training_history=history,
                    training_config={
                        'training_digits': training_digits,
                        'epochs': epochs,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'monitor_digits': monitor_digits,
                        'early_stopping_patience': early_stopping_patience
                    },
                    model_info=self.model.get_model_summary() if hasattr(self.model, 'get_model_summary') else None
                )
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
    
    def get_performance_summary(self, 
                                digits_groups: Dict[str, List[int]],
                                criterion: Optional[nn.Module] = None) -> Dict[str, Dict[str, float]]:
        """
        Get performance summary on multiple digit groups.
        
        Args:
            digits_groups: Dictionary mapping group names to digit lists
                          e.g., {'even': [0,2,4,6,8], 'odd': [1,3,5,7,9]}
        
        Returns:
            Dictionary with performance metrics for each group
        """
        summary = {}
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        for group_name, digits in digits_groups.items():
            train_perf = self.evaluate_on_digits(digits, split='train', criterion=criterion)
            val_perf = self.evaluate_on_digits(digits, split='val', criterion=criterion)
            test_perf = self.evaluate_on_digits(digits, split='test', criterion=criterion)

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

