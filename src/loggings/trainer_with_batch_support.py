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
from loggings.trainer import FlexibleTrainer

class FlexibleTrainerWithBatch(FlexibleTrainer):
    """
    Extended FlexibleTrainer with batch-based training support.
    Inherits all original functionality and adds batch-level training methods.
    """
    
    def __init__(self, model: MNISTModelNN, data_loader, device: str = None):
        if FlexibleTrainer != object:
            super().__init__(model, data_loader, device)
        else:
            # Fallback initialization if parent import failed
            self.model = model
            self.data_loader = data_loader
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            print(f"Trainer initialized on device: {self.device}")
    
    def train_on_digits_batch_based(
        self,
        training_digits: List[int],
        total_batches: int = 1000,
        log_every_n_batches: int = 100,
        val_every_n_batches: int = 200,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        monitor_digits: Optional[List[int]] = None,
        early_stopping: bool = True,
        early_stopping_patience: int = 5,
        verbose: bool = True,
        checkpointer: Optional[object] = None,
        criterion=None, 
        optimizer=None  
    ) -> Dict:
        """
        Train model on specific digits with batch-based logging and evaluation.
        
        Args:
            training_digits: List of digits to train on
            total_batches: Total number of batches to train
            log_every_n_batches: Log training metrics every N batches
            val_every_n_batches: Evaluate on validation/test every N batches
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            monitor_digits: Optional list of digits to monitor for catastrophic forgetting
            early_stopping: Whether to use early stopping
            early_stopping_patience: Patience for early stopping (in validation evaluations)
            verbose: Whether to print progress
            checkpointer: Optional checkpointer object for saving model states
            criterion: The loss function. If None, defaults to nn.CrossEntropyLoss().
            optimizer: The optimizer. If None, defaults to optim.Adam.
        Returns:
            Dictionary with batch-based training history
        """
        if verbose:
            print(f"\nBatch-based training on digits: {training_digits}")
            if monitor_digits:
                print(f"Monitoring digits: {monitor_digits}")
            print(f"Total batches: {total_batches:,}")
            print(f"Logging every: {log_every_n_batches} batches")
            print(f"Validation every: {val_every_n_batches} batches")
        
        # Create data loaders for training digits
        train_loader, val_loader, test_loader = self.create_digit_loaders(training_digits, batch_size)
        
        if verbose:
            print(f"Training samples: {len(train_loader.dataset):,}")
            print(f"Validation samples: {len(val_loader.dataset):,}")
        
        # Setup training
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
                    
        history = {
            'training_digits': training_digits,
            'monitor_digits': monitor_digits,
            'total_batches': total_batches,
            'log_every_n_batches': log_every_n_batches,
            'val_every_n_batches': val_every_n_batches,
            'batch_numbers': [],
            'train_loss': [],
            'train_accuracy': [],  # ADD THIS
            'val_loss': [],
            'val_accuracy': [],    # ADD THIS
            'test_loss': [],
            'test_accuracy': [],   # ADD THIS
        }        
        # Add monitoring history if specified
        if monitor_digits:
            history.update({
                'monitor_train_loss': [],
                'monitor_train_accuracy': [],    # ADD THIS
                'monitor_val_loss': [],
                'monitor_val_accuracy': [],      # ADD THIS
                'monitor_test_loss': [],
                'monitor_test_accuracy': [],     # ADD THIS
            })        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        val_evaluations = 0
        
        # Create infinite data iterator
        train_iterator = iter(train_loader)
        
        if verbose:
            print(f"\nStarting batch-based training...")
            print("-" * 80)
        
        # Training loop
        self.model.train()
        batch_train_losses = []
        batch_train_accuracies = []
        with tqdm(total=total_batches, desc="Training Batches") as pbar:
            for batch_idx in range(total_batches):
                
                # Get next batch (cycle through dataset if needed)
                try:
                    images, labels = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    images, labels = next(train_iterator)
                
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Track batch loss
                batch_train_losses.append(loss.item())
                
                _, predicted = torch.max(outputs.data, 1)
                batch_accuracy = (predicted == labels).float().mean().item() * 100  # Convert to percentage
                batch_train_accuracies.append(batch_accuracy)
                # Log training metrics every N batches
                if (batch_idx + 1) % log_every_n_batches == 0:
                    avg_train_loss = np.mean(batch_train_losses[-log_every_n_batches:])
                    avg_train_accuracy = np.mean(batch_train_accuracies[-log_every_n_batches:])
                    
                    history['batch_numbers'].append(batch_idx + 1)
                    history['train_loss'].append(avg_train_loss)
                    history['train_accuracy'].append(avg_train_accuracy)
                    
                    # Add placeholders for val/test loss (will be filled during validation)
                    history['val_loss'].append(None)
                    history['val_accuracy'].append(None)
                    history['test_loss'].append(None)
                    history['test_accuracy'].append(None)

                    # Add monitor placeholders
                    if monitor_digits:
                        history['monitor_train_loss'].append(None)
                        history['monitor_train_accuracy'].append(None)  
                        history['monitor_val_loss'].append(None)
                        history['monitor_val_accuracy'].append(None)    
                        history['monitor_test_loss'].append(None)
                        history['monitor_test_accuracy'].append(None)                
                          
                # Validation every N batches
                if (batch_idx + 1) % log_every_n_batches == 0:
                    # Evaluate on training digits
                    val_metrics = self.evaluate_on_digits(training_digits, split='val')
                    test_metrics = self.evaluate_on_digits(training_digits, split='test')

                    # Checkpointing after each batch
                    if checkpointer:
                        checkpointer.maybe_save_batch(
                            current_batch=batch_idx + 1,
                            training_history=history,
                            training_config={
                                'training_digits': training_digits,
                                'total_batches': total_batches,
                                'learning_rate': learning_rate,
                                'batch_size': batch_size
                            },
                            model_info=self.model.get_model_summary() if hasattr(self.model, 'get_model_summary') else None
                        )

                # Update the most recent logged entry with validation results
                if history['batch_numbers']:  # If we have logged entries
                    latest_idx = len(history['batch_numbers']) - 1
                    history['val_loss'][latest_idx] = val_metrics['loss']
                    history['val_accuracy'][latest_idx] = val_metrics['accuracy']     # ADD THIS
                    history['test_loss'][latest_idx] = test_metrics['loss']
                    history['test_accuracy'][latest_idx] = test_metrics['accuracy']   # ADD THIS                    
                    # Monitor metrics if specified
                    if monitor_digits:
                        monitor_train = self.evaluate_on_digits(monitor_digits, split='train')
                        monitor_val = self.evaluate_on_digits(monitor_digits, split='val')
                        monitor_test = self.evaluate_on_digits(monitor_digits, split='test')
                        
                        if history['batch_numbers']:  # If we have logged entries
                            latest_idx = len(history['batch_numbers']) - 1
                            history['monitor_train_loss'][latest_idx] = monitor_train['loss']
                            history['monitor_train_accuracy'][latest_idx] = monitor_train['accuracy']   # ADD THIS
                            history['monitor_val_loss'][latest_idx] = monitor_val['loss']
                            history['monitor_val_accuracy'][latest_idx] = monitor_val['accuracy']       # ADD THIS
                            history['monitor_test_loss'][latest_idx] = monitor_test['loss']
                            history['monitor_test_accuracy'][latest_idx] = monitor_test['accuracy']     # ADD THIS                    
                    val_evaluations += 1
                    
                    # Early stopping check
                    if early_stopping:
                        if val_metrics['loss'] < best_val_loss:
                            best_val_loss = val_metrics['loss']
                            patience_counter = 0
                            best_model_state = copy.deepcopy(self.model.state_dict())
                        else:
                            patience_counter += 1
                            
                            if patience_counter >= early_stopping_patience:
                                if verbose:
                                    print(f"\nEarly stopping at batch {batch_idx + 1}")
                                break
                    
                    # Print progress
                    if verbose:
                        progress_str = f"Batch {batch_idx + 1:6d}: "
                        progress_str += f"Train Acc: {avg_train_accuracy:6.2f}% | "     # ADD THIS
                        progress_str += f"Val Acc: {val_metrics['accuracy']:6.2f}% | "  # ADD THIS
                        progress_str += f"Train Loss: {avg_train_loss:.4f} | "
                        progress_str += f"Val Loss: {val_metrics['loss']:.4f}"

                        if monitor_digits and history['monitor_test_loss'][latest_idx] is not None:
                            monitor_loss = history['monitor_test_loss'][latest_idx]
                            progress_str += f" | Monitor Loss: {monitor_loss:.4f}"
                        
                        print(progress_str)
                pbar.update(1)
        
        # Load best model if early stopping was used
        if early_stopping and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Clean up None values in history (remove entries without validation)
        cleaned_history = self._clean_batch_history(history)
        
        # Add final metrics
        cleaned_history['batches_trained'] = batch_idx + 1
        cleaned_history['best_val_loss'] = best_val_loss
        cleaned_history['val_evaluations'] = val_evaluations
        
        if verbose:
            print(f"\nBatch-based training completed!")
            print(f"Batches trained: {batch_idx + 1:,}")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print(f"Validation evaluations: {val_evaluations}")
        
        return cleaned_history
    
    def _clean_batch_history(self, history: Dict) -> Dict:
        """
        Clean up batch history by removing entries with None validation values.
        Keep only entries that have both training and validation metrics.
        """
        cleaned = {}
        
        # Find indices where we have validation data
        valid_indices = []
        for i, val_loss in enumerate(history['val_loss']):
            if val_loss is not None:
                valid_indices.append(i)
        
        # Copy metadata
        for key in ['training_digits', 'monitor_digits', 'total_batches', 
                   'log_every_n_batches', 'val_every_n_batches']:
            if key in history:
                cleaned[key] = history[key]
        
        # Copy arrays with valid indices only
        for key in ['batch_numbers', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 
                'test_loss', 'test_accuracy']:  # ADD train_accuracy, val_accuracy, test_accuracy
            if key in history:
                cleaned[key] = [history[key][i] for i in valid_indices]        
        # Copy monitor arrays if they exist
        for key in ['monitor_train_loss', 'monitor_train_accuracy', 'monitor_val_loss', 
                'monitor_val_accuracy', 'monitor_test_loss', 'monitor_test_accuracy']:  # ADD accuracy versions
            if key in history:
                cleaned[key] = [history[key][i] for i in valid_indices]      
                  
        return cleaned
    
    # Include original methods if not inherited
    if FlexibleTrainer == object:
        def create_digit_loaders(self, digits: List[int], batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
            """Create train, validation, and test loaders for specific digits."""
            # This is a simplified version - use the full implementation from original trainer
            train_filtered = DigitFilter.filter_by_digits(self.data_loader.train_subset, digits)
            val_filtered = DigitFilter.filter_by_digits(self.data_loader.val_subset, digits)
            test_filtered = DigitFilter.filter_by_digits(self.data_loader.test_dataset, digits)
            
            train_loader = DataLoader(train_filtered, batch_size=batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_filtered, batch_size=batch_size, shuffle=False, num_workers=2)
            test_loader = DataLoader(test_filtered, batch_size=batch_size, shuffle=False, num_workers=2)
            
            return train_loader, val_loader, test_loader
        
        def evaluate_on_digits(self, digits: List[int], split: str = 'test') -> Dict[str, float]:
            """Evaluate model performance on specific digits."""
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
        
        def get_performance_summary(self, digits_groups: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
            """Get performance summary on multiple digit groups."""
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


# Example usage and testing
if __name__ == "__main__":
    print("Testing FlexibleTrainer with Batch-Based Training")
    print("="*60)
    
    # This would normally import your actual data and model
    print("Note: This is a standalone version for testing batch-based training.")
    print("To use with your actual MNIST setup, ensure all imports are available.")
