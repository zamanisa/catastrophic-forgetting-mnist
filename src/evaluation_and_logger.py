"""
Training Logger Module

A flexible logging system for deep learning training experiments.
Supports logging to both CSV (for time-series data) and JSON (for metadata).

Usage:
    from training_logger import TrainingLogger
    
    # Initialize logger
    logger = TrainingLogger(experiment_name="mnist_digits_02")
    
    # After training
    logger.log_training_run(
        training_history=history,
        model_info=model.get_model_summary(),
        training_config=config
    )
"""

import json
import csv
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class TrainingLogger:
    """
    A comprehensive logging system for deep learning training experiments.
    
    Features:
    - CSV logging for epoch-by-epoch metrics
    - JSON logging for metadata and summaries
    - Timestamped experiment tracking
    - Automatic directory creation
    - Clean separation of different data types
    """
    
    def __init__(
        self, 
        experiment_name: str,
        log_dir: str = "training_logs",
        include_timestamp: bool = True
    ):
        """
        Initialize the training logger.
        
        Args:
            experiment_name: Name for this experiment (e.g., "mnist_digits_02")
            log_dir: Base directory for all logs
            include_timestamp: Whether to add timestamp to experiment folder
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.include_timestamp = include_timestamp
        
        # Create experiment-specific directory
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        else:
            self.experiment_dir = self.log_dir / experiment_name
        
        # Create directories
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Track experiment start time
        self.experiment_start_time = time.time()
        
        print(f"TrainingLogger initialized:")
        print(f"  Experiment: {experiment_name}")
        print(f"  Log directory: {self.experiment_dir}")
    
    def log_training_run(
        self,
        training_history: Dict,
        model_info: Dict,
        training_config: Dict,
        additional_info: Optional[Dict] = None
    ):
        """
        Log a complete training run with all associated data.
        
        Args:
            training_history: History dictionary from FlexibleTrainer.train_on_digits()
            model_info: Model architecture info from model.get_model_summary()
            training_config: Training configuration parameters
            additional_info: Any additional information to log
        """
        print(f"\nLogging training run to: {self.experiment_dir}")
        
        # Calculate training duration
        training_duration = time.time() - self.experiment_start_time
        
        # 1. Log epoch-by-epoch metrics to CSV
        self._log_epoch_metrics_csv(training_history, training_config)
        
        # 2. Log everything else to single JSON file
        self._log_complete_experiment_json(
            training_history, model_info, training_config, 
            training_duration, additional_info
        )
        
        print("✓ Training run logged successfully!")
        print(f"  Files created in: {self.experiment_dir}")
        print(f"  - epoch_metrics.csv (per-epoch training data)")
        print(f"  - experiment_data.json (all metadata and summaries)")
    
    def _log_epoch_metrics_csv(self, history: Dict, training_config: Dict):
        """Log epoch-by-epoch training metrics to CSV with test loss calculation."""
        csv_file = self.experiment_dir / "epoch_metrics.csv"
        
        # Determine all available metrics
        epochs = len(history.get('train_loss', []))
        if epochs == 0:
            print("⚠ No epoch data to log")
            return
        
        # Fixed headers as requested
        headers = ['epoch', 'train_loss', 'val_loss', 'test_loss', 
                  'monitor_train_loss', 'monitor_val_loss', 'monitor_test_loss']
        
        # Get training digits for test loss calculation
        training_digits = history.get('training_digits', [])
        
        # Write CSV
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for epoch in range(epochs):
                row = [epoch + 1]  # 1-indexed epochs
                
                # Add train_loss
                train_loss = history.get('train_loss', [None])[epoch] if epoch < len(history.get('train_loss', [])) else None
                row.append(train_loss)
                
                # Add val_loss
                val_loss = history.get('val_loss', [None])[epoch] if epoch < len(history.get('val_loss', [])) else None
                row.append(val_loss)
                
                # Add test_loss (we'll need to calculate this or use a placeholder)
                # For now, we'll use None and note that this needs to be calculated during training
                test_loss = None  # This would need to be calculated during training
                row.append(test_loss)
                
                # Add monitor metrics
                monitor_train_loss = history.get('monitor_train_loss', [None])[epoch] if epoch < len(history.get('monitor_train_loss', [])) else None
                monitor_val_loss = history.get('monitor_val_loss', [None])[epoch] if epoch < len(history.get('monitor_val_loss', [])) else None
                monitor_test_loss = history.get('monitor_test_loss', [None])[epoch] if epoch < len(history.get('monitor_test_loss', [])) else None
                
                row.extend([monitor_train_loss, monitor_val_loss, monitor_test_loss])
                
                writer.writerow(row)
        
        print(f"✓ Epoch metrics saved to: epoch_metrics.csv ({epochs} epochs)")
        print(f"  Note: test_loss column requires modification to FlexibleTrainer to calculate test loss per epoch")
    
    def _log_complete_experiment_json(
        self, 
        history: Dict, 
        model_info: Dict, 
        training_config: Dict, 
        duration: float,
        additional_info: Optional[Dict]
    ):
        """Log all experiment data to a single comprehensive JSON file."""
        json_file = self.experiment_dir / "experiment_data.json"
        
        # Create comprehensive experiment data structure
        experiment_data = {
            'experiment_metadata': {
                'experiment_name': self.experiment_name,
                'logged_at': datetime.now().isoformat(),
                'experiment_directory': str(self.experiment_dir),
                'training_duration_seconds': round(duration, 2),
                'training_duration_formatted': self._format_duration(duration)
            },
            
            'model_architecture': model_info,
            
            'training_configuration': training_config,
            
            'training_summary': {
                'training_digits': history.get('training_digits', []),
                'monitor_digits': history.get('monitor_digits', []),
                'epochs_trained': history.get('epochs_trained', 0),
                'best_val_accuracy': history.get('best_val_accuracy', 0.0)
            },
            
            'final_performance': {
                'final_train_accuracy': history.get('train_accuracy', [0])[-1] if history.get('train_accuracy') else 0,
                'final_train_loss': history.get('train_loss', [0])[-1] if history.get('train_loss') else 0,
                'final_val_accuracy': history.get('val_accuracy', [0])[-1] if history.get('val_accuracy') else 0,
                'final_val_loss': history.get('val_loss', [0])[-1] if history.get('val_loss') else 0,
            },
            
            'catastrophic_forgetting_analysis': None,
            
            'additional_info': additional_info or {},
            
            'log_files_created': [
                'epoch_metrics.csv',
                'experiment_data.json'
            ]
        }
        
        # Add catastrophic forgetting analysis if monitor data exists
        if history.get('monitor_test_accuracy'):
            initial_monitor = history['monitor_test_accuracy'][0]
            final_monitor = history['monitor_test_accuracy'][-1]
            experiment_data['catastrophic_forgetting_analysis'] = {
                'initial_monitor_accuracy': initial_monitor,
                'final_monitor_accuracy': final_monitor,
                'accuracy_change': final_monitor - initial_monitor,
                'final_monitor_train_accuracy': history.get('monitor_train_accuracy', [0])[-1] if history.get('monitor_train_accuracy') else 0,
                'final_monitor_val_accuracy': history.get('monitor_val_accuracy', [0])[-1] if history.get('monitor_val_accuracy') else 0,
                'final_monitor_test_loss': history.get('monitor_test_loss', [0])[-1] if history.get('monitor_test_loss') else 0
            }
        
        with open(json_file, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        print(f"✓ Complete experiment data saved to: experiment_data.json")
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
    
    def log_custom_metrics(self, metrics_dict: Dict, filename: str = "custom_metrics"):
        """
        Log custom metrics to separate JSON file (useful for post-training analysis).
        
        Args:
            metrics_dict: Dictionary of metrics to log
            filename: Name for the JSON file (without extension)
        """
        json_file = self.experiment_dir / f"{filename}.json"
        
        metrics_with_time = {
            'logged_at': datetime.now().isoformat(),
            'custom_metrics': metrics_dict
        }
        
        with open(json_file, 'w') as f:
            json.dump(metrics_with_time, f, indent=2)
        
        print(f"✓ Custom metrics saved to: {filename}.json")
    
    def get_experiment_path(self) -> Path:
        """Get the path to the experiment directory."""
        return self.experiment_dir


# Convenience function for quick usage
def create_training_logger(experiment_name: str, log_dir: str = "training_logs") -> TrainingLogger:
    """
    Quick function to create a training logger.
    
    Args:
        experiment_name: Name for this experiment
        log_dir: Base directory for logs
        
    Returns:
        TrainingLogger instance
    """
    return TrainingLogger(experiment_name=experiment_name, log_dir=log_dir)


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

# Example usage demonstration
if __name__ == "__main__":
    # Example of how to use the logger
    print("TrainingLogger Example Usage")
    print("="*50)
    
    # Create logger
    logger = TrainingLogger("example_experiment")
    
    # Example data structures (matching your FlexibleTrainer output)
    example_history = {
        'training_digits': [0, 1],
        'monitor_digits': [2, 3, 4],
        'train_loss': [0.8, 0.6, 0.4, 0.3],
        'train_accuracy': [75.0, 80.0, 85.0, 90.0],
        'val_loss': [0.9, 0.7, 0.5, 0.4],
        'val_accuracy': [70.0, 75.0, 80.0, 85.0],
        'monitor_train_loss': [0.5, 0.6, 0.7, 0.8],
        'monitor_train_accuracy': [85.0, 82.0, 80.0, 78.0],
        'monitor_val_loss': [0.6, 0.7, 0.8, 0.9],
        'monitor_val_accuracy': [80.0, 78.0, 75.0, 72.0],
        'monitor_test_loss': [0.55, 0.65, 0.75, 0.85],
        'monitor_test_accuracy': [82.0, 80.0, 77.0, 74.0],
        'epochs_trained': 4,
        'best_val_accuracy': 85.0
    }
    
    example_model_info = {
        'architecture': '784-2000-1500-1000-500-10',
        'total_parameters': 2500000,
        'model_size_mb': 9.5,
        'dropout_rate': 0.3
    }
    
    example_config = {
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 50,
        'early_stopping_patience': 10
    }
    
    # Log the training run
    logger.log_training_run(
        training_history=example_history,
        model_info=example_model_info,
        training_config=example_config,
        additional_info={'experiment_notes': 'Testing catastrophic forgetting on digits 0,1 vs 2,3,4'}
    )
    
    print(f"\nExample logs created in: {logger.get_experiment_path()}")
