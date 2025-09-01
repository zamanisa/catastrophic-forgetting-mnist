"""
Model Checkpointer Module

A comprehensive checkpointing system for deep learning training experiments.
Saves model weights, optimizer state, and training metadata with resume functionality.

Usage:
    from model_checkpointer import ModelCheckpointer
    
    # Initialize checkpointer
    checkpointer = ModelCheckpointer(
        model=model,
        optimizer=optimizer,
        experiment_name="mnist_digits_02",
        save_every_n_epochs=5
    )
    
    # During training
    checkpointer.maybe_save_epoch(current_epoch)
    
    # Resume training
    start_epoch = checkpointer.load_latest_checkpoint()
"""

import torch
import json
import os
import glob
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


class ModelCheckpointer:
    """
    A comprehensive checkpointing system for deep learning training experiments.
    
    Features:
    - Saves model weights, optimizer state, and training metadata
    - Supports both epoch-based and batch-based training
    - Resume from latest checkpoint functionality  
    - Automatic directory creation and file management
    - Compatible with all model architectures (MNISTModelNN, MNISTTransformer, MNISTGraphNN)
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        experiment_name: str,
        save_every_n_epochs: Optional[int] = None,
        save_every_n_batches: Optional[int] = None,
        checkpoint_dir: str = "model_checkpoints",
        include_timestamp: bool = True,
        device: str = None
    ):
        """
        Initialize the model checkpointer.
        
        Args:
            model: PyTorch model to checkpoint
            optimizer: Optimizer to save state for
            experiment_name: Name for this experiment (e.g., "mnist_digits_02")
            save_every_n_epochs: Save checkpoint every N epochs (for epoch-based training)
            save_every_n_batches: Save checkpoint every N batches (for batch-based training)
            checkpoint_dir: Base directory for all checkpoints
            include_timestamp: Whether to add timestamp to experiment folder
            device: Device to save tensors on
        """
        self.model = model
        self.optimizer = optimizer
        self.experiment_name = experiment_name
        self.save_every_n_epochs = save_every_n_epochs
        self.save_every_n_batches = save_every_n_batches
        self.checkpoint_dir = Path(checkpoint_dir)
        self.include_timestamp = include_timestamp
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create experiment-specific directory (matching TrainingLogger pattern)
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_dir = self.checkpoint_dir / f"{experiment_name}_{timestamp}"
        else:
            self.experiment_dir = self.checkpoint_dir / experiment_name
        
        # Create directories
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Track checkpointer initialization time
        self.checkpointer_start_time = time.time()
        
        print(f"ModelCheckpointer initialized:")
        print(f"  Experiment: {experiment_name}")
        print(f"  Checkpoint directory: {self.experiment_dir}")
        if save_every_n_epochs:
            print(f"  Save frequency: every {save_every_n_epochs} epochs")
        if save_every_n_batches:
            print(f"  Save frequency: every {save_every_n_batches} batches")
    
    def maybe_save_epoch(
        self,
        current_epoch: int,
        training_history: Optional[Dict] = None,
        training_config: Optional[Dict] = None,
        model_info: Optional[Dict] = None,
        force_save: bool = False
    ):
        """
        Save checkpoint if current epoch matches save frequency.
        
        Args:
            current_epoch: Current epoch number
            training_history: Optional training history dictionary
            training_config: Optional training configuration
            model_info: Optional model architecture info
            force_save: Force save regardless of frequency
        """
        if not self.save_every_n_epochs and not force_save:
            return
        
        should_save = force_save or (current_epoch % self.save_every_n_epochs == 0)
        
        if should_save:
            checkpoint_name = f"model_weights_epoch_{current_epoch:04d}.pt"
            self._save_checkpoint(
                checkpoint_name=checkpoint_name,
                epoch=current_epoch,
                batch=None,
                training_history=training_history,
                training_config=training_config,
                model_info=model_info
            )
    
    def maybe_save_batch(
        self,
        current_batch: int,
        training_history: Optional[Dict] = None,
        training_config: Optional[Dict] = None,
        model_info: Optional[Dict] = None,
        force_save: bool = False
    ):
        """
        Save checkpoint if current batch matches save frequency.
        
        Args:
            current_batch: Current batch number
            training_history: Optional training history dictionary
            training_config: Optional training configuration
            model_info: Optional model architecture info
            force_save: Force save regardless of frequency
        """
        if not self.save_every_n_batches and not force_save:
            return
        
        should_save = force_save or (current_batch % self.save_every_n_batches == 0)
        
        if should_save:
            checkpoint_name = f"model_weights_batch_{current_batch:06d}.pt"
            self._save_checkpoint(
                checkpoint_name=checkpoint_name,
                epoch=None,
                batch=current_batch,
                training_history=training_history,
                training_config=training_config,
                model_info=model_info
            )
    
    def _save_checkpoint(
        self,
        checkpoint_name: str,
        epoch: Optional[int] = None,
        batch: Optional[int] = None,
        training_history: Optional[Dict] = None,
        training_config: Optional[Dict] = None,
        model_info: Optional[Dict] = None
    ):
        """Internal method to save a complete checkpoint."""
        checkpoint_path = self.experiment_dir / checkpoint_name
        
        # Calculate training duration
        training_duration = time.time() - self.checkpointer_start_time
        
        # Create comprehensive checkpoint data
        checkpoint_data = {
            # Essential for resuming training
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            
            # Training progress
            'epoch': epoch,
            'batch': batch,
            
            # Model architecture info
            'model_info': model_info or self._get_model_info(),
            
            # Training configuration
            'training_config': training_config or {},
            
            # Training history (if provided)
            'training_history': training_history,
            
            # Checkpoint metadata
            'checkpoint_metadata': {
                'experiment_name': self.experiment_name,
                'saved_at': datetime.now().isoformat(),
                'checkpoint_file': checkpoint_name,
                'training_duration_seconds': round(training_duration, 2),
                'device': str(self.device),
                'pytorch_version': torch.__version__
            }
        }
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Create human-readable summary
        self._save_checkpoint_summary(checkpoint_name, checkpoint_data)
        
        print(f"Checkpoint saved: {checkpoint_name}")
        if epoch is not None:
            print(f"  Epoch: {epoch}")
        if batch is not None:
            print(f"  Batch: {batch:,}")
        print(f"  File: {checkpoint_path}")
    
    def _save_checkpoint_summary(self, checkpoint_name: str, checkpoint_data: Dict):
        """Save human-readable checkpoint summary as JSON."""
        summary_name = checkpoint_name.replace('.pt', '_summary.json')
        summary_path = self.experiment_dir / summary_name
        
        # Create readable summary (exclude large tensors)
        summary = {
            'checkpoint_info': checkpoint_data['checkpoint_metadata'],
            'training_progress': {
                'epoch': checkpoint_data.get('epoch'),
                'batch': checkpoint_data.get('batch')
            },
            'model_architecture': checkpoint_data.get('model_info', {}),
            'training_configuration': checkpoint_data.get('training_config', {}),
        }
        
        # Add training history summary if available
        if checkpoint_data.get('training_history'):
            history = checkpoint_data['training_history']
            if 'train_loss' in history and history['train_loss']:
                summary['training_performance'] = {
                    'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
                    'final_train_accuracy': history.get('train_accuracy', [None])[-1],
                    'final_val_loss': history.get('val_loss', [None])[-1],
                    'final_val_accuracy': history.get('val_accuracy', [None])[-1],
                    'training_points': len(history['train_loss'])
                }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def load_latest_checkpoint(self) -> Tuple[int, int]:
        """
        Load the most recent checkpoint and restore model/optimizer state.
        
        Returns:
            Tuple of (epoch, batch) where training should resume from.
            Returns (0, 0) if no checkpoints found.
        """
        checkpoint_files = list(self.experiment_dir.glob("model_weights_*.pt"))
        
        if not checkpoint_files:
            print(f"No checkpoints found in {self.experiment_dir}")
            return 0, 0
        
        # Sort by modification time to get the latest
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        
        print(f"Loading checkpoint: {latest_checkpoint.name}")
        
        # Load checkpoint data
        checkpoint_data = torch.load(latest_checkpoint, map_location=self.device)
        
        # Restore model and optimizer states
        self.model.load_state_dict(checkpoint_data['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # Get resume information
        resume_epoch = checkpoint_data.get('epoch', 0)
        resume_batch = checkpoint_data.get('batch', 0)
        
        # Print resume information
        saved_at = checkpoint_data.get('checkpoint_metadata', {}).get('saved_at', 'Unknown')
        print(f"Checkpoint loaded successfully!")
        print(f"  Saved at: {saved_at}")
        print(f"  Resume from epoch: {resume_epoch}")
        print(f"  Resume from batch: {resume_batch}")
        
        return resume_epoch or 0, resume_batch or 0
    
    def load_specific_checkpoint(self, checkpoint_name: str) -> Tuple[int, int]:
        """
        Load a specific checkpoint by filename.
        
        Args:
            checkpoint_name: Name of checkpoint file (e.g., "model_weights_epoch_0010.pt")
            
        Returns:
            Tuple of (epoch, batch) from the loaded checkpoint.
        """
        checkpoint_path = self.experiment_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading specific checkpoint: {checkpoint_name}")
        
        # Load checkpoint data
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore model and optimizer states
        self.model.load_state_dict(checkpoint_data['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # Get checkpoint information
        resume_epoch = checkpoint_data.get('epoch', 0)
        resume_batch = checkpoint_data.get('batch', 0)
        
        print(f"Specific checkpoint loaded successfully!")
        print(f"  Epoch: {resume_epoch}")
        print(f"  Batch: {resume_batch}")
        
        return resume_epoch or 0, resume_batch or 0
    
    def list_checkpoints(self) -> List[Dict]:
        """
        List all available checkpoints with their metadata.
        
        Returns:
            List of dictionaries containing checkpoint information.
        """
        checkpoint_files = list(self.experiment_dir.glob("model_weights_*.pt"))
        
        if not checkpoint_files:
            print(f"No checkpoints found in {self.experiment_dir}")
            return []
        
        checkpoints_info = []
        
        for checkpoint_file in sorted(checkpoint_files):
            try:
                # Load just the metadata (not the full model)
                checkpoint_data = torch.load(checkpoint_file, map_location='cpu')
                
                info = {
                    'filename': checkpoint_file.name,
                    'epoch': checkpoint_data.get('epoch'),
                    'batch': checkpoint_data.get('batch'),
                    'saved_at': checkpoint_data.get('checkpoint_metadata', {}).get('saved_at'),
                    'file_size_mb': checkpoint_file.stat().st_size / (1024**2)
                }
                checkpoints_info.append(info)
                
            except Exception as e:
                print(f"Warning: Could not read checkpoint {checkpoint_file.name}: {e}")
        
        return checkpoints_info
    
    def _get_model_info(self) -> Dict:
        """Get model architecture information if available."""
        try:
            # Try to get model summary (works with our custom models)
            if hasattr(self.model, 'get_model_summary'):
                return self.model.get_model_summary()
            else:
                # Fallback to basic info
                total_params = sum(p.numel() for p in self.model.parameters())
                return {
                    'model_type': type(self.model).__name__,
                    'total_parameters': total_params,
                    'model_size_mb': total_params * 4 / (1024**2)
                }
        except Exception:
            return {'model_type': type(self.model).__name__}
    
    def get_experiment_path(self) -> Path:
        """Get the path to the experiment directory."""
        return self.experiment_dir
    
    def save_best_model(
        self,
        current_metric: float,
        epoch: Optional[int] = None,
        batch: Optional[int] = None,
        is_higher_better: bool = True,
        training_history: Optional[Dict] = None,
        training_config: Optional[Dict] = None,
        model_info: Optional[Dict] = None
    ):
        """
        Save model if it's the best so far based on a metric.
        
        Args:
            current_metric: Current metric value (e.g., validation accuracy)
            epoch: Current epoch (optional)
            batch: Current batch (optional)
            is_higher_better: True if higher metric is better (accuracy), False if lower is better (loss)
            training_history: Optional training history
            training_config: Optional training configuration
            model_info: Optional model info
        """
        best_metric_file = self.experiment_dir / "best_metric.txt"
        
        should_save = False
        
        # Check if this is the best model so far
        if best_metric_file.exists():
            with open(best_metric_file, 'r') as f:
                previous_best = float(f.read().strip())
            
            if is_higher_better:
                should_save = current_metric > previous_best
            else:
                should_save = current_metric < previous_best
        else:
            should_save = True  # First time saving
        
        if should_save:
            # Update best metric file
            with open(best_metric_file, 'w') as f:
                f.write(str(current_metric))
            
            # Save best model
            checkpoint_name = "model_weights_best.pt"
            self._save_checkpoint(
                checkpoint_name=checkpoint_name,
                epoch=epoch,
                batch=batch,
                training_history=training_history,
                training_config=training_config,
                model_info=model_info
            )
            
            metric_type = "higher" if is_higher_better else "lower"
            print(f"New best model saved! ({metric_type} is better: {current_metric})")


# Convenience function for quick usage
def create_model_checkpointer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    experiment_name: str,
    save_every_n_epochs: Optional[int] = None,
    save_every_n_batches: Optional[int] = None,
    checkpoint_dir: str = "model_checkpoints"
) -> ModelCheckpointer:
    """
    Quick function to create a model checkpointer.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        experiment_name: Name for this experiment
        save_every_n_epochs: Save frequency for epoch-based training
        save_every_n_batches: Save frequency for batch-based training
        checkpoint_dir: Base directory for checkpoints
        
    Returns:
        ModelCheckpointer instance
    """
    return ModelCheckpointer(
        model=model,
        optimizer=optimizer,
        experiment_name=experiment_name,
        save_every_n_epochs=save_every_n_epochs,
        save_every_n_batches=save_every_n_batches,
        checkpoint_dir=checkpoint_dir
    )


# Example usage demonstration
if __name__ == "__main__":
    import torch.optim as optim
    from mnist_model import MNISTModelNN  # Assuming this exists
    
    print("ModelCheckpointer Example Usage")
    print("="*50)
    
    # Create dummy model and optimizer for example
    model = MNISTModelNN([128, 64])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create checkpointer
    checkpointer = ModelCheckpointer(
        model=model,
        optimizer=optimizer,
        experiment_name="example_experiment",
        save_every_n_epochs=5,
        save_every_n_batches=100
    )
    
    print(f"\nExample checkpointer created in: {checkpointer.get_experiment_path()}")
    
    # Example usage in training loops
    print("\nExample usage in epoch-based training:")
    print("for epoch in range(1, 51):")
    print("    # ... training code ...")
    print("    checkpointer.maybe_save_epoch(epoch)")
    
    print("\nExample usage in batch-based training:")
    print("for batch in range(1, 1001):")
    print("    # ... training code ...")
    print("    checkpointer.maybe_save_batch(batch)")
    
    print("\nExample resuming from checkpoint:")
    print("resume_epoch, resume_batch = checkpointer.load_latest_checkpoint()")
    print("# Continue training from resume_epoch or resume_batch")
    
    print(f"\nModelCheckpointer is ready for use!")