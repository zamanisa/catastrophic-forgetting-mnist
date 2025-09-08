"""
Unified Training Experiment Function

This script provides a single function that replaces all the individual experiment functions,
with automatic detection of training mode and flexible configuration.
"""
import torch
import torch.optim as optim
import warnings
from utils.mnist_data_prep import MNISTDataLoader
from loggings.trainer_with_batch_support import FlexibleTrainerWithBatch
from loggings.training_logger import TrainingLogger


def run_training_experiment(
    model,
    training_config,
    experiment_name,
    checkpointer=None,
    optimizer=None,
    criterion=None
):
    """
    Unified training experiment function that handles both epoch-based and batch-based training.
    
    Args:
        model: The neural network model to train
        training_config (dict): Configuration dictionary with training parameters
        experiment_name (str): Name for the experiment (used for logging)
        checkpointer (ModelCheckpointer, optional): Checkpointer object for saving models
        optimizer (torch.optim.Optimizer, optional): Optimizer (created automatically if None and checkpointer provided)
        criterion (torch.nn.Module, optional): Loss criterion
        
    Returns:
        tuple: (history, logger) - Training history and logger object
        
    Config Keys:
        Required:
            - training_digits (list): List of digits to train on
            
        Training Mode (one required):
            - epochs (int): Number of epochs for epoch-based training
            - total_batches (int): Number of batches for batch-based training
            - If neither provided, defaults to epochs=5
            
        Optional:
            - batch_size (int, default=64): Batch size for training
            - validation_split (float, default=0.2): Validation split ratio
            - learning_rate (float, default=0.001): Learning rate
            - monitor_digits (list, optional): Digits to monitor for catastrophic forgetting
            
        Epoch-based specific:
            - early_stopping_patience (int, default=3): Patience for early stopping
            
        Batch-based specific:
            - log_every_n_batches (int, default=100): Logging frequency
            - val_every_n_batches (int, default=100): Validation frequency
            - early_stopping (bool, default=True): Whether to use early stopping
            - early_stopping_patience (int, default=3): Patience for early stopping
    """
    
    # Define allowed configuration keys
    allowed_keys = {
        # Required
        'training_digits',
        # Training mode
        'epochs', 'total_batches',
        # Common optional
        'batch_size', 'validation_split', 'learning_rate', 'monitor_digits',
        # Epoch-based specific
        'early_stopping_patience',
        # Batch-based specific
        'log_every_n_batches', 'val_every_n_batches', 'early_stopping',
        # Additional
        'training_type'  # For backwards compatibility
    }
    
    # Validate configuration keys
    config_keys = set(training_config.keys())
    unexpected_keys = config_keys - allowed_keys
    if unexpected_keys:
        warnings.warn(f"Unexpected configuration keys found: {unexpected_keys}. "
                     f"Allowed keys are: {sorted(allowed_keys)}")
    
    # Validate required keys
    if 'training_digits' not in training_config:
        raise ValueError("'training_digits' is required in training_config")
    
    # Set default values
    config = training_config.copy()
    config.setdefault('batch_size', 64)
    config.setdefault('validation_split', 0.2)
    config.setdefault('learning_rate', 0.001)
    
    # Auto-detect training mode
    has_epochs = 'epochs' in config
    has_total_batches = 'total_batches' in config
    
    if has_epochs and has_total_batches:
        raise ValueError("Configuration cannot have both 'epochs' and 'total_batches'. "
                        "Please specify only one to determine training mode.")
    elif has_epochs:
        training_mode = 'epoch_based'
        config.setdefault('early_stopping_patience', 3)
    elif has_total_batches:
        training_mode = 'batch_based'
        config.setdefault('log_every_n_batches', 100)
        config.setdefault('val_every_n_batches', 100)
        config.setdefault('early_stopping', True)
        config.setdefault('early_stopping_patience', 3)
    else:
        # Default to epoch-based training
        training_mode = 'epoch_based'
        config['epochs'] = 5
        config.setdefault('early_stopping_patience', 3)
        warnings.warn("Neither 'epochs' nor 'total_batches' specified. "
                     "Defaulting to epoch-based training with 5 epochs.")
    
    # Setup data loader
    data_loader = MNISTDataLoader(
        batch_size=config['batch_size'],
        validation_split=config['validation_split']
    )
    
    # Setup trainer
    trainer = FlexibleTrainerWithBatch(model, data_loader)
    
    # Setup logger
    logger = TrainingLogger(experiment_name)
    
    # Setup optimizer if checkpointer is provided but optimizer is not
    if checkpointer is not None and optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Add training mode to config for logging
    config['training_type'] = training_mode
    
    print(f"Training Mode: {training_mode.upper()}")
    print(f"Training digits: {config['training_digits']}")
    if 'monitor_digits' in config:
        print(f"Monitoring digits: {config['monitor_digits']}")
    
    # Execute training based on detected mode
    if training_mode == 'epoch_based':
        print(f"Epochs: {config['epochs']}")
        print(f"Early stopping patience: {config['early_stopping_patience']}")
        
        # Prepare arguments for epoch-based training
        train_args = {
            'training_digits': config['training_digits'],
            'epochs': config['epochs'],
            'learning_rate': config['learning_rate'],
            'early_stopping_patience': config['early_stopping_patience'],
            'verbose': True
        }
        
        # Add optional parameters
        if 'monitor_digits' in config:
            train_args['monitor_digits'] = config['monitor_digits']
        if checkpointer is not None:
            train_args['checkpointer'] = checkpointer
            train_args['optimizer'] = optimizer
        if criterion is not None:
            train_args['criterion'] = criterion
            
        # Train using epoch-based method
        history = trainer.train_on_digits(**train_args)
        
    else:  # batch_based
        print(f"Total batches: {config['total_batches']:,}")
        print(f"Logging every: {config['log_every_n_batches']} batches")
        print(f"Validation every: {config['val_every_n_batches']} batches")
        print(f"Early stopping: {config['early_stopping']}")
        if config['early_stopping']:
            print(f"Early stopping patience: {config['early_stopping_patience']}")
        
        # Prepare arguments for batch-based training
        train_args = {
            'training_digits': config['training_digits'],
            'total_batches': config['total_batches'],
            'log_every_n_batches': config['log_every_n_batches'],
            'val_every_n_batches': config['val_every_n_batches'],
            'learning_rate': config['learning_rate'],
            'batch_size': config['batch_size'],
            'early_stopping': config['early_stopping'],
            'early_stopping_patience': config['early_stopping_patience'],
            'verbose': True
        }
        
        # Add optional parameters
        if 'monitor_digits' in config:
            train_args['monitor_digits'] = config['monitor_digits']
        if checkpointer is not None:
            train_args['checkpointer'] = checkpointer
            train_args['optimizer'] = optimizer
        if criterion is not None:
            train_args['criterion'] = criterion
            
        # Train using batch-based method
        history = trainer.train_on_digits_batch_based(**train_args)
    
    # Log the results
    additional_info = {
        'experiment_notes': f'{training_mode.replace("_", "-").title()} training experiment',
        'training_mode': training_mode
    }
    
    if training_mode == 'batch_based':
        additional_info['data_points_captured'] = len(history.get('train_loss', []))
        additional_info['granularity_analysis'] = f'Logged every {config["log_every_n_batches"]} batches'
    
    logger.log_training_run(
        training_history=history,
        model_info=model.get_model_summary(),
        training_config=config,
        additional_info=additional_info
    )
    
    # Print completion message
    print(f"✓ {training_mode.replace('_', '-').title()} experiment logged to: {logger.get_experiment_path()}")
    if checkpointer is not None:
        print(f"✓ Checkpoints saved to: {checkpointer.get_experiment_path()}")
    
    return history, logger


# Example usage and test configurations
if __name__ == "__main__":
    from models.ff_nn import MNISTModelNN
    from models.graph_nn import MNISTGraphNN
    from models.transformer import MNISTTransformer
    from loggings.model_checkpointer import ModelCheckpointer
    
    # Example model
    model = MNISTGraphNN(
        hidden_dims=[32, 64, 32],
        connectivity_radius=1,
        use_position_features=False,
        pooling_method='max',
        dropout_rate=0.2
    )
    
    print("Unified Training Experiment Function - Example Usage")
    print("=" * 60)
    
    # Example 1: Simple epoch-based training
    print("\n1. Simple Epoch-based Training:")
    epoch_config = {
        'training_digits': [0, 1, 2],
        'epochs': 3,
        'learning_rate': 0.001,
        'monitor_digits': [8, 9]
    }
    
    history1, logger1 = run_training_experiment(
        model=model,
        training_config=epoch_config,
        experiment_name="unified_epoch_example"
    )
    
    # Example 2: Batch-based training with checkpointing
    print("\n2. Batch-based Training with Checkpointing:")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    checkpointer = ModelCheckpointer(
        model=model,
        optimizer=optimizer,
        experiment_name="unified_batch_example_with_checkpointing",
        save_every_n_batches=100
    )
    
    batch_config = {
        'training_digits': [3, 4, 5],
        'total_batches': 300,
        'log_every_n_batches': 50,
        'val_every_n_batches': 100,
        'batch_size': 32,
        'learning_rate': 0.0015,
        'monitor_digits': [6, 7, 8, 9]
    }
    
    history2, logger2 = run_training_experiment(
        model=model,
        training_config=batch_config,
        experiment_name="unified_batch_example",
        checkpointer=checkpointer,
        optimizer=optimizer
    )
    
    print("\n" + "=" * 60)
    print("Examples completed! Check training_logs directory for results.")
