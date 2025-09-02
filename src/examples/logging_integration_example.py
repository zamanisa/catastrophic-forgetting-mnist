"""
Example: How to integrate TrainingLogger with FlexibleTrainer (both epoch and batch-based training)

This script demonstrates:
- Epoch-based training with logging
- Batch-based training with logging
- How the logger automatically handles both formats
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from mnist_data_prep import MNISTDataLoader
from models.ff_nn import MNISTModelNN
from trainer_with_batch_support import FlexibleTrainer  # Updated trainer
from training_logger import TrainingLogger
from loggings.model_checkpointer import ModelCheckpointer

model_type = MNISTModelNN(hidden_layers=[512, 256], dropout_rate=0.3)

def run_epoch_based_experiment():
    """
    Example of epoch-based training with logging (original functionality).
    """
    print("EPOCH-BASED TRAINING EXPERIMENT")
    print("="*60)
    
    # Setup
    data_loader = MNISTDataLoader(batch_size=64, validation_split=0.2)
    model = model_type
    trainer = FlexibleTrainer(model, data_loader)
    logger = TrainingLogger("epoch_based_experiment")
    
    # Training configuration
    training_config = {
        'training_type': 'epoch_based',
        'training_digits': [0, 1],
        'monitor_digits': [8, 9],
        'epochs': 2,
        'learning_rate': 0.001,
        'batch_size': 64,
        'early_stopping_patience': 3
    }
    
    print(f"Training on digits: {training_config['training_digits']}")
    print(f"Monitoring digits: {training_config['monitor_digits']}")
    
    # Train using original epoch-based method
    history = trainer.train_on_digits(
        training_digits=training_config['training_digits'],
        epochs=training_config['epochs'],
        learning_rate=training_config['learning_rate'],
        monitor_digits=training_config['monitor_digits'],
        early_stopping_patience=training_config['early_stopping_patience'],
        verbose=True
    )
    
    # Log results
    logger.log_training_run(
        training_history=history,
        model_info=model.get_model_summary(),
        training_config=training_config,
        additional_info={'experiment_notes': 'Epoch-based training baseline'}
    )
    
    print(f"✓ Epoch-based experiment logged to: {logger.get_experiment_path()}")
    return history


def run_batch_based_experiment():
    """
    Example of batch-based training with granular logging.
    """
    print("BATCH-BASED TRAINING EXPERIMENT")
    print("="*60)
    
    # Setup
    data_loader = MNISTDataLoader(batch_size=32, validation_split=0.2)
    model = model_type
    trainer = FlexibleTrainer(model, data_loader)
    logger = TrainingLogger("batch_based_experiment")
    
    # Training configuration
    training_config = {
        'training_type': 'batch_based',
        'training_digits': [2, 3, 4],
        'monitor_digits': [7, 8, 9],
        'total_batches': 500,
        'log_every_n_batches': 50,
        'val_every_n_batches': 100,
        'learning_rate': 0.0015,
        'batch_size': 32,
        'early_stopping': True,
        'early_stopping_patience': 3
    }
    
    print(f"Training on digits: {training_config['training_digits']}")
    print(f"Monitoring digits: {training_config['monitor_digits']}")
    print(f"Total batches: {training_config['total_batches']:,}")
    print(f"Logging every: {training_config['log_every_n_batches']} batches")
    print(f"Validation every: {training_config['val_every_n_batches']} batches")
    
    # Train using new batch-based method
    history = trainer.train_on_digits_batch_based(
        training_digits=training_config['training_digits'],
        total_batches=training_config['total_batches'],
        log_every_n_batches=training_config['log_every_n_batches'],
        val_every_n_batches=training_config['val_every_n_batches'],
        learning_rate=training_config['learning_rate'],
        batch_size=training_config['batch_size'],
        monitor_digits=training_config['monitor_digits'],
        early_stopping=training_config['early_stopping'],
        early_stopping_patience=training_config['early_stopping_patience'],
        verbose=True
    )
    
    # Log results (logger automatically detects batch-based format)
    logger.log_training_run(
        training_history=history,
        model_info=model.get_model_summary(),
        training_config=training_config,
        additional_info={
            'experiment_notes': 'Batch-based training with granular logging',
            'granularity_analysis': f'Logged every {training_config["log_every_n_batches"]} batches for detailed training dynamics'
        }
    )
    
    print(f"✓ Batch-based experiment logged to: {logger.get_experiment_path()}")
    return history


def run_comparison_experiment():
    """
    Run both training types for direct comparison.
    """
    print("COMPARISON EXPERIMENT")
    print("="*60)
    print("Running both epoch-based and batch-based training for comparison...")
    
    # Run epoch-based
    print("\n1. Running epoch-based training...")
    epoch_history = run_epoch_based_experiment()
    
    # Run batch-based  
    print("\n2. Running batch-based training...")
    batch_history = run_batch_based_experiment()
    
    # Create comparison summary
    logger = TrainingLogger("training_comparison_summary")
    
    comparison_data = {
        'epoch_based_results': {
            'final_train_loss': epoch_history['train_loss'][-1] if epoch_history.get('train_loss') else None,
            'final_val_loss': epoch_history['val_loss'][-1] if epoch_history.get('val_loss') else None,
            'epochs_trained': epoch_history.get('epochs_trained', 0),
            'data_points': len(epoch_history.get('train_loss', []))
        },
        'batch_based_results': {
            'final_train_loss': batch_history['train_loss'][-1] if batch_history.get('train_loss') else None,
            'final_val_loss': batch_history['val_loss'][-1] if batch_history.get('val_loss') else None,
            'batches_trained': batch_history.get('batches_trained', 0),
            'data_points': len(batch_history.get('train_loss', []))
        },
        'comparison_notes': {
            'epoch_granularity': 'Epoch-based training logs once per epoch',
            'batch_granularity': f'Batch-based training logs every {batch_history.get("log_every_n_batches", "N")} batches',
            'data_density': f'Batch-based provides {len(batch_history.get("train_loss", []))} vs {len(epoch_history.get("train_loss", []))} data points'
        }
    }
    
    # Log comparison
    logger.log_custom_metrics(comparison_data, "training_methods_comparison")
    
    print(f"\n✓ Comparison summary logged to: {logger.get_experiment_path()}")
    
    return epoch_history, batch_history


def run_high_granularity_experiment():
    """
    Example of very high granularity batch-based training.
    """
    print("HIGH GRANULARITY EXPERIMENT")
    print("="*60)
    
    data_loader = MNISTDataLoader(batch_size=64)
    model = model_type
    trainer = FlexibleTrainer(model, data_loader)
    logger = TrainingLogger("high_granularity_experiment")
    
    # Very granular logging settings
    training_config = {
        'training_type': 'high_granularity_batch_based',
        'training_digits': [0, 1, 2, 3, 4],
        'monitor_digits': [5, 6, 7, 8, 9],
        'total_batches': 1000,
        'log_every_n_batches': 25,   # Very frequent logging
        'val_every_n_batches': 50,   # Frequent validation
        'learning_rate': 0.001,
        'batch_size': 64,
        'early_stopping': True,
        'early_stopping_patience': 5
    }
    
    print(f"Ultra-granular training:")
    print(f"  - Logging every {training_config['log_every_n_batches']} batches")
    print(f"  - Validation every {training_config['val_every_n_batches']} batches")
    print(f"  - Expected ~{training_config['total_batches'] // training_config['log_every_n_batches']} log points")
    
    # Train with high granularity
    history = trainer.train_on_digits_batch_based(
        training_digits=training_config['training_digits'],
        total_batches=training_config['total_batches'],
        log_every_n_batches=training_config['log_every_n_batches'],
        val_every_n_batches=training_config['val_every_n_batches'],
        learning_rate=training_config['learning_rate'],
        batch_size=training_config['batch_size'],
        monitor_digits=training_config['monitor_digits'],
        early_stopping=training_config['early_stopping'],
        early_stopping_patience=training_config['early_stopping_patience'],
        verbose=True
    )
    
    # Log with detailed analysis
    logger.log_training_run(
        training_history=history,
        model_info=model.get_model_summary(),
        training_config=training_config,
        additional_info={
            'experiment_type': 'high_granularity_analysis',
            'purpose': 'Detailed training dynamics analysis',
            'data_points_captured': len(history.get('train_loss', [])),
            'granularity_benefit': 'Can observe micro-patterns in loss progression',
            'use_case': 'Research into training instabilities, loss spikes, convergence patterns'
        }
    )
    
    print(f"✓ High granularity experiment logged to: {logger.get_experiment_path()}")
    print(f"  Captured {len(history.get('train_loss', []))} training data points")
    
    return history


def run_simple_batch_experiment():
    """
    Minimal example of batch-based training - just the essentials.
    """
    print("SIMPLE BATCH-BASED EXPERIMENT")
    print("="*60)
    
    # Quick setup
    data_loader = MNISTDataLoader(batch_size=64)
    model = model_type
    trainer = FlexibleTrainer(model, data_loader)
    logger = TrainingLogger("simple_batch_experiment")
    
    # Simple configuration
    training_config = {
        'training_type': 'simple_batch_based',
        'training_digits': [0, 1, 2],
        'total_batches': 300,
        'log_every_n_batches': 100,
        'val_every_n_batches': 100,
        'learning_rate': 0.001,
        'batch_size': 64,
        'early_stopping': False  # Disabled for simplicity
    }
    
    print("Simple batch-based training:")
    print(f"  - Training digits: {training_config['training_digits']}")
    print(f"  - Total batches: {training_config['total_batches']}")
    print(f"  - No early stopping")
    
    # Train
    history = trainer.train_on_digits_batch_based(
        training_digits=training_config['training_digits'],
        total_batches=training_config['total_batches'],
        log_every_n_batches=training_config['log_every_n_batches'],
        val_every_n_batches=training_config['val_every_n_batches'],
        learning_rate=training_config['learning_rate'],
        batch_size=training_config['batch_size'],
        early_stopping=training_config['early_stopping'],
        verbose=True
    )
    
    # Log
    logger.log_training_run(
        training_history=history,
        model_info=model.get_model_summary(),
        training_config=training_config,
        additional_info={'experiment_notes': 'Simple batch-based training example'}
    )
    
    print(f"✓ Simple batch experiment logged to: {logger.get_experiment_path()}")
    return history

def run_simple_batch_experiment_with_checkpointer():
    """
    Minimal example of batch-based training with checkpointing.
    """
    print("SIMPLE BATCH-BASED EXPERIMENT WITH CHECKPOINTING")
    print("="*60)
    experiment_name = "simple_batch_experiment_with_checkpointer"
    # Quick setup
    data_loader = MNISTDataLoader(batch_size=64)
    model = model_type
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = FlexibleTrainer(model, data_loader)
    logger = TrainingLogger(experiment_name)
    # Simple configuration
    training_config = {
        'training_type': 'simple_batch_based',
        'training_digits': [0, 1, 2],
        'total_batches': 200,
        'log_every_n_batches': 100,
        'val_every_n_batches': 100,
        'learning_rate': 0.001,
        'batch_size': 64,
        'early_stopping': False  # Disabled for simplicity
    }
    
    print("Simple batch-based training:")
    print(f"  - Training digits: {training_config['training_digits']}")
    print(f"  - Total batches: {training_config['total_batches']}")
    print(f"  - No early stopping")
    
    checkpointer = ModelCheckpointer(
                                model=model,
                                optimizer=optimizer,
                                experiment_name=experiment_name,
                                save_every_n_batches=100  # Save every 100 batches (3 saves total for 300 batches)
                                )

    # Train
    history = trainer.train_on_digits_batch_based(
        training_digits=training_config['training_digits'],
        total_batches=training_config['total_batches'],
        log_every_n_batches=training_config['log_every_n_batches'],
        val_every_n_batches=training_config['val_every_n_batches'],
        learning_rate=training_config['learning_rate'],
        batch_size=training_config['batch_size'],
        early_stopping=training_config['early_stopping'],
        verbose=True,
        checkpointer=checkpointer,
        optimizer=optimizer
    )

    # Final checkpoint (best model based on final validation loss)
    if 'val_loss' in history and history['val_loss']:
        final_val_loss = [loss for loss in history['val_loss'] if loss is not None][-1]
        checkpointer.save_best_model(
            current_metric=final_val_loss,
            batch=training_config['total_batches'],
            is_higher_better=False,  # Lower loss is better
            training_history=history,
            training_config=training_config,
            model_info=model.get_model_summary()
        )
    # Log
    logger.log_training_run(
        training_history=history,
        model_info=model.get_model_summary(),
        training_config=training_config,
        additional_info={'experiment_notes': 'Simple batch-based training example'}
    )
    
    print(f"✓ Simple batch experiment logged to: {logger.get_experiment_path()}")
    print(f"✓ Checkpoints saved to: {checkpointer.get_experiment_path()}")

    return history, checkpointer



if __name__ == "__main__":
    print("FlexibleTrainer + TrainingLogger Integration Examples")
    print("="*70)
    
    print("Available experiments:")
    print("1. Epoch-based training (original)")
    print("2. Batch-based training")
    print("3. Simple batch-based training")
    print("4. High granularity batch-based training")
    print("5. Compare epoch vs batch training")
    print("6. Simple batch-based training with checkpointing")
    print("7. Run all experiments")
    
    choice = input("\nEnter choice (1-7): ").strip()

    if choice == "1":
        run_epoch_based_experiment()
    
    elif choice == "2":
        run_batch_based_experiment()
    
    elif choice == "3":
        run_simple_batch_experiment()
    
    elif choice == "4":
        run_high_granularity_experiment()
    
    elif choice == "5":
        run_comparison_experiment()

    elif choice == "6":
        run_simple_batch_experiment_with_checkpointer()
        
    elif choice == "7":
        print("Running all experiments...")
        print("\n" + "="*70)
        
        # Run all experiments
        run_epoch_based_experiment()
        print("\n" + "-"*50)
        run_simple_batch_experiment()
        print("\n" + "-"*50)
        run_batch_based_experiment() 
        print("\n" + "-"*50)
        run_high_granularity_experiment()
        print("\n" + "-"*50)
        run_comparison_experiment()
        
        print("\n" + "="*70)
        print("ALL EXPERIMENTS COMPLETED!")
        print("Check the 'training_logs' directory for all results.")
    
    else:
        print("Invalid choice. Running simple batch-based experiment by default...")
        run_simple_batch_experiment()
    
    print(f"\n{'='*70}")
    print("Integration examples completed!")
    print("Key features demonstrated:")
    print("✓ Epoch-based training with logging")
    print("✓ Batch-based training with configurable granularity")  
    print("✓ Automatic detection of training type by logger")
    print("✓ Catastrophic forgetting monitoring")
    print("✓ Early stopping with batch-based training")
    print("✓ Flexible validation frequency")
    print("✓ Clean separation: Trainer trains, Logger logs")