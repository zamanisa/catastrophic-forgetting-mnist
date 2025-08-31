"""
Example: How to integrate TrainingLogger with your existing FlexibleTrainer

This script demonstrates the clean integration pattern where:
- FlexibleTrainer handles training
- TrainingLogger handles logging
- Both remain independent and reusable
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from mnist_data_prep import MNISTDataLoader
from mnist_model import MNISTModelNN
from trainer import FlexibleTrainer
from evaluation_and_logger import TrainingLogger


def run_experiment_with_logging():
    """
    Example of running a complete experiment with comprehensive logging.
    """
    print("Starting MNIST experiment with comprehensive logging...")
    print("="*60)
    
    # 1. Setup data
    print("Setting up data...")
    data_loader = MNISTDataLoader(batch_size=64, validation_split=0.2)
    print("✓ Data loaded")
    
    # 2. Create model
    print("\nCreating model...")
    model = MNISTModelNN(hidden_layers=[1024, 512, 256], dropout_rate=0.3)
    print("✓ Model created")
    
    # 3. Create trainer
    print("\nInitializing trainer...")
    trainer = FlexibleTrainer(model, data_loader)
    print("✓ Trainer ready")
    
    # 4. Create logger
    print("\nInitializing logger...")
    logger = TrainingLogger(
        experiment_name="mnist_catastrophic_forgetting_test",
        log_dir="training_logs"
    )
    print("✓ Logger ready")
    
    # 5. Define experiment parameters
    training_digits = [0, 1, 2]  # Train on these digits
    monitor_digits = [7, 8, 9]   # Monitor these for catastrophic forgetting
    
    training_config = {
        'training_digits': training_digits,
        'monitor_digits': monitor_digits,
        'epochs': 2,
        'learning_rate': 0.001,
        'batch_size': 64,
        'early_stopping_patience': 5,
        'architecture_type': 'feedforward'
    }
    
    print(f"\nExperiment Configuration:")
    print(f"  Training digits: {training_digits}")
    print(f"  Monitor digits: {monitor_digits}")
    print(f"  Max epochs: {training_config['epochs']}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    
    # 6. Run training
    print(f"\nStarting training...")
    print("-" * 60)
    
    history = trainer.train_on_digits(
        training_digits=training_digits,
        epochs=training_config['epochs'],
        learning_rate=training_config['learning_rate'],
        batch_size=training_config['batch_size'],
        monitor_digits=monitor_digits,
        early_stopping_patience=training_config['early_stopping_patience'],
        verbose=True
    )
    
    print("-" * 60)
    print("✓ Training completed")
    
    # 7. Log everything
    print(f"\nLogging experiment results...")
    
    # Get model info
    model_info = model.get_model_summary()
    
    # Additional experiment info
    additional_info = {
        'experiment_type': 'catastrophic_forgetting_analysis',
        'hypothesis': 'Training on digits 0,1,2 should maintain performance on digits 7,8,9',
        'notes': 'Baseline experiment for catastrophic forgetting research',
        'dataset_info': data_loader.get_dataset_info()
    }
    
    # Log the complete experiment
    logger.log_training_run(
        training_history=history,
        model_info=model_info,
        training_config=training_config,
        additional_info=additional_info
    )
    
    # 8. Log additional custom metrics (optional)
    print("\nComputing additional metrics...")
    
    # Get performance on all digit groups for comprehensive analysis
    digit_groups = {
        'trained_digits': training_digits,
        'monitored_digits': monitor_digits,
        'all_even': [0, 2, 4, 6, 8],
        'all_odd': [1, 3, 5, 7, 9],
        'low_digits': [0, 1, 2, 3, 4],
        'high_digits': [5, 6, 7, 8, 9]
    }
    
    performance_breakdown = trainer.get_performance_summary(digit_groups)
    
    logger.log_custom_metrics(
        metrics_dict=performance_breakdown,
        filename="detailed_performance_breakdown"
    )
    
    print("✓ Additional metrics logged")
    
    # 9. Print summary
    print(f"\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Experiment logs saved to: {logger.get_experiment_path()}")
    print(f"\nKey Results:")
    print(f"  Best validation accuracy: {history['best_val_accuracy']:.2f}%")
    print(f"  Epochs trained: {history['epochs_trained']}")
    
    if history.get('monitor_test_accuracy'):
        initial_monitor = history['monitor_test_accuracy'][0]
        final_monitor = history['monitor_test_accuracy'][-1]
        change = final_monitor - initial_monitor
        print(f"  Monitor digits performance change: {change:+.2f}% ({initial_monitor:.2f}% → {final_monitor:.2f}%)")
    
    print(f"\nFiles created:")
    for file in logger.get_experiment_path().glob("*.csv"):
        print(f"  📊 {file.name}")
    for file in logger.get_experiment_path().glob("*.json"):
        print(f"  📄 {file.name}")
    
    return logger.get_experiment_path()


def run_simple_experiment():
    """
    Minimal example - just the essentials.
    """
    print("Running simple experiment with basic logging...")
    
    # Quick setup
    data_loader = MNISTDataLoader(batch_size=32)
    model = MNISTModelNN([512, 256])
    trainer = FlexibleTrainer(model, data_loader)
    logger = TrainingLogger("simple_mnist_test")
    
    # Train
    history = trainer.train_on_digits(
        training_digits=[0, 1, 2, 3, 5],
        epochs=3,
        verbose=True
    )
    
    # Log
    logger.log_training_run(
        training_history=history,
        model_info=model.get_model_summary(),
        training_config={
            'training_digits': [0, 1],
            'epochs': 10,
            'learning_rate': 0.001,
            'batch_size': 32
        }
    )
    
    print(f"Simple experiment logged to: {logger.get_experiment_path()}")


if __name__ == "__main__":
    # You can run either example
    
    print("Choose experiment type:")
    print("1. Comprehensive experiment (with catastrophic forgetting analysis)")
    print("2. Simple experiment (basic training)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        experiment_path = run_experiment_with_logging()
    elif choice == "2":
        run_simple_experiment()
    else:
        print("Running comprehensive experiment by default...")
        experiment_path = run_experiment_with_logging()
