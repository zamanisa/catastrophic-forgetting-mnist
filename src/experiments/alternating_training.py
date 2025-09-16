"""
Alternating Training Script

This script trains a model by alternating between two digit sets for specified periods.
Useful for studying catastrophic forgetting and continual learning patterns.

Usage:
    from alternating_training import run_alternating_experiment
    
    config = {
        'set_A': [0, 1, 2],
        'set_B': [7, 8, 9], 
        'period_type': 'epochs',  # or 'batches'
        'period_length': 3,       # 3 epochs or 3 batches per set
        'total_cycles': 4,        # A->B->A->B (4 total periods)
        'learning_rate': 0.001,
        'batch_size': 64,
        # ... other training params
    }
    
    history, logger = run_alternating_experiment(model, config, "my_alternating_experiment")
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from typing import Dict, List, Tuple, Any
from utils.digit_filter import DigitFilter  

from utils.unified_training_function import run_training_experiment
from loggings.training_logger import TrainingLogger


def run_alternating_experiment(
    model,
    alternating_config: Dict,
    experiment_name: str,
    checkpointer=None,
    optimizer=None,
    criterion=None,
    verbose: bool = True
) -> Tuple[Dict, TrainingLogger]:
    """
    Run alternating training experiment between two digit sets.
    
    Args:
        model: The neural network model to train
        alternating_config (dict): Configuration for alternating training
        experiment_name (str): Name for the experiment
        checkpointer: Optional checkpointer for saving models
        optimizer: Optional optimizer (created if None)
        criterion: Optional loss criterion
        verbose: Whether to print progress
        
    Returns:
        tuple: (complete_history, logger) - Combined history and logger
        
    Alternating Config Keys:
        Required:
            - set_A (list): First set of digits to train on
            - set_B (list): Second set of digits to train on
            - period_type (str): 'epochs' or 'batches'
            - period_length (int): Length of each training period
            - total_cycles (int): Total number of periods (A->B->A->B...)
            
        Optional:
            - learning_rate (float, default=0.001): Learning rate
            - batch_size (int, default=64): Batch size
            - validation_split (float, default=0.2): Validation split
            - early_stopping_patience (int, default=3): Early stopping patience
            
        Period-specific (inherited from unified_training_function):
            For epochs: early_stopping_patience
            For batches: log_every_n_batches, val_every_n_batches, early_stopping
    """
    
    # Validate required configuration
    required_keys = ['set_A', 'set_B', 'period_type', 'period_length', 'total_cycles']
    for key in required_keys:
        if key not in alternating_config:
            raise ValueError(f"'{key}' is required in alternating_config")
    
    if alternating_config['period_type'] not in ['epochs', 'batches']:
        raise ValueError("period_type must be 'epochs' or 'batches'")
    
    # Set defaults
    config = alternating_config.copy()
    config.setdefault('learning_rate', 0.001)
    config.setdefault('batch_size', 64)
    config.setdefault('validation_split', 0.2)
    config.setdefault('early_stopping_patience', 3)
    
    # For batch-based training defaults
    if config['period_type'] == 'batches':
        config.setdefault('log_every_n_batches', min(100, config['period_length'] // 4))
        config.setdefault('val_every_n_batches', min(100, config['period_length'] // 2))
        config.setdefault('early_stopping', True)
    
    # Setup logger for the complete experiment
    logger = TrainingLogger(experiment_name)
    
    # Initialize tracking
    complete_history = {
        'alternating_config': config,
        'periods': [],
        'set_A_performance_over_time': [],
        'set_B_performance_over_time': [],
        'period_summaries': []
    }
    
    if verbose:
        print(f"ALTERNATING TRAINING EXPERIMENT: {experiment_name}")
        print("=" * 60)
        print(f"Set A: {config['set_A']}")
        print(f"Set B: {config['set_B']}")
        print(f"Period type: {config['period_type']}")
        print(f"Period length: {config['period_length']}")
        print(f"Total cycles: {config['total_cycles']}")
        print(f"Training sequence: {_generate_sequence_preview(config)}")
        print("-" * 60)
    
    # Run alternating training cycles
    for cycle in range(config['total_cycles']):
        # Determine which set to train on (alternate A->B->A->B...)
        if cycle % 2 == 0:
            current_set = config['set_A']
            current_set_name = 'A'
            monitor_set = config['set_B']
            monitor_set_name = 'B'
        else:
            current_set = config['set_B']
            current_set_name = 'B'
            monitor_set = config['set_A']
            monitor_set_name = 'A'
        
        if verbose:
            print(f"\nCYCLE {cycle + 1}/{config['total_cycles']}: Training on Set {current_set_name} {current_set}")
            print(f"Monitoring Set {monitor_set_name} {monitor_set} for forgetting")
        
        # Create training config for this period
        period_config = {
            'training_digits': current_set,
            'monitor_digits': monitor_set,
            'learning_rate': config['learning_rate'],
            'batch_size': config['batch_size'],
            'validation_split': config['validation_split'],
            'early_stopping_patience': config['early_stopping_patience']
        }
        
        # Add period-specific parameters
        if config['period_type'] == 'epochs':
            period_config['epochs'] = config['period_length']
        else:  # batches
            period_config['total_batches'] = config['period_length']
            period_config['log_every_n_batches'] = config['log_every_n_batches']
            period_config['val_every_n_batches'] = config['val_every_n_batches']
            period_config['early_stopping'] = config['early_stopping']
        
        # Run training for this period
        period_name = f"{experiment_name}_cycle_{cycle+1}_set_{current_set_name}"
        period_history, period_logger = run_training_experiment(
            model=model,
            training_config=period_config,
            experiment_name=period_name,
            checkpointer=checkpointer,
            optimizer=optimizer,
            criterion=criterion
        )
        
        # Store period results
        period_summary = {
            'cycle': cycle + 1,
            'set_trained': current_set_name,
            'digits_trained': current_set,
            'digits_monitored': monitor_set,
            'period_config': period_config,
            'final_train_accuracy': _get_final_value(period_history, 'train_accuracy'),
            'final_val_accuracy': _get_final_value(period_history, 'val_accuracy'),
            'final_monitor_accuracy': _get_final_value(period_history, 'monitor_test_accuracy'),
            'period_logger_path': str(period_logger.get_experiment_path())
        }
        
        complete_history['periods'].append(period_history)
        complete_history['period_summaries'].append(period_summary)
        
        # Track performance on both sets over time
        # (This would require evaluating the model on both sets after each period)
        # For now, we'll store the available information
        if current_set_name == 'A':
            complete_history['set_A_performance_over_time'].append({
                'cycle': cycle + 1,
                'train_accuracy': period_summary['final_train_accuracy'],
                'val_accuracy': period_summary['final_val_accuracy']
            })
            complete_history['set_B_performance_over_time'].append({
                'cycle': cycle + 1,
                'monitor_accuracy': period_summary['final_monitor_accuracy']
            })
        else:
            complete_history['set_B_performance_over_time'].append({
                'cycle': cycle + 1,
                'train_accuracy': period_summary['final_train_accuracy'],
                'val_accuracy': period_summary['final_val_accuracy']
            })
            complete_history['set_A_performance_over_time'].append({
                'cycle': cycle + 1,
                'monitor_accuracy': period_summary['final_monitor_accuracy']
            })
        
        if verbose:
            print(f"Cycle {cycle + 1} completed:")
            print(f"  Set {current_set_name} accuracy: {period_summary['final_val_accuracy']:.2f}%")
            print(f"  Set {monitor_set_name} accuracy: {period_summary['final_monitor_accuracy']:.2f}%")
    
    # Log the complete alternating experiment
    if verbose:
        print(f"\nALTERNATING EXPERIMENT COMPLETED!")
        print("=" * 60)
        _print_experiment_summary(complete_history, config)
    
    # Create summary for logging
    experiment_summary = _create_experiment_summary(complete_history, config)
    
    # Log the complete experiment
    logger.log_custom_metrics(
        experiment_summary,
        "alternating_experiment_summary"
    )
    
    if verbose:
        print(f"\nComplete experiment logged to: {logger.get_experiment_path()}")
        print("Individual period logs available in their respective directories.")
    
    return complete_history, logger


def _generate_sequence_preview(config: Dict) -> str:
    """Generate a preview of the training sequence."""
    set_A, set_B = config['set_A'], config['set_B']
    cycles = min(config['total_cycles'], 6)  # Show max 6 for brevity
    
    sequence = []
    for i in range(cycles):
        if i % 2 == 0:
            sequence.append(f"A{set_A}")
        else:
            sequence.append(f"B{set_B}")
    
    if config['total_cycles'] > 6:
        sequence.append("...")
    
    return " -> ".join(sequence)


def _get_final_value(history: Dict, key: str) -> float:
    """Get the final value from a history list, handling None values."""
    if key not in history or not history[key]:
        return 0.0
    
    values = [v for v in history[key] if v is not None]
    return values[-1] if values else 0.0


def _print_experiment_summary(complete_history: Dict, config: Dict):
    """Print a summary of the alternating experiment."""
    print("EXPERIMENT SUMMARY:")
    print("-" * 40)
    
    set_A, set_B = config['set_A'], config['set_B']
    
    for i, summary in enumerate(complete_history['period_summaries']):
        cycle = summary['cycle']
        set_name = summary['set_trained']
        train_acc = summary['final_val_accuracy']
        monitor_acc = summary['final_monitor_accuracy']
        
        if set_name == 'A':
            print(f"Cycle {cycle}: Set A{set_A} -> {train_acc:.1f}%, Set B{set_B} -> {monitor_acc:.1f}%")
        else:
            print(f"Cycle {cycle}: Set B{set_B} -> {train_acc:.1f}%, Set A{set_A} -> {monitor_acc:.1f}%")


def _create_experiment_summary(complete_history: Dict, config: Dict) -> Dict:
    """Create a comprehensive summary of the alternating experiment."""
    
    # Extract performance trends
    set_A_accuracies = []
    set_B_accuracies = []
    
    for summary in complete_history['period_summaries']:
        if summary['set_trained'] == 'A':
            set_A_accuracies.append(summary['final_val_accuracy'])
            set_B_accuracies.append(summary['final_monitor_accuracy'])
        else:
            set_A_accuracies.append(summary['final_monitor_accuracy'])
            set_B_accuracies.append(summary['final_val_accuracy'])
    
    return {
        'experiment_type': 'alternating_training',
        'configuration': config,
        'results_summary': {
            'total_cycles_completed': len(complete_history['period_summaries']),
            'set_A_final_accuracy': set_A_accuracies[-1] if set_A_accuracies else 0,
            'set_B_final_accuracy': set_B_accuracies[-1] if set_B_accuracies else 0,
            'set_A_accuracy_trend': set_A_accuracies,
            'set_B_accuracy_trend': set_B_accuracies
        },
        'forgetting_analysis': {
            'set_A_max_drop': max(set_A_accuracies) - min(set_A_accuracies) if set_A_accuracies else 0,
            'set_B_max_drop': max(set_B_accuracies) - min(set_B_accuracies) if set_B_accuracies else 0,
            'average_forgetting': _calculate_average_forgetting(set_A_accuracies, set_B_accuracies)
        },
        'period_details': complete_history['period_summaries']
    }


def _calculate_average_forgetting(set_A_accs: List[float], set_B_accs: List[float]) -> float:
    """Calculate average forgetting across all transitions."""
    if len(set_A_accs) < 2 or len(set_B_accs) < 2:
        return 0.0
    
    forgetting_events = []
    
    # When we switch from A to B, we measure A's forgetting
    # When we switch from B to A, we measure B's forgetting
    for i in range(1, len(set_A_accs)):
        if i % 2 == 1:  # Odd cycles train on B, so A might be forgotten
            prev_A = set_A_accs[i-1]
            curr_A = set_A_accs[i]
            if prev_A > 0:
                forgetting_events.append(max(0, prev_A - curr_A))
        else:  # Even cycles train on A, so B might be forgotten
            prev_B = set_B_accs[i-1]
            curr_B = set_B_accs[i]
            if prev_B > 0:
                forgetting_events.append(max(0, prev_B - curr_B))
    
    return sum(forgetting_events) / len(forgetting_events) if forgetting_events else 0.0


# Example usage and testing
if __name__ == "__main__":
    from models.ff_nn import MNISTModelNN
    from models.graph_nn import MNISTGraphNN
    
    print("Alternating Training Script - Example Usage")
    print("=" * 60)
    
    # Example model
    model = MNISTGraphNN(
        hidden_dims=[32, 64, 32],
        connectivity_radius=1,
        use_position_features=False,
        pooling_method='max',
        dropout_rate=0.2
    )
    
    # Example configuration
    alternating_config = {
        'set_A': [0, 1, 2],
        'set_B': [7, 8, 9],
        'period_type': 'epochs',  # or 'batches'
        'period_length': 2,       # 2 epochs per set
        'total_cycles': 4,        # A->B->A->B (4 total periods)
        'learning_rate': 0.001,
        'batch_size': 64
    }
    
    print("Configuration:")
    print(f"  Set A: {alternating_config['set_A']}")
    print(f"  Set B: {alternating_config['set_B']}")
    print(f"  Period: {alternating_config['period_length']} {alternating_config['period_type']}")
    print(f"  Total cycles: {alternating_config['total_cycles']}")
    
    # Run alternating experiment
    history, logger = run_alternating_experiment(
        model=model,
        alternating_config=alternating_config,
        experiment_name="example_alternating_experiment"
    )
    
    print(f"\nExample alternating experiment completed!")
    print(f"Results logged to: {logger.get_experiment_path()}")
