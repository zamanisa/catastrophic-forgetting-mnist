from typing import Dict

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
