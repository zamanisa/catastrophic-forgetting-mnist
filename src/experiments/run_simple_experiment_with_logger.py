#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import FlexibleTrainer
from mnist_data_prep import MNISTDataLoader
from models.ff_nn import MNISTModelNN
from training_logger import TrainingLogger
print('we are here')
# %%
# Setup
whole_set = [i for i in range(10)]
first_set = [0, 2, 4, 6, 8]
second_set = [i for i in whole_set if i not in first_set]
data_loader = MNISTDataLoader(batch_size=64)
model = MNISTModelNN([512, 512], dropout_rate=0.5)

# Create loggers for each phase
phase1_logger = TrainingLogger("phase1_even_digits")
phase2_logger = TrainingLogger("phase2_odd_digits_with_monitoring")

print(f"Phase 1 training set (even): {first_set}")
print(f"Phase 2 training set (odd): {second_set}")

#%%
trainer = FlexibleTrainer(model, data_loader)

# Phase 1: Train on your custom subset (e.g., even digits)
print("\n" + "="*60)
print("PHASE 1: Training on even digits")
print("="*60)

phase1_history = trainer.train_on_digits(
    training_digits=first_set,
    epochs=2,
    learning_rate=0.001
)

# Log Phase 1
phase1_config = {
    'phase': 1,
    'training_digits': first_set,
    'epochs': 5,
    'learning_rate': 0.001,
    'batch_size': 64,
    'experiment_type': 'baseline_training'
}

phase1_logger.log_training_run(
    training_history=phase1_history,
    model_info=model.get_model_summary(),
    training_config=phase1_config,
    additional_info={
        'phase_description': 'Training on even digits (0,2,4,6,8) - baseline phase',
        'next_phase_plan': 'Will train on odd digits while monitoring even digit performance'
    }
)

print(f"✅ Phase 1 logged to: {phase1_logger.get_experiment_path()}")

#%%
# Phase 2: Train on different subset while monitoring the first
print("\n" + "="*60)
print("PHASE 2: Training on odd digits while monitoring even digits")
print("="*60)

phase2_history = trainer.train_on_digits(
    training_digits=second_set,
    epochs=2,
    monitor_digits=first_set  # This tracks forgetting!
)

# Log Phase 2
phase2_config = {
    'phase': 2,
    'training_digits': second_set,
    'monitor_digits': first_set,
    'epochs': 5,
    'learning_rate': 0.001,  # Default learning rate
    'batch_size': 64,
    'experiment_type': 'catastrophic_forgetting_analysis'
}

phase2_logger.log_training_run(
    training_history=phase2_history,
    model_info=model.get_model_summary(),
    training_config=phase2_config,
    additional_info={
        'phase_description': 'Training on odd digits (1,3,5,7,9) while monitoring even digit performance',
        'previous_phase': 'Trained on even digits first',
        'catastrophic_forgetting_hypothesis': 'Performance on even digits should decrease during odd digit training'
    }
)

print(f"✅ Phase 2 logged to: {phase2_logger.get_experiment_path()}")

