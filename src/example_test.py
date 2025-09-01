#%%
from trainer import FlexibleTrainer
from mnist_data_prep import MNISTDataLoader
from models.ff_nn import MNISTModelNN

#%%
# Setup
whole_set = [i for i in range(10)]
first_set = [0, 2, 4, 6, 8]
second_set = [i for i in whole_set if i not in first_set]
data_loader = MNISTDataLoader(batch_size=64)
model = MNISTModelNN([1024, 512], dropout_rate=0.5)
#%%
trainer = FlexibleTrainer(model, data_loader)
# Phase 1: Train on your custom subset (e.g., even digits)
phase1_history = trainer.train_on_digits(
    training_digits=first_set,
    epochs=5,
    learning_rate=0.001
)
#%%
# Phase 2: Train on different subset while monitoring the first
c = trainer.train_on_digits(
    training_digits=second_set,
    epochs=5,
    monitor_digits=first_set  # This tracks forgetting!
)
