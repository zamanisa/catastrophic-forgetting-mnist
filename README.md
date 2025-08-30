# catastrophic-forgetting-mnist

This project studies catastrophic forgetting in neural networks by training on different subsets of MNIST digits.

## Setup

### 1. Create Conda Virtual Environment

```bash
# Create a new conda environment
conda create -n mnist python=3.11

# Activate the environment
conda activate mnist
```

### 2. Install Required Packages

```bash
# Install all necessary packages
pip install torch torchvision torchaudio datasets matplotlib numpy scikit-learn pillow tqdm transformers
```

**Alternative: Using requirements.txt**
```bash
# If you have requirements.txt file
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Test that PyTorch is working
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Running the Experiment

### Basic Usage

```bash
# Run the main experiment
python example_test.py
```

### What the Experiment Does

1. **Phase 1**: Trains a neural network on even digits (0, 2, 4, 6, 8)
2. **Phase 2**: Trains the same network on odd digits (1, 3, 5, 7, 9) 
3. **Measures**: How much the network "forgets" the even digits (catastrophic forgetting)

### Expected Output

```
MODEL ARCHITECTURE
======================================================================
Input Layer:    784 (28x28 flattened)
Hidden Layer 1: 1024 neurons (ReLU + Dropout)
Hidden Layer 2:  512 neurons (ReLU + Dropout)
Output Layer:    10 neurons (no activation)
----------------------------------------------------------------------

PHASE 1: Training on Even Digits
Training samples: 23,604
Validation samples: 5,888

Epoch   1: Train Acc:  94.83% | Val Acc:  97.84% | ...
Epoch   2: Train Acc:  97.74% | Val Acc:  97.98% | ...
...

PHASE 2: Training on Odd Digits (monitoring even digits)
...

FINAL RESULTS
======================================================================
Even digits accuracy after Phase 1: 98.25%
Even digits accuracy after Phase 2: 73.20%
Catastrophic Forgetting Amount:     25.05%
```

## Project Structure

```
mnist_catastrophic_forgetting/
├── requirements.txt                    # Package dependencies
├── mnist_data_prep.py                 # Data loading utilities
├── mnist_catastrophic_forgetting.py   # Model and training code
├── run_experiment.py                  # Main experiment script
└── results/                           # Output folder (optional)
```

## Customization

### Change Model Architecture

```python
# In your script, modify the hidden layers
model = MNISTModel([2000, 1000, 500])  # 784-2000-1000-500-10
model = MNISTModel([512])              # 784-512-10
model = MNISTModel([1024, 1024, 1024]) # Deep network
```

### Change Training Digits

```python
# Train on different digit subsets
phase1_history = trainer.train_on_digits([0, 1, 2], epochs=30)
phase2_history = trainer.train_on_digits([7, 8, 9], monitor_digits=[0, 1, 2])
```

### Adjust Training Parameters

```python
# Modify training settings
trainer.train_on_digits(
    training_digits=[0, 2, 4, 6, 8],
    epochs=50,           # More epochs
    learning_rate=0.0001, # Lower learning rate
    batch_size=128       # Larger batch size
)
```

## Troubleshooting

### Common Issues

**1. PyTorch Installation Problems**
```bash
# Uninstall and reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**2. Memory Issues**
- Reduce batch size: `MNISTDataLoader(batch_size=32)`
- Use smaller model: `MNISTModel([512, 256])`

**3. Import Errors**
- Make sure all files are in the same directory
- Check that conda environment is activated: `conda activate mnist`

### Getting Help

If you encounter issues:
1. Check that all packages are installed correctly
2. Verify your conda environment is activated
3. Make sure Python version is 3.8 or higher: `python --version`

## Deactivating Environment

When done working:
```bash
# Deactivate the conda environment
conda deactivate
```

## System Requirements

- **RAM**: 8GB+ recommended (16GB for larger models)
- **Python**: 3.8 or higher
- **Storage**: ~2GB for packages and datasets
- **GPU**: Optional (will use CPU by default)
