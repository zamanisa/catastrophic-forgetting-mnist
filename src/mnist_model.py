import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

class MNISTModel(nn.Module):
    """
    Flexible feedforward neural network for MNIST digit classification.
    Automatically adapts to any architecture you specify.
    """
    
    def __init__(self, hidden_layers: List[int] = [2000, 1500, 1000, 500], dropout_rate: float = 0.3):
        """
        Initialize model with flexible architecture.
        
        Args:
            hidden_layers: List of hidden layer sizes (e.g., [2000, 1500, 1000, 500])
            dropout_rate: Dropout probability for regularization
        """
        super(MNISTModel, self).__init__()
        
        # Store architecture info
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.layer_sizes = [784] + hidden_layers + [10]  # Input + hidden + output
        
        # Create layers dynamically
        self.layers = nn.ModuleList()
        for i in range(len(self.layer_sizes) - 1):
            self.layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._init_weights()
        
        # Print architecture automatically
        self.print_architecture()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass through all layers dynamically."""
        # Flatten input if needed
        x = x.view(x.size(0), -1)
        
        # Pass through all layers except the last one
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final layer (no activation, no dropout)
        x = self.layers[-1](x)
        
        return x
    
    def get_layer_outputs(self, x):
        """Get outputs from all intermediate layers for analysis."""
        x = x.view(x.size(0), -1)
        
        outputs = {}
        outputs['input'] = x.clone()
        
        # Pass through all layers and store outputs
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
            outputs[f'layer{i+1}'] = x.clone()
        
        # Final layer
        x = self.layers[-1](x)
        outputs['output'] = x
        
        return outputs

    def get_model_summary(self):
        """Get architecture summary as dictionary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Dynamic architecture string
        arch_str = '-'.join(map(str, self.layer_sizes))
        
        # Dynamic layer info
        layer_info = [{'name': 'Input', 'size': 784, 'activation': None}]
        
        for i, size in enumerate(self.hidden_layers):
            layer_info.append({
                'name': f'Hidden{i+1}', 
                'size': size, 
                'activation': 'ReLU'
            })
        
        layer_info.append({'name': 'Output', 'size': 10, 'activation': None})
        
        return {
            'architecture': arch_str,
            'layers': layer_info,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024**2),
            'dropout_rate': self.dropout_rate,
            'num_hidden_layers': len(self.hidden_layers)
        }
    
    def print_architecture(self):
        """Print model architecture summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("\n" + "="*70)
        print("MODEL ARCHITECTURE")
        print("="*70)
        print(f"Input Layer:    784 (28x28 flattened)")
        
        for i, size in enumerate(self.hidden_layers):
            print(f"Hidden Layer {i+1}: {size:4d} neurons (ReLU + Dropout)")
        
        print(f"Output Layer:    10 neurons (no activation)")
        print("-"*70)
        print(f"Architecture: {'-'.join(map(str, self.layer_sizes))}")
        print(f"Total Layers: {len(self.layer_sizes)} ({len(self.hidden_layers)} hidden)")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: ~{total_params * 4 / (1024**2):.2f} MB")
        print(f"Dropout rate: {self.dropout_rate}")
        print("="*70)

if __name__ == "__main__":
    # Example usage
    model = MNISTModel([1024, 512], dropout_rate=0.5)
    model.print_architecture()
    
    # Test forward pass
    dummy_input = torch.randn(32, 1, 28, 28)
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")