import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input."""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MNISTTransformer(nn.Module):
    """
    Transformer model for MNIST digit classification.
    Drop-in compatible with MNISTModelNN interface.
    
    Architecture: Input → Patch Embedding → Positional Encoding → Transformer Layers → Classification Head
    """
    
    def __init__(
        self,
        patch_size: int = 4,           # 4x4 patches (49 patches from 28x28)
        d_model: int = 256,            # Embedding dimension
        n_heads: int = 8,              # Number of attention heads
        n_layers: int = 6,             # Number of transformer layers
        dim_feedforward: int = 512,    # FFN dimension
        dropout_rate: float = 0.1,     # Dropout rate
        max_patches: int = 100         # Maximum number of patches for positional encoding
    ):
        """
        Initialize MNIST Transformer.
        
        Args:
            patch_size: Size of square patches (4 = 4x4 patches)
            d_model: Model dimension (embedding size)
            n_heads: Number of attention heads
            n_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout_rate: Dropout probability
            max_patches: Maximum patches for positional encoding
        """
        super(MNISTTransformer, self).__init__()
        
        # Store architecture parameters
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        self.max_patches = max_patches
        
        # Calculate number of patches (28/patch_size)^2
        self.img_size = 28
        self.n_patches_per_side = self.img_size // patch_size
        self.n_patches = self.n_patches_per_side ** 2
        self.patch_dim = patch_size * patch_size  # Features per patch
        
        # Patch embedding: Linear projection of flattened patches
        self.patch_embedding = nn.Linear(self.patch_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_patches)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True  # Important: batch dimension first
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, 10)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._init_weights()
        
        # Auto-print architecture (matching MNISTModelNN behavior)
        self.print_architecture()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _patchify(self, images):
        """
        Convert images to patches.
        
        Args:
            images: (batch_size, 1, 28, 28) or (batch_size, 784)
            
        Returns:
            patches: (batch_size, n_patches, patch_dim)
        """
        batch_size = images.size(0)
        
        # Handle both input formats
        if len(images.shape) == 2:  # (batch, 784)
            images = images.view(batch_size, 1, 28, 28)
        elif len(images.shape) == 4:  # (batch, 1, 28, 28)
            pass
        else:
            raise ValueError(f"Unsupported input shape: {images.shape}")
        
        # Extract patches using unfold
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # patches shape: (batch, 1, n_patches_per_side, n_patches_per_side, patch_size, patch_size)
        
        # Reshape to (batch, n_patches, patch_dim)
        patches = patches.contiguous().view(
            batch_size, 
            self.n_patches, 
            self.patch_dim
        )
        
        return patches
    
    def forward(self, x):
        """
        Forward pass through transformer.
        
        Args:
            x: Input tensor of shape (batch, 1, 28, 28) or (batch, 784)
            
        Returns:
            Output tensor of shape (batch, 10)
        """
        # Convert to patches
        patches = self._patchify(x)  # (batch, n_patches, patch_dim)
        
        # Embed patches
        embeddings = self.patch_embedding(patches)  # (batch, n_patches, d_model)
        
        # Add positional encoding
        embeddings = self.pos_encoding(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Pass through transformer encoder
        transformer_out = self.transformer_encoder(embeddings)  # (batch, n_patches, d_model)
        
        # Global average pooling across patches
        pooled = torch.mean(transformer_out, dim=1)  # (batch, d_model)
        
        # Classification
        output = self.classifier(pooled)  # (batch, 10)
        
        return output
    
    def get_layer_outputs(self, x):
        """Get outputs from all intermediate layers for analysis."""
        outputs = {}
        
        # Input
        if len(x.shape) == 2:
            x_reshaped = x.view(x.size(0), 1, 28, 28)
        else:
            x_reshaped = x
        outputs['input'] = x_reshaped.clone()
        
        # Patches
        patches = self._patchify(x)
        outputs['patches'] = patches.clone()
        
        # Embeddings
        embeddings = self.patch_embedding(patches)
        outputs['patch_embeddings'] = embeddings.clone()
        
        # With positional encoding
        embeddings = self.pos_encoding(embeddings)
        embeddings = self.dropout(embeddings)
        outputs['pos_encoded'] = embeddings.clone()
        
        # Through transformer (we can't easily get intermediate layer outputs)
        transformer_out = self.transformer_encoder(embeddings)
        outputs['transformer_out'] = transformer_out.clone()
        
        # Pooled
        pooled = torch.mean(transformer_out, dim=1)
        outputs['pooled'] = pooled.clone()
        
        # Final output
        output = self.classifier(pooled)
        outputs['output'] = output
        
        return outputs
    
    def get_model_summary(self):
        """Get architecture summary as dictionary (compatible with MNISTModelNN)."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Create architecture string
        arch_components = [
            f"Patches({self.patch_size}x{self.patch_size})",
            f"Embed({self.patch_dim}→{self.d_model})",
            f"Trans({self.n_layers}L,{self.n_heads}H)",
            f"FFN({self.dim_feedforward})",
            f"Cls({self.d_model}→10)"
        ]
        arch_str = "-".join(arch_components)
        
        # Layer information (matching MNISTModelNN format)
        layer_info = [
            {'name': 'Input', 'size': 784, 'activation': None},
            {'name': f'Patches({self.patch_size}x{self.patch_size})', 'size': self.n_patches, 'activation': None},
            {'name': 'PatchEmbed', 'size': self.d_model, 'activation': None},
            {'name': f'Transformer({self.n_layers}L)', 'size': self.d_model, 'activation': 'Attention'},
            {'name': 'GlobalPool', 'size': self.d_model, 'activation': None},
            {'name': 'Classifier', 'size': 10, 'activation': None}
        ]
        
        return {
            'architecture': arch_str,
            'layers': layer_info,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024**2),
            'dropout_rate': self.dropout_rate,
            'num_hidden_layers': self.n_layers,
            # Transformer-specific info
            'patch_size': self.patch_size,
            'n_patches': self.n_patches,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'dim_feedforward': self.dim_feedforward
        }
    
    def print_architecture(self):
        """Print model architecture summary (matching MNISTModelNN format)."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("\n" + "="*70)
        print("MNIST TRANSFORMER ARCHITECTURE")
        print("="*70)
        print(f"Input:          784 (28x28 → {self.n_patches} patches of {self.patch_size}x{self.patch_size})")
        print(f"Patch Embed:    {self.patch_dim} → {self.d_model}")
        print(f"Pos Encoding:   {self.d_model} dimensions")
        print(f"Transformer:    {self.n_layers} layers × {self.n_heads} heads")
        print(f"  - d_model:    {self.d_model}")
        print(f"  - FFN dim:    {self.dim_feedforward}")
        print(f"Global Pool:    {self.n_patches} patches → 1")
        print(f"Classifier:     {self.d_model} → 10")
        print("-"*70)
        
        # Create compact architecture string
        arch_str = f"{self.patch_size}x{self.patch_size}patch-{self.d_model}d-{self.n_layers}L{self.n_heads}H-{self.dim_feedforward}ffn-10out"
        print(f"Architecture: {arch_str}")
        print(f"Total Layers: {self.n_layers + 3} (embed + {self.n_layers} transformer + classifier)")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: ~{total_params * 4 / (1024**2):.2f} MB")
        print(f"Dropout rate: {self.dropout_rate}")
        print(f"Patches: {self.n_patches} ({self.n_patches_per_side}×{self.n_patches_per_side})")
        print("="*70)


if __name__ == "__main__":
    print("Testing MNISTTransformer")
    print("="*50)
    
    # Test with different configurations
    print("\n1. Default Configuration:")
    model1 = MNISTTransformer()
    
    print("\n2. Smaller Configuration:")
    model2 = MNISTTransformer(
        patch_size=7,          # 7x7 patches (16 patches)
        d_model=128,
        n_heads=4,
        n_layers=3,
        dim_feedforward=256,
        dropout_rate=0.2
    )
    
    print("\n3. Testing Forward Pass:")
    # Test both input formats
    batch_size = 8
    
    # Format 1: (batch, 1, 28, 28)
    input1 = torch.randn(batch_size, 1, 28, 28)
    output1 = model1(input1)
    print(f"Input shape (1,28,28): {input1.shape} → Output: {output1.shape}")
    
    # Format 2: (batch, 784)
    input2 = torch.randn(batch_size, 784)
    output2 = model1(input2)
    print(f"Input shape (784,): {input2.shape} → Output: {output2.shape}")
    
    print("\n4. Model Summary:")
    summary = model1.get_model_summary()
    print(f"Architecture: {summary['architecture']}")
    print(f"Parameters: {summary['total_parameters']:,}")
    print(f"Model Size: {summary['model_size_mb']:.2f} MB")
    
    print("\n5. Layer Outputs Test:")
    layer_outputs = model1.get_layer_outputs(input1[:2])  # Test with 2 samples
    for layer_name, output in layer_outputs.items():
        print(f"{layer_name}: {output.shape}")
    
    print("\nMNISTTransformer is ready for use with FlexibleTrainer!")