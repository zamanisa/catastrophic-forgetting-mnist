import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional


class GraphConvLayer(nn.Module):
    """
    Graph Convolutional Layer that performs message passing between pixel nodes.
    
    Each pixel is a node, and edges connect spatially neighboring pixels.
    """
    
    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Linear transformations for node features
        self.linear_self = nn.Linear(in_features, out_features)
        self.linear_neighbor = nn.Linear(in_features, out_features)
        
        # Attention mechanism for weighted aggregation
        self.attention = nn.Linear(in_features * 2, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(out_features)
        
    def forward(self, node_features, adjacency_matrix):
        """
        Args:
            node_features: (batch, num_nodes, in_features)
            adjacency_matrix: (num_nodes, num_nodes) - spatial connectivity
        Returns:
            updated_features: (batch, num_nodes, out_features)
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # Self transformation
        self_features = self.linear_self(node_features)  # (batch, num_nodes, out_features)
        
        # Neighbor aggregation with attention
        neighbor_features = []
        
        for i in range(num_nodes):
            # Find neighbors of node i
            neighbors = torch.nonzero(adjacency_matrix[i], as_tuple=True)[0]
            
            if len(neighbors) == 0:
                # No neighbors - just use self features
                aggregated = torch.zeros_like(self_features[:, i])
            else:
                # Get neighbor features
                neighbor_feats = node_features[:, neighbors]  # (batch, num_neighbors, in_features)
                current_feat = node_features[:, i:i+1].expand(-1, len(neighbors), -1)  # (batch, num_neighbors, in_features)
                
                # Compute attention weights
                attention_input = torch.cat([current_feat, neighbor_feats], dim=2)  # (batch, num_neighbors, 2*in_features)
                attention_weights = torch.softmax(self.attention(attention_input).squeeze(-1), dim=1)  # (batch, num_neighbors)
                
                # Weighted aggregation
                aggregated = torch.sum(attention_weights.unsqueeze(-1) * self.linear_neighbor(neighbor_feats), dim=1)  # (batch, out_features)
            
            neighbor_features.append(aggregated)
        
        neighbor_features = torch.stack(neighbor_features, dim=1)  # (batch, num_nodes, out_features)
        
        # Combine self and neighbor features
        combined = self_features + neighbor_features
        combined = self.dropout(combined)
        combined = self.layer_norm(combined)
        
        return F.relu(combined)


class MNISTGraphNN(nn.Module):
    """
    Graph Neural Network for MNIST digit classification.
    Treats each pixel as a node in a graph with spatial connectivity.
    Drop-in compatible with MNISTModelNN interface.
    """
    
    def __init__(
        self,
        hidden_dims: List[int] = [64, 128, 256, 128],  # Hidden dimensions for graph conv layers
        dropout_rate: float = 0.1,
        connectivity_radius: int = 2,  # How far to connect pixels (1=4-neighbors, 2=8-neighbors+more)
        use_position_features: bool = True,  # Include (x,y) coordinates as features
        pooling_method: str = 'attention'  # 'mean', 'max', 'attention'
    ):
        """
        Initialize MNIST Graph Neural Network.
        
        Args:
            hidden_dims: List of hidden dimensions for graph conv layers
            dropout_rate: Dropout probability
            connectivity_radius: Spatial radius for pixel connections
            use_position_features: Whether to include pixel coordinates as features
            pooling_method: Method for graph-level pooling ('mean', 'max', 'attention')
        """
        super(MNISTGraphNN, self).__init__()
        
        # Store architecture parameters
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.connectivity_radius = connectivity_radius
        self.use_position_features = use_position_features
        self.pooling_method = pooling_method
        
        # Graph structure
        self.img_size = 28
        self.num_nodes = self.img_size * self.img_size  # 784 pixels as nodes
        
        # Input features per node
        if use_position_features:
            self.input_dim = 3  # pixel_value + x_coord + y_coord
        else:
            self.input_dim = 1  # just pixel_value
        
        # Create adjacency matrix (spatial connectivity)
        self.register_buffer('adjacency_matrix', self._create_adjacency_matrix())
        
        # Graph convolution layers
        self.graph_layers = nn.ModuleList()
        layer_dims = [self.input_dim] + hidden_dims
        
        for i in range(len(layer_dims) - 1):
            self.graph_layers.append(
                GraphConvLayer(layer_dims[i], layer_dims[i+1], dropout_rate)
            )
        
        # Graph pooling
        if pooling_method == 'attention':
            self.pooling = nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1] // 2, 1)
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1] // 2, 10)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Auto-print architecture (matching MNISTModelNN behavior)
        self.print_architecture()
    
    def _create_adjacency_matrix(self):
        """Create adjacency matrix for spatial pixel connectivity."""
        adj = torch.zeros(self.num_nodes, self.num_nodes)
        
        for i in range(self.img_size):
            for j in range(self.img_size):
                center_idx = i * self.img_size + j
                
                # Connect to spatially nearby pixels
                for di in range(-self.connectivity_radius, self.connectivity_radius + 1):
                    for dj in range(-self.connectivity_radius, self.connectivity_radius + 1):
                        if di == 0 and dj == 0:
                            continue  # Don't connect to self
                        
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.img_size and 0 <= nj < self.img_size:
                            neighbor_idx = ni * self.img_size + nj
                            
                            # Distance-based weighting (closer pixels have stronger connections)
                            distance = np.sqrt(di**2 + dj**2)
                            if distance <= self.connectivity_radius:
                                adj[center_idx, neighbor_idx] = 1.0 / (1.0 + distance)
        
        return adj
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _prepare_node_features(self, images):
        """
        Convert images to node features.
        
        Args:
            images: (batch_size, 1, 28, 28) or (batch_size, 784)
            
        Returns:
            node_features: (batch_size, num_nodes, input_dim)
        """
        batch_size = images.size(0)
        
        # Handle both input formats
        if len(images.shape) == 2:  # (batch, 784)
            pixel_values = images
        elif len(images.shape) == 4:  # (batch, 1, 28, 28)
            pixel_values = images.view(batch_size, -1)  # Flatten to (batch, 784)
        else:
            raise ValueError(f"Unsupported input shape: {images.shape}")
        
        # Each pixel is a node
        node_features = pixel_values.unsqueeze(-1)  # (batch, 784, 1)
        
        # Add positional features if enabled
        if self.use_position_features:
            # Create coordinate features
            coords = []
            for i in range(self.img_size):
                for j in range(self.img_size):
                    # Normalize coordinates to [-1, 1]
                    x_coord = (j - self.img_size // 2) / (self.img_size // 2)
                    y_coord = (i - self.img_size // 2) / (self.img_size // 2)
                    coords.append([x_coord, y_coord])
            
            coords = torch.tensor(coords, dtype=torch.float32, device=images.device)
            coords = coords.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, 784, 2)
            
            # Concatenate pixel values with coordinates
            node_features = torch.cat([node_features, coords], dim=-1)  # (batch, 784, 3)
        
        return node_features
    
    def forward(self, x):
        """
        Forward pass through graph neural network.
        
        Args:
            x: Input tensor of shape (batch, 1, 28, 28) or (batch, 784)
            
        Returns:
            Output tensor of shape (batch, 10)
        """
        # Prepare node features
        node_features = self._prepare_node_features(x)  # (batch, num_nodes, input_dim)
        
        # Pass through graph convolution layers
        for graph_layer in self.graph_layers:
            node_features = graph_layer(node_features, self.adjacency_matrix)
        
        # Graph-level pooling
        if self.pooling_method == 'mean':
            graph_representation = torch.mean(node_features, dim=1)  # (batch, hidden_dim)
        elif self.pooling_method == 'max':
            graph_representation = torch.max(node_features, dim=1)[0]  # (batch, hidden_dim)
        elif self.pooling_method == 'attention':
            # Attention-based pooling
            attention_weights = torch.softmax(self.pooling(node_features).squeeze(-1), dim=1)  # (batch, num_nodes)
            graph_representation = torch.sum(attention_weights.unsqueeze(-1) * node_features, dim=1)  # (batch, hidden_dim)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")
        
        # Classification
        output = self.classifier(graph_representation)  # (batch, 10)
        
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
        
        # Node features
        node_features = self._prepare_node_features(x)
        outputs['node_features'] = node_features.clone()
        
        # Through graph layers
        for i, graph_layer in enumerate(self.graph_layers):
            node_features = graph_layer(node_features, self.adjacency_matrix)
            outputs[f'graph_layer_{i+1}'] = node_features.clone()
        
        # Pooled representation
        if self.pooling_method == 'mean':
            graph_representation = torch.mean(node_features, dim=1)
        elif self.pooling_method == 'max':
            graph_representation = torch.max(node_features, dim=1)[0]
        elif self.pooling_method == 'attention':
            attention_weights = torch.softmax(self.pooling(node_features).squeeze(-1), dim=1)
            graph_representation = torch.sum(attention_weights.unsqueeze(-1) * node_features, dim=1)
        
        outputs['graph_representation'] = graph_representation.clone()
        
        # Final output
        output = self.classifier(graph_representation)
        outputs['output'] = output
        
        return outputs
    
    def get_model_summary(self):
        """Get architecture summary as dictionary (compatible with MNISTModelNN)."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Create architecture string
        hidden_str = '-'.join(map(str, self.hidden_dims))
        arch_str = f"Graph({self.input_dim}→{hidden_str}→10)"
        
        # Layer information (matching MNISTModelNN format)
        layer_info = [
            {'name': 'Input', 'size': 784, 'activation': None},
            {'name': 'NodeFeatures', 'size': self.input_dim, 'activation': None}
        ]
        
        for i, dim in enumerate(self.hidden_dims):
            layer_info.append({
                'name': f'GraphConv{i+1}', 
                'size': dim, 
                'activation': 'ReLU+Attention'
            })
        
        layer_info.extend([
            {'name': f'Pooling({self.pooling_method})', 'size': self.hidden_dims[-1], 'activation': None},
            {'name': 'Classifier', 'size': 10, 'activation': None}
        ])
        
        return {
            'architecture': arch_str,
            'layers': layer_info,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024**2),
            'dropout_rate': self.dropout_rate,
            'num_hidden_layers': len(self.hidden_dims),
            # Graph-specific info
            'num_nodes': self.num_nodes,
            'connectivity_radius': self.connectivity_radius,
            'use_position_features': self.use_position_features,
            'pooling_method': self.pooling_method,
            'graph_edges': int(self.adjacency_matrix.sum().item())
        }
    
    def print_architecture(self):
        """Print model architecture summary (matching MNISTModelNN format)."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_edges = int(self.adjacency_matrix.sum().item())
        
        print("\n" + "="*70)
        print("MNIST GRAPH NEURAL NETWORK ARCHITECTURE")
        print("="*70)
        print(f"Input:          784 pixels → {self.num_nodes} nodes")
        if self.use_position_features:
            print(f"Node Features:  {self.input_dim} (pixel_value + x_coord + y_coord)")
        else:
            print(f"Node Features:  {self.input_dim} (pixel_value only)")
        
        print(f"Graph Edges:    {num_edges:,} (radius={self.connectivity_radius})")
        print(f"Graph Layers:   {len(self.hidden_dims)} layers")
        
        for i, dim in enumerate(self.hidden_dims):
            print(f"  Layer {i+1}:    {self.hidden_dims[i-1] if i > 0 else self.input_dim} → {dim} (GraphConv + Attention)")
        
        print(f"Pooling:        {self.pooling_method} ({self.num_nodes} nodes → 1)")
        print(f"Classifier:     {self.hidden_dims[-1]} → 10")
        print("-"*70)
        
        # Create compact architecture string  
        arch_str = f"Graph{self.connectivity_radius}r-{'-'.join(map(str, self.hidden_dims))}-{self.pooling_method}pool-10out"
        print(f"Architecture: {arch_str}")
        print(f"Total Layers: {len(self.hidden_dims) + 2} ({len(self.hidden_dims)} graph + pooling + classifier)")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: ~{total_params * 4 / (1024**2):.2f} MB")
        print(f"Dropout rate: {self.dropout_rate}")
        print(f"Nodes: {self.num_nodes} | Edges: {num_edges:,} | Avg degree: {num_edges/self.num_nodes:.1f}")
        print("="*70)


if __name__ == "__main__":
    print("Testing MNISTGraphNN")
    print("="*50)
    
    # Test with different configurations
    print("\n1. Default Configuration:")
    model1 = MNISTGraphNN()
    
    print("\n2. Smaller Configuration:")
    model2 = MNISTGraphNN(
        hidden_dims=[32, 64, 32],
        connectivity_radius=1,    # Only 4-connected
        use_position_features=False,
        pooling_method='max',
        dropout_rate=0.2
    )
    
    print("\n3. Testing Forward Pass:")
    # Test both input formats
    batch_size = 4  # Smaller batch for testing
    
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
    print(f"Graph Edges: {summary['graph_edges']:,}")
    
    print("\n5. Layer Outputs Test:")
    layer_outputs = model1.get_layer_outputs(input1[:2])  # Test with 2 samples
    for layer_name, output in layer_outputs.items():
        print(f"{layer_name}: {output.shape}")
    
    print("\nMNISTGraphNN is ready for use with FlexibleTrainer!")