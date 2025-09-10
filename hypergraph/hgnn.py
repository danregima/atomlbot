"""
HyperGraph Neural Network (HGNN) components for AtomBot.

This module implements neural network layers that operate on hypergraphs,
enabling learning and message passing over higher-order relationships.
"""

import numpy as np
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Callable
from abc import ABC, abstractmethod

from .hypergraph import HyperGraph, HyperNode, HyperEdge


class HyperGraphLayer(ABC):
    """Abstract base class for hypergraph neural network layers."""
    
    def __init__(self, input_dim: int, output_dim: int, layer_name: str = "HyperGraphLayer"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_name = layer_name
        
        # Learnable parameters (initialized randomly)
        self._initialize_parameters()
        
        # Training state
        self.training = True
        self.dropout_rate = 0.1
    
    @abstractmethod
    def _initialize_parameters(self) -> None:
        """Initialize learnable parameters."""
        pass
    
    @abstractmethod
    def forward(self, hypergraph: HyperGraph) -> np.ndarray:
        """Forward pass through the layer."""
        pass
    
    def set_training(self, training: bool) -> None:
        """Set training mode."""
        self.training = training
    
    def set_dropout_rate(self, rate: float) -> None:
        """Set dropout rate."""
        self.dropout_rate = max(0.0, min(1.0, rate))


class HyperGraphConvolution(HyperGraphLayer):
    """
    HyperGraph Convolution layer that performs message passing over hyperedges.
    
    This layer aggregates information from hyperedges to update node representations.
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 aggregation: str = "mean", use_edge_features: bool = True):
        self.aggregation = aggregation  # "mean", "sum", "max"
        self.use_edge_features = use_edge_features
        
        super().__init__(input_dim, output_dim, "HyperGraphConvolution")
    
    def _initialize_parameters(self) -> None:
        """Initialize learnable parameters."""
        # Node transformation matrix
        self.W_node = np.random.randn(self.input_dim, self.output_dim) * 0.1
        
        # Edge transformation matrix (if using edge features)
        if self.use_edge_features:
            # Edge features should match input_dim for consistency
            self.W_edge = np.random.randn(self.input_dim, self.output_dim) * 0.1
        
        # Bias terms
        self.bias = np.zeros(self.output_dim)
        
        # Attention weights for hyperedge aggregation
        self.attention_weights = np.random.randn(self.output_dim, 1) * 0.1
    
    def forward(self, hypergraph: HyperGraph) -> np.ndarray:
        """
        Forward pass: aggregate information from hyperedges to nodes.
        
        Args:
            hypergraph: Input hypergraph
            
        Returns:
            Updated node features matrix
        """
        nodes = list(hypergraph._nodes.values())
        edges = list(hypergraph._edges.values())
        
        if not nodes:
            return np.array([])
        
        n_nodes = len(nodes)
        node_to_idx = {node.id: i for i, node in enumerate(nodes)}
        
        # Get current node features
        node_features = hypergraph.get_node_features()
        
        # Initialize output features
        output_features = np.zeros((n_nodes, self.output_dim))
        
        # Transform node features
        transformed_nodes = node_features @ self.W_node
        
        # Process each hyperedge
        for edge in edges:
            edge_nodes = list(edge.get_nodes())
            if len(edge_nodes) < 2:
                continue
            
            # Get node indices for this edge
            node_indices = [node_to_idx[node.id] for node in edge_nodes if node.id in node_to_idx]
            
            if not node_indices:
                continue
            
            # Get features for nodes in this hyperedge
            edge_node_features = transformed_nodes[node_indices]
            
            # Add edge features if available
            if self.use_edge_features and edge.features is not None:
                edge_feature = edge.features @ self.W_edge
                # Broadcast edge feature to all nodes in the hyperedge
                edge_node_features = edge_node_features + edge_feature.reshape(1, -1)
            
            # Apply edge weight
            edge_node_features = edge_node_features * edge.weight
            
            # Aggregate features within the hyperedge
            if self.aggregation == "mean":
                aggregated = np.mean(edge_node_features, axis=0)
            elif self.aggregation == "sum":
                aggregated = np.sum(edge_node_features, axis=0)
            elif self.aggregation == "max":
                aggregated = np.max(edge_node_features, axis=0)
            else:
                aggregated = np.mean(edge_node_features, axis=0)
            
            # Compute attention weights
            attention_score = np.tanh(aggregated @ self.attention_weights).flatten()[0]
            attention_weight = np.exp(attention_score)
            
            # Update features for all nodes in the hyperedge
            weighted_aggregated = aggregated * attention_weight
            for idx in node_indices:
                output_features[idx] += weighted_aggregated
        
        # Add bias and apply activation
        output_features += self.bias
        output_features = np.tanh(output_features)  # Tanh activation
        
        # Apply dropout during training
        if self.training and self.dropout_rate > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, output_features.shape)
            output_features = output_features * dropout_mask / (1 - self.dropout_rate)
        
        # Update node features in the hypergraph
        for i, node in enumerate(nodes):
            node.update_features(output_features[i])
        
        return output_features


class HyperAttention(HyperGraphLayer):
    """
    Hypergraph Attention layer that learns attention weights over hyperedges.
    
    This layer computes attention scores for each hyperedge based on node features.
    """
    
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 4):
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        super().__init__(input_dim, output_dim, "HyperAttention")
    
    def _initialize_parameters(self) -> None:
        """Initialize learnable parameters."""
        # Multi-head attention parameters
        self.W_query = np.random.randn(self.input_dim, self.output_dim) * 0.1
        self.W_key = np.random.randn(self.input_dim, self.output_dim) * 0.1
        self.W_value = np.random.randn(self.input_dim, self.output_dim) * 0.1
        
        # Output projection
        self.W_output = np.random.randn(self.output_dim, self.output_dim) * 0.1
        self.bias = np.zeros(self.output_dim)
    
    def forward(self, hypergraph: HyperGraph) -> np.ndarray:
        """
        Forward pass: apply attention mechanism over hyperedges.
        
        Args:
            hypergraph: Input hypergraph
            
        Returns:
            Attention-weighted node features
        """
        nodes = list(hypergraph._nodes.values())
        edges = list(hypergraph._edges.values())
        
        if not nodes:
            return np.array([])
        
        n_nodes = len(nodes)
        node_to_idx = {node.id: i for i, node in enumerate(nodes)}
        
        # Get current node features
        node_features = hypergraph.get_node_features()
        
        # Compute queries, keys, and values
        queries = node_features @ self.W_query
        keys = node_features @ self.W_key
        values = node_features @ self.W_value
        
        # Reshape for multi-head attention
        queries = queries.reshape(n_nodes, self.num_heads, self.head_dim)
        keys = keys.reshape(n_nodes, self.num_heads, self.head_dim)
        values = values.reshape(n_nodes, self.num_heads, self.head_dim)
        
        # Initialize output
        output_features = np.zeros((n_nodes, self.output_dim))
        
        # Apply attention for each head
        for head in range(self.num_heads):
            head_queries = queries[:, head, :]
            head_keys = keys[:, head, :]
            head_values = values[:, head, :]
            
            head_output = np.zeros((n_nodes, self.head_dim))
            
            # Process each hyperedge
            for edge in edges:
                edge_nodes = list(edge.get_nodes())
                if len(edge_nodes) < 2:
                    continue
                
                node_indices = [node_to_idx[node.id] for node in edge_nodes if node.id in node_to_idx]
                if not node_indices:
                    continue
                
                # Compute attention scores within the hyperedge
                edge_queries = head_queries[node_indices]
                edge_keys = head_keys[node_indices]
                edge_values = head_values[node_indices]
                
                # Scaled dot-product attention
                scores = edge_queries @ edge_keys.T / np.sqrt(self.head_dim)
                attention_weights = self._softmax(scores)
                
                # Apply attention weights
                attended_values = attention_weights @ edge_values
                
                # Update node features
                for i, idx in enumerate(node_indices):
                    head_output[idx] += attended_values[i] * edge.weight
            
            # Store head output
            start_idx = head * self.head_dim
            end_idx = start_idx + self.head_dim
            output_features[:, start_idx:end_idx] = head_output
        
        # Output projection
        output_features = output_features @ self.W_output + self.bias
        
        # Apply activation
        output_features = np.tanh(output_features)
        
        # Apply dropout during training
        if self.training and self.dropout_rate > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, output_features.shape)
            output_features = output_features * dropout_mask / (1 - self.dropout_rate)
        
        # Update node features in the hypergraph
        for i, node in enumerate(nodes):
            node.update_features(output_features[i])
        
        return output_features
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax activation."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class HGNNLayer(HyperGraphLayer):
    """
    Complete HGNN layer combining convolution and attention mechanisms.
    
    This layer performs both hypergraph convolution and attention, 
    providing a comprehensive update rule for node features.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_attention_heads: int = 4, aggregation: str = "mean"):
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        
        super().__init__(input_dim, output_dim, "HGNNLayer")
        
        # Sub-layers (disable edge features to avoid dimension mismatch)
        self.conv_layer = HyperGraphConvolution(input_dim, hidden_dim, aggregation, use_edge_features=False)
        self.attention_layer = HyperAttention(hidden_dim, hidden_dim, num_attention_heads)
        
        # Final projection layer
        self.output_projection = np.random.randn(hidden_dim, output_dim) * 0.1
        self.output_bias = np.zeros(output_dim)
    
    def _initialize_parameters(self) -> None:
        """Parameters are initialized in sub-layers."""
        pass
    
    def forward(self, hypergraph: HyperGraph) -> np.ndarray:
        """
        Forward pass: apply convolution followed by attention.
        
        Args:
            hypergraph: Input hypergraph
            
        Returns:
            Updated node features
        """
        # Apply hypergraph convolution
        conv_output = self.conv_layer.forward(hypergraph)
        
        if conv_output.size == 0:
            return conv_output
        
        # Apply attention mechanism
        attention_output = self.attention_layer.forward(hypergraph)
        
        # Final projection
        output_features = attention_output @ self.output_projection + self.output_bias
        output_features = np.maximum(0, output_features)  # ReLU activation
        
        # Update node features in the hypergraph
        nodes = list(hypergraph._nodes.values())
        for i, node in enumerate(nodes):
            node.update_features(output_features[i])
        
        return output_features
    
    def set_training(self, training: bool) -> None:
        """Set training mode for all sub-layers."""
        super().set_training(training)
        self.conv_layer.set_training(training)
        self.attention_layer.set_training(training)
    
    def set_dropout_rate(self, rate: float) -> None:
        """Set dropout rate for all sub-layers."""
        super().set_dropout_rate(rate)
        self.conv_layer.set_dropout_rate(rate)
        self.attention_layer.set_dropout_rate(rate)


class HyperGraphNeuralNetwork:
    """
    Complete Hypergraph Neural Network with multiple HGNN layers.
    
    This network can be used for node classification, link prediction,
    and other graph-level tasks on hypergraphs.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 num_attention_heads: int = 4, aggregation: str = "mean"):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_attention_heads = num_attention_heads
        self.aggregation = aggregation
        
        # Build layers
        self.layers = []
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(layer_dims) - 1):
            layer = HGNNLayer(
                input_dim=layer_dims[i],
                hidden_dim=layer_dims[i+1],
                output_dim=layer_dims[i+1],
                num_attention_heads=num_attention_heads,
                aggregation=aggregation
            )
            self.layers.append(layer)
        
        # Training state
        self.training = True
    
    def forward(self, hypergraph: HyperGraph) -> np.ndarray:
        """
        Forward pass through the entire network.
        
        Args:
            hypergraph: Input hypergraph
            
        Returns:
            Final node embeddings
        """
        output = None
        
        for layer in self.layers:
            output = layer.forward(hypergraph)
            if output.size == 0:
                break
        
        return output if output is not None else np.array([])
    
    def set_training(self, training: bool) -> None:
        """Set training mode for all layers."""
        self.training = training
        for layer in self.layers:
            layer.set_training(training)
    
    def set_dropout_rate(self, rate: float) -> None:
        """Set dropout rate for all layers."""
        for layer in self.layers:
            layer.set_dropout_rate(rate)
    
    async def async_forward(self, hypergraph: HyperGraph) -> np.ndarray:
        """
        Asynchronous forward pass for large hypergraphs.
        
        Args:
            hypergraph: Input hypergraph
            
        Returns:
            Final node embeddings
        """
        # For now, just call the synchronous version
        # Could be extended for true async processing
        return self.forward(hypergraph)
    
    def get_node_embeddings(self, hypergraph: HyperGraph) -> Dict[str, np.ndarray]:
        """
        Get embeddings for all nodes in the hypergraph.
        
        Args:
            hypergraph: Input hypergraph
            
        Returns:
            Dictionary mapping node IDs to their embeddings
        """
        embeddings = self.forward(hypergraph)
        nodes = list(hypergraph._nodes.values())
        
        if embeddings.size == 0:
            return {}
        
        return {
            node.id: embeddings[i] 
            for i, node in enumerate(nodes)
        }
    
    def predict_node_labels(self, hypergraph: HyperGraph) -> Dict[str, np.ndarray]:
        """
        Predict labels for nodes using the learned embeddings.
        
        Args:
            hypergraph: Input hypergraph
            
        Returns:
            Dictionary mapping node IDs to predicted label probabilities
        """
        embeddings = self.forward(hypergraph)
        nodes = list(hypergraph._nodes.values())
        
        if embeddings.size == 0:
            return {}
        
        # Apply softmax to get probabilities
        exp_embeddings = np.exp(embeddings - np.max(embeddings, axis=1, keepdims=True))
        probabilities = exp_embeddings / np.sum(exp_embeddings, axis=1, keepdims=True)
        
        return {
            node.id: probabilities[i] 
            for i, node in enumerate(nodes)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the network architecture."""
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "output_dim": self.output_dim,
            "num_layers": len(self.layers),
            "num_attention_heads": self.num_attention_heads,
            "aggregation": self.aggregation,
            "training": self.training
        }