"""
Core HyperGraph implementation for AtomBot.

HyperGraphs extend traditional graphs by allowing hyperedges that connect 
multiple nodes simultaneously, enabling more complex relationship modeling.
"""

import uuid
import asyncio
from typing import Dict, List, Set, Optional, Any, Union, Tuple
from datetime import datetime
from collections import defaultdict

import numpy as np


class HyperNode:
    """
    A HyperNode represents a node in a hypergraph that can participate 
    in multiple hyperedges simultaneously.
    """
    
    def __init__(self, node_id: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        self.id = node_id or str(uuid.uuid4())
        self.data = data or {}
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        
        # Track hyperedges this node participates in
        self._hyperedges: Set['HyperEdge'] = set()
        
        # Node features for neural network processing
        self.features: Optional[np.ndarray] = None
        self.embedding_dim = 64  # Default embedding dimension
        
        # Initialize random features if not provided
        if self.features is None:
            self.features = np.random.randn(self.embedding_dim)
    
    def add_hyperedge(self, hyperedge: 'HyperEdge') -> None:
        """Add this node to a hyperedge."""
        self._hyperedges.add(hyperedge)
        self.updated_at = datetime.now()
    
    def remove_hyperedge(self, hyperedge: 'HyperEdge') -> None:
        """Remove this node from a hyperedge."""
        self._hyperedges.discard(hyperedge)
        self.updated_at = datetime.now()
    
    def get_hyperedges(self) -> Set['HyperEdge']:
        """Get all hyperedges this node participates in."""
        return self._hyperedges.copy()
    
    def get_neighbors(self) -> Set['HyperNode']:
        """Get all nodes connected to this node through hyperedges."""
        neighbors = set()
        for hyperedge in self._hyperedges:
            neighbors.update(hyperedge.get_nodes())
        neighbors.discard(self)  # Remove self
        return neighbors
    
    def get_degree(self) -> int:
        """Get the degree (number of hyperedges) of this node."""
        return len(self._hyperedges)
    
    def update_features(self, features: np.ndarray) -> None:
        """Update the node features."""
        self.features = features
        self.updated_at = datetime.now()
    
    def set_data(self, key: str, value: Any) -> None:
        """Set data associated with this node."""
        self.data[key] = value
        self.updated_at = datetime.now()
    
    def get_data(self, key: str, default: Any = None) -> Any:
        """Get data associated with this node."""
        return self.data.get(key, default)
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HyperNode):
            return False
        return self.id == other.id
    
    def __str__(self) -> str:
        return f"HyperNode({self.id[:8]})"
    
    def __repr__(self) -> str:
        return f"HyperNode(id='{self.id}', degree={self.get_degree()})"


class HyperEdge:
    """
    A HyperEdge connects multiple nodes in a hypergraph, representing 
    a higher-order relationship.
    """
    
    def __init__(self, edge_id: Optional[str] = None, nodes: Optional[List[HyperNode]] = None,
                 edge_type: str = "generic", data: Optional[Dict[str, Any]] = None):
        self.id = edge_id or str(uuid.uuid4())
        self.edge_type = edge_type
        self.data = data or {}
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        
        # Nodes in this hyperedge
        self._nodes: Set[HyperNode] = set()
        
        # Edge features for neural network processing (match node embedding dimension)
        self.features: Optional[np.ndarray] = None
        self.embedding_dim = 64  # Should match node embedding dimension
        
        # Weight of this hyperedge (used in neural computations)
        self.weight = 1.0
        
        # Initialize with provided nodes
        if nodes:
            for node in nodes:
                self.add_node(node)
        
        # Initialize random features if not provided (match node dimension)
        if self.features is None:
            self.features = np.random.randn(self.embedding_dim)
    
    def add_node(self, node: HyperNode) -> None:
        """Add a node to this hyperedge."""
        self._nodes.add(node)
        node.add_hyperedge(self)
        self.updated_at = datetime.now()
    
    def remove_node(self, node: HyperNode) -> None:
        """Remove a node from this hyperedge."""
        self._nodes.discard(node)
        node.remove_hyperedge(self)
        self.updated_at = datetime.now()
    
    def get_nodes(self) -> Set[HyperNode]:
        """Get all nodes in this hyperedge."""
        return self._nodes.copy()
    
    def get_cardinality(self) -> int:
        """Get the cardinality (number of nodes) of this hyperedge."""
        return len(self._nodes)
    
    def contains_node(self, node: HyperNode) -> bool:
        """Check if this hyperedge contains a specific node."""
        return node in self._nodes
    
    def update_features(self, features: np.ndarray) -> None:
        """Update the hyperedge features."""
        self.features = features
        self.updated_at = datetime.now()
    
    def set_weight(self, weight: float) -> None:
        """Set the weight of this hyperedge."""
        self.weight = weight
        self.updated_at = datetime.now()
    
    def set_data(self, key: str, value: Any) -> None:
        """Set data associated with this hyperedge."""
        self.data[key] = value
        self.updated_at = datetime.now()
    
    def get_data(self, key: str, default: Any = None) -> Any:
        """Get data associated with this hyperedge."""
        return self.data.get(key, default)
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HyperEdge):
            return False
        return self.id == other.id
    
    def __str__(self) -> str:
        return f"HyperEdge({self.edge_type}:{self.id[:8]})"
    
    def __repr__(self) -> str:
        return f"HyperEdge(id='{self.id}', type='{self.edge_type}', cardinality={self.get_cardinality()})"


class HyperGraph:
    """
    A HyperGraph manages collections of HyperNodes and HyperEdges,
    providing the foundation for neural network computations.
    """
    
    def __init__(self, name: str = "HyperGraph"):
        self.name = name
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        
        # Storage
        self._nodes: Dict[str, HyperNode] = {}
        self._edges: Dict[str, HyperEdge] = {}
        
        # Indices for efficient queries
        self._node_type_index: Dict[str, Set[str]] = defaultdict(set)
        self._edge_type_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Neural network matrices (computed on demand)
        self._incidence_matrix: Optional[np.ndarray] = None
        self._adjacency_matrix: Optional[np.ndarray] = None
        self._matrices_dirty = True
    
    def add_node(self, node: HyperNode) -> HyperNode:
        """Add a node to the hypergraph."""
        if node.id in self._nodes:
            return self._nodes[node.id]
        
        self._nodes[node.id] = node
        self._matrices_dirty = True
        self.updated_at = datetime.now()
        
        # Update type index
        node_type = node.get_data("type", "generic")
        self._node_type_index[node_type].add(node.id)
        
        return node
    
    def add_edge(self, edge: HyperEdge) -> HyperEdge:
        """Add a hyperedge to the hypergraph."""
        if edge.id in self._edges:
            return self._edges[edge.id]
        
        self._edges[edge.id] = edge
        self._matrices_dirty = True
        self.updated_at = datetime.now()
        
        # Ensure all nodes in the edge are in the graph
        for node in edge.get_nodes():
            self.add_node(node)
        
        # Update type index
        self._edge_type_index[edge.edge_type].add(edge.id)
        
        return edge
    
    def remove_node(self, node: Union[HyperNode, str]) -> bool:
        """Remove a node from the hypergraph."""
        node_id = node.id if isinstance(node, HyperNode) else node
        
        if node_id not in self._nodes:
            return False
        
        node_obj = self._nodes[node_id]
        
        # Remove from all hyperedges
        for edge in list(node_obj.get_hyperedges()):
            edge.remove_node(node_obj)
            # Remove empty hyperedges
            if edge.get_cardinality() == 0:
                self.remove_edge(edge)
        
        # Remove from indices
        node_type = node_obj.get_data("type", "generic")
        self._node_type_index[node_type].discard(node_id)
        
        del self._nodes[node_id]
        self._matrices_dirty = True
        self.updated_at = datetime.now()
        
        return True
    
    def remove_edge(self, edge: Union[HyperEdge, str]) -> bool:
        """Remove a hyperedge from the hypergraph."""
        edge_id = edge.id if isinstance(edge, HyperEdge) else edge
        
        if edge_id not in self._edges:
            return False
        
        edge_obj = self._edges[edge_id]
        
        # Remove edge from all nodes
        for node in list(edge_obj.get_nodes()):
            edge_obj.remove_node(node)
        
        # Remove from indices
        self._edge_type_index[edge_obj.edge_type].discard(edge_id)
        
        del self._edges[edge_id]
        self._matrices_dirty = True
        self.updated_at = datetime.now()
        
        return True
    
    def get_node(self, node_id: str) -> Optional[HyperNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[HyperEdge]:
        """Get a hyperedge by ID."""
        return self._edges.get(edge_id)
    
    def get_nodes(self, node_type: Optional[str] = None) -> List[HyperNode]:
        """Get all nodes, optionally filtered by type."""
        if node_type is None:
            return list(self._nodes.values())
        
        node_ids = self._node_type_index.get(node_type, set())
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]
    
    def get_edges(self, edge_type: Optional[str] = None) -> List[HyperEdge]:
        """Get all hyperedges, optionally filtered by type."""
        if edge_type is None:
            return list(self._edges.values())
        
        edge_ids = self._edge_type_index.get(edge_type, set())
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]
    
    def create_hyperedge(self, nodes: List[HyperNode], edge_type: str = "generic",
                        edge_id: Optional[str] = None, data: Optional[Dict[str, Any]] = None) -> HyperEdge:
        """Create and add a new hyperedge connecting the given nodes."""
        edge = HyperEdge(edge_id, nodes, edge_type, data)
        return self.add_edge(edge)
    
    def get_incidence_matrix(self) -> np.ndarray:
        """
        Get the incidence matrix of the hypergraph.
        
        Returns:
            Binary matrix where M[i,j] = 1 if node i is in hyperedge j
        """
        if self._incidence_matrix is None or self._matrices_dirty:
            self._compute_matrices()
        return self._incidence_matrix
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get the adjacency matrix of the hypergraph.
        
        Returns:
            Matrix where A[i,j] is the number of hyperedges connecting nodes i and j
        """
        if self._adjacency_matrix is None or self._matrices_dirty:
            self._compute_matrices()
        return self._adjacency_matrix
    
    def _compute_matrices(self) -> None:
        """Compute incidence and adjacency matrices."""
        nodes = list(self._nodes.values())
        edges = list(self._edges.values())
        
        n_nodes = len(nodes)
        n_edges = len(edges)
        
        if n_nodes == 0 or n_edges == 0:
            self._incidence_matrix = np.zeros((n_nodes, n_edges))
            self._adjacency_matrix = np.zeros((n_nodes, n_nodes))
            self._matrices_dirty = False
            return
        
        # Create node and edge indices
        node_to_idx = {node.id: i for i, node in enumerate(nodes)}
        edge_to_idx = {edge.id: i for i, edge in enumerate(edges)}
        
        # Compute incidence matrix
        incidence = np.zeros((n_nodes, n_edges))
        for j, edge in enumerate(edges):
            for node in edge.get_nodes():
                i = node_to_idx[node.id]
                incidence[i, j] = edge.weight
        
        # Compute adjacency matrix (H @ H.T)
        adjacency = incidence @ incidence.T
        np.fill_diagonal(adjacency, 0)  # Remove self-loops
        
        self._incidence_matrix = incidence
        self._adjacency_matrix = adjacency
        self._matrices_dirty = False
    
    def get_node_features(self) -> np.ndarray:
        """Get node features as a matrix."""
        nodes = list(self._nodes.values())
        if not nodes:
            return np.array([])
        
        return np.stack([node.features for node in nodes])
    
    def get_edge_features(self) -> np.ndarray:
        """Get hyperedge features as a matrix."""
        edges = list(self._edges.values())
        if not edges:
            return np.array([])
        
        return np.stack([edge.features for edge in edges])
    
    def find_connected_components(self) -> List[Set[HyperNode]]:
        """Find connected components in the hypergraph."""
        visited = set()
        components = []
        
        for node in self._nodes.values():
            if node not in visited:
                component = set()
                stack = [node]
                
                while stack:
                    current = stack.pop()
                    if current in visited:
                        continue
                    
                    visited.add(current)
                    component.add(current)
                    
                    # Add neighbors
                    for neighbor in current.get_neighbors():
                        if neighbor not in visited:
                            stack.append(neighbor)
                
                components.append(component)
        
        return components
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the hypergraph."""
        nodes = list(self._nodes.values())
        edges = list(self._edges.values())
        
        if not nodes:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "avg_node_degree": 0,
                "avg_edge_cardinality": 0,
                "connected_components": 0
            }
        
        node_degrees = [node.get_degree() for node in nodes]
        edge_cardinalities = [edge.get_cardinality() for edge in edges]
        components = self.find_connected_components()
        
        return {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "avg_node_degree": np.mean(node_degrees) if node_degrees else 0,
            "max_node_degree": max(node_degrees) if node_degrees else 0,
            "avg_edge_cardinality": np.mean(edge_cardinalities) if edge_cardinalities else 0,
            "max_edge_cardinality": max(edge_cardinalities) if edge_cardinalities else 0,
            "connected_components": len(components),
            "node_types": list(self._node_type_index.keys()),
            "edge_types": list(self._edge_type_index.keys())
        }
    
    def __len__(self) -> int:
        """Get the number of nodes in the hypergraph."""
        return len(self._nodes)
    
    def __str__(self) -> str:
        return f"HyperGraph('{self.name}', nodes={len(self._nodes)}, edges={len(self._edges)})"
    
    def __repr__(self) -> str:
        return f"HyperGraph(name='{self.name}', nodes={len(self._nodes)}, edges={len(self._edges)})"