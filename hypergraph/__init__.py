# HyperGraph Neural Network components for AtomBot
"""
HyperGraph Neural Network (HGNN) implementation for AtomBot.

This module extends the AtomBot architecture to support HyperGraphs where:
- HyperNodes are Distributed HyperAgents
- HyperEdges are Discrete-Event HyperChannels  
- HyperGraph is a Dynamical HyperSystem
"""

__version__ = "0.1.0"

from .hypergraph import HyperGraph, HyperNode, HyperEdge
from .hgnn import HyperGraphConvolution, HyperAttention, HGNNLayer
from .hyperagent import HyperAgent
from .hyperchannel import HyperChannel, DiscreteEventChannel
from .dynamical_system import DynamicalHyperSystem

__all__ = [
    "HyperGraph",
    "HyperNode", 
    "HyperEdge",
    "HyperGraphConvolution",
    "HyperAttention",
    "HGNNLayer",
    "HyperAgent",
    "HyperChannel",
    "DiscreteEventChannel", 
    "DynamicalHyperSystem",
]