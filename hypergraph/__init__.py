# HyperGraph Neural Network components for AtomBot
"""
HyperGraph Neural Network (HGNN) implementation for AtomBot.

This module extends the AtomBot architecture to support HyperGraphs where:
- HyperNodes are Distributed HyperAgents
- HyperEdges are Discrete-Event HyperChannels  
- HyperGraph is a Dynamical HyperSystem
"""

__version__ = "0.1.0"

# Core components that work standalone
from .hypergraph import HyperGraph, HyperNode, HyperEdge
from .hgnn import HyperGraphConvolution, HyperAttention, HGNNLayer
from .hyperchannel import HyperChannel, DiscreteEventChannel, EventType
from .hypergraphql import (
    HyperGraphQLSchema, HyperGraphQLServer, GraphQLType, GraphQLField,
    create_hypergraphql_schema, create_hypergraphql_server,
    introspect_hypergraph_schema, introspect_hypergraph_type
)

# Components requiring AtomBot integration (commented out for now)
# from .hyperagent import HyperAgent
# from .dynamical_system import DynamicalHyperSystem

__all__ = [
    "HyperGraph",
    "HyperNode", 
    "HyperEdge",
    "HyperGraphConvolution",
    "HyperAttention",
    "HGNNLayer",
    "HyperChannel",
    "DiscreteEventChannel",
    "EventType",
    "HyperGraphQLSchema",
    "HyperGraphQLServer", 
    "GraphQLType",
    "GraphQLField",
    "create_hypergraphql_schema",
    "create_hypergraphql_server",
    "introspect_hypergraph_schema",
    "introspect_hypergraph_type",
    # "HyperAgent",
    # "DynamicalHyperSystem",
]