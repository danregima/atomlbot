# AtomBot package
"""
A variant of OpenCog AtomSpace where each atom is an instance of a Shopify shop-chat-agent.

This package combines knowledge representation with conversational AI, creating a system
where each knowledge node is also an autonomous chat agent.

Now extended with HyperGraph Neural Networks (HGNN) for distributed hypergraph cognition.
"""

__version__ = "0.1.0"
__author__ = "AtomBot Contributors"

from .core.atom import Atom
from .core.value import Value, FloatValue, StringValue, LinkValue
from .core.atomspace import AtomSpace
from .agents.chat_agent import ChatAgent
from .agents.mcp_client import MCPClient
from .atombot import AtomBot
from .nodes.concept_node import ConceptNode
from .nodes.predicate_node import PredicateNode
from .links.evaluation_link import EvaluationLink
from .links.inheritance_link import InheritanceLink
from .links.list_link import ListLink

# HGNN components
from .hypergraph.hypergraph import HyperGraph, HyperNode, HyperEdge
from .hypergraph.hgnn import (
    HyperGraphConvolution, HyperAttention, HGNNLayer, HyperGraphNeuralNetwork
)
from .hypergraph.hyperagent import HyperAgent
from .hypergraph.hyperchannel import HyperChannel, DiscreteEventChannel, EventType
from .hypergraph.dynamical_system import DynamicalHyperSystem

__all__ = [
    # Core AtomBot components
    "Atom",
    "Value",
    "FloatValue", 
    "StringValue",
    "LinkValue",
    "AtomSpace",
    "ChatAgent",
    "MCPClient",
    "AtomBot",
    "ConceptNode",
    "PredicateNode", 
    "EvaluationLink",
    "InheritanceLink",
    "ListLink",
    
    # HGNN components
    "HyperGraph",
    "HyperNode",
    "HyperEdge", 
    "HyperGraphConvolution",
    "HyperAttention",
    "HGNNLayer",
    "HyperGraphNeuralNetwork",
    "HyperAgent",
    "HyperChannel",
    "DiscreteEventChannel",
    "EventType",
    "DynamicalHyperSystem",
]