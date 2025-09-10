# AtomBot package
"""
A variant of OpenCog AtomSpace where each atom is an instance of a Shopify shop-chat-agent.

This package combines knowledge representation with conversational AI, creating a system
where each knowledge node is also an autonomous chat agent.
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

__all__ = [
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
]