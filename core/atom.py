"""
Core Atom class for AtomBot.

Atoms are the fundamental units of knowledge representation, inspired by OpenCog AtomSpace.
In AtomBot, each atom also has chat agent capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set, TYPE_CHECKING
import uuid
from datetime import datetime

if TYPE_CHECKING:
    from .atomspace import AtomSpace
    from .value import Value


class AtomType:
    """Atom type hierarchy similar to OpenCog."""
    
    # Base types
    ATOM = "Atom"
    NODE = "Node" 
    LINK = "Link"
    
    # Node types
    CONCEPT_NODE = "ConceptNode"
    PREDICATE_NODE = "PredicateNode"
    SCHEMA_NODE = "SchemaNode"
    
    # Link types  
    INHERITANCE_LINK = "InheritanceLink"
    EVALUATION_LINK = "EvaluationLink"
    SIMILARITY_LINK = "SimilarityLink"
    LIST_LINK = "ListLink"
    
    # Chat Agent types (new for AtomBot)
    CHAT_AGENT_NODE = "ChatAgentNode"
    CONVERSATION_LINK = "ConversationLink"
    
    @classmethod
    def is_node(cls, atom_type: str) -> bool:
        """Check if an atom type is a node type."""
        return atom_type in [
            cls.NODE, cls.CONCEPT_NODE, cls.PREDICATE_NODE, 
            cls.SCHEMA_NODE, cls.CHAT_AGENT_NODE
        ]
    
    @classmethod
    def is_link(cls, atom_type: str) -> bool:
        """Check if an atom type is a link type."""
        return atom_type in [
            cls.LINK, cls.INHERITANCE_LINK, cls.EVALUATION_LINK,
            cls.SIMILARITY_LINK, cls.LIST_LINK, cls.CONVERSATION_LINK
        ]


class Atom(ABC):
    """Abstract base class for all atoms in AtomBot."""
    
    def __init__(self, atom_type: str, name: str = "", atomspace: Optional['AtomSpace'] = None):
        self.id = str(uuid.uuid4())
        self.atom_type = atom_type
        self.name = name
        self.atomspace = atomspace
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        
        # Values attached to this atom (key-value store)
        self._values: Dict[str, 'Value'] = {}
        
        # Incoming and outgoing links
        self._incoming: Set['Atom'] = set()
        self._outgoing: List['Atom'] = []
        
        # Register with atomspace if provided
        if atomspace:
            atomspace.add_atom(self)
    
    @property
    def incoming(self) -> Set['Atom']:
        """Get atoms that link to this atom."""
        return self._incoming.copy()
    
    @property  
    def outgoing(self) -> List['Atom']:
        """Get atoms this atom links to."""
        return self._outgoing.copy()
    
    def add_incoming(self, atom: 'Atom') -> None:
        """Add an incoming link."""
        self._incoming.add(atom)
        self.updated_at = datetime.now()
    
    def remove_incoming(self, atom: 'Atom') -> None:
        """Remove an incoming link."""
        self._incoming.discard(atom)
        self.updated_at = datetime.now()
    
    def add_outgoing(self, atom: 'Atom') -> None:
        """Add an outgoing link."""
        if atom not in self._outgoing:
            self._outgoing.append(atom)
            atom.add_incoming(self)
            self.updated_at = datetime.now()
    
    def remove_outgoing(self, atom: 'Atom') -> None:
        """Remove an outgoing link."""
        if atom in self._outgoing:
            self._outgoing.remove(atom)
            atom.remove_incoming(self)
            self.updated_at = datetime.now()
    
    def set_value(self, key: str, value: 'Value') -> None:
        """Set a value for this atom."""
        self._values[key] = value
        self.updated_at = datetime.now()
    
    def get_value(self, key: str) -> Optional['Value']:
        """Get a value from this atom."""
        return self._values.get(key)
    
    def remove_value(self, key: str) -> None:
        """Remove a value from this atom."""
        if key in self._values:
            del self._values[key]
            self.updated_at = datetime.now()
    
    def get_all_values(self) -> Dict[str, 'Value']:
        """Get all values attached to this atom."""
        return self._values.copy()
    
    def get_arity(self) -> int:
        """Get the arity (number of outgoing atoms) of this atom."""
        return len(self._outgoing)
    
    def is_node(self) -> bool:
        """Check if this atom is a node."""
        return AtomType.is_node(self.atom_type)
    
    def is_link(self) -> bool:
        """Check if this atom is a link."""
        return AtomType.is_link(self.atom_type)
    
    def get_neighbors(self) -> Set['Atom']:
        """Get all neighboring atoms (incoming + outgoing)."""
        neighbors = self._incoming.copy()
        neighbors.update(self._outgoing)
        return neighbors
    
    def __eq__(self, other: object) -> bool:
        """Check equality based on atom type, name, and outgoing atoms."""
        if not isinstance(other, Atom):
            return False
        return (
            self.atom_type == other.atom_type and
            self.name == other.name and
            self._outgoing == other._outgoing
        )
    
    def __hash__(self) -> int:
        """Hash based on atom type and name."""
        return hash((self.atom_type, self.name))
    
    def __str__(self) -> str:
        """String representation of the atom."""
        if self.name:
            return f"{self.atom_type}:{self.name}"
        else:
            return f"{self.atom_type}({len(self._outgoing)})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.atom_type}, name='{self.name}', id={self.id[:8]})"


class Node(Atom):
    """Base class for all node atoms."""
    
    def __init__(self, atom_type: str, name: str, atomspace: Optional['AtomSpace'] = None):
        super().__init__(atom_type, name, atomspace)
    
    def add_outgoing(self, atom: 'Atom') -> None:
        """Nodes cannot have outgoing links directly."""
        raise TypeError("Nodes cannot have outgoing links. Use Links to connect atoms.")


class Link(Atom):
    """Base class for all link atoms."""
    
    def __init__(self, atom_type: str, outgoing: List[Atom], atomspace: Optional['AtomSpace'] = None):
        super().__init__(atom_type, "", atomspace)
        
        # Set up outgoing connections
        for atom in outgoing:
            self.add_outgoing(atom)
    
    def get_target(self, index: int) -> Optional[Atom]:
        """Get the target atom at a specific index."""
        if 0 <= index < len(self._outgoing):
            return self._outgoing[index]
        return None
    
    def get_targets(self) -> List[Atom]:
        """Get all target atoms."""
        return self._outgoing.copy()
    
    def __str__(self) -> str:
        """String representation showing the link structure."""
        targets = ", ".join(str(atom) for atom in self._outgoing)
        return f"{self.atom_type}({targets})"