"""
ListLink - A link that represents an ordered list of atoms.

ListLinks are used to group atoms together in a specific order,
commonly used as arguments to predicates or in other structures.
"""

from typing import List, Optional, TYPE_CHECKING
from atombot.core.atom import Link, AtomType, Atom

if TYPE_CHECKING:
    from atombot.core.atomspace import AtomSpace


class ListLink(Link):
    """
    A ListLink represents an ordered collection of atoms.
    
    Structure: ListLink(atom1, atom2, atom3, ...)
    Example: ListLink(ConceptNode("John"), ConceptNode("Pizza"))
    Used in: EvaluationLink arguments, function parameters, etc.
    """
    
    def __init__(self, items: List[Atom], atomspace: Optional['AtomSpace'] = None):
        super().__init__(AtomType.LIST_LINK, items, atomspace)
    
    def get_items(self) -> List[Atom]:
        """Get all items in this list."""
        return self.get_targets()
    
    def get_item(self, index: int) -> Optional[Atom]:
        """Get an item at a specific index."""
        return self.get_target(index)
    
    def get_size(self) -> int:
        """Get the number of items in this list."""
        return len(self._outgoing)
    
    def is_empty(self) -> bool:
        """Check if this list is empty."""
        return self.get_size() == 0
    
    def contains(self, atom: Atom) -> bool:
        """Check if this list contains a specific atom."""
        return atom in self._outgoing
    
    def index_of(self, atom: Atom) -> int:
        """Get the index of a specific atom in this list."""
        try:
            return self._outgoing.index(atom)
        except ValueError:
            return -1
    
    def to_list(self) -> List[str]:
        """Convert to a list of string representations."""
        return [str(item) for item in self._outgoing]
    
    def __len__(self) -> int:
        """Get the length of this list."""
        return self.get_size()
    
    def __iter__(self):
        """Make this list iterable."""
        return iter(self._outgoing)
    
    def __getitem__(self, index: int) -> Atom:
        """Support indexing."""
        return self._outgoing[index]
    
    def __str__(self) -> str:
        items_str = ", ".join(str(item) for item in self._outgoing)
        return f"ListLink([{items_str}])"