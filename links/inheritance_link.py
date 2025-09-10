"""
InheritanceLink - A link representing inheritance or "is-a" relationships.

InheritanceLinks represent hierarchical relationships between concepts,
indicating that one concept is a specialization or instance of another.
"""

from typing import Optional, TYPE_CHECKING
from atombot.core.atom import Link, AtomType, Atom
from atombot.core.value import TruthValue

if TYPE_CHECKING:
    from atombot.core.atomspace import AtomSpace


class InheritanceLink(Link):
    """
    An InheritanceLink represents an "is-a" or inheritance relationship.
    
    Structure: InheritanceLink(child, parent)
    Example: InheritanceLink(ConceptNode("Dog"), ConceptNode("Animal"))
    Meaning: "Dog is a type of Animal" or "Dog inherits from Animal"
    """
    
    def __init__(self, child: Atom, parent: Atom, atomspace: Optional['AtomSpace'] = None,
                 truth_value: Optional[TruthValue] = None):
        super().__init__(AtomType.INHERITANCE_LINK, [child, parent], atomspace)
        
        # Set truth value for the inheritance relationship
        if truth_value:
            self.set_value("truth_value", truth_value)
        else:
            # Default: high confidence inheritance
            self.set_value("truth_value", TruthValue(0.9, 0.9))
    
    @property
    def child(self) -> Atom:
        """Get the child (specialized/specific) concept."""
        return self.get_target(0)
    
    @property
    def parent(self) -> Atom:
        """Get the parent (general/abstract) concept."""
        return self.get_target(1)
    
    def get_truth_value(self) -> TruthValue:
        """Get the truth value of this inheritance relationship."""
        tv = self.get_value("truth_value")
        if isinstance(tv, TruthValue):
            return tv
        return TruthValue(0.9, 0.9)  # Default high confidence
    
    def set_truth_value(self, truth_value: TruthValue) -> None:
        """Set the truth value of this inheritance relationship."""
        self.set_value("truth_value", truth_value)
    
    def is_valid_inheritance(self) -> bool:
        """Check if this inheritance relationship is considered valid."""
        return self.get_truth_value().is_true()
    
    def get_inheritance_strength(self) -> float:
        """Get the strength of the inheritance relationship."""
        return self.get_truth_value().strength
    
    def find_all_ancestors(self) -> list:
        """Find all ancestors of the child concept through inheritance chains."""
        if not self.atomspace:
            return [self.parent]
        
        ancestors = []
        visited = set()
        queue = [self.parent]
        
        while queue:
            current = queue.pop(0)
            if current.id in visited:
                continue
            
            visited.add(current.id)
            ancestors.append(current)
            
            # Find inheritance links where current is the child
            for atom in self.atomspace.get_all_atoms():
                if (isinstance(atom, InheritanceLink) and 
                    atom.child == current and 
                    atom.parent.id not in visited):
                    queue.append(atom.parent)
        
        return ancestors
    
    def find_all_descendants(self) -> list:
        """Find all descendants of the parent concept through inheritance chains."""
        if not self.atomspace:
            return [self.child]
        
        descendants = []
        visited = set()
        queue = [self.child]
        
        while queue:
            current = queue.pop(0)
            if current.id in visited:
                continue
            
            visited.add(current.id)
            descendants.append(current)
            
            # Find inheritance links where current is the parent
            for atom in self.atomspace.get_all_atoms():
                if (isinstance(atom, InheritanceLink) and 
                    atom.parent == current and 
                    atom.child.id not in visited):
                    queue.append(atom.child)
        
        return descendants
    
    def check_inheritance_consistency(self) -> dict:
        """Check for potential issues with this inheritance relationship."""
        issues = []
        
        # Check for circular inheritance
        ancestors = self.find_all_ancestors()
        if self.child in ancestors:
            issues.append("Circular inheritance detected")
        
        # Check for multiple inheritance conflicts
        # (This is a simplified check - real conflicts would be more complex)
        if not self.atomspace:
            return {"issues": issues, "consistent": len(issues) == 0}
        
        # Find other inheritance links with the same child
        other_parents = []
        for atom in self.atomspace.get_all_atoms():
            if (isinstance(atom, InheritanceLink) and 
                atom.child == self.child and 
                atom != self):
                other_parents.append(atom.parent)
        
        if len(other_parents) > 2:  # Arbitrary threshold
            issues.append(f"Multiple inheritance: {len(other_parents)} additional parents")
        
        return {
            "issues": issues,
            "consistent": len(issues) == 0,
            "other_parents": [str(p) for p in other_parents]
        }
    
    def get_inheritance_depth(self) -> int:
        """Get the depth of this concept in the inheritance hierarchy."""
        return len(self.find_all_ancestors())
    
    def __str__(self) -> str:
        child_name = str(self.child) if self.child else "Unknown"
        parent_name = str(self.parent) if self.parent else "Unknown"
        tv = self.get_truth_value()
        return f"InheritanceLink({child_name} → {parent_name}) {tv}"