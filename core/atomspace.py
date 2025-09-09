"""
AtomSpace class for AtomBot.

The AtomSpace is the central repository for atoms and provides pattern matching
and query capabilities across the agent network.
"""

from typing import Dict, List, Set, Optional, Any, Callable, Tuple, Union
import asyncio
from collections import defaultdict
import re

from .atom import Atom, AtomType, Node, Link
from .value import Value


class PatternMatcher:
    """Pattern matching engine for finding atoms that match given patterns."""
    
    def __init__(self, atomspace: 'AtomSpace'):
        self.atomspace = atomspace
    
    def match_atom_type(self, pattern_type: str, target: Atom) -> bool:
        """Check if an atom matches a type pattern."""
        if pattern_type == "*":  # Wildcard matches any type
            return True
        return target.atom_type == pattern_type
    
    def match_name_pattern(self, pattern: str, target: Atom) -> bool:
        """Check if an atom name matches a pattern (supports regex)."""
        if pattern == "*":  # Wildcard matches any name
            return True
        if pattern.startswith("regex:"):
            regex_pattern = pattern[6:]  # Remove "regex:" prefix
            return bool(re.match(regex_pattern, target.name))
        return target.name == pattern
    
    def find_atoms_by_type(self, atom_type: str) -> List[Atom]:
        """Find all atoms of a specific type."""
        return [atom for atom in self.atomspace.get_all_atoms() 
                if self.match_atom_type(atom_type, atom)]
    
    def find_atoms_by_name(self, name_pattern: str) -> List[Atom]:
        """Find all atoms matching a name pattern."""
        return [atom for atom in self.atomspace.get_all_atoms()
                if self.match_name_pattern(name_pattern, atom)]
    
    def find_atoms_by_predicate(self, predicate: Callable[[Atom], bool]) -> List[Atom]:
        """Find all atoms matching a custom predicate."""
        return [atom for atom in self.atomspace.get_all_atoms() if predicate(atom)]


class AtomSpace:
    """The AtomSpace manages a collection of atoms and their relationships."""
    
    def __init__(self, name: str = "AtomSpace"):
        self.name = name
        self._atoms: Dict[str, Atom] = {}  # id -> atom
        self._type_index: Dict[str, Set[str]] = defaultdict(set)  # type -> atom_ids
        self._name_index: Dict[str, Set[str]] = defaultdict(set)  # name -> atom_ids
        self.pattern_matcher = PatternMatcher(self)
        
        # Statistics
        self._atom_count = 0
        self._node_count = 0
        self._link_count = 0
    
    def add_atom(self, atom: Atom) -> Atom:
        """Add an atom to the atomspace."""
        # Check if atom already exists
        existing = self.get_atom(atom.id)
        if existing:
            return existing
        
        # Add to main storage
        self._atoms[atom.id] = atom
        
        # Update indexes
        self._type_index[atom.atom_type].add(atom.id)
        if atom.name:
            self._name_index[atom.name].add(atom.id)
        
        # Update statistics
        self._atom_count += 1
        if atom.is_node():
            self._node_count += 1
        elif atom.is_link():
            self._link_count += 1
        
        # Set atomspace reference
        atom.atomspace = self
        
        return atom
    
    def remove_atom(self, atom: Union[Atom, str]) -> bool:
        """Remove an atom from the atomspace."""
        atom_id = atom.id if isinstance(atom, Atom) else atom
        
        if atom_id not in self._atoms:
            return False
        
        atom_obj = self._atoms[atom_id]
        
        # Remove from indexes
        self._type_index[atom_obj.atom_type].discard(atom_id)
        if atom_obj.name:
            self._name_index[atom_obj.name].discard(atom_id)
        
        # Clean up relationships
        for incoming in atom_obj.incoming:
            incoming.remove_outgoing(atom_obj)
        for outgoing in atom_obj.outgoing:
            outgoing.remove_incoming(atom_obj)
        
        # Remove from main storage
        del self._atoms[atom_id]
        
        # Update statistics
        self._atom_count -= 1
        if atom_obj.is_node():
            self._node_count -= 1
        elif atom_obj.is_link():
            self._link_count -= 1
        
        return True
    
    def get_atom(self, atom_id: str) -> Optional[Atom]:
        """Get an atom by its ID."""
        return self._atoms.get(atom_id)
    
    def get_atoms_by_type(self, atom_type: str) -> List[Atom]:
        """Get all atoms of a specific type."""
        atom_ids = self._type_index.get(atom_type, set())
        return [self._atoms[aid] for aid in atom_ids if aid in self._atoms]
    
    def get_atoms_by_name(self, name: str) -> List[Atom]:
        """Get all atoms with a specific name."""
        atom_ids = self._name_index.get(name, set())
        return [self._atoms[aid] for aid in atom_ids if aid in self._atoms]
    
    def get_all_atoms(self) -> List[Atom]:
        """Get all atoms in the atomspace."""
        return list(self._atoms.values())
    
    def find_similar_atoms(self, target: Atom, threshold: float = 0.7) -> List[Tuple[Atom, float]]:
        """Find atoms similar to the target atom."""
        similar = []
        
        for atom in self._atoms.values():
            if atom.id == target.id:
                continue
            
            similarity = self.calculate_similarity(target, atom)
            if similarity >= threshold:
                similar.append((atom, similarity))
        
        # Sort by similarity score
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar
    
    def calculate_similarity(self, atom1: Atom, atom2: Atom) -> float:
        """Calculate similarity between two atoms."""
        similarity = 0.0
        factors = 0
        
        # Type similarity
        if atom1.atom_type == atom2.atom_type:
            similarity += 0.4
        factors += 0.4
        
        # Name similarity (simple string comparison)
        if atom1.name and atom2.name:
            if atom1.name == atom2.name:
                similarity += 0.3
            elif atom1.name.lower() in atom2.name.lower() or atom2.name.lower() in atom1.name.lower():
                similarity += 0.15
        factors += 0.3
        
        # Structural similarity (shared neighbors)
        neighbors1 = atom1.get_neighbors()
        neighbors2 = atom2.get_neighbors()
        if neighbors1 or neighbors2:
            shared = len(neighbors1.intersection(neighbors2))
            total = len(neighbors1.union(neighbors2))
            if total > 0:
                similarity += 0.3 * (shared / total)
        factors += 0.3
        
        return similarity / factors if factors > 0 else 0.0
    
    def evaluate_similarity(self, atom1: Atom, atom2: Atom) -> float:
        """Public method to evaluate similarity between atoms."""
        return self.calculate_similarity(atom1, atom2)
    
    def query(self, pattern: Dict[str, Any]) -> List[Atom]:
        """Query atoms using a pattern specification."""
        results = self.get_all_atoms()
        
        # Filter by type
        if "type" in pattern:
            results = [a for a in results if self.pattern_matcher.match_atom_type(pattern["type"], a)]
        
        # Filter by name pattern
        if "name" in pattern:
            results = [a for a in results if self.pattern_matcher.match_name_pattern(pattern["name"], a)]
        
        # Filter by custom predicate
        if "predicate" in pattern:
            results = [a for a in results if pattern["predicate"](a)]
        
        # Filter by values
        if "values" in pattern:
            value_filters = pattern["values"]
            for key, expected in value_filters.items():
                results = [a for a in results 
                          if a.get_value(key) and a.get_value(key).value == expected]
        
        return results
    
    async def propagate_values(self, source: Atom, value_key: str, max_hops: int = 3) -> None:
        """Propagate values through the network."""
        source_value = source.get_value(value_key)
        if not source_value:
            return
        
        visited = set()
        queue = [(source, 0)]
        
        while queue:
            current, hops = queue.pop(0)
            
            if current.id in visited or hops >= max_hops:
                continue
            
            visited.add(current.id)
            
            # Propagate to neighbors
            for neighbor in current.get_neighbors():
                if neighbor.id not in visited:
                    # Simple propagation: copy value with decay
                    if hasattr(source_value, 'value') and isinstance(source_value.value, (int, float)):
                        decay_factor = 0.8 ** hops
                        neighbor.set_value(value_key, type(source_value)(source_value.value * decay_factor))
                    else:
                        neighbor.set_value(value_key, source_value)
                    
                    queue.append((neighbor, hops + 1))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the atomspace."""
        return {
            "name": self.name,
            "total_atoms": self._atom_count,
            "nodes": self._node_count,
            "links": self._link_count,
            "types": len(self._type_index),
            "named_atoms": len(self._name_index)
        }
    
    def clear(self) -> None:
        """Clear all atoms from the atomspace."""
        self._atoms.clear()
        self._type_index.clear()
        self._name_index.clear()
        self._atom_count = 0
        self._node_count = 0
        self._link_count = 0
    
    def __len__(self) -> int:
        """Get the number of atoms in the atomspace."""
        return self._atom_count
    
    def __contains__(self, atom: Union[Atom, str]) -> bool:
        """Check if an atom is in the atomspace."""
        atom_id = atom.id if isinstance(atom, Atom) else atom
        return atom_id in self._atoms
    
    def __str__(self) -> str:
        return f"AtomSpace(name='{self.name}', atoms={self._atom_count})"
    
    def __repr__(self) -> str:
        return f"AtomSpace(name='{self.name}', nodes={self._node_count}, links={self._link_count})"