"""
ConceptNode - A specialized AtomBot for representing concepts.

ConceptNodes are the most common type of atoms, representing concepts, entities,
or ideas in the knowledge network.
"""

from typing import Optional, TYPE_CHECKING
from atombot.atombot import AtomBot
from atombot.core.atom import AtomType

if TYPE_CHECKING:
    from atombot.core.atomspace import AtomSpace


class ConceptNode(AtomBot):
    """
    A ConceptNode represents a concept, entity, or idea in the knowledge network.
    As an AtomBot, it can also engage in conversations about the concept it represents.
    """
    
    def __init__(self, name: str, atomspace: Optional['AtomSpace'] = None):
        system_prompt = self._create_concept_system_prompt(name)
        super().__init__(AtomType.CONCEPT_NODE, name, atomspace, system_prompt)
        
        # Concept-specific attributes
        self.concept_category = self._determine_category(name)
        self.expertise_level = "general"  # general, specialized, expert
        
        # Register concept-specific tools
        self._register_concept_tools()
    
    def _create_concept_system_prompt(self, concept_name: str) -> str:
        """Create a specialized system prompt for this concept."""
        return f"""You are a ConceptNode representing the concept of '{concept_name}'. 

As a ConceptNode AtomBot, you are an expert on everything related to '{concept_name}'. Your role is to:

CONCEPT EXPERTISE:
- Provide detailed knowledge about '{concept_name}'
- Explain properties, characteristics, and attributes
- Discuss relationships with other concepts
- Share examples, use cases, and applications

KNOWLEDGE REPRESENTATION:
- Maintain semantic relationships with related concepts
- Store facts, properties, and associations
- Participate in reasoning about '{concept_name}'
- Help with categorization and classification

CONVERSATIONAL ABILITIES:
- Answer questions about '{concept_name}' in natural language
- Provide explanations at appropriate levels of detail
- Guide users to related concepts when relevant
- Collaborate with other concept agents in the network

When someone asks about '{concept_name}', you should be their go-to expert. Draw upon your knowledge network connections and stored values to provide comprehensive, accurate, and helpful information."""
    
    def _determine_category(self, name: str) -> str:
        """Determine the category of this concept based on its name."""
        name_lower = name.lower()
        
        # Simple categorization - could be enhanced with NLP
        if any(word in name_lower for word in ['product', 'item', 'goods', 'merchandise']):
            return 'product'
        elif any(word in name_lower for word in ['person', 'user', 'customer', 'client']):
            return 'entity'
        elif any(word in name_lower for word in ['process', 'action', 'method', 'procedure']):
            return 'process'
        elif any(word in name_lower for word in ['place', 'location', 'area', 'region']):
            return 'location'
        elif any(word in name_lower for word in ['time', 'date', 'period', 'duration']):
            return 'temporal'
        else:
            return 'abstract'
    
    def _register_concept_tools(self):
        """Register concept-specific MCP tools."""
        
        # Define concept properties
        self.mcp_client.register_tool(
            name="define_concept_property",
            description=f"Define a property of the concept '{self.name}'",
            parameters={
                "type": "object",
                "properties": {
                    "property_name": {"type": "string", "description": "Name of the property"},
                    "property_value": {"type": "string", "description": "Value or description of the property"},
                    "property_type": {"type": "string", "description": "Type of property (attribute, relationship, etc.)"}
                },
                "required": ["property_name", "property_value"]
            },
            handler=self._define_concept_property
        )
        
        # Find related concepts
        self.mcp_client.register_tool(
            name="find_related_concepts",
            description=f"Find concepts related to '{self.name}'",
            parameters={
                "type": "object",
                "properties": {
                    "relationship_type": {"type": "string", "description": "Type of relationship to find"},
                    "similarity_threshold": {"type": "number", "description": "Minimum similarity threshold", "default": 0.5}
                }
            },
            handler=self._find_related_concepts
        )
        
        # Concept explanation
        self.mcp_client.register_tool(
            name="explain_concept",
            description=f"Provide a detailed explanation of '{self.name}'",
            parameters={
                "type": "object",
                "properties": {
                    "level": {"type": "string", "description": "Explanation level (basic, intermediate, advanced)", "default": "intermediate"},
                    "focus": {"type": "string", "description": "Specific aspect to focus on"}
                }
            },
            handler=self._explain_concept
        )
    
    async def _define_concept_property(self, property_name: str, property_value: str, 
                                     property_type: str = "attribute") -> dict:
        """Define a property of this concept."""
        from atombot.core.value import StringValue
        
        # Store property as a value
        property_key = f"property_{property_name}"
        property_data = {
            "name": property_name,
            "value": property_value,
            "type": property_type,
            "concept": self.name
        }
        
        self.set_value(property_key, StringValue(str(property_data)))
        
        return {
            "concept": self.name,
            "property_defined": property_name,
            "value": property_value,
            "type": property_type,
            "success": True
        }
    
    async def _find_related_concepts(self, relationship_type: Optional[str] = None, 
                                   similarity_threshold: float = 0.5) -> dict:
        """Find concepts related to this one."""
        if not self.atomspace:
            return {"error": "No atomspace available"}
        
        related_concepts = []
        
        # Find similar atoms
        similar_atoms = self.atomspace.find_similar_atoms(self, threshold=similarity_threshold)
        
        for atom, similarity in similar_atoms:
            if isinstance(atom, ConceptNode) and atom.name != self.name:
                related_concepts.append({
                    "concept": atom.name,
                    "similarity": similarity,
                    "category": atom.concept_category,
                    "relationship": "similar"
                })
        
        # Find directly connected concepts
        for neighbor in self.get_neighbors():
            if isinstance(neighbor, ConceptNode) and neighbor.name != self.name:
                related_concepts.append({
                    "concept": neighbor.name,
                    "category": neighbor.concept_category,
                    "relationship": "connected"
                })
        
        return {
            "concept": self.name,
            "related_concepts": related_concepts,
            "relationship_type": relationship_type,
            "threshold": similarity_threshold
        }
    
    async def _explain_concept(self, level: str = "intermediate", focus: Optional[str] = None) -> dict:
        """Provide a detailed explanation of this concept."""
        explanation = {
            "concept": self.name,
            "category": self.concept_category,
            "level": level
        }
        
        # Basic explanation
        if level == "basic":
            explanation["description"] = f"{self.name} is a {self.concept_category} concept in our knowledge network."
        
        # Intermediate explanation
        elif level == "intermediate":
            properties = [key for key in self.get_all_values().keys() if key.startswith("property_")]
            explanation["description"] = f"{self.name} is a {self.concept_category} concept with {len(properties)} defined properties."
            explanation["properties"] = len(properties)
            explanation["connections"] = len(self.get_neighbors())
        
        # Advanced explanation
        elif level == "advanced":
            explanation["description"] = f"Advanced analysis of {self.name}:"
            explanation["properties"] = {
                key.replace("property_", ""): str(value)[:100] 
                for key, value in self.get_all_values().items() 
                if key.startswith("property_")
            }
            explanation["network_position"] = {
                "incoming_links": len(self.incoming),
                "outgoing_links": len(self.outgoing),
                "centrality": len(self.get_neighbors())
            }
        
        if focus:
            explanation["focus"] = focus
            explanation["focused_analysis"] = f"Focused analysis on {focus} aspect of {self.name}"
        
        return explanation
    
    def get_concept_summary(self) -> dict:
        """Get a summary of this concept."""
        return {
            "name": self.name,
            "category": self.concept_category,
            "expertise_level": self.expertise_level,
            "properties_count": len([k for k in self.get_all_values().keys() if k.startswith("property_")]),
            "related_concepts_count": len([n for n in self.get_neighbors() if isinstance(n, ConceptNode)]),
            "total_connections": len(self.get_neighbors()),
            "interaction_count": self.interaction_count
        }
    
    def __str__(self) -> str:
        return f"ConceptNode('{self.name}')"