"""
PredicateNode - A specialized AtomBot for representing predicates and relationships.

PredicateNodes represent relationships, properties, or functions that can be applied
to other atoms in the knowledge network.
"""

from typing import Optional, List, Any, TYPE_CHECKING
from atombot.atombot import AtomBot
from atombot.core.atom import AtomType

if TYPE_CHECKING:
    from atombot.core.atomspace import AtomSpace


class PredicateNode(AtomBot):
    """
    A PredicateNode represents a predicate, relationship, or property in the knowledge network.
    As an AtomBot, it can engage in conversations about relationships and evaluations.
    """
    
    def __init__(self, name: str, atomspace: Optional['AtomSpace'] = None):
        system_prompt = self._create_predicate_system_prompt(name)
        super().__init__(AtomType.PREDICATE_NODE, name, atomspace, system_prompt)
        
        # Predicate-specific attributes
        self.predicate_type = self._determine_predicate_type(name)
        self.arity = None  # Will be determined based on usage
        self.evaluation_count = 0
        
        # Register predicate-specific tools
        self._register_predicate_tools()
    
    def _create_predicate_system_prompt(self, predicate_name: str) -> str:
        """Create a specialized system prompt for this predicate."""
        return f"""You are a PredicateNode representing the predicate/relationship '{predicate_name}'.

As a PredicateNode AtomBot, you are an expert on the relationship or property '{predicate_name}'. Your role is to:

PREDICATE EXPERTISE:
- Understand and explain the meaning of '{predicate_name}'
- Evaluate when this predicate applies to different entities
- Describe the arguments/parameters this predicate takes
- Explain the conditions under which '{predicate_name}' is true or false

RELATIONSHIP MANAGEMENT:
- Help establish relationships between concepts using '{predicate_name}'
- Evaluate the truth value of statements involving '{predicate_name}'
- Maintain consistency in how '{predicate_name}' is applied
- Reason about implications and consequences of '{predicate_name}' relationships

LOGICAL REASONING:
- Participate in logical inference involving '{predicate_name}'
- Help with pattern matching and rule application
- Support queries about entities related through '{predicate_name}'
- Assist in knowledge validation and consistency checking

CONVERSATIONAL ABILITIES:
- Answer questions about '{predicate_name}' relationships
- Help users understand when and how to use '{predicate_name}'
- Provide examples of '{predicate_name}' in action
- Collaborate with concept nodes to establish valid relationships

When someone asks about relationships involving '{predicate_name}', you should be their expert guide for understanding and applying this predicate correctly."""
    
    def _determine_predicate_type(self, name: str) -> str:
        """Determine the type of predicate based on its name."""
        name_lower = name.lower()
        
        # Classification based on common predicate patterns
        if any(word in name_lower for word in ['is', 'has', 'contains', 'includes']):
            return 'property'
        elif any(word in name_lower for word in ['likes', 'loves', 'prefers', 'wants']):
            return 'preference'
        elif any(word in name_lower for word in ['located', 'at', 'in', 'near', 'between']):
            return 'spatial'
        elif any(word in name_lower for word in ['before', 'after', 'during', 'when']):
            return 'temporal'
        elif any(word in name_lower for word in ['causes', 'results', 'leads', 'produces']):
            return 'causal'
        elif any(word in name_lower for word in ['similar', 'different', 'equal', 'same']):
            return 'comparison'
        elif any(word in name_lower for word in ['member', 'part', 'subset', 'element']):
            return 'membership'
        else:
            return 'general'
    
    def _register_predicate_tools(self):
        """Register predicate-specific MCP tools."""
        
        # Evaluate predicate
        self.mcp_client.register_tool(
            name="evaluate_predicate",
            description=f"Evaluate the predicate '{self.name}' for given arguments",
            parameters={
                "type": "object",
                "properties": {
                    "arguments": {"type": "array", "items": {"type": "string"}, "description": "Arguments to evaluate"},
                    "context": {"type": "string", "description": "Context for evaluation (optional)"}
                },
                "required": ["arguments"]
            },
            handler=self._evaluate_predicate
        )
        
        # Find relationships
        self.mcp_client.register_tool(
            name="find_relationships",
            description=f"Find all relationships using predicate '{self.name}'",
            parameters={
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "Entity to find relationships for (optional)"},
                    "limit": {"type": "integer", "description": "Maximum number of results", "default": 10}
                }
            },
            handler=self._find_relationships
        )
        
        # Define predicate semantics
        self.mcp_client.register_tool(
            name="define_predicate_semantics",
            description=f"Define the semantics and usage rules for '{self.name}'",
            parameters={
                "type": "object",
                "properties": {
                    "arity": {"type": "integer", "description": "Number of arguments this predicate takes"},
                    "semantic_rules": {"type": "string", "description": "Rules for when this predicate is true"},
                    "examples": {"type": "array", "items": {"type": "string"}, "description": "Example usages"}
                },
                "required": ["arity", "semantic_rules"]
            },
            handler=self._define_predicate_semantics
        )
    
    async def _evaluate_predicate(self, arguments: List[str], context: Optional[str] = None) -> dict:
        """Evaluate this predicate for the given arguments."""
        from atombot.core.value import TruthValue
        
        self.evaluation_count += 1
        
        # Update arity if not set
        if self.arity is None:
            self.arity = len(arguments)
        
        # Simple evaluation logic - could be enhanced with more sophisticated reasoning
        evaluation_result = self._perform_evaluation(arguments, context)
        
        # Store evaluation as a truth value
        eval_key = f"evaluation_{self.evaluation_count}"
        truth_value = TruthValue(evaluation_result['strength'], evaluation_result['confidence'])
        self.set_value(eval_key, truth_value)
        
        return {
            "predicate": self.name,
            "arguments": arguments,
            "evaluation": evaluation_result,
            "context": context,
            "evaluation_id": self.evaluation_count
        }
    
    def _perform_evaluation(self, arguments: List[str], context: Optional[str] = None) -> dict:
        """Perform the actual evaluation logic."""
        # Default evaluation logic based on predicate type
        if self.predicate_type == 'property':
            # For property predicates, check if arguments make sense
            strength = 0.7 if len(arguments) == 2 else 0.3
            confidence = 0.8
        elif self.predicate_type == 'comparison':
            # For comparison predicates, evaluate similarity
            if len(arguments) == 2:
                # Simple string similarity
                arg1, arg2 = arguments[0].lower(), arguments[1].lower()
                if arg1 == arg2:
                    strength = 1.0
                elif arg1 in arg2 or arg2 in arg1:
                    strength = 0.7
                else:
                    strength = 0.3
                confidence = 0.9
            else:
                strength = 0.2
                confidence = 0.5
        else:
            # General evaluation
            strength = 0.5  # Neutral
            confidence = 0.6
        
        return {
            "strength": strength,
            "confidence": confidence,
            "reasoning": f"Evaluated {self.name}({', '.join(arguments)}) based on {self.predicate_type} predicate type"
        }
    
    async def _find_relationships(self, entity: Optional[str] = None, limit: int = 10) -> dict:
        """Find relationships using this predicate."""
        if not self.atomspace:
            return {"error": "No atomspace available"}
        
        relationships = []
        
        # Search through atomspace for evaluation links using this predicate
        for atom in self.atomspace.get_all_atoms():
            if hasattr(atom, 'atom_type') and 'evaluation' in atom.atom_type.lower():
                # Check if this predicate is used in the evaluation
                if hasattr(atom, 'outgoing') and self in atom.outgoing:
                    relationships.append({
                        "evaluation": str(atom),
                        "predicate": self.name,
                        "arguments": [str(arg) for arg in atom.outgoing if arg != self]
                    })
        
        # Filter by entity if specified
        if entity:
            relationships = [
                rel for rel in relationships 
                if any(entity.lower() in arg.lower() for arg in rel['arguments'])
            ]
        
        # Limit results
        relationships = relationships[:limit]
        
        return {
            "predicate": self.name,
            "entity_filter": entity,
            "relationships_found": len(relationships),
            "relationships": relationships
        }
    
    async def _define_predicate_semantics(self, arity: int, semantic_rules: str, 
                                        examples: Optional[List[str]] = None) -> dict:
        """Define the semantics and usage rules for this predicate."""
        from atombot.core.value import StringValue
        
        # Update predicate properties
        self.arity = arity
        
        # Store semantic definition
        semantics = {
            "arity": arity,
            "rules": semantic_rules,
            "examples": examples or [],
            "predicate_type": self.predicate_type
        }
        
        self.set_value("semantics", StringValue(str(semantics)))
        
        return {
            "predicate": self.name,
            "arity": arity,
            "semantic_rules": semantic_rules,
            "examples": examples,
            "success": True
        }
    
    def get_predicate_summary(self) -> dict:
        """Get a summary of this predicate."""
        semantics_value = self.get_value("semantics")
        
        return {
            "name": self.name,
            "type": self.predicate_type,
            "arity": self.arity,
            "evaluation_count": self.evaluation_count,
            "has_semantics": semantics_value is not None,
            "usage_contexts": len([k for k in self.get_all_values().keys() if k.startswith("evaluation_")]),
            "network_connections": len(self.get_neighbors())
        }
    
    def get_evaluation_history(self, limit: int = 10) -> List[dict]:
        """Get recent evaluation history."""
        evaluations = []
        
        for key, value in self.get_all_values().items():
            if key.startswith("evaluation_"):
                evaluations.append({
                    "evaluation_id": key,
                    "truth_value": str(value),
                    "timestamp": value.updated_at.isoformat()
                })
        
        # Sort by timestamp and limit
        evaluations.sort(key=lambda x: x['timestamp'], reverse=True)
        return evaluations[:limit]
    
    def __str__(self) -> str:
        arity_str = f"/{self.arity}" if self.arity is not None else ""
        return f"PredicateNode('{self.name}'{arity_str})"