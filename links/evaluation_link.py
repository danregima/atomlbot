"""
EvaluationLink - A link that evaluates a predicate with arguments.

EvaluationLinks represent the application of a predicate to specific arguments,
forming statements that can be true or false.
"""

from typing import List, Optional, TYPE_CHECKING
from atombot.core.atom import Link, AtomType, Atom
from atombot.core.value import TruthValue

if TYPE_CHECKING:
    from atombot.core.atomspace import AtomSpace
    from atombot.nodes.predicate_node import PredicateNode


class EvaluationLink(Link):
    """
    An EvaluationLink applies a predicate to arguments, creating an evaluable statement.
    
    Structure: EvaluationLink(PredicateNode, ListLink(arguments...))
    Example: EvaluationLink(PredicateNode("likes"), ListLink(ConceptNode("John"), ConceptNode("Pizza")))
    """
    
    def __init__(self, predicate: 'PredicateNode', arguments: List[Atom], 
                 atomspace: Optional['AtomSpace'] = None, truth_value: Optional[TruthValue] = None):
        # Create the standard evaluation structure
        from atombot.links.list_link import ListLink
        
        # Create argument list
        arg_list = ListLink(arguments, atomspace)
        
        # Initialize as link with predicate and argument list
        super().__init__(AtomType.EVALUATION_LINK, [predicate, arg_list], atomspace)
        
        # Set truth value if provided
        if truth_value:
            self.set_value("truth_value", truth_value)
        else:
            # Default truth value
            self.set_value("truth_value", TruthValue(0.5, 0.5))
    
    @property
    def predicate(self) -> 'PredicateNode':
        """Get the predicate of this evaluation."""
        return self.get_target(0)
    
    @property  
    def arguments(self) -> List[Atom]:
        """Get the arguments of this evaluation."""
        arg_list = self.get_target(1)
        if arg_list and hasattr(arg_list, 'get_targets'):
            return arg_list.get_targets()
        return []
    
    def get_truth_value(self) -> TruthValue:
        """Get the truth value of this evaluation."""
        tv = self.get_value("truth_value")
        if isinstance(tv, TruthValue):
            return tv
        return TruthValue(0.5, 0.5)  # Default
    
    def set_truth_value(self, truth_value: TruthValue) -> None:
        """Set the truth value of this evaluation."""
        self.set_value("truth_value", truth_value)
    
    async def evaluate(self) -> TruthValue:
        """Evaluate this statement and update its truth value."""
        if hasattr(self.predicate, 'evaluate_predicate'):
            # Use the predicate's evaluation capabilities
            arg_names = [str(arg) for arg in self.arguments]
            result = await self.predicate._evaluate_predicate(arg_names)
            
            if result.get('evaluation'):
                eval_data = result['evaluation']
                new_tv = TruthValue(eval_data['strength'], eval_data['confidence'])
                self.set_truth_value(new_tv)
                return new_tv
        
        # Fallback evaluation
        return self.get_truth_value()
    
    def is_true(self) -> bool:
        """Check if this evaluation is true."""
        return self.get_truth_value().is_true()
    
    def is_false(self) -> bool:
        """Check if this evaluation is false.""" 
        return self.get_truth_value().is_false()
    
    def is_unknown(self) -> bool:
        """Check if this evaluation is unknown."""
        return self.get_truth_value().is_unknown()
    
    def __str__(self) -> str:
        pred_name = str(self.predicate) if self.predicate else "Unknown"
        arg_names = [str(arg) for arg in self.arguments]
        tv = self.get_truth_value()
        return f"EvaluationLink({pred_name}({', '.join(arg_names)}) {tv})"