"""
Test suite for AtomBot core functionality.
"""

import pytest
import asyncio
from atombot import AtomSpace, ConceptNode, PredicateNode, EvaluationLink, InheritanceLink
from atombot.core.value import TruthValue, StringValue, FloatValue


class TestAtomSpace:
    """Test AtomSpace functionality."""
    
    def test_atomspace_creation(self):
        """Test basic atomspace creation."""
        atomspace = AtomSpace("TestSpace")
        assert atomspace.name == "TestSpace"
        assert len(atomspace) == 0
    
    def test_atom_addition(self):
        """Test adding atoms to atomspace."""
        atomspace = AtomSpace()
        concept = ConceptNode("TestConcept", atomspace)
        
        assert len(atomspace) == 1
        assert concept in atomspace
        assert atomspace.get_atom(concept.id) == concept
    
    def test_atom_removal(self):
        """Test removing atoms from atomspace."""
        atomspace = AtomSpace()
        concept = ConceptNode("TestConcept", atomspace)
        atom_id = concept.id
        
        assert len(atomspace) == 1
        atomspace.remove_atom(concept)
        assert len(atomspace) == 0
        assert atomspace.get_atom(atom_id) is None
    
    def test_query_by_type(self):
        """Test querying atoms by type."""
        atomspace = AtomSpace()
        concept1 = ConceptNode("Concept1", atomspace)
        concept2 = ConceptNode("Concept2", atomspace)
        predicate = PredicateNode("TestPredicate", atomspace)
        
        concepts = atomspace.get_atoms_by_type("ConceptNode")
        predicates = atomspace.get_atoms_by_type("PredicateNode")
        
        assert len(concepts) == 2
        assert len(predicates) == 1
        assert concept1 in concepts
        assert concept2 in concepts
        assert predicate in predicates
    
    def test_similarity_calculation(self):
        """Test similarity calculation between atoms."""
        atomspace = AtomSpace()
        concept1 = ConceptNode("Dog", atomspace)
        concept2 = ConceptNode("Cat", atomspace)
        concept3 = ConceptNode("Canine", atomspace)
        
        similarity1 = atomspace.calculate_similarity(concept1, concept2)
        similarity2 = atomspace.calculate_similarity(concept1, concept3)
        
        assert 0 <= similarity1 <= 1
        assert 0 <= similarity2 <= 1


class TestConceptNode:
    """Test ConceptNode functionality."""
    
    def test_concept_creation(self):
        """Test creating concept nodes."""
        concept = ConceptNode("TestConcept")
        
        assert concept.name == "TestConcept"
        assert concept.atom_type == "ConceptNode"
        assert concept.is_node()
        assert not concept.is_link()
    
    def test_concept_category_detection(self):
        """Test automatic category detection."""
        product = ConceptNode("Product Catalog")
        person = ConceptNode("Customer Smith")
        
        assert product.concept_category in ['product', 'abstract']
        assert person.concept_category in ['entity', 'abstract']
    
    @pytest.mark.asyncio
    async def test_concept_tools(self):
        """Test concept-specific MCP tools."""
        concept = ConceptNode("TestConcept")
        
        # Test property definition
        result = await concept._define_concept_property(
            "color", "blue", "attribute"
        )
        
        assert result["success"] is True
        assert result["property_defined"] == "color"
        assert result["value"] == "blue"
        
        # Check that property was stored
        property_value = concept.get_value("property_color")
        assert property_value is not None


class TestPredicateNode:
    """Test PredicateNode functionality."""
    
    def test_predicate_creation(self):
        """Test creating predicate nodes."""
        predicate = PredicateNode("likes")
        
        assert predicate.name == "likes"
        assert predicate.atom_type == "PredicateNode"
        assert predicate.predicate_type in ['preference', 'general']
    
    @pytest.mark.asyncio 
    async def test_predicate_evaluation(self):
        """Test predicate evaluation."""
        predicate = PredicateNode("likes")
        
        result = await predicate._evaluate_predicate(["John", "Pizza"])
        
        assert "evaluation" in result
        assert "strength" in result["evaluation"]
        assert "confidence" in result["evaluation"]
        assert 0 <= result["evaluation"]["strength"] <= 1
        assert 0 <= result["evaluation"]["confidence"] <= 1


class TestLinks:
    """Test link functionality."""
    
    def test_inheritance_link_creation(self):
        """Test creating inheritance links."""
        child = ConceptNode("Dog")
        parent = ConceptNode("Animal")
        
        inheritance = InheritanceLink(child, parent)
        
        assert inheritance.child == child
        assert inheritance.parent == parent
        assert inheritance.atom_type == "InheritanceLink"
        assert inheritance.is_link()
        assert not inheritance.is_node()
    
    def test_evaluation_link_creation(self):
        """Test creating evaluation links."""
        predicate = PredicateNode("likes")
        arg1 = ConceptNode("John")
        arg2 = ConceptNode("Pizza")
        
        evaluation = EvaluationLink(predicate, [arg1, arg2])
        
        assert evaluation.predicate == predicate
        assert len(evaluation.arguments) == 2
        assert arg1 in evaluation.arguments
        assert arg2 in evaluation.arguments
    
    @pytest.mark.asyncio
    async def test_evaluation_link_evaluation(self):
        """Test evaluating evaluation links."""
        predicate = PredicateNode("likes")
        arg1 = ConceptNode("John")
        arg2 = ConceptNode("Pizza")
        
        evaluation = EvaluationLink(predicate, [arg1, arg2])
        truth_value = await evaluation.evaluate()
        
        assert isinstance(truth_value, TruthValue)
        assert 0 <= truth_value.strength <= 1
        assert 0 <= truth_value.confidence <= 1


class TestValues:
    """Test value functionality."""
    
    def test_truth_value(self):
        """Test truth value creation and operations."""
        tv = TruthValue(0.8, 0.9)
        
        assert tv.strength == 0.8
        assert tv.confidence == 0.9
        assert tv.is_true()
        assert not tv.is_false()
        assert not tv.is_unknown()
    
    def test_string_value(self):
        """Test string value operations."""
        sv1 = StringValue("Hello")
        sv2 = StringValue(" World")
        
        combined = sv1 + sv2
        assert str(combined) == "Hello World"
    
    def test_float_value(self):
        """Test float value operations."""
        fv1 = FloatValue(3.5)
        fv2 = FloatValue(2.0)
        
        result = fv1 + fv2
        assert float(result) == 5.5


class TestAtomBot:
    """Test AtomBot hybrid functionality."""
    
    def test_atombot_creation(self):
        """Test creating AtomBot instances."""
        atomspace = AtomSpace()
        atombot = ConceptNode("TestBot", atomspace)
        
        # Should have both atom and agent properties
        assert hasattr(atombot, 'atom_type')
        assert hasattr(atombot, 'chat')
        assert hasattr(atombot, 'mcp_client')
        assert atombot.agent_type == "AtomBot"
    
    @pytest.mark.asyncio
    async def test_atombot_tools(self):
        """Test AtomBot-specific tools."""
        atomspace = AtomSpace()
        atombot = ConceptNode("TestBot", atomspace)
        
        # Test knowledge search
        search_result = await atombot._search_my_knowledge("test")
        assert "query" in search_result
        assert "results" in search_result
    
    def test_atombot_status(self):
        """Test getting AtomBot status."""
        atomspace = AtomSpace()
        atombot = ConceptNode("TestBot", atomspace)
        
        status = atombot.get_atombot_status()
        
        assert "atom_info" in status
        assert "agent_info" in status
        assert "network_info" in status
        assert status["atom_info"]["name"] == "TestBot"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])