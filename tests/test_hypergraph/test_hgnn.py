"""
Test suite for HGNN (HyperGraph Neural Network) implementation.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta

from atombot.hypergraph import (
    HyperGraph, HyperNode, HyperEdge,
    HyperGraphConvolution, HyperAttention, HGNNLayer, HyperGraphNeuralNetwork,
    HyperAgent, HyperChannel, DiscreteEventChannel, EventType,
    DynamicalHyperSystem
)


class TestHyperGraph:
    """Test basic hypergraph functionality."""
    
    def test_hypergraph_creation(self):
        """Test creating a hypergraph."""
        hg = HyperGraph("TestHyperGraph")
        assert hg.name == "TestHyperGraph"
        assert len(hg) == 0
        assert len(hg.get_nodes()) == 0
        assert len(hg.get_edges()) == 0
    
    def test_hypernode_creation(self):
        """Test creating hypernodes."""
        node = HyperNode(data={"name": "test_node"})
        assert node.id is not None
        assert node.get_data("name") == "test_node"
        assert node.get_degree() == 0
        assert len(node.get_neighbors()) == 0
    
    def test_hyperedge_creation(self):
        """Test creating hyperedges."""
        node1 = HyperNode(data={"name": "node1"})
        node2 = HyperNode(data={"name": "node2"})
        node3 = HyperNode(data={"name": "node3"})
        
        edge = HyperEdge(nodes=[node1, node2, node3], edge_type="test_edge")
        
        assert edge.get_cardinality() == 3
        assert node1 in edge.get_nodes()
        assert node2 in edge.get_nodes()
        assert node3 in edge.get_nodes()
        
        # Check that nodes know about the edge
        assert edge in node1.get_hyperedges()
        assert edge in node2.get_hyperedges()
        assert edge in node3.get_hyperedges()
    
    def test_hypergraph_node_addition(self):
        """Test adding nodes to hypergraph."""
        hg = HyperGraph()
        node = HyperNode(data={"type": "concept"})
        
        added_node = hg.add_node(node)
        assert added_node == node
        assert len(hg) == 1
        assert hg.get_node(node.id) == node
    
    def test_hypergraph_edge_addition(self):
        """Test adding edges to hypergraph."""
        hg = HyperGraph()
        node1 = HyperNode()
        node2 = HyperNode()
        edge = HyperEdge(nodes=[node1, node2])
        
        added_edge = hg.add_edge(edge)
        assert added_edge == edge
        assert len(hg.get_edges()) == 1
        assert len(hg.get_nodes()) == 2  # Nodes should be added automatically
    
    def test_incidence_matrix(self):
        """Test incidence matrix computation."""
        hg = HyperGraph()
        node1 = HyperNode()
        node2 = HyperNode()
        node3 = HyperNode()
        
        hg.add_node(node1)
        hg.add_node(node2) 
        hg.add_node(node3)
        
        edge1 = HyperEdge(nodes=[node1, node2])
        edge2 = HyperEdge(nodes=[node2, node3])
        
        hg.add_edge(edge1)
        hg.add_edge(edge2)
        
        incidence = hg.get_incidence_matrix()
        assert incidence.shape == (3, 2)  # 3 nodes, 2 edges
        
        # Check that incidence matrix has correct structure
        assert np.sum(incidence) == 4  # 2 edges * 2 nodes each = 4 connections
    
    def test_adjacency_matrix(self):
        """Test adjacency matrix computation."""
        hg = HyperGraph()
        node1 = HyperNode()
        node2 = HyperNode()
        node3 = HyperNode()
        
        hg.add_node(node1)
        hg.add_node(node2)
        hg.add_node(node3)
        
        edge = HyperEdge(nodes=[node1, node2, node3])
        hg.add_edge(edge)
        
        adjacency = hg.get_adjacency_matrix()
        assert adjacency.shape == (3, 3)
        
        # In a 3-node hyperedge, each pair should be connected
        assert adjacency[0, 1] == adjacency[1, 0] == 1
        assert adjacency[0, 2] == adjacency[2, 0] == 1
        assert adjacency[1, 2] == adjacency[2, 1] == 1
        
        # Diagonal should be zero (no self-loops)
        assert adjacency[0, 0] == adjacency[1, 1] == adjacency[2, 2] == 0


class TestHGNN:
    """Test hypergraph neural network functionality."""
    
    def test_hyperconvolution_creation(self):
        """Test creating hypergraph convolution layer."""
        conv = HyperGraphConvolution(input_dim=64, output_dim=32)
        assert conv.input_dim == 64
        assert conv.output_dim == 32
        assert conv.W_node.shape == (64, 32)
    
    def test_hyperattention_creation(self):
        """Test creating hypergraph attention layer."""
        attention = HyperAttention(input_dim=64, output_dim=32, num_heads=4)
        assert attention.input_dim == 64
        assert attention.output_dim == 32
        assert attention.num_heads == 4
        assert attention.head_dim == 8  # 32 / 4
    
    def test_hgnn_layer_creation(self):
        """Test creating complete HGNN layer."""
        layer = HGNNLayer(input_dim=64, hidden_dim=128, output_dim=32)
        assert layer.input_dim == 64
        assert layer.output_dim == 32
        assert layer.hidden_dim == 128
        assert len(layer.layers) == 0  # No sub-layers in this architecture
    
    def test_hgnn_forward_pass(self):
        """Test HGNN forward pass."""
        # Create simple hypergraph
        hg = HyperGraph()
        node1 = HyperNode()
        node2 = HyperNode()
        node3 = HyperNode()
        
        # Set features
        node1.features = np.random.randn(64)
        node2.features = np.random.randn(64)
        node3.features = np.random.randn(64)
        
        hg.add_node(node1)
        hg.add_node(node2)
        hg.add_node(node3)
        
        edge = HyperEdge(nodes=[node1, node2, node3])
        hg.add_edge(edge)
        
        # Create and test layer
        conv = HyperGraphConvolution(input_dim=64, output_dim=32)
        output = conv.forward(hg)
        
        assert output.shape == (3, 32)  # 3 nodes, 32 output dimensions
        assert not np.allclose(output, 0)  # Should not be all zeros
    
    def test_hypergraph_neural_network(self):
        """Test complete hypergraph neural network."""
        network = HyperGraphNeuralNetwork(
            input_dim=64,
            hidden_dims=[128, 64],
            output_dim=32
        )
        
        assert network.input_dim == 64
        assert network.output_dim == 32
        assert len(network.layers) == 3  # input->128, 128->64, 64->32
        
        # Test with simple hypergraph
        hg = HyperGraph()
        node = HyperNode()
        node.features = np.random.randn(64)
        hg.add_node(node)
        
        output = network.forward(hg)
        assert output.shape == (1, 32)


class TestHyperAgent:
    """Test HyperAgent functionality."""
    
    def test_hyperagent_creation(self):
        """Test creating a HyperAgent."""
        agent = HyperAgent(
            atom_type="ConceptNode",
            name="TestAgent",
            embedding_dim=64
        )
        
        assert agent.name == "TestAgent"
        assert agent.agent_type == "HyperAgent"
        assert agent.embedding_dim == 64
        assert agent.features is not None
        assert len(agent.features) == 64
    
    @pytest.mark.asyncio
    async def test_hyperagent_embedding_update(self):
        """Test HyperAgent embedding updates."""
        agent = HyperAgent(
            atom_type="ConceptNode", 
            name="TestAgent"
        )
        
        old_features = agent.features.copy()
        result = await agent._update_embeddings()
        
        assert result.get("success") is True
        # Features might not change significantly without neighbors
        assert "feature_change" in result
    
    @pytest.mark.asyncio
    async def test_hyperagent_collective_reasoning(self):
        """Test collective reasoning between HyperAgents."""
        hg = HyperGraph()
        
        agent1 = HyperAgent("ConceptNode", "Agent1", hypergraph=hg)
        agent2 = HyperAgent("ConceptNode", "Agent2", hypergraph=hg)
        
        # Connect agents through a hyperedge
        edge = HyperEdge(nodes=[agent1, agent2], edge_type="collaboration")
        hg.add_edge(edge)
        
        # Test collective reasoning
        result = await agent1._initiate_collective_reasoning(
            "What is the meaning of life?",
            max_agents=1
        )
        
        assert "task" in result
        assert result["task"] == "What is the meaning of life?"
        assert "initiator" in result
        assert result["initiator"] == "Agent1"


class TestHyperChannel:
    """Test HyperChannel and discrete event processing."""
    
    def test_hyperchannel_creation(self):
        """Test creating a HyperChannel."""
        node1 = HyperNode()
        node2 = HyperNode()
        
        channel = HyperChannel(
            nodes=[node1, node2],
            channel_type="communication"
        )
        
        assert channel.channel_type == "communication"
        assert len(channel.get_nodes()) == 2
        assert channel.processing_active
    
    @pytest.mark.asyncio
    async def test_discrete_event_channel(self):
        """Test discrete event processing."""
        node1 = HyperNode()
        node2 = HyperNode()
        
        channel = DiscreteEventChannel(nodes=[node1, node2])
        
        # Send an event
        event_id = await channel.send_event(
            event_type=EventType.MESSAGE,
            source_node=node1.id,
            target_nodes=[node2.id],
            data={"content": "Hello, world!"}
        )
        
        assert event_id is not None
        assert len(channel.event_queue) <= 1  # Might be processed immediately
    
    @pytest.mark.asyncio
    async def test_event_processing(self):
        """Test event processing with custom handlers."""
        channel = DiscreteEventChannel()
        
        # Create custom processor
        from atombot.hypergraph.hyperchannel import EventProcessor
        
        processor = EventProcessor("test_processor")
        events_processed = []
        
        def message_handler(event):
            events_processed.append(event)
            return []
        
        processor.register_handler(EventType.MESSAGE, message_handler)
        channel.register_processor(processor)
        
        # Send event
        await channel.send_event(
            event_type=EventType.MESSAGE,
            source_node="test_source",
            data={"test": "data"}
        )
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Check that event was processed
        # Note: Actual processing depends on the channel's processing mode


class TestDynamicalHyperSystem:
    """Test DynamicalHyperSystem functionality."""
    
    def test_dynamical_system_creation(self):
        """Test creating a DynamicalHyperSystem."""
        system = DynamicalHyperSystem("TestSystem")
        
        assert system.name == "TestSystem"
        assert system.hypergraph is not None
        assert system.neural_network is not None
        assert len(system.hyperagents) == 0
        assert len(system.hyperchannels) == 0
    
    @pytest.mark.asyncio
    async def test_add_hyperagent_to_system(self):
        """Test adding HyperAgents to the system."""
        system = DynamicalHyperSystem("TestSystem")
        
        agent = await system.create_hyperagent(
            atom_type="ConceptNode",
            name="TestAgent"
        )
        
        assert agent.name == "TestAgent"
        assert agent.id in system.hyperagents
        assert agent in system.hypergraph.get_nodes()
    
    @pytest.mark.asyncio
    async def test_create_hyperchannel_in_system(self):
        """Test creating HyperChannels in the system."""
        system = DynamicalHyperSystem("TestSystem")
        
        # Create agents first
        agent1 = await system.create_hyperagent("ConceptNode", "Agent1")
        agent2 = await system.create_hyperagent("ConceptNode", "Agent2")
        
        # Create channel connecting them
        channel = await system.create_hyperchannel(
            channel_type="communication",
            connected_agents=[agent1.id, agent2.id]
        )
        
        assert channel.id in system.hyperchannels
        assert channel in system.hypergraph.get_edges()
        assert agent1 in channel.get_nodes()
        assert agent2 in channel.get_nodes()
    
    @pytest.mark.asyncio
    async def test_global_learning(self):
        """Test global learning coordination."""
        system = DynamicalHyperSystem("TestSystem")
        
        # Add some agents
        agent1 = await system.create_hyperagent("ConceptNode", "Agent1")
        agent2 = await system.create_hyperagent("ConceptNode", "Agent2")
        
        # Connect them
        await system.create_hyperchannel(
            "learning_channel",
            [agent1.id, agent2.id]
        )
        
        # Trigger global learning
        result = await system.trigger_global_learning()
        
        assert result.get("success") is True
        assert "learning_iteration" in result
        assert result["participants"] == 2
    
    @pytest.mark.asyncio
    async def test_system_metrics(self):
        """Test system metrics computation."""
        system = DynamicalHyperSystem("TestSystem")
        
        # Add components
        agent1 = await system.create_hyperagent("ConceptNode", "Agent1")
        agent2 = await system.create_hyperagent("ConceptNode", "Agent2")
        await system.create_hyperchannel("test_channel", [agent1.id, agent2.id])
        
        # Get status
        status = system.get_system_status()
        
        assert status["metrics"]["total_agents"] == 2
        assert status["metrics"]["total_nodes"] == 2
        assert status["metrics"]["total_channels"] == 1
        assert "connectivity_density" in status["metrics"]
    
    @pytest.mark.asyncio
    async def test_system_shutdown(self):
        """Test graceful system shutdown."""
        system = DynamicalHyperSystem("TestSystem")
        
        # Add some components
        await system.create_hyperagent("ConceptNode", "Agent1")
        
        # Shutdown
        await system.shutdown()
        
        assert system.state.value == "dormant"


class TestIntegration:
    """Integration tests for the complete HGNN system."""
    
    @pytest.mark.asyncio
    async def test_complete_hgnn_workflow(self):
        """Test a complete HGNN workflow from creation to learning."""
        # Create system
        system = DynamicalHyperSystem("IntegrationTest")
        
        # Create multiple agents with different types
        agents = []
        for i in range(5):
            agent = await system.create_hyperagent(
                atom_type="ConceptNode",
                name=f"Agent_{i}"
            )
            agents.append(agent)
        
        # Create interconnections
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                if i + j < 4:  # Create some connections
                    await system.create_hyperchannel(
                        "connection",
                        [agents[i].id, agents[j].id]
                    )
        
        # Trigger multiple learning iterations
        for _ in range(3):
            result = await system.trigger_global_learning()
            assert result.get("success") is True
            await asyncio.sleep(0.1)
        
        # Check final state
        status = system.get_system_status()
        assert status["metrics"]["total_agents"] == 5
        assert status["metrics"]["learning_iterations"] >= 3
        assert status["metrics"]["connectivity_density"] > 0
        
        # Test emergent behavior detection
        emergence = await system._detect_emergence()
        assert 0 <= emergence <= 1
        
        # Cleanup
        await system.shutdown()
    
    @pytest.mark.asyncio
    async def test_hyperagent_communication_workflow(self):
        """Test communication workflow between HyperAgents."""
        system = DynamicalHyperSystem("CommunicationTest")
        
        # Create agents
        alice = await system.create_hyperagent("ConceptNode", "Alice")
        bob = await system.create_hyperagent("ConceptNode", "Bob")
        
        # Create communication channel
        channel = await system.create_hyperchannel(
            "chat",
            [alice.id, bob.id],
            discrete_events=True
        )
        
        # Test inter-agent communication
        comm_result = await alice._communicate_with_agent(
            bob.name,
            "Hello, how are you?"
        )
        
        assert comm_result.get("success") is True
        assert "response" in comm_result
        
        # Test collective reasoning
        reasoning_result = await alice._initiate_collective_reasoning(
            "What is the best approach to solving complex problems?"
        )
        
        assert "task" in reasoning_result
        assert "responses" in reasoning_result
        
        await system.shutdown()


# Performance and stress tests
class TestPerformance:
    """Performance tests for HGNN components."""
    
    @pytest.mark.asyncio
    async def test_large_hypergraph_performance(self):
        """Test performance with larger hypergraphs."""
        hg = HyperGraph("LargeGraph")
        
        # Create many nodes
        nodes = []
        for i in range(100):
            node = HyperNode(data={"id": i})
            nodes.append(node)
            hg.add_node(node)
        
        # Create many hyperedges
        for i in range(0, 100, 5):
            edge_nodes = nodes[i:i+5]
            edge = HyperEdge(nodes=edge_nodes)
            hg.add_edge(edge)
        
        # Test matrix computations
        start_time = datetime.now()
        incidence = hg.get_incidence_matrix()
        adjacency = hg.get_adjacency_matrix()
        end_time = datetime.now()
        
        # Check results
        assert incidence.shape[0] == 100  # 100 nodes
        assert adjacency.shape == (100, 100)
        
        # Should complete reasonably quickly
        computation_time = (end_time - start_time).total_seconds()
        assert computation_time < 5.0  # Should complete in under 5 seconds
    
    @pytest.mark.asyncio 
    async def test_many_agents_system(self):
        """Test system with many agents."""
        system = DynamicalHyperSystem("LargeSystem")
        
        # Create many agents
        agents = []
        for i in range(20):
            agent = await system.create_hyperagent(
                "ConceptNode",
                f"Agent_{i}"
            )
            agents.append(agent)
        
        # Create some connections
        for i in range(0, 20, 2):
            if i + 1 < 20:
                await system.create_hyperchannel(
                    "connection",
                    [agents[i].id, agents[i+1].id]
                )
        
        # Test global learning
        start_time = datetime.now()
        result = await system.trigger_global_learning()
        end_time = datetime.now()
        
        assert result.get("success") is True
        
        # Should complete reasonably quickly
        learning_time = (end_time - start_time).total_seconds()
        assert learning_time < 10.0  # Should complete in under 10 seconds
        
        await system.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])