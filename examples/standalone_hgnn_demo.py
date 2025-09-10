#!/usr/bin/env python3
"""
Standalone HGNN Demo - Core HyperGraph Neural Network functionality.

This example demonstrates the core HGNN capabilities without dependencies
on the full AtomBot architecture, showing:

1. HyperGraph creation and manipulation
2. HyperGraph Neural Network processing
3. Discrete-event processing through HyperChannels
4. Basic dynamical system behavior

Run this to see the HGNN components in action!
"""

import sys
import os
import asyncio
import numpy as np
from datetime import datetime, timedelta

# Add the current directory to path for imports
sys.path.insert(0, '.')

from hypergraph.hypergraph import HyperGraph, HyperNode, HyperEdge
from hypergraph.hgnn import HyperGraphConvolution, HyperAttention, HyperGraphNeuralNetwork
from hypergraph.hyperchannel import HyperChannel, DiscreteEventChannel, EventType


def demonstrate_hypergraph_basics():
    """Demonstrate basic hypergraph operations."""
    print("🔗 HYPERGRAPH BASICS")
    print("-" * 50)
    
    # Create a hypergraph
    hg = HyperGraph("DemoHyperGraph")
    print(f"Created hypergraph: {hg}")
    
    # Create nodes representing different concepts
    concepts = ["AI", "Machine Learning", "Deep Learning", "Neural Networks", "Optimization"]
    nodes = []
    
    for i, concept in enumerate(concepts):
        node = HyperNode(data={"name": concept, "type": "concept"})
        node.features = np.random.randn(64)  # Random embeddings
        nodes.append(node)
        hg.add_node(node)
        print(f"  Added node: {concept}")
    
    # Create hyperedges representing relationships
    # Traditional graphs connect pairs, hypergraphs can connect any number of nodes
    
    # ML concepts hyperedge (connects multiple related concepts)
    ml_edge = HyperEdge(
        nodes=[nodes[1], nodes[2], nodes[3]],  # ML, DL, NN
        edge_type="field_relationship"
    )
    hg.add_edge(ml_edge)
    print(f"  Added hyperedge connecting ML concepts: {ml_edge.get_cardinality()} nodes")
    
    # AI foundation hyperedge (connects broader concepts)
    ai_edge = HyperEdge(
        nodes=[nodes[0], nodes[1], nodes[4]],  # AI, ML, Optimization
        edge_type="foundational_relationship"
    )
    hg.add_edge(ai_edge)
    print(f"  Added hyperedge connecting AI foundations: {ai_edge.get_cardinality()} nodes")
    
    # Technical detail hyperedge (deep learning specifics)
    tech_edge = HyperEdge(
        nodes=[nodes[2], nodes[3], nodes[4]],  # DL, NN, Optimization
        edge_type="technical_relationship"
    )
    hg.add_edge(tech_edge)
    print(f"  Added hyperedge connecting technical concepts: {tech_edge.get_cardinality()} nodes")
    
    # Display hypergraph statistics
    stats = hg.get_statistics()
    print(f"\n📊 Hypergraph Statistics:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Hyperedges: {stats['num_edges']}")
    print(f"  Average node degree: {stats['avg_node_degree']:.2f}")
    print(f"  Average hyperedge cardinality: {stats['avg_edge_cardinality']:.2f}")
    print(f"  Connected components: {stats['connected_components']}")
    
    # Show incidence matrix
    incidence = hg.get_incidence_matrix()
    print(f"\n🔢 Incidence Matrix Shape: {incidence.shape}")
    print("   (nodes × hyperedges)")
    
    return hg, nodes


def demonstrate_neural_network(hg):
    """Demonstrate hypergraph neural network processing."""
    print("\n🧠 HYPERGRAPH NEURAL NETWORK")
    print("-" * 50)
    
    # Create different types of neural network layers
    print("Creating HGNN layers...")
    
    # Convolution layer
    conv_layer = HyperGraphConvolution(
        input_dim=64,
        output_dim=32,
        aggregation="mean",
        use_edge_features=False  # Disable edge features for this demo
    )
    print(f"  ✅ Created convolution layer: {conv_layer.input_dim} → {conv_layer.output_dim}")
    
    # Attention layer
    attention_layer = HyperAttention(
        input_dim=64,
        output_dim=32,
        num_heads=4
    )
    print(f"  ✅ Created attention layer: {attention_layer.num_heads} heads")
    
    # Complete neural network - use simpler architecture
    network = HyperGraphNeuralNetwork(
        input_dim=64,
        hidden_dims=[32],  # Simpler single hidden layer
        output_dim=16
    )
    print(f"  ✅ Created complete network: {len(network.layers)} layers")
    
    # Test forward pass
    print("\nRunning neural network forward pass...")
    
    # Get initial node features
    initial_features = hg.get_node_features()
    print(f"  Input features shape: {initial_features.shape}")
    
    # Convolution layer forward pass
    conv_output = conv_layer.forward(hg)
    print(f"  Convolution output shape: {conv_output.shape}")
    
    # Reset features for attention test
    for i, node in enumerate(hg.get_nodes()):
        node.features = initial_features[i]
    
    # Attention layer forward pass
    attention_output = attention_layer.forward(hg)
    print(f"  Attention output shape: {attention_output.shape}")
    
    # Full network forward pass
    # Reset features again
    for i, node in enumerate(hg.get_nodes()):
        node.features = initial_features[i]
    
    final_output = network.forward(hg)
    print(f"  Final network output shape: {final_output.shape}")
    
    # Don't call get_node_embeddings since it would re-process already processed features
    # Instead, use the final_output as embeddings
    nodes = list(hg.get_nodes())
    embeddings = {nodes[i].id: final_output[i] for i in range(len(nodes))}
    print(f"  Generated embeddings for {len(embeddings)} nodes")
    
    # Show embedding norms (measure of activation)
    print("  Node embedding norms:")
    for i, (node_id, embedding) in enumerate(embeddings.items()):
        norm = np.linalg.norm(embedding)
        node_name = hg.get_node(node_id).data.get("name", f"Node_{i}")
        print(f"    {node_name}: {norm:.3f}")
    
    return network, embeddings


async def demonstrate_hyperchannel(nodes):
    """Demonstrate discrete-event processing through HyperChannels."""
    print("\n⚡ HYPERCHANNEL DISCRETE EVENTS")
    print("-" * 50)
    
    # Create a HyperChannel connecting multiple nodes
    channel = DiscreteEventChannel(
        nodes=nodes[:3],  # Connect first 3 nodes
        data={"purpose": "knowledge_sharing"}
    )
    print(f"Created channel connecting {len(channel.get_nodes())} nodes")
    
    # Send different types of events
    print("\nSending discrete events...")
    
    events = []
    
    # Message event
    msg_event_id = await channel.send_event(
        event_type=EventType.MESSAGE,
        source_node=nodes[0].id,
        target_nodes=[nodes[1].id, nodes[2].id],
        data={
            "content": "New research paper on hypergraph attention!",
            "priority": "high",
            "timestamp": datetime.now().isoformat()
        }
    )
    events.append(msg_event_id)
    print(f"  📨 Sent message event: {msg_event_id[:8]}")
    
    # Signal event (propagates through network)
    signal_event_id = await channel.send_event(
        event_type=EventType.SIGNAL,
        source_node=nodes[1].id,
        data={
            "signal_type": "discovery",
            "strength": 0.9,
            "decay": 0.8,
            "content": "Breakthrough in optimization found!"
        }
    )
    events.append(signal_event_id)
    print(f"  📡 Sent signal event: {signal_event_id[:8]}")
    
    # Learning event
    learning_event_id = await channel.send_event(
        event_type=EventType.LEARNING,
        source_node=nodes[2].id,
        data={
            "learning_type": "pattern_recognition",
            "pattern": "hypergraph_structure_advantage",
            "confidence": 0.85
        }
    )
    events.append(learning_event_id)
    print(f"  🎓 Sent learning event: {learning_event_id[:8]}")
    
    # Update event
    update_event_id = await channel.send_event(
        event_type=EventType.UPDATE,
        source_node=nodes[0].id,
        data={
            "update_type": "feature_change",
            "old_features": nodes[0].features[:3].tolist(),  # First 3 features
            "new_features": (nodes[0].features[:3] * 1.1).tolist()  # Updated
        }
    )
    events.append(update_event_id)
    print(f"  🔄 Sent update event: {update_event_id[:8]}")
    
    # Wait for events to be processed
    print("\nWaiting for event processing...")
    await asyncio.sleep(0.2)
    
    # Check channel statistics
    stats = channel.get_statistics()
    print(f"\n📊 Channel Statistics:")
    print(f"  Events queued: {stats['stats']['events_queued']}")
    print(f"  Events processed: {stats['stats']['events_processed']}")
    print(f"  Events dropped: {stats['stats']['events_dropped']}")
    print(f"  Queue size: {stats['queue_status']['queue_size']}")
    print(f"  Processing mode: {stats['queue_status']['processing_mode']}")
    
    # Get performance metrics if available
    if hasattr(channel, 'get_performance_metrics'):
        perf_metrics = channel.get_performance_metrics()
        print(f"\n⚡ Performance Metrics:")
        print(f"  Simulation time: {perf_metrics.get('simulation_time', 0):.3f}")
        if 'latency' in perf_metrics:
            lat = perf_metrics['latency']
            print(f"  Average latency: {lat.get('avg', 0):.3f}s")
        if 'throughput' in perf_metrics:
            tput = perf_metrics['throughput']
            print(f"  Throughput: {tput.get('events_per_second', 0):.1f} events/sec")
    
    return channel, events


def demonstrate_graph_analysis(hg):
    """Demonstrate hypergraph analysis capabilities."""
    print("\n📈 HYPERGRAPH ANALYSIS")
    print("-" * 50)
    
    nodes = list(hg.get_nodes())
    
    # Analyze node connectivity
    print("Node connectivity analysis:")
    for node in nodes:
        name = node.data.get("name", node.id[:8])
        degree = node.get_degree()
        neighbors = len(node.get_neighbors())
        hyperedges = len(node.get_hyperedges())
        
        print(f"  {name}:")
        print(f"    Degree: {degree} | Neighbors: {neighbors} | HyperEdges: {hyperedges}")
    
    # Analyze hyperedge patterns
    print("\nHyperedge analysis:")
    edges = hg.get_edges()
    edge_types = {}
    
    for edge in edges:
        edge_type = edge.edge_type
        cardinality = edge.get_cardinality()
        
        if edge_type not in edge_types:
            edge_types[edge_type] = []
        edge_types[edge_type].append(cardinality)
        
        print(f"  {edge_type}: {cardinality} nodes")
    
    # Edge type statistics
    print("\nEdge type patterns:")
    for edge_type, cardinalities in edge_types.items():
        avg_cardinality = np.mean(cardinalities)
        print(f"  {edge_type}: avg cardinality {avg_cardinality:.1f}")
    
    # Find connected components
    components = hg.find_connected_components()
    print(f"\nConnected components: {len(components)}")
    for i, component in enumerate(components):
        comp_nodes = [node.data.get("name", node.id[:8]) for node in component]
        print(f"  Component {i+1}: {comp_nodes}")
    
    # Adjacency matrix analysis
    adjacency = hg.get_adjacency_matrix()
    print(f"\nAdjacency matrix analysis:")
    print(f"  Shape: {adjacency.shape}")
    print(f"  Density: {np.count_nonzero(adjacency) / adjacency.size:.3f}")
    print(f"  Max connections: {np.max(adjacency)}")
    
    return {"components": components, "edge_types": edge_types}


def demonstrate_learning_simulation(hg, network):
    """Simulate learning and adaptation over multiple iterations."""
    print("\n🔄 LEARNING SIMULATION")
    print("-" * 50)
    
    print("Simulating multiple learning iterations...")
    
    # Store initial embeddings
    initial_embeddings = {}
    for node in hg.get_nodes():
        initial_embeddings[node.id] = node.features.copy()
    
    # Run multiple learning iterations
    iterations = 5
    embedding_changes = []
    
    for iteration in range(iterations):
        print(f"\n  Iteration {iteration + 1}:")
        
        # Reset node features to original dimensions for consistent processing
        for node in hg.get_nodes():
            node.features = np.random.randn(64) * 0.1  # Reset to 64 dims
        
        # Run neural network forward pass
        output = network.forward(hg)
        
        # Calculate embedding changes
        total_change = 0
        for node in hg.get_nodes():
            old_embedding = initial_embeddings[node.id]
            new_embedding = node.features
            change = np.linalg.norm(new_embedding - old_embedding)
            total_change += change
        
        avg_change = total_change / len(hg.get_nodes())
        embedding_changes.append(avg_change)
        
        print(f"    Average embedding change: {avg_change:.4f}")
        print(f"    Output norm: {np.linalg.norm(output):.4f}")
        
        # Add some noise for continued learning
        for node in hg.get_nodes():
            noise = np.random.randn(len(node.features)) * 0.01
            node.features += noise
    
    print(f"\n📊 Learning progression:")
    for i, change in enumerate(embedding_changes):
        print(f"  Iteration {i+1}: {change:.4f} change")
    
    # Calculate convergence
    if len(embedding_changes) > 1:
        convergence_rate = (embedding_changes[-1] - embedding_changes[0]) / iterations
        print(f"  Convergence rate: {convergence_rate:.6f} per iteration")
    
    return embedding_changes


async def main():
    """Main demonstration function."""
    print("=" * 80)
    print("🌟 STANDALONE HGNN (HYPERGRAPH NEURAL NETWORK) DEMONSTRATION")
    print("   Core Components • Neural Processing • Discrete Events • Analysis")
    print("=" * 80)
    
    try:
        # 1. Basic hypergraph operations
        hg, nodes = demonstrate_hypergraph_basics()
        
        # 2. Neural network processing
        network, embeddings = demonstrate_neural_network(hg)
        
        # 3. Discrete-event processing
        channel, events = await demonstrate_hyperchannel(nodes)
        
        # 4. Graph analysis
        analysis = demonstrate_graph_analysis(hg)
        
        # 5. Learning simulation
        learning_progress = demonstrate_learning_simulation(hg, network)
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 DEMONSTRATION SUMMARY")
        print("=" * 80)
        
        print(f"🔗 HyperGraph: {len(hg.get_nodes())} nodes, {len(hg.get_edges())} hyperedges")
        print(f"🧠 Neural Network: {len(network.layers)} layers, {network.output_dim}D embeddings")
        print(f"⚡ Events: {len(events)} events processed through {channel.channel_type} channel")
        print(f"📈 Analysis: {len(analysis['components'])} connected components")
        print(f"🔄 Learning: {len(learning_progress)} iterations simulated")
        
        print("\n✅ CAPABILITIES DEMONSTRATED:")
        print("   🔗 Hypergraph representation with multi-node relationships")
        print("   🧠 Neural message passing and attention over hyperedges")
        print("   ⚡ Discrete-event processing and temporal dynamics")
        print("   📊 Graph analysis and structural insights")
        print("   🔄 Learning simulation and adaptation")
        
        print("\n🎯 KEY ADVANTAGES OF HGNN:")
        print("   • Higher-order relationships beyond pairwise connections")
        print("   • Attention mechanisms for complex pattern recognition")
        print("   • Event-driven processing for temporal dynamics")
        print("   • Distributed learning and emergent behaviors")
        print("   • Scalable neural computation over graph structures")
        
        print("\n🚀 This demonstrates the foundation for:")
        print("   • Distributed HyperAgents as intelligent nodes")
        print("   • Real-time discrete-event communication")
        print("   • Collective intelligence and emergence")
        print("   • Adaptive hypergraph neural systems")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("🎉 STANDALONE HGNN DEMONSTRATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())