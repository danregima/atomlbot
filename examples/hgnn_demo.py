#!/usr/bin/env python3
"""
HGNN Example - Demonstrating HyperGraph Neural Network capabilities in AtomBot.

This example shows how to:
1. Create a DynamicalHyperSystem
2. Add HyperAgents as distributed nodes
3. Connect them with HyperChannels for communication
4. Run distributed learning and reasoning
5. Observe emergent behaviors
"""

import asyncio
import time
from datetime import datetime

from atombot.hypergraph import (
    DynamicalHyperSystem, HyperAgent, EventType
)


async def create_research_team_simulation():
    """Create a simulation of a research team using HGNN."""
    print("🚀 Creating DynamicalHyperSystem for research team simulation...")
    
    # Create the dynamical hypersystem
    system = DynamicalHyperSystem("ResearchTeamHGNN")
    
    # Create HyperAgents representing different roles
    roles_and_names = [
        ("ConceptNode", "Alice_ML_Researcher"),
        ("ConceptNode", "Bob_DataScientist"), 
        ("ConceptNode", "Carol_DomainExpert"),
        ("ConceptNode", "David_SystemArchitect"),
        ("ConceptNode", "Eve_ProductManager")
    ]
    
    agents = {}
    print("\n👥 Creating HyperAgents...")
    for atom_type, name in roles_and_names:
        agent = await system.create_hyperagent(
            atom_type=atom_type,
            name=name,
            embedding_dim=64
        )
        agents[name] = agent
        print(f"  ✅ Created {name}")
    
    # Create specialized knowledge for each agent
    print("\n🧠 Initializing specialized knowledge...")
    
    # Alice (ML Researcher) - knows about algorithms
    alice = agents["Alice_ML_Researcher"]
    alice.set_data("expertise", "machine_learning")
    alice.set_data("knowledge_areas", ["neural_networks", "optimization", "deep_learning"])
    
    # Bob (Data Scientist) - knows about data
    bob = agents["Bob_DataScientist"]
    bob.set_data("expertise", "data_science")
    bob.set_data("knowledge_areas", ["statistics", "data_preprocessing", "visualization"])
    
    # Carol (Domain Expert) - knows business domain
    carol = agents["Carol_DomainExpert"]
    carol.set_data("expertise", "domain_knowledge")
    carol.set_data("knowledge_areas", ["business_requirements", "user_needs", "market_analysis"])
    
    # David (System Architect) - knows infrastructure  
    david = agents["David_SystemArchitect"]
    david.set_data("expertise", "system_architecture")
    david.set_data("knowledge_areas", ["scalability", "performance", "infrastructure"])
    
    # Eve (Product Manager) - knows coordination
    eve = agents["Eve_ProductManager"]
    eve.set_data("expertise", "product_management")
    eve.set_data("knowledge_areas", ["strategy", "coordination", "stakeholder_management"])
    
    return system, agents


async def create_collaboration_channels(system, agents):
    """Create HyperChannels for different types of collaboration."""
    print("\n🔗 Creating collaboration HyperChannels...")
    
    # Technical discussion channel (Alice, Bob, David)
    tech_channel = await system.create_hyperchannel(
        channel_type="technical_discussion",
        connected_agents=[
            agents["Alice_ML_Researcher"].id,
            agents["Bob_DataScientist"].id, 
            agents["David_SystemArchitect"].id
        ],
        discrete_events=True
    )
    print("  ✅ Created technical discussion channel")
    
    # Strategy channel (Carol, Eve)
    strategy_channel = await system.create_hyperchannel(
        channel_type="strategy_planning",
        connected_agents=[
            agents["Carol_DomainExpert"].id,
            agents["Eve_ProductManager"].id
        ],
        discrete_events=True
    )
    print("  ✅ Created strategy planning channel")
    
    # Cross-functional channel (everyone)
    all_agent_ids = [agent.id for agent in agents.values()]
    cross_func_channel = await system.create_hyperchannel(
        channel_type="cross_functional",
        connected_agents=all_agent_ids,
        discrete_events=True
    )
    print("  ✅ Created cross-functional channel")
    
    return {
        "technical": tech_channel,
        "strategy": strategy_channel, 
        "cross_functional": cross_func_channel
    }


async def simulate_research_collaboration(system, agents, channels):
    """Simulate collaborative research activities."""
    print("\n🧪 Starting research collaboration simulation...")
    
    # Phase 1: Individual knowledge sharing
    print("\n📚 Phase 1: Knowledge sharing and learning...")
    
    alice = agents["Alice_ML_Researcher"]
    bob = agents["Bob_DataScientist"]
    carol = agents["Carol_DomainExpert"]
    
    # Alice shares ML knowledge
    ml_knowledge = await alice._synthesize_network_knowledge("machine_learning", depth=1)
    print(f"  Alice synthesized ML knowledge: {len(ml_knowledge.get('knowledge_sources', []))} sources")
    
    # Bob analyzes data patterns
    data_analysis = await bob._analyze_hypergraph_position("centrality")
    print(f"  Bob analyzed network position: {data_analysis.get('degree_centrality', 0)} centrality")
    
    # Phase 2: Collective reasoning
    print("\n🤝 Phase 2: Collective reasoning...")
    
    research_question = "How can we build an AI system that learns from user feedback in real-time?"
    
    collective_result = await alice._initiate_collective_reasoning(
        research_question,
        max_agents=3
    )
    print(f"  Collective reasoning initiated: {len(collective_result.get('responses', []))} participants")
    print(f"  Research question: {research_question}")
    
    # Phase 3: Global learning
    print("\n🧠 Phase 3: Global learning coordination...")
    
    learning_result = await system.trigger_global_learning()
    print(f"  Global learning completed: iteration {learning_result.get('learning_iteration', 0)}")
    print(f"  Participants: {learning_result.get('participants', 0)}")
    
    # Phase 4: Emergent behavior detection
    print("\n✨ Phase 4: Emergence detection...")
    
    emergence_level = await system._detect_emergence()
    print(f"  Current emergence level: {emergence_level:.3f}")
    
    if emergence_level > 0.5:
        print("  🎉 High emergence detected! The team is showing collective intelligence.")
    else:
        print("  📈 Building towards emergence...")
    
    return {
        "collective_reasoning": collective_result,
        "global_learning": learning_result,
        "emergence_level": emergence_level
    }


async def demonstrate_discrete_events(system, agents, channels):
    """Demonstrate discrete event processing through HyperChannels."""
    print("\n⚡ Demonstrating discrete event processing...")
    
    tech_channel = channels["technical"]
    alice = agents["Alice_ML_Researcher"] 
    bob = agents["Bob_DataScientist"]
    
    # Send various types of events
    events = []
    
    # Message event
    msg_event_id = await tech_channel.send_event(
        event_type=EventType.MESSAGE,
        source_node=alice.id,
        target_nodes=[bob.id],
        data={
            "content": "I found an interesting paper on hypergraph neural networks!",
            "paper_title": "HGNN: Hypergraph Neural Networks for Learning on Hypergraphs",
            "urgency": "medium"
        }
    )
    events.append(msg_event_id)
    print(f"  📨 Sent message event: {msg_event_id[:8]}")
    
    # Signal event for network-wide notification
    signal_event_id = await tech_channel.send_event(
        event_type=EventType.SIGNAL,
        source_node=bob.id,
        data={
            "signal_type": "discovery",
            "strength": 0.8,
            "content": "Found breakthrough in data preprocessing technique"
        }
    )
    events.append(signal_event_id)
    print(f"  📡 Sent signal event: {signal_event_id[:8]}")
    
    # Learning event
    learning_event_id = await tech_channel.send_event(
        event_type=EventType.LEARNING,
        source_node=alice.id,
        data={
            "learning_type": "pattern_recognition",
            "insight": "Hypergraph structures capture complex relationships better than regular graphs",
            "confidence": 0.9
        }
    )
    events.append(learning_event_id)
    print(f"  🎓 Sent learning event: {learning_event_id[:8]}")
    
    # Wait for events to be processed
    await asyncio.sleep(0.5)
    
    # Check channel statistics
    stats = tech_channel.get_statistics()
    print(f"  📊 Channel processed {stats['stats']['events_processed']} events")
    
    return events


async def monitor_system_evolution(system, duration_minutes=2):
    """Monitor the system's evolution over time."""
    print(f"\n📈 Monitoring system evolution for {duration_minutes} minutes...")
    
    start_time = time.time()
    monitoring_data = []
    
    while time.time() - start_time < duration_minutes * 60:
        # Get system status
        status = system.get_system_status()
        metrics = status["metrics"]
        
        # Record key metrics
        data_point = {
            "timestamp": datetime.now().isoformat(),
            "emergence_level": metrics["emergence_level"],
            "connectivity_density": metrics["connectivity_density"],
            "system_coherence": metrics["system_coherence"],
            "learning_iterations": metrics["learning_iterations"]
        }
        monitoring_data.append(data_point)
        
        print(f"  🔍 Emergence: {metrics['emergence_level']:.3f}, "
              f"Coherence: {metrics['system_coherence']:.3f}, "
              f"Learning: {metrics['learning_iterations']}")
        
        # Trigger periodic learning
        if len(monitoring_data) % 3 == 0:
            await system.trigger_global_learning()
        
        await asyncio.sleep(10)  # Check every 10 seconds
    
    print(f"  ✅ Collected {len(monitoring_data)} data points")
    return monitoring_data


async def demonstrate_adaptation(system):
    """Demonstrate system adaptation capabilities."""
    print("\n🔄 Demonstrating system adaptation...")
    
    # Check if system needs adaptation
    should_adapt = await system._should_adapt()
    print(f"  🔍 System adaptation needed: {should_adapt}")
    
    if should_adapt:
        print("  🔧 Triggering system adaptation...")
        await system._trigger_system_adaptation()
        print("  ✅ Adaptation completed")
    else:
        print("  ℹ️  System is currently well-adapted")
    
    # Show adaptation metrics
    status = system.get_system_status()
    adaptation_events = status["metrics"]["adaptation_events"]
    print(f"  📊 Total adaptation events: {adaptation_events}")


async def main():
    """Main demonstration function."""
    print("=" * 80)
    print("🌟 HGNN (HyperGraph Neural Network) Demonstration")
    print("   Distributed HyperAgents • Discrete-Event HyperChannels • Dynamical HyperSystem")
    print("=" * 80)
    
    try:
        # Create the research team simulation
        system, agents = await create_research_team_simulation()
        
        # Create collaboration channels
        channels = await create_collaboration_channels(system, agents)
        
        # Simulate research collaboration
        collaboration_results = await simulate_research_collaboration(system, agents, channels)
        
        # Demonstrate discrete event processing
        event_results = await demonstrate_discrete_events(system, agents, channels)
        
        # Monitor system evolution
        evolution_data = await monitor_system_evolution(system, duration_minutes=1)
        
        # Demonstrate adaptation
        await demonstrate_adaptation(system)
        
        # Final system summary
        print("\n" + "=" * 80)
        print("📊 FINAL SYSTEM SUMMARY")
        print("=" * 80)
        
        final_status = system.get_system_status()
        metrics = final_status["metrics"]
        
        print(f"🤖 HyperAgents: {metrics['total_agents']}")
        print(f"🔗 HyperChannels: {metrics['total_channels']}")
        print(f"🧠 Learning Iterations: {metrics['learning_iterations']}")
        print(f"🔄 Adaptation Events: {metrics['adaptation_events']}")
        print(f"✨ Emergence Level: {metrics['emergence_level']:.3f}")
        print(f"🌐 Connectivity Density: {metrics['connectivity_density']:.3f}")
        print(f"🎯 System Coherence: {metrics['system_coherence']:.3f}")
        
        hypergraph_stats = final_status["hypergraph_stats"]
        print(f"📈 Hypergraph Nodes: {hypergraph_stats['num_nodes']}")
        print(f"📈 Hypergraph Edges: {hypergraph_stats['num_edges']}")
        print(f"🔗 Connected Components: {hypergraph_stats['connected_components']}")
        
        print("\n🎉 HGNN demonstration completed successfully!")
        print("   The system demonstrated:")
        print("   ✅ Distributed cognition through HyperAgents")
        print("   ✅ Discrete-event processing through HyperChannels")  
        print("   ✅ Global learning and adaptation")
        print("   ✅ Emergent behavior detection")
        print("   ✅ Dynamic system evolution")
        
        # Graceful shutdown
        print("\n🔄 Shutting down system gracefully...")
        await system.shutdown()
        print("✅ System shutdown complete.")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())