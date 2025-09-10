"""
Dynamical HyperSystem - Orchestrates the entire hypergraph as a dynamic system.

The DynamicalHyperSystem manages the global behavior of the hypergraph,
including learning, adaptation, emergence, and system-level coordination.
"""

import asyncio
import numpy as np
from typing import Dict, List, Set, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

from .hypergraph import HyperGraph, HyperNode, HyperEdge
from .hgnn import HyperGraphNeuralNetwork
from .hyperagent import HyperAgent
from .hyperchannel import HyperChannel, DiscreteEventChannel, EventType, DiscreteEvent


class SystemState(Enum):
    """States of the dynamical hypersystem."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    LEARNING = "learning"
    ADAPTING = "adapting"
    SYNCHRONIZING = "synchronizing"
    DORMANT = "dormant"
    ERROR = "error"


class SystemMode(Enum):
    """Operating modes of the hypersystem."""
    AUTONOMOUS = "autonomous"
    GUIDED = "guided"
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"


@dataclass
class SystemMetrics:
    """Metrics for monitoring system performance."""
    total_nodes: int = 0
    total_edges: int = 0
    total_agents: int = 0
    total_channels: int = 0
    learning_iterations: int = 0
    adaptation_events: int = 0
    emergence_level: float = 0.0
    connectivity_density: float = 0.0
    information_flow_rate: float = 0.0
    system_coherence: float = 0.0
    last_update: Optional[datetime] = None


class DynamicalHyperSystem:
    """
    The DynamicalHyperSystem orchestrates a hypergraph as a dynamic, 
    learning, and adaptive system.
    
    Key capabilities:
    - Global learning coordination
    - Emergent behavior detection
    - System-level adaptation
    - Information flow optimization
    - Distributed synchronization
    """
    
    def __init__(self, name: str = "DynamicalHyperSystem",
                 hypergraph: Optional[HyperGraph] = None,
                 neural_network: Optional[HyperGraphNeuralNetwork] = None):
        
        self.name = name
        self.system_id = f"dhs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.created_at = datetime.now()
        
        # Core components
        self.hypergraph = hypergraph or HyperGraph(f"{name}_hypergraph")
        self.neural_network = neural_network
        self.hyperagents: Dict[str, HyperAgent] = {}
        self.hyperchannels: Dict[str, HyperChannel] = {}
        
        # System state
        self.state = SystemState.INITIALIZING
        self.mode = SystemMode.AUTONOMOUS
        self.metrics = SystemMetrics()
        
        # Learning and adaptation
        self.global_learning_rate = 0.01
        self.adaptation_threshold = 0.1
        self.emergence_detection_enabled = True
        self.system_memory: Dict[str, Any] = {}
        
        # Coordination and synchronization
        self.coordination_channels: Dict[str, DiscreteEventChannel] = {}
        self.global_clock = 0.0
        self.sync_interval = 1.0
        
        # Event processing
        self.event_history: List[DiscreteEvent] = []
        self.event_patterns: Dict[str, int] = {}
        
        # Tasks and scheduling
        self.background_tasks: Set[asyncio.Task] = set()
        self.update_interval = timedelta(seconds=5)
        self.last_update = datetime.now()
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self) -> None:
        """Initialize the dynamical hypersystem."""
        # Create default coordination channel
        coord_channel = DiscreteEventChannel(
            channel_id="system_coordination",
            nodes=[],
            data={"type": "coordination", "global": True}
        )
        self.coordination_channels["main"] = coord_channel
        
        # Set up logging
        self.logger = logging.getLogger(f"DHS_{self.name}")
        
        # Initialize neural network if not provided
        if self.neural_network is None:
            self.neural_network = HyperGraphNeuralNetwork(
                input_dim=64,
                hidden_dims=[128, 64],
                output_dim=32
            )
        
        # Start background processes
        self._start_background_tasks()
        
        self.state = SystemState.ACTIVE
        self.logger.info(f"DynamicalHyperSystem '{self.name}' initialized")
    
    def _start_background_tasks(self) -> None:
        """Start background tasks for system maintenance."""
        tasks = [
            self._system_monitor_loop(),
            self._learning_coordination_loop(),
            self._adaptation_loop(),
            self._emergence_detection_loop()
        ]
        
        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
    
    async def _system_monitor_loop(self) -> None:
        """Monitor system health and performance."""
        try:
            while self.state != SystemState.ERROR:
                await self._update_system_metrics()
                await self._check_system_health()
                await asyncio.sleep(self.update_interval.total_seconds())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"System monitor error: {e}")
            self.state = SystemState.ERROR
    
    async def _learning_coordination_loop(self) -> None:
        """Coordinate learning across the hypergraph."""
        try:
            while self.state != SystemState.ERROR:
                if self.state == SystemState.LEARNING:
                    await self._coordinate_global_learning()
                await asyncio.sleep(2.0)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Learning coordination error: {e}")
    
    async def _adaptation_loop(self) -> None:
        """Monitor and trigger system adaptations."""
        try:
            while self.state != SystemState.ERROR:
                if await self._should_adapt():
                    await self._trigger_system_adaptation()
                await asyncio.sleep(3.0)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Adaptation loop error: {e}")
    
    async def _emergence_detection_loop(self) -> None:
        """Detect emergent behaviors in the system."""
        try:
            while self.state != SystemState.ERROR and self.emergence_detection_enabled:
                emergence_level = await self._detect_emergence()
                self.metrics.emergence_level = emergence_level
                
                if emergence_level > 0.8:  # High emergence detected
                    await self._handle_high_emergence(emergence_level)
                
                await asyncio.sleep(5.0)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Emergence detection error: {e}")
    
    async def add_hyperagent(self, hyperagent: HyperAgent) -> None:
        """Add a HyperAgent to the system."""
        # Add to hypergraph
        self.hypergraph.add_node(hyperagent)
        
        # Register in system
        self.hyperagents[hyperagent.id] = hyperagent
        
        # Set hypergraph reference
        hyperagent.hypergraph = self.hypergraph
        
        # Connect to coordination channel
        coord_channel = self.coordination_channels["main"]
        coord_channel.add_node(hyperagent)
        
        # Update metrics
        await self._update_system_metrics()
        
        self.logger.info(f"Added HyperAgent '{hyperagent.name}' to system")
    
    async def add_hyperchannel(self, hyperchannel: HyperChannel) -> None:
        """Add a HyperChannel to the system."""
        # Add to hypergraph
        self.hypergraph.add_edge(hyperchannel)
        
        # Register in system
        self.hyperchannels[hyperchannel.id] = hyperchannel
        
        # Update metrics
        await self._update_system_metrics()
        
        self.logger.info(f"Added HyperChannel '{hyperchannel.id}' to system")
    
    async def create_hyperagent(self, atom_type: str, name: str, 
                              embedding_dim: int = 64) -> HyperAgent:
        """Create and add a new HyperAgent to the system."""
        hyperagent = HyperAgent(
            atom_type=atom_type,
            name=name,
            hypergraph=self.hypergraph,
            embedding_dim=embedding_dim
        )
        
        await self.add_hyperagent(hyperagent)
        return hyperagent
    
    async def create_hyperchannel(self, channel_type: str, 
                                 connected_agents: List[str],
                                 discrete_events: bool = True) -> HyperChannel:
        """Create and add a new HyperChannel connecting specified agents."""
        # Find agent nodes
        nodes = []
        for agent_id in connected_agents:
            if agent_id in self.hyperagents:
                nodes.append(self.hyperagents[agent_id])
        
        # Create channel
        if discrete_events:
            channel = DiscreteEventChannel(
                nodes=nodes,
                data={"type": channel_type}
            )
        else:
            channel = HyperChannel(
                nodes=nodes,
                channel_type=channel_type,
                data={"type": channel_type}
            )
        
        await self.add_hyperchannel(channel)
        return channel
    
    async def trigger_global_learning(self) -> Dict[str, Any]:
        """Trigger global learning across the entire system."""
        self.state = SystemState.LEARNING
        
        try:
            # Run HGNN forward pass
            if self.neural_network and len(self.hypergraph.get_nodes()) > 0:
                embeddings = await self.neural_network.async_forward(self.hypergraph)
                
                # Update agent embeddings
                for i, node in enumerate(self.hypergraph.get_nodes()):
                    if isinstance(node, HyperAgent) and i < len(embeddings):
                        node.features = embeddings[i]
                        await node.learn_from_interaction({
                            "type": "global_learning",
                            "embedding_update": True,
                            "system_iteration": self.metrics.learning_iterations
                        })
            
            # Coordinate learning across agents
            learning_results = await self._coordinate_global_learning()
            
            self.metrics.learning_iterations += 1
            self.state = SystemState.ACTIVE
            
            return {
                "success": True,
                "learning_iteration": self.metrics.learning_iterations,
                "participants": len(self.hyperagents),
                "results": learning_results
            }
        
        except Exception as e:
            self.state = SystemState.ERROR
            return {"error": f"Global learning failed: {str(e)}"}
    
    async def _coordinate_global_learning(self) -> Dict[str, Any]:
        """Coordinate learning across all agents."""
        results = {
            "agents_updated": 0,
            "total_adaptations": 0,
            "convergence_metric": 0.0
        }
        
        # Trigger learning updates for all agents
        tasks = []
        for agent in self.hyperagents.values():
            task = agent._update_embeddings(trigger_learning=False)
            tasks.append(task)
        
        # Wait for all updates
        update_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in update_results:
            if isinstance(result, dict) and result.get("success"):
                results["agents_updated"] += 1
                results["total_adaptations"] += result.get("adaptation_count", 0)
        
        # Compute convergence metric
        if len(self.hyperagents) > 1:
            embeddings = [agent.features for agent in self.hyperagents.values() 
                         if agent.features is not None]
            if embeddings:
                mean_embedding = np.mean(embeddings, axis=0)
                variances = [np.linalg.norm(emb - mean_embedding) for emb in embeddings]
                results["convergence_metric"] = 1.0 / (1.0 + np.mean(variances))
        
        return results
    
    async def _should_adapt(self) -> bool:
        """Determine if the system should trigger adaptation."""
        # Check adaptation triggers
        triggers = [
            self.metrics.emergence_level > 0.7,
            self.metrics.system_coherence < 0.3,
            self.metrics.information_flow_rate < 0.1,
            datetime.now() - self.last_update > timedelta(minutes=10)
        ]
        
        return any(triggers)
    
    async def _trigger_system_adaptation(self) -> None:
        """Trigger system-level adaptation."""
        self.state = SystemState.ADAPTING
        
        try:
            # Analyze system state
            analysis = await self._analyze_system_state()
            
            # Adapt based on analysis
            if analysis["connectivity_low"]:
                await self._increase_connectivity()
            
            if analysis["information_stagnant"]:
                await self._stimulate_information_flow()
            
            if analysis["learning_plateau"]:
                await self._adjust_learning_parameters()
            
            self.metrics.adaptation_events += 1
            self.state = SystemState.ACTIVE
            
            self.logger.info("System adaptation completed")
        
        except Exception as e:
            self.logger.error(f"System adaptation failed: {e}")
            self.state = SystemState.ACTIVE
    
    async def _analyze_system_state(self) -> Dict[str, bool]:
        """Analyze current system state for adaptation decisions."""
        analysis = {
            "connectivity_low": self.metrics.connectivity_density < 0.3,
            "information_stagnant": self.metrics.information_flow_rate < 0.1,
            "learning_plateau": self.metrics.learning_iterations > 100 and self.metrics.emergence_level < 0.2
        }
        return analysis
    
    async def _increase_connectivity(self) -> None:
        """Increase connectivity between agents."""
        agents = list(self.hyperagents.values())
        
        # Find pairs of agents with low connectivity
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                
                # Check if they share any hyperedges
                shared_edges = agent1.get_hyperedges().intersection(agent2.get_hyperedges())
                
                if len(shared_edges) == 0:
                    # Create new connection
                    await self.create_hyperchannel(
                        "adaptive_connection",
                        [agent1.id, agent2.id]
                    )
                    break  # Limit to one new connection per adaptation
    
    async def _stimulate_information_flow(self) -> None:
        """Stimulate information flow through the system."""
        # Send stimulation signals through coordination channels
        for channel in self.coordination_channels.values():
            await channel.send_event(
                event_type=EventType.SIGNAL,
                source_node="system",
                data={
                    "type": "stimulation",
                    "strength": 1.0,
                    "purpose": "increase_information_flow"
                }
            )
    
    async def _adjust_learning_parameters(self) -> None:
        """Adjust global learning parameters."""
        # Increase learning rate if system is stagnant
        self.global_learning_rate = min(0.1, self.global_learning_rate * 1.1)
        
        # Update neural network parameters
        if self.neural_network:
            self.neural_network.set_dropout_rate(max(0.05, self.neural_network.layers[0].dropout_rate * 0.9))
        
        # Update agent learning rates
        for agent in self.hyperagents.values():
            agent.learning_rate = self.global_learning_rate
    
    async def _detect_emergence(self) -> float:
        """Detect emergent behaviors in the system."""
        if len(self.hyperagents) < 2:
            return 0.0
        
        try:
            # Measure various emergence indicators
            indicators = []
            
            # 1. Synchronization measure
            sync_measure = await self._measure_synchronization()
            indicators.append(sync_measure)
            
            # 2. Information integration
            integration_measure = await self._measure_information_integration()
            indicators.append(integration_measure)
            
            # 3. Collective behavior
            collective_measure = await self._measure_collective_behavior()
            indicators.append(collective_measure)
            
            # 4. System coherence
            coherence_measure = await self._measure_system_coherence()
            indicators.append(coherence_measure)
            
            # Combine indicators
            emergence_level = np.mean(indicators) if indicators else 0.0
            return float(np.clip(emergence_level, 0.0, 1.0))
        
        except Exception as e:
            self.logger.error(f"Emergence detection error: {e}")
            return 0.0
    
    async def _measure_synchronization(self) -> float:
        """Measure synchronization level among agents."""
        if len(self.hyperagents) < 2:
            return 0.0
        
        # Simple synchronization measure based on feature similarity
        embeddings = [agent.features for agent in self.hyperagents.values() 
                     if agent.features is not None]
        
        if len(embeddings) < 2:
            return 0.0
        
        # Compute pairwise correlations
        correlations = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                corr = np.corrcoef(embeddings[i], embeddings[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    async def _measure_information_integration(self) -> float:
        """Measure information integration across the system."""
        # Based on mutual information between connected agents
        # Simplified measure using embedding distances
        
        total_integration = 0.0
        edge_count = 0
        
        for edge in self.hypergraph.get_edges():
            nodes = list(edge.get_nodes())
            if len(nodes) >= 2:
                # Measure integration within this hyperedge
                embeddings = [node.features for node in nodes 
                             if hasattr(node, 'features') and node.features is not None]
                
                if len(embeddings) >= 2:
                    # Information integration as inverse of average distance
                    distances = []
                    for i in range(len(embeddings)):
                        for j in range(i + 1, len(embeddings)):
                            dist = np.linalg.norm(embeddings[i] - embeddings[j])
                            distances.append(dist)
                    
                    if distances:
                        avg_distance = np.mean(distances)
                        integration = 1.0 / (1.0 + avg_distance)
                        total_integration += integration
                        edge_count += 1
        
        return total_integration / max(1, edge_count)
    
    async def _measure_collective_behavior(self) -> float:
        """Measure collective behavior emergence."""
        # Based on coordination in recent interactions
        recent_interactions = 0
        coordinated_interactions = 0
        
        # Count recent collaborative interactions
        cutoff_time = datetime.now() - timedelta(minutes=5)
        
        for agent in self.hyperagents.values():
            for interaction in agent.collaboration_history[-10:]:  # Recent interactions
                interaction_time = datetime.fromisoformat(interaction["timestamp"])
                if interaction_time > cutoff_time:
                    recent_interactions += 1
                    if interaction["type"] == "collective_reasoning":
                        coordinated_interactions += 1
        
        if recent_interactions == 0:
            return 0.0
        
        return coordinated_interactions / recent_interactions
    
    async def _measure_system_coherence(self) -> float:
        """Measure overall system coherence."""
        # Based on consistency of agent states and behaviors
        if len(self.hyperagents) < 2:
            return 1.0
        
        # Measure coherence in learning metrics
        learning_scores = [agent.learning_metrics["collaboration_score"] 
                          for agent in self.hyperagents.values()]
        
        if not learning_scores:
            return 0.0
        
        # Coherence as inverse of variance
        mean_score = np.mean(learning_scores)
        variance = np.var(learning_scores)
        coherence = 1.0 / (1.0 + variance)
        
        self.metrics.system_coherence = coherence
        return coherence
    
    async def _handle_high_emergence(self, emergence_level: float) -> None:
        """Handle high emergence scenarios."""
        self.logger.info(f"High emergence detected: {emergence_level:.3f}")
        
        # Broadcast emergence alert
        for channel in self.coordination_channels.values():
            await channel.send_event(
                event_type=EventType.ALERT,
                source_node="system",
                data={
                    "type": "high_emergence",
                    "level": emergence_level,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Trigger global learning to consolidate emergent patterns
        await self.trigger_global_learning()
    
    async def _update_system_metrics(self) -> None:
        """Update system metrics."""
        self.metrics.total_nodes = len(self.hypergraph.get_nodes())
        self.metrics.total_edges = len(self.hypergraph.get_edges())
        self.metrics.total_agents = len(self.hyperagents)
        self.metrics.total_channels = len(self.hyperchannels)
        
        # Compute connectivity density
        if self.metrics.total_nodes > 1:
            max_connections = self.metrics.total_nodes * (self.metrics.total_nodes - 1) / 2
            actual_connections = sum(node.get_degree() for node in self.hypergraph.get_nodes()) / 2
            self.metrics.connectivity_density = actual_connections / max_connections
        
        # Compute information flow rate
        total_events = sum(len(channel.event_queue) for channel in self.hyperchannels.values()
                          if hasattr(channel, 'event_queue'))
        self.metrics.information_flow_rate = total_events / max(1, self.metrics.total_channels)
        
        self.metrics.last_update = datetime.now()
        self.last_update = self.metrics.last_update
    
    async def _check_system_health(self) -> None:
        """Check overall system health."""
        health_issues = []
        
        # Check for inactive agents
        inactive_agents = [agent for agent in self.hyperagents.values()
                          if agent.last_interaction and 
                          (datetime.now() - agent.last_interaction).total_seconds() > 300]
        
        if len(inactive_agents) > len(self.hyperagents) * 0.5:
            health_issues.append("Many agents inactive")
        
        # Check for disconnected components
        components = self.hypergraph.find_connected_components()
        if len(components) > 1:
            health_issues.append(f"System fragmented into {len(components)} components")
        
        # Check for error states
        error_agents = [agent for agent in self.hyperagents.values()
                       if hasattr(agent, 'state') and agent.state == 'error']
        
        if error_agents:
            health_issues.append(f"{len(error_agents)} agents in error state")
        
        if health_issues:
            self.logger.warning(f"System health issues: {health_issues}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "system_info": {
                "name": self.name,
                "id": self.system_id,
                "state": self.state.value,
                "mode": self.mode.value,
                "created_at": self.created_at.isoformat(),
                "uptime": (datetime.now() - self.created_at).total_seconds()
            },
            "metrics": {
                "total_nodes": self.metrics.total_nodes,
                "total_edges": self.metrics.total_edges,
                "total_agents": self.metrics.total_agents,
                "total_channels": self.metrics.total_channels,
                "learning_iterations": self.metrics.learning_iterations,
                "adaptation_events": self.metrics.adaptation_events,
                "emergence_level": self.metrics.emergence_level,
                "connectivity_density": self.metrics.connectivity_density,
                "information_flow_rate": self.metrics.information_flow_rate,
                "system_coherence": self.metrics.system_coherence
            },
            "hypergraph_stats": self.hypergraph.get_statistics(),
            "background_tasks": len(self.background_tasks),
            "coordination_channels": len(self.coordination_channels)
        }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the system."""
        self.logger.info("Shutting down DynamicalHyperSystem")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Stop channel processing
        for channel in self.hyperchannels.values():
            if hasattr(channel, '_stop_processing'):
                channel._stop_processing()
        
        self.state = SystemState.DORMANT
        self.logger.info("DynamicalHyperSystem shutdown complete")
    
    def __str__(self) -> str:
        return f"DynamicalHyperSystem('{self.name}', agents={len(self.hyperagents)}, state={self.state.value})"
    
    def __repr__(self) -> str:
        return f"DynamicalHyperSystem(name='{self.name}', agents={len(self.hyperagents)}, channels={len(self.hyperchannels)}, state={self.state.value})"