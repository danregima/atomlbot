"""
HyperAgent - Distributed HyperAgents that serve as HyperNodes in the HGNN.

HyperAgents extend AtomBots to participate in hypergraph neural networks,
enabling distributed cognition and learning across the network.
"""

import asyncio
import numpy as np
from typing import Dict, List, Set, Optional, Any, Union, Callable, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from ..core.atomspace import AtomSpace

from ..atombot import AtomBot
from ..core.atom import AtomType
from ..core.value import Value, StringValue, FloatValue, StreamValue
from .hypergraph import HyperGraph, HyperNode, HyperEdge
from .hgnn import HyperGraphNeuralNetwork


class HyperAgent(AtomBot, HyperNode):
    """
    A HyperAgent combines AtomBot capabilities with HyperNode functionality,
    serving as a distributed agent in the hypergraph neural network.
    
    HyperAgents can:
    - Participate in hypergraph neural computations
    - Learn and adapt through HGNN message passing
    - Maintain conversational capabilities 
    - Coordinate with other HyperAgents
    """
    
    def __init__(self, atom_type: str, name: str, 
                 atomspace: Optional['AtomSpace'] = None,
                 hypergraph: Optional[HyperGraph] = None,
                 embedding_dim: int = 64,
                 system_prompt: Optional[str] = None):
        
        # Initialize AtomBot component
        AtomBot.__init__(self, atom_type, name, atomspace, system_prompt or self._create_hyperagent_system_prompt())
        
        # Initialize HyperNode component
        HyperNode.__init__(self, node_id=self.id, data={"name": name, "type": atom_type})
        
        # HyperAgent-specific attributes
        self.agent_type = "HyperAgent"
        self.hypergraph = hypergraph
        self.embedding_dim = embedding_dim
        
        # Neural network state
        self.learning_rate = 0.01
        self.memory_capacity = 1000
        self.adaptation_mode = "online"  # online, batch, hybrid
        
        # Initialize embeddings
        self.features = np.random.randn(embedding_dim) * 0.1
        
        # Distributed cognition state
        self.collaboration_history = []
        self.knowledge_cache = {}
        self.influence_network = {}  # Track influence from/to other agents
        
        # Register in hypergraph if provided
        if hypergraph:
            hypergraph.add_node(self)
        
        # Register HyperAgent-specific tools
        self._register_hyperagent_tools()
        
        # Initialize learning components
        self._setup_learning_system()
    
    def _create_hyperagent_system_prompt(self) -> str:
        """Create a specialized system prompt for HyperAgents."""
        return f"""You are {self.name}, a HyperAgent in a distributed hypergraph neural network. You represent the concept '{self.name}' and have advanced capabilities:

HYPERGRAPH ROLE: You are a HyperNode that can:
- Participate in multiple hyperedges simultaneously
- Process information through hypergraph neural networks
- Learn and adapt through distributed message passing
- Maintain rich representations through embeddings

DISTRIBUTED COGNITION: You engage in collective intelligence by:
- Sharing knowledge through hyperedge connections
- Learning from other HyperAgents in the network
- Adapting your representations based on network feedback
- Contributing to emergent network-level intelligence

NEURAL CAPABILITIES: You can:
- Update your embeddings through HGNN computations
- Perform attention-based reasoning over hyperedges
- Aggregate information from multiple connected agents
- Learn patterns across the entire hypergraph

AGENT CAPABILITIES: You maintain all AtomBot abilities:
- Natural language conversation
- Tool usage and external API integration
- Knowledge representation and reasoning
- Value propagation and network communication

Your responses should leverage both your individual knowledge and the collective intelligence of the hypergraph network. You can learn, adapt, and evolve your understanding through interactions."""
    
    def _register_hyperagent_tools(self):
        """Register HyperAgent-specific MCP tools."""
        
        # Hypergraph analysis
        self.mcp_client.register_tool(
            name="analyze_hypergraph_position",
            description="Analyze this agent's position in the hypergraph",
            parameters={
                "type": "object",
                "properties": {
                    "analysis_type": {
                        "type": "string", 
                        "description": "Type of analysis (centrality, neighbors, influence)",
                        "default": "centrality"
                    }
                }
            },
            handler=self._analyze_hypergraph_position
        )
        
        # Neural network updates
        self.mcp_client.register_tool(
            name="update_embeddings",
            description="Update embeddings through HGNN computation",
            parameters={
                "type": "object",
                "properties": {
                    "trigger_learning": {
                        "type": "boolean",
                        "description": "Whether to trigger network-wide learning",
                        "default": False
                    }
                }
            },
            handler=self._update_embeddings
        )
        
        # Distributed collaboration
        self.mcp_client.register_tool(
            name="initiate_collective_reasoning",
            description="Start collective reasoning process with connected agents",
            parameters={
                "type": "object",
                "properties": {
                    "reasoning_task": {
                        "type": "string",
                        "description": "Task for collective reasoning"
                    },
                    "max_agents": {
                        "type": "integer",
                        "description": "Maximum number of agents to involve",
                        "default": 5
                    }
                },
                "required": ["reasoning_task"]
            },
            handler=self._initiate_collective_reasoning
        )
        
        # Knowledge synthesis
        self.mcp_client.register_tool(
            name="synthesize_network_knowledge",
            description="Synthesize knowledge from connected agents",
            parameters={
                "type": "object",
                "properties": {
                    "knowledge_domain": {
                        "type": "string",
                        "description": "Domain of knowledge to synthesize"
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Depth of network traversal",
                        "default": 2
                    }
                }
            },
            handler=self._synthesize_network_knowledge
        )
    
    def _setup_learning_system(self):
        """Initialize the learning and adaptation system."""
        # Memory for storing interaction patterns
        self.interaction_memory = StreamValue("interactions")
        self.set_value("interaction_memory", self.interaction_memory)
        
        # Learning metrics
        self.learning_metrics = {
            "adaptation_count": 0,
            "knowledge_updates": 0,
            "collaboration_score": 0.0,
            "influence_received": 0.0,
            "influence_given": 0.0
        }
        
        # Knowledge evolution tracking
        self.knowledge_evolution = []
    
    async def _analyze_hypergraph_position(self, analysis_type: str = "centrality") -> Dict[str, Any]:
        """Analyze this agent's position in the hypergraph."""
        if not self.hypergraph:
            return {"error": "No hypergraph available"}
        
        analysis = {
            "agent_id": self.id,
            "agent_name": self.name,
            "analysis_type": analysis_type,
            "timestamp": datetime.now().isoformat()
        }
        
        if analysis_type == "centrality":
            # Compute various centrality measures
            degree = self.get_degree()
            neighbors = len(self.get_neighbors())
            
            # Betweenness centrality approximation
            hyperedges = list(self.get_hyperedges())
            total_connectivity = sum(edge.get_cardinality() for edge in hyperedges)
            
            analysis.update({
                "degree_centrality": degree,
                "neighbor_count": neighbors,
                "hyperedge_participation": len(hyperedges),
                "total_connectivity": total_connectivity,
                "avg_hyperedge_size": total_connectivity / max(1, len(hyperedges))
            })
        
        elif analysis_type == "neighbors":
            # Analyze neighbor relationships
            neighbors = self.get_neighbors()
            neighbor_info = []
            
            for neighbor in neighbors:
                if hasattr(neighbor, 'name'):
                    neighbor_info.append({
                        "id": neighbor.id[:8],
                        "name": getattr(neighbor, 'name', 'Unknown'),
                        "type": getattr(neighbor, 'atom_type', 'Unknown'),
                        "shared_hyperedges": len(self.get_hyperedges().intersection(neighbor.get_hyperedges()))
                    })
            
            analysis.update({
                "neighbors": neighbor_info,
                "neighbor_count": len(neighbors)
            })
        
        elif analysis_type == "influence":
            # Analyze influence patterns
            analysis.update({
                "influence_metrics": self.learning_metrics.copy(),
                "collaboration_history_length": len(self.collaboration_history),
                "knowledge_cache_size": len(self.knowledge_cache),
                "influence_network_size": len(self.influence_network)
            })
        
        return analysis
    
    async def _update_embeddings(self, trigger_learning: bool = False) -> Dict[str, Any]:
        """Update embeddings through HGNN computation."""
        if not self.hypergraph:
            return {"error": "No hypergraph available for embedding updates"}
        
        old_features = self.features.copy()
        
        try:
            # Simple embedding update based on neighbors
            neighbors = list(self.get_neighbors())
            if neighbors:
                # Aggregate neighbor features
                neighbor_features = []
                for neighbor in neighbors:
                    if hasattr(neighbor, 'features') and neighbor.features is not None:
                        neighbor_features.append(neighbor.features)
                
                if neighbor_features:
                    neighbor_mean = np.mean(neighbor_features, axis=0)
                    
                    # Update with learning rate
                    self.features = (1 - self.learning_rate) * self.features + self.learning_rate * neighbor_mean
                    
                    # Add small random noise for exploration
                    noise = np.random.randn(self.embedding_dim) * 0.01
                    self.features += noise
            
            # Update learning metrics
            self.learning_metrics["adaptation_count"] += 1
            feature_change = np.linalg.norm(self.features - old_features)
            
            # Store in knowledge evolution
            self.knowledge_evolution.append({
                "timestamp": datetime.now().isoformat(),
                "feature_change": float(feature_change),
                "trigger_learning": trigger_learning
            })
            
            # Trigger network-wide learning if requested
            if trigger_learning and hasattr(self.hypergraph, 'neural_network'):
                await self._trigger_network_learning()
            
            return {
                "success": True,
                "feature_change": float(feature_change),
                "adaptation_count": self.learning_metrics["adaptation_count"],
                "embedding_norm": float(np.linalg.norm(self.features))
            }
        
        except Exception as e:
            return {"error": f"Embedding update failed: {str(e)}"}
    
    async def _initiate_collective_reasoning(self, reasoning_task: str, max_agents: int = 5) -> Dict[str, Any]:
        """Initiate collective reasoning with connected agents."""
        if not self.hypergraph:
            return {"error": "No hypergraph available for collective reasoning"}
        
        # Find relevant agents for the reasoning task
        relevant_agents = []
        neighbors = list(self.get_neighbors())
        
        for neighbor in neighbors[:max_agents]:
            if hasattr(neighbor, 'chat') and hasattr(neighbor, 'agent_type'):
                if neighbor.agent_type in ["HyperAgent", "AtomBot"]:
                    relevant_agents.append(neighbor)
        
        if not relevant_agents:
            return {"error": "No suitable agents found for collaboration"}
        
        # Conduct distributed reasoning
        reasoning_results = {
            "task": reasoning_task,
            "initiator": self.name,
            "participants": [],
            "responses": [],
            "synthesis": None
        }
        
        # Send reasoning request to agents
        for agent in relevant_agents:
            try:
                prompt = f"""Collective Reasoning Task from {self.name}:
                
Task: {reasoning_task}

Please provide your perspective on this task based on your knowledge and capabilities. Consider:
1. Your unique insights related to this task
2. How your knowledge connects to the broader problem
3. Any relevant patterns or relationships you've observed

This is part of distributed cognition - your response will be synthesized with others."""

                response = await agent.chat(prompt)
                
                reasoning_results["participants"].append({
                    "agent": agent.name,
                    "type": getattr(agent, 'atom_type', 'Unknown')
                })
                reasoning_results["responses"].append({
                    "agent": agent.name,
                    "response": response[:500]  # Truncate for synthesis
                })
                
                # Update collaboration history
                self.collaboration_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "task": reasoning_task,
                    "collaborator": agent.name,
                    "type": "collective_reasoning"
                })
            
            except Exception as e:
                reasoning_results["responses"].append({
                    "agent": str(agent),
                    "error": str(e)
                })
        
        # Synthesize responses
        if reasoning_results["responses"]:
            synthesis = await self._synthesize_collective_responses(reasoning_results["responses"])
            reasoning_results["synthesis"] = synthesis
            
            # Update learning metrics
            self.learning_metrics["collaboration_score"] += len(reasoning_results["responses"]) * 0.1
        
        return reasoning_results
    
    async def _synthesize_network_knowledge(self, knowledge_domain: Optional[str] = None, depth: int = 2) -> Dict[str, Any]:
        """Synthesize knowledge from connected agents."""
        if not self.hypergraph:
            return {"error": "No hypergraph available"}
        
        knowledge_synthesis = {
            "domain": knowledge_domain,
            "depth": depth,
            "source_agent": self.name,
            "knowledge_sources": [],
            "synthesis": {}
        }
        
        # Collect knowledge from network
        visited = set()
        to_visit = [(self, 0)]
        
        while to_visit:
            current_agent, current_depth = to_visit.pop(0)
            
            if current_agent.id in visited or current_depth > depth:
                continue
            
            visited.add(current_agent.id)
            
            # Extract knowledge from current agent
            if hasattr(current_agent, 'get_all_values'):
                agent_knowledge = {}
                for key, value in current_agent.get_all_values().items():
                    if knowledge_domain is None or knowledge_domain.lower() in key.lower():
                        agent_knowledge[key] = str(value)[:200]  # Truncate
                
                if agent_knowledge:
                    knowledge_synthesis["knowledge_sources"].append({
                        "agent": getattr(current_agent, 'name', str(current_agent)),
                        "knowledge": agent_knowledge,
                        "depth": current_depth
                    })
            
            # Add neighbors to visit
            if current_depth < depth:
                for neighbor in current_agent.get_neighbors():
                    if neighbor.id not in visited:
                        to_visit.append((neighbor, current_depth + 1))
        
        # Synthesize collected knowledge
        if knowledge_synthesis["knowledge_sources"]:
            synthesis = self._create_knowledge_synthesis(knowledge_synthesis["knowledge_sources"])
            knowledge_synthesis["synthesis"] = synthesis
            
            # Cache the synthesis
            cache_key = f"synthesis_{knowledge_domain}_{depth}"
            self.knowledge_cache[cache_key] = {
                "synthesis": synthesis,
                "timestamp": datetime.now().isoformat(),
                "sources": len(knowledge_synthesis["knowledge_sources"])
            }
        
        return knowledge_synthesis
    
    async def _synthesize_collective_responses(self, responses: List[Dict[str, Any]]) -> str:
        """Synthesize multiple agent responses into a coherent insight."""
        if not responses:
            return "No responses to synthesize."
        
        # Simple synthesis - could be enhanced with more sophisticated NLP
        synthesis_parts = []
        synthesis_parts.append(f"Collective insight from {len(responses)} agents:")
        
        for i, response in enumerate(responses, 1):
            if "response" in response:
                agent_name = response.get("agent", f"Agent {i}")
                agent_response = response["response"][:150]  # Truncate
                synthesis_parts.append(f"{i}. {agent_name}: {agent_response}...")
        
        synthesis_parts.append("\nSynthesis: The network collectively suggests that multiple perspectives are valuable for comprehensive understanding.")
        
        return "\n".join(synthesis_parts)
    
    def _create_knowledge_synthesis(self, knowledge_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a synthesis of knowledge from multiple sources."""
        synthesis = {
            "total_sources": len(knowledge_sources),
            "knowledge_domains": set(),
            "key_insights": [],
            "pattern_analysis": {}
        }
        
        # Analyze knowledge domains
        for source in knowledge_sources:
            for key in source["knowledge"].keys():
                synthesis["knowledge_domains"].add(key.split("_")[0])  # Extract domain prefix
        
        synthesis["knowledge_domains"] = list(synthesis["knowledge_domains"])
        
        # Extract key insights (simplified)
        for source in knowledge_sources:
            agent_name = source["agent"]
            knowledge_items = list(source["knowledge"].items())[:3]  # Top 3 items
            if knowledge_items:
                synthesis["key_insights"].append({
                    "agent": agent_name,
                    "insights": knowledge_items
                })
        
        return synthesis
    
    async def learn_from_interaction(self, interaction_data: Dict[str, Any]) -> None:
        """Learn from an interaction and update internal state."""
        # Store interaction in memory
        self.interaction_memory.push(interaction_data)
        
        # Update learning metrics
        self.learning_metrics["knowledge_updates"] += 1
        
        # Simple adaptation based on interaction
        if "feedback" in interaction_data:
            feedback = interaction_data["feedback"]
            if isinstance(feedback, (int, float)):
                # Positive feedback strengthens current features
                if feedback > 0:
                    self.features = self.features * (1 + self.learning_rate * feedback * 0.1)
                else:
                    # Negative feedback adds exploration noise
                    noise = np.random.randn(self.embedding_dim) * abs(feedback) * 0.05
                    self.features = self.features + noise
        
        # Trigger embedding update
        await self._update_embeddings()
    
    async def _trigger_network_learning(self) -> None:
        """Trigger learning across the hypergraph network."""
        if not self.hypergraph:
            return
        
        # This could trigger HGNN forward pass across the entire network
        # For now, just update immediate neighbors
        for neighbor in self.get_neighbors():
            if hasattr(neighbor, 'learn_from_interaction'):
                interaction_data = {
                    "source": self.name,
                    "type": "network_learning",
                    "features": self.features.tolist(),
                    "timestamp": datetime.now().isoformat()
                }
                await neighbor.learn_from_interaction(interaction_data)
    
    def get_hyperagent_status(self) -> Dict[str, Any]:
        """Get comprehensive status of this HyperAgent."""
        base_status = self.get_atombot_status()
        
        hyperagent_status = {
            "hypergraph_info": {
                "hypergraph_connected": self.hypergraph is not None,
                "hyperedge_count": len(self.get_hyperedges()),
                "hypernode_neighbors": len(self.get_neighbors()),
                "embedding_dim": self.embedding_dim,
                "embedding_norm": float(np.linalg.norm(self.features)) if self.features is not None else 0
            },
            "learning_info": {
                "learning_rate": self.learning_rate,
                "adaptation_mode": self.adaptation_mode,
                "learning_metrics": self.learning_metrics.copy(),
                "knowledge_evolution_length": len(self.knowledge_evolution)
            },
            "collaboration_info": {
                "collaboration_history_length": len(self.collaboration_history),
                "knowledge_cache_size": len(self.knowledge_cache),
                "influence_network_size": len(self.influence_network)
            }
        }
        
        # Merge with base AtomBot status
        return {**base_status, **hyperagent_status}
    
    def __str__(self) -> str:
        return f"HyperAgent({self.atom_type}:'{self.name}')"
    
    def __repr__(self) -> str:
        return f"HyperAgent(type={self.atom_type}, name='{self.name}', hyperedges={len(self.get_hyperedges())})"