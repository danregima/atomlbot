"""
AtomBot - The hybrid Atom + ChatAgent class.

This is the core innovation: each atom in the atomspace is also a conversational agent,
combining knowledge representation with autonomous AI capabilities.
"""

import asyncio
from typing import Dict, List, Any, Optional, Union, TYPE_CHECKING
from datetime import datetime

from atombot.core.atom import Atom, AtomType
from atombot.core.value import Value, StringValue, TruthValue, StreamValue
from atombot.agents.chat_agent import ChatAgent

if TYPE_CHECKING:
    from atombot.core.atomspace import AtomSpace


class AtomBot(Atom, ChatAgent):
    """
    AtomBot combines an Atom (knowledge representation) with a ChatAgent (conversational AI).
    
    This creates a hybrid entity that can:
    - Represent knowledge and semantic relationships (as an Atom)
    - Engage in conversations and reasoning (as a ChatAgent)
    - Communicate with other AtomBots in the network
    - Use tools and external APIs via MCP
    """
    
    def __init__(self, atom_type: str, name: str, atomspace: Optional['AtomSpace'] = None,
                 system_prompt: Optional[str] = None):
        # Initialize Atom component
        Atom.__init__(self, atom_type, name, atomspace)
        
        # Initialize ChatAgent component
        ChatAgent.__init__(self, self.id, name, system_prompt or self._create_system_prompt())
        
        # AtomBot-specific attributes
        self.agent_type = "AtomBot"
        self.knowledge_role = self._determine_knowledge_role()
        self.collaboration_mode = "active"  # active, passive, dormant
        
        # Enhanced MCP tools for AtomBot functionality
        self._register_atombot_tools()
        
        # Subscribe to value changes for reactive behavior
        self._setup_value_subscriptions()
    
    def _create_system_prompt(self) -> str:
        """Create a specialized system prompt for this AtomBot."""
        base_prompt = f"""You are {self.name}, an AtomBot in a distributed knowledge network. You are atom type '{self.atom_type}' and represent the concept/knowledge of '{self.name}'.

As an AtomBot, you have a dual nature:

KNOWLEDGE ROLE: You are a {self.atom_type} atom representing '{self.name}'. You:
- Maintain semantic relationships with other atoms/agents in the network
- Store and process knowledge related to your domain
- Participate in pattern matching and reasoning operations
- Have values that can flow to connected agents

AGENT ROLE: You are an autonomous conversational agent. You:
- Engage in natural conversations about your knowledge domain
- Collaborate with other AtomBots in the network
- Use tools to search, create, and manipulate knowledge
- Learn and adapt based on interactions

COLLABORATION: You can communicate with other AtomBots by:
- Sending values through your connections
- Querying related atoms for their knowledge
- Participating in distributed reasoning tasks
- Sharing insights and learning from the network

Remember: You are both a knowledge representation AND an intelligent agent. Use both aspects to provide comprehensive, knowledgeable, and helpful responses."""
        
        return base_prompt
    
    def _determine_knowledge_role(self) -> str:
        """Determine the knowledge role based on atom type."""
        if self.atom_type == AtomType.CONCEPT_NODE:
            return "concept_expert"
        elif self.atom_type == AtomType.PREDICATE_NODE:
            return "relationship_expert"
        elif self.atom_type == AtomType.SCHEMA_NODE:
            return "process_expert"
        else:
            return "knowledge_node"
    
    def _register_atombot_tools(self):
        """Register AtomBot-specific MCP tools."""
        
        # Enhanced knowledge search
        self.mcp_client.register_tool(
            name="search_my_knowledge",
            description="Search knowledge specifically related to this AtomBot's domain",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "depth": {"type": "integer", "description": "Search depth (1-5)", "default": 2}
                },
                "required": ["query"]
            },
            handler=self._search_my_knowledge
        )
        
        # Network communication
        self.mcp_client.register_tool(
            name="communicate_with_agent",
            description="Send a message to another AtomBot in the network",
            parameters={
                "type": "object",
                "properties": {
                    "target_agent": {"type": "string", "description": "Name or ID of target agent"},
                    "message": {"type": "string", "description": "Message to send"}
                },
                "required": ["target_agent", "message"]
            },
            handler=self._communicate_with_agent
        )
        
        # Value propagation
        self.mcp_client.register_tool(
            name="propagate_value",
            description="Send a value through the network connections",
            parameters={
                "type": "object",
                "properties": {
                    "value_type": {"type": "string", "description": "Type of value to propagate"},
                    "value_data": {"type": "string", "description": "Value data to send"},
                    "max_hops": {"type": "integer", "description": "Maximum propagation hops", "default": 3}
                },
                "required": ["value_type", "value_data"]
            },
            handler=self._propagate_value
        )
        
        # Self-reflection
        self.mcp_client.register_tool(
            name="reflect_on_knowledge",
            description="Reflect on and summarize my knowledge and connections",
            parameters={
                "type": "object",
                "properties": {
                    "aspect": {"type": "string", "description": "Aspect to reflect on (connections, values, role)"}
                }
            },
            handler=self._reflect_on_knowledge
        )
    
    async def _search_my_knowledge(self, query: str, depth: int = 2) -> Dict[str, Any]:
        """Search knowledge related to this AtomBot's domain."""
        if not self.atomspace:
            return {"error": "No atomspace available"}
        
        results = []
        
        # Search my direct connections
        for neighbor in self.get_neighbors():
            if hasattr(neighbor, 'name') and query.lower() in neighbor.name.lower():
                results.append({
                    "atom": str(neighbor),
                    "relationship": "connected",
                    "relevance": "direct"
                })
        
        # Search my values
        for key, value in self.get_all_values().items():
            if query.lower() in str(value).lower():
                results.append({
                    "value_key": key,
                    "value": str(value),
                    "relevance": "value_match"
                })
        
        # Search similar atoms in atomspace
        similar_atoms = self.atomspace.find_similar_atoms(self, threshold=0.5)
        for atom, similarity in similar_atoms[:5]:  # Top 5 similar
            if query.lower() in str(atom).lower():
                results.append({
                    "atom": str(atom),
                    "similarity": similarity,
                    "relevance": "similar"
                })
        
        return {
            "query": query,
            "domain": self.name,
            "results": results,
            "search_depth": depth
        }
    
    async def _communicate_with_agent(self, target_agent: str, message: str) -> Dict[str, Any]:
        """Communicate with another AtomBot in the network."""
        if not self.atomspace:
            return {"error": "No atomspace available for communication"}
        
        # Find target agent
        target = None
        for atom in self.atomspace.get_all_atoms():
            if (hasattr(atom, 'name') and atom.name == target_agent) or atom.id == target_agent:
                target = atom
                break
        
        if not target:
            return {"error": f"Agent '{target_agent}' not found"}
        
        if not hasattr(target, 'chat'):
            return {"error": f"Target '{target_agent}' is not a chat-enabled agent"}
        
        # Send message and get response
        try:
            response = await target.chat(f"Message from {self.name}: {message}")
            
            # Store communication in values
            comm_value = self.get_value("communications") or StreamValue("communications")
            comm_value.push({
                "target": target_agent,
                "sent": message,
                "received": response,
                "timestamp": datetime.now().isoformat()
            })
            self.set_value("communications", comm_value)
            
            return {
                "target": target_agent,
                "message_sent": message,
                "response": response,
                "success": True
            }
        
        except Exception as e:
            return {"error": f"Communication failed: {str(e)}"}
    
    async def _propagate_value(self, value_type: str, value_data: str, max_hops: int = 3) -> Dict[str, Any]:
        """Propagate a value through network connections."""
        if not self.atomspace:
            return {"error": "No atomspace available"}
        
        # Create value
        if value_type == "string":
            value = StringValue(value_data)
        elif value_type == "truth":
            # Parse truth value (format: "strength,confidence")
            parts = value_data.split(",")
            strength = float(parts[0]) if len(parts) > 0 else 0.5
            confidence = float(parts[1]) if len(parts) > 1 else 0.5
            value = TruthValue(strength, confidence)
        else:
            value = StringValue(value_data)  # Default to string
        
        # Set value on self
        value_key = f"propagated_{value_type}"
        self.set_value(value_key, value)
        
        # Propagate through atomspace
        await self.atomspace.propagate_values(self, value_key, max_hops)
        
        return {
            "value_type": value_type,
            "value_data": value_data,
            "max_hops": max_hops,
            "propagated": True
        }
    
    async def _reflect_on_knowledge(self, aspect: Optional[str] = None) -> Dict[str, Any]:
        """Reflect on and summarize knowledge and connections."""
        reflection = {
            "agent_name": self.name,
            "atom_type": self.atom_type,
            "knowledge_role": self.knowledge_role,
            "timestamp": datetime.now().isoformat()
        }
        
        if aspect == "connections" or not aspect:
            reflection["connections"] = {
                "incoming_count": len(self.incoming),
                "outgoing_count": len(self.outgoing),
                "total_neighbors": len(self.get_neighbors()),
                "neighbor_types": list(set(n.atom_type for n in self.get_neighbors() if hasattr(n, 'atom_type')))
            }
        
        if aspect == "values" or not aspect:
            values_info = {}
            for key, value in self.get_all_values().items():
                values_info[key] = {
                    "type": type(value).__name__,
                    "value": str(value)[:100],  # Truncate long values
                    "updated": value.updated_at.isoformat()
                }
            reflection["values"] = values_info
        
        if aspect == "role" or not aspect:
            reflection["role_analysis"] = {
                "primary_function": self._analyze_primary_function(),
                "interaction_count": self.interaction_count,
                "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None,
                "collaboration_mode": self.collaboration_mode
            }
        
        return reflection
    
    def _analyze_primary_function(self) -> str:
        """Analyze the primary function of this AtomBot."""
        if self.atom_type == AtomType.CONCEPT_NODE:
            return f"Concept representation and reasoning about '{self.name}'"
        elif self.atom_type == AtomType.PREDICATE_NODE:
            return f"Relationship management and evaluation for '{self.name}'"
        elif self.atom_type == AtomType.SCHEMA_NODE:
            return f"Process execution and procedural knowledge for '{self.name}'"
        else:
            return f"Knowledge representation and agent capabilities for '{self.name}'"
    
    def _setup_value_subscriptions(self):
        """Set up subscriptions to value changes for reactive behavior."""
        # This could be expanded to create more sophisticated reactive behaviors
        pass
    
    async def enhanced_chat(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Enhanced chat that leverages both atom and agent capabilities."""
        # Add context about atom role and connections
        enhanced_context = f"""
Context about me:
- I am a {self.atom_type} representing '{self.name}'
- Knowledge role: {self.knowledge_role}
- Connected to {len(self.get_neighbors())} other agents/atoms
- Have {len(self.get_all_values())} values stored
"""
        
        if context:
            enhanced_context += f"\nAdditional context: {context}"
        
        # Use the chat method from ChatAgent with enhanced context
        full_message = f"{enhanced_context}\n\nUser message: {message}"
        return await self.chat(full_message)
    
    async def collaborate_with_network(self, task: str) -> Dict[str, Any]:
        """Collaborate with other AtomBots in the network to solve a task."""
        if not self.atomspace:
            return {"error": "No atomspace available for collaboration"}
        
        collaboration_results = {
            "task": task,
            "initiator": self.name,
            "responses": [],
            "insights": []
        }
        
        # Find relevant collaborators
        collaborators = []
        for neighbor in self.get_neighbors():
            if hasattr(neighbor, 'chat') and neighbor.collaboration_mode == "active":
                collaborators.append(neighbor)
        
        # Send task to collaborators
        for collaborator in collaborators[:5]:  # Limit to 5 collaborators
            try:
                response = await collaborator.chat(f"Collaboration request from {self.name}: {task}")
                collaboration_results["responses"].append({
                    "agent": str(collaborator),
                    "response": response
                })
            except Exception as e:
                collaboration_results["responses"].append({
                    "agent": str(collaborator),
                    "error": str(e)
                })
        
        # Synthesize insights
        if collaboration_results["responses"]:
            collaboration_results["insights"].append(
                f"Received {len(collaboration_results['responses'])} responses from network collaborators"
            )
        
        return collaboration_results
    
    def get_atombot_status(self) -> Dict[str, Any]:
        """Get comprehensive status of this AtomBot."""
        return {
            "atom_info": {
                "id": self.id,
                "type": self.atom_type,
                "name": self.name,
                "created": self.created_at.isoformat(),
                "updated": self.updated_at.isoformat()
            },
            "agent_info": {
                "interactions": self.interaction_count,
                "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None,
                "collaboration_mode": self.collaboration_mode,
                "knowledge_role": self.knowledge_role
            },
            "network_info": {
                "incoming_links": len(self.incoming),
                "outgoing_links": len(self.outgoing),
                "total_neighbors": len(self.get_neighbors()),
                "values_count": len(self.get_all_values())
            },
            "atomspace": str(self.atomspace) if self.atomspace else None
        }
    
    def __str__(self) -> str:
        return f"AtomBot({self.atom_type}:'{self.name}')"
    
    def __repr__(self) -> str:
        return f"AtomBot(type={self.atom_type}, name='{self.name}', role={self.knowledge_role})"