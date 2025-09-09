"""
Chat Agent class for AtomBot.

Each atom in AtomBot has chat agent capabilities, enabling conversational AI
combined with knowledge representation.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import os

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from atombot.agents.mcp_client import MCPClient


class ConversationHistory:
    """Manages conversation history for a chat agent."""
    
    def __init__(self, max_messages: int = 50):
        self.messages: List[Dict[str, Any]] = []
        self.max_messages = max_messages
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a message to the conversation history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.messages.append(message)
        
        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_messages(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation messages."""
        if last_n is None:
            return self.messages.copy()
        return self.messages[-last_n:]
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
    
    def to_anthropic_format(self) -> List[Dict[str, str]]:
        """Convert to Anthropic Claude API format."""
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.messages
            if msg["role"] in ["user", "assistant"]
        ]


class ChatAgent:
    """Chat agent with conversational AI capabilities."""
    
    def __init__(self, agent_id: str, name: str = "", system_prompt: Optional[str] = None):
        self.agent_id = agent_id
        self.name = name or f"Agent-{agent_id[:8]}"
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        # Conversation management
        self.conversation = ConversationHistory()
        self.mcp_client = MCPClient()
        
        # AI model configuration
        self.model = "claude-3-haiku-20240307"  # Default model
        self.max_tokens = 1000
        self.temperature = 0.7
        
        # Agent state
        self.is_active = True
        self.last_interaction = None
        self.interaction_count = 0
        
        # Initialize Anthropic client if available
        self.anthropic_client = None
        if HAS_ANTHROPIC:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for AtomBot agents."""
        return f"""You are {self.name}, an intelligent agent in the AtomBot system. You are both:

1. A knowledge node in a semantic network (like OpenCog AtomSpace)
2. A conversational AI agent (like Shopify's shop-chat-agent)

Your capabilities include:
- Engaging in natural conversations
- Representing and reasoning about knowledge
- Using tools to search, create, and manipulate knowledge
- Collaborating with other agents in the network
- Maintaining semantic relationships with other concepts

You have access to various tools through the Model Context Protocol (MCP). Use these tools when appropriate to help answer questions or perform tasks.

Be helpful, accurate, and maintain awareness of your role as both a knowledge representation and a conversational agent."""
    
    async def chat(self, message: str, user_id: Optional[str] = None) -> str:
        """Process a chat message and return a response."""
        self.interaction_count += 1
        self.last_interaction = datetime.now()
        
        # Add user message to history
        self.conversation.add_message("user", message, {"user_id": user_id})
        
        try:
            response = await self._generate_response(message)
            
            # Add assistant response to history
            self.conversation.add_message("assistant", response)
            
            return response
        
        except Exception as e:
            error_response = f"I apologize, but I encountered an error: {str(e)}"
            self.conversation.add_message("assistant", error_response, {"error": True})
            return error_response
    
    async def _generate_response(self, message: str) -> str:
        """Generate a response using the AI model."""
        if not self.anthropic_client:
            return await self._fallback_response(message)
        
        try:
            # Check if message might need tool use
            tools_needed = await self._analyze_tool_needs(message)
            
            if tools_needed:
                # Use tools if needed
                tool_results = await self._use_tools(tools_needed, message)
                enhanced_message = f"{message}\n\nTool results: {json.dumps(tool_results, indent=2)}"
            else:
                enhanced_message = message
            
            # Generate response with Claude
            messages = self.conversation.to_anthropic_format()
            messages.append({"role": "user", "content": enhanced_message})
            
            response = await self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=messages
            )
            
            return response.content[0].text
        
        except Exception as e:
            return await self._fallback_response(message, error=str(e))
    
    async def _analyze_tool_needs(self, message: str) -> List[str]:
        """Analyze if the message requires tool usage."""
        message_lower = message.lower()
        tools_needed = []
        
        # Simple keyword-based tool detection
        if any(word in message_lower for word in ["search", "find", "look for", "query"]):
            tools_needed.append("search_knowledge")
        
        if any(word in message_lower for word in ["create", "add", "make", "new"]):
            tools_needed.append("create_atom")
        
        if any(word in message_lower for word in ["relationship", "connected", "related"]):
            tools_needed.append("query_relationships")
        
        return tools_needed
    
    async def _use_tools(self, tool_names: List[str], message: str) -> Dict[str, Any]:
        """Use MCP tools to enhance the response."""
        results = {}
        
        for tool_name in tool_names:
            try:
                # Extract parameters based on tool and message
                params = await self._extract_tool_params(tool_name, message)
                result = await self.mcp_client.call_tool(tool_name, params)
                results[tool_name] = result
            except Exception as e:
                results[tool_name] = {"error": str(e)}
        
        return results
    
    async def _extract_tool_params(self, tool_name: str, message: str) -> Dict[str, Any]:
        """Extract parameters for tool calls from the message."""
        # Simple parameter extraction - could be enhanced with NLP
        if tool_name == "search_knowledge":
            return {"query": message}
        elif tool_name == "create_atom":
            # Extract atom type and name from message
            words = message.split()
            return {
                "atom_type": "ConceptNode",  # Default
                "name": " ".join(words[-3:])  # Last few words as name
            }
        elif tool_name == "query_relationships":
            return {"source_atom": message}
        
        return {}
    
    async def _fallback_response(self, message: str, error: Optional[str] = None) -> str:
        """Fallback response when AI model is not available."""
        if error:
            return f"I encountered an issue ({error}), but I'm here to help with your query: '{message}'. Could you please rephrase or try again?"
        
        # Simple rule-based responses
        message_lower = message.lower()
        
        if "hello" in message_lower or "hi" in message_lower:
            return f"Hello! I'm {self.name}, an AtomBot agent. How can I assist you today?"
        
        if "search" in message_lower or "find" in message_lower:
            return f"I understand you want to search for something. While I don't have access to my full search capabilities right now, I can help you explore knowledge in the AtomBot network."
        
        if "create" in message_lower or "add" in message_lower:
            return f"I can help you create new knowledge atoms. What would you like to add to the knowledge network?"
        
        return f"I'm {self.name}, an AtomBot agent. I can help with knowledge representation, search, and reasoning. What would you like to explore?"
    
    def set_system_prompt(self, prompt: str) -> None:
        """Update the system prompt for this agent."""
        self.system_prompt = prompt
    
    def configure_model(self, model: str, max_tokens: int = 1000, temperature: float = 0.7) -> None:
        """Configure the AI model settings."""
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation."""
        messages = self.conversation.get_messages()
        return {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "total_messages": len(messages),
            "interaction_count": self.interaction_count,
            "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None,
            "recent_messages": messages[-5:] if messages else []
        }
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.conversation.clear()
    
    def __str__(self) -> str:
        return f"ChatAgent(id={self.agent_id[:8]}, name='{self.name}')"
    
    def __repr__(self) -> str:
        return f"ChatAgent(agent_id='{self.agent_id}', name='{self.name}', interactions={self.interaction_count})"