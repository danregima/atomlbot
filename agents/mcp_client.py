"""
Model Context Protocol (MCP) client for AtomBot.

This enables chat agents to use tools and interact with external systems,
following the MCP specification from Anthropic.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Callable
import aiohttp
from datetime import datetime


class MCPTool:
    """Represents an MCP tool that can be called by chat agents."""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any], 
                 handler: Callable):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.call_count = 0
        self.last_called = None
    
    async def call(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call the tool with given arguments."""
        self.call_count += 1
        self.last_called = datetime.now()
        
        try:
            if asyncio.iscoroutinefunction(self.handler):
                result = await self.handler(**arguments)
            else:
                result = self.handler(**arguments)
            
            return {
                "success": True,
                "result": result,
                "tool": self.name,
                "timestamp": self.last_called.isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": self.name,
                "timestamp": self.last_called.isoformat()
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary for MCP protocol."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


class MCPClient:
    """MCP client for managing tools and external integrations."""
    
    def __init__(self):
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, Any] = {}
        self.prompts: Dict[str, str] = {}
        
        # Register default AtomBot tools
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools for AtomBot functionality."""
        
        # Knowledge search tool
        self.register_tool(
            name="search_knowledge",
            description="Search for knowledge in the atomspace",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "Search query"
                    },
                    "atom_type": {
                        "type": "string",
                        "description": "Filter by atom type (optional)"
                    }
                },
                "required": ["query"]
            },
            handler=self._search_knowledge
        )
        
        # Create atom tool
        self.register_tool(
            name="create_atom",
            description="Create a new atom in the atomspace",
            parameters={
                "type": "object", 
                "properties": {
                    "atom_type": {
                        "type": "string",
                        "description": "Type of atom to create"
                    },
                    "name": {
                        "type": "string",
                        "description": "Name of the atom"
                    },
                    "values": {
                        "type": "object",
                        "description": "Initial values to set (optional)"
                    }
                },
                "required": ["atom_type", "name"]
            },
            handler=self._create_atom
        )
        
        # Query relationships tool
        self.register_tool(
            name="query_relationships", 
            description="Query relationships between atoms",
            parameters={
                "type": "object",
                "properties": {
                    "source_atom": {
                        "type": "string",
                        "description": "Source atom name or ID"
                    },
                    "relationship_type": {
                        "type": "string", 
                        "description": "Type of relationship to query"
                    }
                },
                "required": ["source_atom"]
            },
            handler=self._query_relationships
        )
    
    async def _search_knowledge(self, query: str, atom_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Default knowledge search implementation."""
        # This would be overridden by actual AtomBot instances
        return [{
            "message": f"Knowledge search for '{query}' (type: {atom_type})",
            "results": []
        }]
    
    async def _create_atom(self, atom_type: str, name: str, values: Optional[Dict] = None) -> Dict[str, Any]:
        """Default atom creation implementation."""
        return {
            "message": f"Would create {atom_type} named '{name}'",
            "atom_type": atom_type,
            "name": name,
            "values": values or {}
        }
    
    async def _query_relationships(self, source_atom: str, relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Default relationship query implementation."""
        return [{
            "message": f"Querying relationships for '{source_atom}' (type: {relationship_type})",
            "relationships": []
        }]
    
    def register_tool(self, name: str, description: str, parameters: Dict[str, Any], 
                     handler: Callable) -> None:
        """Register a new MCP tool."""
        tool = MCPTool(name, description, parameters, handler)
        self.tools[name] = tool
    
    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        return [tool.to_dict() for tool in self.tools.values()]
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool with given arguments."""
        tool = self.get_tool(name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{name}' not found",
                "available_tools": list(self.tools.keys())
            }
        
        return await tool.call(arguments)
    
    def register_resource(self, name: str, resource: Any) -> None:
        """Register a resource that tools can access."""
        self.resources[name] = resource
    
    def get_resource(self, name: str) -> Any:
        """Get a registered resource."""
        return self.resources.get(name)
    
    def register_prompt(self, name: str, prompt: str) -> None:
        """Register a prompt template."""
        self.prompts[name] = prompt
    
    def get_prompt(self, name: str, **kwargs) -> str:
        """Get a prompt template with variable substitution."""
        template = self.prompts.get(name, "")
        try:
            return template.format(**kwargs)
        except KeyError as e:
            return f"Prompt template error: missing variable {e}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about tool usage."""
        return {
            "total_tools": len(self.tools),
            "tool_stats": {
                name: {
                    "call_count": tool.call_count,
                    "last_called": tool.last_called.isoformat() if tool.last_called else None
                }
                for name, tool in self.tools.items()
            }
        }