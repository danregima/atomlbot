"""
HyperGraphQL - GraphQL introspection for HyperGraph structures.

This module provides GraphQL introspection capabilities for exploring
HyperGraph schemas, types, and fields through standard GraphQL queries.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

from .hypergraph import HyperGraph, HyperNode, HyperEdge


@dataclass
class GraphQLType:
    """Represents a GraphQL type in the schema."""
    name: str
    kind: str
    description: Optional[str] = None
    fields: Optional[List[Dict[str, Any]]] = None
    enumValues: Optional[List[Dict[str, str]]] = None
    interfaces: Optional[List[str]] = None
    possibleTypes: Optional[List[str]] = None
    inputFields: Optional[List[Dict[str, Any]]] = None


@dataclass
class GraphQLField:
    """Represents a field in a GraphQL type."""
    name: str
    description: Optional[str] = None
    type: Optional[Dict[str, Any]] = None
    args: Optional[List[Dict[str, Any]]] = None
    isDeprecated: bool = False
    deprecationReason: Optional[str] = None


class HyperGraphQLSchema:
    """
    GraphQL schema introspection for HyperGraph structures.
    
    Provides __schema and __type queries for exploring the hypergraph
    structure through standard GraphQL introspection.
    """
    
    def __init__(self, hypergraph: HyperGraph):
        self.hypergraph = hypergraph
        self.types = self._build_schema_types()
        self.query_type = self._build_query_type()
        self.mutation_type = None  # Not implemented yet
        self.subscription_type = None  # Not implemented yet
    
    def _build_schema_types(self) -> List[GraphQLType]:
        """Build the list of types available in the schema."""
        types = []
        
        # Add fundamental GraphQL introspection types
        types.extend(self._get_introspection_types())
        
        # Add HyperGraph-specific types
        types.extend(self._get_hypergraph_types())
        
        return types
    
    def _get_introspection_types(self) -> List[GraphQLType]:
        """Get standard GraphQL introspection types."""
        return [
            GraphQLType(
                name="__Schema",
                kind="OBJECT",
                description="A GraphQL Schema defines the capabilities of a GraphQL server.",
                fields=[
                    {"name": "types", "type": {"kind": "NON_NULL", "ofType": {"kind": "LIST", "ofType": {"kind": "NON_NULL", "ofType": {"name": "__Type", "kind": "OBJECT"}}}}},
                    {"name": "queryType", "type": {"kind": "NON_NULL", "ofType": {"name": "__Type", "kind": "OBJECT"}}},
                    {"name": "mutationType", "type": {"name": "__Type", "kind": "OBJECT"}},
                    {"name": "subscriptionType", "type": {"name": "__Type", "kind": "OBJECT"}},
                    {"name": "directives", "type": {"kind": "NON_NULL", "ofType": {"kind": "LIST", "ofType": {"kind": "NON_NULL", "ofType": {"name": "__Directive", "kind": "OBJECT"}}}}},
                ]
            ),
            GraphQLType(
                name="__Type",
                kind="OBJECT",
                description="The fundamental unit of any GraphQL Schema is the type.",
                fields=[
                    {"name": "kind", "type": {"kind": "NON_NULL", "ofType": {"name": "__TypeKind", "kind": "ENUM"}}},
                    {"name": "name", "type": {"name": "String", "kind": "SCALAR"}},
                    {"name": "description", "type": {"name": "String", "kind": "SCALAR"}},
                    {"name": "fields", "type": {"kind": "LIST", "ofType": {"kind": "NON_NULL", "ofType": {"name": "__Field", "kind": "OBJECT"}}}},
                    {"name": "interfaces", "type": {"kind": "LIST", "ofType": {"kind": "NON_NULL", "ofType": {"name": "__Type", "kind": "OBJECT"}}}},
                    {"name": "possibleTypes", "type": {"kind": "LIST", "ofType": {"kind": "NON_NULL", "ofType": {"name": "__Type", "kind": "OBJECT"}}}},
                    {"name": "enumValues", "type": {"kind": "LIST", "ofType": {"kind": "NON_NULL", "ofType": {"name": "__EnumValue", "kind": "OBJECT"}}}},
                    {"name": "inputFields", "type": {"kind": "LIST", "ofType": {"kind": "NON_NULL", "ofType": {"name": "__InputValue", "kind": "OBJECT"}}}},
                    {"name": "ofType", "type": {"name": "__Type", "kind": "OBJECT"}},
                ]
            ),
            GraphQLType(
                name="__Field",
                kind="OBJECT", 
                description="Object and Interface types are described by a list of Fields.",
                fields=[
                    {"name": "name", "type": {"kind": "NON_NULL", "ofType": {"name": "String", "kind": "SCALAR"}}},
                    {"name": "description", "type": {"name": "String", "kind": "SCALAR"}},
                    {"name": "args", "type": {"kind": "NON_NULL", "ofType": {"kind": "LIST", "ofType": {"kind": "NON_NULL", "ofType": {"name": "__InputValue", "kind": "OBJECT"}}}}},
                    {"name": "type", "type": {"kind": "NON_NULL", "ofType": {"name": "__Type", "kind": "OBJECT"}}},
                    {"name": "isDeprecated", "type": {"kind": "NON_NULL", "ofType": {"name": "Boolean", "kind": "SCALAR"}}},
                    {"name": "deprecationReason", "type": {"name": "String", "kind": "SCALAR"}},
                ]
            ),
            GraphQLType(
                name="String",
                kind="SCALAR",
                description="The `String` scalar type represents textual data."
            ),
            GraphQLType(
                name="Boolean", 
                kind="SCALAR",
                description="The `Boolean` scalar type represents `true` or `false`."
            ),
            GraphQLType(
                name="ID",
                kind="SCALAR", 
                description="The `ID` scalar type represents a unique identifier."
            ),
            GraphQLType(
                name="Int",
                kind="SCALAR",
                description="The `Int` scalar type represents non-fractional signed whole numeric values."
            ),
            GraphQLType(
                name="Float",
                kind="SCALAR",
                description="The `Float` scalar type represents signed double-precision fractional values."
            ),
        ]
    
    def _get_hypergraph_types(self) -> List[GraphQLType]:
        """Get HyperGraph-specific types."""
        return [
            GraphQLType(
                name="HyperGraph",
                kind="OBJECT",
                description="A hypergraph containing nodes and hyperedges.",
                fields=[
                    {"name": "id", "type": {"kind": "NON_NULL", "ofType": {"name": "ID", "kind": "SCALAR"}}},
                    {"name": "nodes", "type": {"kind": "LIST", "ofType": {"kind": "NON_NULL", "ofType": {"name": "HyperNode", "kind": "OBJECT"}}}},
                    {"name": "edges", "type": {"kind": "LIST", "ofType": {"kind": "NON_NULL", "ofType": {"name": "HyperEdge", "kind": "OBJECT"}}}},
                    {"name": "nodeCount", "type": {"kind": "NON_NULL", "ofType": {"name": "Int", "kind": "SCALAR"}}},
                    {"name": "edgeCount", "type": {"kind": "NON_NULL", "ofType": {"name": "Int", "kind": "SCALAR"}}},
                    {"name": "createdAt", "type": {"kind": "NON_NULL", "ofType": {"name": "String", "kind": "SCALAR"}}},
                ]
            ),
            GraphQLType(
                name="HyperNode",
                kind="OBJECT",
                description="A node in a hypergraph that can participate in multiple hyperedges.",
                fields=[
                    {"name": "id", "type": {"kind": "NON_NULL", "ofType": {"name": "ID", "kind": "SCALAR"}}},
                    {"name": "data", "type": {"name": "String", "kind": "SCALAR"}},
                    {"name": "features", "type": {"kind": "LIST", "ofType": {"kind": "NON_NULL", "ofType": {"name": "Float", "kind": "SCALAR"}}}},
                    {"name": "hyperedges", "type": {"kind": "LIST", "ofType": {"kind": "NON_NULL", "ofType": {"name": "HyperEdge", "kind": "OBJECT"}}}},
                    {"name": "degree", "type": {"kind": "NON_NULL", "ofType": {"name": "Int", "kind": "SCALAR"}}},
                    {"name": "createdAt", "type": {"kind": "NON_NULL", "ofType": {"name": "String", "kind": "SCALAR"}}},
                ]
            ),
            GraphQLType(
                name="HyperEdge", 
                kind="OBJECT",
                description="A hyperedge connecting multiple nodes in a hypergraph.",
                fields=[
                    {"name": "id", "type": {"kind": "NON_NULL", "ofType": {"name": "ID", "kind": "SCALAR"}}},
                    {"name": "nodes", "type": {"kind": "LIST", "ofType": {"kind": "NON_NULL", "ofType": {"name": "HyperNode", "kind": "OBJECT"}}}},
                    {"name": "weight", "type": {"name": "Float", "kind": "SCALAR"}},
                    {"name": "data", "type": {"name": "String", "kind": "SCALAR"}},
                    {"name": "cardinality", "type": {"kind": "NON_NULL", "ofType": {"name": "Int", "kind": "SCALAR"}}},
                    {"name": "createdAt", "type": {"kind": "NON_NULL", "ofType": {"name": "String", "kind": "SCALAR"}}},
                ]
            ),
            GraphQLType(
                name="Query",
                kind="OBJECT",
                description="The root query type.",
                fields=[
                    {"name": "hypergraph", "type": {"name": "HyperGraph", "kind": "OBJECT"}},
                    {"name": "node", "args": [{"name": "id", "type": {"kind": "NON_NULL", "ofType": {"name": "ID", "kind": "SCALAR"}}}], "type": {"name": "HyperNode", "kind": "OBJECT"}},
                    {"name": "edge", "args": [{"name": "id", "type": {"kind": "NON_NULL", "ofType": {"name": "ID", "kind": "SCALAR"}}}], "type": {"name": "HyperEdge", "kind": "OBJECT"}},
                    {"name": "nodes", "type": {"kind": "LIST", "ofType": {"kind": "NON_NULL", "ofType": {"name": "HyperNode", "kind": "OBJECT"}}}},
                    {"name": "edges", "type": {"kind": "LIST", "ofType": {"kind": "NON_NULL", "ofType": {"name": "HyperEdge", "kind": "OBJECT"}}}},
                ]
            ),
        ]
    
    def _build_query_type(self) -> GraphQLType:
        """Build the root Query type."""
        return next(t for t in self.types if t.name == "Query")
    
    def introspect_schema(self) -> Dict[str, Any]:
        """Handle __schema introspection query."""
        return {
            "data": {
                "__schema": {
                    "types": [asdict(t) for t in self.types],
                    "queryType": asdict(self.query_type) if self.query_type else None,
                    "mutationType": asdict(self.mutation_type) if self.mutation_type else None,
                    "subscriptionType": asdict(self.subscription_type) if self.subscription_type else None,
                    "directives": []  # Not implemented
                }
            }
        }
    
    def introspect_type(self, type_name: str) -> Dict[str, Any]:
        """Handle __type introspection query."""
        type_def = next((t for t in self.types if t.name == type_name), None)
        
        if not type_def:
            return {
                "data": {
                    "__type": None
                }
            }
        
        return {
            "data": {
                "__type": asdict(type_def)
            }
        }
    
    def to_idl(self) -> str:
        """Convert schema to GraphQL IDL (Interface Definition Language) format."""
        idl_parts = []
        
        # Add scalar definitions
        scalars = [t for t in self.types if t.kind == "SCALAR" and t.name not in ["String", "Boolean", "ID", "Int", "Float"]]
        for scalar in scalars:
            if scalar.description:
                idl_parts.append(f'"""{scalar.description}"""')
            idl_parts.append(f"scalar {scalar.name}")
            idl_parts.append("")
        
        # Add object type definitions
        objects = [t for t in self.types if t.kind == "OBJECT" and not t.name.startswith("__")]
        for obj in objects:
            if obj.description:
                idl_parts.append(f'"""{obj.description}"""')
            
            idl_parts.append(f"type {obj.name} {{")
            
            if obj.fields:
                for field in obj.fields:
                    field_line = f"  {field['name']}"
                    
                    # Add arguments if present
                    if field.get('args'):
                        args = []
                        for arg in field['args']:
                            arg_type = self._format_type_reference(arg['type'])
                            args.append(f"{arg['name']}: {arg_type}")
                        field_line += f"({', '.join(args)})"
                    
                    # Add return type
                    return_type = self._format_type_reference(field['type'])
                    field_line += f": {return_type}"
                    
                    idl_parts.append(field_line)
            
            idl_parts.append("}")
            idl_parts.append("")
        
        return "\n".join(idl_parts)
    
    def _format_type_reference(self, type_ref: Dict[str, Any]) -> str:
        """Format a type reference for IDL output."""
        if type_ref.get('kind') == 'NON_NULL':
            return f"{self._format_type_reference(type_ref['ofType'])}!"
        elif type_ref.get('kind') == 'LIST':
            return f"[{self._format_type_reference(type_ref['ofType'])}]"
        else:
            return type_ref.get('name', 'Unknown')


class HyperGraphQLServer:
    """
    HTTP server for HyperGraphQL introspection queries.
    
    Serves GraphQL introspection queries over HTTP with support for both
    JSON and IDL response formats.
    """
    
    def __init__(self, hypergraph: HyperGraph, port: int = 8080):
        self.hypergraph = hypergraph
        self.schema = HyperGraphQLSchema(hypergraph)
        self.port = port
    
    async def handle_introspection(self, query: str, accept_header: str = "application/json") -> Dict[str, Any]:
        """Handle GraphQL introspection queries."""
        query = query.strip()
        
        # Handle __schema query
        if "__schema" in query and "types" in query:
            result = self.schema.introspect_schema()
            
            # Return IDL format if requested
            if "application/vnd.github.v4.idl" in accept_header:
                return {
                    "data": self.schema.to_idl(),
                    "content_type": "text/plain"
                }
            
            return result
        
        # Handle __type query  
        if "__type" in query:
            # Extract type name from query (simplified parsing)
            import re
            type_match = re.search(r'__type\s*\(\s*name:\s*"([^"]+)"', query)
            if type_match:
                type_name = type_match.group(1)
                return self.schema.introspect_type(type_name)
        
        # Default empty response for unsupported queries
        return {"data": None}
    
    def get_introspection_endpoint(self):
        """Get the introspection endpoint URL."""
        return f"http://localhost:{self.port}/graphql"
    
    async def serve_introspection(self, request_method: str = "GET", query: str = "", accept_header: str = "application/json") -> Dict[str, Any]:
        """
        Serve introspection requests.
        
        This simulates HTTP server behavior for testing purposes.
        In a real implementation, this would be integrated with aiohttp or similar.
        """
        if request_method == "GET":
            # Default introspection query for GET requests
            if not query:
                query = """
                query {
                  __schema {
                    types {
                      name
                      kind
                      description
                      fields {
                        name
                      }
                    }
                  }
                }
                """
            
            result = await self.handle_introspection(query, accept_header)
            
            # Add metadata for HTTP response simulation
            return {
                **result,
                "status": 200,
                "method": request_method,
                "endpoint": self.get_introspection_endpoint()
            }
        
        return {
            "error": "Method not supported",
            "status": 405
        }


# Convenience functions for easy usage
def create_hypergraphql_schema(hypergraph: HyperGraph) -> HyperGraphQLSchema:
    """Create a HyperGraphQL schema from a hypergraph."""
    return HyperGraphQLSchema(hypergraph)


def create_hypergraphql_server(hypergraph: HyperGraph, port: int = 8080) -> HyperGraphQLServer:
    """Create a HyperGraphQL server from a hypergraph."""
    return HyperGraphQLServer(hypergraph, port)


async def introspect_hypergraph_schema(hypergraph: HyperGraph, output_format: str = "json") -> Union[Dict[str, Any], str]:
    """
    Perform schema introspection on a hypergraph.
    
    Args:
        hypergraph: The hypergraph to introspect
        output_format: "json" or "idl"
    
    Returns:
        Schema introspection result in the requested format
    """
    schema = HyperGraphQLSchema(hypergraph)
    
    if output_format.lower() == "idl":
        return schema.to_idl()
    else:
        return schema.introspect_schema()


async def introspect_hypergraph_type(hypergraph: HyperGraph, type_name: str) -> Dict[str, Any]:
    """
    Perform type introspection on a hypergraph.
    
    Args:
        hypergraph: The hypergraph to introspect
        type_name: Name of the type to introspect
    
    Returns:
        Type introspection result
    """
    schema = HyperGraphQLSchema(hypergraph)
    return schema.introspect_type(type_name)