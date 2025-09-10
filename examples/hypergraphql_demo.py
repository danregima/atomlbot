"""
HyperGraphQL Example - Demonstrating GraphQL introspection for HyperGraphs

This example demonstrates the GraphQL introspection capabilities implemented for
AtomBot's HyperGraph structures, as requested in the problem statement.

It shows how to:
1. Discover the GraphQL API using __schema introspection
2. Query specific types using __type introspection  
3. Get both JSON and IDL format responses
4. Simulate HTTP GET requests for introspection
"""

import sys
import os
import asyncio
import json

# Add parent directory to path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from hypergraph.hypergraph import HyperGraph, HyperNode, HyperEdge
from hypergraph.hypergraphql import (
    HyperGraphQLServer, create_hypergraphql_schema,
    introspect_hypergraph_schema, introspect_hypergraph_type
)


def create_knowledge_hypergraph():
    """Create a sample knowledge hypergraph for demonstration."""
    hg = HyperGraph()
    
    # Create knowledge concept nodes
    concepts = [
        ("ai", {"type": "concept", "name": "Artificial Intelligence", "domain": "computer_science"}),
        ("ml", {"type": "concept", "name": "Machine Learning", "domain": "computer_science"}),
        ("dl", {"type": "concept", "name": "Deep Learning", "domain": "computer_science"}),
        ("nlp", {"type": "concept", "name": "Natural Language Processing", "domain": "computer_science"}),
        ("cv", {"type": "concept", "name": "Computer Vision", "domain": "computer_science"}),
        ("repo", {"type": "repository", "name": "AtomBot Repository", "url": "github.com/danregima/atomlbot"}),
    ]
    
    nodes = {}
    for node_id, data in concepts:
        node = HyperNode(node_id=node_id, data=data)
        nodes[node_id] = node
        hg.add_node(node)
    
    # Create relationship hyperedges
    relationships = [
        ("subsumes_1", [nodes["ai"], nodes["ml"]], "subsumes"),
        ("subsumes_2", [nodes["ml"], nodes["dl"]], "subsumes"),
        ("specializes_1", [nodes["ml"], nodes["nlp"]], "specializes"),  
        ("specializes_2", [nodes["ml"], nodes["cv"]], "specializes"),
        ("hierarchy", [nodes["ai"], nodes["ml"], nodes["dl"]], "hierarchy"),
        ("implements", [nodes["repo"], nodes["ai"], nodes["ml"]], "implements"),
    ]
    
    for edge_id, edge_nodes, relation_type in relationships:
        edge = HyperEdge(edge_id=edge_id, nodes=edge_nodes, edge_type=relation_type, 
                        data={"relation": relation_type})
        edge.weight = 0.8  # Set weight after creation
        hg.add_edge(edge)
    
    return hg


async def demonstrate_schema_introspection():
    """Demonstrate __schema introspection query as mentioned in the problem statement."""
    print("=" * 80)
    print("DEMONSTRATING GRAPHQL SCHEMA INTROSPECTION")
    print("=" * 80)
    
    hg = create_knowledge_hypergraph()
    server = HyperGraphQLServer(hg)
    
    # Example from problem statement: Query __schema to list all types
    schema_query = """
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
    
    print("Query:")
    print(schema_query)
    print("\nResponse:")
    
    result = await server.handle_introspection(schema_query)
    
    # Pretty print the response
    types = result['data']['__schema']['types']
    print(f"Found {len(types)} types in the schema:")
    
    for type_def in types:
        print(f"\n• {type_def['name']} ({type_def['kind']})")
        if type_def.get('description'):
            print(f"  Description: {type_def['description']}")
        if type_def.get('fields'):
            field_names = [f['name'] for f in type_def['fields']]
            print(f"  Fields: {', '.join(field_names)}")
    
    print("\n" + "=" * 80 + "\n")


async def demonstrate_type_introspection():
    """Demonstrate __type introspection query as mentioned in the problem statement."""
    print("DEMONSTRATING GRAPHQL TYPE INTROSPECTION")
    print("=" * 80)
    
    hg = create_knowledge_hypergraph()
    server = HyperGraphQLServer(hg)
    
    # Example from problem statement: Query __type to get details about any type
    # Note: Using "HyperNode" instead of "Repository" since that's our hypergraph type
    type_query = """
    query {
      __type(name: "HyperNode") {
        name
        kind
        description
        fields {
          name
        }
      }
    }
    """
    
    print("Query:")
    print(type_query)
    print("\nResponse:")
    
    result = await server.handle_introspection(type_query)
    type_data = result['data']['__type']
    
    if type_data:
        print(f"Type: {type_data['name']}")
        print(f"Kind: {type_data['kind']}")
        print(f"Description: {type_data.get('description', 'None')}")
        print("Fields:")
        for field in type_data.get('fields', []):
            print(f"  • {field['name']}")
    else:
        print("Type not found")
    
    print("\n" + "=" * 80 + "\n")


async def demonstrate_get_request_introspection():
    """Demonstrate introspection via GET request as mentioned in the problem statement."""
    print("DEMONSTRATING GET REQUEST INTROSPECTION")
    print("=" * 80)
    
    hg = create_knowledge_hypergraph()
    server = HyperGraphQLServer(hg, port=8080)
    
    print(f"Introspection endpoint: {server.get_introspection_endpoint()}")
    print("Simulating GET request for introspection...")
    
    # Simulate the curl request mentioned in problem statement
    result = await server.serve_introspection("GET")
    
    print(f"Status: {result['status']}")
    print(f"Method: {result['method']}")
    print(f"Endpoint: {result['endpoint']}")
    
    if result.get('data'):
        types_count = len(result['data']['__schema']['types'])
        print(f"Schema contains {types_count} types")
        
        # Show some key types
        types = result['data']['__schema']['types']
        hypergraph_types = [t for t in types if t['name'] in ['HyperGraph', 'HyperNode', 'HyperEdge']]
        
        print("\nHyperGraph-specific types found:")
        for htype in hypergraph_types:
            print(f"  • {htype['name']} - {htype.get('description', 'No description')}")
    
    print("\n" + "=" * 80 + "\n")


async def demonstrate_idl_format():
    """Demonstrate IDL format response as mentioned in the problem statement."""
    print("DEMONSTRATING IDL FORMAT RESPONSE")
    print("=" * 80)
    
    hg = create_knowledge_hypergraph()
    server = HyperGraphQLServer(hg)
    
    # Simulate the curl command with IDL media type from problem statement
    print("Simulating request with Accept: application/vnd.github.v4.idl")
    
    schema_query = "__schema { types { name } }"
    result = await server.handle_introspection(schema_query, "application/vnd.github.v4.idl")
    
    print("Response format: IDL (Interface Definition Language)")
    print("Content-Type:", result.get('content_type', 'text/plain'))
    print("\nIDL Schema Definition:")
    print("-" * 40)
    print(result['data'])
    print("-" * 40)
    
    print("\n" + "=" * 80 + "\n")


async def demonstrate_repository_type_query():
    """Demonstrate the exact Repository type query from the problem statement."""
    print("DEMONSTRATING REPOSITORY TYPE QUERY (from problem statement)")
    print("=" * 80)
    
    hg = create_knowledge_hypergraph()
    server = HyperGraphQLServer(hg)
    
    # This is the exact query from the problem statement
    repository_query = """
    query {
      __type(name: "Repository") {
        name
        kind
        description
        fields {
          name
        }
      }
    }
    """
    
    print("Query (from problem statement):")
    print(repository_query)
    print("\nResponse:")
    
    result = await server.handle_introspection(repository_query)
    type_data = result['data']['__type']
    
    if type_data:
        print(f"Type: {type_data['name']}")
        print(f"Kind: {type_data['kind']}")
        print(f"Description: {type_data.get('description', 'None')}")
    else:
        print("Repository type not found in HyperGraph schema")
        print("(This is expected since we're working with HyperGraph types, not GitHub repository types)")
    
    print("\nNote: To see available types in our HyperGraph schema, try:")
    print("__type(name: \"HyperNode\") or __type(name: \"HyperEdge\")")
    
    print("\n" + "=" * 80 + "\n")


async def demonstrate_practical_usage():
    """Show practical usage examples."""
    print("PRACTICAL USAGE EXAMPLES")
    print("=" * 80)
    
    hg = create_knowledge_hypergraph()
    
    print("1. Quick schema introspection:")
    schema_result = await introspect_hypergraph_schema(hg, "json")
    types_count = len(schema_result['data']['__schema']['types'])
    print(f"   Found {types_count} types in schema")
    
    print("\n2. Quick type introspection:")
    type_result = await introspect_hypergraph_type(hg, "HyperEdge")
    if type_result['data']['__type']:
        fields_count = len(type_result['data']['__type']['fields'])
        print(f"   HyperEdge type has {fields_count} fields")
    
    print("\n3. IDL format generation:")
    idl_result = await introspect_hypergraph_schema(hg, "idl")
    idl_lines = len(idl_result.split('\n'))
    print(f"   Generated IDL schema with {idl_lines} lines")
    
    print("\n4. Server setup:")
    server = HyperGraphQLServer(hg, port=8080)
    print(f"   GraphQL endpoint: {server.get_introspection_endpoint()}")
    
    print("\n" + "=" * 80 + "\n")


async def main():
    """Run all demonstration examples."""
    print("HyperGraphQL Introspection Demo")
    print("Implementing GraphQL introspection for HyperGraph structures")
    print("as specified in the problem statement")
    print("\n")
    
    # Run all demonstrations
    await demonstrate_schema_introspection()
    await demonstrate_type_introspection() 
    await demonstrate_get_request_introspection()
    await demonstrate_idl_format()
    await demonstrate_repository_type_query()
    await demonstrate_practical_usage()
    
    print("✅ HyperGraphQL implementation complete!")
    print("\nFeatures implemented:")
    print("• __schema introspection query support")
    print("• __type introspection query support") 
    print("• JSON response format")
    print("• IDL (Interface Definition Language) response format")
    print("• HTTP GET request simulation")
    print("• Full hypergraph type definitions")
    print("• Convenient utility functions")


if __name__ == "__main__":
    asyncio.run(main())