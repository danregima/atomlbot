#!/usr/bin/env python3
"""
HyperGraphQL CLI - Command-line interface for GraphQL introspection

This CLI tool allows users to perform GraphQL introspection queries on
HyperGraph structures, similar to the curl commands mentioned in the problem statement.

Usage:
    python hypergraphql_cli.py --schema                    # Get full schema
    python hypergraphql_cli.py --type HyperNode           # Get specific type
    python hypergraphql_cli.py --schema --format idl      # Get schema in IDL format
    python hypergraphql_cli.py --endpoint                 # Show GraphQL endpoint
"""

import sys
import os
import argparse
import asyncio
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from hypergraph.hypergraph import HyperGraph, HyperNode, HyperEdge
from hypergraph.hypergraphql import (
    HyperGraphQLServer, create_hypergraphql_schema,
    introspect_hypergraph_schema, introspect_hypergraph_type
)


def create_sample_hypergraph():
    """Create a sample hypergraph for CLI demonstration."""
    hg = HyperGraph()
    
    # Add sample knowledge graph
    concepts = [
        ("ai", {"name": "Artificial Intelligence", "domain": "CS"}),
        ("ml", {"name": "Machine Learning", "domain": "CS"}),
        ("dl", {"name": "Deep Learning", "domain": "CS"}),
        ("nlp", {"name": "Natural Language Processing", "domain": "CS"}),
    ]
    
    nodes = {}
    for node_id, data in concepts:
        node = HyperNode(node_id=node_id, data=data)
        nodes[node_id] = node
        hg.add_node(node)
    
    # Add relationships
    edge1 = HyperEdge(edge_id="subsumes_1", nodes=[nodes["ai"], nodes["ml"]], 
                     edge_type="subsumes", data={"relation": "subsumes"})
    edge2 = HyperEdge(edge_id="subsumes_2", nodes=[nodes["ml"], nodes["dl"]], 
                     edge_type="subsumes", data={"relation": "subsumes"})
    edge3 = HyperEdge(edge_id="applies_to", nodes=[nodes["ml"], nodes["nlp"]], 
                     edge_type="applies_to", data={"relation": "applies_to"})
    
    hg.add_edge(edge1)
    hg.add_edge(edge2) 
    hg.add_edge(edge3)
    
    return hg


async def cmd_schema(format_type="json"):
    """Handle --schema command."""
    hg = create_sample_hypergraph()
    
    if format_type.lower() == "idl":
        result = await introspect_hypergraph_schema(hg, "idl")
        print(result)
    else:
        result = await introspect_hypergraph_schema(hg, "json")
        print(json.dumps(result, indent=2))


async def cmd_type(type_name):
    """Handle --type command."""
    hg = create_sample_hypergraph()
    result = await introspect_hypergraph_type(hg, type_name)
    print(json.dumps(result, indent=2))


async def cmd_endpoint():
    """Handle --endpoint command."""
    hg = create_sample_hypergraph()
    server = HyperGraphQLServer(hg, port=8080)
    endpoint = server.get_introspection_endpoint()
    
    print(f"GraphQL Introspection Endpoint: {endpoint}")
    print("\nSupported introspection queries:")
    print("• GET /graphql - Default schema introspection")
    print("• POST /graphql - Custom introspection queries")
    print("\nAccept headers:")
    print("• application/json - JSON format (default)")
    print("• application/vnd.github.v4.idl - IDL format")


async def cmd_list_types():
    """Handle --list-types command."""
    hg = create_sample_hypergraph()
    schema = create_hypergraphql_schema(hg)
    
    print("Available types in HyperGraph schema:")
    print("=" * 40)
    
    for gql_type in schema.types:
        print(f"• {gql_type.name} ({gql_type.kind})")
        if gql_type.description:
            print(f"  {gql_type.description}")
        print()


async def cmd_query(query_string, format_type="json"):
    """Handle --query command."""
    hg = create_sample_hypergraph()
    server = HyperGraphQLServer(hg)
    
    accept_header = "application/json"
    if format_type.lower() == "idl":
        accept_header = "application/vnd.github.v4.idl"
    
    result = await server.handle_introspection(query_string, accept_header)
    
    if isinstance(result.get('data'), str):
        print(result['data'])
    else:
        print(json.dumps(result, indent=2))


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="HyperGraphQL CLI - GraphQL introspection for HyperGraphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --schema                          Get full schema in JSON
  %(prog)s --schema --format idl             Get schema in IDL format
  %(prog)s --type HyperNode                  Get HyperNode type details
  %(prog)s --type Repository                 Try non-existent type
  %(prog)s --list-types                      List all available types
  %(prog)s --endpoint                        Show GraphQL endpoint info
  %(prog)s --query "query { __schema { types { name } } }"
        """
    )
    
    parser.add_argument('--schema', action='store_true',
                       help='Get the full GraphQL schema (__schema query)')
    
    parser.add_argument('--type', metavar='TYPE_NAME',
                       help='Get details about a specific type (__type query)')
    
    parser.add_argument('--list-types', action='store_true',
                       help='List all available types in the schema')
    
    parser.add_argument('--endpoint', action='store_true',
                       help='Show GraphQL endpoint information')
    
    parser.add_argument('--query', metavar='QUERY',
                       help='Execute a custom GraphQL introspection query')
    
    parser.add_argument('--format', choices=['json', 'idl'], default='json',
                       help='Output format (json or idl)')
    
    parser.add_argument('--pretty', action='store_true',
                       help='Pretty-print JSON output (default: true)')
    
    return parser


async def main():
    """Main CLI function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Show help if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    try:
        if args.schema:
            await cmd_schema(args.format)
        elif args.type:
            await cmd_type(args.type)
        elif args.list_types:
            await cmd_list_types()
        elif args.endpoint:
            await cmd_endpoint()
        elif args.query:
            await cmd_query(args.query, args.format)
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())