# HyperGraphQL

HyperGraphQL adds GraphQL introspection capabilities to AtomBot's HyperGraph structures, enabling discovery and exploration of hypergraph schemas through standard GraphQL queries.

## Features

- **__schema introspection**: Query all types, fields, and schema structure
- **__type introspection**: Query specific type details and field information  
- **JSON format support**: Standard GraphQL JSON response format
- **IDL format support**: Interface Definition Language output format
- **HTTP simulation**: Simulated GET request handling for introspection
- **CLI tool**: Command-line interface for introspection queries

## Quick Start

```python
from atombot.hypergraph import HyperGraph, HyperNode, HyperEdge
from atombot.hypergraph.hypergraphql import (
    create_hypergraphql_schema, introspect_hypergraph_schema
)

# Create a hypergraph
hg = HyperGraph()
node1 = HyperNode("node1", {"name": "AI"})
node2 = HyperNode("node2", {"name": "ML"})
hg.add_node(node1)
hg.add_node(node2)

edge = HyperEdge("edge1", [node1, node2], "contains")
hg.add_edge(edge)

# Introspect the schema
schema_result = await introspect_hypergraph_schema(hg, "json")
print(f"Found {len(schema_result['data']['__schema']['types'])} types")

# Get IDL format
idl_schema = await introspect_hypergraph_schema(hg, "idl")
print(idl_schema)
```

## GraphQL Introspection Queries

### Schema Introspection

Query all types in the schema:

```graphql
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
```

### Type Introspection

Query specific type details:

```graphql
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
```

## HTTP Server Simulation

```python
from atombot.hypergraph.hypergraphql import HyperGraphQLServer

server = HyperGraphQLServer(hypergraph, port=8080)

# Simulate GET request
result = await server.serve_introspection("GET")

# Handle custom query
query = 'query { __schema { types { name } } }'
result = await server.handle_introspection(query)

# Get IDL format
result = await server.handle_introspection(query, "application/vnd.github.v4.idl")
```

## CLI Usage

The package includes a command-line tool for introspection:

```bash
# List all available types
python examples/hypergraphql_cli.py --list-types

# Get schema in JSON format
python examples/hypergraphql_cli.py --schema

# Get schema in IDL format  
python examples/hypergraphql_cli.py --schema --format idl

# Query specific type
python examples/hypergraphql_cli.py --type HyperNode

# Custom query
python examples/hypergraphql_cli.py --query "query { __schema { types { name } } }"

# Show endpoint info
python examples/hypergraphql_cli.py --endpoint
```

## Types Available for Introspection

The HyperGraphQL schema includes these types:

- **HyperGraph**: Root hypergraph container
- **HyperNode**: Individual nodes in the hypergraph
- **HyperEdge**: Hyperedges connecting multiple nodes
- **Query**: Root query type for data access
- Standard GraphQL introspection types (__Schema, __Type, __Field)
- Scalar types (String, Boolean, ID, Int, Float)

## IDL Output Example

```graphql
"""A hypergraph containing nodes and hyperedges."""
type HyperGraph {
  id: ID!
  nodes: [HyperNode!]
  edges: [HyperEdge!]
  nodeCount: Int!
  edgeCount: Int!
  createdAt: String!
}

"""A node in a hypergraph that can participate in multiple hyperedges."""
type HyperNode {
  id: ID!
  data: String
  features: [Float!]
  hyperedges: [HyperEdge!]
  degree: Int!
  createdAt: String!
}
```

## Integration with Existing Components

HyperGraphQL integrates seamlessly with existing AtomBot hypergraph components:

- Works with existing HyperGraph, HyperNode, HyperEdge classes
- Compatible with HGNN (HyperGraph Neural Networks)
- Supports HyperAgent and HyperChannel architectures
- Can introspect any hypergraph structure in the system

This makes it easy to explore and understand complex hypergraph schemas through standard GraphQL introspection protocols.