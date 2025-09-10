"""
Tests for HyperGraphQL functionality.

Tests GraphQL introspection capabilities for HyperGraph structures,
including __schema and __type queries in both JSON and IDL formats.
"""

import pytest
import json
import asyncio
from datetime import datetime

from atombot.hypergraph.hypergraph import HyperGraph, HyperNode, HyperEdge
from atombot.hypergraph.hypergraphql import (
    HyperGraphQLSchema, HyperGraphQLServer, GraphQLType, GraphQLField,
    create_hypergraphql_schema, create_hypergraphql_server,
    introspect_hypergraph_schema, introspect_hypergraph_type
)


@pytest.fixture
def sample_hypergraph():
    """Create a sample hypergraph for testing."""
    hg = HyperGraph()
    
    # Add some nodes
    node1 = HyperNode(node_id="node1", data={"type": "concept", "name": "AI"})
    node2 = HyperNode(node_id="node2", data={"type": "concept", "name": "ML"})
    node3 = HyperNode(node_id="node3", data={"type": "concept", "name": "DL"})
    
    hg.add_node(node1)
    hg.add_node(node2)
    hg.add_node(node3)
    
    # Add some hyperedges
    edge1 = HyperEdge(edge_id="edge1", nodes={node1, node2}, weight=0.8, data={"relation": "contains"})
    edge2 = HyperEdge(edge_id="edge2", nodes={node2, node3}, weight=0.9, data={"relation": "specializes"})
    edge3 = HyperEdge(edge_id="edge3", nodes={node1, node2, node3}, weight=0.7, data={"relation": "hierarchy"})
    
    hg.add_hyperedge(edge1)
    hg.add_hyperedge(edge2)
    hg.add_hyperedge(edge3)
    
    return hg


class TestGraphQLType:
    """Test GraphQLType functionality."""
    
    def test_graphql_type_creation(self):
        """Test creating GraphQL types."""
        gql_type = GraphQLType(
            name="TestType",
            kind="OBJECT",
            description="A test type",
            fields=[{"name": "id", "type": {"name": "ID", "kind": "SCALAR"}}]
        )
        
        assert gql_type.name == "TestType"
        assert gql_type.kind == "OBJECT"
        assert gql_type.description == "A test type"
        assert len(gql_type.fields) == 1
        assert gql_type.fields[0]["name"] == "id"


class TestHyperGraphQLSchema:
    """Test HyperGraphQLSchema functionality."""
    
    def test_schema_creation(self, sample_hypergraph):
        """Test creating a GraphQL schema from a hypergraph."""
        schema = HyperGraphQLSchema(sample_hypergraph)
        
        assert schema.hypergraph == sample_hypergraph
        assert len(schema.types) > 0
        assert schema.query_type is not None
        assert schema.query_type.name == "Query"
    
    def test_schema_introspection_types(self, sample_hypergraph):
        """Test that schema contains expected introspection types."""
        schema = HyperGraphQLSchema(sample_hypergraph)
        
        type_names = [t.name for t in schema.types]
        
        # Check for standard GraphQL introspection types
        assert "__Schema" in type_names
        assert "__Type" in type_names
        assert "__Field" in type_names
        
        # Check for scalar types
        assert "String" in type_names
        assert "Boolean" in type_names
        assert "ID" in type_names
        assert "Int" in type_names
        assert "Float" in type_names
        
        # Check for hypergraph types
        assert "HyperGraph" in type_names
        assert "HyperNode" in type_names
        assert "HyperEdge" in type_names
        assert "Query" in type_names
    
    def test_hypergraph_type_fields(self, sample_hypergraph):
        """Test that HyperGraph type has expected fields."""
        schema = HyperGraphQLSchema(sample_hypergraph)
        
        hypergraph_type = next(t for t in schema.types if t.name == "HyperGraph")
        field_names = [f["name"] for f in hypergraph_type.fields]
        
        assert "id" in field_names
        assert "nodes" in field_names
        assert "edges" in field_names
        assert "nodeCount" in field_names
        assert "edgeCount" in field_names
        assert "createdAt" in field_names
    
    def test_hypernode_type_fields(self, sample_hypergraph):
        """Test that HyperNode type has expected fields."""
        schema = HyperGraphQLSchema(sample_hypergraph)
        
        hypernode_type = next(t for t in schema.types if t.name == "HyperNode")
        field_names = [f["name"] for f in hypernode_type.fields]
        
        assert "id" in field_names
        assert "data" in field_names
        assert "features" in field_names
        assert "hyperedges" in field_names
        assert "degree" in field_names
        assert "createdAt" in field_names
    
    def test_hyperedge_type_fields(self, sample_hypergraph):
        """Test that HyperEdge type has expected fields."""
        schema = HyperGraphQLSchema(sample_hypergraph)
        
        hyperedge_type = next(t for t in schema.types if t.name == "HyperEdge")
        field_names = [f["name"] for f in hyperedge_type.fields]
        
        assert "id" in field_names
        assert "nodes" in field_names
        assert "weight" in field_names
        assert "data" in field_names
        assert "cardinality" in field_names
        assert "createdAt" in field_names


class TestGraphQLIntrospection:
    """Test GraphQL introspection queries."""
    
    def test_schema_introspection(self, sample_hypergraph):
        """Test __schema introspection query."""
        schema = HyperGraphQLSchema(sample_hypergraph)
        result = schema.introspect_schema()
        
        assert "data" in result
        assert "__schema" in result["data"]
        
        schema_data = result["data"]["__schema"]
        assert "types" in schema_data
        assert "queryType" in schema_data
        
        # Check that we have the expected number of types
        types = schema_data["types"]
        assert len(types) > 10  # Should have introspection + hypergraph types
        
        # Check that query type is properly set
        query_type = schema_data["queryType"]
        assert query_type["name"] == "Query"
    
    def test_type_introspection_existing_type(self, sample_hypergraph):
        """Test __type introspection query for existing type."""
        schema = HyperGraphQLSchema(sample_hypergraph)
        result = schema.introspect_type("HyperNode")
        
        assert "data" in result
        assert "__type" in result["data"]
        
        type_data = result["data"]["__type"]
        assert type_data["name"] == "HyperNode"
        assert type_data["kind"] == "OBJECT"
        assert "fields" in type_data
        assert len(type_data["fields"]) > 0
    
    def test_type_introspection_nonexistent_type(self, sample_hypergraph):
        """Test __type introspection query for non-existent type."""
        schema = HyperGraphQLSchema(sample_hypergraph)
        result = schema.introspect_type("NonExistentType")
        
        assert "data" in result
        assert "__type" in result["data"]
        assert result["data"]["__type"] is None
    
    def test_idl_format_output(self, sample_hypergraph):
        """Test IDL format output."""
        schema = HyperGraphQLSchema(sample_hypergraph)
        idl = schema.to_idl()
        
        assert isinstance(idl, str)
        assert "type HyperGraph" in idl
        assert "type HyperNode" in idl
        assert "type HyperEdge" in idl
        assert "type Query" in idl
        
        # Check for field definitions
        assert "id: ID!" in idl
        assert "nodes: [HyperNode!]" in idl
        assert "edges: [HyperEdge!]" in idl


class TestHyperGraphQLServer:
    """Test HyperGraphQLServer functionality."""
    
    def test_server_creation(self, sample_hypergraph):
        """Test creating a GraphQL server."""
        server = HyperGraphQLServer(sample_hypergraph, port=8080)
        
        assert server.hypergraph == sample_hypergraph
        assert server.port == 8080
        assert isinstance(server.schema, HyperGraphQLSchema)
    
    def test_server_endpoint(self, sample_hypergraph):
        """Test server endpoint URL."""
        server = HyperGraphQLServer(sample_hypergraph, port=9000)
        endpoint = server.get_introspection_endpoint()
        
        assert endpoint == "http://localhost:9000/graphql"
    
    @pytest.mark.asyncio
    async def test_handle_schema_introspection(self, sample_hypergraph):
        """Test handling __schema introspection."""
        server = HyperGraphQLServer(sample_hypergraph)
        
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
        
        result = await server.handle_introspection(query)
        
        assert "data" in result
        assert "__schema" in result["data"]
        assert "types" in result["data"]["__schema"]
    
    @pytest.mark.asyncio
    async def test_handle_type_introspection(self, sample_hypergraph):
        """Test handling __type introspection."""
        server = HyperGraphQLServer(sample_hypergraph)
        
        query = """
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
        
        result = await server.handle_introspection(query)
        
        assert "data" in result
        assert "__type" in result["data"]
        assert result["data"]["__type"]["name"] == "HyperNode"
    
    @pytest.mark.asyncio
    async def test_serve_get_request(self, sample_hypergraph):
        """Test serving GET requests."""
        server = HyperGraphQLServer(sample_hypergraph)
        
        result = await server.serve_introspection("GET")
        
        assert result["status"] == 200
        assert result["method"] == "GET"
        assert "endpoint" in result
        assert "data" in result
    
    @pytest.mark.asyncio
    async def test_serve_idl_format(self, sample_hypergraph):
        """Test serving IDL format responses."""
        server = HyperGraphQLServer(sample_hypergraph)
        
        query = "__schema { types { name } }"
        result = await server.handle_introspection(query, "application/vnd.github.v4.idl")
        
        assert "data" in result
        assert isinstance(result["data"], str)
        assert "content_type" in result
        assert result["content_type"] == "text/plain"


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_hypergraphql_schema(self, sample_hypergraph):
        """Test create_hypergraphql_schema function."""
        schema = create_hypergraphql_schema(sample_hypergraph)
        
        assert isinstance(schema, HyperGraphQLSchema)
        assert schema.hypergraph == sample_hypergraph
    
    def test_create_hypergraphql_server(self, sample_hypergraph):
        """Test create_hypergraphql_server function."""
        server = create_hypergraphql_server(sample_hypergraph, port=8888)
        
        assert isinstance(server, HyperGraphQLServer)
        assert server.hypergraph == sample_hypergraph
        assert server.port == 8888
    
    @pytest.mark.asyncio
    async def test_introspect_hypergraph_schema_json(self, sample_hypergraph):
        """Test introspect_hypergraph_schema function with JSON output."""
        result = await introspect_hypergraph_schema(sample_hypergraph, "json")
        
        assert isinstance(result, dict)
        assert "data" in result
        assert "__schema" in result["data"]
    
    @pytest.mark.asyncio
    async def test_introspect_hypergraph_schema_idl(self, sample_hypergraph):
        """Test introspect_hypergraph_schema function with IDL output."""
        result = await introspect_hypergraph_schema(sample_hypergraph, "idl")
        
        assert isinstance(result, str)
        assert "type HyperGraph" in result
    
    @pytest.mark.asyncio
    async def test_introspect_hypergraph_type(self, sample_hypergraph):
        """Test introspect_hypergraph_type function."""
        result = await introspect_hypergraph_type(sample_hypergraph, "HyperEdge")
        
        assert isinstance(result, dict)
        assert "data" in result
        assert "__type" in result["data"]
        assert result["data"]["__type"]["name"] == "HyperEdge"


class TestGraphQLQueryParsing:
    """Test GraphQL query parsing and response formatting."""
    
    @pytest.mark.asyncio
    async def test_complex_schema_query(self, sample_hypergraph):
        """Test complex __schema query with nested fields."""
        server = HyperGraphQLServer(sample_hypergraph)
        
        query = """
        query IntrospectionQuery {
          __schema {
            queryType { name }
            mutationType { name }
            subscriptionType { name }
            types {
              ...FullType
            }
          }
        }
        
        fragment FullType on __Type {
          kind
          name
          description
          fields(includeDeprecated: true) {
            name
            description
            args {
              ...InputValue
            }
            type {
              ...TypeRef
            }
            isDeprecated
            deprecationReason
          }
        }
        
        fragment InputValue on __InputValue {
          name
          description
          type { ...TypeRef }
          defaultValue
        }
        
        fragment TypeRef on __Type {
          kind
          name
          ofType {
            kind
            name
            ofType {
              kind
              name
              ofType {
                kind
                name
                ofType {
                  kind
                  name
                  ofType {
                    kind
                    name
                    ofType {
                      kind
                      name
                      ofType {
                        kind
                        name
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """
        
        result = await server.handle_introspection(query)
        
        # Even with complex queries, we should get valid results
        assert "data" in result
        assert "__schema" in result["data"]
    
    @pytest.mark.asyncio
    async def test_repository_type_query(self, sample_hypergraph):
        """Test the specific __type query mentioned in the problem statement."""
        server = HyperGraphQLServer(sample_hypergraph)
        
        # This tests the exact query from the problem statement
        query = """
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
        
        result = await server.handle_introspection(query)
        
        assert "data" in result
        assert "__type" in result["data"]
        # Repository type doesn't exist in hypergraph, so should be null
        assert result["data"]["__type"] is None


if __name__ == "__main__":
    pytest.main([__file__])