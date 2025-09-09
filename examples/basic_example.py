"""
Basic example demonstrating AtomBot functionality.

This example shows how to create an AtomSpace with AtomBot agents,
have them chat, and demonstrate their hybrid capabilities.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables (for API keys)
load_dotenv()

from atombot import AtomSpace, ConceptNode, PredicateNode, EvaluationLink, InheritanceLink
from atombot.core.value import TruthValue, StringValue


async def basic_example():
    """Basic example of AtomBot functionality."""
    print("🤖 AtomBot Basic Example")
    print("=" * 40)
    
    # Create an AtomSpace
    atomspace = AtomSpace("BasicExample")
    print(f"Created AtomSpace: {atomspace}")
    
    # Create some concept agents
    print("\n📝 Creating Concept Agents...")
    product_agent = ConceptNode("Product Catalog", atomspace)
    customer_agent = ConceptNode("Customer Service", atomspace)
    snowboard_agent = ConceptNode("Snowboard", atomspace)
    
    print(f"  • {product_agent}")
    print(f"  • {customer_agent}")
    print(f"  • {snowboard_agent}")
    
    # Create a predicate agent
    print("\n🔗 Creating Predicate Agent...")
    has_property = PredicateNode("has_property", atomspace)
    print(f"  • {has_property}")
    
    # Create relationships
    print("\n🌐 Creating Relationships...")
    
    # Inheritance: Snowboard is a Product
    inheritance = InheritanceLink(snowboard_agent, product_agent, atomspace)
    print(f"  • {inheritance}")
    
    # Evaluation: Snowboard has_property "winter_sport_equipment"
    evaluation = EvaluationLink(
        has_property, 
        [snowboard_agent, ConceptNode("winter_sport_equipment", atomspace)],
        atomspace,
        TruthValue(0.95, 0.9)
    )
    print(f"  • {evaluation}")
    
    # Show atomspace statistics
    print(f"\n📊 AtomSpace Statistics:")
    stats = atomspace.get_statistics()
    for key, value in stats.items():
        print(f"  • {key}: {value}")
    
    # Test chat capabilities
    print("\n💬 Testing Chat Capabilities...")
    
    # Chat with product agent
    print(f"\n🗣️  Chatting with {product_agent.name}:")
    try:
        response = await product_agent.chat("What snowboards do you have in your catalog?")
        print(f"  Response: {response}")
    except Exception as e:
        print(f"  Note: {e} (API key may not be configured)")
    
    # Chat with customer service agent
    print(f"\n🗣️  Chatting with {customer_agent.name}:")
    try:
        response = await customer_agent.chat("Can you tell me about winter sports equipment?")
        print(f"  Response: {response}")
    except Exception as e:
        print(f"  Note: {e} (API key may not be configured)")
    
    # Demonstrate tool usage
    print("\n🔧 Testing MCP Tools...")
    
    # Search knowledge
    search_result = await product_agent.mcp_client.call_tool(
        "search_my_knowledge", 
        {"query": "snowboard"}
    )
    print(f"  Search result: {search_result}")
    
    # Query relationships
    relationships = await has_property.mcp_client.call_tool(
        "find_relationships",
        {"entity": "snowboard"}
    )
    print(f"  Relationships: {relationships}")
    
    # Demonstrate value propagation
    print("\n📡 Testing Value Propagation...")
    
    # Set a value on snowboard agent
    snowboard_agent.set_value("popularity", StringValue("high"))
    
    # Propagate through network
    await atomspace.propagate_values(snowboard_agent, "popularity", max_hops=2)
    print("  Values propagated through network!")
    
    # Check what values other agents received
    for agent in [product_agent, customer_agent]:
        popularity = agent.get_value("popularity")
        if popularity:
            print(f"  • {agent.name} received popularity: {popularity}")
    
    # Demonstrate agent collaboration
    print("\n🤝 Testing Agent Collaboration...")
    
    try:
        collaboration_result = await product_agent.collaborate_with_network(
            "Find information about winter sports equipment"
        )
        print(f"  Collaboration result: {collaboration_result}")
    except Exception as e:
        print(f"  Collaboration note: {e}")
    
    # Show final status
    print("\n📋 Final Agent Status:")
    for agent in [product_agent, customer_agent, snowboard_agent, has_property]:
        if hasattr(agent, 'get_atombot_status'):
            status = agent.get_atombot_status()
            print(f"  • {agent.name}: {status['network_info']['total_neighbors']} connections, "
                  f"{status['agent_info']['interactions']} interactions")


async def pattern_matching_example():
    """Example demonstrating pattern matching capabilities."""
    print("\n🔍 Pattern Matching Example")
    print("=" * 40)
    
    atomspace = AtomSpace("PatternExample")
    
    # Create a small knowledge network
    animals = [
        ConceptNode("Dog", atomspace),
        ConceptNode("Cat", atomspace), 
        ConceptNode("Bird", atomspace),
        ConceptNode("Animal", atomspace)
    ]
    
    # Create inheritance relationships
    for animal in animals[:-1]:  # All except "Animal"
        InheritanceLink(animal, animals[-1], atomspace)  # -> Animal
    
    # Query patterns
    print("🔍 Querying patterns...")
    
    # Find all concept nodes
    concepts = atomspace.query({"type": "ConceptNode"})
    print(f"  Found {len(concepts)} concept nodes")
    
    # Find animals (by name pattern)
    animal_pattern = atomspace.query({
        "type": "ConceptNode",
        "name": "regex:.*[Aa]nimal.*"
    })
    print(f"  Found {len(animal_pattern)} nodes matching 'animal' pattern")
    
    # Find inheritance links
    inheritance_links = atomspace.query({"type": "InheritanceLink"})
    print(f"  Found {len(inheritance_links)} inheritance relationships")
    
    # Custom predicate search
    def is_pet(atom):
        return hasattr(atom, 'name') and atom.name.lower() in ['dog', 'cat']
    
    pets = atomspace.query({"predicate": is_pet})
    print(f"  Found {len(pets)} pets using custom predicate")


if __name__ == "__main__":
    print("🚀 Starting AtomBot Examples")
    
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Note: ANTHROPIC_API_KEY not found. Chat features will use fallback responses.")
        print("   To enable full chat capabilities, set your Anthropic API key in .env file")
    
    # Run examples
    asyncio.run(basic_example())
    asyncio.run(pattern_matching_example())
    
    print("\n✅ Examples completed!")
    print("\nTo explore more:")
    print("  1. Set ANTHROPIC_API_KEY for full chat capabilities")
    print("  2. Check out other examples in the examples/ directory")
    print("  3. Run tests with: python -m pytest tests/")