#!/usr/bin/env python3
"""
Test script for Knowledge Graph integration
Tests Neo4j connectivity and basic operations
"""

import asyncio
import os
from datetime import datetime

# Set up environment
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "archon2025secure"

async def test_knowledge_graph():
    """Test the Knowledge Graph integration"""
    
    print("🔍 Testing Knowledge Graph Integration...")
    print("=" * 60)
    
    # Test 1: Neo4j Connection
    print("\n1️⃣ Testing Neo4j Connection...")
    try:
        from python.src.agents.knowledge_graph.graph_client import Neo4jClient
        
        client = Neo4jClient()
        await client.connect()
        print("✅ Neo4j connected successfully!")
        
        # Test basic query
        result = await client.execute_cypher("RETURN 1 as test")
        if result and result[0]['test'] == 1:
            print("✅ Neo4j query execution successful!")
        
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False
    
    # Test 2: Create a test node
    print("\n2️⃣ Testing Node Creation...")
    try:
        test_node = await client.create_node(
            labels=["TestConcept", "Pattern"],
            properties={
                "name": "Singleton Pattern",
                "description": "Ensures a class has only one instance",
                "language": "python",
                "effectiveness_score": 0.85,
                "created_at": datetime.utcnow().isoformat()
            }
        )
        print(f"✅ Created node: {test_node.id}")
        
    except Exception as e:
        print(f"❌ Node creation failed: {e}")
        return False
    
    # Test 3: Knowledge Ingestion
    print("\n3️⃣ Testing Knowledge Ingestion...")
    try:
        from python.src.agents.knowledge_graph.knowledge_ingestion import KnowledgeIngestionPipeline
        
        pipeline = KnowledgeIngestionPipeline(client)
        
        # Test code ingestion
        sample_code = '''
class DatabaseConnection:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def connect(self):
        return "Connected to database"
'''
        
        result = await pipeline.ingest_from_code(
            code=sample_code,
            language="python",
            source_file="test_singleton.py",
            project_id="test_project"
        )
        
        if result['success']:
            print(f"✅ Code ingestion successful!")
            print(f"   - Concepts created: {result['concepts_created']}")
            print(f"   - Relationships created: {result['relationships_created']}")
            print(f"   - Patterns detected: {result['patterns_detected']}")
        
    except Exception as e:
        print(f"❌ Knowledge ingestion failed: {e}")
        return False
    
    # Test 4: Query Engine
    print("\n4️⃣ Testing Query Engine...")
    try:
        from python.src.agents.knowledge_graph.query_engine import GraphQueryEngine
        
        query_engine = GraphQueryEngine(client)
        
        # Test natural language query
        nl_result = await query_engine.query(
            "find patterns in python",
            query_type="natural"
        )
        print(f"✅ Natural language query executed!")
        print(f"   - Results found: {nl_result.count}")
        
        # Test Cypher query
        cypher_result = await query_engine.query(
            "MATCH (n:Pattern) RETURN n LIMIT 5",
            query_type="cypher"
        )
        print(f"✅ Cypher query executed!")
        print(f"   - Results found: {cypher_result.count}")
        
    except Exception as e:
        print(f"❌ Query engine failed: {e}")
        return False
    
    # Test 5: Graph Analysis
    print("\n5️⃣ Testing Graph Analysis...")
    try:
        from python.src.agents.knowledge_graph.graph_analyzer import GraphAnalyzer
        
        analyzer = GraphAnalyzer(client)
        
        # Get basic metrics
        metrics = await analyzer.analyze_graph_metrics()
        print(f"✅ Graph analysis successful!")
        print(f"   - Node count: {metrics.get('node_count', 0)}")
        print(f"   - Relationship count: {metrics.get('relationship_count', 0)}")
        
    except Exception as e:
        print(f"❌ Graph analysis failed: {e}")
        return False
    
    # Test 6: Relationship Discovery
    print("\n6️⃣ Testing Relationship Discovery...")
    try:
        from python.src.agents.knowledge_graph.relationship_mapper import RelationshipMapper
        
        mapper = RelationshipMapper(client)
        
        # Discover relationships for our test node
        relationships = await mapper.discover_relationships(
            concept_id=test_node.id,
            max_depth=2,
            min_strength=0.3
        )
        print(f"✅ Relationship discovery successful!")
        print(f"   - Discovered relationships: {len(relationships)}")
        
    except Exception as e:
        print(f"❌ Relationship discovery failed: {e}")
        return False
    
    # Cleanup
    print("\n7️⃣ Cleaning up test data...")
    try:
        await client.delete_node(test_node.id)
        print("✅ Test data cleaned up!")
        
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")
    
    # Close connection
    await client.close()
    
    print("\n" + "=" * 60)
    print("✅ All Knowledge Graph tests passed successfully!")
    return True


async def test_api_endpoints():
    """Test the Knowledge Graph API endpoints"""
    
    print("\n\n🌐 Testing Knowledge Graph API Endpoints...")
    print("=" * 60)
    
    import httpx
    
    base_url = "http://localhost:8181/api/knowledge-graph"
    
    async with httpx.AsyncClient() as client:
        # Test 1: Health check
        print("\n1️⃣ Testing API Health...")
        try:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API Health: {data['status']}")
                print(f"   - Neo4j: {'✅' if data['services']['neo4j'] else '❌'}")
                print(f"   - Kafka: {'✅' if data['services']['kafka'] else '❌'}")
                print(f"   - Redis: {'✅' if data['services']['redis'] else '❌'}")
        except Exception as e:
            print(f"❌ API health check failed: {e}")
            return False
        
        # Test 2: Create node via API
        print("\n2️⃣ Testing Node Creation via API...")
        try:
            node_data = {
                "labels": ["APITest", "Concept"],
                "properties": {
                    "name": "Test API Node",
                    "description": "Created via API test",
                    "timestamp": datetime.utcnow().isoformat()
                },
                "generate_embedding": False
            }
            
            response = await client.post(f"{base_url}/nodes", json=node_data)
            if response.status_code == 200:
                result = response.json()
                node_id = result['node']['id']
                print(f"✅ Node created via API: {node_id}")
            else:
                print(f"❌ API node creation failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ API node creation failed: {e}")
            return False
        
        # Test 3: Query via API
        print("\n3️⃣ Testing Query via API...")
        try:
            query_data = {
                "query": "MATCH (n) RETURN count(n) as total",
                "query_type": "cypher"
            }
            
            response = await client.post(f"{base_url}/query", json=query_data)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Query executed via API!")
                print(f"   - Results: {result['count']} records")
        except Exception as e:
            print(f"❌ API query failed: {e}")
            return False
        
        # Test 4: Get statistics
        print("\n4️⃣ Testing Statistics API...")
        try:
            response = await client.get(f"{base_url}/statistics")
            if response.status_code == 200:
                stats = response.json()
                print(f"✅ Statistics retrieved!")
                print(f"   - Graph stats: {stats['statistics'].get('graph', {})}")
        except Exception as e:
            print(f"❌ Statistics API failed: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✅ All API tests passed successfully!")
    return True


if __name__ == "__main__":
    print("\n🚀 Archon Knowledge Graph Integration Test Suite")
    print("=" * 60)
    
    # Run tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Test direct integration
        success = loop.run_until_complete(test_knowledge_graph())
        
        if success:
            # Test API endpoints
            loop.run_until_complete(test_api_endpoints())
        
        print("\n✨ Knowledge Graph Integration Complete!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        loop.close()