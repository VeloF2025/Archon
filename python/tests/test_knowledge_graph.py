"""
Comprehensive tests for Knowledge Graph with Neo4j
Tests graph operations, ingestion, querying, and analysis
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import json
from typing import List, Dict, Any

from src.agents.knowledge_graph.graph_client import Neo4jClient, GraphNode, GraphRelationship
from src.agents.knowledge_graph.knowledge_ingestion import (
    KnowledgeIngestionPipeline,
    KnowledgeConcept,
    KnowledgeRelation
)
from src.agents.knowledge_graph.query_engine import GraphQueryEngine, QueryResult
from src.agents.knowledge_graph.relationship_mapper import RelationshipMapper
from src.agents.knowledge_graph.graph_analyzer import GraphAnalyzer


class TestGraphClient:
    """Test Neo4j graph client functionality"""
    
    @pytest.fixture
    async def client(self):
        """Create a mock Neo4j client"""
        with patch('src.agents.knowledge_graph.graph_client.AsyncGraphDatabase') as mock_db:
            client = Neo4jClient()
            client.driver = MagicMock()
            
            # Mock session
            mock_session = AsyncMock()
            client.driver.session.return_value.__aenter__.return_value = mock_session
            
            yield client
    
    @pytest.mark.asyncio
    async def test_connect_to_neo4j(self):
        """Test connecting to Neo4j database"""
        with patch('src.agents.knowledge_graph.graph_client.AsyncGraphDatabase') as mock_db:
            mock_driver = MagicMock()
            mock_db.driver.return_value = mock_driver
            mock_driver.verify_connectivity = AsyncMock()
            
            client = Neo4jClient()
            await client.connect()
            
            mock_db.driver.assert_called_once()
            mock_driver.verify_connectivity.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_node(self, client):
        """Test creating a node in the graph"""
        mock_session = client.driver.session.return_value.__aenter__.return_value
        mock_session.run = AsyncMock()
        
        # Mock the response
        mock_record = MagicMock()
        mock_record.__getitem__.return_value = {
            "id": "test-node-1",
            "name": "Test Node",
            "created_at": datetime.utcnow().isoformat()
        }
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=mock_record)
        mock_session.run.return_value = mock_result
        
        node = await client.create_node(
            labels=["Concept", "Test"],
            properties={"name": "Test Node", "description": "Test description"}
        )
        
        assert node.id == "test-node-1"
        assert node.labels == ["Concept", "Test"]
        assert node.properties["name"] == "Test Node"
        mock_session.run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_relationship(self, client):
        """Test creating a relationship between nodes"""
        mock_session = client.driver.session.return_value.__aenter__.return_value
        
        mock_record = MagicMock()
        mock_record.__getitem__.side_effect = lambda k: {
            "rel_id": 123,
            "r": {"strength": 0.8}
        }[k]
        
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=mock_record)
        mock_session.run = AsyncMock(return_value=mock_result)
        
        relationship = await client.create_relationship(
            source_id="node-1",
            target_id="node-2",
            relationship_type="DEPENDS_ON",
            properties={"strength": 0.8}
        )
        
        assert relationship.id == "123"
        assert relationship.type == "DEPENDS_ON"
        assert relationship.source_id == "node-1"
        assert relationship.target_id == "node-2"
    
    @pytest.mark.asyncio
    async def test_find_node(self, client):
        """Test finding a node by ID"""
        mock_session = client.driver.session.return_value.__aenter__.return_value
        
        mock_record = MagicMock()
        mock_record.__getitem__.side_effect = lambda k: {
            "n": {"id": "test-id", "name": "Test"},
            "labels": ["Concept"]
        }[k]
        
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=mock_record)
        mock_session.run = AsyncMock(return_value=mock_result)
        
        node = await client.find_node(node_id="test-id")
        
        assert node is not None
        assert node.id == "test-id"
        assert "Concept" in node.labels
    
    @pytest.mark.asyncio
    async def test_execute_cypher_query(self, client):
        """Test executing arbitrary Cypher queries"""
        mock_session = client.driver.session.return_value.__aenter__.return_value
        
        mock_records = [
            {"count": 42, "name": "Result1"},
            {"count": 24, "name": "Result2"}
        ]
        
        mock_result = AsyncMock()
        mock_result.__aiter__.return_value = (MagicMock(data=r) for r in mock_records)
        mock_session.run = AsyncMock(return_value=mock_result)
        
        results = await client.execute_cypher(
            "MATCH (n) RETURN count(n) as count, n.name as name",
            {"limit": 10}
        )
        
        assert len(results) == 2
        mock_session.run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_find_shortest_path(self, client):
        """Test finding shortest path between nodes"""
        mock_session = client.driver.session.return_value.__aenter__.return_value
        
        mock_record = MagicMock()
        mock_record.__getitem__.side_effect = lambda k: {
            "nodes": [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
            "relationships": [{"type": "CONNECTS"}]
        }[k]
        
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=mock_record)
        mock_session.run = AsyncMock(return_value=mock_result)
        
        path = await client.find_shortest_path("n1", "n3", max_depth=5)
        
        assert path is not None
        assert len(path["nodes"]) == 3
        assert path["nodes"][0]["id"] == "n1"
        assert path["nodes"][-1]["id"] == "n3"


class TestKnowledgeIngestion:
    """Test knowledge ingestion pipeline"""
    
    @pytest.fixture
    def pipeline(self):
        """Create an ingestion pipeline with mock client"""
        mock_client = MagicMock()
        pipeline = KnowledgeIngestionPipeline(mock_client)
        
        # Mock pattern detector
        with patch.object(pipeline, 'pattern_detector') as mock_detector:
            mock_detector.detect_patterns = AsyncMock(return_value=[])
            yield pipeline
    
    @pytest.mark.asyncio
    async def test_ingest_from_code(self, pipeline):
        """Test ingesting knowledge from source code"""
        code = '''
def calculate_sum(a: int, b: int) -> int:
    """Calculate sum of two numbers"""
    return a + b
'''
        
        # Mock the graph client methods
        pipeline.graph.create_node = AsyncMock(return_value=MagicMock(id="node-1"))
        pipeline.graph.find_node = AsyncMock(return_value=None)
        pipeline.graph.create_relationship = AsyncMock()
        
        # Mock embedding creation
        with patch('src.agents.knowledge_graph.knowledge_ingestion.create_embedding') as mock_embed:
            mock_embed.return_value = [0.1] * 1536
            
            result = await pipeline.ingest_from_code(
                code=code,
                language="python",
                source_file="test.py",
                project_id="test-project"
            )
        
        assert result["success"] is True
        assert "concepts_created" in result
        assert "relationships_created" in result
        assert result["source_file"] == "test.py"
    
    @pytest.mark.asyncio
    async def test_ingest_from_documentation(self, pipeline):
        """Test ingesting knowledge from documentation"""
        content = """
# API Documentation

This API provides REST endpoints for user management.

## Endpoints
- GET /users - List all users
- POST /users - Create a new user
- GET /users/{id} - Get user by ID
"""
        
        pipeline.graph.create_node = AsyncMock(return_value=MagicMock(id="doc-1"))
        pipeline.graph.find_node = AsyncMock(return_value=None)
        pipeline.graph.create_relationship = AsyncMock()
        
        with patch('src.agents.knowledge_graph.knowledge_ingestion.create_embedding') as mock_embed:
            mock_embed.return_value = [0.1] * 1536
            
            result = await pipeline.ingest_from_documentation(
                content=content,
                doc_type="API",
                source="api_docs.md",
                project_id="test-project"
            )
        
        assert result["success"] is True
        assert result["source"] == "api_docs.md"
    
    @pytest.mark.asyncio
    async def test_ingest_project_structure(self, pipeline):
        """Test ingesting project structure"""
        project_data = {
            "name": "TestProject",
            "description": "Test project description",
            "components": [
                {"name": "Frontend", "description": "React frontend"},
                {"name": "Backend", "description": "FastAPI backend"}
            ],
            "dependencies": [
                {"name": "react", "version": "18.0.0"},
                {"name": "fastapi", "version": "0.100.0"}
            ]
        }
        
        pipeline.graph.create_node = AsyncMock(return_value=MagicMock(id="proj-1"))
        pipeline.graph.find_node = AsyncMock(return_value=None)
        pipeline.graph.create_relationship = AsyncMock()
        
        with patch('src.agents.knowledge_graph.knowledge_ingestion.create_embedding') as mock_embed:
            mock_embed.return_value = [0.1] * 1536
            
            result = await pipeline.ingest_project_structure(
                project_id="test-project",
                project_data=project_data
            )
        
        assert result["success"] is True
        assert result["project_id"] == "test-project"
        
        # Should create nodes for project, components, and dependencies
        assert pipeline.graph.create_node.call_count >= 5  # 1 project + 2 components + 2 deps


class TestQueryEngine:
    """Test graph query engine functionality"""
    
    @pytest.fixture
    def query_engine(self):
        """Create a query engine with mock client"""
        mock_client = MagicMock()
        return GraphQueryEngine(mock_client)
    
    @pytest.mark.asyncio
    async def test_cypher_query(self, query_engine):
        """Test executing Cypher queries"""
        query_engine.graph.execute_cypher = AsyncMock(return_value=[
            {"name": "Node1", "count": 10},
            {"name": "Node2", "count": 20}
        ])
        
        result = await query_engine.query(
            query="MATCH (n) RETURN n.name as name, count(*) as count",
            query_type="cypher"
        )
        
        assert isinstance(result, QueryResult)
        assert result.query_type == "cypher"
        assert result.count == 2
        assert len(result.results) == 2
    
    @pytest.mark.asyncio
    async def test_natural_language_query(self, query_engine):
        """Test natural language to Cypher conversion"""
        query_engine.graph.execute_cypher = AsyncMock(return_value=[
            {"name": "Pattern1"},
            {"name": "Pattern2"}
        ])
        
        result = await query_engine.query(
            query="find patterns related to singleton",
            query_type="natural"
        )
        
        assert result.query_type == "natural"
        assert len(result.results) >= 0
        
        # Should convert to appropriate Cypher query
        query_engine.graph.execute_cypher.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, query_engine):
        """Test semantic search using embeddings"""
        query_engine.graph.execute_cypher = AsyncMock(return_value=[
            {"n": {"id": "1", "name": "Concept1", "embedding": [0.1] * 1536}},
            {"n": {"id": "2", "name": "Concept2", "embedding": [0.2] * 1536}}
        ])
        
        with patch('src.agents.knowledge_graph.query_engine.create_embedding') as mock_embed:
            mock_embed.return_value = [0.15] * 1536
            
            result = await query_engine.semantic_search(
                query_text="design patterns",
                limit=10,
                threshold=0.5
            )
        
        assert result.query_type == "semantic"
        assert isinstance(result.results, list)
    
    @pytest.mark.asyncio
    async def test_query_recommendations(self, query_engine):
        """Test getting query recommendations"""
        recommendations = await query_engine.recommend_queries(
            context="pattern analysis"
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all("query" in r for r in recommendations)
        assert any("pattern" in r["title"].lower() for r in recommendations)
    
    @pytest.mark.asyncio
    async def test_export_subgraph(self, query_engine):
        """Test exporting a subgraph"""
        query_engine.graph.execute_cypher = AsyncMock(return_value=[{
            "nodes": [{"id": "1", "name": "Node1"}],
            "relationships": [{"type": "CONNECTS"}]
        }])
        
        export_data = await query_engine.export_subgraph(
            center_node="node-1",
            depth=2,
            format="json"
        )
        
        assert "nodes" in export_data
        assert "relationships" in export_data


class TestRelationshipMapper:
    """Test relationship discovery and mapping"""
    
    @pytest.fixture
    def mapper(self):
        """Create a relationship mapper with mock client"""
        mock_client = MagicMock()
        return RelationshipMapper(mock_client)
    
    @pytest.mark.asyncio
    async def test_discover_relationships(self, mapper):
        """Test discovering relationships for a concept"""
        # Mock the graph client methods
        mapper.graph.find_node = AsyncMock(return_value=MagicMock(
            properties={"embedding": [0.1] * 1536, "name": "TestConcept"},
            labels=["Concept"]
        ))
        
        mapper.graph.execute_cypher = AsyncMock(return_value=[
            {"id": "2", "name": "Related1", "embedding": [0.11] * 1536},
            {"id": "3", "name": "Related2", "embedding": [0.09] * 1536}
        ])
        
        mapper.graph.get_node_neighbors = AsyncMock(return_value={
            "neighbors": [{"id": "4"}, {"id": "5"}]
        })
        
        relationships = await mapper.discover_relationships(
            concept_id="test-concept",
            max_depth=2,
            min_strength=0.5
        )
        
        assert isinstance(relationships, list)
        assert all("source_id" in r for r in relationships)
        assert all("target_id" in r for r in relationships)
        assert all("type" in r for r in relationships)
    
    @pytest.mark.asyncio
    async def test_create_discovered_relationships(self, mapper):
        """Test creating discovered relationships in the graph"""
        mapper.discovered_relationships = [
            {
                "source_id": "1",
                "target_id": "2",
                "type": "SIMILAR_TO",
                "strength": 0.8,
                "evidence": "High similarity"
            }
        ]
        
        mapper.graph.find_relationships = AsyncMock(return_value=[])
        mapper.graph.create_relationship = AsyncMock()
        
        result = await mapper.create_discovered_relationships(
            min_strength=0.5,
            dry_run=False
        )
        
        assert result["created"] == 1
        assert result["skipped"] == 0
        mapper.graph.create_relationship.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_relationship_patterns(self, mapper):
        """Test analyzing patterns in relationships"""
        mapper.graph.execute_cypher = AsyncMock(side_effect=[
            [{"avg_clustering": 0.7, "max_clustering": 0.9}],  # clustering
            [{"node_id": "hub1", "degree": 20}],  # centrality
            [{"category": "Pattern", "count": 10}],  # communities
            [{"node_id": "bottleneck1", "in_degree": 10, "out_degree": 1}]  # bottlenecks
        ])
        
        patterns = await mapper.analyze_relationship_patterns()
        
        assert "clustering" in patterns
        assert "centrality" in patterns
        assert "communities" in patterns
        assert "bottlenecks" in patterns


class TestGraphAnalyzer:
    """Test graph analysis functionality"""
    
    @pytest.fixture
    def analyzer(self):
        """Create a graph analyzer with mock client"""
        mock_client = MagicMock()
        return GraphAnalyzer(mock_client)
    
    @pytest.mark.asyncio
    async def test_analyze_graph_metrics(self, analyzer):
        """Test analyzing graph metrics"""
        analyzer.graph.get_statistics = AsyncMock(return_value={
            "node_count": 100,
            "relationship_count": 250,
            "label_counts": [{"labels": ["Concept"], "count": 50}]
        })
        
        analyzer.graph.execute_cypher = AsyncMock(return_value=[
            {"avg_degree": 5.0, "max_degree": 20}
        ])
        
        metrics = await analyzer.analyze_graph_metrics()
        
        assert "basic_stats" in metrics
        assert "degree_metrics" in metrics
        assert metrics["basic_stats"]["node_count"] == 100
    
    @pytest.mark.asyncio
    async def test_detect_communities(self, analyzer):
        """Test community detection"""
        analyzer.graph.execute_cypher = AsyncMock(return_value=[
            {"community_id": 1, "nodes": 20},
            {"community_id": 2, "nodes": 15}
        ])
        
        communities = await analyzer.detect_communities()
        
        assert "communities" in communities
        assert "count" in communities
        assert communities["count"] >= 0
    
    @pytest.mark.asyncio
    async def test_find_anomalies(self, analyzer):
        """Test finding anomalies in the graph"""
        analyzer.graph.execute_cypher = AsyncMock(side_effect=[
            [{"id": "orphan1", "name": "OrphanNode"}],  # orphans
            [{"id": "hub1", "degree": 100}],  # hubs
            [],  # no duplicate relationships
            []  # no self-loops
        ])
        
        anomalies = await analyzer.find_anomalies()
        
        assert "orphan_nodes" in anomalies
        assert "hub_nodes" in anomalies
        assert len(anomalies["orphan_nodes"]) == 1
    
    @pytest.mark.asyncio  
    async def test_generate_insights(self, analyzer):
        """Test generating graph insights"""
        # Mock all the analysis methods
        analyzer.analyze_graph_metrics = AsyncMock(return_value={
            "basic_stats": {"node_count": 100}
        })
        analyzer.analyze_patterns = AsyncMock(return_value={
            "most_common": ["Pattern1"]
        })
        analyzer.detect_communities = AsyncMock(return_value={
            "count": 5
        })
        analyzer.find_anomalies = AsyncMock(return_value={
            "orphan_nodes": []
        })
        analyzer.analyze_trends = AsyncMock(return_value={
            "growth_rate": 0.1
        })
        
        insights = await analyzer.generate_insights()
        
        assert "summary" in insights
        assert "metrics" in insights
        assert "patterns" in insights
        assert "recommendations" in insights
        assert isinstance(insights["recommendations"], list)


@pytest.mark.integration
class TestKnowledgeGraphIntegration:
    """Integration tests for the complete knowledge graph system"""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("NEO4J_URI"), reason="Neo4j not available")
    async def test_full_knowledge_pipeline(self):
        """Test the complete knowledge graph pipeline"""
        # This would connect to a real Neo4j instance for integration testing
        
        client = Neo4jClient()
        await client.connect()
        
        try:
            # Create a test node
            node = await client.create_node(
                labels=["IntegrationTest"],
                properties={"name": "TestNode", "timestamp": datetime.utcnow().isoformat()}
            )
            assert node.id is not None
            
            # Ingest some knowledge
            pipeline = KnowledgeIngestionPipeline(client)
            result = await pipeline.ingest_from_code(
                code="def test(): pass",
                language="python",
                source_file="test.py"
            )
            assert result["success"] is True
            
            # Query the graph
            query_engine = GraphQueryEngine(client)
            query_result = await query_engine.query(
                "MATCH (n:IntegrationTest) RETURN n",
                query_type="cypher"
            )
            assert query_result.count >= 1
            
            # Clean up
            await client.delete_node(node.id)
            
        finally:
            await client.close()