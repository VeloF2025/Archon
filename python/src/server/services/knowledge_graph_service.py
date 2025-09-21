"""
Knowledge Graph Service
Manages all knowledge graph operations and integrations
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import os

from ...agents.knowledge_graph.graph_client import Neo4jClient, GraphNode, GraphRelationship
from ...agents.knowledge_graph.knowledge_ingestion import KnowledgeIngestionPipeline
from ...agents.knowledge_graph.relationship_mapper import RelationshipMapper
from ...agents.knowledge_graph.graph_analyzer import GraphAnalyzer
from ...agents.knowledge_graph.query_engine import GraphQueryEngine
from ..services.embeddings import create_embedding

logger = logging.getLogger(__name__)


class KnowledgeGraphService:
    """Service for managing knowledge graph operations"""
    
    def __init__(self):
        self.graph_client = None
        self.ingestion_pipeline = None
        self.relationship_mapper = None
        self.graph_analyzer = None
        self.query_engine = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the knowledge graph service"""
        if self._initialized:
            return
        
        try:
            # Initialize Neo4j client
            self.graph_client = Neo4jClient(
                uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                username=os.getenv("NEO4J_USERNAME", "neo4j"),
                password=os.getenv("NEO4J_PASSWORD", "archon2025secure")
            )
            await self.graph_client.connect()
            
            # Initialize components
            self.ingestion_pipeline = KnowledgeIngestionPipeline(self.graph_client)
            self.relationship_mapper = RelationshipMapper(self.graph_client)
            self.graph_analyzer = GraphAnalyzer(self.graph_client)
            self.query_engine = GraphQueryEngine(self.graph_client)
            
            self._initialized = True
            logger.info("Knowledge Graph Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Knowledge Graph Service: {e}")
            raise
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def close(self):
        """Close the knowledge graph service"""
        if self.graph_client:
            await self.graph_client.close()
        self._initialized = False
    
    async def create_node(
        self,
        labels: List[str],
        properties: Dict[str, Any],
        generate_embedding: bool = True
    ) -> GraphNode:
        """Create a node in the knowledge graph"""
        await self.initialize()
        
        # Generate embedding if requested
        if generate_embedding and "name" in properties:
            embedding_text = f"{properties.get('name')} {' '.join(labels)} {properties.get('description', '')}"
            properties["embedding"] = await create_embedding(embedding_text)
        
        return await self.graph_client.create_node(labels, properties)
    
    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
        strength: float = 1.0
    ) -> GraphRelationship:
        """Create a relationship between nodes"""
        await self.initialize()
        
        props = properties or {}
        props["strength"] = strength
        
        return await self.graph_client.create_relationship(
            source_id, target_id, relationship_type, props
        )
    
    async def query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        query_type: str = "cypher"
    ):
        """Execute a query on the knowledge graph"""
        await self.initialize()
        
        if query_type == "semantic":
            return await self.query_engine.semantic_search(query)
        else:
            return await self.query_engine.query(query, parameters, query_type)
    
    async def ingest_code(
        self,
        code: str,
        language: str,
        source_file: str,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ingest code into the knowledge graph"""
        await self.initialize()
        
        return await self.ingestion_pipeline.ingest_from_code(
            code, language, source_file, project_id
        )
    
    async def ingest_documentation(
        self,
        content: str,
        doc_type: str,
        source: str,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ingest documentation into the knowledge graph"""
        await self.initialize()
        
        return await self.ingestion_pipeline.ingest_from_documentation(
            content, doc_type, source, project_id
        )
    
    async def discover_relationships(
        self,
        concept_id: str,
        max_depth: int = 2,
        min_strength: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Discover relationships for a concept"""
        await self.initialize()
        
        return await self.relationship_mapper.discover_relationships(
            concept_id, max_depth, min_strength
        )
    
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID"""
        await self.initialize()
        
        node = await self.graph_client.find_node(node_id=node_id)
        if node:
            return {
                "id": node.id,
                "labels": node.labels,
                "properties": node.properties
            }
        return None
    
    async def get_neighbors(
        self,
        node_id: str,
        depth: int = 1
    ) -> Dict[str, Any]:
        """Get neighboring nodes"""
        await self.initialize()
        
        return await self.graph_client.get_node_neighbors(node_id, depth)
    
    async def search_nodes(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search for nodes"""
        await self.initialize()
        
        labels = [category] if category else None
        nodes = await self.graph_client.find_nodes(
            labels=labels,
            properties={"name": query} if query else None,
            limit=limit
        )
        
        return [
            {
                "id": node.id,
                "labels": node.labels,
                "properties": node.properties
            }
            for node in nodes
        ]
    
    async def semantic_search(
        self,
        query: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Perform semantic search"""
        await self.initialize()
        
        result = await self.query_engine.semantic_search(query, limit)
        return result.results
    
    async def find_shortest_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> Optional[Dict[str, Any]]:
        """Find shortest path between nodes"""
        await self.initialize()
        
        return await self.graph_client.find_shortest_path(
            source_id, target_id, max_depth
        )
    
    async def get_visualization_data(
        self,
        center_node_id: Optional[str] = None,
        depth: int = 2,
        max_nodes: int = 100,
        include_orphans: bool = False
    ) -> Dict[str, Any]:
        """Get graph data for visualization"""
        await self.initialize()
        
        if center_node_id:
            # Get subgraph around center node
            query = f"""
                MATCH (center {{id: $center_id}})
                MATCH path = (center)-[*0..{depth}]-(neighbor)
                WITH collect(DISTINCT neighbor) as nodes,
                     collect(DISTINCT relationships(path)) as rels_nested
                UNWIND rels_nested as rels_list
                UNWIND rels_list as rel
                WITH nodes, collect(DISTINCT rel) as relationships
                RETURN nodes, relationships
                LIMIT $limit
            """
            params = {"center_id": center_node_id, "limit": max_nodes}
        else:
            # Get general graph sample
            query = """
                MATCH (n)
                WITH n LIMIT $limit
                OPTIONAL MATCH (n)-[r]-(m)
                WHERE m IN nodes
                RETURN collect(DISTINCT n) as nodes, collect(DISTINCT r) as relationships
            """
            params = {"limit": max_nodes}
        
        result = await self.graph_client.execute_cypher(query, params)
        
        if not result:
            return {"nodes": [], "edges": []}
        
        nodes = result[0].get("nodes", [])
        relationships = result[0].get("relationships", [])
        
        # Format for visualization
        vis_nodes = [
            {
                "id": node.get("id"),
                "label": node.get("name", node.get("id")),
                "category": node.get("category", "default"),
                "properties": dict(node)
            }
            for node in nodes if node
        ]
        
        vis_edges = []
        for rel in relationships:
            if rel and hasattr(rel, '__dict__'):
                vis_edges.append({
                    "source": rel.start_node.id if hasattr(rel, 'start_node') else None,
                    "target": rel.end_node.id if hasattr(rel, 'end_node') else None,
                    "type": rel.type if hasattr(rel, 'type') else "RELATED",
                    "properties": dict(rel) if rel else {}
                })
        
        return {
            "nodes": vis_nodes,
            "edges": vis_edges
        }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        await self.initialize()
        
        graph_stats = await self.graph_client.get_statistics()
        ingestion_stats = await self.ingestion_pipeline.get_ingestion_statistics()
        
        return {
            "graph": graph_stats,
            "ingestion": ingestion_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def analyze_graph(
        self,
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Analyze the graph structure"""
        await self.initialize()
        
        if analysis_type == "patterns":
            return await self.relationship_mapper.analyze_relationship_patterns()
        elif analysis_type == "centrality":
            return await self.graph_analyzer.analyze_graph_metrics()
        elif analysis_type == "communities":
            return await self.graph_analyzer.detect_communities()
        else:  # comprehensive
            return await self.graph_analyzer.generate_insights()
    
    async def get_query_recommendations(
        self,
        context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Get query recommendations"""
        await self.initialize()
        
        return await self.query_engine.recommend_queries(context)
    
    async def export_subgraph(
        self,
        center_node_id: str,
        depth: int = 2,
        format: str = "json"
    ) -> Dict[str, Any]:
        """Export a subgraph"""
        await self.initialize()
        
        return await self.query_engine.export_subgraph(
            center_node_id, depth, format
        )
    
    async def delete_node(self, node_id: str) -> bool:
        """Delete a node"""
        await self.initialize()
        
        return await self.graph_client.delete_node(node_id)
    
    async def health_check(self) -> Dict[str, bool]:
        """Check service health"""
        health = {
            "neo4j": False,
            "kafka": False,
            "redis": False
        }
        
        try:
            # Check Neo4j
            await self.initialize()
            test_result = await self.graph_client.execute_cypher("RETURN 1 as test")
            health["neo4j"] = bool(test_result)
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
        
        # Kafka and Redis checks would go here
        # For now, we'll assume they're healthy if Neo4j is up
        if health["neo4j"]:
            health["kafka"] = True
            health["redis"] = True
        
        return health


# Dependency injection function
def get_knowledge_graph_service():
    """Get knowledge graph service instance for dependency injection"""
    return KnowledgeGraphService()