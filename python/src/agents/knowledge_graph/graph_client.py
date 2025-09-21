"""
Neo4j Graph Database Client
Handles connections and operations with Neo4j
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import os
from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import Neo4jError
from pydantic import BaseModel, Field
import asyncio

logger = logging.getLogger(__name__)


class GraphNode(BaseModel):
    """Represents a node in the knowledge graph"""
    id: str = Field(description="Unique node identifier")
    labels: List[str] = Field(description="Node labels/types")
    properties: Dict[str, Any] = Field(description="Node properties")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class GraphRelationship(BaseModel):
    """Represents a relationship between nodes"""
    id: str = Field(description="Unique relationship identifier")
    type: str = Field(description="Relationship type")
    source_id: str = Field(description="Source node ID")
    target_id: str = Field(description="Target node ID")
    properties: Dict[str, Any] = Field(default_factory=dict)
    strength: float = Field(default=1.0, description="Relationship strength")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class Neo4jClient:
    """Async client for Neo4j graph database operations"""
    
    def __init__(
        self,
        uri: str = None,
        username: str = None,
        password: str = None
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "archon2025secure")
        self.driver: Optional[AsyncDriver] = None
        
    async def connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=3600
            )
            await self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
            
            # Initialize schema
            await self._initialize_schema()
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    async def close(self):
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")
    
    async def _initialize_schema(self):
        """Initialize graph schema with constraints and indexes"""
        async with self.driver.session() as session:
            queries = [
                # Constraints for unique IDs
                "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT pattern_id IF NOT EXISTS FOR (p:Pattern) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT project_id IF NOT EXISTS FOR (pr:Project) REQUIRE pr.id IS UNIQUE",
                "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT task_id IF NOT EXISTS FOR (t:Task) REQUIRE t.id IS UNIQUE",
                
                # Indexes for common queries
                "CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)",
                "CREATE INDEX concept_category IF NOT EXISTS FOR (c:Concept) ON (c.category)",
                "CREATE INDEX pattern_language IF NOT EXISTS FOR (p:Pattern) ON (p.language)",
                "CREATE INDEX document_type IF NOT EXISTS FOR (d:Document) ON (d.type)",
                "CREATE FULLTEXT INDEX concept_search IF NOT EXISTS FOR (c:Concept) ON EACH [c.name, c.description]",
            ]
            
            for query in queries:
                try:
                    await session.run(query)
                except Neo4jError as e:
                    if "already exists" not in str(e):
                        logger.warning(f"Schema query warning: {e}")
            
            logger.info("Neo4j schema initialized")
    
    async def create_node(
        self,
        labels: List[str],
        properties: Dict[str, Any]
    ) -> GraphNode:
        """
        Create a node in the graph
        
        Args:
            labels: Node labels (types)
            properties: Node properties
            
        Returns:
            Created node
        """
        try:
            # Ensure ID exists
            if "id" not in properties:
                properties["id"] = f"{labels[0].lower()}_{datetime.utcnow().timestamp()}"
            
            # Add timestamps
            properties["created_at"] = datetime.utcnow().isoformat()
            properties["updated_at"] = properties["created_at"]
            
            # Build query
            labels_str = ":".join(labels)
            query = f"""
                CREATE (n:{labels_str})
                SET n = $properties
                RETURN n
            """
            
            async with self.driver.session() as session:
                result = await session.run(query, properties=properties)
                record = await result.single()
                
                if record:
                    node_data = dict(record["n"])
                    return GraphNode(
                        id=node_data["id"],
                        labels=labels,
                        properties=node_data
                    )
                
                raise Exception("Failed to create node")
                
        except Exception as e:
            logger.error(f"Error creating node: {e}")
            raise
    
    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> GraphRelationship:
        """
        Create a relationship between nodes
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_type: Type of relationship
            properties: Relationship properties
            
        Returns:
            Created relationship
        """
        try:
            props = properties or {}
            props["created_at"] = datetime.utcnow().isoformat()
            
            query = f"""
                MATCH (source {{id: $source_id}})
                MATCH (target {{id: $target_id}})
                CREATE (source)-[r:{relationship_type}]->(target)
                SET r = $properties
                SET r.id = id(r)
                RETURN r, id(r) as rel_id
            """
            
            async with self.driver.session() as session:
                result = await session.run(
                    query,
                    source_id=source_id,
                    target_id=target_id,
                    properties=props
                )
                record = await result.single()
                
                if record:
                    return GraphRelationship(
                        id=str(record["rel_id"]),
                        type=relationship_type,
                        source_id=source_id,
                        target_id=target_id,
                        properties=dict(record["r"]),
                        strength=props.get("strength", 1.0)
                    )
                
                raise Exception(f"Failed to create relationship between {source_id} and {target_id}")
                
        except Exception as e:
            logger.error(f"Error creating relationship: {e}")
            raise
    
    async def find_node(
        self,
        node_id: Optional[str] = None,
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> Optional[GraphNode]:
        """Find a node by ID, labels, or properties"""
        try:
            # Build query
            conditions = []
            params = {}
            
            if node_id:
                conditions.append("n.id = $node_id")
                params["node_id"] = node_id
            
            if labels:
                labels_str = ":".join(labels)
                query_start = f"MATCH (n:{labels_str})"
            else:
                query_start = "MATCH (n)"
            
            if properties:
                for key, value in properties.items():
                    conditions.append(f"n.{key} = ${key}")
                    params[key] = value
            
            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            query = f"{query_start} {where_clause} RETURN n, labels(n) as labels LIMIT 1"
            
            async with self.driver.session() as session:
                result = await session.run(query, **params)
                record = await result.single()
                
                if record:
                    return GraphNode(
                        id=record["n"]["id"],
                        labels=record["labels"],
                        properties=dict(record["n"])
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Error finding node: {e}")
            return None
    
    async def find_nodes(
        self,
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[GraphNode]:
        """Find multiple nodes matching criteria"""
        try:
            conditions = []
            params = {"limit": limit}
            
            if labels:
                labels_str = ":".join(labels)
                query_start = f"MATCH (n:{labels_str})"
            else:
                query_start = "MATCH (n)"
            
            if properties:
                for key, value in properties.items():
                    conditions.append(f"n.{key} = ${key}")
                    params[key] = value
            
            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            query = f"{query_start} {where_clause} RETURN n, labels(n) as labels LIMIT $limit"
            
            async with self.driver.session() as session:
                result = await session.run(query, **params)
                nodes = []
                
                async for record in result:
                    nodes.append(GraphNode(
                        id=record["n"]["id"],
                        labels=record["labels"],
                        properties=dict(record["n"])
                    ))
                
                return nodes
                
        except Exception as e:
            logger.error(f"Error finding nodes: {e}")
            return []
    
    async def find_relationships(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relationship_type: Optional[str] = None,
        limit: int = 100
    ) -> List[GraphRelationship]:
        """Find relationships matching criteria"""
        try:
            conditions = []
            params = {"limit": limit}
            
            query_parts = ["MATCH (source)"]
            
            if source_id:
                conditions.append("source.id = $source_id")
                params["source_id"] = source_id
            
            if relationship_type:
                query_parts.append(f"-[r:{relationship_type}]->")
            else:
                query_parts.append("-[r]->")
            
            query_parts.append("(target)")
            
            if target_id:
                conditions.append("target.id = $target_id")
                params["target_id"] = target_id
            
            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            query = f"{' '.join(query_parts)} {where_clause} RETURN r, type(r) as type, source.id as source_id, target.id as target_id LIMIT $limit"
            
            async with self.driver.session() as session:
                result = await session.run(query, **params)
                relationships = []
                
                async for record in result:
                    relationships.append(GraphRelationship(
                        id=str(record["r"].id),
                        type=record["type"],
                        source_id=record["source_id"],
                        target_id=record["target_id"],
                        properties=dict(record["r"])
                    ))
                
                return relationships
                
        except Exception as e:
            logger.error(f"Error finding relationships: {e}")
            return []
    
    async def update_node(
        self,
        node_id: str,
        properties: Dict[str, Any]
    ) -> bool:
        """Update node properties"""
        try:
            properties["updated_at"] = datetime.utcnow().isoformat()
            
            query = """
                MATCH (n {id: $node_id})
                SET n += $properties
                RETURN n
            """
            
            async with self.driver.session() as session:
                result = await session.run(
                    query,
                    node_id=node_id,
                    properties=properties
                )
                record = await result.single()
                return record is not None
                
        except Exception as e:
            logger.error(f"Error updating node: {e}")
            return False
    
    async def delete_node(self, node_id: str) -> bool:
        """Delete a node and its relationships"""
        try:
            query = """
                MATCH (n {id: $node_id})
                DETACH DELETE n
                RETURN count(n) as deleted
            """
            
            async with self.driver.session() as session:
                result = await session.run(query, node_id=node_id)
                record = await result.single()
                return record["deleted"] > 0
                
        except Exception as e:
            logger.error(f"Error deleting node: {e}")
            return False
    
    async def find_shortest_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> List[Dict[str, Any]]:
        """Find shortest path between two nodes"""
        try:
            query = """
                MATCH path = shortestPath(
                    (source {id: $source_id})-[*..%d]-(target {id: $target_id})
                )
                RETURN [n in nodes(path) | {id: n.id, labels: labels(n)}] as nodes,
                       [r in relationships(path) | {type: type(r), properties: properties(r)}] as relationships
            """ % max_depth
            
            async with self.driver.session() as session:
                result = await session.run(
                    query,
                    source_id=source_id,
                    target_id=target_id
                )
                record = await result.single()
                
                if record:
                    return {
                        "nodes": record["nodes"],
                        "relationships": record["relationships"]
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Error finding path: {e}")
            return None
    
    async def get_node_neighbors(
        self,
        node_id: str,
        depth: int = 1,
        relationship_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get neighboring nodes up to specified depth"""
        try:
            rel_filter = ""
            if relationship_types:
                rel_filter = ":" + "|".join(relationship_types)
            
            query = f"""
                MATCH (center {{id: $node_id}})
                OPTIONAL MATCH path = (center)-[r{rel_filter}*1..{depth}]-(neighbor)
                WITH center, collect(DISTINCT neighbor) as neighbors, 
                     collect(DISTINCT r) as relationships
                RETURN center, neighbors, relationships
            """
            
            async with self.driver.session() as session:
                result = await session.run(query, node_id=node_id)
                record = await result.single()
                
                if record:
                    return {
                        "center": dict(record["center"]),
                        "neighbors": [dict(n) for n in record["neighbors"] if n],
                        "relationships": record["relationships"]
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting neighbors: {e}")
            return None
    
    async def execute_cypher(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute arbitrary Cypher query"""
        try:
            async with self.driver.session() as session:
                result = await session.run(query, **(parameters or {}))
                records = []
                
                async for record in result:
                    records.append(dict(record))
                
                return records
                
        except Exception as e:
            logger.error(f"Error executing Cypher: {e}")
            raise
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        try:
            queries = {
                "node_count": "MATCH (n) RETURN count(n) as count",
                "relationship_count": "MATCH ()-[r]->() RETURN count(r) as count",
                "label_counts": "MATCH (n) RETURN labels(n) as labels, count(n) as count",
                "relationship_type_counts": "MATCH ()-[r]->() RETURN type(r) as type, count(r) as count"
            }
            
            stats = {}
            
            async with self.driver.session() as session:
                for key, query in queries.items():
                    result = await session.run(query)
                    
                    if "counts" in key:
                        records = []
                        async for record in result:
                            records.append(dict(record))
                        stats[key] = records
                    else:
                        record = await result.single()
                        stats[key] = record["count"] if record else 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}