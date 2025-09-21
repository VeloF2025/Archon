"""
Relationship Mapper for Knowledge Graph
Maps and discovers relationships between concepts
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import numpy as np
from collections import defaultdict
from pydantic import BaseModel, Field

from .graph_client import Neo4jClient

logger = logging.getLogger(__name__)


class RelationshipType(BaseModel):
    """Defines a type of relationship"""
    name: str
    inverse_name: Optional[str] = None
    transitivity: bool = False  # If A->B and B->C implies A->C
    symmetry: bool = False  # If A->B implies B->A
    strength_threshold: float = 0.5
    description: str = ""


class RelationshipMapper:
    """Maps relationships between concepts in the knowledge graph"""
    
    def __init__(self, graph_client: Neo4jClient):
        self.graph = graph_client
        self.relationship_types = self._initialize_relationship_types()
        self.discovered_relationships = []
    
    def _initialize_relationship_types(self) -> Dict[str, RelationshipType]:
        """Initialize known relationship types"""
        return {
            "DEPENDS_ON": RelationshipType(
                name="DEPENDS_ON",
                inverse_name="REQUIRED_BY",
                transitivity=True,
                description="Dependency relationship"
            ),
            "IMPLEMENTS": RelationshipType(
                name="IMPLEMENTS",
                inverse_name="IMPLEMENTED_BY",
                description="Implementation relationship"
            ),
            "EXTENDS": RelationshipType(
                name="EXTENDS",
                inverse_name="EXTENDED_BY",
                transitivity=True,
                description="Extension/inheritance relationship"
            ),
            "USES": RelationshipType(
                name="USES",
                inverse_name="USED_BY",
                transitivity=False,
                description="Usage relationship"
            ),
            "SIMILAR_TO": RelationshipType(
                name="SIMILAR_TO",
                symmetry=True,
                description="Similarity relationship"
            ),
            "ALTERNATIVE_TO": RelationshipType(
                name="ALTERNATIVE_TO",
                symmetry=True,
                description="Alternative option"
            ),
            "CONFLICTS_WITH": RelationshipType(
                name="CONFLICTS_WITH",
                symmetry=True,
                description="Conflicting concepts"
            ),
            "PREREQUISITE_FOR": RelationshipType(
                name="PREREQUISITE_FOR",
                inverse_name="REQUIRES",
                transitivity=True,
                description="Prerequisite relationship"
            ),
            "RELATED_TO": RelationshipType(
                name="RELATED_TO",
                symmetry=True,
                strength_threshold=0.3,
                description="General relationship"
            )
        }
    
    async def discover_relationships(
        self,
        concept_id: str,
        max_depth: int = 2,
        min_strength: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Discover relationships for a concept
        
        Args:
            concept_id: Concept to analyze
            max_depth: Maximum search depth
            min_strength: Minimum relationship strength
            
        Returns:
            List of discovered relationships
        """
        try:
            discovered = []
            
            # Get concept and its properties
            concept = await self.graph.find_node(node_id=concept_id)
            if not concept:
                logger.warning(f"Concept {concept_id} not found")
                return []
            
            # Find similar concepts by embedding
            if concept.properties.get("embedding"):
                similar = await self._find_similar_by_embedding(
                    concept_id,
                    concept.properties["embedding"],
                    min_strength
                )
                discovered.extend(similar)
            
            # Find related by category
            category_related = await self._find_by_category(
                concept_id,
                concept.labels,
                min_strength
            )
            discovered.extend(category_related)
            
            # Find transitive relationships
            transitive = await self._find_transitive_relationships(
                concept_id,
                max_depth
            )
            discovered.extend(transitive)
            
            # Infer new relationships
            inferred = await self._infer_relationships(concept_id)
            discovered.extend(inferred)
            
            # Store discovered relationships
            self.discovered_relationships.extend(discovered)
            
            return discovered
            
        except Exception as e:
            logger.error(f"Error discovering relationships: {e}")
            return []
    
    async def _find_similar_by_embedding(
        self,
        concept_id: str,
        embedding: List[float],
        min_similarity: float
    ) -> List[Dict[str, Any]]:
        """Find similar concepts using embeddings"""
        discovered = []
        
        try:
            # Query for concepts with embeddings
            query = """
                MATCH (c:Concept)
                WHERE c.id <> $concept_id AND c.embedding IS NOT NULL
                RETURN c.id as id, c.name as name, c.embedding as embedding
                LIMIT 100
            """
            
            results = await self.graph.execute_cypher(query, {"concept_id": concept_id})
            
            for result in results:
                if result.get("embedding"):
                    similarity = self._cosine_similarity(embedding, result["embedding"])
                    
                    if similarity >= min_similarity:
                        discovered.append({
                            "source_id": concept_id,
                            "target_id": result["id"],
                            "type": "SIMILAR_TO",
                            "strength": similarity,
                            "evidence": f"Embedding similarity: {similarity:.2f}"
                        })
            
        except Exception as e:
            logger.error(f"Error finding similar by embedding: {e}")
        
        return discovered
    
    async def _find_by_category(
        self,
        concept_id: str,
        labels: List[str],
        min_strength: float
    ) -> List[Dict[str, Any]]:
        """Find related concepts by category"""
        discovered = []
        
        try:
            # Find concepts with same labels
            for label in labels:
                if label == "Concept":
                    continue
                    
                query = f"""
                    MATCH (c:{label})
                    WHERE c.id <> $concept_id
                    RETURN c.id as id, c.name as name
                    LIMIT 20
                """
                
                results = await self.graph.execute_cypher(query, {"concept_id": concept_id})
                
                for result in results:
                    discovered.append({
                        "source_id": concept_id,
                        "target_id": result["id"],
                        "type": "RELATED_TO",
                        "strength": min_strength,
                        "evidence": f"Same category: {label}"
                    })
            
        except Exception as e:
            logger.error(f"Error finding by category: {e}")
        
        return discovered
    
    async def _find_transitive_relationships(
        self,
        concept_id: str,
        max_depth: int
    ) -> List[Dict[str, Any]]:
        """Find transitive relationships"""
        discovered = []
        
        try:
            # Check each transitive relationship type
            for rel_name, rel_type in self.relationship_types.items():
                if not rel_type.transitivity:
                    continue
                
                query = f"""
                    MATCH path = (start {{id: $concept_id}})-[:{rel_name}*2..{max_depth}]->(end)
                    WHERE NOT (start)-[:{rel_name}]->(end)
                    RETURN DISTINCT end.id as target_id, end.name as target_name, length(path) as distance
                    LIMIT 10
                """
                
                results = await self.graph.execute_cypher(query, {"concept_id": concept_id})
                
                for result in results:
                    strength = 1.0 / result["distance"]  # Weaker with distance
                    discovered.append({
                        "source_id": concept_id,
                        "target_id": result["target_id"],
                        "type": rel_name,
                        "strength": strength,
                        "evidence": f"Transitive {rel_name} (distance: {result['distance']})"
                    })
            
        except Exception as e:
            logger.error(f"Error finding transitive relationships: {e}")
        
        return discovered
    
    async def _infer_relationships(self, concept_id: str) -> List[Dict[str, Any]]:
        """Infer new relationships based on patterns"""
        discovered = []
        
        try:
            # Get concept's neighbors
            neighbors = await self.graph.get_node_neighbors(concept_id, depth=1)
            if not neighbors:
                return []
            
            neighbor_ids = [n["id"] for n in neighbors.get("neighbors", [])]
            
            # Find common neighbors (triangle patterns)
            for neighbor_id in neighbor_ids:
                neighbor_neighbors = await self.graph.get_node_neighbors(neighbor_id, depth=1)
                if neighbor_neighbors:
                    common = set(n["id"] for n in neighbor_neighbors.get("neighbors", []))
                    common = common.intersection(set(neighbor_ids))
                    
                    for common_id in common:
                        if common_id != concept_id:
                            discovered.append({
                                "source_id": concept_id,
                                "target_id": common_id,
                                "type": "RELATED_TO",
                                "strength": 0.6,
                                "evidence": f"Common neighbor: {neighbor_id}"
                            })
            
        except Exception as e:
            logger.error(f"Error inferring relationships: {e}")
        
        return discovered[:10]  # Limit to avoid too many
    
    async def create_discovered_relationships(
        self,
        min_strength: float = 0.5,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Create discovered relationships in the graph
        
        Args:
            min_strength: Minimum strength to create relationship
            dry_run: If True, don't actually create relationships
            
        Returns:
            Creation results
        """
        created = 0
        skipped = 0
        
        for rel in self.discovered_relationships:
            if rel["strength"] < min_strength:
                skipped += 1
                continue
            
            try:
                # Check if relationship already exists
                existing = await self.graph.find_relationships(
                    source_id=rel["source_id"],
                    target_id=rel["target_id"],
                    relationship_type=rel["type"]
                )
                
                if not existing and not dry_run:
                    await self.graph.create_relationship(
                        source_id=rel["source_id"],
                        target_id=rel["target_id"],
                        relationship_type=rel["type"],
                        properties={
                            "strength": rel["strength"],
                            "evidence": rel["evidence"],
                            "discovered_at": datetime.utcnow().isoformat()
                        }
                    )
                    created += 1
                    
            except Exception as e:
                logger.error(f"Error creating relationship: {e}")
        
        return {
            "created": created,
            "skipped": skipped,
            "total_discovered": len(self.discovered_relationships)
        }
    
    async def analyze_relationship_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in relationships"""
        try:
            patterns = {
                "clustering": await self._analyze_clustering(),
                "centrality": await self._analyze_centrality(),
                "communities": await self._detect_communities(),
                "bottlenecks": await self._find_bottlenecks()
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return {}
    
    async def _analyze_clustering(self) -> Dict[str, Any]:
        """Analyze clustering in the graph"""
        query = """
            MATCH (n)
            WITH n, size((n)--()) as degree
            WHERE degree > 0
            MATCH (n)--(neighbor)
            WITH n, collect(DISTINCT neighbor) as neighbors
            WITH n, neighbors, size(neighbors) as degree
            WHERE degree > 1
            UNWIND range(0, size(neighbors)-2) as i
            UNWIND range(i+1, size(neighbors)-1) as j
            WITH n, neighbors[i] as n1, neighbors[j] as n2
            WHERE (n1)--(n2)
            WITH n, count(*) as triangles, size((n)--()) as degree
            WITH n.id as node_id, 
                 2.0 * triangles / (degree * (degree - 1)) as clustering_coefficient
            RETURN avg(clustering_coefficient) as avg_clustering,
                   max(clustering_coefficient) as max_clustering,
                   min(clustering_coefficient) as min_clustering
        """
        
        result = await self.graph.execute_cypher(query)
        return result[0] if result else {}
    
    async def _analyze_centrality(self) -> Dict[str, Any]:
        """Analyze node centrality"""
        query = """
            MATCH (n)
            WITH n, size((n)--()) as degree
            ORDER BY degree DESC
            LIMIT 10
            RETURN n.id as node_id, n.name as name, degree
        """
        
        results = await self.graph.execute_cypher(query)
        
        return {
            "most_connected": results,
            "count": len(results)
        }
    
    async def _detect_communities(self) -> Dict[str, Any]:
        """Detect communities in the graph"""
        # Simplified community detection
        query = """
            MATCH (n)-[r]-(m)
            WITH n, count(DISTINCT m) as connections
            WHERE connections > 3
            RETURN n.category as category, count(n) as count
            ORDER BY count DESC
        """
        
        results = await self.graph.execute_cypher(query)
        
        return {
            "communities": results,
            "total": len(results)
        }
    
    async def _find_bottlenecks(self) -> Dict[str, Any]:
        """Find bottleneck nodes"""
        query = """
            MATCH (n)
            WHERE size((n)<--()) > 5 AND size((n)-->()) < 2
            RETURN n.id as node_id, n.name as name,
                   size((n)<--()) as in_degree,
                   size((n)-->()) as out_degree
            ORDER BY in_degree DESC
            LIMIT 10
        """
        
        results = await self.graph.execute_cypher(query)
        
        return {
            "bottlenecks": results,
            "count": len(results)
        }
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))