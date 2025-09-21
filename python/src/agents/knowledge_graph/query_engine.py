"""
Query Engine for Knowledge Graph
Provides advanced querying capabilities with natural language support
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import re
from pydantic import BaseModel, Field

from .graph_client import Neo4jClient
from ...server.services.embeddings import create_embedding

logger = logging.getLogger(__name__)


class QueryResult(BaseModel):
    """Result from a graph query"""
    query: str
    query_type: str  # "cypher", "natural", "template"
    results: List[Dict[str, Any]]
    count: int
    execution_time: float = 0.0
    explanation: Optional[str] = None
    visualizable: bool = False


class GraphQueryEngine:
    """Advanced query engine for the knowledge graph"""
    
    def __init__(self, graph_client: Neo4jClient):
        self.graph = graph_client
        self.query_templates = self._initialize_templates()
        self.query_history = []
    
    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize common query templates"""
        return {
            "find_related": """
                MATCH (n {name: $name})-[r]-(related)
                RETURN n, type(r) as relationship, related
                LIMIT $limit
            """,
            "shortest_path": """
                MATCH (source {name: $source}), (target {name: $target})
                MATCH path = shortestPath((source)-[*..%d]-(target))
                RETURN path, length(path) as distance
            """,
            "common_neighbors": """
                MATCH (a {name: $node1})--(common)--(b {name: $node2})
                WHERE a <> b
                RETURN common, count(*) as connections
                ORDER BY connections DESC
                LIMIT $limit
            """,
            "subgraph": """
                MATCH (center {name: $center})
                MATCH path = (center)-[*..%d]-(neighbor)
                RETURN path
                LIMIT $limit
            """,
            "pattern_search": """
                MATCH (n:Pattern {language: $language})
                WHERE n.is_antipattern = $is_antipattern
                RETURN n
                ORDER BY n.effectiveness_score DESC
                LIMIT $limit
            """,
            "concept_search": """
                MATCH (n:Concept)
                WHERE n.name CONTAINS $search_term
                OR n.description CONTAINS $search_term
                RETURN n
                ORDER BY n.relevance_score DESC
                LIMIT $limit
            """,
            "impact_analysis": """
                MATCH (source {name: $name})
                MATCH (source)-[*1..%d]->(impacted)
                RETURN DISTINCT impacted, 
                       size((source)-[*]->(impacted)) as impact_distance
                ORDER BY impact_distance
                LIMIT $limit
            """
        }
    
    async def query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        query_type: str = "cypher"
    ) -> QueryResult:
        """
        Execute a query on the knowledge graph
        
        Args:
            query: Query string (Cypher or natural language)
            parameters: Query parameters
            query_type: Type of query ("cypher", "natural", "template")
            
        Returns:
            Query results
        """
        start_time = datetime.utcnow()
        
        try:
            if query_type == "natural":
                cypher_query, params = await self._natural_to_cypher(query, parameters)
            elif query_type == "template":
                cypher_query = self.query_templates.get(query, query)
                params = parameters or {}
            else:
                cypher_query = query
                params = parameters or {}
            
            # Execute query
            results = await self.graph.execute_cypher(cypher_query, params)
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Determine if results are visualizable
            visualizable = self._is_visualizable(cypher_query, results)
            
            # Create result
            result = QueryResult(
                query=query,
                query_type=query_type,
                results=results,
                count=len(results),
                execution_time=execution_time,
                visualizable=visualizable
            )
            
            # Store in history
            self.query_history.append({
                "query": query,
                "timestamp": datetime.utcnow().isoformat(),
                "result_count": len(results)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return QueryResult(
                query=query,
                query_type=query_type,
                results=[],
                count=0,
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
                explanation=f"Error: {str(e)}"
            )
    
    async def _natural_to_cypher(
        self,
        natural_query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Convert natural language query to Cypher"""
        
        query_lower = natural_query.lower()
        params = parameters or {}
        
        # Pattern matching for common queries
        
        # Find related to X
        if "related to" in query_lower or "connected to" in query_lower:
            match = re.search(r"(?:related|connected) to (\w+)", query_lower)
            if match:
                params["name"] = match.group(1)
                params["limit"] = params.get("limit", 20)
                return self.query_templates["find_related"], params
        
        # Path between X and Y
        if "path between" in query_lower or "route from" in query_lower:
            match = re.search(r"(?:between|from) (\w+) (?:and|to) (\w+)", query_lower)
            if match:
                params["source"] = match.group(1)
                params["target"] = match.group(2)
                return self.query_templates["shortest_path"] % 10, params
        
        # Common between X and Y
        if "common" in query_lower and "between" in query_lower:
            match = re.search(r"between (\w+) and (\w+)", query_lower)
            if match:
                params["node1"] = match.group(1)
                params["node2"] = match.group(2)
                params["limit"] = params.get("limit", 10)
                return self.query_templates["common_neighbors"], params
        
        # Search for patterns
        if "antipattern" in query_lower or "anti-pattern" in query_lower:
            params["is_antipattern"] = True
            params["language"] = params.get("language", "python")
            params["limit"] = params.get("limit", 10)
            return self.query_templates["pattern_search"], params
        
        # Search for concepts
        if "find" in query_lower or "search" in query_lower:
            # Extract search term
            words = natural_query.split()
            search_terms = [w for w in words if w.lower() not in ["find", "search", "for", "the", "a", "an"]]
            if search_terms:
                params["search_term"] = " ".join(search_terms)
                params["limit"] = params.get("limit", 20)
                return self.query_templates["concept_search"], params
        
        # Impact analysis
        if "impact" in query_lower or "affects" in query_lower:
            match = re.search(r"(?:impact of|affects) (\w+)", query_lower)
            if match:
                params["name"] = match.group(1)
                params["limit"] = params.get("limit", 30)
                return self.query_templates["impact_analysis"] % 3, params
        
        # Default: try to extract entities and search
        entities = self._extract_entities(natural_query)
        if entities:
            params["search_term"] = entities[0]
            params["limit"] = params.get("limit", 20)
            return self.query_templates["concept_search"], params
        
        # Fallback to simple match
        return "MATCH (n) RETURN n LIMIT 10", params
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential entities from text"""
        # Simple entity extraction - in production use NER
        words = text.split()
        # Filter out common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "find", "search", "show", "get", "list"}
        entities = [w for w in words if w.lower() not in stopwords and len(w) > 2]
        return entities
    
    def _is_visualizable(self, query: str, results: List[Dict[str, Any]]) -> bool:
        """Check if results can be visualized as a graph"""
        # Check if query returns nodes and relationships
        query_lower = query.lower()
        
        if "path" in query_lower or "relationship" in query_lower:
            return True
        
        if results and len(results) > 0:
            # Check if results contain graph elements
            first_result = results[0]
            for value in first_result.values():
                if isinstance(value, dict) and "id" in value:
                    return True
        
        return False
    
    async def semantic_search(
        self,
        query_text: str,
        limit: int = 10,
        threshold: float = 0.7
    ) -> QueryResult:
        """
        Semantic search using embeddings
        
        Args:
            query_text: Search query
            limit: Maximum results
            threshold: Similarity threshold
            
        Returns:
            Search results
        """
        try:
            # Generate embedding for query
            query_embedding = await create_embedding(query_text)
            
            # Search using embedding similarity
            # Note: This would use pgvector in PostgreSQL or vector index in Neo4j
            query = """
                MATCH (n:Concept)
                WHERE n.embedding IS NOT NULL
                RETURN n, n.name as name, n.description as description
                LIMIT $limit
            """
            
            results = await self.graph.execute_cypher(query, {"limit": limit})
            
            # Calculate similarities (simplified - would be done in database)
            if query_embedding and results:
                for result in results:
                    if result.get("n", {}).get("embedding"):
                        similarity = self._cosine_similarity(
                            query_embedding,
                            result["n"]["embedding"]
                        )
                        result["similarity"] = similarity
                
                # Filter by threshold and sort
                results = [r for r in results if r.get("similarity", 0) >= threshold]
                results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            
            return QueryResult(
                query=query_text,
                query_type="semantic",
                results=results,
                count=len(results),
                explanation=f"Semantic search for: {query_text}"
            )
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return QueryResult(
                query=query_text,
                query_type="semantic",
                results=[],
                count=0,
                explanation=f"Error: {str(e)}"
            )
    
    async def recommend_queries(
        self,
        context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Recommend useful queries based on context
        
        Args:
            context: Optional context for recommendations
            
        Returns:
            List of recommended queries
        """
        recommendations = []
        
        # Basic recommendations
        recommendations.extend([
            {
                "title": "Find Hub Nodes",
                "query": "MATCH (n) WITH n, size((n)--()) as degree WHERE degree > 10 RETURN n ORDER BY degree DESC LIMIT 10",
                "description": "Identify the most connected concepts"
            },
            {
                "title": "Recent Additions",
                "query": "MATCH (n) WHERE n.created_at IS NOT NULL RETURN n ORDER BY n.created_at DESC LIMIT 20",
                "description": "View recently added concepts"
            },
            {
                "title": "Find Antipatterns",
                "query": "MATCH (n:Pattern) WHERE n.is_antipattern = true RETURN n",
                "description": "List all detected antipatterns"
            },
            {
                "title": "Isolated Concepts",
                "query": "MATCH (n) WHERE NOT (n)--() RETURN n",
                "description": "Find concepts with no connections"
            }
        ])
        
        # Context-based recommendations
        if context:
            context_lower = context.lower()
            
            if "pattern" in context_lower:
                recommendations.append({
                    "title": "Effective Patterns",
                    "query": "MATCH (n:Pattern) WHERE n.effectiveness_score > 0.7 RETURN n ORDER BY n.effectiveness_score DESC",
                    "description": "Find highly effective patterns"
                })
            
            if "project" in context_lower:
                recommendations.append({
                    "title": "Project Dependencies",
                    "query": "MATCH (p:Project)-[:DEPENDS_ON]->(d) RETURN p, d",
                    "description": "View project dependencies"
                })
            
            if "performance" in context_lower:
                recommendations.append({
                    "title": "Performance Bottlenecks",
                    "query": "MATCH (n) WHERE size((n)<--()) > 10 AND size((n)-->()) < 2 RETURN n",
                    "description": "Find potential bottleneck nodes"
                })
        
        return recommendations
    
    async def export_subgraph(
        self,
        center_node: str,
        depth: int = 2,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Export a subgraph centered on a node
        
        Args:
            center_node: Center node ID
            depth: Depth of subgraph
            format: Export format (json, cypher, graphml)
            
        Returns:
            Exported subgraph data
        """
        try:
            # Get subgraph
            query = f"""
                MATCH (center {{id: $center}})
                MATCH path = (center)-[*0..{depth}]-(neighbor)
                WITH collect(DISTINCT neighbor) as nodes,
                     collect(DISTINCT relationships(path)) as rels_nested
                UNWIND rels_nested as rels_list
                UNWIND rels_list as rel
                WITH nodes, collect(DISTINCT rel) as relationships
                RETURN nodes, relationships
            """
            
            result = await self.graph.execute_cypher(query, {"center": center_node})
            
            if not result:
                return {}
            
            nodes = result[0].get("nodes", [])
            relationships = result[0].get("relationships", [])
            
            if format == "json":
                return {
                    "nodes": [dict(n) for n in nodes],
                    "relationships": [
                        {
                            "source": r.start_node.id if hasattr(r, 'start_node') else str(r.id),
                            "target": r.end_node.id if hasattr(r, 'end_node') else str(r.id),
                            "type": r.type if hasattr(r, 'type') else "RELATED_TO"
                        } for r in relationships
                    ]
                }
            
            elif format == "cypher":
                # Generate Cypher statements
                statements = []
                
                # Create nodes
                for node in nodes:
                    labels = ":".join(node.labels if hasattr(node, 'labels') else ["Concept"])
                    props = json.dumps(dict(node))
                    statements.append(f"CREATE (n:{labels} {props})")
                
                # Create relationships
                for rel in relationships:
                    statements.append(
                        f"MATCH (a {{id: '{rel.start_node.id}'}}), (b {{id: '{rel.end_node.id}'}}) "
                        f"CREATE (a)-[:{rel.type}]->(b)"
                    )
                
                return {"cypher": "\n".join(statements)}
            
            else:
                return {"error": f"Unsupported format: {format}"}
                
        except Exception as e:
            logger.error(f"Error exporting subgraph: {e}")
            return {"error": str(e)}
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        import numpy as np
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    async def get_query_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent query history"""
        return self.query_history[-limit:]