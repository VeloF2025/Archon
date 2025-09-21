"""
Graph Analyzer for Knowledge Graph
Analyzes graph structure, patterns, and provides insights
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, Counter
from pydantic import BaseModel, Field

from .graph_client import Neo4jClient

logger = logging.getLogger(__name__)


class GraphMetrics(BaseModel):
    """Graph metrics and statistics"""
    node_count: int = 0
    edge_count: int = 0
    average_degree: float = 0.0
    density: float = 0.0
    clustering_coefficient: float = 0.0
    connected_components: int = 0
    largest_component_size: int = 0
    average_path_length: float = 0.0
    diameter: int = 0


class GraphInsight(BaseModel):
    """Insights derived from graph analysis"""
    type: str  # "pattern", "anomaly", "trend", "recommendation"
    title: str
    description: str
    severity: str  # "info", "warning", "critical"
    affected_nodes: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class GraphAnalyzer:
    """Analyzes the knowledge graph for insights and patterns"""
    
    def __init__(self, graph_client: Neo4jClient):
        self.graph = graph_client
        self.metrics_cache = {}
        self.insights = []
    
    async def analyze_graph(self) -> Dict[str, Any]:
        """
        Comprehensive graph analysis
        
        Returns:
            Analysis results with metrics and insights
        """
        try:
            logger.info("Starting comprehensive graph analysis")
            
            # Calculate basic metrics
            metrics = await self._calculate_metrics()
            
            # Analyze patterns
            patterns = await self._analyze_patterns()
            
            # Detect anomalies
            anomalies = await self._detect_anomalies()
            
            # Analyze trends
            trends = await self._analyze_trends()
            
            # Generate insights
            insights = await self._generate_insights(metrics, patterns, anomalies, trends)
            
            # Store insights
            self.insights.extend(insights)
            
            return {
                "metrics": metrics.dict() if isinstance(metrics, GraphMetrics) else metrics,
                "patterns": patterns,
                "anomalies": anomalies,
                "trends": trends,
                "insights": [i.dict() for i in insights],
                "analyzed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing graph: {e}")
            return {}
    
    async def _calculate_metrics(self) -> GraphMetrics:
        """Calculate basic graph metrics"""
        metrics = GraphMetrics()
        
        try:
            # Node and edge counts
            stats = await self.graph.get_statistics()
            metrics.node_count = stats.get("node_count", 0)
            metrics.edge_count = stats.get("relationship_count", 0)
            
            # Average degree
            if metrics.node_count > 0:
                metrics.average_degree = (2 * metrics.edge_count) / metrics.node_count
            
            # Density
            if metrics.node_count > 1:
                max_edges = metrics.node_count * (metrics.node_count - 1) / 2
                metrics.density = metrics.edge_count / max_edges if max_edges > 0 else 0
            
            # Clustering coefficient
            clustering_query = """
                MATCH (n)
                WITH n, size((n)--()) as degree
                WHERE degree > 1
                MATCH (n)--(neighbor)
                WITH n, collect(DISTINCT neighbor) as neighbors, degree
                UNWIND range(0, size(neighbors)-2) as i
                UNWIND range(i+1, size(neighbors)-1) as j
                WITH n, neighbors[i] as n1, neighbors[j] as n2, degree
                WHERE (n1)--(n2)
                WITH n, count(*) as triangles, degree
                WITH n, 2.0 * triangles / (degree * (degree - 1)) as node_clustering
                RETURN avg(node_clustering) as avg_clustering
            """
            
            result = await self.graph.execute_cypher(clustering_query)
            if result and result[0].get("avg_clustering"):
                metrics.clustering_coefficient = result[0]["avg_clustering"]
            
            # Connected components
            components_query = """
                CALL gds.graph.project.cypher(
                    'components',
                    'MATCH (n) RETURN id(n) as id',
                    'MATCH (n)-[r]-(m) RETURN id(n) as source, id(m) as target'
                )
                YIELD graphName
                CALL gds.wcc.stats('components')
                YIELD componentCount, componentDistribution
                RETURN componentCount, componentDistribution.max as largestComponent
            """
            
            # Simplified version without GDS
            simple_components_query = """
                MATCH (n)
                WITH collect(DISTINCT n) as nodes
                RETURN size(nodes) as component_count
            """
            
            result = await self.graph.execute_cypher(simple_components_query)
            if result:
                metrics.connected_components = result[0].get("component_count", 0)
            
            self.metrics_cache = metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
        
        return metrics
    
    async def _analyze_patterns(self) -> Dict[str, Any]:
        """Analyze graph patterns"""
        patterns = {
            "hubs": [],
            "clusters": [],
            "bridges": [],
            "cycles": []
        }
        
        try:
            # Find hub nodes (high degree centrality)
            hub_query = """
                MATCH (n)
                WITH n, size((n)--()) as degree
                WHERE degree > 10
                RETURN n.id as node_id, n.name as name, degree
                ORDER BY degree DESC
                LIMIT 10
            """
            
            hubs = await self.graph.execute_cypher(hub_query)
            patterns["hubs"] = hubs
            
            # Find dense clusters
            cluster_query = """
                MATCH (n)-[r]-(m)
                WITH n, count(DISTINCT m) as connections
                WHERE connections > 5
                WITH n.category as category, collect(n.name) as nodes, avg(connections) as avg_connections
                WHERE size(nodes) > 3
                RETURN category, size(nodes) as cluster_size, avg_connections
                ORDER BY cluster_size DESC
                LIMIT 10
            """
            
            clusters = await self.graph.execute_cypher(cluster_query)
            patterns["clusters"] = clusters
            
            # Find bridge nodes (betweenness centrality)
            bridge_query = """
                MATCH (n)
                WHERE size((n)--()) > 2
                WITH n, size((n)--()) as degree
                MATCH p = shortestPath((a)-[*]-(b))
                WHERE a <> b AND n IN nodes(p) AND a <> n AND b <> n
                WITH n, count(p) as paths_through
                WHERE paths_through > 10
                RETURN n.id as node_id, n.name as name, paths_through
                ORDER BY paths_through DESC
                LIMIT 10
            """
            
            # Simplified bridge detection
            simple_bridge_query = """
                MATCH (n)
                WHERE size((n)<--()) > 0 AND size((n)-->()) > 0
                WITH n, size((n)<--()) as in_degree, size((n)-->()) as out_degree
                WHERE in_degree > 2 AND out_degree > 2
                RETURN n.id as node_id, n.name as name, in_degree + out_degree as total_degree
                ORDER BY total_degree DESC
                LIMIT 10
            """
            
            bridges = await self.graph.execute_cypher(simple_bridge_query)
            patterns["bridges"] = bridges
            
            # Find cycles
            cycle_query = """
                MATCH path = (n)-[*3..5]-(n)
                WITH path, nodes(path) as cycle_nodes
                WHERE size(cycle_nodes) = size(collect(DISTINCT id(n) FOR n IN cycle_nodes))
                RETURN size(cycle_nodes) as cycle_length, [n IN cycle_nodes | n.name] as nodes
                LIMIT 10
            """
            
            cycles = await self.graph.execute_cypher(cycle_query)
            patterns["cycles"] = cycles
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
        
        return patterns
    
    async def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in the graph"""
        anomalies = []
        
        try:
            # Isolated nodes
            isolated_query = """
                MATCH (n)
                WHERE NOT (n)--()
                RETURN n.id as node_id, n.name as name, n.category as category
                LIMIT 20
            """
            
            isolated = await self.graph.execute_cypher(isolated_query)
            if isolated:
                anomalies.append({
                    "type": "isolated_nodes",
                    "count": len(isolated),
                    "nodes": isolated[:5],
                    "severity": "warning"
                })
            
            # Unusually high degree nodes
            high_degree_query = """
                MATCH (n)
                WITH n, size((n)--()) as degree
                WITH avg(degree) as avg_degree, stDev(degree) as std_degree
                MATCH (n)
                WITH n, size((n)--()) as degree, avg_degree, std_degree
                WHERE degree > avg_degree + 3 * std_degree
                RETURN n.id as node_id, n.name as name, degree
                ORDER BY degree DESC
                LIMIT 10
            """
            
            high_degree = await self.graph.execute_cypher(high_degree_query)
            if high_degree:
                anomalies.append({
                    "type": "unusual_high_degree",
                    "count": len(high_degree),
                    "nodes": high_degree,
                    "severity": "info"
                })
            
            # Duplicate relationships
            duplicate_query = """
                MATCH (a)-[r1]->(b)
                MATCH (a)-[r2]->(b)
                WHERE id(r1) < id(r2) AND type(r1) = type(r2)
                RETURN a.id as source, b.id as target, type(r1) as rel_type, count(*) as duplicates
                LIMIT 10
            """
            
            duplicates = await self.graph.execute_cypher(duplicate_query)
            if duplicates:
                anomalies.append({
                    "type": "duplicate_relationships",
                    "count": len(duplicates),
                    "relationships": duplicates,
                    "severity": "warning"
                })
            
            # Self-loops
            self_loop_query = """
                MATCH (n)-[r]-(n)
                RETURN n.id as node_id, n.name as name, type(r) as rel_type
                LIMIT 10
            """
            
            self_loops = await self.graph.execute_cypher(self_loop_query)
            if self_loops:
                anomalies.append({
                    "type": "self_loops",
                    "count": len(self_loops),
                    "nodes": self_loops,
                    "severity": "info"
                })
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
        
        return anomalies
    
    async def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze temporal trends in the graph"""
        trends = {
            "growth_rate": {},
            "recent_activity": {},
            "popular_concepts": []
        }
        
        try:
            # Growth rate
            growth_query = """
                MATCH (n)
                WHERE n.created_at IS NOT NULL
                WITH date(n.created_at) as creation_date, count(n) as nodes_created
                ORDER BY creation_date DESC
                LIMIT 30
                RETURN creation_date, nodes_created
            """
            
            growth = await self.graph.execute_cypher(growth_query)
            if growth:
                trends["growth_rate"] = {
                    "daily_counts": growth,
                    "average": sum(g["nodes_created"] for g in growth) / len(growth) if growth else 0
                }
            
            # Recent activity
            recent_query = """
                MATCH (n)
                WHERE n.updated_at IS NOT NULL
                WITH n
                ORDER BY n.updated_at DESC
                LIMIT 20
                RETURN n.id as node_id, n.name as name, n.updated_at as last_updated
            """
            
            recent = await self.graph.execute_cypher(recent_query)
            trends["recent_activity"] = {
                "recently_updated": recent[:10],
                "count": len(recent)
            }
            
            # Popular concepts (most connected)
            popular_query = """
                MATCH (n)
                WITH n, size((n)--()) as connections
                ORDER BY connections DESC
                LIMIT 10
                RETURN n.id as node_id, n.name as name, n.category as category, connections
            """
            
            popular = await self.graph.execute_cypher(popular_query)
            trends["popular_concepts"] = popular
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
        
        return trends
    
    async def _generate_insights(
        self,
        metrics: GraphMetrics,
        patterns: Dict[str, Any],
        anomalies: List[Dict[str, Any]],
        trends: Dict[str, Any]
    ) -> List[GraphInsight]:
        """Generate insights from analysis"""
        insights = []
        
        # Insight: Graph density
        if metrics.density < 0.01:
            insights.append(GraphInsight(
                type="pattern",
                title="Sparse Graph",
                description=f"Graph density is very low ({metrics.density:.4f}), indicating weak connectivity",
                severity="warning",
                metrics={"density": metrics.density},
                recommendations=["Consider relationship discovery", "Enhance knowledge ingestion"]
            ))
        elif metrics.density > 0.5:
            insights.append(GraphInsight(
                type="pattern",
                title="Dense Graph",
                description=f"Graph density is high ({metrics.density:.2f}), indicating strong connectivity",
                severity="info",
                metrics={"density": metrics.density}
            ))
        
        # Insight: Hub nodes
        if patterns.get("hubs"):
            top_hubs = patterns["hubs"][:3]
            insights.append(GraphInsight(
                type="pattern",
                title="Key Hub Nodes Identified",
                description=f"Found {len(patterns['hubs'])} hub nodes that are highly connected",
                severity="info",
                affected_nodes=[h["node_id"] for h in top_hubs],
                metrics={"hub_count": len(patterns["hubs"])},
                recommendations=["Monitor hub nodes for bottlenecks", "Consider load distribution"]
            ))
        
        # Insight: Isolated nodes
        isolated_anomaly = next((a for a in anomalies if a["type"] == "isolated_nodes"), None)
        if isolated_anomaly and isolated_anomaly["count"] > 0:
            insights.append(GraphInsight(
                type="anomaly",
                title="Isolated Nodes Detected",
                description=f"Found {isolated_anomaly['count']} isolated nodes with no connections",
                severity="warning",
                affected_nodes=[n["node_id"] for n in isolated_anomaly["nodes"][:5]],
                metrics={"isolated_count": isolated_anomaly["count"]},
                recommendations=["Review isolated nodes", "Consider removing or connecting them"]
            ))
        
        # Insight: Growth trend
        if trends.get("growth_rate", {}).get("average", 0) > 100:
            insights.append(GraphInsight(
                type="trend",
                title="Rapid Graph Growth",
                description="Graph is growing rapidly with high daily node creation rate",
                severity="info",
                metrics={"daily_average": trends["growth_rate"]["average"]},
                recommendations=["Monitor performance", "Consider scaling strategy"]
            ))
        
        # Insight: Clustering
        if metrics.clustering_coefficient > 0.7:
            insights.append(GraphInsight(
                type="pattern",
                title="High Clustering Detected",
                description=f"High clustering coefficient ({metrics.clustering_coefficient:.2f}) indicates tight communities",
                severity="info",
                metrics={"clustering": metrics.clustering_coefficient}
            ))
        
        return insights
    
    async def analyze_concept_importance(
        self,
        concept_id: str
    ) -> Dict[str, Any]:
        """
        Analyze importance of a specific concept
        
        Args:
            concept_id: Concept to analyze
            
        Returns:
            Importance metrics
        """
        try:
            # Degree centrality
            degree_query = """
                MATCH (n {id: $concept_id})
                WITH n, size((n)--()) as degree,
                     size((n)<--()) as in_degree,
                     size((n)-->()) as out_degree
                RETURN degree, in_degree, out_degree
            """
            
            degree_result = await self.graph.execute_cypher(
                degree_query,
                {"concept_id": concept_id}
            )
            
            # Shortest paths through node
            betweenness_query = """
                MATCH (n {id: $concept_id})
                MATCH p = shortestPath((a)-[*]-(b))
                WHERE a <> b AND n IN nodes(p) AND a <> n AND b <> n
                RETURN count(p) as paths_through
            """
            
            betweenness_result = await self.graph.execute_cypher(
                betweenness_query,
                {"concept_id": concept_id}
            )
            
            # PageRank approximation
            pagerank_query = """
                MATCH (n {id: $concept_id})
                MATCH (n)<-[:*1..2]-(influencer)
                WITH n, count(DISTINCT influencer) as influence_score
                RETURN influence_score
            """
            
            pagerank_result = await self.graph.execute_cypher(
                pagerank_query,
                {"concept_id": concept_id}
            )
            
            importance = {
                "concept_id": concept_id,
                "degree_centrality": degree_result[0] if degree_result else {},
                "betweenness_centrality": betweenness_result[0]["paths_through"] if betweenness_result else 0,
                "influence_score": pagerank_result[0]["influence_score"] if pagerank_result else 0
            }
            
            # Calculate overall importance score
            if degree_result:
                degree = degree_result[0]["degree"]
                betweenness = betweenness_result[0]["paths_through"] if betweenness_result else 0
                influence = pagerank_result[0]["influence_score"] if pagerank_result else 0
                
                # Normalize and combine scores
                importance["overall_score"] = (
                    0.4 * min(degree / 20, 1) +
                    0.3 * min(betweenness / 100, 1) +
                    0.3 * min(influence / 50, 1)
                )
            
            return importance
            
        except Exception as e:
            logger.error(f"Error analyzing concept importance: {e}")
            return {}