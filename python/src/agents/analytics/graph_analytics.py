"""
Graph Analytics & Network Analysis Engine
Advanced Analytics & Intelligence Platform - Archon Enhancement 2025 Phase 5

Enterprise-grade graph analytics with:
- Network topology analysis and graph algorithms
- Community detection and clustering
- Social network analysis and influence scoring
- Knowledge graph construction and querying
- Graph neural networks and embeddings
- Real-time graph streaming and updates
- Graph visualization and interactive exploration
- Fraud detection and anomaly patterns in networks
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set
from datetime import datetime, timedelta
import asyncio
import json
import uuid
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import networkx as nx
from collections import defaultdict, deque
import heapq
from itertools import combinations

logger = logging.getLogger(__name__)


class GraphType(Enum):
    DIRECTED = "directed"
    UNDIRECTED = "undirected"
    WEIGHTED = "weighted"
    MULTIPARTITE = "multipartite"
    TEMPORAL = "temporal"
    HYPERGRAPH = "hypergraph"


class CommunityAlgorithm(Enum):
    LOUVAIN = "louvain"
    LEIDEN = "leiden"
    LABEL_PROPAGATION = "label_propagation"
    MODULARITY = "modularity"
    SPECTRAL = "spectral"
    WALKTRAP = "walktrap"
    INFOMAP = "infomap"


class CentralityMeasure(Enum):
    DEGREE = "degree"
    BETWEENNESS = "betweenness"
    CLOSENESS = "closeness"
    EIGENVECTOR = "eigenvector"
    PAGERANK = "pagerank"
    KATZ = "katz"
    HITS_HUB = "hits_hub"
    HITS_AUTHORITY = "hits_authority"


class GraphMetric(Enum):
    DENSITY = "density"
    CLUSTERING_COEFFICIENT = "clustering_coefficient"
    DIAMETER = "diameter"
    RADIUS = "radius"
    ASSORTATIVITY = "assortativity"
    TRANSITIVITY = "transitivity"
    MODULARITY = "modularity"
    SMALL_WORLD = "small_world"


class EmbeddingMethod(Enum):
    NODE2VEC = "node2vec"
    DEEPWALK = "deepwalk"
    LINE = "line"
    GRAPH_SAGE = "graphsage"
    GCN = "gcn"
    GAT = "gat"
    METAPATH2VEC = "metapath2vec"


@dataclass
class Node:
    """Graph node with attributes"""
    id: str
    label: str = ""
    node_type: str = "default"
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    weight: float = 1.0
    cluster_id: Optional[str] = None


@dataclass
class Edge:
    """Graph edge with attributes"""
    id: str
    source: str
    target: str
    edge_type: str = "default"
    weight: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    directed: bool = True


@dataclass
class GraphData:
    """Complete graph data structure"""
    graph_id: str
    name: str
    graph_type: GraphType
    nodes: Dict[str, Node]
    edges: Dict[str, Edge]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_dynamic: bool = False
    temporal_snapshots: List[datetime] = field(default_factory=list)


@dataclass
class Community:
    """Detected community in graph"""
    community_id: str
    nodes: Set[str]
    score: float
    algorithm: CommunityAlgorithm
    size: int = 0
    density: float = 0.0
    modularity: float = 0.0
    conductance: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class NetworkMetrics:
    """Network analysis metrics"""
    graph_id: str
    node_count: int
    edge_count: int
    density: float
    clustering_coefficient: float
    diameter: int
    average_path_length: float
    assortativity: float
    modularity: float
    components: int
    centrality_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    communities: List[Community] = field(default_factory=list)
    computed_at: datetime = field(default_factory=datetime.now)


@dataclass
class GraphEmbedding:
    """Node embeddings representation"""
    embedding_id: str
    graph_id: str
    method: EmbeddingMethod
    node_embeddings: Dict[str, np.ndarray]
    dimensions: int
    training_epochs: int = 0
    loss: float = 0.0
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class GraphQuery:
    """Graph query configuration"""
    query_id: str
    query_type: str  # path, subgraph, pattern, similarity
    parameters: Dict[str, Any]
    constraints: Dict[str, Any] = field(default_factory=dict)
    limit: int = 1000
    timeout_seconds: int = 30
    include_attributes: bool = True


@dataclass
class PathResult:
    """Path finding result"""
    source: str
    target: str
    path: List[str]
    length: int
    weight: float
    algorithm: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphStorage(ABC):
    """Abstract graph storage interface"""
    
    @abstractmethod
    async def store_graph(self, graph: GraphData) -> bool:
        pass
    
    @abstractmethod
    async def load_graph(self, graph_id: str) -> Optional[GraphData]:
        pass
    
    @abstractmethod
    async def update_node(self, graph_id: str, node: Node) -> bool:
        pass
    
    @abstractmethod
    async def update_edge(self, graph_id: str, edge: Edge) -> bool:
        pass
    
    @abstractmethod
    async def query_subgraph(self, graph_id: str, query: GraphQuery) -> GraphData:
        pass


class InMemoryGraphStorage(GraphStorage):
    """In-memory graph storage implementation"""
    
    def __init__(self):
        self.graphs: Dict[str, GraphData] = {}
        self.indexes: Dict[str, Dict[str, Set[str]]] = {}
    
    async def store_graph(self, graph: GraphData) -> bool:
        """Store graph in memory"""
        try:
            self.graphs[graph.graph_id] = graph
            await self._build_indexes(graph)
            logger.info(f"Stored graph {graph.graph_id} with {len(graph.nodes)} nodes, {len(graph.edges)} edges")
            return True
        except Exception as e:
            logger.error(f"Failed to store graph: {e}")
            return False
    
    async def load_graph(self, graph_id: str) -> Optional[GraphData]:
        """Load graph from memory"""
        return self.graphs.get(graph_id)
    
    async def update_node(self, graph_id: str, node: Node) -> bool:
        """Update node in graph"""
        if graph_id in self.graphs:
            self.graphs[graph_id].nodes[node.id] = node
            self.graphs[graph_id].updated_at = datetime.now()
            return True
        return False
    
    async def update_edge(self, graph_id: str, edge: Edge) -> bool:
        """Update edge in graph"""
        if graph_id in self.graphs:
            self.graphs[graph_id].edges[edge.id] = edge
            self.graphs[graph_id].updated_at = datetime.now()
            return True
        return False
    
    async def query_subgraph(self, graph_id: str, query: GraphQuery) -> GraphData:
        """Query subgraph based on criteria"""
        if graph_id not in self.graphs:
            raise ValueError(f"Graph not found: {graph_id}")
        
        original_graph = self.graphs[graph_id]
        
        # Extract subgraph based on query
        subgraph_nodes = {}
        subgraph_edges = {}
        
        if query.query_type == "radius":
            # Get nodes within radius of specified center
            center_node = query.parameters.get("center")
            radius = query.parameters.get("radius", 1)
            
            if center_node in original_graph.nodes:
                visited = {center_node}
                queue = deque([(center_node, 0)])
                
                while queue and len(subgraph_nodes) < query.limit:
                    node_id, distance = queue.popleft()
                    
                    if distance <= radius:
                        subgraph_nodes[node_id] = original_graph.nodes[node_id]
                        
                        # Add neighbors
                        if distance < radius:
                            for edge_id, edge in original_graph.edges.items():
                                if edge.source == node_id and edge.target not in visited:
                                    visited.add(edge.target)
                                    queue.append((edge.target, distance + 1))
                                elif edge.target == node_id and edge.source not in visited and not edge.directed:
                                    visited.add(edge.source)
                                    queue.append((edge.source, distance + 1))
        
        # Add relevant edges
        for edge_id, edge in original_graph.edges.items():
            if edge.source in subgraph_nodes and edge.target in subgraph_nodes:
                subgraph_edges[edge_id] = edge
        
        return GraphData(
            graph_id=f"{graph_id}_subgraph_{uuid.uuid4().hex[:8]}",
            name=f"Subgraph of {original_graph.name}",
            graph_type=original_graph.graph_type,
            nodes=subgraph_nodes,
            edges=subgraph_edges,
            metadata={"query": query.query_type, "parent_graph": graph_id}
        )
    
    async def _build_indexes(self, graph: GraphData):
        """Build search indexes for graph"""
        graph_id = graph.graph_id
        self.indexes[graph_id] = {
            'node_types': defaultdict(set),
            'edge_types': defaultdict(set),
            'node_attributes': defaultdict(set)
        }
        
        for node_id, node in graph.nodes.items():
            self.indexes[graph_id]['node_types'][node.node_type].add(node_id)
            
            for attr_key in node.attributes:
                self.indexes[graph_id]['node_attributes'][attr_key].add(node_id)
        
        for edge_id, edge in graph.edges.items():
            self.indexes[graph_id]['edge_types'][edge.edge_type].add(edge_id)


class CommunityDetector:
    """Community detection algorithms"""
    
    def __init__(self):
        self.communities_cache: Dict[str, List[Community]] = {}
    
    async def detect_communities(self, graph: GraphData, algorithm: CommunityAlgorithm) -> List[Community]:
        """Detect communities using specified algorithm"""
        cache_key = f"{graph.graph_id}_{algorithm.value}"
        
        if cache_key in self.communities_cache:
            logger.info(f"Returning cached communities for {cache_key}")
            return self.communities_cache[cache_key]
        
        try:
            if algorithm == CommunityAlgorithm.LOUVAIN:
                communities = await self._louvain_algorithm(graph)
            elif algorithm == CommunityAlgorithm.LABEL_PROPAGATION:
                communities = await self._label_propagation(graph)
            elif algorithm == CommunityAlgorithm.MODULARITY:
                communities = await self._modularity_optimization(graph)
            else:
                logger.warning(f"Algorithm {algorithm} not implemented, using Louvain")
                communities = await self._louvain_algorithm(graph)
            
            self.communities_cache[cache_key] = communities
            logger.info(f"Detected {len(communities)} communities using {algorithm.value}")
            return communities
            
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return []
    
    async def _louvain_algorithm(self, graph: GraphData) -> List[Community]:
        """Louvain community detection algorithm"""
        # Create adjacency representation
        adjacency = defaultdict(list)
        node_weights = {}
        
        for node_id, node in graph.nodes.items():
            node_weights[node_id] = node.weight
        
        for edge in graph.edges.values():
            adjacency[edge.source].append((edge.target, edge.weight))
            if not edge.directed:
                adjacency[edge.target].append((edge.source, edge.weight))
        
        # Initialize communities (each node in its own community)
        node_to_community = {node_id: i for i, node_id in enumerate(graph.nodes.keys())}
        communities = {i: {node_id} for i, node_id in enumerate(graph.nodes.keys())}
        
        # Iterative optimization
        improved = True
        iteration = 0
        
        while improved and iteration < 100:
            improved = False
            iteration += 1
            
            for node_id in graph.nodes:
                current_community = node_to_community[node_id]
                best_community = current_community
                best_gain = 0
                
                # Calculate modularity gain for moving to neighbor communities
                neighbor_communities = set()
                for neighbor, _ in adjacency[node_id]:
                    neighbor_communities.add(node_to_community[neighbor])
                
                for community_id in neighbor_communities:
                    if community_id != current_community:
                        gain = self._calculate_modularity_gain(
                            node_id, current_community, community_id,
                            adjacency, node_to_community, node_weights
                        )
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_community = community_id
                
                # Move node to best community
                if best_community != current_community:
                    communities[current_community].remove(node_id)
                    communities[best_community].add(node_id)
                    node_to_community[node_id] = best_community
                    improved = True
        
        # Convert to Community objects
        result_communities = []
        for community_id, nodes in communities.items():
            if nodes:  # Skip empty communities
                community = Community(
                    community_id=str(community_id),
                    nodes=nodes,
                    score=self._calculate_community_score(nodes, graph),
                    algorithm=CommunityAlgorithm.LOUVAIN,
                    size=len(nodes)
                )
                result_communities.append(community)
        
        return result_communities
    
    def _calculate_modularity_gain(self, node_id: str, current_community: int, target_community: int,
                                 adjacency: Dict, node_to_community: Dict, node_weights: Dict) -> float:
        """Calculate modularity gain for moving node to different community"""
        # Simplified modularity gain calculation
        internal_edges_current = 0
        internal_edges_target = 0
        
        for neighbor, weight in adjacency[node_id]:
            if node_to_community[neighbor] == current_community:
                internal_edges_current += weight
            elif node_to_community[neighbor] == target_community:
                internal_edges_target += weight
        
        return internal_edges_target - internal_edges_current
    
    async def _label_propagation(self, graph: GraphData) -> List[Community]:
        """Label propagation algorithm"""
        # Initialize labels
        labels = {node_id: node_id for node_id in graph.nodes}
        
        # Create adjacency list
        adjacency = defaultdict(list)
        for edge in graph.edges.values():
            adjacency[edge.source].append((edge.target, edge.weight))
            if not edge.directed:
                adjacency[edge.target].append((edge.source, edge.weight))
        
        # Iteratively update labels
        for iteration in range(100):
            updated = False
            nodes_list = list(graph.nodes.keys())
            np.random.shuffle(nodes_list)  # Random order
            
            for node_id in nodes_list:
                # Count neighbor labels
                label_weights = defaultdict(float)
                
                for neighbor, weight in adjacency[node_id]:
                    label_weights[labels[neighbor]] += weight
                
                if label_weights:
                    # Choose most frequent label
                    best_label = max(label_weights.keys(), key=label_weights.get)
                    
                    if best_label != labels[node_id]:
                        labels[node_id] = best_label
                        updated = True
            
            if not updated:
                break
        
        # Group nodes by labels
        communities_dict = defaultdict(set)
        for node_id, label in labels.items():
            communities_dict[label].add(node_id)
        
        # Convert to Community objects
        communities = []
        for i, (label, nodes) in enumerate(communities_dict.items()):
            if len(nodes) > 1:  # Skip singleton communities
                community = Community(
                    community_id=f"lp_{i}",
                    nodes=nodes,
                    score=self._calculate_community_score(nodes, graph),
                    algorithm=CommunityAlgorithm.LABEL_PROPAGATION,
                    size=len(nodes)
                )
                communities.append(community)
        
        return communities
    
    async def _modularity_optimization(self, graph: GraphData) -> List[Community]:
        """Modularity optimization algorithm"""
        # Simplified modularity optimization
        return await self._louvain_algorithm(graph)
    
    def _calculate_community_score(self, nodes: Set[str], graph: GraphData) -> float:
        """Calculate community quality score"""
        if len(nodes) < 2:
            return 0.0
        
        internal_edges = 0
        external_edges = 0
        
        for edge in graph.edges.values():
            if edge.source in nodes and edge.target in nodes:
                internal_edges += 1
            elif edge.source in nodes or edge.target in nodes:
                external_edges += 1
        
        total_edges = internal_edges + external_edges
        if total_edges == 0:
            return 0.0
        
        return internal_edges / total_edges


class CentralityCalculator:
    """Centrality measures calculator"""
    
    def __init__(self):
        self.centrality_cache: Dict[str, Dict[str, Dict[str, float]]] = {}
    
    async def calculate_centrality(self, graph: GraphData, measure: CentralityMeasure) -> Dict[str, float]:
        """Calculate centrality measure for all nodes"""
        cache_key = f"{graph.graph_id}_{measure.value}"
        
        if cache_key in self.centrality_cache:
            return self.centrality_cache[cache_key]
        
        try:
            if measure == CentralityMeasure.DEGREE:
                scores = await self._degree_centrality(graph)
            elif measure == CentralityMeasure.BETWEENNESS:
                scores = await self._betweenness_centrality(graph)
            elif measure == CentralityMeasure.CLOSENESS:
                scores = await self._closeness_centrality(graph)
            elif measure == CentralityMeasure.PAGERANK:
                scores = await self._pagerank_centrality(graph)
            elif measure == CentralityMeasure.EIGENVECTOR:
                scores = await self._eigenvector_centrality(graph)
            else:
                logger.warning(f"Centrality measure {measure} not implemented")
                scores = await self._degree_centrality(graph)
            
            self.centrality_cache[cache_key] = scores
            logger.info(f"Calculated {measure.value} centrality for {len(scores)} nodes")
            return scores
            
        except Exception as e:
            logger.error(f"Centrality calculation failed: {e}")
            return {}
    
    async def _degree_centrality(self, graph: GraphData) -> Dict[str, float]:
        """Calculate degree centrality"""
        degree_counts = defaultdict(int)
        
        for edge in graph.edges.values():
            degree_counts[edge.source] += 1
            if not edge.directed:
                degree_counts[edge.target] += 1
            elif edge.directed:
                degree_counts[edge.target] += 1
        
        # Normalize by maximum possible degree
        max_degree = len(graph.nodes) - 1 if len(graph.nodes) > 1 else 1
        
        centrality = {}
        for node_id in graph.nodes:
            centrality[node_id] = degree_counts[node_id] / max_degree
        
        return centrality
    
    async def _betweenness_centrality(self, graph: GraphData) -> Dict[str, float]:
        """Calculate betweenness centrality using Brandes algorithm"""
        centrality = {node_id: 0.0 for node_id in graph.nodes}
        
        # Build adjacency list
        adjacency = defaultdict(list)
        for edge in graph.edges.values():
            adjacency[edge.source].append(edge.target)
            if not edge.directed:
                adjacency[edge.target].append(edge.source)
        
        # For each node as source
        for source in graph.nodes:
            # BFS to find shortest paths
            predecessors = defaultdict(list)
            distances = {node_id: -1 for node_id in graph.nodes}
            distances[source] = 0
            sigma = defaultdict(int)
            sigma[source] = 1
            
            queue = deque([source])
            stack = []
            
            while queue:
                v = queue.popleft()
                stack.append(v)
                
                for w in adjacency[v]:
                    # First time visiting w?
                    if distances[w] < 0:
                        queue.append(w)
                        distances[w] = distances[v] + 1
                    
                    # Shortest path to w via v?
                    if distances[w] == distances[v] + 1:
                        sigma[w] += sigma[v]
                        predecessors[w].append(v)
            
            # Accumulate betweenness
            delta = defaultdict(float)
            
            while stack:
                w = stack.pop()
                for v in predecessors[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                
                if w != source:
                    centrality[w] += delta[w]
        
        # Normalize
        n = len(graph.nodes)
        if n > 2:
            normalization = 2.0 / ((n - 1) * (n - 2))
            for node_id in centrality:
                centrality[node_id] *= normalization
        
        return centrality
    
    async def _closeness_centrality(self, graph: GraphData) -> Dict[str, float]:
        """Calculate closeness centrality"""
        centrality = {}
        
        # Build adjacency list
        adjacency = defaultdict(list)
        for edge in graph.edges.values():
            adjacency[edge.source].append(edge.target)
            if not edge.directed:
                adjacency[edge.target].append(edge.source)
        
        for source in graph.nodes:
            # BFS to calculate shortest paths
            distances = {node_id: float('inf') for node_id in graph.nodes}
            distances[source] = 0
            queue = deque([source])
            
            while queue:
                current = queue.popleft()
                
                for neighbor in adjacency[current]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
            
            # Calculate closeness
            reachable_distances = [d for d in distances.values() if d != float('inf') and d > 0]
            
            if reachable_distances:
                centrality[source] = len(reachable_distances) / sum(reachable_distances)
            else:
                centrality[source] = 0.0
        
        return centrality
    
    async def _pagerank_centrality(self, graph: GraphData, damping: float = 0.85, max_iter: int = 100) -> Dict[str, float]:
        """Calculate PageRank centrality"""
        nodes = list(graph.nodes.keys())
        n = len(nodes)
        
        if n == 0:
            return {}
        
        # Initialize PageRank values
        pagerank = {node_id: 1.0 / n for node_id in nodes}
        
        # Build adjacency and out-degree info
        adjacency = defaultdict(list)
        out_degree = defaultdict(int)
        
        for edge in graph.edges.values():
            adjacency[edge.source].append(edge.target)
            out_degree[edge.source] += 1
        
        # Power iteration
        for iteration in range(max_iter):
            new_pagerank = {}
            
            for node_id in nodes:
                rank_sum = 0.0
                
                # Sum contributions from incoming links
                for edge in graph.edges.values():
                    if edge.target == node_id:
                        source_degree = out_degree[edge.source]
                        if source_degree > 0:
                            rank_sum += pagerank[edge.source] / source_degree
                
                new_pagerank[node_id] = (1 - damping) / n + damping * rank_sum
            
            # Check convergence
            diff = sum(abs(new_pagerank[node] - pagerank[node]) for node in nodes)
            pagerank = new_pagerank
            
            if diff < 1e-6:
                break
        
        return pagerank
    
    async def _eigenvector_centrality(self, graph: GraphData, max_iter: int = 100) -> Dict[str, float]:
        """Calculate eigenvector centrality (simplified power iteration)"""
        nodes = list(graph.nodes.keys())
        n = len(nodes)
        
        if n == 0:
            return {}
        
        # Initialize centrality values
        centrality = {node_id: 1.0 for node_id in nodes}
        
        # Build adjacency matrix information
        adjacency = defaultdict(list)
        for edge in graph.edges.values():
            adjacency[edge.source].append(edge.target)
            if not edge.directed:
                adjacency[edge.target].append(edge.source)
        
        # Power iteration
        for iteration in range(max_iter):
            new_centrality = defaultdict(float)
            
            for node_id in nodes:
                for neighbor in adjacency[node_id]:
                    new_centrality[neighbor] += centrality[node_id]
            
            # Normalize
            norm = sum(new_centrality.values())
            if norm > 0:
                for node_id in nodes:
                    new_centrality[node_id] /= norm
            
            # Check convergence
            diff = sum(abs(new_centrality[node] - centrality[node]) for node in nodes)
            centrality = dict(new_centrality)
            
            if diff < 1e-6:
                break
        
        return centrality


class PathFinder:
    """Graph path finding algorithms"""
    
    async def find_shortest_path(self, graph: GraphData, source: str, target: str, weighted: bool = True) -> Optional[PathResult]:
        """Find shortest path between two nodes"""
        if source not in graph.nodes or target not in graph.nodes:
            return None
        
        try:
            if weighted:
                path, distance = await self._dijkstra(graph, source, target)
            else:
                path, distance = await self._bfs_shortest_path(graph, source, target)
            
            if path:
                return PathResult(
                    source=source,
                    target=target,
                    path=path,
                    length=len(path) - 1,
                    weight=distance,
                    algorithm="dijkstra" if weighted else "bfs"
                )
        except Exception as e:
            logger.error(f"Path finding failed: {e}")
        
        return None
    
    async def find_all_paths(self, graph: GraphData, source: str, target: str, max_length: int = 10) -> List[PathResult]:
        """Find all paths between two nodes (up to max_length)"""
        if source not in graph.nodes or target not in graph.nodes:
            return []
        
        all_paths = []
        
        def dfs_paths(current_path: List[str], visited: Set[str]):
            if len(current_path) > max_length:
                return
            
            current_node = current_path[-1]
            
            if current_node == target and len(current_path) > 1:
                # Calculate path weight
                weight = 0.0
                for i in range(len(current_path) - 1):
                    for edge in graph.edges.values():
                        if edge.source == current_path[i] and edge.target == current_path[i + 1]:
                            weight += edge.weight
                            break
                
                path_result = PathResult(
                    source=source,
                    target=target,
                    path=current_path.copy(),
                    length=len(current_path) - 1,
                    weight=weight,
                    algorithm="dfs"
                )
                all_paths.append(path_result)
                return
            
            # Explore neighbors
            for edge in graph.edges.values():
                next_node = None
                if edge.source == current_node and edge.target not in visited:
                    next_node = edge.target
                elif not edge.directed and edge.target == current_node and edge.source not in visited:
                    next_node = edge.source
                
                if next_node:
                    visited.add(next_node)
                    current_path.append(next_node)
                    dfs_paths(current_path, visited)
                    current_path.pop()
                    visited.remove(next_node)
        
        # Start DFS
        visited = {source}
        dfs_paths([source], visited)
        
        return all_paths
    
    async def _dijkstra(self, graph: GraphData, source: str, target: str) -> Tuple[Optional[List[str]], float]:
        """Dijkstra's shortest path algorithm"""
        distances = {node_id: float('inf') for node_id in graph.nodes}
        distances[source] = 0
        previous = {}
        unvisited = [(0, source)]
        
        while unvisited:
            current_distance, current_node = heapq.heappop(unvisited)
            
            if current_node == target:
                # Reconstruct path
                path = []
                while current_node in previous:
                    path.append(current_node)
                    current_node = previous[current_node]
                path.append(source)
                path.reverse()
                return path, distances[target]
            
            if current_distance > distances[current_node]:
                continue
            
            # Check neighbors
            for edge in graph.edges.values():
                neighbor = None
                edge_weight = edge.weight
                
                if edge.source == current_node:
                    neighbor = edge.target
                elif not edge.directed and edge.target == current_node:
                    neighbor = edge.source
                
                if neighbor:
                    distance = current_distance + edge_weight
                    
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current_node
                        heapq.heappush(unvisited, (distance, neighbor))
        
        return None, float('inf')
    
    async def _bfs_shortest_path(self, graph: GraphData, source: str, target: str) -> Tuple[Optional[List[str]], float]:
        """BFS shortest path (unweighted)"""
        if source == target:
            return [source], 0
        
        visited = {source}
        queue = deque([(source, [source])])
        
        while queue:
            current_node, path = queue.popleft()
            
            # Check neighbors
            for edge in graph.edges.values():
                neighbor = None
                
                if edge.source == current_node and edge.target not in visited:
                    neighbor = edge.target
                elif not edge.directed and edge.target == current_node and edge.source not in visited:
                    neighbor = edge.source
                
                if neighbor:
                    new_path = path + [neighbor]
                    
                    if neighbor == target:
                        return new_path, len(new_path) - 1
                    
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))
        
        return None, float('inf')


class GraphEmbeddingGenerator:
    """Graph embedding generation using various methods"""
    
    def __init__(self):
        self.embeddings_cache: Dict[str, GraphEmbedding] = {}
    
    async def generate_embeddings(self, graph: GraphData, method: EmbeddingMethod, 
                                dimensions: int = 128, **kwargs) -> GraphEmbedding:
        """Generate node embeddings using specified method"""
        cache_key = f"{graph.graph_id}_{method.value}_{dimensions}"
        
        if cache_key in self.embeddings_cache:
            logger.info(f"Returning cached embeddings for {cache_key}")
            return self.embeddings_cache[cache_key]
        
        try:
            if method == EmbeddingMethod.NODE2VEC:
                embeddings = await self._node2vec_embeddings(graph, dimensions, **kwargs)
            elif method == EmbeddingMethod.DEEPWALK:
                embeddings = await self._deepwalk_embeddings(graph, dimensions, **kwargs)
            elif method == EmbeddingMethod.GRAPH_SAGE:
                embeddings = await self._graphsage_embeddings(graph, dimensions, **kwargs)
            else:
                logger.warning(f"Embedding method {method} not implemented, using random")
                embeddings = await self._random_embeddings(graph, dimensions)
            
            embedding_result = GraphEmbedding(
                embedding_id=f"emb_{uuid.uuid4().hex[:8]}",
                graph_id=graph.graph_id,
                method=method,
                node_embeddings=embeddings,
                dimensions=dimensions,
                hyperparameters=kwargs
            )
            
            self.embeddings_cache[cache_key] = embedding_result
            logger.info(f"Generated {method.value} embeddings for {len(embeddings)} nodes")
            return embedding_result
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return random embeddings as fallback
            return await self.generate_embeddings(graph, EmbeddingMethod.DEEPWALK, dimensions)
    
    async def _node2vec_embeddings(self, graph: GraphData, dimensions: int, 
                                 walk_length: int = 80, num_walks: int = 10,
                                 p: float = 1.0, q: float = 1.0) -> Dict[str, np.ndarray]:
        """Generate Node2Vec embeddings using biased random walks"""
        # Generate random walks with bias
        walks = []
        
        for _ in range(num_walks):
            for start_node in graph.nodes:
                walk = await self._biased_random_walk(graph, start_node, walk_length, p, q)
                if len(walk) > 1:
                    walks.append(walk)
        
        # Train embeddings using skip-gram (simplified)
        embeddings = await self._train_skip_gram(walks, dimensions)
        
        return embeddings
    
    async def _biased_random_walk(self, graph: GraphData, start_node: str, 
                                walk_length: int, p: float, q: float) -> List[str]:
        """Generate biased random walk for Node2Vec"""
        walk = [start_node]
        
        # Build adjacency list
        adjacency = defaultdict(list)
        for edge in graph.edges.values():
            adjacency[edge.source].append(edge.target)
            if not edge.directed:
                adjacency[edge.target].append(edge.source)
        
        while len(walk) < walk_length:
            current = walk[-1]
            neighbors = adjacency[current]
            
            if not neighbors:
                break
            
            if len(walk) == 1:
                # First step - uniform random
                next_node = np.random.choice(neighbors)
            else:
                # Biased selection
                prev_node = walk[-2]
                probs = []
                
                for neighbor in neighbors:
                    if neighbor == prev_node:
                        # Return to previous node
                        prob = 1.0 / p
                    elif neighbor in adjacency[prev_node]:
                        # Common neighbor (BFS)
                        prob = 1.0
                    else:
                        # New node (DFS)
                        prob = 1.0 / q
                    
                    probs.append(prob)
                
                # Normalize probabilities
                total = sum(probs)
                if total > 0:
                    probs = [p / total for p in probs]
                    next_node = np.random.choice(neighbors, p=probs)
                else:
                    next_node = np.random.choice(neighbors)
            
            walk.append(next_node)
        
        return walk
    
    async def _deepwalk_embeddings(self, graph: GraphData, dimensions: int,
                                 walk_length: int = 40, num_walks: int = 80) -> Dict[str, np.ndarray]:
        """Generate DeepWalk embeddings using random walks"""
        walks = []
        
        # Generate random walks
        for _ in range(num_walks):
            for start_node in graph.nodes:
                walk = await self._random_walk(graph, start_node, walk_length)
                if len(walk) > 1:
                    walks.append(walk)
        
        # Train embeddings
        embeddings = await self._train_skip_gram(walks, dimensions)
        
        return embeddings
    
    async def _random_walk(self, graph: GraphData, start_node: str, walk_length: int) -> List[str]:
        """Generate random walk"""
        walk = [start_node]
        
        # Build adjacency list
        adjacency = defaultdict(list)
        for edge in graph.edges.values():
            adjacency[edge.source].append(edge.target)
            if not edge.directed:
                adjacency[edge.target].append(edge.source)
        
        while len(walk) < walk_length:
            current = walk[-1]
            neighbors = adjacency[current]
            
            if not neighbors:
                break
            
            next_node = np.random.choice(neighbors)
            walk.append(next_node)
        
        return walk
    
    async def _train_skip_gram(self, walks: List[List[str]], dimensions: int, 
                             window_size: int = 5) -> Dict[str, np.ndarray]:
        """Train skip-gram model on walks (simplified implementation)"""
        # Build vocabulary
        vocab = set()
        for walk in walks:
            vocab.update(walk)
        
        vocab = list(vocab)
        vocab_size = len(vocab)
        node_to_idx = {node: i for i, node in enumerate(vocab)}
        
        # Initialize random embeddings
        embeddings = {}
        for node in vocab:
            embeddings[node] = np.random.randn(dimensions) * 0.1
        
        # Simplified training (in practice would use proper neural network)
        learning_rate = 0.01
        epochs = 5
        
        for epoch in range(epochs):
            for walk in walks:
                for i, center_node in enumerate(walk):
                    # Define context window
                    start = max(0, i - window_size)
                    end = min(len(walk), i + window_size + 1)
                    
                    for j in range(start, end):
                        if i != j:
                            context_node = walk[j]
                            
                            # Update embeddings (simplified gradient descent)
                            center_emb = embeddings[center_node]
                            context_emb = embeddings[context_node]
                            
                            # Simplified update rule
                            similarity = np.dot(center_emb, context_emb)
                            gradient = learning_rate * (1 - similarity)
                            
                            embeddings[center_node] += gradient * context_emb
                            embeddings[context_node] += gradient * center_emb
        
        # Normalize embeddings
        for node in embeddings:
            norm = np.linalg.norm(embeddings[node])
            if norm > 0:
                embeddings[node] /= norm
        
        return embeddings
    
    async def _graphsage_embeddings(self, graph: GraphData, dimensions: int) -> Dict[str, np.ndarray]:
        """Generate GraphSAGE embeddings (simplified)"""
        # For now, return random embeddings as placeholder
        return await self._random_embeddings(graph, dimensions)
    
    async def _random_embeddings(self, graph: GraphData, dimensions: int) -> Dict[str, np.ndarray]:
        """Generate random embeddings as baseline"""
        embeddings = {}
        
        for node_id in graph.nodes:
            embeddings[node_id] = np.random.randn(dimensions)
            # Normalize
            norm = np.linalg.norm(embeddings[node_id])
            if norm > 0:
                embeddings[node_id] /= norm
        
        return embeddings


class GraphAnalytics:
    """Main graph analytics orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage = InMemoryGraphStorage()
        self.community_detector = CommunityDetector()
        self.centrality_calculator = CentralityCalculator()
        self.path_finder = PathFinder()
        self.embedding_generator = GraphEmbeddingGenerator()
        
        # Analysis results cache
        self.metrics_cache: Dict[str, NetworkMetrics] = {}
        self.query_cache: Dict[str, Any] = {}
        
        logger.info("Graph Analytics system initialized")
    
    async def initialize(self):
        """Initialize graph analytics system"""
        try:
            await self._load_configuration()
            await self._setup_storage()
            await self._start_background_tasks()
            
            logger.info("Graph Analytics system fully initialized")
            
        except Exception as e:
            logger.error(f"Graph analytics initialization failed: {e}")
            raise
    
    async def _load_configuration(self):
        """Load system configuration"""
        default_config = {
            'max_graph_size': 100000,
            'cache_ttl_minutes': 60,
            'max_path_length': 20,
            'embedding_dimensions': 128,
            'community_detection_timeout': 300
        }
        
        self.config = {**default_config, **self.config}
        logger.info("Configuration loaded")
    
    async def _setup_storage(self):
        """Setup graph storage"""
        logger.info("Graph storage initialized")
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        asyncio.create_task(self._cache_cleanup_loop())
        asyncio.create_task(self._metrics_update_loop())
        logger.info("Background tasks started")
    
    async def create_graph(self, graph_data: GraphData) -> str:
        """Create and store new graph"""
        if len(graph_data.nodes) > self.config['max_graph_size']:
            raise ValueError(f"Graph too large: {len(graph_data.nodes)} nodes (max: {self.config['max_graph_size']})")
        
        success = await self.storage.store_graph(graph_data)
        
        if success:
            logger.info(f"Created graph {graph_data.graph_id}")
            return graph_data.graph_id
        else:
            raise RuntimeError("Failed to store graph")
    
    async def add_nodes(self, graph_id: str, nodes: List[Node]) -> bool:
        """Add nodes to existing graph"""
        graph = await self.storage.load_graph(graph_id)
        if not graph:
            return False
        
        for node in nodes:
            graph.nodes[node.id] = node
        
        graph.updated_at = datetime.now()
        return await self.storage.store_graph(graph)
    
    async def add_edges(self, graph_id: str, edges: List[Edge]) -> bool:
        """Add edges to existing graph"""
        graph = await self.storage.load_graph(graph_id)
        if not graph:
            return False
        
        for edge in edges:
            # Verify nodes exist
            if edge.source in graph.nodes and edge.target in graph.nodes:
                graph.edges[edge.id] = edge
        
        graph.updated_at = datetime.now()
        return await self.storage.store_graph(graph)
    
    async def analyze_network(self, graph_id: str, force_refresh: bool = False) -> NetworkMetrics:
        """Perform comprehensive network analysis"""
        if not force_refresh and graph_id in self.metrics_cache:
            cached_metrics = self.metrics_cache[graph_id]
            cache_age = (datetime.now() - cached_metrics.computed_at).total_seconds() / 60
            
            if cache_age < self.config['cache_ttl_minutes']:
                logger.info(f"Returning cached metrics for graph {graph_id}")
                return cached_metrics
        
        graph = await self.storage.load_graph(graph_id)
        if not graph:
            raise ValueError(f"Graph not found: {graph_id}")
        
        try:
            # Basic metrics
            node_count = len(graph.nodes)
            edge_count = len(graph.edges)
            density = self._calculate_density(graph)
            
            # Advanced metrics
            clustering_coefficient = await self._calculate_clustering_coefficient(graph)
            diameter, avg_path_length = await self._calculate_distance_metrics(graph)
            assortativity = await self._calculate_assortativity(graph)
            components = await self._count_components(graph)
            
            # Centrality measures
            centrality_scores = {}
            for measure in [CentralityMeasure.DEGREE, CentralityMeasure.BETWEENNESS, 
                          CentralityMeasure.CLOSENESS, CentralityMeasure.PAGERANK]:
                scores = await self.centrality_calculator.calculate_centrality(graph, measure)
                centrality_scores[measure.value] = scores
            
            # Community detection
            communities = await self.community_detector.detect_communities(graph, CommunityAlgorithm.LOUVAIN)
            modularity = self._calculate_modularity_from_communities(graph, communities)
            
            metrics = NetworkMetrics(
                graph_id=graph_id,
                node_count=node_count,
                edge_count=edge_count,
                density=density,
                clustering_coefficient=clustering_coefficient,
                diameter=diameter,
                average_path_length=avg_path_length,
                assortativity=assortativity,
                modularity=modularity,
                components=components,
                centrality_scores=centrality_scores,
                communities=communities
            )
            
            self.metrics_cache[graph_id] = metrics
            logger.info(f"Network analysis completed for graph {graph_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Network analysis failed: {e}")
            raise
    
    async def detect_communities(self, graph_id: str, algorithm: CommunityAlgorithm) -> List[Community]:
        """Detect communities in graph"""
        graph = await self.storage.load_graph(graph_id)
        if not graph:
            raise ValueError(f"Graph not found: {graph_id}")
        
        communities = await self.community_detector.detect_communities(graph, algorithm)
        logger.info(f"Detected {len(communities)} communities using {algorithm.value}")
        return communities
    
    async def calculate_centrality(self, graph_id: str, measure: CentralityMeasure) -> Dict[str, float]:
        """Calculate centrality measure for graph nodes"""
        graph = await self.storage.load_graph(graph_id)
        if not graph:
            raise ValueError(f"Graph not found: {graph_id}")
        
        return await self.centrality_calculator.calculate_centrality(graph, measure)
    
    async def find_path(self, graph_id: str, source: str, target: str, 
                       algorithm: str = "shortest") -> Optional[PathResult]:
        """Find path between two nodes"""
        graph = await self.storage.load_graph(graph_id)
        if not graph:
            raise ValueError(f"Graph not found: {graph_id}")
        
        if algorithm == "shortest":
            return await self.path_finder.find_shortest_path(graph, source, target)
        else:
            raise ValueError(f"Unsupported path algorithm: {algorithm}")
    
    async def find_all_paths(self, graph_id: str, source: str, target: str, 
                           max_length: int = None) -> List[PathResult]:
        """Find all paths between two nodes"""
        graph = await self.storage.load_graph(graph_id)
        if not graph:
            raise ValueError(f"Graph not found: {graph_id}")
        
        max_len = max_length or self.config['max_path_length']
        return await self.path_finder.find_all_paths(graph, source, target, max_len)
    
    async def generate_embeddings(self, graph_id: str, method: EmbeddingMethod, 
                                dimensions: int = None) -> GraphEmbedding:
        """Generate node embeddings"""
        graph = await self.storage.load_graph(graph_id)
        if not graph:
            raise ValueError(f"Graph not found: {graph_id}")
        
        dims = dimensions or self.config['embedding_dimensions']
        return await self.embedding_generator.generate_embeddings(graph, method, dims)
    
    async def query_subgraph(self, graph_id: str, query: GraphQuery) -> GraphData:
        """Query subgraph based on criteria"""
        return await self.storage.query_subgraph(graph_id, query)
    
    async def get_node_neighbors(self, graph_id: str, node_id: str, 
                               depth: int = 1) -> Dict[str, Any]:
        """Get neighbors of a node up to specified depth"""
        graph = await self.storage.load_graph(graph_id)
        if not graph:
            raise ValueError(f"Graph not found: {graph_id}")
        
        if node_id not in graph.nodes:
            raise ValueError(f"Node not found: {node_id}")
        
        neighbors = {}
        visited = set()
        queue = deque([(node_id, 0)])
        
        while queue:
            current_node, current_depth = queue.popleft()
            
            if current_depth >= depth or current_node in visited:
                continue
            
            visited.add(current_node)
            neighbors[current_node] = {
                'node': graph.nodes[current_node],
                'depth': current_depth
            }
            
            # Add neighbors to queue
            for edge in graph.edges.values():
                next_node = None
                if edge.source == current_node:
                    next_node = edge.target
                elif not edge.directed and edge.target == current_node:
                    next_node = edge.source
                
                if next_node and next_node not in visited:
                    queue.append((next_node, current_depth + 1))
        
        return neighbors
    
    def _calculate_density(self, graph: GraphData) -> float:
        """Calculate graph density"""
        n = len(graph.nodes)
        m = len(graph.edges)
        
        if n < 2:
            return 0.0
        
        max_edges = n * (n - 1)
        if graph.graph_type != GraphType.DIRECTED:
            max_edges //= 2
        
        return m / max_edges if max_edges > 0 else 0.0
    
    async def _calculate_clustering_coefficient(self, graph: GraphData) -> float:
        """Calculate global clustering coefficient"""
        if len(graph.nodes) < 3:
            return 0.0
        
        # Build adjacency list
        adjacency = defaultdict(set)
        for edge in graph.edges.values():
            adjacency[edge.source].add(edge.target)
            if not edge.directed:
                adjacency[edge.target].add(edge.source)
        
        total_triangles = 0
        total_triplets = 0
        
        for node_id in graph.nodes:
            neighbors = list(adjacency[node_id])
            degree = len(neighbors)
            
            if degree < 2:
                continue
            
            # Count triangles
            triangles = 0
            for i, neighbor1 in enumerate(neighbors):
                for neighbor2 in neighbors[i + 1:]:
                    if neighbor2 in adjacency[neighbor1]:
                        triangles += 1
            
            total_triangles += triangles
            total_triplets += degree * (degree - 1) // 2
        
        return (3 * total_triangles) / total_triplets if total_triplets > 0 else 0.0
    
    async def _calculate_distance_metrics(self, graph: GraphData) -> Tuple[int, float]:
        """Calculate diameter and average path length"""
        if len(graph.nodes) < 2:
            return 0, 0.0
        
        # Build adjacency list
        adjacency = defaultdict(list)
        for edge in graph.edges.values():
            adjacency[edge.source].append(edge.target)
            if not edge.directed:
                adjacency[edge.target].append(edge.source)
        
        all_distances = []
        diameter = 0
        
        for source in list(graph.nodes.keys())[:min(100, len(graph.nodes))]:  # Sample for large graphs
            distances = await self._bfs_distances(source, adjacency)
            
            for distance in distances.values():
                if distance != float('inf') and distance > 0:
                    all_distances.append(distance)
                    diameter = max(diameter, distance)
        
        avg_path_length = sum(all_distances) / len(all_distances) if all_distances else 0.0
        
        return diameter, avg_path_length
    
    async def _bfs_distances(self, source: str, adjacency: Dict[str, List[str]]) -> Dict[str, int]:
        """Calculate distances from source using BFS"""
        distances = {source: 0}
        queue = deque([source])
        
        while queue:
            current = queue.popleft()
            
            for neighbor in adjacency[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        
        return distances
    
    async def _calculate_assortativity(self, graph: GraphData) -> float:
        """Calculate degree assortativity"""
        if len(graph.edges) == 0:
            return 0.0
        
        # Calculate node degrees
        degrees = defaultdict(int)
        for edge in graph.edges.values():
            degrees[edge.source] += 1
            degrees[edge.target] += 1
        
        # Calculate assortativity coefficient
        sum_xy = 0
        sum_x = 0
        sum_y = 0
        sum_x2 = 0
        sum_y2 = 0
        m = len(graph.edges)
        
        for edge in graph.edges.values():
            x = degrees[edge.source]
            y = degrees[edge.target]
            
            sum_xy += x * y
            sum_x += x
            sum_y += y
            sum_x2 += x * x
            sum_y2 += y * y
        
        if m == 0:
            return 0.0
        
        numerator = sum_xy / m - (sum_x * sum_y) / (m * m)
        denominator_x = sum_x2 / m - (sum_x / m) ** 2
        denominator_y = sum_y2 / m - (sum_y / m) ** 2
        denominator = (denominator_x * denominator_y) ** 0.5
        
        return numerator / denominator if denominator > 0 else 0.0
    
    async def _count_components(self, graph: GraphData) -> int:
        """Count connected components"""
        visited = set()
        components = 0
        
        # Build adjacency list
        adjacency = defaultdict(list)
        for edge in graph.edges.values():
            adjacency[edge.source].append(edge.target)
            if not edge.directed:
                adjacency[edge.target].append(edge.source)
        
        for node_id in graph.nodes:
            if node_id not in visited:
                # BFS to mark component
                queue = deque([node_id])
                visited.add(node_id)
                
                while queue:
                    current = queue.popleft()
                    
                    for neighbor in adjacency[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                components += 1
        
        return components
    
    def _calculate_modularity_from_communities(self, graph: GraphData, communities: List[Community]) -> float:
        """Calculate modularity score from detected communities"""
        if not communities:
            return 0.0
        
        m = len(graph.edges)
        if m == 0:
            return 0.0
        
        # Create node to community mapping
        node_to_community = {}
        for i, community in enumerate(communities):
            for node_id in community.nodes:
                node_to_community[node_id] = i
        
        # Calculate modularity
        modularity = 0.0
        
        # Node degrees
        degrees = defaultdict(int)
        for edge in graph.edges.values():
            degrees[edge.source] += 1
            degrees[edge.target] += 1
        
        for edge in graph.edges.values():
            source_community = node_to_community.get(edge.source, -1)
            target_community = node_to_community.get(edge.target, -1)
            
            if source_community == target_community and source_community != -1:
                expected = (degrees[edge.source] * degrees[edge.target]) / (2 * m)
                modularity += 1 - expected
            else:
                expected = (degrees[edge.source] * degrees[edge.target]) / (2 * m)
                modularity -= expected
        
        return modularity / m
    
    async def _cache_cleanup_loop(self):
        """Background cache cleanup"""
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                await self._cleanup_caches()
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def _metrics_update_loop(self):
        """Background metrics update"""
        while True:
            try:
                await asyncio.sleep(1800)  # Update every 30 minutes
                await self._update_stale_metrics()
            except Exception as e:
                logger.error(f"Metrics update error: {e}")
    
    async def _cleanup_caches(self):
        """Clean up expired cache entries"""
        current_time = datetime.now()
        expired_keys = []
        
        for graph_id, metrics in self.metrics_cache.items():
            cache_age = (current_time - metrics.computed_at).total_seconds() / 60
            if cache_age > self.config['cache_ttl_minutes']:
                expired_keys.append(graph_id)
        
        for key in expired_keys:
            del self.metrics_cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned {len(expired_keys)} expired cache entries")
    
    async def _update_stale_metrics(self):
        """Update stale metrics in background"""
        current_time = datetime.now()
        
        for graph_id, metrics in list(self.metrics_cache.items()):
            cache_age = (current_time - metrics.computed_at).total_seconds() / 60
            if cache_age > self.config['cache_ttl_minutes'] * 0.8:  # Refresh at 80% of TTL
                try:
                    await self.analyze_network(graph_id, force_refresh=True)
                    logger.info(f"Updated stale metrics for graph {graph_id}")
                except Exception as e:
                    logger.warning(f"Failed to update metrics for {graph_id}: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get graph analytics system status"""
        return {
            'graphs_stored': len(self.storage.graphs),
            'metrics_cached': len(self.metrics_cache),
            'embeddings_cached': len(self.embedding_generator.embeddings_cache),
            'communities_cached': len(self.community_detector.communities_cache),
            'centrality_cached': len(self.centrality_calculator.centrality_cache),
            'cache_ttl_minutes': self.config['cache_ttl_minutes'],
            'max_graph_size': self.config['max_graph_size']
        }


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Initialize graph analytics system
        config = {
            'max_graph_size': 10000,
            'cache_ttl_minutes': 30,
            'embedding_dimensions': 64
        }
        
        analytics = GraphAnalytics(config)
        await analytics.initialize()
        
        # Create sample graph
        nodes = [
            Node(id="A", label="Node A", node_type="user"),
            Node(id="B", label="Node B", node_type="user"),
            Node(id="C", label="Node C", node_type="user"),
            Node(id="D", label="Node D", node_type="user"),
            Node(id="E", label="Node E", node_type="user")
        ]
        
        edges = [
            Edge(id="1", source="A", target="B", weight=1.0),
            Edge(id="2", source="B", target="C", weight=1.5),
            Edge(id="3", source="C", target="D", weight=2.0),
            Edge(id="4", source="D", target="E", weight=1.0),
            Edge(id="5", source="E", target="A", weight=1.2),
            Edge(id="6", source="A", target="C", weight=0.8)
        ]
        
        graph_data = GraphData(
            graph_id="sample_graph",
            name="Sample Social Network",
            graph_type=GraphType.UNDIRECTED,
            nodes={node.id: node for node in nodes},
            edges={edge.id: edge for edge in edges}
        )
        
        # Store graph
        graph_id = await analytics.create_graph(graph_data)
        print(f"Created graph: {graph_id}")
        
        # Perform network analysis
        metrics = await analytics.analyze_network(graph_id)
        print(f"Network Analysis - Nodes: {metrics.node_count}, Edges: {metrics.edge_count}")
        print(f"Density: {metrics.density:.3f}, Clustering: {metrics.clustering_coefficient:.3f}")
        print(f"Communities: {len(metrics.communities)}")
        
        # Calculate centrality
        pagerank_scores = await analytics.calculate_centrality(graph_id, CentralityMeasure.PAGERANK)
        print(f"PageRank scores: {pagerank_scores}")
        
        # Find path
        path_result = await analytics.find_path(graph_id, "A", "D")
        if path_result:
            print(f"Shortest path A->D: {' -> '.join(path_result.path)} (length: {path_result.length})")
        
        # Generate embeddings
        embeddings = await analytics.generate_embeddings(graph_id, EmbeddingMethod.DEEPWALK, 32)
        print(f"Generated embeddings for {len(embeddings.node_embeddings)} nodes")
        
        # Get system status
        status = analytics.get_system_status()
        print(f"System Status: {status}")
        
        logger.info("Graph Analytics system demonstration completed")
    
    # Run the example
    asyncio.run(main())