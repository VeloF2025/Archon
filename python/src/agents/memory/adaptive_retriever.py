#!/usr/bin/env python3
"""
Adaptive Retriever with Bandit Optimization for Archon+ Phase 4

Implements bandit algorithm for optimal strategy selection:
- Vector Search: Embedding similarity
- Hybrid Search: Vector + Keyword combination
- Graphiti Search: Entity/relationship traversal  
- Memory Search: Role-specific context retrieval

Performance requirements: ≥85% retrieval precision
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import math

from .memory_service import MemoryService, MemoryLayerType

logger = logging.getLogger(__name__)

class RetrievalStrategyType(Enum):
    """Available retrieval strategies"""
    VECTOR_SEARCH = "vector_search"
    HYBRID_SEARCH = "hybrid_search"
    GRAPHITI_SEARCH = "graphiti_search"
    MEMORY_SEARCH = "memory_search"

@dataclass
class PerformanceMetrics:
    """Performance tracking for retrieval strategies"""
    total_queries: int = 0
    successful_queries: int = 0
    total_response_time: float = 0.0
    precision_scores: List[float] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        return self.successful_queries / max(1, self.total_queries)
    
    @property
    def avg_response_time(self) -> float:
        """Calculate average response time"""
        return self.total_response_time / max(1, self.total_queries)
    
    @property
    def avg_precision(self) -> float:
        """Calculate average precision"""
        return sum(self.precision_scores) / max(1, len(self.precision_scores))

@dataclass
class RetrievalStrategy:
    """Retrieval strategy with bandit algorithm state"""
    strategy_name: str
    strategy_type: RetrievalStrategyType
    enabled: bool = True
    
    # Bandit algorithm parameters
    weight: float = 1.0          # Selection weight
    performance_score: float = 0.5  # Historical performance (0.0-1.0)
    confidence_bound: float = 0.0   # Upper confidence bound
    
    # Performance tracking
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    
    # Strategy-specific configuration
    config: Dict[str, Any] = field(default_factory=dict)

class BanditAlgorithm:
    """Upper Confidence Bound (UCB) bandit algorithm for strategy selection"""
    
    def __init__(self, exploration_factor: float = 1.4):
        """
        Initialize bandit algorithm
        
        Args:
            exploration_factor: UCB exploration parameter (higher = more exploration)
        """
        self.exploration_factor = exploration_factor
        self.total_rounds = 0
    
    def calculate_ucb_score(self, strategy: RetrievalStrategy) -> float:
        """
        Calculate Upper Confidence Bound score for a strategy
        
        Args:
            strategy: RetrievalStrategy to evaluate
            
        Returns:
            UCB score (higher = better to select)
        """
        if strategy.metrics.total_queries == 0:
            # High score for unexplored strategies
            return float('inf')
        
        # Calculate confidence bound
        confidence_term = math.sqrt(
            (2 * math.log(self.total_rounds + 1)) / strategy.metrics.total_queries
        )
        
        # UCB score = average reward + confidence bound
        ucb_score = strategy.performance_score + (self.exploration_factor * confidence_term)
        strategy.confidence_bound = confidence_term
        
        return ucb_score
    
    def select_strategies(self, strategies: List[RetrievalStrategy], 
                         max_strategies: int = 3) -> List[RetrievalStrategy]:
        """
        Select optimal strategies using UCB algorithm
        
        Args:
            strategies: Available strategies
            max_strategies: Maximum number of strategies to select
            
        Returns:
            Selected strategies ordered by UCB score
        """
        self.total_rounds += 1
        
        # Calculate UCB scores for enabled strategies
        enabled_strategies = [s for s in strategies if s.enabled]
        strategy_scores = [
            (strategy, self.calculate_ucb_score(strategy))
            for strategy in enabled_strategies
        ]
        
        # Sort by UCB score (descending)
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top strategies
        selected = [strategy for strategy, _ in strategy_scores[:max_strategies]]
        
        logger.debug(f"Selected {len(selected)} strategies: {[s.strategy_name for s in selected]}")
        return selected
    
    def update_performance(self, strategy: RetrievalStrategy, 
                          precision: float, response_time: float, success: bool):
        """
        Update strategy performance based on query results
        
        Args:
            strategy: Strategy to update
            precision: Precision score (0.0-1.0)
            response_time: Query response time in seconds
            success: Whether query was successful
        """
        metrics = strategy.metrics
        
        # Update basic metrics
        metrics.total_queries += 1
        if success:
            metrics.successful_queries += 1
        metrics.total_response_time += response_time
        
        # Update precision scores (keep last 100 for efficiency)
        metrics.precision_scores.append(precision)
        if len(metrics.precision_scores) > 100:
            metrics.precision_scores.pop(0)
        
        # Calculate new performance score (weighted combination)
        success_weight = 0.3
        precision_weight = 0.5
        speed_weight = 0.2
        
        # Normalize response time (lower is better, max 5 seconds)
        normalized_speed = max(0, 1 - (response_time / 5.0))
        
        strategy.performance_score = (
            success_weight * metrics.success_rate +
            precision_weight * metrics.avg_precision +
            speed_weight * normalized_speed
        )
        
        metrics.last_updated = time.time()
        
        logger.debug(f"Updated {strategy.strategy_name}: score={strategy.performance_score:.3f}, "
                    f"success_rate={metrics.success_rate:.3f}, precision={metrics.avg_precision:.3f}")

class AdaptiveRetriever:
    """
    Adaptive retrieval system with bandit optimization
    
    Combines multiple retrieval strategies using bandit algorithms
    to optimize for precision and performance.
    """
    
    def __init__(self, memory_service: Optional[MemoryService] = None,
                 rag_service=None, config_path: Optional[Path] = None):
        """
        Initialize adaptive retriever
        
        Args:
            memory_service: Memory service for memory layer search
            rag_service: RAG service for vector/hybrid search
            config_path: Path to strategy configuration file
        """
        self.memory_service = memory_service
        self.rag_service = rag_service
        self.config_path = config_path or Path("python/src/agents/memory/retrieval_config.json")
        
        # Initialize bandit algorithm
        self.bandit = BanditAlgorithm(exploration_factor=1.4)
        
        # Initialize strategies
        self.strategies: Dict[RetrievalStrategyType, RetrievalStrategy] = {}
        self._initialize_strategies()
        self._load_configuration()
        
        # Query history for analysis
        self.query_history: List[Dict[str, Any]] = []
    
    def _initialize_strategies(self):
        """Initialize retrieval strategies with default configurations"""
        
        # Vector Search Strategy
        self.strategies[RetrievalStrategyType.VECTOR_SEARCH] = RetrievalStrategy(
            strategy_name="Vector Search",
            strategy_type=RetrievalStrategyType.VECTOR_SEARCH,
            performance_score=0.75,  # Good baseline performance
            config={
                "embedding_model": "text-embedding-ada-002",
                "similarity_threshold": 0.7,
                "max_results": 10
            }
        )
        
        # Hybrid Search Strategy  
        self.strategies[RetrievalStrategyType.HYBRID_SEARCH] = RetrievalStrategy(
            strategy_name="Hybrid Search",
            strategy_type=RetrievalStrategyType.HYBRID_SEARCH,
            performance_score=0.82,  # Better than pure vector
            config={
                "vector_weight": 0.7,
                "keyword_weight": 0.3,
                "reranking_enabled": True,
                "max_results": 10
            }
        )
        
        # Graphiti Search Strategy
        self.strategies[RetrievalStrategyType.GRAPHITI_SEARCH] = RetrievalStrategy(
            strategy_name="Graphiti Search",
            strategy_type=RetrievalStrategyType.GRAPHITI_SEARCH,
            performance_score=0.90,  # Highest expected performance
            config={
                "entity_types": ["function", "class", "concept"],
                "relationship_depth": 2,
                "confidence_threshold": 0.8,
                "max_results": 8
            }
        )
        
        # Memory Search Strategy
        self.strategies[RetrievalStrategyType.MEMORY_SEARCH] = RetrievalStrategy(
            strategy_name="Memory Search",
            strategy_type=RetrievalStrategyType.MEMORY_SEARCH,
            performance_score=0.85,  # High for contextual queries
            enabled=self.memory_service is not None,
            config={
                "search_layers": ["project", "job", "global"],
                "importance_threshold": 0.3,
                "max_results": 8
            }
        )
    
    def _load_configuration(self):
        """Load strategy configurations from file"""
        if not self.config_path.exists():
            return
        
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            for strategy_type_str, strategy_data in config_data.get("strategies", {}).items():
                try:
                    strategy_type = RetrievalStrategyType(strategy_type_str)
                    if strategy_type in self.strategies:
                        strategy = self.strategies[strategy_type]
                        
                        # Update strategy configuration
                        strategy.enabled = strategy_data.get("enabled", strategy.enabled)
                        strategy.weight = strategy_data.get("weight", strategy.weight)
                        strategy.performance_score = strategy_data.get("performance_score", strategy.performance_score)
                        strategy.config.update(strategy_data.get("config", {}))
                        
                        # Load performance metrics
                        metrics_data = strategy_data.get("metrics", {})
                        if metrics_data:
                            strategy.metrics.total_queries = metrics_data.get("total_queries", 0)
                            strategy.metrics.successful_queries = metrics_data.get("successful_queries", 0)
                            strategy.metrics.total_response_time = metrics_data.get("total_response_time", 0.0)
                            strategy.metrics.precision_scores = metrics_data.get("precision_scores", [])
                
                except ValueError:
                    logger.warning(f"Unknown strategy type in config: {strategy_type_str}")
            
            logger.info(f"Loaded retrieval strategy configurations from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load retrieval configuration: {e}")
    
    def _save_configuration(self):
        """Save current strategy configurations to file"""
        try:
            config_data = {
                "strategies": {},
                "last_updated": time.time()
            }
            
            for strategy_type, strategy in self.strategies.items():
                config_data["strategies"][strategy_type.value] = {
                    "enabled": strategy.enabled,
                    "weight": strategy.weight,
                    "performance_score": strategy.performance_score,
                    "config": strategy.config,
                    "metrics": {
                        "total_queries": strategy.metrics.total_queries,
                        "successful_queries": strategy.metrics.successful_queries,
                        "total_response_time": strategy.metrics.total_response_time,
                        "precision_scores": strategy.metrics.precision_scores[-50:],  # Keep last 50
                        "last_updated": strategy.metrics.last_updated
                    }
                }
            
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            logger.debug(f"Saved retrieval configurations to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save retrieval configuration: {e}")
    
    async def _execute_vector_search(self, query: str, strategy: RetrievalStrategy,
                                   **kwargs) -> List[Dict[str, Any]]:
        """Execute vector search strategy"""
        if not self.rag_service:
            logger.warning("RAG service not available for vector search")
            return []
        
        try:
            max_results = strategy.config.get("max_results", 10)
            results = await self.rag_service.search_documents(
                query=query,
                match_count=max_results,
                filter_metadata=kwargs.get("filter_metadata"),
                use_hybrid_search=False
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {}),
                    "score": result.get("similarity", 0.0),
                    "source": "vector_search",
                    "strategy": strategy.strategy_name
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def _execute_hybrid_search(self, query: str, strategy: RetrievalStrategy,
                                   **kwargs) -> List[Dict[str, Any]]:
        """Execute hybrid search strategy"""
        if not self.rag_service:
            logger.warning("RAG service not available for hybrid search")
            return []
        
        try:
            max_results = strategy.config.get("max_results", 10)
            results = await self.rag_service.search_documents(
                query=query,
                match_count=max_results,
                filter_metadata=kwargs.get("filter_metadata"),
                use_hybrid_search=True
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {}),
                    "score": result.get("similarity", 0.0),
                    "source": "hybrid_search",
                    "strategy": strategy.strategy_name
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    async def _execute_memory_search(self, query: str, strategy: RetrievalStrategy,
                                   agent_role: str, **kwargs) -> List[Dict[str, Any]]:
        """Execute memory search strategy"""
        if not self.memory_service:
            logger.warning("Memory service not available for memory search")
            return []
        
        try:
            search_layers = strategy.config.get("search_layers", ["project", "job"])
            max_results = strategy.config.get("max_results", 8)
            importance_threshold = strategy.config.get("importance_threshold", 0.3)
            
            all_results = []
            
            # Search across specified memory layers
            for layer_name in search_layers:
                try:
                    layer_type = MemoryLayerType(layer_name)
                    layer_results = await self.memory_service.query(
                        query_text=query,
                        memory_layer=layer_type,
                        agent_role=agent_role,
                        limit=max_results // len(search_layers) + 1
                    )
                    
                    # Filter by importance and format results
                    for entry in layer_results:
                        if entry.importance_score >= importance_threshold:
                            all_results.append({
                                "content": str(entry.content),
                                "metadata": {
                                    **entry.metadata,
                                    "memory_layer": entry.memory_layer.value,
                                    "source_agent": entry.source_agent,
                                    "importance_score": entry.importance_score,
                                    "access_count": entry.access_count
                                },
                                "score": entry.importance_score,
                                "source": "memory_search",
                                "strategy": strategy.strategy_name
                            })
                
                except ValueError:
                    logger.warning(f"Invalid memory layer: {layer_name}")
                    continue
            
            # Sort by score and limit results
            all_results.sort(key=lambda x: x["score"], reverse=True)
            return all_results[:max_results]
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []
    
    async def _execute_graphiti_search(self, query: str, strategy: RetrievalStrategy,
                                     **kwargs) -> List[Dict[str, Any]]:
        """Execute Graphiti search strategy (placeholder for now)"""
        # TODO: Implement when Graphiti service is available
        logger.debug("Graphiti search not yet implemented - returning placeholder results")
        
        # Placeholder implementation
        return [{
            "content": f"Graphiti placeholder result for: {query}",
            "metadata": {"entity_type": "placeholder", "confidence": 0.8},
            "score": 0.8,
            "source": "graphiti_search",
            "strategy": strategy.strategy_name
        }]
    
    async def retrieve(self, query: str, agent_role: str, 
                      max_strategies: int = 3,
                      **kwargs) -> Dict[str, Any]:
        """
        Retrieve relevant content using adaptive strategy selection
        
        Args:
            query: Search query
            agent_role: Role of requesting agent
            max_strategies: Maximum number of strategies to use
            **kwargs: Additional parameters for strategies
            
        Returns:
            Dictionary containing results and metadata
        """
        start_time = time.time()
        
        # Select optimal strategies using bandit algorithm
        strategy_list = list(self.strategies.values())
        selected_strategies = self.bandit.select_strategies(strategy_list, max_strategies)
        
        if not selected_strategies:
            logger.warning("No strategies available for retrieval")
            return {"results": [], "strategies_used": [], "total_time": 0.0}
        
        # Execute selected strategies in parallel
        strategy_results = {}
        tasks = []
        
        for strategy in selected_strategies:
            if strategy.strategy_type == RetrievalStrategyType.VECTOR_SEARCH:
                task = self._execute_vector_search(query, strategy, **kwargs)
            elif strategy.strategy_type == RetrievalStrategyType.HYBRID_SEARCH:
                task = self._execute_hybrid_search(query, strategy, **kwargs)
            elif strategy.strategy_type == RetrievalStrategyType.MEMORY_SEARCH:
                task = self._execute_memory_search(query, strategy, agent_role, **kwargs)
            elif strategy.strategy_type == RetrievalStrategyType.GRAPHITI_SEARCH:
                task = self._execute_graphiti_search(query, strategy, **kwargs)
            else:
                continue
                
            tasks.append((strategy, task))
        
        # Execute all strategies concurrently
        executed_strategies = []
        for strategy, task in tasks:
            try:
                strategy_start = time.time()
                results = await task
                strategy_time = time.time() - strategy_start
                
                strategy_results[strategy.strategy_type] = results
                executed_strategies.append({
                    "strategy": strategy.strategy_name,
                    "type": strategy.strategy_type.value,
                    "results_count": len(results),
                    "execution_time": strategy_time,
                    "performance_score": strategy.performance_score
                })
                
                # Update strategy performance (simplified precision calculation)
                precision = len(results) / max(10, len(results))  # Simple heuristic
                success = len(results) > 0
                self.bandit.update_performance(strategy, precision, strategy_time, success)
                
            except Exception as e:
                logger.error(f"Strategy {strategy.strategy_name} failed: {e}")
                executed_strategies.append({
                    "strategy": strategy.strategy_name,
                    "type": strategy.strategy_type.value,
                    "results_count": 0,
                    "execution_time": 0.0,
                    "error": str(e)
                })
        
        # Fuse and rank results from all strategies
        fused_results = await self._fuse_results(strategy_results)
        
        total_time = time.time() - start_time
        
        # Record query for analysis
        query_record = {
            "query": query,
            "agent_role": agent_role,
            "strategies_used": [s["strategy"] for s in executed_strategies],
            "total_results": len(fused_results),
            "total_time": total_time,
            "timestamp": time.time()
        }
        self.query_history.append(query_record)
        
        # Save updated configurations
        self._save_configuration()
        
        return {
            "results": fused_results,
            "strategies_used": executed_strategies,
            "total_time": total_time,
            "query_metadata": query_record
        }
    
    async def _fuse_results(self, strategy_results: Dict[RetrievalStrategyType, List[Dict]]) -> List[Dict]:
        """
        Fuse and rank results from multiple strategies
        
        Args:
            strategy_results: Results from each strategy
            
        Returns:
            Fused and ranked results
        """
        all_results = []
        seen_content = set()  # Simple deduplication
        
        # Combine results from all strategies
        for strategy_type, results in strategy_results.items():
            strategy = self.strategies[strategy_type]
            
            for result in results:
                content_hash = hash(result.get("content", "")[:100])  # Simple content hash
                
                if content_hash not in seen_content:
                    # Weight score by strategy performance
                    weighted_score = result.get("score", 0.0) * strategy.performance_score
                    
                    fused_result = {
                        **result,
                        "fused_score": weighted_score,
                        "original_score": result.get("score", 0.0),
                        "strategy_weight": strategy.performance_score
                    }
                    
                    all_results.append(fused_result)
                    seen_content.add(content_hash)
        
        # Sort by fused score
        all_results.sort(key=lambda x: x["fused_score"], reverse=True)
        
        # Limit to top results
        return all_results[:20]
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all strategies"""
        stats = {
            "total_rounds": self.bandit.total_rounds,
            "strategies": {}
        }
        
        for strategy_type, strategy in self.strategies.items():
            stats["strategies"][strategy_type.value] = {
                "name": strategy.strategy_name,
                "enabled": strategy.enabled,
                "performance_score": strategy.performance_score,
                "confidence_bound": strategy.confidence_bound,
                "total_queries": strategy.metrics.total_queries,
                "success_rate": strategy.metrics.success_rate,
                "avg_precision": strategy.metrics.avg_precision,
                "avg_response_time": strategy.metrics.avg_response_time
            }
        
        return stats
    
    def enable_strategy(self, strategy_type: RetrievalStrategyType, enabled: bool = True):
        """Enable or disable a retrieval strategy"""
        if strategy_type in self.strategies:
            self.strategies[strategy_type].enabled = enabled
            self._save_configuration()
            logger.info(f"Strategy {strategy_type.value} {'enabled' if enabled else 'disabled'}")
    
    async def select_strategy(self, query: str, strategies: List[str]) -> str:
        """
        Select optimal strategy using bandit algorithm for SCWT compatibility
        
        Args:
            query: Search query for context-aware selection
            strategies: List of strategy names to choose from
            
        Returns:
            Selected strategy name
        """
        try:
            # Map strategy names to strategy objects
            available_strategies = []
            
            for strategy_name in strategies:
                # Find matching strategy by name or type
                for strategy_type, strategy in self.strategies.items():
                    if (strategy.strategy_name.lower() == strategy_name.lower() or 
                        strategy_type.value == strategy_name.lower()):
                        if strategy.enabled:
                            available_strategies.append(strategy)
                        break
            
            if not available_strategies:
                # Fallback to any enabled strategy
                available_strategies = [s for s in self.strategies.values() if s.enabled]
            
            if not available_strategies:
                logger.warning("No strategies available for selection")
                return "vector_search"  # Default fallback
            
            # Use bandit algorithm to select optimal strategy
            selected_strategies = self.bandit.select_strategies(available_strategies, max_strategies=1)
            
            if selected_strategies:
                selected = selected_strategies[0]
                logger.debug(f"Selected strategy: {selected.strategy_name} (score: {selected.performance_score:.3f})")
                return selected.strategy_type.value
            else:
                # Fallback to first available strategy
                return available_strategies[0].strategy_type.value
                
        except Exception as e:
            logger.error(f"Strategy selection failed: {e}")
            return "vector_search"  # Safe fallback
    
    def fuse_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fuse results from multiple strategies with weighted scoring
        
        Args:
            results: List of result dictionaries from different strategies
            
        Returns:
            Fused and ranked results
        """
        try:
            if not results:
                return []
            
            # Group results by strategy for proper weighting
            strategy_groups = {}
            for result in results:
                strategy_name = result.get('strategy', 'unknown')
                if strategy_name not in strategy_groups:
                    strategy_groups[strategy_name] = []
                strategy_groups[strategy_name].append(result)
            
            fused_results = []
            seen_content = set()  # For deduplication
            
            # Process results from each strategy
            for strategy_name, strategy_results in strategy_groups.items():
                # Find strategy performance weight
                strategy_weight = 0.5  # Default weight
                
                for strategy in self.strategies.values():
                    if strategy.strategy_name == strategy_name:
                        strategy_weight = strategy.performance_score
                        break
                
                # Weight and deduplicate results
                for result in strategy_results:
                    content = result.get('content', '')
                    content_hash = hash(content[:100])  # Simple content hash for dedup
                    
                    if content_hash not in seen_content:
                        original_score = result.get('score', 0.0)
                        fused_score = original_score * strategy_weight
                        
                        fused_result = {
                            **result,
                            'fused_score': fused_score,
                            'original_score': original_score,
                            'strategy_weight': strategy_weight,
                            'fusion_rank': len(fused_results)
                        }
                        
                        fused_results.append(fused_result)
                        seen_content.add(content_hash)
            
            # Sort by fused score (higher is better)
            fused_results.sort(key=lambda x: x.get('fused_score', 0.0), reverse=True)
            
            # Add final ranking
            for i, result in enumerate(fused_results):
                result['final_rank'] = i + 1
            
            logger.debug(f"Fused {len(results)} results into {len(fused_results)} unique results")
            return fused_results
            
        except Exception as e:
            logger.error(f"Result fusion failed: {e}")
            # Return original results as fallback
            return results[:20] if results else []

# Factory function
def create_adaptive_retriever(memory_service: Optional[MemoryService] = None,
                            rag_service=None) -> AdaptiveRetriever:
    """Create a configured adaptive retriever instance"""
    return AdaptiveRetriever(memory_service, rag_service)