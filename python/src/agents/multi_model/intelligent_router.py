"""
Intelligent Model Routing System with Advanced Cost Optimization
Combines ML-powered routing with real-time cost optimization across all providers.
"""

import asyncio
import time
import logging
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from .model_ensemble import ModelEnsemble, TaskType, TaskRequest, ModelResponse, ModelProvider
from .predictive_scaler import PredictiveAgentScaler
from .benchmark_system import BenchmarkSuite
from ..monitoring.metrics import track_agent_execution, agent_cost_dollars

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Different routing strategies."""
    PERFORMANCE_FIRST = "performance_first"
    COST_FIRST = "cost_first"
    BALANCED = "balanced"
    SPEED_FIRST = "speed_first"
    QUALITY_FIRST = "quality_first"
    ADAPTIVE = "adaptive"


@dataclass
class RoutingDecision:
    """Detailed routing decision with reasoning."""
    selected_model: str
    strategy_used: RoutingStrategy
    confidence_score: float
    estimated_cost: float
    estimated_time: float
    estimated_quality: float
    fallback_models: List[str]
    reasoning: List[str]
    factors_considered: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CostOptimizationRule:
    """Cost optimization rule."""
    name: str
    condition: str  # Python expression
    action: str     # routing action
    priority: int
    active: bool = True
    savings_target: float = 0.0  # Target cost savings


@dataclass
class RoutingMetrics:
    """Metrics for routing decisions."""
    total_requests: int = 0
    successful_routes: int = 0
    avg_cost_per_request: float = 0.0
    avg_response_time: float = 0.0
    avg_quality_score: float = 0.0
    cost_savings_achieved: float = 0.0
    strategy_distribution: Dict[str, int] = field(default_factory=dict)
    model_usage_stats: Dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


class IntelligentModelRouter:
    """
    Advanced intelligent routing system with ML-powered decision making
    and comprehensive cost optimization.
    """
    
    def __init__(
        self,
        model_ensemble: ModelEnsemble,
        predictive_scaler: Optional[PredictiveAgentScaler] = None,
        benchmark_suite: Optional[BenchmarkSuite] = None,
        redis_client=None
    ):
        self.ensemble = model_ensemble
        self.scaler = predictive_scaler
        self.benchmark = benchmark_suite
        self.redis_client = redis_client
        
        # ML models for routing decisions
        self.routing_model: Optional[RandomForestClassifier] = None
        self.feature_scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        
        # Historical routing data for ML training
        self.routing_history: List[Dict[str, Any]] = []
        
        # Cost optimization rules
        self.cost_optimization_rules: Dict[str, CostOptimizationRule] = {}
        self._initialize_cost_optimization_rules()
        
        # Routing strategies configuration
        self.strategy_configs = {
            RoutingStrategy.PERFORMANCE_FIRST: {
                "quality_weight": 0.5,
                "speed_weight": 0.3,
                "cost_weight": 0.1,
                "reliability_weight": 0.1
            },
            RoutingStrategy.COST_FIRST: {
                "quality_weight": 0.2,
                "speed_weight": 0.2,
                "cost_weight": 0.5,
                "reliability_weight": 0.1
            },
            RoutingStrategy.BALANCED: {
                "quality_weight": 0.3,
                "speed_weight": 0.25,
                "cost_weight": 0.25,
                "reliability_weight": 0.2
            },
            RoutingStrategy.SPEED_FIRST: {
                "quality_weight": 0.2,
                "speed_weight": 0.6,
                "cost_weight": 0.1,
                "reliability_weight": 0.1
            },
            RoutingStrategy.QUALITY_FIRST: {
                "quality_weight": 0.6,
                "speed_weight": 0.1,
                "cost_weight": 0.1,
                "reliability_weight": 0.2
            }
        }
        
        # Dynamic pricing and budget management
        self.budget_manager = BudgetManager()
        self.dynamic_pricing: Dict[str, float] = {}
        
        # Real-time routing metrics
        self.routing_metrics = RoutingMetrics()
        
        # A/B testing for routing strategies
        self.ab_test_config = {
            "enabled": True,
            "test_percentage": 0.1,  # 10% of traffic for testing
            "current_test": None,
            "baseline_strategy": RoutingStrategy.BALANCED
        }
        
        # Circuit breaker for failed models
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Load historical data and models
        asyncio.create_task(self._load_routing_model())
    
    def _initialize_cost_optimization_rules(self):
        """Initialize cost optimization rules."""
        
        # Rule 1: Use cheaper models for simple queries
        self.cost_optimization_rules["simple_query_cost_optimization"] = CostOptimizationRule(
            name="Simple Query Cost Optimization",
            condition="task_type == 'simple_query' and urgency != 'critical'",
            action="prefer_models(['anthropic:claude-3-haiku-20240307', 'openai:gpt-3.5-turbo'])",
            priority=1,
            savings_target=0.7  # 70% cost reduction target
        )
        
        # Rule 2: Budget-based routing
        self.cost_optimization_rules["budget_constraint"] = CostOptimizationRule(
            name="Budget Constraint Routing",
            condition="estimated_cost > remaining_budget * 0.1",
            action="route_to_cheapest_viable_model()",
            priority=2,
            savings_target=0.5
        )
        
        # Rule 3: Time-based cost optimization
        self.cost_optimization_rules["off_peak_optimization"] = CostOptimizationRule(
            name="Off-Peak Cost Optimization",
            condition="is_off_peak_hours() and urgency == 'low'",
            action="prefer_cost_efficient_models()",
            priority=3,
            savings_target=0.3
        )
        
        # Rule 4: Bulk processing optimization
        self.cost_optimization_rules["bulk_processing"] = CostOptimizationRule(
            name="Bulk Processing Optimization",
            condition="batch_size > 10",
            action="use_bulk_pricing_models()",
            priority=4,
            savings_target=0.4
        )
    
    async def route_request(
        self,
        request: TaskRequest,
        strategy: Optional[RoutingStrategy] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """
        Intelligently route request to optimal model.
        """
        start_time = time.time()
        
        # Determine strategy
        if strategy is None:
            strategy = await self._determine_optimal_strategy(request, context)
        
        # Apply A/B testing if enabled
        if self.ab_test_config["enabled"] and self._should_ab_test():
            strategy = await self._apply_ab_testing(strategy, request)
        
        # Get candidate models
        candidates = await self._get_candidate_models(request, strategy, context)
        
        # Apply cost optimization rules
        candidates = await self._apply_cost_optimization(candidates, request, context)
        
        # Score candidates using ML model or heuristics
        if self.routing_model:
            scored_candidates = await self._score_candidates_ml(candidates, request, context)
        else:
            scored_candidates = await self._score_candidates_heuristic(candidates, request, strategy)
        
        # Select best model
        decision = await self._make_final_decision(scored_candidates, request, strategy, context)
        
        # Track routing decision
        await self._track_routing_decision(decision, request, time.time() - start_time)
        
        return decision
    
    async def _determine_optimal_strategy(
        self,
        request: TaskRequest,
        context: Optional[Dict[str, Any]]
    ) -> RoutingStrategy:
        """Determine optimal routing strategy based on context."""
        
        # Check for explicit strategy in context
        if context and "preferred_strategy" in context:
            return RoutingStrategy(context["preferred_strategy"])
        
        # Use adaptive strategy based on current conditions
        current_hour = datetime.now().hour
        
        # Business hours: prefer balanced approach
        if 9 <= current_hour <= 17:
            base_strategy = RoutingStrategy.BALANCED
        # Off hours: prefer cost optimization
        else:
            base_strategy = RoutingStrategy.COST_FIRST
        
        # Adjust based on urgency
        if request.urgency == "critical":
            return RoutingStrategy.PERFORMANCE_FIRST
        elif request.urgency == "high":
            return RoutingStrategy.SPEED_FIRST
        elif request.urgency == "low":
            return RoutingStrategy.COST_FIRST
        
        # Check budget constraints
        remaining_budget = await self.budget_manager.get_remaining_budget()
        if remaining_budget < 0.2:  # Less than 20% budget remaining
            return RoutingStrategy.COST_FIRST
        
        return base_strategy
    
    async def _get_candidate_models(
        self,
        request: TaskRequest,
        strategy: RoutingStrategy,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Get candidate models based on task type and strategy."""
        
        # Start with task type preferences
        task_candidates = self.ensemble.routing_preferences.get(
            request.task_type, 
            list(self.ensemble.model_configs.keys())
        )
        
        # Filter out unavailable models
        available_candidates = []
        for model_id in task_candidates:
            config = self.ensemble.model_configs.get(model_id)
            if config and config.availability:
                # Check circuit breaker
                if not self._is_circuit_open(model_id):
                    available_candidates.append(model_id)
        
        # If strategy is cost-first, sort by cost
        if strategy == RoutingStrategy.COST_FIRST:
            available_candidates = sorted(
                available_candidates,
                key=lambda m: self.ensemble.model_configs[m].cost_per_1k_tokens
            )
        
        # If strategy is speed-first, prefer faster models
        elif strategy == RoutingStrategy.SPEED_FIRST:
            # Get models known for speed (lower cost often correlates with speed)
            speed_models = [
                "anthropic:claude-3-haiku-20240307",
                "openai:gpt-3.5-turbo"
            ]
            # Prioritize speed models if available
            prioritized = [m for m in available_candidates if m in speed_models]
            others = [m for m in available_candidates if m not in speed_models]
            available_candidates = prioritized + others
        
        # Limit candidates for performance
        max_candidates = 5
        return available_candidates[:max_candidates]
    
    async def _apply_cost_optimization(
        self,
        candidates: List[str],
        request: TaskRequest,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Apply cost optimization rules to filter candidates."""
        
        optimized_candidates = candidates.copy()
        
        for rule_name, rule in self.cost_optimization_rules.items():
            if not rule.active:
                continue
            
            try:
                # Evaluate rule condition
                evaluation_context = {
                    "task_type": request.task_type.value,
                    "urgency": request.urgency,
                    "estimated_cost": 0,  # Will be calculated per model
                    "remaining_budget": await self.budget_manager.get_remaining_budget(),
                    "is_off_peak_hours": lambda: datetime.now().hour < 8 or datetime.now().hour > 20,
                    "batch_size": context.get("batch_size", 1) if context else 1
                }
                
                if eval(rule.condition, {"__builtins__": {}}, evaluation_context):
                    logger.info(f"Applying cost optimization rule: {rule.name}")
                    
                    # Apply rule action
                    if "prefer_models" in rule.action:
                        # Extract preferred models from action
                        import re
                        models_match = re.search(r"prefer_models\(\[(.*?)\]\)", rule.action)
                        if models_match:
                            preferred_models = [
                                m.strip().strip("'\"") 
                                for m in models_match.group(1).split(",")
                            ]
                            # Filter to only preferred models that are in candidates
                            filtered = [m for m in preferred_models if m in optimized_candidates]
                            if filtered:
                                optimized_candidates = filtered
                    
                    elif "route_to_cheapest_viable_model" in rule.action:
                        # Sort by cost and take cheapest viable options
                        optimized_candidates = sorted(
                            optimized_candidates,
                            key=lambda m: self.ensemble.model_configs[m].cost_per_1k_tokens
                        )[:2]  # Top 2 cheapest
                    
                    elif "prefer_cost_efficient_models" in rule.action:
                        # Prefer models with good cost efficiency
                        cost_efficient = []
                        for model_id in optimized_candidates:
                            config = self.ensemble.model_configs[model_id]
                            if config.cost_per_1k_tokens < 5.0:  # Under $5 per 1k tokens
                                cost_efficient.append(model_id)
                        if cost_efficient:
                            optimized_candidates = cost_efficient
                            
            except Exception as e:
                logger.warning(f"Cost optimization rule {rule_name} failed: {e}")
        
        return optimized_candidates
    
    async def _score_candidates_ml(
        self,
        candidates: List[str],
        request: TaskRequest,
        context: Optional[Dict[str, Any]]
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Score candidates using ML model."""
        if not self.routing_model or not self.feature_scaler:
            return await self._score_candidates_heuristic(
                candidates, request, RoutingStrategy.BALANCED
            )
        
        scored_candidates = []
        
        for model_id in candidates:
            try:
                # Extract features
                features = await self._extract_routing_features(model_id, request, context)
                
                if features:
                    # Scale features
                    features_scaled = self.feature_scaler.transform([features])
                    
                    # Predict probability of success
                    probability = self.routing_model.predict_proba(features_scaled)[0]
                    success_prob = probability[1] if len(probability) > 1 else probability[0]
                    
                    # Calculate detailed scores
                    detailed_scores = await self._calculate_detailed_scores(model_id, request)
                    
                    scored_candidates.append((model_id, success_prob, detailed_scores))
                
            except Exception as e:
                logger.warning(f"ML scoring failed for {model_id}: {e}")
                # Fallback to heuristic scoring
                heuristic_scores = await self._score_candidates_heuristic(
                    [model_id], request, RoutingStrategy.BALANCED
                )
                if heuristic_scores:
                    scored_candidates.append(heuristic_scores[0])
        
        return sorted(scored_candidates, key=lambda x: x[1], reverse=True)
    
    async def _score_candidates_heuristic(
        self,
        candidates: List[str],
        request: TaskRequest,
        strategy: RoutingStrategy
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Score candidates using heuristic approach."""
        scored_candidates = []
        strategy_weights = self.strategy_configs[strategy]
        
        for model_id in candidates:
            try:
                config = self.ensemble.model_configs[model_id]
                
                # Get performance metrics
                metrics = await self.ensemble._get_performance_metrics(model_id)
                
                # Calculate component scores
                quality_score = metrics.avg_quality_score if metrics else config.performance_score
                speed_score = self._calculate_speed_score(config, metrics)
                cost_score = self._calculate_cost_score(config, request)
                reliability_score = metrics.success_rate if metrics else 0.9
                
                # Calculate weighted score
                overall_score = (
                    quality_score * strategy_weights["quality_weight"] +
                    speed_score * strategy_weights["speed_weight"] +
                    cost_score * strategy_weights["cost_weight"] +
                    reliability_score * strategy_weights["reliability_weight"]
                )
                
                detailed_scores = {
                    "quality_score": quality_score,
                    "speed_score": speed_score,
                    "cost_score": cost_score,
                    "reliability_score": reliability_score,
                    "estimated_cost": self.ensemble._estimate_cost(model_id, request),
                    "estimated_time": self._estimate_response_time(config, metrics)
                }
                
                scored_candidates.append((model_id, overall_score, detailed_scores))
                
            except Exception as e:
                logger.warning(f"Heuristic scoring failed for {model_id}: {e}")
        
        return sorted(scored_candidates, key=lambda x: x[1], reverse=True)
    
    def _calculate_speed_score(self, config, metrics) -> float:
        """Calculate speed score for model."""
        if metrics and metrics.avg_response_time > 0:
            # Normalize response time (lower is better)
            baseline_time = 2.0  # 2 seconds baseline
            if metrics.avg_response_time <= baseline_time / 2:
                return 1.0
            elif metrics.avg_response_time <= baseline_time:
                return 0.8
            elif metrics.avg_response_time <= baseline_time * 2:
                return 0.6
            else:
                return 0.3
        else:
            # Estimate based on model cost (often correlates with speed)
            if config.cost_per_1k_tokens < 1.0:
                return 0.9  # Cheap models are often fast
            elif config.cost_per_1k_tokens < 5.0:
                return 0.7
            else:
                return 0.5
    
    def _calculate_cost_score(self, config, request) -> float:
        """Calculate cost score for model."""
        estimated_cost = self.ensemble._estimate_cost(config.model_name, request)
        
        # Normalize cost (lower is better)
        if estimated_cost < 0.01:
            return 1.0
        elif estimated_cost < 0.05:
            return 0.8
        elif estimated_cost < 0.1:
            return 0.6
        elif estimated_cost < 0.5:
            return 0.4
        else:
            return 0.2
    
    def _estimate_response_time(self, config, metrics) -> float:
        """Estimate response time for model."""
        if metrics and metrics.avg_response_time > 0:
            return metrics.avg_response_time
        else:
            # Rough estimate based on cost
            if config.cost_per_1k_tokens < 1.0:
                return 1.0  # Fast, cheap models
            elif config.cost_per_1k_tokens < 5.0:
                return 2.0  # Medium speed/cost
            else:
                return 4.0  # Slower, expensive models
    
    async def _make_final_decision(
        self,
        scored_candidates: List[Tuple[str, float, Dict[str, Any]]],
        request: TaskRequest,
        strategy: RoutingStrategy,
        context: Optional[Dict[str, Any]]
    ) -> RoutingDecision:
        """Make final routing decision."""
        
        if not scored_candidates:
            # Emergency fallback
            fallback_model = "anthropic:claude-3-haiku-20240307"
            return RoutingDecision(
                selected_model=fallback_model,
                strategy_used=strategy,
                confidence_score=0.1,
                estimated_cost=0.01,
                estimated_time=2.0,
                estimated_quality=0.5,
                fallback_models=[],
                reasoning=["Emergency fallback - no candidates available"],
                factors_considered={}
            )
        
        # Select top candidate
        selected_model, score, detailed_scores = scored_candidates[0]
        
        # Prepare fallback models (top 3 alternatives)
        fallback_models = [candidate[0] for candidate in scored_candidates[1:4]]
        
        # Build reasoning
        reasoning = []
        reasoning.append(f"Selected based on {strategy.value} strategy")
        reasoning.append(f"Overall score: {score:.3f}")
        
        if detailed_scores.get("quality_score", 0) > 0.8:
            reasoning.append("High quality expected")
        if detailed_scores.get("cost_score", 0) > 0.8:
            reasoning.append("Cost efficient")
        if detailed_scores.get("speed_score", 0) > 0.8:
            reasoning.append("Fast response expected")
        
        return RoutingDecision(
            selected_model=selected_model,
            strategy_used=strategy,
            confidence_score=score,
            estimated_cost=detailed_scores.get("estimated_cost", 0),
            estimated_time=detailed_scores.get("estimated_time", 0),
            estimated_quality=detailed_scores.get("quality_score", 0),
            fallback_models=fallback_models,
            reasoning=reasoning,
            factors_considered=detailed_scores
        )
    
    async def _extract_routing_features(
        self,
        model_id: str,
        request: TaskRequest,
        context: Optional[Dict[str, Any]]
    ) -> Optional[List[float]]:
        """Extract features for ML routing model."""
        try:
            config = self.ensemble.model_configs[model_id]
            metrics = await self.ensemble._get_performance_metrics(model_id)
            
            features = []
            
            # Model features
            features.extend([
                hash(config.provider.value) % 1000 / 1000.0,  # Provider hash
                config.cost_per_1k_tokens / 30.0,  # Normalized cost
                config.max_tokens / 10000.0,  # Normalized max tokens
                config.performance_score,
                len(config.strengths),  # Number of task type strengths
            ])
            
            # Task features
            features.extend([
                hash(request.task_type.value) % 1000 / 1000.0,  # Task type hash
                len(request.prompt) / 1000.0,  # Normalized prompt length
                request.temperature,
                request.max_tokens / 2000.0 if request.max_tokens else 0.5,
                {"low": 0.2, "normal": 0.5, "high": 0.8, "critical": 1.0}[request.urgency],
            ])
            
            # Historical metrics
            if metrics:
                features.extend([
                    metrics.success_rate,
                    metrics.avg_response_time / 10.0,  # Normalized
                    metrics.avg_quality_score,
                    metrics.cost_efficiency / 5.0,  # Normalized
                    min(metrics.total_requests / 100.0, 1.0),  # Experience factor
                ])
            else:
                features.extend([0.9, 0.2, 0.8, 0.5, 0.1])  # Default values
            
            # Time-based features
            now = datetime.now()
            features.extend([
                now.hour / 24.0,  # Hour of day
                now.weekday() / 7.0,  # Day of week
                int(9 <= now.hour <= 17 and now.weekday() < 5),  # Business hours
            ])
            
            # Context features
            if context:
                features.extend([
                    context.get("batch_size", 1) / 10.0,  # Normalized batch size
                    context.get("priority_score", 0.5),
                    int(context.get("requires_accuracy", False)),
                ])
            else:
                features.extend([0.1, 0.5, 0])
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    async def _calculate_detailed_scores(
        self,
        model_id: str,
        request: TaskRequest
    ) -> Dict[str, Any]:
        """Calculate detailed scores for model."""
        config = self.ensemble.model_configs[model_id]
        metrics = await self.ensemble._get_performance_metrics(model_id)
        
        return {
            "quality_score": metrics.avg_quality_score if metrics else config.performance_score,
            "speed_score": self._calculate_speed_score(config, metrics),
            "cost_score": self._calculate_cost_score(config, request),
            "reliability_score": metrics.success_rate if metrics else 0.9,
            "estimated_cost": self.ensemble._estimate_cost(model_id, request),
            "estimated_time": self._estimate_response_time(config, metrics)
        }
    
    def _is_circuit_open(self, model_id: str) -> bool:
        """Check if circuit breaker is open for model."""
        return self.ensemble._is_circuit_open(model_id)
    
    def _should_ab_test(self) -> bool:
        """Determine if this request should be part of A/B test."""
        return (
            self.ab_test_config["enabled"] and 
            np.random.random() < self.ab_test_config["test_percentage"]
        )
    
    async def _apply_ab_testing(
        self,
        original_strategy: RoutingStrategy,
        request: TaskRequest
    ) -> RoutingStrategy:
        """Apply A/B testing to routing strategy."""
        if self.ab_test_config["current_test"]:
            test_strategy = RoutingStrategy(self.ab_test_config["current_test"])
            logger.debug(f"A/B test: using {test_strategy.value} instead of {original_strategy.value}")
            return test_strategy
        return original_strategy
    
    async def _track_routing_decision(
        self,
        decision: RoutingDecision,
        request: TaskRequest,
        decision_time: float
    ):
        """Track routing decision for analysis and ML training."""
        
        # Update routing metrics
        self.routing_metrics.total_requests += 1
        self.routing_metrics.avg_cost_per_request = (
            (self.routing_metrics.avg_cost_per_request * (self.routing_metrics.total_requests - 1) +
             decision.estimated_cost) / self.routing_metrics.total_requests
        )
        
        # Update strategy distribution
        strategy_name = decision.strategy_used.value
        if strategy_name not in self.routing_metrics.strategy_distribution:
            self.routing_metrics.strategy_distribution[strategy_name] = 0
        self.routing_metrics.strategy_distribution[strategy_name] += 1
        
        # Update model usage stats
        if decision.selected_model not in self.routing_metrics.model_usage_stats:
            self.routing_metrics.model_usage_stats[decision.selected_model] = 0
        self.routing_metrics.model_usage_stats[decision.selected_model] += 1
        
        self.routing_metrics.last_updated = datetime.now()
        
        # Store detailed routing history for ML training
        routing_record = {
            "timestamp": decision.timestamp.isoformat(),
            "selected_model": decision.selected_model,
            "strategy": decision.strategy_used.value,
            "task_type": request.task_type.value,
            "urgency": request.urgency,
            "estimated_cost": decision.estimated_cost,
            "estimated_time": decision.estimated_time,
            "estimated_quality": decision.estimated_quality,
            "confidence_score": decision.confidence_score,
            "decision_time": decision_time,
            "factors": decision.factors_considered
        }
        
        self.routing_history.append(routing_record)
        
        # Keep history bounded
        if len(self.routing_history) > 10000:
            self.routing_history = self.routing_history[-5000:]  # Keep last 5k records
        
        # Store in Redis for persistence
        if self.redis_client:
            try:
                self.redis_client.lpush("routing_history", json.dumps(routing_record))
                self.redis_client.ltrim("routing_history", 0, 10000)  # Keep last 10k
                
                # Store current metrics
                metrics_data = {
                    "total_requests": self.routing_metrics.total_requests,
                    "avg_cost_per_request": self.routing_metrics.avg_cost_per_request,
                    "strategy_distribution": self.routing_metrics.strategy_distribution,
                    "model_usage_stats": self.routing_metrics.model_usage_stats,
                    "last_updated": self.routing_metrics.last_updated.isoformat()
                }
                self.redis_client.setex("routing_metrics", 3600, json.dumps(metrics_data))
                
            except Exception as e:
                logger.warning(f"Failed to store routing data: {e}")
    
    async def _load_routing_model(self):
        """Load pre-trained routing model if available."""
        try:
            # This would load from persistent storage
            # For now, we'll train a new model if we have enough data
            if len(self.routing_history) > 100:
                await self._train_routing_model()
        except Exception as e:
            logger.warning(f"Failed to load routing model: {e}")
    
    async def _train_routing_model(self):
        """Train ML model for routing decisions."""
        try:
            if len(self.routing_history) < 50:
                logger.info("Not enough data to train routing model")
                return
            
            logger.info("Training routing model...")
            
            # Prepare training data
            X, y = await self._prepare_training_data()
            
            if len(X) < 20:
                return
            
            # Train model
            self.routing_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            self.feature_scaler = StandardScaler()
            self.label_encoder = LabelEncoder()
            
            # Scale features and encode labels
            X_scaled = self.feature_scaler.fit_transform(X)
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Train model
            self.routing_model.fit(X_scaled, y_encoded)
            
            # Calculate training accuracy
            train_score = self.routing_model.score(X_scaled, y_encoded)
            logger.info(f"Routing model trained with accuracy: {train_score:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to train routing model: {e}")
    
    async def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from routing history."""
        X = []
        y = []
        
        for record in self.routing_history[-1000:]:  # Use last 1000 records
            try:
                # Extract features (similar to real-time feature extraction)
                features = await self._extract_features_from_record(record)
                if features:
                    X.append(features)
                    y.append(record["selected_model"])
            except Exception as e:
                logger.warning(f"Failed to extract features from record: {e}")
        
        return np.array(X), np.array(y)
    
    async def _extract_features_from_record(self, record: Dict[str, Any]) -> Optional[List[float]]:
        """Extract features from historical routing record."""
        try:
            features = []
            
            # Basic features from record
            features.extend([
                {"low": 0.2, "normal": 0.5, "high": 0.8, "critical": 1.0}.get(record.get("urgency", "normal"), 0.5),
                record.get("estimated_cost", 0),
                record.get("estimated_time", 0),
                record.get("confidence_score", 0),
            ])
            
            # Time-based features
            timestamp = datetime.fromisoformat(record["timestamp"])
            features.extend([
                timestamp.hour / 24.0,
                timestamp.weekday() / 7.0,
                int(9 <= timestamp.hour <= 17 and timestamp.weekday() < 5),
            ])
            
            # Strategy encoding
            strategies = ["performance_first", "cost_first", "balanced", "speed_first", "quality_first"]
            strategy_encoded = strategies.index(record.get("strategy", "balanced")) / len(strategies)
            features.append(strategy_encoded)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction from record failed: {e}")
            return None
    
    async def get_routing_status(self) -> Dict[str, Any]:
        """Get current routing system status."""
        return {
            "routing_metrics": {
                "total_requests": self.routing_metrics.total_requests,
                "avg_cost_per_request": self.routing_metrics.avg_cost_per_request,
                "strategy_distribution": self.routing_metrics.strategy_distribution,
                "model_usage_stats": self.routing_metrics.model_usage_stats,
            },
            "ml_model_status": {
                "trained": self.routing_model is not None,
                "training_data_size": len(self.routing_history),
                "feature_scaler_fitted": self.feature_scaler is not None,
            },
            "cost_optimization": {
                "active_rules": len([r for r in self.cost_optimization_rules.values() if r.active]),
                "total_rules": len(self.cost_optimization_rules),
            },
            "circuit_breakers": {
                "active_breakers": len([cb for cb in self.circuit_breakers.values() if cb.get("state") == "open"]),
                "total_monitored": len(self.circuit_breakers),
            },
            "ab_testing": self.ab_test_config,
            "budget_status": await self.budget_manager.get_budget_status()
        }


class BudgetManager:
    """Manages budget tracking and cost constraints."""
    
    def __init__(self, initial_budget: float = 1000.0):
        self.daily_budget = initial_budget
        self.spent_today = 0.0
        self.last_reset = datetime.now().date()
        self.spending_history: List[Dict[str, Any]] = []
        
    async def get_remaining_budget(self) -> float:
        """Get remaining budget for today."""
        await self._check_daily_reset()
        return max(0, self.daily_budget - self.spent_today)
    
    async def record_spending(self, amount: float, model_id: str, task_type: str):
        """Record spending against budget."""
        await self._check_daily_reset()
        
        self.spent_today += amount
        self.spending_history.append({
            "timestamp": datetime.now().isoformat(),
            "amount": amount,
            "model_id": model_id,
            "task_type": task_type,
            "remaining_budget": await self.get_remaining_budget()
        })
        
        # Keep history bounded
        if len(self.spending_history) > 1000:
            self.spending_history = self.spending_history[-500:]
    
    async def _check_daily_reset(self):
        """Check if we need to reset daily budget."""
        today = datetime.now().date()
        if today > self.last_reset:
            self.spent_today = 0.0
            self.last_reset = today
    
    async def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        await self._check_daily_reset()
        
        return {
            "daily_budget": self.daily_budget,
            "spent_today": self.spent_today,
            "remaining_today": await self.get_remaining_budget(),
            "utilization_percentage": (self.spent_today / self.daily_budget) * 100,
            "last_reset": self.last_reset.isoformat(),
            "spending_entries_today": len([
                s for s in self.spending_history
                if datetime.fromisoformat(s["timestamp"]).date() == self.last_reset
            ])
        }