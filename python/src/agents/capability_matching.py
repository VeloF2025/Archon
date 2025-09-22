"""
Capability Matching - Intelligent agent-task matching.

This module provides intelligent matching algorithms for pairing tasks
with the most suitable agents based on capabilities, expertise, and context.

Key Features:
- Multi-factor capability matching
- Context-aware agent selection
- Performance-based weighting
- Load balancing considerations
- Confidence scoring for matches
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
from collections import defaultdict

from .enhanced_agent_capabilities import (
    EnhancedAgentCapabilitySystem, AgentCapabilityProfile, CapabilityCategory
)
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class MatchingStrategy(Enum):
    """Strategies for agent-task matching."""
    EXACT_MATCH = "exact_match"           # Perfect capability match required
    BEST_FIT = "best_fit"               # Best overall match
    BALANCED = "balanced"               # Balance expertise and availability
    SPECIALIST = "specialist"           # Prefer specialists for critical tasks
    GENERALIST = "generalist"           # Prefer generalists for broad tasks
    LOAD_BALANCED = "load_balanced"     # Consider agent workload


class MatchingFactor(Enum):
    """Factors considered in matching."""
    CAPABILITY_MATCH = "capability_match"
    EXPERTISE_LEVEL = "expertise_level"
    PERFORMANCE_HISTORY = "performance_history"
    AVAILABILITY = "availability"
    RECENT_ACTIVITY = "recent_activity"
    LOAD_BALANCE = "load_balance"
    CONFIDENCE_CALIBRATION = "confidence_calibration"


@dataclass
class TaskRequirements:
    """Requirements for a task."""
    task_id: str
    description: str
    required_capabilities: Set[str]
    capability_weights: Dict[str, float]  # capability -> importance weight
    complexity: float  # 0-1
    urgency: float     # 0-1
    estimated_duration: float  # Minutes
    required_confidence: float = 0.7
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Current state of an agent."""
    agent_name: str
    current_load: float  # 0-1, where 1 is fully loaded
    recent_performance: float  # 0-1, recent success rate
    last_active: datetime
    current_tasks: int
    max_concurrent_tasks: int
    availability_schedule: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchScore:
    """Score for a potential agent-task match."""
    agent_name: str
    task_id: str
    overall_score: float
    factor_scores: Dict[MatchingFactor, float]
    confidence_interval: Tuple[float, float]
    reasoning: List[str]
    match_strength: str  # "weak", "moderate", "strong", "excellent"


@dataclass
class MatchingResult:
    """Result of agent-task matching."""
    task_id: str
    best_matches: List[MatchScore]
    all_scores: List[MatchScore]
    matching_strategy: MatchingStrategy
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class CapabilityMatcher:
    """
    Advanced capability matching system.

    This system provides:
    - Multi-factor matching algorithms
    - Context-aware agent selection
    - Performance-based scoring
    - Load balancing
    - Confidence estimation
    """

    def __init__(self, capability_system: EnhancedAgentCapabilitySystem):
        """Initialize the capability matcher."""
        self.capability_system = capability_system
        self.agent_states: Dict[str, AgentState] = {}
        self.matching_history: List[MatchingResult] = []
        self.factor_weights: Dict[MatchingFactor, float] = {
            MatchingFactor.CAPABILITY_MATCH: 0.3,
            MatchingFactor.EXPERTISE_LEVEL: 0.25,
            MatchingFactor.PERFORMANCE_HISTORY: 0.2,
            MatchingFactor.AVAILABILITY: 0.15,
            MatchingFactor.LOAD_BALANCE: 0.1
        }
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def update_agent_state(
        self,
        agent_name: str,
        current_load: float = 0.0,
        recent_performance: float = 0.8,
        current_tasks: int = 0,
        max_concurrent_tasks: int = 5
    ) -> None:
        """Update the state of an agent."""
        self.agent_states[agent_name] = AgentState(
            agent_name=agent_name,
            current_load=current_load,
            recent_performance=recent_performance,
            last_active=datetime.utcnow(),
            current_tasks=current_tasks,
            max_concurrent_tasks=max_concurrent_tasks
        )

    def find_best_matches(
        self,
        task: TaskRequirements,
        available_agents: List[str],
        strategy: MatchingStrategy = MatchingStrategy.BALANCED,
        top_k: int = 3
    ) -> MatchingResult:
        """
        Find the best agent matches for a task.

        Args:
            task: Task requirements
            available_agents: List of available agent names
            strategy: Matching strategy to use
            top_k: Number of top matches to return

        Returns:
            MatchingResult with ranked matches
        """
        start_time = datetime.utcnow()

        try:
            self.logger.info(f"Finding matches for task {task.task_id} using {strategy.value}")

            # Calculate scores for all agents
            all_scores = []
            for agent_name in available_agents:
                match_score = self._calculate_match_score(agent_name, task, strategy)
                if match_score.overall_score > 0.1:  # Minimum threshold
                    all_scores.append(match_score)

            # Sort by overall score
            all_scores.sort(key=lambda x: x.overall_score, reverse=True)

            # Get top matches
            best_matches = all_scores[:top_k]

            # Create result
            result = MatchingResult(
                task_id=task.task_id,
                best_matches=best_matches,
                all_scores=all_scores,
                matching_strategy=strategy,
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
                metadata={
                    "total_agents_considered": len(available_agents),
                    "viable_matches": len(all_scores),
                    "strategy_weights": self._get_strategy_weights(strategy)
                }
            )

            # Record in history
            self.matching_history.append(result)

            self.logger.info(f"Found {len(best_matches)} matches for task {task.task_id}")
            return result

        except Exception as e:
            self.logger.error(f"Error finding matches for task {task.task_id}: {e}")
            return MatchingResult(
                task_id=task.task_id,
                best_matches=[],
                all_scores=[],
                matching_strategy=strategy,
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
                metadata={"error": str(e)}
            )

    def _calculate_match_score(self, agent_name: str, task: TaskRequirements, strategy: MatchingStrategy) -> MatchScore:
        """Calculate match score for a specific agent."""
        factor_scores = {}
        reasoning = []

        # Get agent profile and state
        agent_profile = self.capability_system.agent_profiles.get(agent_name)
        agent_state = self.agent_states.get(agent_name)

        if not agent_profile:
            return MatchScore(
                agent_name=agent_name,
                task_id=task.task_id,
                overall_score=0.0,
                factor_scores={},
                confidence_interval=(0.0, 0.0),
                reasoning=["Agent profile not found"],
                match_strength="weak"
            )

        # Calculate factor scores
        factor_scores[MatchingFactor.CAPABILITY_MATCH] = self._calculate_capability_match(agent_profile, task, reasoning)
        factor_scores[MatchingFactor.EXPERTISE_LEVEL] = self._calculate_expertise_level(agent_profile, task, reasoning)
        factor_scores[MatchingFactor.PERFORMANCE_HISTORY] = self._calculate_performance_history(agent_profile, reasoning)
        factor_scores[MatchingFactor.AVAILABILITY] = self._calculate_availability(agent_state, task, reasoning)
        factor_scores[MatchingFactor.LOAD_BALANCE] = self._calculate_load_balance(agent_state, reasoning)

        # Apply strategy-specific weights
        strategy_weights = self._get_strategy_weights(strategy)

        # Calculate weighted overall score
        overall_score = sum(
            factor_scores[factor] * weight
            for factor, weight in strategy_weights.items()
            if factor in factor_scores
        )

        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(factor_scores, strategy_weights)

        # Determine match strength
        match_strength = self._determine_match_strength(overall_score)

        return MatchScore(
            agent_name=agent_name,
            task_id=task.task_id,
            overall_score=overall_score,
            factor_scores=factor_scores,
            confidence_interval=confidence_interval,
            reasoning=reasoning,
            match_strength=match_strength
        )

    def _calculate_capability_match(self, profile: AgentCapabilityProfile, task: TaskRequirements, reasoning: List[str]) -> float:
        """Calculate capability match score."""
        required_capabilities = task.required_capabilities
        if not required_capabilities:
            return 1.0

        # Calculate weighted capability match
        total_weight = 0.0
        weighted_score = 0.0

        for capability in required_capabilities:
            expertise = profile.capabilities.get(capability, 0.0)
            weight = task.capability_weights.get(capability, 1.0)

            weighted_score += expertise * weight
            total_weight += weight

        capability_score = weighted_score / total_weight if total_weight > 0 else 0.0

        # Additional bonus for full coverage
        covered_capabilities = sum(
            1 for cap in required_capabilities
            if cap in profile.capabilities and profile.capabilities[cap] > 0.3
        )
        coverage_bonus = covered_capabilities / len(required_capabilities)

        final_score = (capability_score * 0.8) + (coverage_bonus * 0.2)

        if final_score < 0.5:
            reasoning.append(f"Low capability match: {final_score:.2f}")

        return final_score

    def _calculate_expertise_level(self, profile: AgentCapabilityProfile, task: TaskRequirements, reasoning: List[str]) -> float:
        """Calculate expertise level score."""
        if not task.required_capabilities:
            return 0.8  # Default for no specific requirements

        # Calculate average expertise for required capabilities
        expertise_levels = []
        for capability in task.required_capabilities:
            if capability in profile.capabilities:
                expertise_levels.append(profile.capabilities[capability])

        if not expertise_levels:
            return 0.0

        avg_expertise = statistics.mean(expertise_levels)

        # Apply complexity adjustment
        if task.complexity > 0.7:
            # High complexity tasks need higher expertise
            complexity_penalty = max(0, (task.complexity - 0.7) * 2)
            adjusted_score = max(0, avg_expertise - complexity_penalty)
        else:
            adjusted_score = avg_expertise

        if adjusted_score < 0.6:
            reasoning.append(f"Expertise level may be insufficient: {adjusted_score:.2f}")

        return adjusted_score

    def _calculate_performance_history(self, profile: AgentCapabilityProfile, reasoning: List[str]) -> float:
        """Calculate performance history score."""
        recent_tasks = profile.performance_metrics.get("recent_tasks", 0)
        successful_tasks = profile.performance_metrics.get("successful_tasks", 0)

        if recent_tasks == 0:
            return 0.7  # Default for new agents

        success_rate = successful_tasks / recent_tasks

        # Apply confidence adjustment
        if success_rate > 0.9:
            score = 1.0
        elif success_rate > 0.7:
            score = 0.8
        elif success_rate > 0.5:
            score = 0.6
        else:
            score = 0.4
            reasoning.append(f"Low historical success rate: {success_rate:.1%}")

        return score

    def _calculate_availability(self, agent_state: Optional[AgentState], task: TaskRequirements, reasoning: List[str]) -> float:
        """Calculate availability score."""
        if not agent_state:
            return 0.8  # Default when state unknown

        # Calculate current availability
        current_load = agent_state.current_load
        availability_score = 1.0 - current_load

        # Consider urgency
        if task.urgency > 0.8 and availability_score < 0.5:
            reasoning.append(f"Agent may not be available for urgent task (load: {current_load:.1%})")

        # Consider concurrent tasks
        if agent_state.current_tasks >= agent_state.max_concurrent_tasks:
            availability_score *= 0.5
            reasoning.append("Agent at maximum capacity")

        return max(0.0, availability_score)

    def _calculate_load_balance(self, agent_state: Optional[AgentState], reasoning: List[str]) -> float:
        """Calculate load balance score."""
        if not agent_state:
            return 0.7  # Default when state unknown

        # Prefer agents with lower current load
        load_ratio = agent_state.current_tasks / max(1, agent_state.max_concurrent_tasks)
        load_score = 1.0 - load_ratio

        return load_score

    def _get_strategy_weights(self, strategy: MatchingStrategy) -> Dict[MatchingFactor, float]:
        """Get factor weights for a specific strategy."""
        base_weights = self.factor_weights.copy()

        if strategy == MatchingStrategy.EXACT_MATCH:
            base_weights[MatchingFactor.CAPABILITY_MATCH] = 0.5
            base_weights[MatchingFactor.EXPERTISE_LEVEL] = 0.4
            base_weights[MatchingFactor.AVAILABILITY] = 0.1

        elif strategy == MatchingStrategy.SPECIALIST:
            base_weights[MatchingFactor.EXPERTISE_LEVEL] = 0.4
            base_weights[MatchingFactor.CAPABILITY_MATCH] = 0.3
            base_weights[MatchingFactor.PERFORMANCE_HISTORY] = 0.3

        elif strategy == MatchingStrategy.GENERALIST:
            base_weights[MatchingFactor.CAPABILITY_MATCH] = 0.2
            base_weights[MatchingFactor.AVAILABILITY] = 0.3
            base_weights[MatchingFactor.LOAD_BALANCE] = 0.3

        elif strategy == MatchingStrategy.LOAD_BALANCED:
            base_weights[MatchingFactor.LOAD_BALANCE] = 0.4
            base_weights[MatchingFactor.AVAILABILITY] = 0.4
            base_weights[MatchingFactor.CAPABILITY_MATCH] = 0.2

        return base_weights

    def _calculate_confidence_interval(
        self,
        factor_scores: Dict[MatchingFactor, float],
        weights: Dict[MatchingFactor, float]
    ) -> Tuple[float, float]:
        """Calculate confidence interval for match score."""
        # Calculate variance based on factor score variance
        scores = list(factor_scores.values())
        if len(scores) < 2:
            return (0.0, 0.0)

        variance = statistics.variance(scores)
        std_dev = statistics.sqrt(variance)

        # 95% confidence interval
        margin = 1.96 * std_dev / statistics.sqrt(len(scores))

        mean_score = statistics.mean(scores)
        return (
            max(0.0, mean_score - margin),
            min(1.0, mean_score + margin)
        )

    def _determine_match_strength(self, score: float) -> str:
        """Determine match strength based on score."""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "strong"
        elif score >= 0.4:
            return "moderate"
        else:
            return "weak"

    def predict_task_success(
        self,
        agent_name: str,
        task: TaskRequirements
    ) -> Dict[str, Any]:
        """
        Predict success probability for a specific agent-task combination.

        Args:
            agent_name: Name of the agent
            task: Task requirements

        Returns:
            Success prediction with confidence intervals
        """
        try:
            match_score = self._calculate_match_score(agent_name, task, MatchingStrategy.BALANCED)

            # Base success probability on match score
            base_success_prob = match_score.overall_score

            # Adjust for task complexity and urgency
            complexity_factor = 1.0 - (task.complexity * 0.3)
            urgency_factor = 1.0 - (task.urgency * 0.2)

            adjusted_prob = base_success_prob * complexity_factor * urgency_factor

            # Calculate confidence interval
            confidence_margin = 0.1 * (1 - match_score.overall_score)
            confidence_interval = (
                max(0.0, adjusted_prob - confidence_margin),
                min(1.0, adjusted_prob + confidence_margin)
            )

            return {
                "success_probability": adjusted_prob,
                "confidence_interval": confidence_interval,
                "match_score": match_score.overall_score,
                "match_strength": match_score.match_strength,
                "key_factors": [
                    {"factor": factor.value, "score": score}
                    for factor, score in match_score.factor_scores.items()
                ],
                "reasoning": match_score.reasoning
            }

        except Exception as e:
            self.logger.error(f"Error predicting task success: {e}")
            return {
                "success_probability": 0.5,
                "confidence_interval": (0.0, 1.0),
                "error": str(e)
            }

    def get_matching_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get statistics about recent matching performance."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            recent_matches = [
                match for match in self.matching_history
                if match.execution_time > 0  # Filter out error cases
            ]

            if not recent_matches:
                return {"message": "No matching history available"}

            # Calculate statistics
            avg_matches_per_task = statistics.mean([len(match.all_scores) for match in recent_matches])
            avg_top_score = statistics.mean([
                match.best_matches[0].overall_score
                for match in recent_matches
                if match.best_matches
            ])

            strategy_usage = defaultdict(int)
            for match in recent_matches:
                strategy_usage[match.matching_strategy.value] += 1

            return {
                "period_days": days,
                "total_matches": len(recent_matches),
                "average_candidates_per_task": avg_matches_per_task,
                "average_top_score": avg_top_score,
                "strategy_usage": dict(strategy_usage),
                "most_used_strategy": max(strategy_usage, key=strategy_usage.get) if strategy_usage else None
            }

        except Exception as e:
            self.logger.error(f"Error getting matching statistics: {e}")
            return {"error": str(e)}

    def optimize_matching_weights(self, feedback_data: List[Dict[str, Any]]) -> None:
        """
        Optimize matching weights based on feedback data.

        Args:
            feedback_data: List of feedback entries with actual performance
        """
        try:
            self.logger.info("Optimizing matching weights based on feedback")

            if len(feedback_data) < 10:
                self.logger.warning("Insufficient feedback data for optimization")
                return

            # Analyze which factors correlate with success
            factor_correlations = {}
            for factor in MatchingFactor:
                correlations = []
                for feedback in feedback_data:
                    if factor.value in feedback.get("factor_scores", {}):
                        factor_score = feedback["factor_scores"][factor.value]
                        actual_success = feedback.get("actual_success", 0.5)
                        correlations.append((factor_score, actual_success))

                if len(correlations) >= 5:
                    # Calculate correlation
                    x_values = [c[0] for c in correlations]
                    y_values = [c[1] for c in correlations]

                    if len(set(x_values)) > 1 and len(set(y_values)) > 1:
                        correlation = self._calculate_correlation(x_values, y_values)
                        factor_correlations[factor] = correlation

            # Update weights based on correlations
            if factor_correlations:
                # Normalize correlations to positive weights
                min_corr = min(factor_correlations.values())
                max_corr = max(factor_correlations.values())

                if max_corr > min_corr:
                    total_weight = 0.0
                    for factor, correlation in factor_correlations.items():
                        normalized_weight = (correlation - min_corr) / (max_corr - min_corr)
                        self.factor_weights[factor] = normalized_weight
                        total_weight += normalized_weight

                    # Normalize to sum to 1.0
                    for factor in self.factor_weights:
                        self.factor_weights[factor] /= total_weight

            self.logger.info("Matching weights optimized successfully")

        except Exception as e:
            self.logger.error(f"Error optimizing matching weights: {e}")

    def _calculate_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0

        try:
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)
            sum_y2 = sum(y * y for y in y_values)

            numerator = n * sum_xy - sum_x * sum_y
            denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5

            if denominator == 0:
                return 0.0

            return numerator / denominator
        except:
            return 0.0