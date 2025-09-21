"""
Handoff Learning - Learning from successful handoff patterns.

This module implements learning capabilities to improve handoff decisions
based on historical performance and patterns.

Key Features:
- Performance pattern analysis
- Capability expertise refinement
- Strategy optimization
- Confidence scoring improvements
- Adaptive threshold adjustment
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
from collections import defaultdict

from .agent_handoff_engine import (
    HandoffResult, HandoffDecision, HandoffStatus, HandoffStrategy,
    AgentCapability, HandoffTrigger
)

logger = logging.getLogger(__name__)


class LearningEventType(Enum):
    """Types of learning events."""
    HANDOFF_COMPLETED = "handoff_completed"
    HANDOFF_FAILED = "handoff_failed"
    STRATEGY_PERFORMANCE = "strategy_performance"
    CAPABILITY_MATCH = "capability_match"
    CONFIDENCE_CALIBRATION = "confidence_calibration"


@dataclass
class LearningEvent:
    """Represents a learning event."""
    event_id: str
    event_type: LearningEventType
    timestamp: datetime
    data: Dict[str, Any]
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "confidence": self.confidence
        }


@dataclass
class CapabilityPerformance:
    """Performance metrics for agent capabilities."""
    capability_name: str
    success_count: int = 0
    failure_count: int = 0
    average_confidence: float = 0.0
    average_execution_time: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    def update(self, success: bool, confidence: float, execution_time: float) -> None:
        """Update performance metrics."""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        # Update moving averages
        total_events = self.success_count + self.failure_count
        if total_events == 1:
            self.average_confidence = confidence
            self.average_execution_time = execution_time
        else:
            alpha = 2.0 / (total_events + 1)  # Adaptive learning rate
            self.average_confidence = (1 - alpha) * self.average_confidence + alpha * confidence
            self.average_execution_time = (1 - alpha) * self.average_execution_time + alpha * execution_time

        self.last_updated = datetime.utcnow()


@dataclass
class StrategyPattern:
    """Pattern recognition for handoff strategies."""
    strategy: HandoffStrategy
    success_patterns: List[Dict[str, Any]] = field(default_factory=list)
    failure_patterns: List[Dict[str, Any]] = field(default_factory=list)
    optimal_conditions: Dict[str, Any] = field(default_factory=dict)

    def add_pattern(self, success: bool, context: Dict[str, Any]) -> None:
        """Add a pattern based on handoff outcome."""
        pattern = {
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        }

        if success:
            self.success_patterns.append(pattern)
        else:
            self.failure_patterns.append(pattern)

        # Keep only recent patterns (last 50)
        if len(self.success_patterns) > 50:
            self.success_patterns = self.success_patterns[-50:]
        if len(self.failure_patterns) > 50:
            self.failure_patterns = self.failure_patterns[-50:]

    def get_optimal_conditions(self) -> Dict[str, Any]:
        """Get optimal conditions for this strategy."""
        if not self.success_patterns:
            return {}

        # Analyze successful patterns
        complexity_scores = []
        confidence_scores = []

        for pattern in self.success_patterns:
            context = pattern["context"]
            complexity_scores.append(context.get("task_complexity", 0))
            confidence_scores.append(context.get("confidence_score", 0))

        return {
            "strategy": self.strategy.value,
            "success_rate": len(self.success_patterns) / (len(self.success_patterns) + len(self.failure_patterns)),
            "avg_task_complexity": statistics.mean(complexity_scores) if complexity_scores else 0,
            "avg_confidence_threshold": statistics.mean(confidence_scores) if confidence_scores else 0,
            "optimal_for_complex_tasks": statistics.mean(complexity_scores) > 50 if complexity_scores else False,
            "optimal_for_low_confidence": statistics.mean(confidence_scores) < 0.7 if confidence_scores else False
        }


class HandoffLearningEngine:
    """
    Learning engine for improving handoff decisions.

    This engine learns from:
    - Historical handoff performance
    - Strategy success patterns
    - Capability matching accuracy
    - Confidence scoring calibration
    """

    def __init__(self):
        """Initialize the learning engine."""
        self.learning_events: List[LearningEvent] = []
        self.capability_performance: Dict[str, CapabilityPerformance] = {}
        self.strategy_patterns: Dict[HandoffStrategy, StrategyPattern] = {}
        self.agent_handoff_history: Dict[str, List[HandoffResult]] = defaultdict(list)
        self.learning_confidence_thresholds: Dict[str, float] = {}
        self.last_learning_cycle: Optional[datetime] = None

        # Initialize strategy patterns
        for strategy in HandoffStrategy:
            self.strategy_patterns[strategy] = StrategyPattern(strategy)

        logger.info("HandoffLearningEngine initialized")

    async def record_handoff_result(self, result: HandoffResult) -> None:
        """Record a handoff result for learning."""
        try:
            # Create learning event
            event = LearningEvent(
                event_id=str(uuid.uuid4()),
                event_type=LearningEventType.HANDOFF_COMPLETED if result.status == HandoffStatus.COMPLETED else LearningEventType.HANDOFF_FAILED,
                timestamp=datetime.utcnow(),
                data=result.to_dict()
            )

            self.learning_events.append(event)

            # Update agent handoff history
            target_agent = result.decision.target_agent
            self.agent_handoff_history[target_agent].append(result)

            # Update capability performance
            await self._update_capability_performance(result)

            # Update strategy patterns
            await self._update_strategy_patterns(result)

            # Update confidence thresholds
            await self._update_confidence_thresholds(result)

            logger.debug(f"Recorded handoff result: {result.decision.source_agent} -> {result.decision.target_agent}")

        except Exception as e:
            logger.error(f"Error recording handoff result: {e}")

    async def _update_capability_performance(self, result: HandoffResult) -> None:
        """Update capability performance metrics."""
        try:
            # Extract capabilities from task
            task_capabilities = self._extract_capabilities_from_task(result.decision.context.task_description)

            success = result.status == HandoffStatus.COMPLETED
            confidence = result.decision.confidence_score
            execution_time = result.execution_time

            # Update each capability
            for capability in task_capabilities:
                if capability not in self.capability_performance:
                    self.capability_performance[capability] = CapabilityPerformance(capability)

                self.capability_performance[capability].update(success, confidence, execution_time)

        except Exception as e:
            logger.error(f"Error updating capability performance: {e}")

    def _extract_capabilities_from_task(self, task_description: str) -> List[str]:
        """Extract capabilities from task description."""
        capabilities = []
        task_lower = task_description.lower()

        capability_keywords = {
            "coding": ["code", "program", "develop", "implement", "function", "class"],
            "testing": ["test", "unit", "integration", "assert", "mock"],
            "design": ["design", "architecture", "ui", "ux", "layout"],
            "analysis": ["analyze", "research", "data", "metrics", "statistics"],
            "security": ["security", "authentication", "authorization", "encrypt"],
            "performance": ["performance", "optimize", "speed", "efficiency"],
            "documentation": ["document", "readme", "guide", "manual"],
            "deployment": ["deploy", "release", "publish", "distribute"]
        }

        for capability, keywords in capability_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                capabilities.append(capability)

        return capabilities

    async def _update_strategy_patterns(self, result: HandoffResult) -> None:
        """Update strategy patterns based on handoff result."""
        try:
            strategy = result.decision.strategy
            pattern = self.strategy_patterns.get(strategy)

            if pattern:
                success = result.status == HandoffStatus.COMPLETED
                context = {
                    "task_complexity": len(result.decision.context.task_description.split()),
                    "confidence_score": result.decision.confidence_score,
                    "execution_time": result.execution_time,
                    "trigger": result.decision.trigger.value
                }

                pattern.add_pattern(success, context)

        except Exception as e:
            logger.error(f"Error updating strategy patterns: {e}")

    async def _update_confidence_thresholds(self, result: HandoffResult) -> None:
        """Update confidence thresholds based on results."""
        try:
            source_agent = result.decision.source_agent
            capability_key = f"{source_agent}_confidence_threshold"

            if capability_key not in self.learning_confidence_thresholds:
                self.learning_confidence_thresholds[capability_key] = 0.7

            # Adjust threshold based on results
            success = result.status == HandoffStatus.COMPLETED
            current_threshold = self.learning_confidence_thresholds[capability_key]

            if success and result.decision.confidence_score < current_threshold:
                # Lower threshold for successful low-confidence handoffs
                self.learning_confidence_thresholds[capability_key] *= 0.95
            elif not success and result.decision.confidence_score > current_threshold:
                # Raise threshold for failed high-confidence handoffs
                self.learning_confidence_thresholds[capability_key] *= 1.05

            # Keep within reasonable bounds
            self.learning_confidence_thresholds[capability_key] = max(0.3, min(0.95, self.learning_confidence_thresholds[capability_key]))

        except Exception as e:
            logger.error(f"Error updating confidence thresholds: {e}")

    async def run_learning_cycle(self) -> None:
        """Run a complete learning cycle."""
        try:
            self.last_learning_cycle = datetime.utcnow()

            # Analyze recent events
            recent_events = self._get_recent_events(hours=24)
            if not recent_events:
                logger.debug("No recent events for learning cycle")
                return

            logger.info(f"Running learning cycle with {len(recent_events)} recent events")

            # Learn capability improvements
            await self._learn_capability_improvements(recent_events)

            # Learn strategy optimizations
            await self._learn_strategy_optimizations(recent_events)

            # Learn confidence calibration
            await self._learn_confidence_calibration(recent_events)

            # Clean up old events
            await self._cleanup_old_events()

            logger.info("Learning cycle completed")

        except Exception as e:
            logger.error(f"Error in learning cycle: {e}")

    def _get_recent_events(self, hours: int = 24) -> List[LearningEvent]:
        """Get learning events from the last N hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [event for event in self.learning_events if event.timestamp > cutoff_time]

    async def _learn_capability_improvements(self, events: List[LearningEvent]) -> None:
        """Learn capability improvements from events."""
        try:
            # Analyze which capabilities are most successful
            capability_success_rates = defaultdict(lambda: {"success": 0, "total": 0})

            for event in events:
                if event.event_type in [LearningEventType.HANDOFF_COMPLETED, LearningEventType.HANDOFF_FAILED]:
                    data = event.data
                    if "decision" in data and "context" in data["decision"]:
                        capabilities = self._extract_capabilities_from_task(data["decision"]["context"]["task_description"])
                        success = event.event_type == LearningEventType.HANDOFF_COMPLETED

                        for capability in capabilities:
                            capability_success_rates[capability]["total"] += 1
                            if success:
                                capability_success_rates[capability]["success"] += 1

            # Log insights
            for capability, stats in capability_success_rates.items():
                if stats["total"] >= 3:  # Minimum threshold
                    success_rate = stats["success"] / stats["total"]
                    if success_rate > 0.8:
                        logger.info(f"High-performing capability: {capability} ({success_rate:.2f} success rate)")
                    elif success_rate < 0.5:
                        logger.warning(f"Underperforming capability: {capability} ({success_rate:.2f} success rate)")

        except Exception as e:
            logger.error(f"Error learning capability improvements: {e}")

    async def _learn_strategy_optimizations(self, events: List[LearningEvent]) -> None:
        """Learn strategy optimizations from events."""
        try:
            # Update optimal conditions for each strategy
            for strategy, pattern in self.strategy_patterns.items():
                optimal_conditions = pattern.get_optimal_conditions()
                if optimal_conditions["success_rate"] > 0:
                    logger.info(f"Strategy {strategy.value} optimal conditions: {optimal_conditions}")

        except Exception as e:
            logger.error(f"Error learning strategy optimizations: {e}")

    async def _learn_confidence_calibration(self, events: List[LearningEvent]) -> None:
        """Learn confidence calibration improvements."""
        try:
            # Analyze confidence vs actual success
            confidence_predictions = []

            for event in events:
                if event.event_type in [LearningEventType.HANDOFF_COMPLETED, LearningEventType.HANDOFF_FAILED]:
                    data = event.data
                    if "decision" in data:
                        predicted_confidence = data["decision"].get("confidence_score", 0)
                        actual_success = event.event_type == LearningEventType.HANDOFF_COMPLETED

                        confidence_predictions.append((predicted_confidence, actual_success))

            if confidence_predictions:
                # Calculate calibration metrics
                avg_confidence = statistics.mean(pred for pred, _ in confidence_predictions)
                actual_success_rate = sum(1 for _, success in confidence_predictions if success) / len(confidence_predictions)

                calibration_error = abs(avg_confidence - actual_success_rate)

                if calibration_error > 0.1:
                    logger.warning(f"Confidence calibration issue: predicted {avg_confidence:.2f}, actual {actual_success_rate:.2f}")
                else:
                    logger.info(f"Confidence well calibrated: {avg_confidence:.2f} vs {actual_success_rate:.2f}")

        except Exception as e:
            logger.error(f"Error learning confidence calibration: {e}")

    async def _cleanup_old_events(self, days: int = 30) -> None:
        """Clean up old learning events."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            original_count = len(self.learning_events)

            self.learning_events = [event for event in self.learning_events if event.timestamp > cutoff_time]

            removed_count = original_count - len(self.learning_events)
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old learning events")

        except Exception as e:
            logger.error(f"Error cleaning up old events: {e}")

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get learning insights and recommendations."""
        try:
            insights = {
                "total_learning_events": len(self.learning_events),
                "last_learning_cycle": self.last_learning_cycle.isoformat() if self.last_learning_cycle else None,
                "capability_performance": {},
                "strategy_recommendations": {},
                "confidence_calibration": {}
            }

            # Capability performance insights
            for capability, perf in self.capability_performance.items():
                if perf.success_count + perf.failure_count >= 3:
                    insights["capability_performance"][capability] = {
                        "success_rate": perf.success_rate,
                        "average_confidence": perf.average_confidence,
                        "average_execution_time": perf.average_execution_time,
                        "total_events": perf.success_count + perf.failure_count
                    }

            # Strategy recommendations
            for strategy, pattern in self.strategy_patterns.items():
                optimal = pattern.get_optimal_conditions()
                if optimal.get("success_rate", 0) > 0:
                    insights["strategy_recommendations"][strategy.value] = optimal

            # Confidence thresholds
            insights["confidence_calibration"]["thresholds"] = self.learning_confidence_thresholds.copy()

            return insights

        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return {"error": str(e)}

    def get_capability_recommendations(self, task_description: str) -> List[Dict[str, Any]]:
        """Get capability-based recommendations for a task."""
        try:
            capabilities = self._extract_capabilities_from_task(task_description)
            recommendations = []

            for capability in capabilities:
                if capability in self.capability_performance:
                    perf = self.capability_performance[capability]
                    recommendations.append({
                        "capability": capability,
                        "success_rate": perf.success_rate,
                        "confidence": perf.average_confidence,
                        "execution_time": perf.average_execution_time,
                        "recommended": perf.success_rate > 0.7
                    })

            # Sort by success rate
            recommendations.sort(key=lambda x: x["success_rate"], reverse=True)

            return recommendations

        except Exception as e:
            logger.error(f"Error getting capability recommendations: {e}")
            return []

    async def get_strategy_recommendation(
        self,
        task_description: str,
        current_confidence: float,
        source_agent: str
    ) -> Dict[str, Any]:
        """Get strategy recommendation for a handoff."""
        try:
            task_complexity = len(task_description.split())

            # Get capability recommendations
            capability_recs = self.get_capability_recommendations(task_description)

            # Determine strategy based on patterns
            best_strategy = HandoffStrategy.SEQUENTIAL
            best_score = 0.0

            for strategy, pattern in self.strategy_patterns.items():
                optimal = pattern.get_optimal_conditions()

                # Score based on optimal conditions
                score = 0.0

                # Check complexity match
                if optimal.get("optimal_for_complex_tasks") and task_complexity > 50:
                    score += 0.3
                elif not optimal.get("optimal_for_complex_tasks") and task_complexity <= 50:
                    score += 0.3

                # Check confidence match
                if optimal.get("optimal_for_low_confidence") and current_confidence < 0.7:
                    score += 0.3
                elif not optimal.get("optimal_for_low_confidence") and current_confidence >= 0.7:
                    score += 0.3

                # Base success rate
                score += optimal.get("success_rate", 0) * 0.4

                if score > best_score:
                    best_score = score
                    best_strategy = strategy

            # Get agent-specific threshold
            threshold_key = f"{source_agent}_confidence_threshold"
            confidence_threshold = self.learning_confidence_thresholds.get(threshold_key, 0.7)

            return {
                "recommended_strategy": best_strategy.value,
                "confidence": best_score,
                "confidence_threshold": confidence_threshold,
                "reasoning": f"Based on {len(capability_recs)} capabilities and task complexity {task_complexity}",
                "capability_insights": capability_recs
            }

        except Exception as e:
            logger.error(f"Error getting strategy recommendation: {e}")
            return {
                "recommended_strategy": HandoffStrategy.SEQUENTIAL.value,
                "confidence": 0.5,
                "error": str(e)
            }