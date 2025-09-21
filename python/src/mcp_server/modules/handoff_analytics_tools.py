"""
Handoff Analytics Tools - Advanced analytics for handoff optimization.

This module provides MCP tools for deep analysis of handoff patterns,
performance optimization, and learning insights.

Key Features:
- Pattern recognition and trend analysis
- Performance optimization recommendations
- Capability gap analysis
- ROI analysis for handoffs
- Predictive analytics
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from pydantic import BaseModel, Field
import statistics

from ...agents.orchestration.agent_handoff_engine import HandoffResult, HandoffStatus
from ...agents.orchestration.archon_agency import ArchonAgency

logger = logging.getLogger(__name__)


class AnalyticsTimeRange(BaseModel):
    """Time range for analytics queries."""
    start_time: datetime
    end_time: datetime


class HandoffPattern(BaseModel):
    """Represents a handoff pattern."""
    pattern_id: str
    source_agents: List[str]
    target_agents: List[str]
    strategies: List[str]
    success_rate: float
    frequency: int
    avg_execution_time: float
    confidence_trend: List[float]
    discovered_at: datetime


class OptimizationRecommendation(BaseModel):
    """Represents an optimization recommendation."""
    recommendation_id: str
    type: str  # "capability", "strategy", "threshold", "workflow"
    priority: str  # "low", "medium", "high", "critical"
    description: str
    expected_improvement: float  # Percentage improvement expected
    implementation_effort: str  # "low", "medium", "high"
    agents_affected: List[str]
    created_at: datetime


class CapabilityGapAnalysis(BaseModel):
    """Analysis of capability gaps in the agency."""
    capability_name: str
    current_agents: List[str]
    demand_frequency: int
    success_rate: float
    recommended_agents: List[str]
    priority: str


class HandoffROI(BaseModel):
    """Return on investment analysis for handoffs."""
    handoff_type: str
    time_saved_per_handoff: float  # Minutes
    success_rate_improvement: float  # Percentage points
    implementation_cost: str
    roi_score: float
    break_even_point: str  # Time to break even


class HandoffAnalyticsTools:
    """
    Advanced analytics tools for handoff optimization.

    This class provides tools for:
    - Pattern recognition and trend analysis
    - Performance optimization recommendations
    - Capability gap analysis
    - ROI analysis
    - Predictive analytics
    """

    def __init__(self, agency: ArchonAgency):
        """Initialize analytics tools with agency reference."""
        self.agency = agency
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def analyze_handoff_patterns(
        self,
        time_range_hours: int = 168,  # 7 days default
        min_frequency: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Analyze handoff patterns over time.

        Args:
            time_range_hours: Time range to analyze
            min_frequency: Minimum frequency to consider a pattern

        Returns:
            List of discovered patterns
        """
        try:
            self.logger.info(f"Analyzing handoff patterns over {time_range_hours} hours")

            # Get time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=time_range_hours)

            # Filter handoff history by time range
            patterns = []
            recent_handoffs = self._filter_handoffs_by_time(start_time, end_time)

            # Analyze source-target patterns
            source_target_patterns = self._analyze_source_target_patterns(recent_handoffs, min_frequency)

            # Analyze strategy patterns
            strategy_patterns = self._analyze_strategy_patterns(recent_handoffs)

            # Analyze temporal patterns
            temporal_patterns = self._analyze_temporal_patterns(recent_handoffs)

            # Combine all patterns
            all_patterns = source_target_patterns + strategy_patterns + temporal_patterns

            return [pattern.dict() for pattern in all_patterns]

        except Exception as e:
            self.logger.error(f"Error analyzing handoff patterns: {e}")
            return []

    def _filter_handoffs_by_time(self, start_time: datetime, end_time: datetime) -> List[HandoffResult]:
        """Filter handoffs by time range."""
        # This would need to be implemented based on how handoff history is stored
        # For now, return empty list
        return []

    def _analyze_source_target_patterns(self, handoffs: List[HandoffResult], min_frequency: int) -> List[HandoffPattern]:
        """Analyze source-target agent patterns."""
        patterns = []

        # Count source-target pairs
        pair_counts = {}
        pair_success = {}

        for handoff in handoffs:
            key = f"{handoff.decision.source_agent}->{handoff.decision.target_agent}"
            pair_counts[key] = pair_counts.get(key, 0) + 1

            if key not in pair_success:
                pair_success[key] = {"success": 0, "total": 0}

            pair_success[key]["total"] += 1
            if handoff.status == HandoffStatus.COMPLETED:
                pair_success[key]["success"] += 1

        # Create patterns for frequent pairs
        for pair, count in pair_counts.items():
            if count >= min_frequency:
                source, target = pair.split("->")
                success_stats = pair_success[pair]
                success_rate = success_stats["success"] / success_stats["total"]

                pattern = HandoffPattern(
                    pattern_id=f"pattern_{len(patterns)}",
                    source_agents=[source],
                    target_agents=[target],
                    strategies=[],  # Would need to extract from handoffs
                    success_rate=success_rate,
                    frequency=count,
                    avg_execution_time=0.0,  # Would calculate from handoffs
                    confidence_trend=[],  # Would extract from handoffs
                    discovered_at=datetime.utcnow()
                )
                patterns.append(pattern)

        return patterns

    def _analyze_strategy_patterns(self, handoffs: List[HandoffResult]) -> List[HandoffPattern]:
        """Analyze strategy usage patterns."""
        # Placeholder implementation
        return []

    def _analyze_temporal_patterns(self, handoffs: List[HandoffResult]) -> List[HandoffPattern]:
        """Analyze temporal patterns in handoffs."""
        # Placeholder implementation
        return []

    async def generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate optimization recommendations based on analytics.

        Returns:
            List of optimization recommendations
        """
        try:
            self.logger.info("Generating optimization recommendations")

            recommendations = []

            # Get current statistics
            stats = await self.agency.get_handoff_statistics()
            handoff_stats = stats.get("handoff_engine", {})

            # Analyze strategy performance
            strategy_stats = handoff_stats.get("strategy_statistics", {})
            strategy_recommendations = self._generate_strategy_recommendations(strategy_stats)
            recommendations.extend(strategy_recommendations)

            # Analyze agent performance
            agent_recommendations = self._generate_agent_recommendations(handoff_stats)
            recommendations.extend(agent_recommendations)

            # Analyze capability utilization
            capability_recommendations = await self._generate_capability_recommendations()
            recommendations.extend(capability_recommendations)

            # Sort by priority
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            recommendations.sort(key=lambda r: priority_order.get(r.priority, 4))

            return [rec.dict() for rec in recommendations]

        except Exception as e:
            self.logger.error(f"Error generating optimization recommendations: {e}")
            return []

    def _generate_strategy_recommendations(self, strategy_stats: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate strategy-specific recommendations."""
        recommendations = []

        for strategy_name, stats in strategy_stats.items():
            success_rate = stats.get("success_rate", 0)
            avg_time = stats.get("avg_execution_time", 0)

            if success_rate < 0.6:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"strategy_{strategy_name}_success",
                    type="strategy",
                    priority="high",
                    description=f"Low success rate ({success_rate:.1%}) for {strategy_name} strategy",
                    expected_improvement=20.0,
                    implementation_effort="medium",
                    agents_affected=[],
                    created_at=datetime.utcnow()
                ))

            if avg_time > 30.0:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"strategy_{strategy_name}_performance",
                    type="strategy",
                    priority="medium",
                    description=f"High execution time ({avg_time:.1f}s) for {strategy_name} strategy",
                    expected_improvement=15.0,
                    implementation_effort="low",
                    agents_affected=[],
                    created_at=datetime.utcnow()
                ))

        return recommendations

    def _generate_agent_recommendations(self, handoff_stats: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate agent-specific recommendations."""
        recommendations = []

        # Analyze agent participation
        total_agents = handoff_stats.get("agents_participating", 0)
        total_handoffs = handoff_stats.get("total_handoffs", 0)

        if total_agents < 3 and total_handoffs > 10:
            recommendations.append(OptimizationRecommendation(
                recommendation_id="agent_diversity",
                type="capability",
                priority="medium",
                description="Low agent diversity in handoffs, consider adding specialized agents",
                expected_improvement=25.0,
                implementation_effort="high",
                agents_affected=[],
                created_at=datetime.utcnow()
            ))

        return recommendations

    async def _generate_capability_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate capability-specific recommendations."""
        recommendations = []

        # Analyze learning insights
        learning_insights = self.agency.learning_engine.get_learning_insights()
        capability_performance = learning_insights.get("capability_performance", {})

        for capability, perf in capability_performance.items():
            if perf.get("success_rate", 0) < 0.5:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"capability_{capability}",
                    type="capability",
                    priority="high",
                    description=f"Underperforming capability: {capability} ({perf.get('success_rate', 0):.1%} success rate)",
                    expected_improvement=30.0,
                    implementation_effort="medium",
                    agents_affected=[],
                    created_at=datetime.utcnow()
                ))

        return recommendations

    async def analyze_capability_gaps(self) -> List[Dict[str, Any]]:
        """
        Analyze capability gaps in the agency.

        Returns:
            List of capability gap analyses
        """
        try:
            self.logger.info("Analyzing capability gaps")

            gap_analyses = []

            # Get current capabilities
            current_capabilities = set()
            for capability in self.agency.handoff_engine.agent_capabilities.values():
                current_capabilities.update(capability.capabilities)

            # Analyze demand based on handoff history (placeholder)
            demand_analysis = {
                "coding": {"frequency": 15, "success_rate": 0.85},
                "testing": {"frequency": 8, "success_rate": 0.72},
                "design": {"frequency": 12, "success_rate": 0.78},
                "security": {"frequency": 5, "success_rate": 0.45},
                "performance": {"frequency": 7, "success_rate": 0.65},
                "documentation": {"frequency": 10, "success_rate": 0.90}
            }

            # Identify gaps
            for capability, demand in demand_analysis.items():
                if capability not in current_capabilities:
                    gap_analysis = CapabilityGapAnalysis(
                        capability_name=capability,
                        current_agents=[],
                        demand_frequency=demand["frequency"],
                        success_rate=0.0,
                        recommended_agents=["Need specialist agent"],
                        priority="high" if demand["frequency"] > 10 else "medium"
                    )
                    gap_analyses.append(gap_analysis)
                elif demand["success_rate"] < 0.6:
                    # Find agents with this capability
                    agents_with_capability = []
                    for agent_name, agent_cap in self.agency.handoff_engine.agent_capabilities.items():
                        if capability in agent_cap.capabilities:
                            agents_with_capability.append(agent_name)

                    gap_analysis = CapabilityGapAnalysis(
                        capability_name=capability,
                        current_agents=agents_with_capability,
                        demand_frequency=demand["frequency"],
                        success_rate=demand["success_rate"],
                        recommended_agents=["Consider training or new agents"],
                        priority="medium"
                    )
                    gap_analyses.append(gap_analysis)

            return [analysis.dict() for analysis in gap_analyses]

        except Exception as e:
            self.logger.error(f"Error analyzing capability gaps: {e}")
            return []

    async def calculate_handoff_roi(self) -> Dict[str, Any]:
        """
        Calculate return on investment for handoffs.

        Returns:
            ROI analysis results
        """
        try:
            self.logger.info("Calculating handoff ROI")

            # Get statistics
            stats = await self.agency.get_handoff_statistics()
            handoff_stats = stats.get("handoff_engine", {})

            total_handoffs = handoff_stats.get("total_handoffs", 0)
            success_rate = handoff_stats.get("success_rate", 0)
            avg_time = handoff_stats.get("strategy_statistics", {}).get("sequential", {}).get("avg_execution_time", 0)

            # Calculate ROI metrics
            if total_handoffs > 0:
                successful_handoffs = int(total_handoffs * success_rate)
                time_saved = successful_handoffs * (10 - avg_time / 60)  # Assume 10 min without handoff
                success_improvement = (success_rate - 0.5) * 100  # Improvement over 50% baseline

                roi_analysis = {
                    "total_handoffs": total_handoffs,
                    "successful_handoffs": successful_handoffs,
                    "success_rate": success_rate,
                    "average_execution_time_minutes": avg_time / 60,
                    "estimated_time_saved_minutes": max(0, time_saved),
                    "success_rate_improvement_points": max(0, success_improvement),
                    "implementation_cost": "Low (existing infrastructure)",
                    "roi_score": min(10, success_rate * 10),  # Scale 0-10
                    "break_even_point": "Immediate (using existing infrastructure)"
                }
            else:
                roi_analysis = {
                    "total_handoffs": 0,
                    "successful_handoffs": 0,
                    "success_rate": 0,
                    "average_execution_time_minutes": 0,
                    "estimated_time_saved_minutes": 0,
                    "success_rate_improvement_points": 0,
                    "implementation_cost": "Low",
                    "roi_score": 0,
                    "break_even_point": "N/A"
                }

            return roi_analysis

        except Exception as e:
            self.logger.error(f"Error calculating handoff ROI: {e}")
            return {"error": str(e)}

    async def predict_handoff_success(
        self,
        source_agent: str,
        target_agent: str,
        task_description: str,
        strategy: str
    ) -> Dict[str, Any]:
        """
        Predict success probability for a handoff.

        Args:
            source_agent: Source agent name
            target_agent: Target agent name
            task_description: Task description
            strategy: Handoff strategy

        Returns:
            Success prediction with confidence intervals
        """
        try:
            self.logger.info(f"Predicting handoff success: {source_agent} -> {target_agent}")

            # Extract required capabilities
            required_capabilities = self.agency.handoff_engine._extract_task_capabilities(
                task_description, task_description
            )

            # Get agent capabilities
            source_cap = self.agency.handoff_engine.agent_capabilities.get(source_agent)
            target_cap = self.agency.handoff_engine.agent_capabilities.get(target_agent)

            if not source_cap or not target_cap:
                return {
                    "error": "Agent capabilities not found",
                    "success_probability": 0.0,
                    "confidence_interval": [0.0, 0.0]
                }

            # Calculate capability match score
            capability_score = target_cap.get_capability_score(required_capabilities)

            # Get historical performance
            historical_success = 0.7  # Default baseline
            if hasattr(self.agency.learning_engine, 'capability_performance'):
                for cap in required_capabilities:
                    if cap in self.agency.learning_engine.capability_performance:
                        perf = self.agency.learning_engine.capability_performance[cap]
                        historical_success = max(historical_success, perf.success_rate)

            # Get strategy performance
            strategy_success = 0.7  # Default baseline
            strategy_stats = self.agency.handoff_engine.get_handoff_statistics().get("strategy_statistics", {})
            if strategy in strategy_stats:
                strategy_success = strategy_stats[strategy].get("success_rate", 0.7)

            # Combine factors for final prediction
            weights = {"capability": 0.4, "historical": 0.3, "strategy": 0.3}
            final_probability = (
                weights["capability"] * capability_score +
                weights["historical"] * historical_success +
                weights["strategy"] * strategy_success
            )

            # Calculate confidence interval
            confidence_margin = 0.1 * (1 - capability_score)  # Wider interval for lower capability match
            confidence_interval = [
                max(0.0, final_probability - confidence_margin),
                min(1.0, final_probability + confidence_margin)
            ]

            return {
                "success_probability": final_probability,
                "confidence_interval": confidence_interval,
                "capability_match_score": capability_score,
                "historical_success_rate": historical_success,
                "strategy_success_rate": strategy_success,
                "prediction_factors": {
                    "capability_weight": weights["capability"],
                    "historical_weight": weights["historical"],
                    "strategy_weight": weights["strategy"]
                }
            }

        except Exception as e:
            self.logger.error(f"Error predicting handoff success: {e}")
            return {
                "error": str(e),
                "success_probability": 0.0,
                "confidence_interval": [0.0, 0.0]
            }

    async def get_comprehensive_dashboard(
        self,
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get comprehensive handoff analytics dashboard.

        Args:
            time_range_hours: Time range for analytics

        Returns:
            Complete dashboard data
        """
        try:
            self.logger.info(f"Generating comprehensive dashboard for {time_range_hours} hours")

            # Get all analytics data
            patterns = await self.analyze_handoff_patterns(time_range_hours)
            recommendations = await self.generate_optimization_recommendations()
            capability_gaps = await self.analyze_capability_gaps()
            roi_data = await self.calculate_handoff_roi()

            # Get basic statistics
            stats = await self.agency.get_handoff_statistics()

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "time_range_hours": time_range_hours,
                "basic_statistics": stats,
                "patterns_discovered": patterns,
                "optimization_recommendations": recommendations,
                "capability_gaps": capability_gaps,
                "roi_analysis": roi_data,
                "dashboard_summary": {
                    "total_patterns": len(patterns),
                    "high_priority_recommendations": len([r for r in recommendations if r.get("priority") == "high"]),
                    "critical_gaps": len([g for g in capability_gaps if g.get("priority") == "high"]),
                    "roi_score": roi_data.get("roi_score", 0)
                }
            }

        except Exception as e:
            self.logger.error(f"Error generating comprehensive dashboard: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}


# MCP Tool functions
async def analyze_handoff_patterns(
    time_range_hours: int = 168,
    min_frequency: int = 3,
    # Agency instance injected by MCP server
    agency: Optional[ArchonAgency] = None
) -> Dict[str, Any]:
    """Analyze handoff patterns over time."""
    if not agency:
        raise ValueError("Agency instance not provided")

    tools = HandoffAnalyticsTools(agency)
    patterns = await tools.analyze_handoff_patterns(time_range_hours, min_frequency)
    return {"patterns": patterns, "time_range_hours": time_range_hours}


async def generate_optimization_recommendations(
    # Agency instance injected by MCP server
    agency: Optional[ArchonAgency] = None
) -> Dict[str, Any]:
    """Generate optimization recommendations."""
    if not agency:
        raise ValueError("Agency instance not provided")

    tools = HandoffAnalyticsTools(agency)
    recommendations = await tools.generate_optimization_recommendations()
    return {"recommendations": recommendations}


async def get_handoff_dashboard(
    time_range_hours: int = 24,
    # Agency instance injected by MCP server
    agency: Optional[ArchonAgency] = None
) -> Dict[str, Any]:
    """Get comprehensive handoff analytics dashboard."""
    if not agency:
        raise ValueError("Agency instance not provided")

    tools = HandoffAnalyticsTools(agency)
    dashboard = await tools.get_comprehensive_dashboard(time_range_hours)
    return dashboard