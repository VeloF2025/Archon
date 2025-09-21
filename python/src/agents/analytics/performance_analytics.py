"""
Performance Analytics Component
Comprehensive performance metrics and insights for conversation analytics
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
import numpy as np
import pandas as pd
from collections import defaultdict, deque, Counter
from concurrent.futures import ThreadPoolExecutor
import statistics

logger = logging.getLogger(__name__)


class PerformanceMetricType(Enum):
    """Types of performance metrics"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    SUCCESS_RATE = "success_rate"
    ERROR_RATE = "error_rate"
    COLLABORATION_EFFICIENCY = "collaboration_efficiency"
    KNOWLEDGE_UTILIZATION = "knowledge_utilization"
    HANDOFF_EFFICIENCY = "handoff_efficiency"
    TASK_COMPLETION_RATE = "task_completion_rate"
    RESOURCE_UTILIZATION = "resource_utilization"
    SCALABILITY_METRIC = "scalability_metric"


class PerformanceTrend(Enum):
    """Performance trend directions"""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"


class PerformanceGrade(Enum):
    """Performance grades"""
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"  # 80-89%
    SATISFACTORY = "satisfactory"  # 70-79%
    NEEDS_IMPROVEMENT = "needs_improvement"  # 60-69%
    POOR = "poor"  # < 60%


@dataclass
class PerformanceMetric:
    """Represents a performance metric"""
    metric_id: str
    metric_type: PerformanceMetricType
    value: float
    timestamp: datetime
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class PerformanceThreshold:
    """Represents performance thresholds"""
    metric_type: PerformanceMetricType
    excellent: float
    good: float
    satisfactory: float
    needs_improvement: float
    unit: str


@dataclass
class AgentPerformanceProfile:
    """Performance profile for an agent"""
    agent_id: str
    metrics: Dict[PerformanceMetricType, List[PerformanceMetric]] = field(default_factory=dict)
    current_score: float = 0.0
    trend: PerformanceTrend = PerformanceTrend.STABLE
    grade: PerformanceGrade = PerformanceGrade.SATISFACTORY
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SystemPerformanceReport:
    """Comprehensive system performance report"""
    report_id: str
    generated_at: datetime = field(default_factory=datetime.utcnow)
    time_range: Tuple[datetime, datetime] = field(default_factory=lambda: (datetime.utcnow() - timedelta(hours=24), datetime.utcnow()))
    overall_score: float = 0.0
    agent_profiles: Dict[str, AgentPerformanceProfile] = field(default_factory=dict)
    system_metrics: Dict[PerformanceMetricType, float] = field(default_factory=dict)
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    optimization_opportunities: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class ConversationPerformanceAnalyzer:
    """
    Advanced performance analytics for conversation systems
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics_history: Dict[PerformanceMetricType, List[PerformanceMetric]] = defaultdict(list)
        self.agent_profiles: Dict[str, AgentPerformanceProfile] = {}
        self.thresholds = self._initialize_thresholds()
        self.benchmark_data = self._initialize_benchmarks()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def _initialize_thresholds(self) -> Dict[PerformanceMetricType, PerformanceThreshold]:
        """Initialize performance thresholds"""
        return {
            PerformanceMetricType.RESPONSE_TIME: PerformanceThreshold(
                metric_type=PerformanceMetricType.RESPONSE_TIME,
                excellent=30.0,    # < 30 seconds
                good=60.0,         # < 60 seconds
                satisfactory=120.0,  # < 2 minutes
                needs_improvement=300.0,  # < 5 minutes
                unit="seconds"
            ),
            PerformanceMetricType.SUCCESS_RATE: PerformanceThreshold(
                metric_type=PerformanceMetricType.SUCCESS_RATE,
                excellent=0.95,   # > 95%
                good=0.90,        # > 90%
                satisfactory=0.80,  # > 80%
                needs_improvement=0.70,  # > 70%
                unit="percentage"
            ),
            PerformanceMetricType.COLLABORATION_EFFICIENCY: PerformanceThreshold(
                metric_type=PerformanceMetricType.COLLABORATION_EFFICIENCY,
                excellent=0.90,   # > 90%
                good=0.80,        # > 80%
                satisfactory=0.70,  # > 70%
                needs_improvement=0.60,  # > 60%
                unit="score"
            ),
            PerformanceMetricType.HANDOFF_EFFICIENCY: PerformanceThreshold(
                metric_type=PerformanceMetricType.HANDOFF_EFFICIENCY,
                excellent=0.95,   # > 95%
                good=0.85,        # > 85%
                satisfactory=0.75,  # > 75%
                needs_improvement=0.65,  # > 65%
                unit="score"
            ),
            PerformanceMetricType.THROUGHPUT: PerformanceThreshold(
                metric_type=PerformanceMetricType.THROUGHPUT,
                excellent=10.0,   # > 10 events/minute
                good=7.0,         # > 7 events/minute
                satisfactory=5.0,   # > 5 events/minute
                needs_improvement=3.0,  # > 3 events/minute
                unit="events_per_minute"
            )
        }

    def _initialize_benchmarks(self) -> Dict[str, Any]:
        """Initialize benchmark data for comparison"""
        return {
            'industry_averages': {
                'response_time': 120.0,  # 2 minutes
                'success_rate': 0.85,    # 85%
                'collaboration_efficiency': 0.75,
                'handoff_efficiency': 0.80,
                'throughput': 6.0
            },
            'top_performers': {
                'response_time': 45.0,   # 45 seconds
                'success_rate': 0.95,    # 95%
                'collaboration_efficiency': 0.90,
                'handoff_efficiency': 0.95,
                'throughput': 12.0
            }
        }

    async def record_metric(self, metric: PerformanceMetric) -> None:
        """
        Record a performance metric

        Args:
            metric: Performance metric to record
        """
        try:
            # Add to history
            self.metrics_history[metric.metric_type].append(metric)

            # Update agent profile if applicable
            if metric.agent_id:
                await self._update_agent_profile(metric)

            # Keep only recent metrics (last 7 days)
            cutoff_time = datetime.utcnow() - timedelta(days=7)
            self.metrics_history[metric.metric_type] = [
                m for m in self.metrics_history[metric.metric_type]
                if m.timestamp > cutoff_time
            ]

            logger.debug(f"Recorded performance metric: {metric.metric_type.value} = {metric.value}")

        except Exception as e:
            logger.error(f"Error recording performance metric: {e}")

    async def _update_agent_profile(self, metric: PerformanceMetric) -> None:
        """Update agent performance profile"""
        agent_id = metric.agent_id
        if agent_id not in self.agent_profiles:
            self.agent_profiles[agent_id] = AgentPerformanceProfile(agent_id=agent_id)

        profile = self.agent_profiles[agent_id]

        # Add metric to profile
        if metric.metric_type not in profile.metrics:
            profile.metrics[metric.metric_type] = []
        profile.metrics[metric.metric_type].append(metric)

        # Keep only recent metrics (last 24 hours)
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        for metric_type in profile.metrics:
            profile.metrics[metric_type] = [
                m for m in profile.metrics[metric_type]
                if m.timestamp > cutoff_time
            ]

        # Recalculate profile
        await self._recalculate_agent_profile(agent_id)

    async def _recalculate_agent_profile(self, agent_id: str) -> None:
        """Recalculate agent performance profile"""
        if agent_id not in self.agent_profiles:
            return

        profile = self.agent_profiles[agent_id]

        # Calculate current score
        current_score = await self._calculate_agent_performance_score(agent_id)
        profile.current_score = current_score

        # Determine grade
        profile.grade = self._determine_performance_grade(current_score)

        # Analyze trends
        profile.trend = await self._analyze_performance_trend(agent_id)

        # Identify strengths and weaknesses
        strengths, weaknesses = await self._identify_strengths_weaknesses(agent_id)
        profile.strengths = strengths
        profile.weaknesses = weaknesses

        # Generate recommendations
        profile.recommendations = await self._generate_agent_recommendations(agent_id)

        profile.last_updated = datetime.utcnow()

    async def _calculate_agent_performance_score(self, agent_id: str) -> float:
        """Calculate overall performance score for an agent"""
        if agent_id not in self.agent_profiles:
            return 0.0

        profile = self.agent_profiles[agent_id]
        if not profile.metrics:
            return 0.0

        # Calculate scores for each metric type
        metric_scores = {}
        for metric_type, metrics in profile.metrics.items():
            if not metrics:
                continue

            # Get recent metrics
            recent_metrics = metrics[-10:]  # Last 10 metrics
            avg_value = np.mean([m.value for m in recent_metrics])

            # Normalize to 0-1 scale based on thresholds
            threshold = self.thresholds.get(metric_type)
            if threshold:
                if metric_type in [PerformanceMetricType.RESPONSE_TIME]:
                    # Lower is better
                    normalized_score = max(0, 1 - (avg_value / threshold.needs_improvement))
                else:
                    # Higher is better
                    normalized_score = min(1, avg_value / threshold.excellent)
            else:
                normalized_score = 0.5  # Default if no threshold

            metric_scores[metric_type] = normalized_score

        # Calculate weighted average
        weights = {
            PerformanceMetricType.SUCCESS_RATE: 0.3,
            PerformanceMetricType.RESPONSE_TIME: 0.25,
            PerformanceMetricType.COLLABORATION_EFFICIENCY: 0.2,
            PerformanceMetricType.HANDOFF_EFFICIENCY: 0.15,
            PerformanceMetricType.THROUGHPUT: 0.1
        }

        total_score = 0.0
        total_weight = 0.0

        for metric_type, score in metric_scores.items():
            weight = weights.get(metric_type, 0.1)
            total_score += score * weight
            total_weight += weight

        return total_score / max(total_weight, 1.0)

    def _determine_performance_grade(self, score: float) -> PerformanceGrade:
        """Determine performance grade from score"""
        if score >= 0.90:
            return PerformanceGrade.EXCELLENT
        elif score >= 0.80:
            return PerformanceGrade.GOOD
        elif score >= 0.70:
            return PerformanceGrade.SATISFACTORY
        elif score >= 0.60:
            return PerformanceGrade.NEEDS_IMPROVEMENT
        else:
            return PerformanceGrade.POOR

    async def _analyze_performance_trend(self, agent_id: str) -> PerformanceTrend:
        """Analyze performance trend for an agent"""
        if agent_id not in self.agent_profiles:
            return PerformanceTrend.STABLE

        profile = self.agent_profiles[agent_id]
        if not profile.metrics:
            return PerformanceTrend.STABLE

        # Calculate trend based on recent performance scores
        recent_scores = []
        time_window = timedelta(hours=6)

        # Calculate scores for different time windows
        for hours_ago in [6, 12, 18, 24]:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_ago)
            window_metrics = {}

            for metric_type, metrics in profile.metrics.items():
                window_metrics[metric_type] = [
                    m for m in metrics if m.timestamp > cutoff_time
                ]

            if window_metrics:
                # Calculate score for this window
                window_score = await self._calculate_window_score(window_metrics)
                recent_scores.append(window_score)

        if len(recent_scores) < 2:
            return PerformanceTrend.STABLE

        # Calculate trend
        score_diff = recent_scores[0] - recent_scores[-1]  # Recent vs older
        volatility = np.std(recent_scores)

        if volatility > 0.2:
            return PerformanceTrend.VOLATILE
        elif score_diff > 0.1:
            return PerformanceTrend.IMPROVING
        elif score_diff < -0.1:
            return PerformanceTrend.DECLINING
        else:
            return PerformanceTrend.STABLE

    async def _calculate_window_score(self, window_metrics: Dict[PerformanceMetricType, List[PerformanceMetric]]) -> float:
        """Calculate performance score for a time window"""
        metric_scores = {}

        for metric_type, metrics in window_metrics.items():
            if not metrics:
                continue

            avg_value = np.mean([m.value for m in metrics])
            threshold = self.thresholds.get(metric_type)

            if threshold:
                if metric_type in [PerformanceMetricType.RESPONSE_TIME]:
                    normalized_score = max(0, 1 - (avg_value / threshold.needs_improvement))
                else:
                    normalized_score = min(1, avg_value / threshold.excellent)
            else:
                normalized_score = 0.5

            metric_scores[metric_type] = normalized_score

        if not metric_scores:
            return 0.0

        return np.mean(list(metric_scores.values()))

    async def _identify_strengths_weaknesses(self, agent_id: str) -> Tuple[List[str], List[str]]:
        """Identify agent strengths and weaknesses"""
        if agent_id not in self.agent_profiles:
            return [], []

        profile = self.agent_profiles[agent_id]
        if not profile.metrics:
            return [], []

        strengths = []
        weaknesses = []

        for metric_type, metrics in profile.metrics.items():
            if not metrics:
                continue

            recent_metrics = metrics[-5:]  # Last 5 metrics
            avg_value = np.mean([m.value for m in recent_metrics])
            threshold = self.thresholds.get(metric_type)

            if threshold:
                if metric_type in [PerformanceMetricType.RESPONSE_TIME]:
                    if avg_value <= threshold.good:
                        strengths.append(f"Fast response time ({avg_value:.1f}s)")
                    elif avg_value > threshold.needs_improvement:
                        weaknesses.append(f"Slow response time ({avg_value:.1f}s)")
                else:
                    if avg_value >= threshold.good:
                        strengths.append(f"High {metric_type.value.replace('_', ' ')} ({avg_value:.2%})")
                    elif avg_value < threshold.needs_improvement:
                        weaknesses.append(f"Low {metric_type.value.replace('_', ' ')} ({avg_value:.2%})")

        return strengths, weaknesses

    async def _generate_agent_recommendations(self, agent_id: str) -> List[str]:
        """Generate recommendations for agent improvement"""
        if agent_id not in self.agent_profiles:
            return []

        profile = self.agent_profiles[agent_id]
        recommendations = []

        # Grade-based recommendations
        if profile.grade == PerformanceGrade.EXCELLENT:
            recommendations.append("Maintain current performance levels")
            recommendations.append("Consider mentoring other agents")
        elif profile.grade == PerformanceGrade.GOOD:
            recommendations.append("Focus on areas of improvement to reach excellence")
        elif profile.grade == PerformanceGrade.SATISFACTORY:
            recommendations.append("Address identified weaknesses to improve performance")
        elif profile.grade == PerformanceGrade.NEEDS_IMPROVEMENT:
            recommendations.append("Immediate attention needed in underperforming areas")
            recommendations.append("Consider additional training or support")
        else:  # POOR
            recommendations.append("Critical performance issues require immediate intervention")
            recommendations.append("Review agent configuration and capabilities")

        # Specific metric-based recommendations
        for metric_type, metrics in profile.metrics.items():
            if not metrics:
                continue

            recent_metrics = metrics[-5:]
            avg_value = np.mean([m.value for m in recent_metrics])
            threshold = self.thresholds.get(metric_type)

            if threshold and avg_value < threshold.needs_improvement:
                if metric_type == PerformanceMetricType.RESPONSE_TIME:
                    recommendations.append("Optimize processing pipelines to reduce response time")
                elif metric_type == PerformanceMetricType.SUCCESS_RATE:
                    recommendations.append("Improve error handling and validation")
                elif metric_type == PerformanceMetricType.COLLABORATION_EFFICIENCY:
                    recommendations.append("Enhance communication and coordination skills")
                elif metric_type == PerformanceMetricType.HANDOFF_EFFICIENCY:
                    recommendations.append("Improve context sharing and handoff processes")

        # Trend-based recommendations
        if profile.trend == PerformanceTrend.DECLINING:
            recommendations.append("Investigate recent performance decline")
        elif profile.trend == PerformanceTrend.VOLATILE:
            recommendations.append("Stabilize performance through consistent processes")

        return recommendations[:5]  # Return top 5 recommendations

    async def generate_system_performance_report(self,
                                               time_range: Optional[Tuple[datetime, datetime]] = None) -> SystemPerformanceReport:
        """
        Generate comprehensive system performance report

        Args:
            time_range: Time range for the report

        Returns:
            System performance report
        """
        if time_range is None:
            time_range = (datetime.utcnow() - timedelta(hours=24), datetime.utcnow())

        try:
            report = SystemPerformanceReport(
                report_id=str(uuid.uuid4()),
                time_range=time_range
            )

            # Calculate overall system score
            report.overall_score = await self._calculate_system_performance_score(time_range)

            # Get system metrics
            report.system_metrics = await self._calculate_system_metrics(time_range)

            # Identify bottlenecks
            report.bottlenecks = await self._identify_system_bottlenecks(time_range)

            # Find optimization opportunities
            report.optimization_opportunities = await self._identify_optimization_opportunities(time_range)

            # Generate system recommendations
            report.recommendations = await self._generate_system_recommendations(report)

            # Include agent profiles
            report.agent_profiles = self.agent_profiles.copy()

            return report

        except Exception as e:
            logger.error(f"Error generating system performance report: {e}")
            raise

    async def _calculate_system_performance_score(self, time_range: Tuple[datetime, datetime]) -> float:
        """Calculate overall system performance score"""
        start_time, end_time = time_range

        # Get metrics within time range
        system_metrics = {}
        for metric_type, metrics in self.metrics_history.items():
            range_metrics = [
                m for m in metrics
                if start_time <= m.timestamp <= end_time
            ]
            if range_metrics:
                system_metrics[metric_type] = range_metrics

        if not system_metrics:
            return 0.0

        # Calculate scores for each metric type
        metric_scores = {}
        for metric_type, metrics in system_metrics.items():
            avg_value = np.mean([m.value for m in metrics])
            threshold = self.thresholds.get(metric_type)

            if threshold:
                if metric_type in [PerformanceMetricType.RESPONSE_TIME]:
                    normalized_score = max(0, 1 - (avg_value / threshold.needs_improvement))
                else:
                    normalized_score = min(1, avg_value / threshold.excellent)
            else:
                normalized_score = 0.5

            metric_scores[metric_type] = normalized_score

        # Calculate weighted average
        weights = {
            PerformanceMetricType.SUCCESS_RATE: 0.3,
            PerformanceMetricType.RESPONSE_TIME: 0.25,
            PerformanceMetricType.COLLABORATION_EFFICIENCY: 0.2,
            PerformanceMetricType.HANDOFF_EFFICIENCY: 0.15,
            PerformanceMetricType.THROUGHPUT: 0.1
        }

        total_score = 0.0
        total_weight = 0.0

        for metric_type, score in metric_scores.items():
            weight = weights.get(metric_type, 0.1)
            total_score += score * weight
            total_weight += weight

        return total_score / max(total_weight, 1.0)

    async def _calculate_system_metrics(self, time_range: Tuple[datetime, datetime]) -> Dict[PerformanceMetricType, float]:
        """Calculate system-level metrics"""
        start_time, end_time = time_range
        system_metrics = {}

        for metric_type, metrics in self.metrics_history.items():
            range_metrics = [
                m for m in metrics
                if start_time <= m.timestamp <= end_time and m.agent_id is None  # System-level metrics
            ]

            if range_metrics:
                avg_value = np.mean([m.value for m in range_metrics])
                system_metrics[metric_type] = avg_value
            else:
                # Aggregate from agent metrics
                agent_metrics = [
                    m for m in metrics
                    if start_time <= m.timestamp <= end_time and m.agent_id is not None
                ]
                if agent_metrics:
                    avg_value = np.mean([m.value for m in agent_metrics])
                    system_metrics[metric_type] = avg_value

        return system_metrics

    async def _identify_system_bottlenecks(self, time_range: Tuple[datetime, datetime]) -> List[Dict[str, Any]]:
        """Identify system performance bottlenecks"""
        bottlenecks = []
        start_time, end_time = time_range

        # Check each metric type for bottlenecks
        for metric_type, threshold in self.thresholds.items():
            # Get recent metrics
            recent_metrics = [
                m for m in self.metrics_history[metric_type]
                if m.timestamp > start_time
            ]

            if not recent_metrics:
                continue

            avg_value = np.mean([m.value for m in recent_metrics])

            # Check if below threshold
            if metric_type in [PerformanceMetricType.RESPONSE_TIME]:
                if avg_value > threshold.needs_improvement:
                    bottlenecks.append({
                        'type': metric_type.value,
                        'severity': 'high' if avg_value > threshold.needs_improvement * 2 else 'medium',
                        'description': f"High response time: {avg_value:.1f}s",
                        'impact': 'User experience and system throughput',
                        'current_value': avg_value,
                        'threshold': threshold.needs_improvement
                    })
            else:
                if avg_value < threshold.needs_improvement:
                    bottlenecks.append({
                        'type': metric_type.value,
                        'severity': 'high' if avg_value < threshold.needs_improvement * 0.5 else 'medium',
                        'description': f"Low {metric_type.value}: {avg_value:.2%}",
                        'impact': 'Overall system effectiveness',
                        'current_value': avg_value,
                        'threshold': threshold.needs_improvement
                    })

        # Agent-specific bottlenecks
        for agent_id, profile in self.agent_profiles.items():
            if profile.current_score < 0.6:  # Poor performance
                bottlenecks.append({
                    'type': 'agent_performance',
                    'agent_id': agent_id,
                    'severity': 'high',
                    'description': f"Agent {agent_id} performing poorly ({profile.current_score:.2%})",
                    'impact': 'Specific agent tasks and collaborations',
                    'current_value': profile.current_score,
                    'threshold': 0.6
                })

        return bottlenecks

    async def _identify_optimization_opportunities(self, time_range: Tuple[datetime, datetime]) -> List[Dict[str, Any]]:
        """Identify optimization opportunities"""
        opportunities = []

        # Analyze metric patterns
        for metric_type, metrics in self.metrics_history.items():
            if len(metrics) < 10:
                continue

            # Check for improvement potential
            recent_metrics = metrics[-10:]
            avg_value = np.mean([m.value for m in recent_metrics])
            threshold = self.thresholds.get(metric_type)

            if threshold:
                if metric_type in [PerformanceMetricType.RESPONSE_TIME]:
                    potential = (avg_value - threshold.excellent) / avg_value
                    if potential > 0.2:  # 20% improvement potential
                        opportunities.append({
                            'type': 'response_time_optimization',
                            'potential_improvement': potential,
                            'description': f"Response time optimization potential: {potential:.1%}",
                            'estimated_impact': 'User experience and throughput',
                            'difficulty': 'medium'
                        })
                else:
                    potential = (threshold.excellent - avg_value) / threshold.excellent
                    if potential > 0.1:  # 10% improvement potential
                        opportunities.append({
                            'type': f'{metric_type.value}_optimization',
                            'potential_improvement': potential,
                            'description': f"{metric_type.value.replace('_', ' ').title()} optimization potential: {potential:.1%}",
                            'estimated_impact': 'Overall system effectiveness',
                            'difficulty': 'low' if potential < 0.3 else 'medium'
                        })

        # Collaboration optimization
        if PerformanceMetricType.COLLABORATION_EFFICIENCY in self.metrics_history:
            collab_metrics = self.metrics_history[PerformanceMetricType.COLLABORATION_EFFICIENCY]
            if collab_metrics:
                avg_collab = np.mean([m.value for m in collab_metrics[-10:]])
                if avg_collab < 0.8:
                    opportunities.append({
                        'type': 'collaboration_optimization',
                        'potential_improvement': 0.2,
                        'description': 'Collaboration process optimization',
                        'estimated_impact': 'Team effectiveness and task completion',
                        'difficulty': 'medium'
                    })

        return opportunities

    async def _generate_system_recommendations(self, report: SystemPerformanceReport) -> List[str]:
        """Generate system-level recommendations"""
        recommendations = []

        # Overall performance recommendations
        if report.overall_score >= 0.90:
            recommendations.append("System performance is excellent - maintain current configuration")
        elif report.overall_score >= 0.80:
            recommendations.append("System performance is good - focus on optimization opportunities")
        elif report.overall_score >= 0.70:
            recommendations.append("System performance is satisfactory - address bottlenecks")
        elif report.overall_score >= 0.60:
            recommendations.append("System performance needs improvement - prioritize bottlenecks")
        else:
            recommendations.append("System performance is poor - immediate intervention required")

        # Bottleneck-specific recommendations
        for bottleneck in report.bottlenecks:
            if bottleneck['type'] == 'response_time':
                recommendations.append("Optimize response times through caching and parallelization")
            elif bottleneck['type'] == 'success_rate':
                recommendations.append("Implement better error handling and validation")
            elif bottleneck['type'] == 'collaboration_efficiency':
                recommendations.append("Improve communication protocols and context sharing")

        # Optimization opportunity recommendations
        for opportunity in report.optimization_opportunities[:3]:  # Top 3 opportunities
            recommendations.append(f"Consider {opportunity['description']}")

        # Agent performance recommendations
        poor_performers = [
            agent_id for agent_id, profile in report.agent_profiles.items()
            if profile.grade == PerformanceGrade.POOR
        ]

        if poor_performers:
            recommendations.append(f"Review and optimize poor-performing agents: {', '.join(poor_performers[:3])}")

        return recommendations[:10]  # Return top 10 recommendations

    async def get_performance_comparison(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Get performance comparison between agents"""
        comparison = {
            'agents': {},
            'metrics': {},
            'rankings': {},
            'insights': []
        }

        # Get profiles for specified agents
        profiles = {
            agent_id: self.agent_profiles.get(agent_id)
            for agent_id in agent_ids
            if agent_id in self.agent_profiles
        }

        if not profiles:
            return comparison

        # Build comparison data
        for agent_id, profile in profiles.items():
            if profile:
                comparison['agents'][agent_id] = {
                    'current_score': profile.current_score,
                    'grade': profile.grade.value,
                    'trend': profile.trend.value,
                    'strengths': profile.strengths,
                    'weaknesses': profile.weaknesses
                }

        # Compare metrics
        metric_types = list(PerformanceMetricType)
        for metric_type in metric_types:
            metric_data = []
            for agent_id, profile in profiles.items():
                if profile and metric_type in profile.metrics:
                    recent_metrics = profile.metrics[metric_type][-5:]
                    avg_value = np.mean([m.value for m in recent_metrics])
                    metric_data.append({'agent_id': agent_id, 'value': avg_value})

            if metric_data:
                comparison['metrics'][metric_type.value] = metric_data

        # Generate rankings
        ranked_agents = sorted(
            [(agent_id, profile.current_score) for agent_id, profile in profiles.items() if profile],
            key=lambda x: x[1],
            reverse=True
        )

        comparison['rankings'] = {
            'by_score': [{'agent_id': agent_id, 'score': score, 'rank': i+1}
                        for i, (agent_id, score) in enumerate(ranked_agents)]
        }

        # Generate insights
        comparison['insights'] = await self._generate_comparison_insights(profiles)

        return comparison

    async def _generate_comparison_insights(self, profiles: Dict[str, AgentPerformanceProfile]) -> List[str]:
        """Generate insights from agent comparison"""
        insights = []

        if not profiles:
            return insights

        # Performance spread analysis
        scores = [profile.current_score for profile in profiles.values() if profile]
        if scores:
            min_score = min(scores)
            max_score = max(scores)
            score_spread = max_score - min_score

            if score_spread > 0.3:
                insights.append(f"Significant performance variance detected ({score_spread:.1%} spread)")
            elif score_spread < 0.1:
                insights.append("Consistent performance across agents")

        # Trend analysis
        improving_agents = [
            agent_id for agent_id, profile in profiles.items()
            if profile and profile.trend == PerformanceTrend.IMPROVING
        ]
        declining_agents = [
            agent_id for agent_id, profile in profiles.items()
            if profile and profile.trend == PerformanceTrend.DECLINING
        ]

        if improving_agents:
            insights.append(f"Improving agents: {', '.join(improving_agents[:3])}")
        if declining_agents:
            insights.append(f"Declining agents: {', '.join(declining_agents[:3])}")

        # Grade distribution
        grade_counts = Counter([profile.grade.value for profile in profiles.values() if profile])
        if grade_counts:
            insights.append(f"Grade distribution: {dict(grade_counts)}")

        return insights

    async def get_performance_forecast(self, agent_id: Optional[str] = None,
                                     forecast_days: int = 7) -> Dict[str, Any]:
        """Generate performance forecast"""
        forecast = {
            'forecast_id': str(uuid.uuid4()),
            'generated_at': datetime.utcnow().isoformat(),
            'forecast_period_days': forecast_days,
            'forecasts': {}
        }

        try:
            if agent_id:
                # Single agent forecast
                if agent_id in self.agent_profiles:
                    forecast['forecasts'][agent_id] = await self._forecast_agent_performance(
                        agent_id, forecast_days
                    )
            else:
                # System-wide forecast
                for agent_id, profile in self.agent_profiles.items():
                    forecast['forecasts'][agent_id] = await self._forecast_agent_performance(
                        agent_id, forecast_days
                    )

        except Exception as e:
            logger.error(f"Error generating performance forecast: {e}")

        return forecast

    async def _forecast_agent_performance(self, agent_id: str, forecast_days: int) -> Dict[str, Any]:
        """Forecast performance for a specific agent"""
        if agent_id not in self.agent_profiles:
            return {}

        profile = self.agent_profiles[agent_id]

        # Simple trend-based forecasting
        current_score = profile.current_score
        trend = profile.trend

        # Calculate trend adjustment
        trend_adjustment = 0.0
        if trend == PerformanceTrend.IMPROVING:
            trend_adjustment = 0.05  # 5% improvement
        elif trend == PerformanceTrend.DECLINING:
            trend_adjustment = -0.05  # 5% decline
        elif trend == PerformanceTrend.VOLATILE:
            trend_adjustment = 0.0  # No change due to volatility

        # Generate daily forecasts
        forecasts = []
        for day in range(1, forecast_days + 1):
            # Apply trend adjustment with diminishing effect
            daily_adjustment = trend_adjustment * (0.8 ** (day - 1))
            forecast_score = max(0.0, min(1.0, current_score + daily_adjustment))

            forecasts.append({
                'day': day,
                'forecasted_score': forecast_score,
                'confidence': max(0.5, 1.0 - (day * 0.05)),  # Decreasing confidence
                'projected_grade': self._determine_performance_grade(forecast_score).value
            })

        return {
            'agent_id': agent_id,
            'current_score': current_score,
            'current_grade': profile.grade.value,
            'trend': trend.value,
            'daily_forecasts': forecasts,
            'summary': {
                'projected_improvement': sum(f['forecasted_score'] for f in forecasts) / len(forecasts) - current_score,
                'risk_level': 'low' if trend != PerformanceTrend.DECLINING else 'medium'
            }
        }


# Factory function for creating performance analyzer
def create_conversation_performance_analyzer(config: Optional[Dict[str, Any]] = None) -> ConversationPerformanceAnalyzer:
    """Factory function to create conversation performance analyzer"""
    return ConversationPerformanceAnalyzer(config)