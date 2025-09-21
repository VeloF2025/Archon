"""
Team Performance Tracking System for Phase 9 Autonomous Development Teams

This module tracks and optimizes autonomous development team performance,
providing real-time analytics, trend analysis, and adaptive improvements
to maximize team effectiveness and project success rates.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from uuid import uuid4, UUID
import json
from pathlib import Path
import statistics

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class PerformanceMetric(str, Enum):
    """Types of performance metrics tracked."""
    TASK_COMPLETION_RATE = "task_completion_rate"
    AVERAGE_TASK_DURATION = "average_task_duration"
    QUALITY_SCORE = "quality_score"
    BUG_INTRODUCTION_RATE = "bug_introduction_rate"
    COLLABORATION_EFFICIENCY = "collaboration_efficiency"
    RESOURCE_UTILIZATION = "resource_utilization"
    CLIENT_SATISFACTION = "client_satisfaction"
    TECHNICAL_DEBT_ACCUMULATION = "technical_debt_accumulation"
    INNOVATION_INDEX = "innovation_index"
    KNOWLEDGE_SHARING_SCORE = "knowledge_sharing_score"


class PerformanceTrend(str, Enum):
    """Performance trend directions."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"


class TeamMaturity(str, Enum):
    """Team maturity levels."""
    FORMING = "forming"      # New team, establishing processes
    STORMING = "storming"    # Working through conflicts and processes
    NORMING = "norming"      # Establishing team norms and practices
    PERFORMING = "performing"  # High performance and efficiency
    TRANSFORMING = "transforming"  # Continuously improving and innovating


@dataclass
class PerformanceDataPoint:
    """Individual performance measurement."""
    timestamp: datetime = field(default_factory=datetime.now)
    metric: PerformanceMetric = PerformanceMetric.TASK_COMPLETION_RATE
    value: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    team_id: str = ""
    project_id: str = ""
    agent_id: Optional[str] = None


@dataclass
class TeamPerformanceProfile:
    """Comprehensive team performance profile."""
    team_id: str
    team_name: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    maturity_level: TeamMaturity = TeamMaturity.FORMING
    
    # Core metrics
    current_metrics: Dict[PerformanceMetric, float] = field(default_factory=dict)
    metric_trends: Dict[PerformanceMetric, PerformanceTrend] = field(default_factory=dict)
    historical_data: Dict[PerformanceMetric, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=100)))
    
    # Team composition and skills
    team_members: List[str] = field(default_factory=list)
    skill_distribution: Dict[str, float] = field(default_factory=dict)
    specialization_balance: float = 0.0  # 0=all generalists, 1=all specialists
    
    # Performance analytics
    performance_score: float = 0.0  # Overall performance score (0-10)
    consistency_score: float = 0.0  # How consistent performance is (0-10)
    improvement_rate: float = 0.0   # Rate of improvement over time
    bottlenecks: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    
    # Predictive analytics
    predicted_performance: Dict[str, float] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)
    optimization_opportunities: List[str] = field(default_factory=list)
    
    # Collaboration metrics
    communication_frequency: float = 0.0
    knowledge_sharing_events: int = 0
    cross_training_sessions: int = 0
    mentoring_relationships: int = 0
    
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceAnalytics:
    """Advanced analytics for team performance."""
    team_id: str
    analysis_period: Tuple[datetime, datetime]
    
    # Statistical analysis
    metric_correlations: Dict[Tuple[PerformanceMetric, PerformanceMetric], float] = field(default_factory=dict)
    performance_patterns: List[str] = field(default_factory=list)
    seasonal_trends: Dict[str, float] = field(default_factory=dict)
    
    # Predictive modeling
    performance_forecast: Dict[PerformanceMetric, List[float]] = field(default_factory=dict)
    forecast_confidence: Dict[PerformanceMetric, float] = field(default_factory=dict)
    
    # Benchmarking
    team_ranking: int = 0
    percentile_scores: Dict[PerformanceMetric, float] = field(default_factory=dict)
    best_practice_recommendations: List[str] = field(default_factory=list)
    
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceOptimization:
    """Performance optimization recommendation."""
    team_id: str
    optimization_type: str = ""
    description: str = ""
    expected_impact: float = 0.0  # Expected performance improvement (0-1)
    implementation_effort: str = "low"  # low, medium, high
    priority: int = 5  # 1-10
    success_probability: float = 0.0
    time_to_impact: int = 7  # days
    prerequisites: List[str] = field(default_factory=list)
    implementation_steps: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class TeamPerformanceTracker:
    """
    Comprehensive team performance tracking and optimization system.
    
    Monitors autonomous development teams in real-time, analyzes performance trends,
    and provides data-driven recommendations for continuous improvement.
    """
    
    def __init__(self, storage_path: str = "data/team_performance"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.team_profiles: Dict[str, TeamPerformanceProfile] = {}
        self.performance_data: List[PerformanceDataPoint] = []
        self.analytics: Dict[str, PerformanceAnalytics] = {}
        self.optimizations: Dict[str, List[PerformanceOptimization]] = defaultdict(list)
        
        # Performance tracking configuration
        self.metric_weights = {
            PerformanceMetric.TASK_COMPLETION_RATE: 0.25,
            PerformanceMetric.QUALITY_SCORE: 0.20,
            PerformanceMetric.COLLABORATION_EFFICIENCY: 0.15,
            PerformanceMetric.RESOURCE_UTILIZATION: 0.10,
            PerformanceMetric.CLIENT_SATISFACTION: 0.15,
            PerformanceMetric.INNOVATION_INDEX: 0.10,
            PerformanceMetric.KNOWLEDGE_SHARING_SCORE: 0.05
        }
        
        self.benchmark_thresholds = {
            PerformanceMetric.TASK_COMPLETION_RATE: {"excellent": 0.95, "good": 0.85, "poor": 0.70},
            PerformanceMetric.QUALITY_SCORE: {"excellent": 9.0, "good": 8.0, "poor": 7.0},
            PerformanceMetric.BUG_INTRODUCTION_RATE: {"excellent": 0.02, "good": 0.05, "poor": 0.10},
            PerformanceMetric.COLLABORATION_EFFICIENCY: {"excellent": 9.0, "good": 7.5, "poor": 6.0}
        }
        
        # Load existing data
        self._load_performance_data()
        
        # Start background analytics
        self._running = False
        self._analytics_task: Optional[asyncio.Task] = None
    
    def _load_performance_data(self):
        """Load existing performance data from storage."""
        try:
            profiles_file = self.storage_path / "team_profiles.json"
            if profiles_file.exists():
                with open(profiles_file, 'r') as f:
                    data = json.load(f)
                    for team_id, profile_data in data.items():
                        profile = TeamPerformanceProfile(**profile_data)
                        # Convert historical data back to deques
                        for metric, values in profile_data.get("historical_data", {}).items():
                            profile.historical_data[PerformanceMetric(metric)] = deque(
                                [PerformanceDataPoint(**v) for v in values], 
                                maxlen=100
                            )
                        self.team_profiles[team_id] = profile
                
                logger.info(f"Loaded {len(self.team_profiles)} team performance profiles")
        
        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
    
    def _save_performance_data(self):
        """Save performance data to storage."""
        try:
            profiles_file = self.storage_path / "team_profiles.json"
            
            # Serialize team profiles
            serializable_profiles = {}
            for team_id, profile in self.team_profiles.items():
                profile_dict = profile.__dict__.copy()
                
                # Convert enums to strings
                if isinstance(profile_dict.get("maturity_level"), TeamMaturity):
                    profile_dict["maturity_level"] = profile_dict["maturity_level"].value
                
                # Convert metrics and trends
                profile_dict["current_metrics"] = {
                    k.value if isinstance(k, PerformanceMetric) else k: v
                    for k, v in profile_dict["current_metrics"].items()
                }
                
                profile_dict["metric_trends"] = {
                    k.value if isinstance(k, PerformanceMetric) else k: v.value if isinstance(v, PerformanceTrend) else v
                    for k, v in profile_dict["metric_trends"].items()
                }
                
                # Convert historical data
                profile_dict["historical_data"] = {
                    k.value if isinstance(k, PerformanceMetric) else k: [
                        point.__dict__ for point in list(v)
                    ]
                    for k, v in profile_dict["historical_data"].items()
                }
                
                # Convert datetime objects
                for key, value in profile_dict.items():
                    if isinstance(value, datetime):
                        profile_dict[key] = value.isoformat()
                
                serializable_profiles[team_id] = profile_dict
            
            with open(profiles_file, 'w') as f:
                json.dump(serializable_profiles, f, indent=2, default=str)
            
            logger.info("Performance data saved successfully")
        
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    async def create_team_profile(self, team_id: str, team_name: str, team_members: List[str]) -> TeamPerformanceProfile:
        """Create a new team performance profile."""
        
        profile = TeamPerformanceProfile(
            team_id=team_id,
            team_name=team_name,
            team_members=team_members
        )
        
        # Initialize default metrics
        profile.current_metrics = {metric: 0.0 for metric in PerformanceMetric}
        profile.metric_trends = {metric: PerformanceTrend.STABLE for metric in PerformanceMetric}
        
        self.team_profiles[team_id] = profile
        
        logger.info(f"Created performance profile for team {team_name} ({team_id})")
        return profile
    
    async def record_performance_data(self, data_point: PerformanceDataPoint):
        """Record a new performance data point."""
        
        # Add to global data
        self.performance_data.append(data_point)
        
        # Update team profile
        if data_point.team_id in self.team_profiles:
            profile = self.team_profiles[data_point.team_id]
            
            # Update current metrics
            profile.current_metrics[data_point.metric] = data_point.value
            
            # Add to historical data
            profile.historical_data[data_point.metric].append(data_point)
            
            # Update trends
            await self._update_metric_trend(profile, data_point.metric)
            
            # Recalculate overall performance score
            await self._calculate_performance_score(profile)
            
            # Update last modified
            profile.last_updated = datetime.now()
            
            logger.debug(f"Recorded {data_point.metric.value}={data_point.value} for team {data_point.team_id}")
    
    async def _update_metric_trend(self, profile: TeamPerformanceProfile, metric: PerformanceMetric):
        """Update trend analysis for a specific metric."""
        
        historical_values = [point.value for point in profile.historical_data[metric]]
        
        if len(historical_values) < 3:
            profile.metric_trends[metric] = PerformanceTrend.STABLE
            return
        
        # Calculate trend using linear regression
        x = np.arange(len(historical_values))
        slope, _, r_value, _, _ = stats.linregress(x, historical_values)
        
        # Determine trend based on slope and correlation
        if abs(r_value) < 0.3:  # Low correlation = volatile
            trend = PerformanceTrend.VOLATILE
        elif slope > 0.01:  # Positive slope = improving
            trend = PerformanceTrend.IMPROVING
        elif slope < -0.01:  # Negative slope = declining
            trend = PerformanceTrend.DECLINING
        else:  # Small slope = stable
            trend = PerformanceTrend.STABLE
        
        profile.metric_trends[metric] = trend
    
    async def _calculate_performance_score(self, profile: TeamPerformanceProfile):
        """Calculate overall performance score for a team."""
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in self.metric_weights.items():
            if metric in profile.current_metrics:
                # Normalize metric value to 0-10 scale
                normalized_value = self._normalize_metric_value(metric, profile.current_metrics[metric])
                weighted_score += normalized_value * weight
                total_weight += weight
        
        profile.performance_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Calculate consistency score based on metric volatility
        consistency_scores = []
        for metric, historical_data in profile.historical_data.items():
            if len(historical_data) >= 5:
                values = [point.value for point in historical_data]
                normalized_values = [self._normalize_metric_value(metric, v) for v in values]
                std_dev = statistics.stdev(normalized_values)
                consistency = max(0, 10 - std_dev * 2)  # Lower std dev = higher consistency
                consistency_scores.append(consistency)
        
        profile.consistency_score = statistics.mean(consistency_scores) if consistency_scores else 5.0
    
    def _normalize_metric_value(self, metric: PerformanceMetric, value: float) -> float:
        """Normalize a metric value to 0-10 scale."""
        
        # Get thresholds for this metric
        thresholds = self.benchmark_thresholds.get(metric, {"excellent": 10, "good": 7, "poor": 4})
        
        if metric == PerformanceMetric.BUG_INTRODUCTION_RATE:
            # Lower is better for bug rate
            if value <= thresholds["excellent"]:
                return 10.0
            elif value <= thresholds["good"]:
                return 7.5
            elif value <= thresholds["poor"]:
                return 5.0
            else:
                return max(0, 5 - (value - thresholds["poor"]) * 10)
        else:
            # Higher is better for most metrics
            if value >= thresholds["excellent"]:
                return 10.0
            elif value >= thresholds["good"]:
                return 7.5
            elif value >= thresholds["poor"]:
                return 5.0
            else:
                return max(0, value / thresholds["poor"] * 5)
    
    async def analyze_team_performance(self, team_id: str, days_back: int = 30) -> Optional[PerformanceAnalytics]:
        """Generate comprehensive performance analytics for a team."""
        
        if team_id not in self.team_profiles:
            logger.error(f"Team {team_id} not found")
            return None
        
        profile = self.team_profiles[team_id]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        analytics = PerformanceAnalytics(
            team_id=team_id,
            analysis_period=(start_date, end_date)
        )
        
        # Calculate metric correlations
        await self._calculate_metric_correlations(analytics, profile)
        
        # Identify performance patterns
        await self._identify_performance_patterns(analytics, profile)
        
        # Generate performance forecast
        await self._generate_performance_forecast(analytics, profile)
        
        # Calculate team ranking and percentiles
        await self._calculate_team_ranking(analytics)
        
        # Generate recommendations
        await self._generate_best_practices(analytics, profile)
        
        self.analytics[team_id] = analytics
        return analytics
    
    async def _calculate_metric_correlations(self, analytics: PerformanceAnalytics, profile: TeamPerformanceProfile):
        """Calculate correlations between different performance metrics."""
        
        correlations = {}
        metrics_with_data = {}
        
        # Collect data for metrics with sufficient history
        for metric, historical_data in profile.historical_data.items():
            if len(historical_data) >= 10:
                metrics_with_data[metric] = [point.value for point in historical_data]
        
        # Calculate pairwise correlations
        for metric1 in metrics_with_data:
            for metric2 in metrics_with_data:
                if metric1 != metric2:
                    try:
                        correlation, _ = stats.pearsonr(metrics_with_data[metric1], metrics_with_data[metric2])
                        if abs(correlation) > 0.3:  # Only store significant correlations
                            correlations[(metric1, metric2)] = correlation
                    except Exception as e:
                        logger.debug(f"Could not calculate correlation between {metric1} and {metric2}: {e}")
        
        analytics.metric_correlations = correlations
    
    async def _identify_performance_patterns(self, analytics: PerformanceAnalytics, profile: TeamPerformanceProfile):
        """Identify patterns in team performance data."""
        
        patterns = []
        
        # Analyze trends
        improving_metrics = [m.value for m, t in profile.metric_trends.items() if t == PerformanceTrend.IMPROVING]
        declining_metrics = [m.value for m, t in profile.metric_trends.items() if t == PerformanceTrend.DECLINING]
        
        if len(improving_metrics) > len(declining_metrics):
            patterns.append(f"Overall improvement trend with {len(improving_metrics)} metrics improving")
        elif len(declining_metrics) > len(improving_metrics):
            patterns.append(f"Performance concerns with {len(declining_metrics)} metrics declining")
        
        # Analyze maturity progression
        if profile.maturity_level == TeamMaturity.FORMING and profile.performance_score > 6.0:
            patterns.append("Rapid team formation with strong early performance")
        elif profile.maturity_level == TeamMaturity.PERFORMING and profile.consistency_score > 8.0:
            patterns.append("High-performing team with consistent delivery")
        
        # Analyze correlation patterns
        strong_positive_correlations = [
            f"{pair[0].value} ↔ {pair[1].value}"
            for pair, corr in analytics.metric_correlations.items()
            if corr > 0.7
        ]
        
        if strong_positive_correlations:
            patterns.append(f"Strong positive correlations: {', '.join(strong_positive_correlations[:3])}")
        
        analytics.performance_patterns = patterns
    
    async def _generate_performance_forecast(self, analytics: PerformanceAnalytics, profile: TeamPerformanceProfile):
        """Generate performance forecasts using trend analysis."""
        
        forecast_days = 14  # 2-week forecast
        
        for metric, historical_data in profile.historical_data.items():
            if len(historical_data) < 5:
                continue
            
            values = [point.value for point in historical_data]
            x = np.arange(len(values))
            
            try:
                # Fit linear trend
                slope, intercept, r_value, _, _ = stats.linregress(x, values)
                
                # Generate forecast
                future_x = np.arange(len(values), len(values) + forecast_days)
                forecast_values = [slope * x + intercept for x in future_x]
                
                analytics.performance_forecast[metric] = forecast_values
                analytics.forecast_confidence[metric] = abs(r_value)
                
            except Exception as e:
                logger.debug(f"Could not generate forecast for {metric}: {e}")
    
    async def _calculate_team_ranking(self, analytics: PerformanceAnalytics):
        """Calculate team ranking against other teams."""
        
        if len(self.team_profiles) < 2:
            analytics.team_ranking = 1
            return
        
        team_scores = [(team_id, profile.performance_score) 
                      for team_id, profile in self.team_profiles.items()]
        team_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Find ranking
        for rank, (team_id, _) in enumerate(team_scores, 1):
            if team_id == analytics.team_id:
                analytics.team_ranking = rank
                break
        
        # Calculate percentiles for each metric
        target_profile = self.team_profiles[analytics.team_id]
        
        for metric in PerformanceMetric:
            all_values = []
            for profile in self.team_profiles.values():
                if metric in profile.current_metrics:
                    all_values.append(profile.current_metrics[metric])
            
            if all_values and metric in target_profile.current_metrics:
                target_value = target_profile.current_metrics[metric]
                percentile = stats.percentileofscore(all_values, target_value)
                analytics.percentile_scores[metric] = percentile
    
    async def _generate_best_practices(self, analytics: PerformanceAnalytics, profile: TeamPerformanceProfile):
        """Generate best practice recommendations based on analytics."""
        
        recommendations = []
        
        # Based on performance score
        if profile.performance_score < 6.0:
            recommendations.append("Focus on improving core metrics: task completion and quality scores")
        elif profile.performance_score > 8.5:
            recommendations.append("Maintain excellence and consider mentoring other teams")
        
        # Based on consistency
        if profile.consistency_score < 6.0:
            recommendations.append("Implement standardized processes to improve consistency")
        
        # Based on trends
        declining_count = sum(1 for trend in profile.metric_trends.values() if trend == PerformanceTrend.DECLINING)
        if declining_count >= 3:
            recommendations.append("Address declining metrics through team retrospectives and process improvements")
        
        # Based on maturity level
        if profile.maturity_level == TeamMaturity.FORMING:
            recommendations.append("Establish clear team processes and communication channels")
        elif profile.maturity_level == TeamMaturity.STORMING:
            recommendations.append("Focus on conflict resolution and role clarification")
        
        # Based on correlations
        quality_collaboration_corr = analytics.metric_correlations.get(
            (PerformanceMetric.QUALITY_SCORE, PerformanceMetric.COLLABORATION_EFFICIENCY), 0
        )
        if quality_collaboration_corr > 0.7:
            recommendations.append("Strong quality-collaboration link detected. Maintain collaborative practices.")
        
        analytics.best_practice_recommendations = recommendations
    
    async def generate_optimization_recommendations(self, team_id: str) -> List[PerformanceOptimization]:
        """Generate specific optimization recommendations for a team."""
        
        if team_id not in self.team_profiles:
            return []
        
        profile = self.team_profiles[team_id]
        optimizations = []
        
        # Analyze bottlenecks and generate optimizations
        optimizations.extend(await self._generate_skill_optimizations(profile))
        optimizations.extend(await self._generate_process_optimizations(profile))
        optimizations.extend(await self._generate_collaboration_optimizations(profile))
        optimizations.extend(await self._generate_quality_optimizations(profile))
        
        # Sort by expected impact and priority
        optimizations.sort(key=lambda x: (x.expected_impact * x.priority), reverse=True)
        
        # Store optimizations
        self.optimizations[team_id] = optimizations
        
        return optimizations
    
    async def _generate_skill_optimizations(self, profile: TeamPerformanceProfile) -> List[PerformanceOptimization]:
        """Generate skill-based optimization recommendations."""
        
        optimizations = []
        
        # Check skill balance
        if profile.specialization_balance < 0.3:  # Too many generalists
            optimization = PerformanceOptimization(
                team_id=profile.team_id,
                optimization_type="skill_specialization",
                description="Increase team specialization by developing deeper expertise in key areas",
                expected_impact=0.15,
                implementation_effort="medium",
                priority=7,
                success_probability=0.8,
                time_to_impact=30,
                implementation_steps=[
                    "Identify critical skill gaps",
                    "Assign specialists to focus areas",
                    "Provide targeted training",
                    "Establish knowledge sharing sessions"
                ],
                success_metrics=["increased_specialization_balance", "improved_quality_scores"]
            )
            optimizations.append(optimization)
        
        elif profile.specialization_balance > 0.8:  # Too many specialists
            optimization = PerformanceOptimization(
                team_id=profile.team_id,
                optimization_type="skill_cross_training",
                description="Improve team flexibility through cross-training initiatives",
                expected_impact=0.12,
                implementation_effort="medium",
                priority=6,
                success_probability=0.75,
                time_to_impact=45,
                implementation_steps=[
                    "Identify cross-training opportunities",
                    "Implement pair programming sessions",
                    "Rotate team members across different areas",
                    "Document knowledge sharing"
                ],
                success_metrics=["reduced_specialization_balance", "improved_collaboration_efficiency"]
            )
            optimizations.append(optimization)
        
        return optimizations
    
    async def _generate_process_optimizations(self, profile: TeamPerformanceProfile) -> List[PerformanceOptimization]:
        """Generate process-based optimization recommendations."""
        
        optimizations = []
        
        # Check task completion rate
        completion_rate = profile.current_metrics.get(PerformanceMetric.TASK_COMPLETION_RATE, 0.0)
        
        if completion_rate < 0.8:
            optimization = PerformanceOptimization(
                team_id=profile.team_id,
                optimization_type="process_improvement",
                description="Implement structured task management and tracking processes",
                expected_impact=0.20,
                implementation_effort="low",
                priority=9,
                success_probability=0.9,
                time_to_impact=14,
                implementation_steps=[
                    "Implement daily standups",
                    "Use task tracking tools",
                    "Define clear acceptance criteria",
                    "Establish WIP limits"
                ],
                success_metrics=["improved_task_completion_rate", "reduced_task_duration"]
            )
            optimizations.append(optimization)
        
        # Check technical debt
        debt_score = profile.current_metrics.get(PerformanceMetric.TECHNICAL_DEBT_ACCUMULATION, 0.0)
        
        if debt_score > 0.3:
            optimization = PerformanceOptimization(
                team_id=profile.team_id,
                optimization_type="technical_debt_reduction",
                description="Implement regular technical debt reduction practices",
                expected_impact=0.18,
                implementation_effort="medium",
                priority=7,
                success_probability=0.8,
                time_to_impact=21,
                implementation_steps=[
                    "Conduct technical debt assessment",
                    "Allocate 20% time for refactoring",
                    "Implement code quality gates",
                    "Regular code reviews"
                ],
                success_metrics=["reduced_technical_debt", "improved_quality_score"]
            )
            optimizations.append(optimization)
        
        return optimizations
    
    async def _generate_collaboration_optimizations(self, profile: TeamPerformanceProfile) -> List[PerformanceOptimization]:
        """Generate collaboration-based optimization recommendations."""
        
        optimizations = []
        
        collaboration_score = profile.current_metrics.get(PerformanceMetric.COLLABORATION_EFFICIENCY, 0.0)
        
        if collaboration_score < 7.0:
            optimization = PerformanceOptimization(
                team_id=profile.team_id,
                optimization_type="collaboration_improvement",
                description="Enhance team collaboration through better communication tools and practices",
                expected_impact=0.16,
                implementation_effort="low",
                priority=8,
                success_probability=0.85,
                time_to_impact=10,
                implementation_steps=[
                    "Implement collaborative development tools",
                    "Establish regular knowledge sharing sessions",
                    "Create team documentation standards",
                    "Set up mentoring pairs"
                ],
                success_metrics=["improved_collaboration_efficiency", "increased_knowledge_sharing_score"]
            )
            optimizations.append(optimization)
        
        return optimizations
    
    async def _generate_quality_optimizations(self, profile: TeamPerformanceProfile) -> List[PerformanceOptimization]:
        """Generate quality-based optimization recommendations."""
        
        optimizations = []
        
        quality_score = profile.current_metrics.get(PerformanceMetric.QUALITY_SCORE, 0.0)
        bug_rate = profile.current_metrics.get(PerformanceMetric.BUG_INTRODUCTION_RATE, 0.0)
        
        if quality_score < 8.0 or bug_rate > 0.05:
            optimization = PerformanceOptimization(
                team_id=profile.team_id,
                optimization_type="quality_improvement",
                description="Implement comprehensive quality assurance practices",
                expected_impact=0.22,
                implementation_effort="medium",
                priority=9,
                success_probability=0.9,
                time_to_impact=21,
                implementation_steps=[
                    "Implement automated testing pipeline",
                    "Establish code review requirements",
                    "Add quality metrics dashboard",
                    "Regular quality retrospectives"
                ],
                success_metrics=["improved_quality_score", "reduced_bug_introduction_rate"]
            )
            optimizations.append(optimization)
        
        return optimizations
    
    async def start_monitoring(self):
        """Start continuous performance monitoring."""
        
        if self._running:
            return
        
        self._running = True
        self._analytics_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started team performance monitoring")
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        
        if not self._running:
            return
        
        self._running = False
        
        if self._analytics_task:
            self._analytics_task.cancel()
            try:
                await self._analytics_task
            except asyncio.CancelledError:
                pass
        
        self._save_performance_data()
        logger.info("Stopped team performance monitoring")
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop."""
        
        while self._running:
            try:
                # Update all team analytics
                for team_id in self.team_profiles.keys():
                    await self.analyze_team_performance(team_id)
                
                # Save data periodically
                self._save_performance_data()
                
                # Sleep until next monitoring cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def get_team_dashboard_data(self, team_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive dashboard data for a team."""
        
        if team_id not in self.team_profiles:
            return None
        
        profile = self.team_profiles[team_id]
        analytics = self.analytics.get(team_id)
        optimizations = self.optimizations.get(team_id, [])
        
        return {
            "team_info": {
                "id": profile.team_id,
                "name": profile.team_name,
                "maturity_level": profile.maturity_level.value,
                "team_size": len(profile.team_members),
                "created_at": profile.created_at.isoformat()
            },
            "performance_scores": {
                "overall": round(profile.performance_score, 2),
                "consistency": round(profile.consistency_score, 2),
                "improvement_rate": round(profile.improvement_rate, 2)
            },
            "current_metrics": {
                metric.value: round(value, 3)
                for metric, value in profile.current_metrics.items()
            },
            "metric_trends": {
                metric.value: trend.value
                for metric, trend in profile.metric_trends.items()
            },
            "analytics": {
                "ranking": analytics.team_ranking if analytics else 0,
                "patterns": analytics.performance_patterns if analytics else [],
                "correlations": len(analytics.metric_correlations) if analytics else 0
            },
            "optimizations": [
                {
                    "type": opt.optimization_type,
                    "description": opt.description,
                    "impact": round(opt.expected_impact, 2),
                    "priority": opt.priority,
                    "effort": opt.implementation_effort
                }
                for opt in optimizations[:5]  # Top 5 recommendations
            ],
            "strengths": profile.strengths,
            "bottlenecks": profile.bottlenecks,
            "last_updated": profile.last_updated.isoformat()
        }


async def main():
    """Test the team performance tracker."""
    
    logging.basicConfig(level=logging.INFO)
    
    tracker = TeamPerformanceTracker()
    
    # Create a test team
    team_profile = await tracker.create_team_profile(
        team_id="team-alpha",
        team_name="Alpha Development Team",
        team_members=["agent-1", "agent-2", "agent-3", "agent-4"]
    )
    
    print(f"Created team profile: {team_profile.team_name}")
    
    # Simulate some performance data
    test_data = [
        PerformanceDataPoint(
            metric=PerformanceMetric.TASK_COMPLETION_RATE,
            value=0.85,
            team_id="team-alpha"
        ),
        PerformanceDataPoint(
            metric=PerformanceMetric.QUALITY_SCORE,
            value=8.2,
            team_id="team-alpha"
        ),
        PerformanceDataPoint(
            metric=PerformanceMetric.COLLABORATION_EFFICIENCY,
            value=7.5,
            team_id="team-alpha"
        )
    ]
    
    for data_point in test_data:
        await tracker.record_performance_data(data_point)
    
    # Analyze performance
    analytics = await tracker.analyze_team_performance("team-alpha")
    print(f"Performance analytics generated: {len(analytics.performance_patterns)} patterns")
    
    # Generate optimizations
    optimizations = await tracker.generate_optimization_recommendations("team-alpha")
    print(f"Generated {len(optimizations)} optimization recommendations")
    
    # Get dashboard data
    dashboard = await tracker.get_team_dashboard_data("team-alpha")
    print(f"Dashboard data: Performance score {dashboard['performance_scores']['overall']}")


if __name__ == "__main__":
    asyncio.run(main())