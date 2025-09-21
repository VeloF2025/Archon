"""
Performance Analyzer
Real-time performance analysis and optimization recommendations
"""

import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import deque, defaultdict
import logging
import statistics
import json

logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """Types of performance metrics"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DATABASE_LATENCY = "database_latency"
    CACHE_HIT_RATE = "cache_hit_rate"
    QUEUE_LENGTH = "queue_length"
    CONCURRENCY = "concurrency"
    SATURATION = "saturation"


class PerformanceStatus(Enum):
    """Performance status levels"""
    OPTIMAL = "optimal"
    GOOD = "good"
    DEGRADED = "degraded"
    POOR = "poor"
    CRITICAL = "critical"


class OptimizationType(Enum):
    """Types of optimization recommendations"""
    CACHING = "caching"
    SCALING = "scaling"
    INDEXING = "indexing"
    QUERY_OPTIMIZATION = "query_optimization"
    CODE_OPTIMIZATION = "code_optimization"
    RESOURCE_ALLOCATION = "resource_allocation"
    CONFIGURATION = "configuration"
    ARCHITECTURE = "architecture"


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration"""
    metric: PerformanceMetric
    optimal: float
    good: float
    degraded: float
    poor: float
    critical: float
    unit: str = ""


@dataclass
class PerformanceProfile:
    """Performance profile for a component"""
    component_name: str
    component_type: str
    baseline_metrics: Dict[str, float]
    thresholds: List[PerformanceThreshold]
    sla_targets: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceAnomaly:
    """Detected performance anomaly"""
    anomaly_id: str
    component: str
    metric: PerformanceMetric
    severity: str
    detected_at: datetime
    current_value: float
    expected_value: float
    deviation_percent: float
    duration_seconds: int
    probable_causes: List[str]
    recommended_actions: List[str]


@dataclass
class PerformanceReport:
    """Performance analysis report"""
    report_id: str
    period_start: datetime
    period_end: datetime
    overall_status: PerformanceStatus
    component_statuses: Dict[str, PerformanceStatus]
    key_metrics: Dict[str, Dict[str, float]]
    anomalies: List[PerformanceAnomaly]
    optimizations: List[Dict[str, Any]]
    trends: Dict[str, str]
    predictions: Dict[str, Any]
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""
    recommendation_id: str
    type: OptimizationType
    priority: str  # high, medium, low
    component: str
    title: str
    description: str
    expected_improvement: Dict[str, float]
    implementation_effort: str  # low, medium, high
    risk_level: str  # low, medium, high
    steps: List[str]
    estimated_cost: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)


class PerformanceBuffer:
    """Circular buffer for performance metrics"""
    
    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self.stats_cache: Dict[str, Any] = {}
        self.cache_timestamp: Optional[datetime] = None
        
    def add(self, value: float, timestamp: datetime = None) -> None:
        """Add value to buffer"""
        if timestamp is None:
            timestamp = datetime.now()
        self.buffer.append((timestamp, value))
        self.stats_cache = {}  # Invalidate cache
        
    def get_recent(self, seconds: int) -> List[Tuple[datetime, float]]:
        """Get recent values within specified seconds"""
        cutoff = datetime.now() - timedelta(seconds=seconds)
        return [(ts, val) for ts, val in self.buffer if ts > cutoff]
    
    def calculate_stats(self) -> Dict[str, float]:
        """Calculate statistics for buffer"""
        # Check cache
        if self.stats_cache and self.cache_timestamp:
            if (datetime.now() - self.cache_timestamp).seconds < 10:
                return self.stats_cache
        
        if not self.buffer:
            return {}
        
        values = [val for _, val in self.buffer]
        
        stats = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "p50": np.percentile(values, 50),
            "p90": np.percentile(values, 90),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99)
        }
        
        # Cache results
        self.stats_cache = stats
        self.cache_timestamp = datetime.now()
        
        return stats


class PerformanceAnalyzer:
    """Advanced performance analysis and optimization system"""
    
    def __init__(self):
        # Performance data storage
        self.metrics_buffers: Dict[str, Dict[str, PerformanceBuffer]] = defaultdict(
            lambda: defaultdict(lambda: PerformanceBuffer(max_size=1000))
        )
        
        # Performance profiles
        self.profiles: Dict[str, PerformanceProfile] = {}
        
        # Anomaly detection
        self.anomalies: List[PerformanceAnomaly] = []
        self.anomaly_detectors: Dict[str, callable] = {}
        
        # Optimization recommendations
        self.recommendations: List[OptimizationRecommendation] = []
        
        # Default thresholds
        self.default_thresholds = self._initialize_default_thresholds()
        
        # Analysis state
        self.analysis_enabled = True
        self.anomaly_detection_enabled = True
        self.optimization_suggestions_enabled = True
        
        # Background tasks
        self._analysis_task: Optional[asyncio.Task] = None
        self._anomaly_detection_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "total_metrics_analyzed": 0,
            "anomalies_detected": 0,
            "recommendations_generated": 0,
            "reports_generated": 0,
            "last_analysis": None
        }
        
        # Start background tasks
        asyncio.create_task(self._start_background_tasks())
    
    def _initialize_default_thresholds(self) -> Dict[PerformanceMetric, PerformanceThreshold]:
        """Initialize default performance thresholds"""
        return {
            PerformanceMetric.RESPONSE_TIME: PerformanceThreshold(
                metric=PerformanceMetric.RESPONSE_TIME,
                optimal=100, good=200, degraded=500, poor=1000, critical=2000,
                unit="ms"
            ),
            PerformanceMetric.THROUGHPUT: PerformanceThreshold(
                metric=PerformanceMetric.THROUGHPUT,
                optimal=1000, good=500, degraded=200, poor=100, critical=50,
                unit="req/s"
            ),
            PerformanceMetric.ERROR_RATE: PerformanceThreshold(
                metric=PerformanceMetric.ERROR_RATE,
                optimal=0.1, good=1, degraded=2, poor=5, critical=10,
                unit="%"
            ),
            PerformanceMetric.CPU_USAGE: PerformanceThreshold(
                metric=PerformanceMetric.CPU_USAGE,
                optimal=40, good=60, degraded=75, poor=85, critical=95,
                unit="%"
            ),
            PerformanceMetric.MEMORY_USAGE: PerformanceThreshold(
                metric=PerformanceMetric.MEMORY_USAGE,
                optimal=50, good=70, degraded=80, poor=90, critical=95,
                unit="%"
            ),
            PerformanceMetric.DATABASE_LATENCY: PerformanceThreshold(
                metric=PerformanceMetric.DATABASE_LATENCY,
                optimal=10, good=25, degraded=50, poor=100, critical=200,
                unit="ms"
            ),
            PerformanceMetric.CACHE_HIT_RATE: PerformanceThreshold(
                metric=PerformanceMetric.CACHE_HIT_RATE,
                optimal=95, good=85, degraded=70, poor=50, critical=30,
                unit="%"
            ),
            PerformanceMetric.QUEUE_LENGTH: PerformanceThreshold(
                metric=PerformanceMetric.QUEUE_LENGTH,
                optimal=10, good=50, degraded=100, poor=500, critical=1000,
                unit="items"
            )
        }
    
    async def _start_background_tasks(self) -> None:
        """Start background analysis tasks"""
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        self._anomaly_detection_task = asyncio.create_task(self._anomaly_detection_loop())
    
    async def _analysis_loop(self) -> None:
        """Background performance analysis loop"""
        while self.analysis_enabled:
            try:
                await asyncio.sleep(60)  # Analyze every minute
                await self._analyze_all_components()
                self.stats["last_analysis"] = datetime.now()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(10)
    
    async def _anomaly_detection_loop(self) -> None:
        """Background anomaly detection loop"""
        while self.anomaly_detection_enabled:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._detect_anomalies()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in anomaly detection loop: {e}")
                await asyncio.sleep(10)
    
    async def record_metric(self, component: str, metric: PerformanceMetric,
                          value: float, timestamp: datetime = None) -> None:
        """Record a performance metric"""
        try:
            self.metrics_buffers[component][metric.value].add(value, timestamp)
            self.stats["total_metrics_analyzed"] += 1
            
            # Check for immediate anomalies
            if self.anomaly_detection_enabled:
                await self._check_immediate_anomaly(component, metric, value)
                
        except Exception as e:
            logger.error(f"Error recording metric: {e}")
    
    async def record_api_performance(self, endpoint: str, method: str,
                                   response_time_ms: float, status_code: int,
                                   timestamp: datetime = None) -> None:
        """Record API endpoint performance"""
        component = f"api_{endpoint}_{method}"
        
        await self.record_metric(component, PerformanceMetric.RESPONSE_TIME, response_time_ms, timestamp)
        
        # Record error if status indicates failure
        if status_code >= 400:
            error_rate = await self._calculate_error_rate(component)
            await self.record_metric(component, PerformanceMetric.ERROR_RATE, error_rate, timestamp)
    
    async def record_database_performance(self, query_type: str, latency_ms: float,
                                        rows_affected: int = 0,
                                        timestamp: datetime = None) -> None:
        """Record database performance"""
        component = f"database_{query_type}"
        
        await self.record_metric(component, PerformanceMetric.DATABASE_LATENCY, latency_ms, timestamp)
        
        # Record throughput if applicable
        if rows_affected > 0:
            throughput = rows_affected / (latency_ms / 1000) if latency_ms > 0 else 0
            await self.record_metric(component, PerformanceMetric.THROUGHPUT, throughput, timestamp)
    
    async def record_system_performance(self, cpu_percent: float, memory_percent: float,
                                      timestamp: datetime = None) -> None:
        """Record system performance metrics"""
        await self.record_metric("system", PerformanceMetric.CPU_USAGE, cpu_percent, timestamp)
        await self.record_metric("system", PerformanceMetric.MEMORY_USAGE, memory_percent, timestamp)
    
    def create_performance_profile(self, component_name: str, component_type: str,
                                  thresholds: List[PerformanceThreshold] = None,
                                  sla_targets: Dict[str, float] = None) -> PerformanceProfile:
        """Create performance profile for a component"""
        try:
            # Calculate baseline metrics from existing data
            baseline_metrics = {}
            if component_name in self.metrics_buffers:
                for metric_name, buffer in self.metrics_buffers[component_name].items():
                    stats = buffer.calculate_stats()
                    if stats:
                        baseline_metrics[metric_name] = stats.get("mean", 0)
            
            profile = PerformanceProfile(
                component_name=component_name,
                component_type=component_type,
                baseline_metrics=baseline_metrics,
                thresholds=thresholds or list(self.default_thresholds.values()),
                sla_targets=sla_targets or {}
            )
            
            self.profiles[component_name] = profile
            logger.info(f"Created performance profile for {component_name}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Error creating performance profile: {e}")
            raise
    
    async def _analyze_all_components(self) -> None:
        """Analyze performance of all components"""
        for component_name, metrics in self.metrics_buffers.items():
            try:
                await self._analyze_component(component_name, metrics)
            except Exception as e:
                logger.error(f"Error analyzing component {component_name}: {e}")
    
    async def _analyze_component(self, component_name: str,
                               metrics: Dict[str, PerformanceBuffer]) -> Dict[str, Any]:
        """Analyze performance of a single component"""
        analysis = {
            "component": component_name,
            "timestamp": datetime.now(),
            "metrics": {},
            "status": PerformanceStatus.OPTIMAL,
            "issues": [],
            "recommendations": []
        }
        
        worst_status = PerformanceStatus.OPTIMAL
        
        for metric_name, buffer in metrics.items():
            stats = buffer.calculate_stats()
            if not stats:
                continue
            
            analysis["metrics"][metric_name] = stats
            
            # Determine status based on thresholds
            try:
                metric_enum = PerformanceMetric(metric_name)
                if metric_enum in self.default_thresholds:
                    threshold = self.default_thresholds[metric_enum]
                    status = self._get_status_from_value(stats["mean"], threshold)
                    
                    if self._status_priority(status) > self._status_priority(worst_status):
                        worst_status = status
                    
                    if status in [PerformanceStatus.POOR, PerformanceStatus.CRITICAL]:
                        analysis["issues"].append({
                            "metric": metric_name,
                            "status": status.value,
                            "current_value": stats["mean"],
                            "threshold": threshold.poor if status == PerformanceStatus.POOR else threshold.critical
                        })
            except ValueError:
                # Unknown metric type
                pass
        
        analysis["status"] = worst_status
        
        # Generate recommendations if needed
        if worst_status in [PerformanceStatus.DEGRADED, PerformanceStatus.POOR, PerformanceStatus.CRITICAL]:
            recommendations = await self._generate_recommendations(component_name, analysis)
            analysis["recommendations"] = recommendations
        
        return analysis
    
    def _get_status_from_value(self, value: float, threshold: PerformanceThreshold) -> PerformanceStatus:
        """Get performance status from value and threshold"""
        # For metrics where lower is better (response time, latency, error rate)
        if threshold.metric in [PerformanceMetric.RESPONSE_TIME, PerformanceMetric.DATABASE_LATENCY,
                               PerformanceMetric.ERROR_RATE, PerformanceMetric.CPU_USAGE,
                               PerformanceMetric.MEMORY_USAGE, PerformanceMetric.QUEUE_LENGTH]:
            if value <= threshold.optimal:
                return PerformanceStatus.OPTIMAL
            elif value <= threshold.good:
                return PerformanceStatus.GOOD
            elif value <= threshold.degraded:
                return PerformanceStatus.DEGRADED
            elif value <= threshold.poor:
                return PerformanceStatus.POOR
            else:
                return PerformanceStatus.CRITICAL
        
        # For metrics where higher is better (throughput, cache hit rate)
        else:
            if value >= threshold.optimal:
                return PerformanceStatus.OPTIMAL
            elif value >= threshold.good:
                return PerformanceStatus.GOOD
            elif value >= threshold.degraded:
                return PerformanceStatus.DEGRADED
            elif value >= threshold.poor:
                return PerformanceStatus.POOR
            else:
                return PerformanceStatus.CRITICAL
    
    def _status_priority(self, status: PerformanceStatus) -> int:
        """Get priority value for status (higher = worse)"""
        priorities = {
            PerformanceStatus.OPTIMAL: 0,
            PerformanceStatus.GOOD: 1,
            PerformanceStatus.DEGRADED: 2,
            PerformanceStatus.POOR: 3,
            PerformanceStatus.CRITICAL: 4
        }
        return priorities.get(status, 0)
    
    async def _detect_anomalies(self) -> None:
        """Detect performance anomalies"""
        for component_name, metrics in self.metrics_buffers.items():
            for metric_name, buffer in metrics.items():
                try:
                    anomaly = await self._detect_anomaly_in_metric(component_name, metric_name, buffer)
                    if anomaly:
                        self.anomalies.append(anomaly)
                        self.stats["anomalies_detected"] += 1
                        logger.warning(f"Anomaly detected: {anomaly.component} - {anomaly.metric.value}")
                except Exception as e:
                    logger.error(f"Error detecting anomaly: {e}")
    
    async def _detect_anomaly_in_metric(self, component: str, metric_name: str,
                                       buffer: PerformanceBuffer) -> Optional[PerformanceAnomaly]:
        """Detect anomaly in a specific metric"""
        recent_data = buffer.get_recent(300)  # Last 5 minutes
        if len(recent_data) < 10:
            return None
        
        recent_values = [val for _, val in recent_data]
        
        # Calculate statistics
        mean = statistics.mean(recent_values)
        std_dev = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
        
        if std_dev == 0:
            return None
        
        # Check for outliers (3 sigma rule)
        latest_value = recent_values[-1]
        z_score = abs((latest_value - mean) / std_dev)
        
        if z_score > 3:
            try:
                metric_enum = PerformanceMetric(metric_name)
            except ValueError:
                metric_enum = PerformanceMetric.RESPONSE_TIME
            
            # Determine severity
            if z_score > 5:
                severity = "critical"
            elif z_score > 4:
                severity = "high"
            else:
                severity = "medium"
            
            deviation_percent = ((latest_value - mean) / mean) * 100
            
            # Generate probable causes and recommendations
            probable_causes = self._analyze_probable_causes(component, metric_enum, deviation_percent)
            recommended_actions = self._generate_remediation_actions(component, metric_enum, severity)
            
            return PerformanceAnomaly(
                anomaly_id=f"anomaly_{datetime.now().timestamp()}",
                component=component,
                metric=metric_enum,
                severity=severity,
                detected_at=datetime.now(),
                current_value=latest_value,
                expected_value=mean,
                deviation_percent=deviation_percent,
                duration_seconds=len(recent_data),
                probable_causes=probable_causes,
                recommended_actions=recommended_actions
            )
        
        return None
    
    async def _check_immediate_anomaly(self, component: str, metric: PerformanceMetric,
                                      value: float) -> None:
        """Check for immediate anomalies in real-time"""
        # Quick threshold check
        if metric in self.default_thresholds:
            threshold = self.default_thresholds[metric]
            status = self._get_status_from_value(value, threshold)
            
            if status == PerformanceStatus.CRITICAL:
                anomaly = PerformanceAnomaly(
                    anomaly_id=f"immediate_{datetime.now().timestamp()}",
                    component=component,
                    metric=metric,
                    severity="critical",
                    detected_at=datetime.now(),
                    current_value=value,
                    expected_value=threshold.critical,
                    deviation_percent=((value - threshold.critical) / threshold.critical) * 100,
                    duration_seconds=0,
                    probable_causes=["Sudden spike detected", "Possible system overload"],
                    recommended_actions=["Immediate investigation required", "Check system resources"]
                )
                
                self.anomalies.append(anomaly)
                self.stats["anomalies_detected"] += 1
    
    def _analyze_probable_causes(self, component: str, metric: PerformanceMetric,
                                deviation_percent: float) -> List[str]:
        """Analyze probable causes for performance anomaly"""
        causes = []
        
        if metric == PerformanceMetric.RESPONSE_TIME:
            if deviation_percent > 0:
                causes.extend([
                    "Increased load or traffic",
                    "Slow database queries",
                    "External service latency",
                    "Resource contention"
                ])
            
        elif metric == PerformanceMetric.CPU_USAGE:
            if deviation_percent > 0:
                causes.extend([
                    "CPU-intensive operations",
                    "Inefficient algorithms",
                    "Memory leaks causing GC pressure",
                    "Runaway processes"
                ])
        
        elif metric == PerformanceMetric.MEMORY_USAGE:
            if deviation_percent > 0:
                causes.extend([
                    "Memory leak",
                    "Large data processing",
                    "Cache overflow",
                    "Insufficient garbage collection"
                ])
        
        elif metric == PerformanceMetric.ERROR_RATE:
            if deviation_percent > 0:
                causes.extend([
                    "Service degradation",
                    "Network issues",
                    "Configuration problems",
                    "Dependency failures"
                ])
        
        elif metric == PerformanceMetric.DATABASE_LATENCY:
            if deviation_percent > 0:
                causes.extend([
                    "Slow queries",
                    "Missing indexes",
                    "Lock contention",
                    "Connection pool exhaustion"
                ])
        
        return causes[:3]  # Return top 3 causes
    
    def _generate_remediation_actions(self, component: str, metric: PerformanceMetric,
                                     severity: str) -> List[str]:
        """Generate remediation actions for anomaly"""
        actions = []
        
        if severity == "critical":
            actions.append("Immediate investigation and intervention required")
        
        if metric == PerformanceMetric.RESPONSE_TIME:
            actions.extend([
                "Enable caching for frequently accessed data",
                "Optimize database queries",
                "Scale horizontally if load is high",
                "Review and optimize code hot paths"
            ])
        
        elif metric == PerformanceMetric.CPU_USAGE:
            actions.extend([
                "Profile CPU usage to identify hot spots",
                "Optimize algorithms and data structures",
                "Consider vertical scaling",
                "Implement request throttling"
            ])
        
        elif metric == PerformanceMetric.MEMORY_USAGE:
            actions.extend([
                "Analyze memory dumps for leaks",
                "Increase heap size if appropriate",
                "Optimize data structures",
                "Implement memory pooling"
            ])
        
        elif metric == PerformanceMetric.ERROR_RATE:
            actions.extend([
                "Review error logs for patterns",
                "Implement circuit breakers",
                "Add retry logic with backoff",
                "Validate input data and configurations"
            ])
        
        elif metric == PerformanceMetric.DATABASE_LATENCY:
            actions.extend([
                "Analyze slow query logs",
                "Add or optimize indexes",
                "Implement query result caching",
                "Consider read replicas for scaling"
            ])
        
        return actions[:4]  # Return top 4 actions
    
    async def _generate_recommendations(self, component: str,
                                       analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations"""
        recommendations = []
        
        for issue in analysis.get("issues", []):
            metric = issue["metric"]
            
            try:
                metric_enum = PerformanceMetric(metric)
                
                # Generate recommendation based on metric type
                if metric_enum == PerformanceMetric.RESPONSE_TIME:
                    rec = await self._recommend_response_time_optimization(component, issue)
                elif metric_enum == PerformanceMetric.CPU_USAGE:
                    rec = await self._recommend_cpu_optimization(component, issue)
                elif metric_enum == PerformanceMetric.MEMORY_USAGE:
                    rec = await self._recommend_memory_optimization(component, issue)
                elif metric_enum == PerformanceMetric.DATABASE_LATENCY:
                    rec = await self._recommend_database_optimization(component, issue)
                else:
                    rec = await self._recommend_generic_optimization(component, issue)
                
                if rec:
                    recommendations.append(rec)
                    self.stats["recommendations_generated"] += 1
                    
            except ValueError:
                pass
        
        return recommendations
    
    async def _recommend_response_time_optimization(self, component: str,
                                                  issue: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response time optimization recommendation"""
        return {
            "type": OptimizationType.CACHING.value,
            "priority": "high",
            "title": f"Implement caching for {component}",
            "description": f"Response time is {issue['current_value']:.0f}ms, implement caching to reduce latency",
            "expected_improvement": {"response_time": -50},
            "steps": [
                "Identify frequently accessed data",
                "Implement Redis or Memcached",
                "Set appropriate TTL values",
                "Monitor cache hit rates"
            ]
        }
    
    async def _recommend_cpu_optimization(self, component: str,
                                        issue: Dict[str, Any]) -> Dict[str, Any]:
        """Generate CPU optimization recommendation"""
        return {
            "type": OptimizationType.CODE_OPTIMIZATION.value,
            "priority": "high",
            "title": f"Optimize CPU usage for {component}",
            "description": f"CPU usage is {issue['current_value']:.0f}%, optimize code to reduce load",
            "expected_improvement": {"cpu_usage": -30},
            "steps": [
                "Profile code to identify hot spots",
                "Optimize algorithms and loops",
                "Implement async/parallel processing",
                "Consider caching computed results"
            ]
        }
    
    async def _recommend_memory_optimization(self, component: str,
                                           issue: Dict[str, Any]) -> Dict[str, Any]:
        """Generate memory optimization recommendation"""
        return {
            "type": OptimizationType.RESOURCE_ALLOCATION.value,
            "priority": "high",
            "title": f"Optimize memory usage for {component}",
            "description": f"Memory usage is {issue['current_value']:.0f}%, optimize to prevent OOM",
            "expected_improvement": {"memory_usage": -25},
            "steps": [
                "Analyze memory allocation patterns",
                "Implement object pooling",
                "Optimize data structures",
                "Configure GC settings"
            ]
        }
    
    async def _recommend_database_optimization(self, component: str,
                                             issue: Dict[str, Any]) -> Dict[str, Any]:
        """Generate database optimization recommendation"""
        return {
            "type": OptimizationType.INDEXING.value,
            "priority": "high",
            "title": f"Optimize database queries for {component}",
            "description": f"Database latency is {issue['current_value']:.0f}ms, optimize queries",
            "expected_improvement": {"database_latency": -60},
            "steps": [
                "Analyze slow query log",
                "Add appropriate indexes",
                "Optimize query structure",
                "Consider query result caching"
            ]
        }
    
    async def _recommend_generic_optimization(self, component: str,
                                            issue: Dict[str, Any]) -> Dict[str, Any]:
        """Generate generic optimization recommendation"""
        return {
            "type": OptimizationType.CONFIGURATION.value,
            "priority": "medium",
            "title": f"Review configuration for {component}",
            "description": f"Performance degradation detected in {issue['metric']}",
            "expected_improvement": {},
            "steps": [
                "Review current configuration",
                "Compare with best practices",
                "Test configuration changes",
                "Monitor impact"
            ]
        }
    
    async def _calculate_error_rate(self, component: str) -> float:
        """Calculate error rate for component"""
        # This would calculate actual error rate from metrics
        # For now, return mock value
        return 0.5
    
    async def generate_performance_report(self, start_time: datetime = None,
                                        end_time: datetime = None) -> PerformanceReport:
        """Generate comprehensive performance report"""
        try:
            if not end_time:
                end_time = datetime.now()
            if not start_time:
                start_time = end_time - timedelta(hours=24)
            
            # Analyze all components
            component_statuses = {}
            key_metrics = {}
            
            for component_name, metrics in self.metrics_buffers.items():
                analysis = await self._analyze_component(component_name, metrics)
                component_statuses[component_name] = analysis["status"]
                key_metrics[component_name] = analysis["metrics"]
            
            # Determine overall status
            overall_status = PerformanceStatus.OPTIMAL
            for status in component_statuses.values():
                if self._status_priority(status) > self._status_priority(overall_status):
                    overall_status = status
            
            # Get recent anomalies
            recent_anomalies = [
                a for a in self.anomalies
                if start_time <= a.detected_at <= end_time
            ]
            
            # Get recent recommendations
            recent_recommendations = [
                {
                    "id": r.recommendation_id,
                    "type": r.type.value,
                    "priority": r.priority,
                    "component": r.component,
                    "title": r.title,
                    "description": r.description
                }
                for r in self.recommendations
                if start_time <= r.created_at <= end_time
            ]
            
            # Analyze trends
            trends = await self._analyze_trends(key_metrics)
            
            # Generate predictions
            predictions = await self._generate_predictions(key_metrics)
            
            report = PerformanceReport(
                report_id=f"report_{datetime.now().timestamp()}",
                period_start=start_time,
                period_end=end_time,
                overall_status=overall_status,
                component_statuses=component_statuses,
                key_metrics=key_metrics,
                anomalies=recent_anomalies,
                optimizations=recent_recommendations,
                trends=trends,
                predictions=predictions
            )
            
            self.stats["reports_generated"] += 1
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            raise
    
    async def _analyze_trends(self, metrics: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Analyze performance trends"""
        trends = {}
        
        for component, component_metrics in metrics.items():
            for metric_name, stats in component_metrics.items():
                if isinstance(stats, dict) and "mean" in stats:
                    # Simple trend analysis (would be more sophisticated in production)
                    key = f"{component}.{metric_name}"
                    
                    # Check if metric is improving or degrading
                    # This would compare with historical data
                    trends[key] = "stable"  # Could be: improving, degrading, stable, volatile
        
        return trends
    
    async def _generate_predictions(self, metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate performance predictions"""
        predictions = {
            "next_hour": {},
            "next_day": {},
            "recommendations": []
        }
        
        # Simple predictions (would use ML models in production)
        for component, component_metrics in metrics.items():
            for metric_name, stats in component_metrics.items():
                if isinstance(stats, dict) and "mean" in stats:
                    # Predict based on current trend
                    predictions["next_hour"][f"{component}.{metric_name}"] = stats["mean"] * 1.05
                    predictions["next_day"][f"{component}.{metric_name}"] = stats["mean"] * 1.1
        
        # Generate proactive recommendations
        predictions["recommendations"] = [
            "Consider scaling if traffic increases by 20%",
            "Schedule maintenance window for optimization",
            "Review and update performance thresholds"
        ]
        
        return predictions
    
    def get_current_performance_status(self) -> Dict[str, Any]:
        """Get current performance status summary"""
        status_summary = {
            "overall_status": PerformanceStatus.OPTIMAL.value,
            "component_count": len(self.metrics_buffers),
            "active_anomalies": len([a for a in self.anomalies if a.severity in ["high", "critical"]]),
            "pending_recommendations": len(self.recommendations),
            "stats": self.stats
        }
        
        # Calculate overall status
        worst_status = PerformanceStatus.OPTIMAL
        for component_name, metrics in self.metrics_buffers.items():
            for metric_name, buffer in metrics.items():
                stats = buffer.calculate_stats()
                if stats:
                    try:
                        metric_enum = PerformanceMetric(metric_name)
                        if metric_enum in self.default_thresholds:
                            threshold = self.default_thresholds[metric_enum]
                            status = self._get_status_from_value(stats["mean"], threshold)
                            
                            if self._status_priority(status) > self._status_priority(worst_status):
                                worst_status = status
                    except ValueError:
                        pass
        
        status_summary["overall_status"] = worst_status.value
        
        return status_summary
    
    async def shutdown(self) -> None:
        """Shutdown performance analyzer"""
        logger.info("Shutting down performance analyzer...")
        
        self.analysis_enabled = False
        self.anomaly_detection_enabled = False
        
        if self._analysis_task:
            self._analysis_task.cancel()
        if self._anomaly_detection_task:
            self._anomaly_detection_task.cancel()
        
        logger.info("Performance analyzer shutdown complete")