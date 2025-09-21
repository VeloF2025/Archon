"""
Analytics Engine
Core engine for enterprise-grade analytics and insights generation
"""

import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import logging
import hashlib

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics tracked"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    RATE = "rate"
    PERCENTILE = "percentile"
    MOVING_AVERAGE = "moving_average"
    CUMULATIVE = "cumulative"


class AggregationType(Enum):
    """Types of aggregation methods"""
    SUM = "sum"
    AVERAGE = "average"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE_50 = "p50"
    PERCENTILE_90 = "p90"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"
    STANDARD_DEVIATION = "std_dev"
    VARIANCE = "variance"


class TimeWindow(Enum):
    """Time windows for analytics"""
    MINUTE = "minute"
    FIVE_MINUTES = "5_minutes"
    FIFTEEN_MINUTES = "15_minutes"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


@dataclass
class MetricPoint:
    """Individual metric data point"""
    metric_id: str
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    unit: Optional[str] = None


@dataclass
class TimeSeriesData:
    """Time series data structure"""
    metric_name: str
    timestamps: List[datetime]
    values: List[float]
    aggregation: AggregationType
    window: TimeWindow
    tags: Dict[str, str] = field(default_factory=dict)
    statistics: Dict[str, float] = field(default_factory=dict)


@dataclass
class AnalyticsQuery:
    """Query for analytics data"""
    metrics: List[str]
    start_time: datetime
    end_time: datetime
    aggregation: AggregationType = AggregationType.AVERAGE
    window: TimeWindow = TimeWindow.HOUR
    filters: Dict[str, Any] = field(default_factory=dict)
    group_by: List[str] = field(default_factory=list)
    order_by: Optional[str] = None
    limit: Optional[int] = None


@dataclass
class AnalyticsResult:
    """Result from analytics query"""
    query: AnalyticsQuery
    data: List[TimeSeriesData]
    summary: Dict[str, Any]
    execution_time_ms: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Alert:
    """Alert configuration and state"""
    alert_id: str
    name: str
    metric: str
    condition: str  # e.g., "value > 100"
    threshold: float
    window: TimeWindow
    severity: str  # critical, warning, info
    enabled: bool = True
    triggered: bool = False
    last_triggered: Optional[datetime] = None
    notification_channels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dashboard:
    """Dashboard configuration"""
    dashboard_id: str
    name: str
    description: str
    widgets: List[Dict[str, Any]]
    layout: Dict[str, Any]
    refresh_interval: int  # seconds
    filters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


class MetricsStorage:
    """In-memory metrics storage with time-based eviction"""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}
        self.indexes: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        
    def add_metric(self, metric: MetricPoint) -> None:
        """Add a metric point to storage"""
        # Store metric
        self.metrics[metric.name].append(metric)
        
        # Update metadata
        if metric.name not in self.metric_metadata:
            self.metric_metadata[metric.name] = {
                "type": metric.metric_type,
                "unit": metric.unit,
                "first_seen": metric.timestamp,
                "last_seen": metric.timestamp,
                "tags": set()
            }
        
        self.metric_metadata[metric.name]["last_seen"] = metric.timestamp
        self.metric_metadata[metric.name]["tags"].update(metric.tags.keys())
        
        # Update indexes
        for tag_key, tag_value in metric.tags.items():
            index_key = f"{tag_key}:{tag_value}"
            if metric.name not in self.indexes[index_key]:
                self.indexes[index_key].append(metric.name)
    
    def query_metrics(self, metric_names: List[str], start_time: datetime, 
                     end_time: datetime, filters: Dict[str, Any] = None) -> List[MetricPoint]:
        """Query metrics from storage"""
        results = []
        
        for metric_name in metric_names:
            if metric_name in self.metrics:
                for metric in self.metrics[metric_name]:
                    # Time filter
                    if not (start_time <= metric.timestamp <= end_time):
                        continue
                    
                    # Tag filters
                    if filters:
                        match = True
                        for key, value in filters.items():
                            if key not in metric.tags or metric.tags[key] != value:
                                match = False
                                break
                        if not match:
                            continue
                    
                    results.append(metric)
        
        return results
    
    def cleanup_old_metrics(self) -> int:
        """Remove metrics older than retention period"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        removed_count = 0
        
        for metric_name in list(self.metrics.keys()):
            # Remove old metrics
            original_len = len(self.metrics[metric_name])
            self.metrics[metric_name] = deque(
                (m for m in self.metrics[metric_name] if m.timestamp > cutoff_time),
                maxlen=10000
            )
            removed_count += original_len - len(self.metrics[metric_name])
            
            # Remove empty metric series
            if not self.metrics[metric_name]:
                del self.metrics[metric_name]
                if metric_name in self.metric_metadata:
                    del self.metric_metadata[metric_name]
        
        return removed_count


class AnalyticsEngine:
    """Core analytics engine for enterprise metrics and insights"""
    
    def __init__(self, retention_hours: int = 168):  # 7 days default
        self.storage = MetricsStorage(retention_hours)
        self.alerts: Dict[str, Alert] = {}
        self.dashboards: Dict[str, Dashboard] = {}
        self.query_cache: Dict[str, AnalyticsResult] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Real-time processing
        self.stream_processors: Dict[str, callable] = {}
        self.aggregation_buffers: Dict[str, List[float]] = defaultdict(list)
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._alert_evaluation_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "total_metrics_received": 0,
            "total_queries_executed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "alerts_triggered": 0,
            "last_cleanup": None
        }
        
        # Start background tasks
        asyncio.create_task(self._start_background_tasks())
    
    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._alert_evaluation_task = asyncio.create_task(self._alert_evaluation_loop())
    
    async def _cleanup_loop(self) -> None:
        """Background task to cleanup old metrics"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                removed = self.storage.cleanup_old_metrics()
                self.stats["last_cleanup"] = datetime.now()
                logger.info(f"Cleaned up {removed} old metrics")
                
                # Clear old cache entries
                self._cleanup_cache()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _alert_evaluation_loop(self) -> None:
        """Background task to evaluate alerts"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._evaluate_all_alerts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert evaluation loop: {e}")
    
    async def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE,
                          tags: Dict[str, str] = None, unit: Optional[str] = None) -> None:
        """Record a single metric point"""
        try:
            metric = MetricPoint(
                metric_id=str(uuid.uuid4()),
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {},
                metric_type=metric_type,
                unit=unit
            )
            
            # Store metric
            self.storage.add_metric(metric)
            self.stats["total_metrics_received"] += 1
            
            # Process through stream processors
            await self._process_metric_stream(metric)
            
            # Check alerts
            await self._check_metric_alerts(metric)
            
        except Exception as e:
            logger.error(f"Error recording metric {name}: {e}")
    
    async def record_batch_metrics(self, metrics: List[Dict[str, Any]]) -> None:
        """Record multiple metrics in batch"""
        try:
            for metric_data in metrics:
                await self.record_metric(
                    name=metric_data["name"],
                    value=metric_data["value"],
                    metric_type=MetricType(metric_data.get("type", "gauge")),
                    tags=metric_data.get("tags", {}),
                    unit=metric_data.get("unit")
                )
        except Exception as e:
            logger.error(f"Error recording batch metrics: {e}")
    
    async def query(self, query: AnalyticsQuery) -> AnalyticsResult:
        """Execute analytics query"""
        start_time = datetime.now()
        
        try:
            # Check cache
            cache_key = self._generate_cache_key(query)
            if cache_key in self.query_cache:
                cached_result = self.query_cache[cache_key]
                if (datetime.now() - cached_result.timestamp).seconds < self.cache_ttl:
                    self.stats["cache_hits"] += 1
                    return cached_result
            
            self.stats["cache_misses"] += 1
            self.stats["total_queries_executed"] += 1
            
            # Query metrics from storage
            raw_metrics = self.storage.query_metrics(
                query.metrics,
                query.start_time,
                query.end_time,
                query.filters
            )
            
            # Group and aggregate data
            time_series_data = await self._aggregate_metrics(raw_metrics, query)
            
            # Calculate summary statistics
            summary = self._calculate_summary(time_series_data)
            
            # Create result
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            result = AnalyticsResult(
                query=query,
                data=time_series_data,
                summary=summary,
                execution_time_ms=execution_time
            )
            
            # Cache result
            self.query_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    async def _aggregate_metrics(self, metrics: List[MetricPoint], 
                                query: AnalyticsQuery) -> List[TimeSeriesData]:
        """Aggregate metrics based on query parameters"""
        if not metrics:
            return []
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame([
            {
                "timestamp": m.timestamp,
                "value": m.value,
                "metric": m.name,
                **m.tags
            }
            for m in metrics
        ])
        
        # Set timestamp as index
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        
        # Determine resampling frequency
        freq_map = {
            TimeWindow.MINUTE: "1T",
            TimeWindow.FIVE_MINUTES: "5T",
            TimeWindow.FIFTEEN_MINUTES: "15T",
            TimeWindow.HOUR: "1H",
            TimeWindow.DAY: "1D",
            TimeWindow.WEEK: "1W",
            TimeWindow.MONTH: "1M",
            TimeWindow.QUARTER: "1Q",
            TimeWindow.YEAR: "1Y"
        }
        
        freq = freq_map.get(query.window, "1H")
        
        # Group by metric and tags if specified
        group_cols = ["metric"]
        if query.group_by:
            group_cols.extend(query.group_by)
        
        time_series_list = []
        
        for group_keys, group_df in df.groupby(group_cols):
            # Resample and aggregate
            resampled = group_df["value"].resample(freq)
            
            if query.aggregation == AggregationType.SUM:
                aggregated = resampled.sum()
            elif query.aggregation == AggregationType.AVERAGE:
                aggregated = resampled.mean()
            elif query.aggregation == AggregationType.MEDIAN:
                aggregated = resampled.median()
            elif query.aggregation == AggregationType.MIN:
                aggregated = resampled.min()
            elif query.aggregation == AggregationType.MAX:
                aggregated = resampled.max()
            elif query.aggregation == AggregationType.COUNT:
                aggregated = resampled.count()
            elif query.aggregation == AggregationType.PERCENTILE_50:
                aggregated = resampled.quantile(0.5)
            elif query.aggregation == AggregationType.PERCENTILE_90:
                aggregated = resampled.quantile(0.9)
            elif query.aggregation == AggregationType.PERCENTILE_95:
                aggregated = resampled.quantile(0.95)
            elif query.aggregation == AggregationType.PERCENTILE_99:
                aggregated = resampled.quantile(0.99)
            elif query.aggregation == AggregationType.STANDARD_DEVIATION:
                aggregated = resampled.std()
            elif query.aggregation == AggregationType.VARIANCE:
                aggregated = resampled.var()
            else:
                aggregated = resampled.mean()
            
            # Remove NaN values
            aggregated = aggregated.fillna(0)
            
            # Create tags dictionary
            tags = {}
            if isinstance(group_keys, tuple):
                tags["metric"] = group_keys[0]
                for i, tag_name in enumerate(query.group_by):
                    if i + 1 < len(group_keys):
                        tags[tag_name] = group_keys[i + 1]
            else:
                tags["metric"] = group_keys
            
            # Calculate statistics
            statistics = {
                "min": float(aggregated.min()),
                "max": float(aggregated.max()),
                "mean": float(aggregated.mean()),
                "median": float(aggregated.median()),
                "std": float(aggregated.std()),
                "count": len(aggregated)
            }
            
            time_series = TimeSeriesData(
                metric_name=tags["metric"],
                timestamps=aggregated.index.tolist(),
                values=aggregated.values.tolist(),
                aggregation=query.aggregation,
                window=query.window,
                tags=tags,
                statistics=statistics
            )
            
            time_series_list.append(time_series)
        
        # Sort if requested
        if query.order_by:
            reverse = query.order_by.startswith("-")
            order_field = query.order_by.lstrip("-")
            
            if order_field in ["min", "max", "mean", "median"]:
                time_series_list.sort(
                    key=lambda ts: ts.statistics.get(order_field, 0),
                    reverse=reverse
                )
        
        # Apply limit
        if query.limit:
            time_series_list = time_series_list[:query.limit]
        
        return time_series_list
    
    def _calculate_summary(self, time_series_data: List[TimeSeriesData]) -> Dict[str, Any]:
        """Calculate summary statistics across all time series"""
        if not time_series_data:
            return {}
        
        all_values = []
        for ts in time_series_data:
            all_values.extend(ts.values)
        
        if not all_values:
            return {}
        
        return {
            "total_series": len(time_series_data),
            "total_points": len(all_values),
            "global_min": min(all_values),
            "global_max": max(all_values),
            "global_mean": np.mean(all_values),
            "global_median": np.median(all_values),
            "global_std": np.std(all_values),
            "time_range": {
                "start": min(ts.timestamps[0] for ts in time_series_data if ts.timestamps),
                "end": max(ts.timestamps[-1] for ts in time_series_data if ts.timestamps)
            }
        }
    
    async def create_alert(self, name: str, metric: str, condition: str, 
                          threshold: float, window: TimeWindow,
                          severity: str = "warning",
                          notification_channels: List[str] = None) -> Alert:
        """Create a new alert rule"""
        try:
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                name=name,
                metric=metric,
                condition=condition,
                threshold=threshold,
                window=window,
                severity=severity,
                notification_channels=notification_channels or []
            )
            
            self.alerts[alert.alert_id] = alert
            logger.info(f"Created alert: {name}")
            
            return alert
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            raise
    
    async def _evaluate_all_alerts(self) -> None:
        """Evaluate all active alerts"""
        for alert_id, alert in self.alerts.items():
            if alert.enabled:
                await self._evaluate_alert(alert)
    
    async def _evaluate_alert(self, alert: Alert) -> bool:
        """Evaluate a single alert"""
        try:
            # Query recent metrics
            end_time = datetime.now()
            window_map = {
                TimeWindow.MINUTE: timedelta(minutes=1),
                TimeWindow.FIVE_MINUTES: timedelta(minutes=5),
                TimeWindow.FIFTEEN_MINUTES: timedelta(minutes=15),
                TimeWindow.HOUR: timedelta(hours=1),
                TimeWindow.DAY: timedelta(days=1),
                TimeWindow.WEEK: timedelta(weeks=1)
            }
            
            start_time = end_time - window_map.get(alert.window, timedelta(hours=1))
            
            query = AnalyticsQuery(
                metrics=[alert.metric],
                start_time=start_time,
                end_time=end_time,
                aggregation=AggregationType.AVERAGE,
                window=alert.window
            )
            
            result = await self.query(query)
            
            if not result.data:
                return False
            
            # Evaluate condition
            latest_value = result.data[0].values[-1] if result.data[0].values else 0
            
            triggered = False
            if ">" in alert.condition:
                triggered = latest_value > alert.threshold
            elif "<" in alert.condition:
                triggered = latest_value < alert.threshold
            elif "=" in alert.condition:
                triggered = abs(latest_value - alert.threshold) < 0.001
            
            # Update alert state
            if triggered and not alert.triggered:
                alert.triggered = True
                alert.last_triggered = datetime.now()
                self.stats["alerts_triggered"] += 1
                await self._send_alert_notification(alert, latest_value)
                logger.warning(f"Alert triggered: {alert.name} (value: {latest_value})")
            elif not triggered and alert.triggered:
                alert.triggered = False
                logger.info(f"Alert resolved: {alert.name}")
            
            return triggered
            
        except Exception as e:
            logger.error(f"Error evaluating alert {alert.name}: {e}")
            return False
    
    async def _check_metric_alerts(self, metric: MetricPoint) -> None:
        """Check if a metric triggers any alerts"""
        for alert in self.alerts.values():
            if alert.enabled and alert.metric == metric.name:
                # Simple threshold check for real-time alerts
                triggered = False
                if ">" in alert.condition:
                    triggered = metric.value > alert.threshold
                elif "<" in alert.condition:
                    triggered = metric.value < alert.threshold
                
                if triggered and not alert.triggered:
                    alert.triggered = True
                    alert.last_triggered = datetime.now()
                    self.stats["alerts_triggered"] += 1
                    await self._send_alert_notification(alert, metric.value)
    
    async def _send_alert_notification(self, alert: Alert, value: float) -> None:
        """Send alert notification"""
        # This would integrate with notification systems
        logger.warning(f"ALERT: {alert.name} - Value {value} {alert.condition} (threshold: {alert.threshold})")
    
    async def _process_metric_stream(self, metric: MetricPoint) -> None:
        """Process metric through stream processors"""
        for processor_name, processor in self.stream_processors.items():
            try:
                await processor(metric)
            except Exception as e:
                logger.error(f"Error in stream processor {processor_name}: {e}")
    
    def register_stream_processor(self, name: str, processor: callable) -> None:
        """Register a stream processor for real-time metric processing"""
        self.stream_processors[name] = processor
    
    async def create_dashboard(self, name: str, description: str,
                             widgets: List[Dict[str, Any]],
                             layout: Dict[str, Any] = None,
                             refresh_interval: int = 60) -> Dashboard:
        """Create a new dashboard"""
        try:
            dashboard = Dashboard(
                dashboard_id=str(uuid.uuid4()),
                name=name,
                description=description,
                widgets=widgets,
                layout=layout or {"type": "grid", "columns": 12},
                refresh_interval=refresh_interval
            )
            
            self.dashboards[dashboard.dashboard_id] = dashboard
            logger.info(f"Created dashboard: {name}")
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            raise
    
    async def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get data for all widgets in a dashboard"""
        try:
            if dashboard_id not in self.dashboards:
                raise ValueError(f"Dashboard {dashboard_id} not found")
            
            dashboard = self.dashboards[dashboard_id]
            widget_data = {}
            
            for widget in dashboard.widgets:
                widget_id = widget.get("id", str(uuid.uuid4()))
                widget_type = widget.get("type", "line_chart")
                
                # Create query from widget configuration
                query = AnalyticsQuery(
                    metrics=widget.get("metrics", []),
                    start_time=datetime.now() - timedelta(hours=widget.get("time_range_hours", 24)),
                    end_time=datetime.now(),
                    aggregation=AggregationType(widget.get("aggregation", "average")),
                    window=TimeWindow(widget.get("window", "hour")),
                    filters=widget.get("filters", {})
                )
                
                # Execute query
                result = await self.query(query)
                
                # Format data for widget type
                if widget_type == "line_chart":
                    widget_data[widget_id] = self._format_line_chart_data(result)
                elif widget_type == "bar_chart":
                    widget_data[widget_id] = self._format_bar_chart_data(result)
                elif widget_type == "gauge":
                    widget_data[widget_id] = self._format_gauge_data(result)
                elif widget_type == "table":
                    widget_data[widget_id] = self._format_table_data(result)
                elif widget_type == "heatmap":
                    widget_data[widget_id] = self._format_heatmap_data(result)
                else:
                    widget_data[widget_id] = {"error": f"Unknown widget type: {widget_type}"}
            
            return {
                "dashboard": {
                    "id": dashboard.dashboard_id,
                    "name": dashboard.name,
                    "description": dashboard.description,
                    "layout": dashboard.layout,
                    "refresh_interval": dashboard.refresh_interval
                },
                "widgets": widget_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            raise
    
    def _format_line_chart_data(self, result: AnalyticsResult) -> Dict[str, Any]:
        """Format data for line chart widget"""
        series = []
        for ts in result.data:
            series.append({
                "name": ts.metric_name,
                "data": [
                    {"x": timestamp.isoformat(), "y": value}
                    for timestamp, value in zip(ts.timestamps, ts.values)
                ]
            })
        
        return {
            "type": "line_chart",
            "series": series,
            "summary": result.summary
        }
    
    def _format_bar_chart_data(self, result: AnalyticsResult) -> Dict[str, Any]:
        """Format data for bar chart widget"""
        categories = []
        values = []
        
        for ts in result.data:
            categories.append(ts.metric_name)
            values.append(ts.statistics.get("mean", 0))
        
        return {
            "type": "bar_chart",
            "categories": categories,
            "values": values,
            "summary": result.summary
        }
    
    def _format_gauge_data(self, result: AnalyticsResult) -> Dict[str, Any]:
        """Format data for gauge widget"""
        if result.data:
            current_value = result.data[0].values[-1] if result.data[0].values else 0
            min_value = result.data[0].statistics.get("min", 0)
            max_value = result.data[0].statistics.get("max", 100)
        else:
            current_value = 0
            min_value = 0
            max_value = 100
        
        return {
            "type": "gauge",
            "value": current_value,
            "min": min_value,
            "max": max_value,
            "summary": result.summary
        }
    
    def _format_table_data(self, result: AnalyticsResult) -> Dict[str, Any]:
        """Format data for table widget"""
        rows = []
        for ts in result.data:
            rows.append({
                "metric": ts.metric_name,
                "min": ts.statistics.get("min", 0),
                "max": ts.statistics.get("max", 0),
                "mean": ts.statistics.get("mean", 0),
                "median": ts.statistics.get("median", 0),
                "std": ts.statistics.get("std", 0),
                "count": ts.statistics.get("count", 0)
            })
        
        return {
            "type": "table",
            "columns": ["metric", "min", "max", "mean", "median", "std", "count"],
            "rows": rows
        }
    
    def _format_heatmap_data(self, result: AnalyticsResult) -> Dict[str, Any]:
        """Format data for heatmap widget"""
        data = []
        for ts in result.data:
            for i, (timestamp, value) in enumerate(zip(ts.timestamps, ts.values)):
                data.append({
                    "x": timestamp.isoformat(),
                    "y": ts.metric_name,
                    "value": value
                })
        
        return {
            "type": "heatmap",
            "data": data,
            "summary": result.summary
        }
    
    def _generate_cache_key(self, query: AnalyticsQuery) -> str:
        """Generate cache key for query"""
        key_parts = [
            ",".join(query.metrics),
            str(query.start_time),
            str(query.end_time),
            query.aggregation.value,
            query.window.value,
            json.dumps(query.filters, sort_keys=True),
            ",".join(query.group_by)
        ]
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _cleanup_cache(self) -> None:
        """Cleanup old cache entries"""
        cutoff_time = datetime.now() - timedelta(seconds=self.cache_ttl)
        keys_to_remove = []
        
        for key, result in self.query_cache.items():
            if result.timestamp < cutoff_time:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.query_cache[key]
    
    async def export_metrics(self, format: str = "json", 
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> Union[str, bytes]:
        """Export metrics in specified format"""
        try:
            if not end_time:
                end_time = datetime.now()
            if not start_time:
                start_time = end_time - timedelta(hours=24)
            
            # Get all metric names
            metric_names = list(self.storage.metrics.keys())
            
            # Query all metrics
            query = AnalyticsQuery(
                metrics=metric_names,
                start_time=start_time,
                end_time=end_time,
                aggregation=AggregationType.AVERAGE,
                window=TimeWindow.HOUR
            )
            
            result = await self.query(query)
            
            if format == "json":
                export_data = {
                    "export_time": datetime.now().isoformat(),
                    "time_range": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat()
                    },
                    "metrics": []
                }
                
                for ts in result.data:
                    export_data["metrics"].append({
                        "name": ts.metric_name,
                        "timestamps": [t.isoformat() for t in ts.timestamps],
                        "values": ts.values,
                        "statistics": ts.statistics,
                        "tags": ts.tags
                    })
                
                return json.dumps(export_data, indent=2)
            
            elif format == "csv":
                # Create CSV format
                lines = ["timestamp,metric,value,tags"]
                for ts in result.data:
                    tags_str = json.dumps(ts.tags)
                    for timestamp, value in zip(ts.timestamps, ts.values):
                        lines.append(f"{timestamp.isoformat()},{ts.metric_name},{value},{tags_str}")
                
                return "\n".join(lines)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            raise
    
    def get_metrics_info(self) -> Dict[str, Any]:
        """Get information about stored metrics"""
        return {
            "total_metrics": len(self.storage.metrics),
            "total_data_points": sum(len(series) for series in self.storage.metrics.values()),
            "metric_types": {
                name: meta["type"].value 
                for name, meta in self.storage.metric_metadata.items()
            },
            "retention_hours": self.storage.retention_hours,
            "stats": self.stats,
            "alerts": {
                "total": len(self.alerts),
                "enabled": sum(1 for a in self.alerts.values() if a.enabled),
                "triggered": sum(1 for a in self.alerts.values() if a.triggered)
            },
            "dashboards": {
                "total": len(self.dashboards),
                "widgets": sum(len(d.widgets) for d in self.dashboards.values())
            },
            "cache": {
                "entries": len(self.query_cache),
                "ttl_seconds": self.cache_ttl
            }
        }
    
    async def shutdown(self) -> None:
        """Shutdown analytics engine"""
        logger.info("Shutting down analytics engine...")
        
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._alert_evaluation_task:
            self._alert_evaluation_task.cancel()
        
        # Export final metrics if needed
        # await self.export_metrics("json")
        
        logger.info("Analytics engine shutdown complete")