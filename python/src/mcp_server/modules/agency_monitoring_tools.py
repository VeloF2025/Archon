"""
Agency Monitoring MCP Tools for Archon MCP Server

This module provides MCP tools for monitoring agency workflows including:
- Real-time workflow status monitoring
- Performance metrics collection
- Agent health monitoring
- Resource utilization tracking
- Error and exception tracking
- Analytics and reporting

Integration with Phase 1 Agency Swarm monitoring system.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

from mcp.server.fastmcp import Context, FastMCP

# Add the project root to Python path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import monitoring components
from src.agents.orchestration.parallel_executor import AgentStatus, ParallelExecutor
from src.server.config.logfire_config import mcp_logger

logger = logging.getLogger(__name__)


class MonitoringLevel(str, Enum):
    """Monitoring detail levels"""
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    DEBUG = "debug"


class MetricType(str, Enum):
    """Types of metrics to collect"""
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    COMMUNICATION = "communication"
    ERROR = "error"
    CUSTOM = "custom"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    agent_name: Optional[str] = None
    workflow_id: Optional[str] = None
    execution_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ResourceUsage:
    """Resource usage information"""
    timestamp: datetime
    agent_name: str
    cpu_percent: float
    memory_mb: float
    disk_mb: float
    network_io: Dict[str, int]
    active_connections: int


@dataclass
class HealthStatus:
    """Agent health status"""
    agent_name: str
    is_healthy: bool
    last_check: datetime
    response_time_ms: float
    error_message: Optional[str] = None
    uptime_seconds: float = 0


@dataclass
class Alert:
    """Monitoring alert"""
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    source: str
    metadata: Optional[Dict[str, Any]] = None
    resolved_at: Optional[datetime] = None


class AgencyMonitor:
    """Main monitoring class for agency workflows"""

    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size

        # Metrics storage
        self.performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.resource_usage: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.health_status: Dict[str, HealthStatus] = {}
        self.alerts: List[Alert] = []

        # Monitoring configuration
        self.monitoring_enabled = True
        self.collection_interval = 30  # seconds
        self.health_check_interval = 60  # seconds
        self.alert_thresholds = {
            "cpu_warning": 80.0,
            "cpu_critical": 95.0,
            "memory_warning": 85.0,
            "memory_critical": 95.0,
            "response_time_warning": 5000.0,  # 5 seconds
            "response_time_critical": 15000.0  # 15 seconds
        }

        # Monitoring tasks
        self._monitoring_task = None
        self._health_check_task = None

    async def start_monitoring(self) -> None:
        """Start background monitoring tasks"""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._collect_metrics_periodically())

        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._perform_health_checks_periodically())

        logger.info("Agency monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            self._monitoring_task = None

        if self._health_check_task:
            self._health_check_task.cancel()
            self._health_check_task = None

        logger.info("Agency monitoring stopped")

    async def _collect_metrics_periodically(self) -> None:
        """Periodically collect metrics from all active agencies"""
        while self.monitoring_enabled:
            try:
                await self._collect_performance_metrics()
                await self._collect_resource_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(self.collection_interval)

    async def _perform_health_checks_periodically(self) -> None:
        """Periodically perform health checks on agents"""
        while self.monitoring_enabled:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health checks: {e}")
                await asyncio.sleep(self.health_check_interval)

    async def _collect_performance_metrics(self) -> None:
        """Collect performance metrics from active agencies"""
        try:
            # This would integrate with actual agency instances
            # For now, we'll collect system-level metrics
            timestamp = datetime.utcnow()

            # Collect execution statistics if available
            try:
                from ..agency_workflow_tools import _workflow_manager
                executions = _workflow_manager.get_workflow_executions()

                # Calculate performance metrics
                running_executions = [ex for ex in executions if ex["status"] == "running"]
                completed_executions = [ex for ex in executions if ex["status"] == "completed"]

                # Active workflows metric
                metric = PerformanceMetric(
                    timestamp=timestamp,
                    metric_name="active_workflows",
                    value=float(len(running_executions)),
                    unit="count",
                    metadata={"total_executions": len(executions)}
                )
                self.add_performance_metric(metric)

                # Completion rate metric (last hour)
                one_hour_ago = timestamp - timedelta(hours=1)
                recent_completed = [
                    ex for ex in completed_executions
                    if ex["completed_at"] and datetime.fromisoformat(ex["completed_at"]) > one_hour_ago
                ]
                completion_rate = len(recent_completed) / 1.0  # per hour

                metric = PerformanceMetric(
                    timestamp=timestamp,
                    metric_name="workflow_completion_rate",
                    value=completion_rate,
                    unit="workflows/hour"
                )
                self.add_performance_metric(metric)

            except ImportError:
                # Workflow manager not available
                pass

        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")

    async def _collect_resource_metrics(self) -> None:
        """Collect resource usage metrics"""
        try:
            import psutil

            timestamp = datetime.utcnow()

            # System-wide resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Network I/O
            network_io = psutil.net_io_counters()
            network_data = {
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv,
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv
            }

            # Store system resource metrics
            metric = PerformanceMetric(
                timestamp=timestamp,
                metric_name="system_cpu_percent",
                value=cpu_percent,
                unit="percent"
            )
            self.add_performance_metric(metric)

            metric = PerformanceMetric(
                timestamp=timestamp,
                metric_name="system_memory_percent",
                value=memory.percent,
                unit="percent"
            )
            self.add_performance_metric(metric)

            metric = PerformanceMetric(
                timestamp=timestamp,
                metric_name="system_disk_percent",
                value=disk.percent,
                unit="percent"
            )
            self.add_performance_metric(metric)

            # Check for resource alerts
            if cpu_percent > self.alert_thresholds["cpu_critical"]:
                await self._create_alert(
                    severity=AlertSeverity.CRITICAL,
                    title="High CPU Usage",
                    message=f"CPU usage at {cpu_percent:.1f}%",
                    source="system_monitoring",
                    metadata={"current_value": cpu_percent, "threshold": self.alert_thresholds["cpu_critical"]}
                )

            if memory.percent > self.alert_thresholds["memory_critical"]:
                await self._create_alert(
                    severity=AlertSeverity.CRITICAL,
                    title="High Memory Usage",
                    message=f"Memory usage at {memory.percent:.1f}%",
                    source="system_monitoring",
                    metadata={"current_value": memory.percent, "threshold": self.alert_thresholds["memory_critical"]}
                )

        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            logger.error(f"Error collecting resource metrics: {e}")

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all registered agents"""
        try:
            # This would integrate with actual agent health checks
            # For now, we'll perform basic system health checks
            timestamp = datetime.utcnow()

            # Check if we can access basic services
            try:
                # Simple health check - can we import required modules
                from src.agents.base_agent import BaseAgent

                # Update system health status
                health_status = HealthStatus(
                    agent_name="system",
                    is_healthy=True,
                    last_check=timestamp,
                    response_time_ms=50.0,  # Mock response time
                    uptime_seconds=time.time()
                )
                self.update_health_status("system", health_status)

            except ImportError as e:
                health_status = HealthStatus(
                    agent_name="system",
                    is_healthy=False,
                    last_check=timestamp,
                    response_time_ms=0.0,
                    error_message=str(e)
                )
                self.update_health_status("system", health_status)

                await self._create_alert(
                    severity=AlertSeverity.ERROR,
                    title="System Health Check Failed",
                    message=f"Cannot import required modules: {e}",
                    source="health_monitor"
                )

        except Exception as e:
            logger.error(f"Error performing health checks: {e}")

    async def _create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create a monitoring alert"""
        alert = Alert(
            alert_id=str(int(time.time() * 1000)),  # Simple ID based on timestamp
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.utcnow(),
            source=source,
            metadata=metadata or {}
        )

        self.alerts.append(alert)
        logger.warning(f"Alert created: {severity.upper()} - {title}")

        # Keep only recent alerts (last 1000)
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]

    def add_performance_metric(self, metric: PerformanceMetric) -> None:
        """Add a performance metric to storage"""
        key = f"{metric.metric_name}_{metric.agent_name or 'system'}"
        self.performance_metrics[key].append(metric)

    def update_health_status(self, agent_name: str, health_status: HealthStatus) -> None:
        """Update health status for an agent"""
        self.health_status[agent_name] = health_status

    def get_performance_metrics(
        self,
        metric_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[PerformanceMetric]:
        """Get performance metrics with filtering"""
        all_metrics = []

        for key, metrics in self.performance_metrics.items():
            for metric in metrics:
                if metric_name and metric.metric_name != metric_name:
                    continue
                if agent_name and metric.agent_name != agent_name:
                    continue
                if start_time and metric.timestamp < start_time:
                    continue
                if end_time and metric.timestamp > end_time:
                    continue

                all_metrics.append(metric)

        # Sort by timestamp and limit
        all_metrics.sort(key=lambda m: m.timestamp, reverse=True)
        return all_metrics[:limit]

    def get_health_status(self, agent_name: Optional[str] = None) -> Dict[str, HealthStatus]:
        """Get health status for agents"""
        if agent_name:
            return {agent_name: self.health_status.get(agent_name)}
        return self.health_status.copy()

    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        source: Optional[str] = None,
        resolved: Optional[bool] = None,
        limit: int = 100
    ) -> List[Alert]:
        """Get alerts with filtering"""
        filtered_alerts = []

        for alert in self.alerts:
            if severity and alert.severity != severity:
                continue
            if source and alert.source != source:
                continue
            if resolved is not None:
                is_resolved = alert.resolved_at is not None
                if is_resolved != resolved:
                    continue

            filtered_alerts.append(alert)

        # Sort by timestamp and limit
        filtered_alerts.sort(key=lambda a: a.timestamp, reverse=True)
        return filtered_alerts[:limit]

    def get_agency_summary(self) -> Dict[str, Any]:
        """Get overall agency monitoring summary"""
        timestamp = datetime.utcnow()

        # Count health status
        healthy_count = sum(1 for health in self.health_status.values() if health.is_healthy)
        total_health = len(self.health_status)

        # Count alerts by severity
        alert_counts = defaultdict(int)
        unresolved_alerts = 0
        for alert in self.alerts:
            if alert.resolved_at is None:
                unresolved_alerts += 1
            alert_counts[alert.severity.value] += 1

        # Get latest metrics summary
        latest_metrics = {}
        for key, metrics in self.performance_metrics.items():
            if metrics:
                latest = metrics[-1]
                latest_metrics[key] = {
                    "value": latest.value,
                    "unit": latest.unit,
                    "timestamp": latest.timestamp.isoformat()
                }

        return {
            "timestamp": timestamp.isoformat(),
            "monitoring_status": "active" if self.monitoring_enabled else "inactive",
            "health_summary": {
                "healthy_agents": healthy_count,
                "total_agents": total_health,
                "health_percentage": (healthy_count / total_health * 100) if total_health > 0 else 0
            },
            "alert_summary": {
                "total_alerts": len(self.alerts),
                "unresolved_alerts": unresolved_alerts,
                "by_severity": dict(alert_counts)
            },
            "latest_metrics": latest_metrics,
            "collection_intervals": {
                "metrics_seconds": self.collection_interval,
                "health_check_seconds": self.health_check_interval
            }
        }


# Global monitoring instance
_agency_monitor = AgencyMonitor()


def register_agency_monitoring_tools(mcp: FastMCP):
    """Register all agency monitoring tools with the MCP server."""

    @mcp.tool()
    async def archon_get_workflow_status(
        ctx: Context,
        workflow_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        include_details: bool = True
    ) -> str:
        """
        Get real-time workflow status and monitoring information.

        Args:
            workflow_id: Optional filter by specific workflow ID
            execution_id: Optional filter by specific execution ID
            include_details: Whether to include detailed execution information

        Returns:
            JSON string with workflow status information
        """
        try:
            # Start monitoring if not already running
            if not _agency_monitor.monitoring_enabled:
                await _agency_monitor.start_monitoring()

            # Get workflow executions from workflow manager
            try:
                from ..agency_workflow_tools import _workflow_manager
                executions = _workflow_manager.get_workflow_executions(workflow_id)

                if execution_id:
                    executions = [ex for ex in executions if ex["execution_id"] == execution_id]
            except ImportError:
                executions = []

            # Process executions
            workflow_status = []
            for execution in executions:
                status_info = {
                    "execution_id": execution["execution_id"],
                    "workflow_id": execution["workflow_id"],
                    "status": execution["status"],
                    "progress": execution.get("progress", 0),
                    "started_at": execution.get("started_at"),
                    "completed_at": execution.get("completed_at"),
                    "error_message": execution.get("error_message")
                }

                if include_details:
                    # Add performance metrics for this execution
                    metrics = _agency_monitor.get_performance_metrics(
                        execution_id=execution["execution_id"],
                        limit=10
                    )
                    status_info["performance_metrics"] = [
                        {
                            "metric_name": m.metric_name,
                            "value": m.value,
                            "unit": m.unit,
                            "timestamp": m.timestamp.isoformat()
                        }
                        for m in metrics
                    ]

                workflow_status.append(status_info)

            result = {
                "success": True,
                "workflow_status": workflow_status,
                "total_executions": len(workflow_status),
                "monitoring_active": _agency_monitor.monitoring_enabled,
                "queried_at": datetime.utcnow().isoformat()
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error getting workflow status: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_get_performance_metrics(
        ctx: Context,
        metric_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        time_range_hours: int = 1,
        limit: int = 100
    ) -> str:
        """
        Get performance metrics for agencies and workflows.

        Args:
            metric_name: Optional filter by specific metric name
            agent_name: Optional filter by agent name
            time_range_hours: Time range in hours for metrics (default: 1 hour)
            limit: Maximum number of metrics to return

        Returns:
            JSON string with performance metrics
        """
        try:
            # Calculate time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=time_range_hours)

            # Get metrics
            metrics = _agency_monitor.get_performance_metrics(
                metric_name=metric_name,
                agent_name=agent_name,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )

            # Format metrics for response
            formatted_metrics = []
            for metric in metrics:
                formatted_metric = {
                    "timestamp": metric.timestamp.isoformat(),
                    "metric_name": metric.metric_name,
                    "value": metric.value,
                    "unit": metric.unit,
                    "agent_name": metric.agent_name,
                    "workflow_id": metric.workflow_id,
                    "execution_id": metric.execution_id
                }
                if metric.metadata:
                    formatted_metric["metadata"] = metric.metadata

                formatted_metrics.append(formatted_metric)

            # Calculate summary statistics
            if metrics:
                values = [m.value for m in metrics]
                summary = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "average": sum(values) / len(values),
                    "latest": values[-1] if values else None
                }
            else:
                summary = {"count": 0}

            result = {
                "success": True,
                "metrics": formatted_metrics,
                "summary": summary,
                "filters": {
                    "metric_name": metric_name,
                    "agent_name": agent_name,
                    "time_range_hours": time_range_hours
                },
                "queried_at": datetime.utcnow().isoformat()
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_get_agent_health(
        ctx: Context,
        agent_name: Optional[str] = None,
        include_history: bool = False
    ) -> str:
        """
        Get health status information for agents.

        Args:
            agent_name: Optional filter by specific agent name
            include_history: Whether to include health status history

        Returns:
            JSON string with agent health information
        """
        try:
            # Ensure monitoring is running
            if not _agency_monitor.monitoring_enabled:
                await _agency_monitor.start_monitoring()

            # Get health status
            health_status = _agency_monitor.get_health_status(agent_name)

            # Format health status
            formatted_health = {}
            for name, health in health_status.items():
                health_info = {
                    "agent_name": health.agent_name,
                    "is_healthy": health.is_healthy,
                    "last_check": health.last_check.isoformat(),
                    "response_time_ms": health.response_time_ms,
                    "uptime_seconds": health.uptime_seconds
                }
                if health.error_message:
                    health_info["error_message"] = health.error_message

                formatted_health[name] = health_info

            result = {
                "success": True,
                "agent_health": formatted_health,
                "total_agents": len(formatted_health),
                "healthy_agents": sum(1 for h in formatted_health.values() if h["is_healthy"]),
                "queried_at": datetime.utcnow().isoformat()
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error getting agent health: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_get_monitoring_alerts(
        ctx: Context,
        severity: Optional[str] = None,
        source: Optional[str] = None,
        resolved: Optional[bool] = None,
        limit: int = 50
    ) -> str:
        """
        Get monitoring alerts and notifications.

        Args:
            severity: Optional filter by alert severity (info, warning, error, critical)
            source: Optional filter by alert source
            resolved: Optional filter by resolved status
            limit: Maximum number of alerts to return

        Returns:
            JSON string with monitoring alerts
        """
        try:
            # Parse severity filter
            severity_enum = None
            if severity:
                try:
                    severity_enum = AlertSeverity(severity.lower())
                except ValueError:
                    return json.dumps({
                        "success": False,
                        "error": f"Invalid severity: {severity}. Valid values: {', '.join(s.value for s in AlertSeverity)}"
                    }, indent=2)

            # Get alerts
            alerts = _agency_monitor.get_alerts(
                severity=severity_enum,
                source=source,
                resolved=resolved,
                limit=limit
            )

            # Format alerts
            formatted_alerts = []
            for alert in alerts:
                alert_info = {
                    "alert_id": alert.alert_id,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "source": alert.source,
                    "is_resolved": alert.resolved_at is not None
                }
                if alert.resolved_at:
                    alert_info["resolved_at"] = alert.resolved_at.isoformat()
                if alert.metadata:
                    alert_info["metadata"] = alert.metadata

                formatted_alerts.append(alert_info)

            result = {
                "success": True,
                "alerts": formatted_alerts,
                "total_alerts": len(formatted_alerts),
                "filters": {
                    "severity": severity,
                    "source": source,
                    "resolved": resolved
                },
                "queried_at": datetime.utcnow().isoformat()
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error getting monitoring alerts: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_get_agency_summary(
        ctx: Context,
        include_health: bool = True,
        include_metrics: bool = True,
        include_alerts: bool = True
    ) -> str:
        """
        Get comprehensive agency monitoring summary.

        Args:
            include_health: Whether to include health summary
            include_metrics: Whether to include metrics summary
            include_alerts: Whether to include alerts summary

        Returns:
            JSON string with agency monitoring summary
        """
        try:
            # Get summary
            summary = _agency_monitor.get_agency_summary()

            # Customize summary based on parameters
            if not include_health:
                summary.pop("health_summary", None)
            if not include_metrics:
                summary.pop("latest_metrics", None)
            if not include_alerts:
                summary.pop("alert_summary", None)

            result = {
                "success": True,
                "agency_summary": summary,
                "queried_at": datetime.utcnow().isoformat()
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error getting agency summary: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_configure_monitoring(
        ctx: Context,
        enabled: bool,
        collection_interval: Optional[int] = None,
        health_check_interval: Optional[int] = None,
        alert_thresholds: Optional[str] = None
    ) -> str:
        """
        Configure monitoring settings and thresholds.

        Args:
            enabled: Whether monitoring should be enabled
            collection_interval: Metrics collection interval in seconds
            health_check_interval: Health check interval in seconds
            alert_thresholds: JSON string with alert threshold configuration

        Returns:
            JSON string with configuration result
        """
        try:
            # Update monitoring enabled state
            _agency_monitor.monitoring_enabled = enabled

            # Update intervals if provided
            if collection_interval is not None:
                _agency_monitor.collection_interval = max(10, collection_interval)  # Minimum 10 seconds

            if health_check_interval is not None:
                _agency_monitor.health_check_interval = max(30, health_check_interval)  # Minimum 30 seconds

            # Update alert thresholds if provided
            if alert_thresholds:
                thresholds = json.loads(alert_thresholds)
                _agency_monitor.alert_thresholds.update(thresholds)

            # Start or stop monitoring based on enabled state
            if enabled:
                await _agency_monitor.start_monitoring()
            else:
                await _agency_monitor.stop_monitoring()

            result = {
                "success": True,
                "monitoring_enabled": enabled,
                "collection_interval_seconds": _agency_monitor.collection_interval,
                "health_check_interval_seconds": _agency_monitor.health_check_interval,
                "alert_thresholds": _agency_monitor.alert_thresholds,
                "configured_at": datetime.utcnow().isoformat()
            }

            mcp_logger.info(f"Monitoring configured: enabled={enabled}")

            return json.dumps(result, indent=2)

        except json.JSONDecodeError as e:
            return json.dumps({
                "success": False,
                "error": f"Invalid JSON in alert_thresholds: {str(e)}"
            }, indent=2)
        except Exception as e:
            logger.error(f"Error configuring monitoring: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_get_resource_utilization(
        ctx: Context,
        agent_name: Optional[str] = None,
        time_range_minutes: int = 30
    ) -> str:
        """
        Get resource utilization metrics for agents and system.

        Args:
            agent_name: Optional filter by specific agent name
            time_range_minutes: Time range in minutes for resource data

        Returns:
            JSON string with resource utilization information
        """
        try:
            # Get resource metrics
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=time_range_minutes)

            # Get CPU metrics
            cpu_metrics = _agency_monitor.get_performance_metrics(
                metric_name="system_cpu_percent",
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )

            # Get memory metrics
            memory_metrics = _agency_monitor.get_performance_metrics(
                metric_name="system_memory_percent",
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )

            # Get disk metrics
            disk_metrics = _agency_monitor.get_performance_metrics(
                metric_name="system_disk_percent",
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )

            # Calculate resource summaries
            def calculate_summary(metrics):
                if not metrics:
                    return {"count": 0}

                values = [m.value for m in metrics]
                return {
                    "count": len(values),
                    "current": values[-1] if values else None,
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }

            resource_data = {
                "time_range_minutes": time_range_minutes,
                "cpu": calculate_summary(cpu_metrics),
                "memory": calculate_summary(memory_metrics),
                "disk": calculate_summary(disk_metrics),
                "queried_at": datetime.utcnow().isoformat()
            }

            result = {
                "success": True,
                "resource_utilization": resource_data
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error getting resource utilization: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    # Log successful registration
    logger.info("✓ Agency monitoring tools registered")