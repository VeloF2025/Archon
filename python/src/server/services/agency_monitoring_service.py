"""
Agency Monitoring Service - Real-time monitoring and analytics for agency workflows.

This service provides comprehensive monitoring capabilities for agency workflows,
including performance metrics, health checks, alerting, and resource utilization tracking.
"""

import asyncio
import psutil
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
import logging
from collections import defaultdict, deque
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
import statistics

from ...database.models import (
    AgencyWorkflow,
    WorkflowExecution,
    AgentCapability,
    MonitoringAlert,
    ResourceUtilization
)
from get_config import get_config

logger = logging.getLogger(__name__)
settings = get_config()


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Alert types."""
    WORKFLOW_FAILURE = "workflow_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    AGENT_UNRESPONSIVE = "agent_unresponsive"
    COMMUNICATION_FAILURE = "communication_failure"
    SECURITY_ALERT = "security_alert"


class AgentHealthStatus(str, Enum):
    """Agent health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"


class MonitoringConfig(BaseModel):
    """Configuration for monitoring settings."""
    enable_performance_monitoring: bool = True
    enable_resource_monitoring: bool = True
    enable_alerting: bool = True
    alert_thresholds: Dict[str, float] = Field(default_factory=dict)
    data_retention_days: int = 30
    monitoring_interval_seconds: int = 30


class PerformanceMetrics(BaseModel):
    """Performance metrics for workflows and agents."""
    execution_time_ms: float
    success_rate: float
    throughput: float
    error_rate: float
    average_response_time: float
    p95_response_time: float
    p99_response_time: float


class ResourceMetrics(BaseModel):
    """Resource utilization metrics."""
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_io_bytes: Dict[str, int]
    active_connections: int
    queue_length: int


class AgentHealth(BaseModel):
    """Agent health information."""
    agent_id: str
    status: AgentHealthStatus
    last_seen: datetime
    response_time_ms: float
    error_count: int
    active_tasks: int
    capabilities: List[str]
    metadata: Dict[str, Any]


class AgencyAlert(BaseModel):
    """Alert information."""
    id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any]
    resolved: bool
    resolved_at: Optional[datetime]


class AgencyMonitoringService:
    """Service for monitoring agency workflows and agent health."""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.config = MonitoringConfig()
        self._metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self._alerts = deque(maxlen=1000)
        self._monitoring_task = None
        self._agent_health_cache = {}
        self._last_alert_check = datetime.utcnow()

    async def start_monitoring(self) -> None:
        """Start the background monitoring task."""
        if self._monitoring_task and not self._monitoring_task.done():
            return

        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Agency monitoring service started")

    async def stop_monitoring(self) -> None:
        """Stop the background monitoring task."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Agency monitoring service stopped")

    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get detailed workflow status with metrics."""
        try:
            # Get workflow information
            stmt = select(AgencyWorkflow).where(AgencyWorkflow.id == workflow_id)
            result = await self.db.execute(stmt)
            workflow = result.scalar_one_or_none()

            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")

            # Get recent executions
            time_threshold = datetime.utcnow() - timedelta(hours=24)
            stmt = select(WorkflowExecution).where(
                and_(
                    WorkflowExecution.workflow_id == workflow_id,
                    WorkflowExecution.created_at >= time_threshold
                )
            ).order_by(WorkflowExecution.created_at.desc())

            result = await self.db.execute(stmt)
            executions = result.scalars().all()

            # Calculate metrics
            performance_metrics = self._calculate_performance_metrics(executions)

            return {
                "workflow": workflow,
                "status": workflow.status,
                "recent_executions": executions[:10],
                "performance_metrics": performance_metrics,
                "agent_status": await self._get_workflow_agent_status(workflow.agent_ids),
                "alerts": await self._get_workflow_alerts(workflow_id)
            }

        except Exception as e:
            logger.error(f"Error getting workflow status {workflow_id}: {e}")
            raise

    async def get_performance_metrics(self,
                                     workflow_id: Optional[str] = None,
                                     time_range_hours: int = 24) -> PerformanceMetrics:
        """Get performance metrics for workflows."""
        try:
            time_threshold = datetime.utcnow() - timedelta(hours=time_range_hours)

            # Query executions
            stmt = select(WorkflowExecution)
            if workflow_id:
                stmt = stmt.where(WorkflowExecution.workflow_id == workflow_id)

            stmt = stmt.where(WorkflowExecution.created_at >= time_threshold)
            result = await self.db.execute(stmt)
            executions = result.scalars().all()

            return self._calculate_performance_metrics(executions)

        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            raise

    async def get_agent_health(self, agent_ids: Optional[List[str]] = None) -> List[AgentHealth]:
        """Get health status for agents."""
        try:
            # Update health cache
            await self._update_agent_health_cache()

            if agent_ids:
                return [self._agent_health_cache.get(agent_id) for agent_id in agent_ids
                       if agent_id in self._agent_health_cache]

            return list(self._agent_health_cache.values())

        except Exception as e:
            logger.error(f"Error getting agent health: {e}")
            raise

    async def get_monitoring_alerts(self,
                                 alert_type: Optional[AlertType] = None,
                                 severity: Optional[AlertSeverity] = None,
                                 resolved: Optional[bool] = None,
                                 limit: int = 100) -> List[AgencyAlert]:
        """Get monitoring alerts with filtering."""
        try:
            stmt = select(MonitoringAlert)

            if alert_type:
                stmt = stmt.where(MonitoringAlert.alert_type == alert_type.value)

            if severity:
                stmt = stmt.where(MonitoringAlert.severity == severity.value)

            if resolved is not None:
                stmt = stmt.where(MonitoringAlert.resolved == resolved)

            stmt = stmt.order_by(MonitoringAlert.created_at.desc())
            stmt = stmt.limit(limit)

            result = await self.db.execute(stmt)
            alerts = result.scalars().all()

            return [
                AgencyAlert(
                    id=alert.id,
                    alert_type=AlertType(alert.alert_type),
                    severity=AlertSeverity(alert.severity),
                    title=alert.title,
                    description=alert.description,
                    timestamp=alert.created_at,
                    source=alert.source,
                    metadata=alert.metadata or {},
                    resolved=alert.resolved,
                    resolved_at=alert.resolved_at
                )
                for alert in alerts
            ]

        except Exception as e:
            logger.error(f"Error getting monitoring alerts: {e}")
            raise

    async def get_agency_summary(self) -> Dict[str, Any]:
        """Get comprehensive agency summary."""
        try:
            # Get workflow statistics
            stmt = select(AgencyWorkflow)
            result = await self.db.execute(stmt)
            workflows = result.scalars().all()

            total_workflows = len(workflows)
            active_workflows = sum(1 for w in workflows if w.status == "running")

            # Get execution statistics
            time_threshold = datetime.utcnow() - timedelta(hours=24)
            stmt = select(WorkflowExecution).where(
                WorkflowExecution.created_at >= time_threshold
            )
            result = await self.db.execute(stmt)
            executions = result.scalars().all()

            total_executions = len(executions)
            successful_executions = sum(1 for e in executions if e.status == "completed")
            failed_executions = sum(1 for e in executions if e.status == "failed")

            # Get agent statistics
            stmt = select(AgentCapability).distinct(AgentCapability.agent_id)
            result = await self.db.execute(stmt)
            agent_ids = [row[0] for row in result.all()]
            agent_health = await self.get_agent_health(agent_ids)

            healthy_agents = sum(1 for a in agent_health if a.status == AgentHealthStatus.HEALTHY)

            # Get alert statistics
            stmt = select(MonitoringAlert).where(
                MonitoringAlert.created_at >= time_threshold
            )
            result = await self.db.execute(stmt)
            alerts = result.scalars().all()

            critical_alerts = sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL.value)

            return {
                "total_workflows": total_workflows,
                "active_workflows": active_workflows,
                "total_executions_24h": total_executions,
                "successful_executions_24h": successful_executions,
                "failed_executions_24h": failed_executions,
                "success_rate_24h": (successful_executions / total_executions * 100) if total_executions > 0 else 0,
                "total_agents": len(agent_ids),
                "healthy_agents": healthy_agents,
                "critical_alerts_24h": critical_alerts,
                "performance_metrics": await self.get_performance_metrics(),
                "resource_metrics": await self.get_resource_utilization(),
                "system_health": self._calculate_system_health(healthy_agents, len(agent_ids), critical_alerts)
            }

        except Exception as e:
            logger.error(f"Error getting agency summary: {e}")
            raise

    async def configure_monitoring(self, config: MonitoringConfig) -> None:
        """Configure monitoring settings."""
        try:
            self.config = config
            logger.info("Monitoring configuration updated")

            # Restart monitoring if needed
            if self._monitoring_task and not self._monitoring_task.done():
                await self.stop_monitoring()
                await self.start_monitoring()

        except Exception as e:
            logger.error(f"Error configuring monitoring: {e}")
            raise

    async def get_resource_utilization(self) -> ResourceMetrics:
        """Get current system resource utilization."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()

            return ResourceMetrics(
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent,
                disk_usage_percent=disk.percent,
                network_io_bytes={
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv
                },
                active_connections=len(self._active_executions),
                queue_length=len(asyncio.all_tasks())
            )

        except Exception as e:
            logger.error(f"Error getting resource utilization: {e}")
            raise

    async def create_alert(self,
                         alert_type: AlertType,
                         severity: AlertSeverity,
                         title: str,
                         description: str,
                         source: str,
                         metadata: Optional[Dict[str, Any]] = None) -> AgencyAlert:
        """Create a monitoring alert."""
        try:
            alert_id = str(uuid.uuid4())

            # Store in database
            alert = MonitoringAlert(
                id=alert_id,
                alert_type=alert_type.value,
                severity=severity.value,
                title=title,
                description=description,
                source=source,
                metadata=metadata or {},
                created_at=datetime.utcnow(),
                resolved=False
            )

            self.db.add(alert)
            await self.db.commit()

            # Add to cache
            agency_alert = AgencyAlert(
                id=alert_id,
                alert_type=alert_type,
                severity=severity,
                title=title,
                description=description,
                timestamp=datetime.utcnow(),
                source=source,
                metadata=metadata or {},
                resolved=False,
                resolved_at=None
            )

            self._alerts.append(agency_alert)

            logger.warning(f"Created alert: {alert_type} - {title}")
            return agency_alert

        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            raise

    # Background monitoring loop
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                # Collect metrics
                await self._collect_metrics()

                # Check for alerts
                await self._check_alerts()

                # Update agent health
                await self._update_agent_health_cache()

                # Sleep for next iteration
                await asyncio.sleep(self.config.monitoring_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.monitoring_interval_seconds)

    async def _collect_metrics(self) -> None:
        """Collect performance and resource metrics."""
        try:
            # Get resource metrics
            resource_metrics = await self.get_resource_utilization()

            # Store in history
            self._metrics_history['resource'].append({
                'timestamp': datetime.utcnow(),
                'metrics': resource_metrics.dict()
            })

            # Check for resource alerts
            await self._check_resource_alerts(resource_metrics)

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")

    async def _check_alerts(self) -> None:
        """Check for alert conditions."""
        try:
            # Check for workflow failures
            await self._check_workflow_failures()

            # Check for performance degradation
            await self._check_performance_degradation()

            # Check for agent health issues
            await self._check_agent_health()

        except Exception as e:
            logger.error(f"Error checking alerts: {e}")

    async def _update_agent_health_cache(self) -> None:
        """Update agent health cache."""
        try:
            # This would integrate with agent registry to get real-time health
            # For now, update with basic health information
            pass

        except Exception as e:
            logger.error(f"Error updating agent health cache: {e}")

    def _calculate_performance_metrics(self, executions: List[WorkflowExecution]) -> PerformanceMetrics:
        """Calculate performance metrics from executions."""
        if not executions:
            return PerformanceMetrics(
                execution_time_ms=0.0,
                success_rate=0.0,
                throughput=0.0,
                error_rate=0.0,
                average_response_time=0.0,
                p95_response_time=0.0,
                p99_response_time=0.0
            )

        successful_executions = [e for e in executions if e.status == "completed"]
        failed_executions = [e for e in executions if e.status == "failed"]

        execution_times = []
        for execution in executions:
            if execution.end_time:
                execution_time = (execution.end_time - execution.start_time).total_seconds() * 1000
                execution_times.append(execution_time)

        return PerformanceMetrics(
            execution_time_ms=statistics.mean(execution_times) if execution_times else 0.0,
            success_rate=len(successful_executions) / len(executions) * 100,
            throughput=len(executions) / 24,  # executions per hour
            error_rate=len(failed_executions) / len(executions) * 100,
            average_response_time=statistics.mean(execution_times) if execution_times else 0.0,
            p95_response_time=statistics.quantiles(execution_times, n=20)[18] if len(execution_times) > 20 else 0.0,
            p99_response_time=statistics.quantiles(execution_times, n=100)[98] if len(execution_times) > 100 else 0.0
        )

    async def _get_workflow_agent_status(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Get status for workflow agents."""
        try:
            agent_health = await self.get_agent_health(agent_ids)
            return {agent.agent_id: agent.dict() for agent in agent_health}

        except Exception as e:
            logger.error(f"Error getting workflow agent status: {e}")
            return {}

    async def _get_workflow_alerts(self, workflow_id: str) -> List[AgencyAlert]:
        """Get alerts for a specific workflow."""
        try:
            stmt = select(MonitoringAlert).where(
                MonitoringAlert.metadata['workflow_id'].astext == workflow_id
            ).order_by(MonitoringAlert.created_at.desc()).limit(10)

            result = await self.db.execute(stmt)
            alerts = result.scalars().all()

            return [
                AgencyAlert(
                    id=alert.id,
                    alert_type=AlertType(alert.alert_type),
                    severity=AlertSeverity(alert.severity),
                    title=alert.title,
                    description=alert.description,
                    timestamp=alert.created_at,
                    source=alert.source,
                    metadata=alert.metadata or {},
                    resolved=alert.resolved,
                    resolved_at=alert.resolved_at
                )
                for alert in alerts
            ]

        except Exception as e:
            logger.error(f"Error getting workflow alerts: {e}")
            return []

    def _calculate_system_health(self, healthy_agents: int, total_agents: int, critical_alerts: int) -> str:
        """Calculate overall system health."""
        if critical_alerts > 0:
            return "critical"
        elif healthy_agents / total_agents < 0.8 if total_agents > 0 else 1:
            return "warning"
        else:
            return "healthy"

    async def _check_resource_alerts(self, metrics: ResourceMetrics) -> None:
        """Check for resource-based alerts."""
        thresholds = self.config.alert_thresholds

        if metrics.cpu_usage_percent > thresholds.get('cpu_warning', 80):
            severity = AlertSeverity.HIGH if metrics.cpu_usage_percent > thresholds.get('cpu_critical', 90) else AlertSeverity.MEDIUM
            await self.create_alert(
                AlertType.RESOURCE_EXHAUSTION,
                severity,
                "High CPU Usage",
                f"CPU usage is {metrics.cpu_usage_percent:.1f}%",
                "system_monitor",
                {"metric": "cpu", "value": metrics.cpu_usage_percent}
            )

        if metrics.memory_usage_percent > thresholds.get('memory_warning', 85):
            severity = AlertSeverity.HIGH if metrics.memory_usage_percent > thresholds.get('memory_critical', 95) else AlertSeverity.MEDIUM
            await self.create_alert(
                AlertType.RESOURCE_EXHAUSTION,
                severity,
                "High Memory Usage",
                f"Memory usage is {metrics.memory_usage_percent:.1f}%",
                "system_monitor",
                {"metric": "memory", "value": metrics.memory_usage_percent}
            )

    async def _check_workflow_failures(self) -> None:
        """Check for workflow execution failures."""
        # Check for recent workflow failures
        time_threshold = datetime.utcnow() - timedelta(minutes=5)
        stmt = select(WorkflowExecution).where(
            and_(
                WorkflowExecution.status == "failed",
                WorkflowExecution.end_time >= time_threshold
            )
        )

        result = await self.db.execute(stmt)
        failed_executions = result.scalars().all()

        for execution in failed_executions:
            await self.create_alert(
                AlertType.WORKFLOW_FAILURE,
                AlertSeverity.HIGH,
                "Workflow Execution Failed",
                f"Workflow {execution.workflow_id} failed: {execution.error_message}",
                "workflow_monitor",
                {"workflow_id": execution.workflow_id, "execution_id": execution.id}
            )

    async def _check_performance_degradation(self) -> None:
        """Check for performance degradation."""
        # Check for performance degradation based on historical metrics
        pass

    async def _check_agent_health(self) -> None:
        """Check for agent health issues."""
        # Check for unresponsive agents
        pass