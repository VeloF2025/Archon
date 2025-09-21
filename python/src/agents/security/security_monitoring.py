"""
Security Monitoring Service
Real-time security monitoring and alerting system
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import threading
from collections import defaultdict, deque
import psutil
import socket
import requests
from concurrent.futures import ThreadPoolExecutor

from .threat_detection import ThreatDetectionSystem, ThreatEvent, ThreatSeverity, ThreatType
from .audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class MonitoringLevel(Enum):
    """Security monitoring levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertPriority(Enum):
    """Alert priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class MonitoringType(Enum):
    """Types of security monitoring"""
    SYSTEM = "system"
    NETWORK = "network"
    APPLICATION = "application"
    DATABASE = "database"
    USER_ACTIVITY = "user_activity"
    COMPLIANCE = "compliance"


@dataclass
class SecurityMetric:
    """Security metric definition"""
    name: str
    description: str
    metric_type: str  # counter, gauge, rate, percentile
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)  # warning, critical


@dataclass
class SecurityAlert:
    """Security alert definition"""
    alert_id: str
    name: str
    description: str
    severity: AlertPriority
    monitoring_type: MonitoringType
    source: str
    timestamp: datetime
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold_triggered: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    escalation_level: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "monitoring_type": self.monitoring_type.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold_triggered": self.threshold_triggered,
            "details": self.details,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved": self.resolved,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "escalation_level": self.escalation_level
        }


@dataclass
class MonitoringRule:
    """Monitoring rule definition"""
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # e.g., "value > threshold"
    threshold: float
    severity: AlertPriority
    monitoring_type: MonitoringType
    enabled: bool = True
    cooldown_period: int = 300  # seconds
    last_triggered: Optional[datetime] = None
    notification_channels: List[str] = field(default_factory=list)
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)


class SecurityMonitor:
    """Real-time security monitoring system"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.monitoring_level = MonitoringLevel(self.config.get('monitoring_level', 'medium'))

        # Core components
        self.threat_detection = ThreatDetectionSystem(config.get('threat_detection', {}))
        self.audit_logger = AuditLogger()

        # Monitoring data
        self.metrics: Dict[str, List[SecurityMetric]] = defaultdict(list)
        self.alerts: deque = deque(maxlen=10000)
        self.active_alerts: Dict[str, SecurityAlert] = {}
        self.monitoring_rules: Dict[str, MonitoringRule] = {}

        # System monitoring
        self.system_metrics_thread: Optional[threading.Thread] = None
        self.network_monitor_thread: Optional[threading.Thread] = None
        self.running = False

        # Alert handling
        self.alert_handlers: Dict[AlertPriority, List[Callable]] = defaultdict(list)
        self.notification_channels: Dict[str, Callable] = {}

        # Metrics tracking
        self.performance_metrics: Dict[str, Any] = defaultdict(int)

        # Initialize monitoring
        self._initialize_monitoring_rules()
        self._initialize_notification_channels()

        # Start monitoring threads
        self.start_monitoring()

    def _initialize_monitoring_rules(self) -> None:
        """Initialize default monitoring rules"""

        # CPU usage monitoring
        cpu_rule = MonitoringRule(
            rule_id="cpu_usage_high",
            name="High CPU Usage",
            description="CPU usage exceeds threshold",
            metric_name="cpu_usage_percent",
            condition="value > threshold",
            threshold=85.0,
            severity=AlertPriority.HIGH,
            monitoring_type=MonitoringType.SYSTEM,
            notification_channels=["email", "slack"]
        )
        self.monitoring_rules[cpu_rule.rule_id] = cpu_rule

        # Memory usage monitoring
        memory_rule = MonitoringRule(
            rule_id="memory_usage_high",
            name="High Memory Usage",
            description="Memory usage exceeds threshold",
            metric_name="memory_usage_percent",
            condition="value > threshold",
            threshold=90.0,
            severity=AlertPriority.HIGH,
            monitoring_type=MonitoringType.SYSTEM,
            notification_channels=["email", "slack"]
        )
        self.monitoring_rules[memory_rule.rule_id] = memory_rule

        # Disk space monitoring
        disk_rule = MonitoringRule(
            rule_id="disk_space_low",
            name="Low Disk Space",
            description="Available disk space below threshold",
            metric_name="disk_available_percent",
            condition="value < threshold",
            threshold=10.0,
            severity=AlertPriority.CRITICAL,
            monitoring_type=MonitoringType.SYSTEM,
            notification_channels=["email", "slack", "pagerduty"]
        )
        self.monitoring_rules[disk_rule.rule_id] = disk_rule

        # Network traffic monitoring
        network_rule = MonitoringRule(
            rule_id="network_traffic_anomaly",
            name="Network Traffic Anomaly",
            description="Unusual network traffic detected",
            metric_name="network_traffic_mbps",
            condition="value > threshold",
            threshold=1000.0,  # 1 Gbps
            severity=AlertPriority.MEDIUM,
            monitoring_type=MonitoringType.NETWORK,
            notification_channels=["email"]
        )
        self.monitoring_rules[network_rule.rule_id] = network_rule

        # Failed login attempts
        auth_rule = MonitoringRule(
            rule_id="failed_logins_high",
            name="High Failed Login Attempts",
            description="Unusual number of failed login attempts",
            metric_name="failed_login_count",
            condition="value > threshold",
            threshold=10.0,
            severity=AlertPriority.HIGH,
            monitoring_type=MonitoringType.USER_ACTIVITY,
            notification_channels=["email", "slack"]
        )
        self.monitoring_rules[auth_rule.rule_id] = auth_rule

        # Threat detection alerts
        threat_rule = MonitoringRule(
            rule_id="threat_detected",
            name="Security Threat Detected",
            description="Threat detection system identified a threat",
            metric_name="threat_count",
            condition="value > threshold",
            threshold=0.0,
            severity=AlertPriority.HIGH,
            monitoring_type=MonitoringType.APPLICATION,
            notification_channels=["email", "slack", "pagerduty"]
        )
        self.monitoring_rules[threat_rule.rule_id] = threat_rule

    def _initialize_notification_channels(self) -> None:
        """Initialize notification channels"""

        # Email notification (placeholder implementation)
        def send_email_notification(alert: SecurityAlert, recipients: List[str]):
            logger.info(f"Email alert sent to {recipients}: {alert.name} - {alert.description}")

        # Slack notification (placeholder implementation)
        def send_slack_notification(alert: SecurityAlert, webhook_url: str):
            logger.info(f"Slack alert sent: {alert.name} - {alert.description}")

        # PagerDuty notification (placeholder implementation)
        def send_pagerduty_notification(alert: SecurityAlert, service_key: str):
            logger.info(f"PagerDuty alert sent: {alert.name} - {alert.description}")

        self.notification_channels = {
            "email": send_email_notification,
            "slack": send_slack_notification,
            "pagerduty": send_pagerduty_notification
        }

    def start_monitoring(self) -> None:
        """Start security monitoring threads"""
        if self.running:
            return

        self.running = True

        # Start system metrics monitoring
        self.system_metrics_thread = threading.Thread(
            target=self._monitor_system_metrics,
            daemon=True
        )
        self.system_metrics_thread.start()

        # Start network monitoring
        self.network_monitor_thread = threading.Thread(
            target=self._monitor_network_traffic,
            daemon=True
        )
        self.network_monitor_thread.start()

        # Start threat detection monitoring
        asyncio.create_task(self._monitor_threats())

        # Start alert processing
        asyncio.create_task(self._process_alerts())

        logger.info("Security monitoring started")

    def stop_monitoring(self) -> None:
        """Stop security monitoring threads"""
        self.running = False

        if self.system_metrics_thread:
            self.system_metrics_thread.join(timeout=5)
        if self.network_monitor_thread:
            self.network_monitor_thread.join(timeout=5)

        logger.info("Security monitoring stopped")

    def _monitor_system_metrics(self) -> None:
        """Monitor system metrics"""
        while self.running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.record_metric(
                    name="cpu_usage_percent",
                    value=cpu_percent,
                    unit="percent",
                    metric_type="gauge",
                    tags={"host": socket.gethostname()}
                )

                # Memory usage
                memory = psutil.virtual_memory()
                self.record_metric(
                    name="memory_usage_percent",
                    value=memory.percent,
                    unit="percent",
                    metric_type="gauge",
                    tags={"host": socket.gethostname()}
                )

                # Disk usage
                disk = psutil.disk_usage('/')
                disk_available_percent = (disk.free / disk.total) * 100
                self.record_metric(
                    name="disk_available_percent",
                    value=disk_available_percent,
                    unit="percent",
                    metric_type="gauge",
                    tags={"host": socket.gethostname(), "mount": "/"}
                )

                # Process count
                process_count = len(psutil.pids())
                self.record_metric(
                    name="process_count",
                    value=process_count,
                    unit="count",
                    metric_type="gauge",
                    tags={"host": socket.gethostname()}
                )

                # Network I/O
                net_io = psutil.net_io_counters()
                self.record_metric(
                    name="network_bytes_sent",
                    value=net_io.bytes_sent,
                    unit="bytes",
                    metric_type="counter",
                    tags={"host": socket.gethostname()}
                )
                self.record_metric(
                    name="network_bytes_received",
                    value=net_io.bytes_recv,
                    unit="bytes",
                    metric_type="counter",
                    tags={"host": socket.gethostname()}
                )

            except Exception as e:
                logger.error(f"Error monitoring system metrics: {str(e)}")

            time.sleep(5)  # Monitor every 5 seconds

    def _monitor_network_traffic(self) -> None:
        """Monitor network traffic"""
        last_net_io = None

        while self.running:
            try:
                current_net_io = psutil.net_io_counters()

                if last_net_io:
                    # Calculate network traffic rate
                    time_diff = time.time() - last_net_io.get('timestamp', time.time())
                    if time_diff > 0:
                        bytes_sent_diff = current_net_io.bytes_sent - last_net_io['bytes_sent']
                        bytes_recv_diff = current_net_io.bytes_recv - last_net_io['bytes_recv']

                        send_rate_mbps = (bytes_sent_diff * 8) / (time_diff * 1e6)
                        recv_rate_mbps = (bytes_recv_diff * 8) / (time_diff * 1e6)

                        self.record_metric(
                            name="network_send_rate_mbps",
                            value=send_rate_mbps,
                            unit="mbps",
                            metric_type="gauge",
                            tags={"host": socket.gethostname()}
                        )
                        self.record_metric(
                            name="network_receive_rate_mbps",
                            value=recv_rate_mbps,
                            unit="mbps",
                            metric_type="gauge",
                            tags={"host": socket.gethostname()}
                        )

                last_net_io = {
                    'bytes_sent': current_net_io.bytes_sent,
                    'bytes_recv': current_net_io.bytes_recv,
                    'timestamp': time.time()
                }

            except Exception as e:
                logger.error(f"Error monitoring network traffic: {str(e)}")

            time.sleep(10)  # Monitor every 10 seconds

    async def _monitor_threats(self) -> None:
        """Monitor for security threats"""
        while self.running:
            try:
                # Get recent threat events
                recent_threats = self.threat_detection.get_recent_threats(hours=1)

                if recent_threats:
                    # Record threat metric
                    self.record_metric(
                        name="threat_count",
                        value=len(recent_threats),
                        unit="count",
                        metric_type="counter",
                        tags={"severity": "recent"}
                    )

                    # Process each threat
                    for threat_data in recent_threats:
                        await self._process_threat_alert(threat_data)

                # Check for high-severity active threats
                active_threats = self.threat_detection.active_threats
                if active_threats:
                    high_severity_threats = [
                        threat for threat in active_threats.values()
                        if threat.severity.value >= ThreatSeverity.HIGH.value
                    ]

                    if high_severity_threats:
                        self.record_metric(
                            name="active_high_severity_threats",
                            value=len(high_severity_threats),
                            unit="count",
                            metric_type="gauge"
                        )

            except Exception as e:
                logger.error(f"Error monitoring threats: {str(e)}")

            await asyncio.sleep(30)  # Check every 30 seconds

    async def _process_threat_alert(self, threat_data: Dict[str, Any]) -> None:
        """Process threat alert"""
        alert = SecurityAlert(
            alert_id=str(uuid.uuid4()),
            name=f"Security Threat: {threat_data.get('threat_type', 'Unknown')}",
            description=threat_data.get('description', 'Security threat detected'),
            severity=self._map_threat_severity_to_alert_priority(threat_data.get('severity', 'medium')),
            monitoring_type=MonitoringType.APPLICATION,
            source="threat_detection",
            timestamp=datetime.now(),
            metric_name="threat_detected",
            metric_value=1.0,
            details=threat_data
        )

        self.create_alert(alert)

    def _map_threat_severity_to_alert_priority(self, threat_severity: str) -> AlertPriority:
        """Map threat severity to alert priority"""
        mapping = {
            'critical': AlertPriority.EMERGENCY,
            'high': AlertPriority.HIGH,
            'medium': AlertPriority.MEDIUM,
            'low': AlertPriority.LOW,
            'info': AlertPriority.LOW
        }
        return mapping.get(threat_severity.lower(), AlertPriority.MEDIUM)

    def record_metric(self, name: str, value: float, unit: str, metric_type: str,
                     tags: Dict[str, str] = None, thresholds: Dict[str, float] = None) -> None:
        """Record a security metric"""
        metric = SecurityMetric(
            name=name,
            description=f"Security metric: {name}",
            metric_type=metric_type,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            tags=tags or {},
            thresholds=thresholds or {}
        )

        self.metrics[name].append(metric)

        # Keep only recent metrics (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.metrics[name] = [
            m for m in self.metrics[name] if m.timestamp > cutoff_time
        ]

        # Check monitoring rules
        self._check_monitoring_rules(metric)

    def _check_monitoring_rules(self, metric: SecurityMetric) -> None:
        """Check if metric triggers any monitoring rules"""
        for rule in self.monitoring_rules.values():
            if not rule.enabled:
                continue

            if rule.metric_name != metric.name:
                continue

            # Check cooldown period
            if (rule.last_triggered and
                (datetime.now() - rule.last_triggered).seconds < rule.cooldown_period):
                continue

            # Evaluate condition
            if self._evaluate_condition(rule.condition, metric.value, rule.threshold):
                # Create alert
                alert = SecurityAlert(
                    alert_id=str(uuid.uuid4()),
                    name=rule.name,
                    description=rule.description,
                    severity=rule.severity,
                    monitoring_type=rule.monitoring_type,
                    source="monitoring_rule",
                    timestamp=datetime.now(),
                    metric_name=metric.name,
                    metric_value=metric.value,
                    threshold_triggered=rule.threshold,
                    details={
                        "rule_id": rule.rule_id,
                        "condition": rule.condition,
                        "metric_tags": metric.tags
                    }
                )

                self.create_alert(alert)
                rule.last_triggered = datetime.now()

    def _evaluate_condition(self, condition: str, value: float, threshold: float) -> bool:
        """Evaluate monitoring condition"""
        try:
            # Simple condition evaluation
            if condition == "value > threshold":
                return value > threshold
            elif condition == "value < threshold":
                return value < threshold
            elif condition == "value == threshold":
                return value == threshold
            elif condition == "value >= threshold":
                return value >= threshold
            elif condition == "value <= threshold":
                return value <= threshold
            else:
                return False
        except Exception:
            return False

    def create_alert(self, alert: SecurityAlert) -> None:
        """Create and process security alert"""
        # Store alert
        self.alerts.append(alert)

        # Add to active alerts if not acknowledged/resolved
        if not alert.acknowledged and not alert.resolved:
            self.active_alerts[alert.alert_id] = alert

        # Log alert
        self.audit_logger.log_security_event(
            event_type="security_alert",
            description=f"Alert created: {alert.name}",
            severity=alert.severity.value,
            details=alert.to_dict()
        )

        # Process alert
        asyncio.create_task(self._process_alert(alert))

    async def _process_alert(self, alert: SecurityAlert) -> None:
        """Process security alert"""
        logger.warning(f"Processing security alert: {alert.name} (Priority: {alert.severity.name})")

        # Send notifications
        await self._send_alert_notifications(alert)

        # Trigger alert handlers
        handlers = self.alert_handlers.get(alert.severity, [])
        for handler in handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {str(e)}")

        # Check escalation rules
        await self._check_escalation_rules(alert)

    async def _send_alert_notifications(self, alert: SecurityAlert) -> None:
        """Send alert notifications"""
        # Find monitoring rule for this alert
        rule = None
        for monitoring_rule in self.monitoring_rules.values():
            if (monitoring_rule.metric_name == alert.metric_name and
                monitoring_rule.monitoring_type == alert.monitoring_type):
                rule = monitoring_rule
                break

        if not rule:
            return

        # Send notifications through configured channels
        for channel in rule.notification_channels:
            if channel in self.notification_channels:
                try:
                    notification_func = self.notification_channels[channel]
                    notification_func(alert, [])
                except Exception as e:
                    logger.error(f"Error sending {channel} notification: {str(e)}")

    async def _check_escalation_rules(self, alert: SecurityAlert) -> None:
        """Check and apply escalation rules"""
        # Simple escalation: if alert not acknowledged within timeout, escalate
        if not alert.acknowledged:
            escalation_timeout = 300  # 5 minutes

            def escalate():
                time.sleep(escalation_timeout)
                if alert.alert_id in self.active_alerts and not alert.acknowledged:
                    alert.escalation_level += 1

                    # Re-send notifications with higher priority
                    if alert.escalation_level >= 2:
                        alert.severity = AlertPriority(alert.severity.value + 1)
                        if alert.severity.value > AlertPriority.EMERGENCY.value:
                            alert.severity = AlertPriority.EMERGENCY

                    asyncio.create_task(self._send_alert_notifications(alert))

            threading.Thread(target=escalate, daemon=True).start()

    async def _process_alerts(self) -> None:
        """Background alert processing"""
        while self.running:
            try:
                # Check for stale alerts
                stale_threshold = datetime.now() - timedelta(hours=24)
                stale_alerts = [
                    alert for alert in self.active_alerts.values()
                    if alert.timestamp < stale_threshold and not alert.acknowledged
                ]

                for alert in stale_alerts:
                    # Escalate stale alerts
                    alert.escalation_level += 1
                    await self._send_alert_notifications(alert)

                # Clean up old resolved alerts
                resolved_threshold = datetime.now() - timedelta(days=7)
                self.alerts = deque(
                    [alert for alert in self.alerts if alert.timestamp > resolved_threshold],
                    maxlen=10000
                )

            except Exception as e:
                logger.error(f"Error processing alerts: {str(e)}")

            await asyncio.sleep(300)  # Check every 5 minutes

    def register_alert_handler(self, priority: AlertPriority, handler: Callable) -> None:
        """Register alert handler for specific priority"""
        self.alert_handlers[priority].append(handler)

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge security alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()

            # Remove from active alerts
            del self.active_alerts[alert_id]

            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True

        return False

    def resolve_alert(self, alert_id: str, resolved_by: str, resolution_notes: str = "") -> bool:
        """Resolve security alert"""
        # Check active alerts
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            del self.active_alerts[alert_id]
        else:
            # Find in alert history
            alert = None
            for a in self.alerts:
                if a.alert_id == alert_id:
                    alert = a
                    break

            if not alert:
                return False

        alert.resolved = True
        alert.resolved_by = resolved_by
        alert.resolved_at = datetime.now()
        alert.details['resolution_notes'] = resolution_notes

        logger.info(f"Alert {alert_id} resolved by {resolved_by}")
        return True

    def get_security_status(self) -> Dict[str, Any]:
        """Get overall security status"""
        active_alert_count = len(self.active_alerts)
        critical_alerts = len([a for a in self.active_alerts.values() if a.severity == AlertPriority.CRITICAL])

        # Calculate system health score
        health_score = self._calculate_health_score()

        # Get recent metrics
        recent_metrics = {}
        for metric_name, metrics in self.metrics.items():
            if metrics:
                recent_metrics[metric_name] = metrics[-1].value

        return {
            "status": "healthy" if health_score > 80 else "warning" if health_score > 60 else "critical",
            "health_score": health_score,
            "active_alerts": active_alert_count,
            "critical_alerts": critical_alerts,
            "monitoring_level": self.monitoring_level.value,
            "recent_metrics": recent_metrics,
            "threat_metrics": self.threat_detection.get_threat_metrics(),
            "system_uptime": time.time() - psutil.boot_time(),
            "last_updated": datetime.now().isoformat()
        }

    def _calculate_health_score(self) -> float:
        """Calculate system health score"""
        score = 100.0

        # Deduct for active alerts
        for alert in self.active_alerts.values():
            if alert.severity == AlertPriority.CRITICAL:
                score -= 20
            elif alert.severity == AlertPriority.HIGH:
                score -= 10
            elif alert.severity == AlertPriority.MEDIUM:
                score -= 5
            elif alert.severity == AlertPriority.LOW:
                score -= 2

        # Check system metrics
        cpu_metric = self.metrics.get("cpu_usage_percent", [])
        if cpu_metric:
            cpu_usage = cpu_metric[-1].value
            if cpu_usage > 90:
                score -= 15
            elif cpu_usage > 80:
                score -= 10
            elif cpu_usage > 70:
                score -= 5

        memory_metric = self.metrics.get("memory_usage_percent", [])
        if memory_metric:
            memory_usage = memory_metric[-1].value
            if memory_usage > 90:
                score -= 15
            elif memory_usage > 80:
                score -= 10
            elif memory_usage > 70:
                score -= 5

        return max(0, min(100, score))

    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_alerts = [
            alert for alert in self.alerts
            if alert.timestamp > cutoff_time
        ]

        # Group by severity
        severity_counts = defaultdict(int)
        type_counts = defaultdict(int)

        for alert in recent_alerts:
            severity_counts[alert.severity.name.lower()] += 1
            type_counts[alert.monitoring_type.value] += 1

        return {
            "total_alerts": len(recent_alerts),
            "active_alerts": len(self.active_alerts),
            "severity_distribution": dict(severity_counts),
            "type_distribution": dict(type_counts),
            "acknowledged_rate": len([a for a in recent_alerts if a.acknowledged]) / len(recent_alerts) if recent_alerts else 0,
            "resolved_rate": len([a for a in recent_alerts if a.resolved]) / len(recent_alerts) if recent_alerts else 0,
            "average_response_time": self._calculate_average_response_time(recent_alerts)
        }

    def _calculate_average_response_time(self, alerts: List[SecurityAlert]) -> float:
        """Calculate average alert response time"""
        acknowledged_alerts = [a for a in alerts if a.acknowledged and a.acknowledged_at]

        if not acknowledged_alerts:
            return 0.0

        total_response_time = sum(
            (a.acknowledged_at - a.timestamp).total_seconds()
            for a in acknowledged_alerts
        )

        return total_response_time / len(acknowledged_alerts)

    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        summary = {}

        for metric_name, metrics in self.metrics.items():
            recent_metrics = [
                m for m in metrics if m.timestamp > cutoff_time
            ]

            if recent_metrics:
                values = [m.value for m in recent_metrics]
                summary[metric_name] = {
                    "current": values[-1],
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "unit": recent_metrics[-1].unit,
                    "count": len(values)
                }

        return summary

    def export_monitoring_data(self, format_type: str = "json", hours: int = 24) -> str:
        """Export monitoring data"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        data = {
            "export_timestamp": datetime.now().isoformat(),
            "time_range_hours": hours,
            "metrics": {},
            "alerts": [],
            "system_info": {
                "hostname": socket.gethostname(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "disk_total": psutil.disk_usage('/').total
            }
        }

        # Export metrics
        for metric_name, metrics in self.metrics.items():
            recent_metrics = [
                m for m in metrics if m.timestamp > cutoff_time
            ]
            data["metrics"][metric_name] = [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "value": m.value,
                    "unit": m.unit,
                    "tags": m.tags
                }
                for m in recent_metrics
            ]

        # Export alerts
        recent_alerts = [
            alert for alert in self.alerts if alert.timestamp > cutoff_time
        ]
        data["alerts"] = [alert.to_dict() for alert in recent_alerts]

        if format_type == "json":
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format_type}")