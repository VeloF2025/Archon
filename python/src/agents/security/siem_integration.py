"""
SIEM Integration Service
Integration with Security Information and Event Management systems
"""

import asyncio
import logging
import json
import aiohttp
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import uuid
import hashlib
import gzip
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as ET

from .security_monitoring import SecurityMonitor, SecurityAlert, AlertPriority
from .threat_detection import ThreatEvent, ThreatSeverity, ThreatType

logger = logging.getLogger(__name__)


class SIEMType(Enum):
    """Supported SIEM systems"""
    SPLUNK = "splunk"
    ELASTIC_SIEM = "elastic_siem"
    QRADAR = "qradar"
    ARC_SIGHT = "arc_sight"
    MICROSOFT_SENTINEL = "microsoft_sentinel"
    SUMO_LOGIC = "sumo_logic"
    LOG_RHYTHM = "log_rhythm"
    CUSTOM = "custom"


class EventType(Enum):
    """Types of security events"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NETWORK = "network"
    SYSTEM = "system"
    APPLICATION = "application"
    DATABASE = "database"
    THREAT_DETECTION = "threat_detection"
    COMPLIANCE = "compliance"
    AUDIT = "audit"


@dataclass
class SIEMEvent:
    """SIEM event format"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    source: str
    message: str
    severity: str
    details: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    raw_data: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "message": self.message,
            "severity": self.severity,
            "details": self.details,
            "tags": self.tags,
            "raw_data": self.raw_data
        }

    def to_celery_format(self) -> str:
        """Convert to Common Event Format (CEF)"""
        cef_version = "0"
        device_vendor = self.details.get("device_vendor", "archon")
        device_product = self.details.get("device_product", "security_system")
        device_version = self.details.get("device_version", "1.0")
        device_event_class_id = self.event_type.value
        name = self.message
        severity = str(self._map_severity_to_cef())

        # Construct CEF header
        cef_header = f"CEF:{cef_version}|{device_vendor}|{device_product}|{device_version}|{device_event_class_id}|{name}|{severity}|"

        # Add extensions
        extensions = []
        for key, value in self.details.items():
            extensions.append(f"{key}={value}")

        cef_extensions = " ".join(extensions)
        return f"{cef_header}{cef_extensions}"

    def _map_severity_to_cef(self) -> int:
        """Map severity to CEF severity scale"""
        severity_mapping = {
            "low": 3,
            "medium": 5,
            "high": 7,
            "critical": 9,
            "emergency": 10
        }
        return severity_mapping.get(self.severity.lower(), 5)


@dataclass
class SIEMConfig:
    """SIEM configuration"""
    siem_type: SIEMType
    endpoint_url: str
    authentication: Dict[str, str]
    enabled: bool = True
    batch_size: int = 100
    batch_timeout: int = 30  # seconds
    retry_attempts: int = 3
    timeout: int = 30
    verify_ssl: bool = True
    compression: bool = True
    event_format: str = "json"  # json, cef, leef, syslog


@dataclass
class SIEMQuery:
    """SIEM search query"""
    query_id: str
    name: str
    description: str
    query: str
    start_time: datetime
    end_time: datetime
    filters: Dict[str, Any] = field(default_factory=dict)
    fields: List[str] = field(default_factory=list)
    limit: int = 1000
    last_run: Optional[datetime] = None
    schedule: str = "manual"  # manual, hourly, daily


class SIEMIntegration:
    """SIEM integration service"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.siem_configs: Dict[str, SIEMConfig] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self.query_results: Dict[str, Any] = {}
        self.session: Optional[aiohttp.ClientSession] = None

        # Metrics
        self.metrics: Dict[str, int] = {
            "events_sent": 0,
            "events_failed": 0,
            "queries_executed": 0,
            "last_connection_error": 0
        }

        # Initialize SIEM configurations
        self._initialize_siem_configs()

        # Start event processing
        self.running = False
        self.processing_task: Optional[asyncio.Task] = None

    def _initialize_siem_configs(self) -> None:
        """Initialize SIEM configurations from config"""

        # Splunk configuration
        if "splunk" in self.config:
            splunk_config = SIEMConfig(
                siem_type=SIEMType.SPLUNK,
                endpoint_url=self.config["splunk"]["endpoint_url"],
                authentication=self.config["splunk"]["authentication"],
                batch_size=self.config["splunk"].get("batch_size", 100),
                batch_timeout=self.config["splunk"].get("batch_timeout", 30)
            )
            self.siem_configs["splunk"] = splunk_config

        # Elastic SIEM configuration
        if "elastic_siem" in self.config:
            elastic_config = SIEMConfig(
                siem_type=SIEMType.ELASTIC_SIEM,
                endpoint_url=self.config["elastic_siem"]["endpoint_url"],
                authentication=self.config["elastic_siem"]["authentication"],
                batch_size=self.config["elastic_siem"].get("batch_size", 100),
                batch_timeout=self.config["elastic_siem"].get("batch_timeout", 30)
            )
            self.siem_configs["elastic_siem"] = elastic_config

        # Microsoft Sentinel configuration
        if "microsoft_sentinel" in self.config:
            sentinel_config = SIEMConfig(
                siem_type=SIEMType.MICROSOFT_SENTINEL,
                endpoint_url=self.config["microsoft_sentinel"]["endpoint_url"],
                authentication=self.config["microsoft_sentinel"]["authentication"],
                batch_size=self.config["microsoft_sentinel"].get("batch_size", 100),
                batch_timeout=self.config["microsoft_sentinel"].get("batch_timeout", 30)
            )
            self.siem_configs["microsoft_sentinel"] = sentinel_config

    async def start(self) -> None:
        """Start SIEM integration service"""
        if self.running:
            return

        self.running = True
        self.session = aiohttp.ClientSession()

        # Start event processing task
        self.processing_task = asyncio.create_task(self._process_events())

        logger.info("SIEM integration service started")

    async def stop(self) -> None:
        """Stop SIEM integration service"""
        if not self.running:
            return

        self.running = False

        # Cancel processing task
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

        # Close session
        if self.session:
            await self.session.close()

        logger.info("SIEM integration service stopped")

    async def send_event(self, event: SIEMEvent, siem_systems: List[str] = None) -> None:
        """Send event to SIEM systems"""
        if siem_systems is None:
            siem_systems = list(self.siem_configs.keys())

        for siem_name in siem_systems:
            if siem_name in self.siem_configs and self.siem_configs[siem_name].enabled:
                await self.event_queue.put((siem_name, event))

    async def send_security_alert(self, alert: SecurityAlert, siem_systems: List[str] = None) -> None:
        """Send security alert to SIEM systems"""
        event = SIEMEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.THREAT_DETECTION,
            timestamp=alert.timestamp,
            source="archon_security_monitor",
            message=alert.description,
            severity=alert.severity.name.lower(),
            details={
                "alert_id": alert.alert_id,
                "alert_name": alert.name,
                "monitoring_type": alert.monitoring_type.value,
                "metric_name": alert.metric_name,
                "metric_value": alert.metric_value,
                "acknowledged": alert.acknowledged,
                "resolved": alert.resolved,
                "escalation_level": alert.escalation_level
            },
            tags=["security", "alert", alert.monitoring_type.value]
        )

        await self.send_event(event, siem_systems)

    async def send_threat_event(self, threat_event: ThreatEvent, siem_systems: List[str] = None) -> None:
        """Send threat event to SIEM systems"""
        event = SIEMEvent(
            event_id=threat_event.event_id,
            event_type=EventType.THREAT_DETECTION,
            timestamp=threat_event.timestamp,
            source="archon_threat_detection",
            message=threat_event.description,
            severity=threat_event.severity.name.lower(),
            details={
                "threat_type": threat_event.threat_type.value,
                "source_ip": threat_event.source_ip,
                "target_ip": threat_event.target_ip,
                "user_id": threat_event.user_id,
                "confidence": threat_event.confidence,
                "false_positive_probability": threat_event.false_positive_probability,
                "status": threat_event.status.value
            },
            tags=["threat", threat_event.threat_type.value],
            raw_data=json.dumps(threat_event.raw_data)
        )

        await self.send_event(event, siem_systems)

    async def send_audit_event(self, event_type: str, description: str, details: Dict[str, Any],
                             severity: str = "medium", siem_systems: List[str] = None) -> None:
        """Send audit event to SIEM systems"""
        event = SIEMEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.AUDIT,
            timestamp=datetime.now(),
            source="archon_audit_system",
            message=description,
            severity=severity,
            details={
                "event_type": event_type,
                **details
            },
            tags=["audit", "compliance"]
        )

        await self.send_event(event, siem_systems)

    async def _process_events(self) -> None:
        """Process events from queue and send to SIEM systems"""
        event_batches: Dict[str, List[SIEMEvent]] = {}

        while self.running:
            try:
                # Collect events for batching
                try:
                    siem_name, event = await asyncio.wait_for(
                        self.event_queue.get(), timeout=1.0
                    )

                    if siem_name not in event_batches:
                        event_batches[siem_name] = []

                    event_batches[siem_name].append(event)

                    # Check if batch is ready to send
                    siem_config = self.siem_configs.get(siem_name)
                    if siem_config and len(event_batches[siem_name]) >= siem_config.batch_size:
                        await self._send_event_batch(siem_name, event_batches[siem_name])
                        del event_batches[siem_name]

                except asyncio.TimeoutError:
                    # Check for timed-out batches
                    current_time = datetime.now()

                    for siem_name, events in list(event_batches.items()):
                        siem_config = self.siem_configs.get(siem_name)
                        if siem_config and events:
                            # Check if batch has timed out
                            batch_age = (current_time - events[0].timestamp).total_seconds()
                            if batch_age >= siem_config.batch_timeout:
                                await self._send_event_batch(siem_name, events)
                                del event_batches[siem_name]

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error processing SIEM events: {str(e)}")
                self.metrics["events_failed"] += 1

        # Send remaining events on shutdown
        for siem_name, events in event_batches.items():
            await self._send_event_batch(siem_name, events)

    async def _send_event_batch(self, siem_name: str, events: List[SIEMEvent]) -> None:
        """Send batch of events to SIEM system"""
        siem_config = self.siem_configs.get(siem_name)
        if not siem_config:
            return

        try:
            if siem_config.siem_type == SIEMType.SPLUNK:
                await self._send_to_splunk(siem_config, events)
            elif siem_config.siem_type == SIEMType.ELASTIC_SIEM:
                await self._send_to_elastic(siem_config, events)
            elif siem_config.siem_type == SIEMType.MICROSOFT_SENTINEL:
                await self._send_to_sentinel(siem_config, events)
            else:
                logger.warning(f"Unsupported SIEM type: {siem_config.siem_type}")
                return

            self.metrics["events_sent"] += len(events)
            logger.debug(f"Sent {len(events)} events to {siem_name}")

        except Exception as e:
            logger.error(f"Error sending events to {siem_name}: {str(e)}")
            self.metrics["events_failed"] += len(events)
            self.metrics["last_connection_error"] = int(datetime.now().timestamp())

    async def _send_to_splunk(self, config: SIEMConfig, events: List[SIEMEvent]) -> None:
        """Send events to Splunk"""
        # Prepare HEC (HTTP Event Collector) payload
        splunk_events = []
        for event in events:
            splunk_event = {
                "event": event.to_dict(),
                "sourcetype": "archon:security",
                "source": "archon_security_system",
                "index": config.authentication.get("index", "main"),
                "host": event.details.get("host", "localhost")
            }
            splunk_events.append(splunk_event)

        # Send to Splunk HEC
        headers = {
            "Authorization": f"Splunk {config.authentication['token']}",
            "Content-Type": "application/json"
        }

        if config.compression:
            headers["Content-Encoding"] = "gzip"

        payload = json.dumps({"events": splunk_events})

        if config.compression:
            payload = gzip.compress(payload.encode())

        async with self.session.post(
            f"{config.endpoint_url}/services/collector/event",
            headers=headers,
            data=payload,
            timeout=aiohttp.ClientTimeout(total=config.timeout)
        ) as response:
            if response.status not in [200, 201]:
                error_text = await response.text()
                raise Exception(f"Splunk HEC error: {response.status} - {error_text}")

    async def _send_to_elastic(self, config: SIEMConfig, events: List[SIEMEvent]) -> None:
        """Send events to Elastic SIEM"""
        # Prepare Elasticsearch bulk API payload
        bulk_lines = []

        for event in events:
            # Index action
            action = {
                "index": {
                    "_index": config.authentication.get("index", "archon-security-events"),
                    "_id": event.event_id
                }
            }
            bulk_lines.append(json.dumps(action))
            bulk_lines.append(json.dumps(event.to_dict()))

        bulk_payload = "\n".join(bulk_lines) + "\n"

        headers = {
            "Content-Type": "application/x-ndjson",
            "Authorization": f"ApiKey {config.authentication['api_key']}"
        }

        async with self.session.post(
            f"{config.endpoint_url}/_bulk",
            headers=headers,
            data=bulk_payload,
            timeout=aiohttp.ClientTimeout(total=config.timeout)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Elasticsearch bulk error: {response.status} - {error_text}")

    async def _send_to_sentinel(self, config: SIEMConfig, events: List[SIEMEvent]) -> None:
        """Send events to Microsoft Sentinel (via Log Analytics)"""
        # Prepare Log Analytics payload
        sentinel_events = []
        for event in events:
            sentinel_event = {
                "TimeGeneratedUtc": event.timestamp.isoformat(),
                "Message": event.message,
                "Severity": event.severity,
                "EventType": event.event_type.value,
                "Source": event.source,
                "EventId": event.event_id,
                **event.details
            }
            sentinel_events.append(sentinel_event)

        # Build Log Analytics API signature
        workspace_id = config.authentication["workspace_id"]
        shared_key = config.authentication["shared_key"]

        date_string = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
        content = json.dumps(sentinel_events)
        content_length = len(content)

        # Create signature
        string_to_hash = f"POST\n{content_length}\napplication/json\nx-ms-date:{date_string}\n/api/logs"
        encoded_hash = base64.b64encode(
            hmac.new(
                base64.b64decode(shared_key),
                string_to_hash.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"SharedKey {workspace_id}:{encoded_hash}",
            "x-ms-date": date_string,
            "Log-Type": "ArchoSecurityEvent"
        }

        async with self.session.post(
            f"{config.endpoint_url}/api/logs?api-version=2016-04-01",
            headers=headers,
            data=content,
            timeout=aiohttp.ClientTimeout(total=config.timeout)
        ) as response:
            if response.status not in [200, 202]:
                error_text = await response.text()
                raise Exception(f"Log Analytics error: {response.status} - {error_text}")

    async def execute_query(self, siem_name: str, query: SIEMQuery) -> Dict[str, Any]:
        """Execute query against SIEM system"""
        siem_config = self.siem_configs.get(siem_name)
        if not siem_config:
            raise ValueError(f"SIEM system {siem_name} not configured")

        try:
            if siem_config.siem_type == SIEMType.SPLUNK:
                result = await self._execute_splunk_query(siem_config, query)
            elif siem_config.siem_type == SIEMType.ELASTIC_SIEM:
                result = await self._execute_elastic_query(siem_config, query)
            elif siem_config.siem_type == SIEMType.MICROSOFT_SENTINEL:
                result = await self._execute_sentinel_query(siem_config, query)
            else:
                raise ValueError(f"Query execution not supported for {siem_config.siem_type}")

            self.metrics["queries_executed"] += 1
            query.last_run = datetime.now()
            self.query_results[query.query_id] = result

            return result

        except Exception as e:
            logger.error(f"Error executing query on {siem_name}: {str(e)}")
            raise

    async def _execute_splunk_query(self, config: SIEMConfig, query: SIEMQuery) -> Dict[str, Any]:
        """Execute query against Splunk"""
        headers = {
            "Authorization": f"Bearer {config.authentication['token']}",
            "Content-Type": "application/json"
        }

        search_params = {
            "search": query.query,
            "earliest_time": query.start_time.isoformat(),
            "latest_time": query.end_time.isoformat(),
            "output_mode": "json",
            "count": query.limit
        }

        async with self.session.post(
            f"{config.endpoint_url}/services/search/jobs",
            headers=headers,
            json=search_params,
            timeout=aiohttp.ClientTimeout(total=config.timeout)
        ) as response:
            if response.status != 201:
                error_text = await response.text()
                raise Exception(f"Splunk job creation error: {response.status} - {error_text}")

            job_data = await response.json()
            sid = job_data["sid"]

        # Wait for job to complete and get results
        job_url = f"{config.endpoint_url}/services/search/jobs/{sid}/results"
        result_url = f"{job_url}?output_mode=json&count={query.limit}"

        async with self.session.get(
            result_url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=config.timeout)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Splunk search error: {response.status} - {error_text}")

            results = await response.json()

        return {
            "query_id": query.query_id,
            "siem_system": "splunk",
            "results": results.get("results", []),
            "fields": results.get("fields", []),
            "total_count": len(results.get("results", [])),
            "execution_time": datetime.now().isoformat()
        }

    async def _execute_elastic_query(self, config: SIEMConfig, query: SIEMQuery) -> Dict[str, Any]:
        """Execute query against Elastic SIEM"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"ApiKey {config.authentication['api_key']}"
        }

        # Build Elasticsearch query
        es_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "@timestamp": {
                                    "gte": query.start_time.isoformat(),
                                    "lte": query.end_time.isoformat()
                                }
                            }
                        }
                    ]
                }
            },
            "size": query.limit,
            "_source": query.fields if query.fields else True
        }

        # Add filters
        if query.filters:
            es_query["query"]["bool"]["filter"] = query.filters

        async with self.session.post(
            f"{config.endpoint_url}/{config.authentication.get('index', 'archon-security-events')}/_search",
            headers=headers,
            json=es_query,
            timeout=aiohttp.ClientTimeout(total=config.timeout)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Elasticsearch search error: {response.status} - {error_text}")

            results = await response.json()

        return {
            "query_id": query.query_id,
            "siem_system": "elastic_siem",
            "results": results.get("hits", {}).get("hits", []),
            "total_count": results.get("hits", {}).get("total", {}).get("value", 0),
            "execution_time": datetime.now().isoformat()
        }

    async def _execute_sentinel_query(self, config: SIEMConfig, query: SIEMQuery) -> Dict[str, Any]:
        """Execute query against Microsoft Sentinel"""
        # This would typically use the Azure Sentinel REST API
        # For now, return placeholder implementation

        return {
            "query_id": query.query_id,
            "siem_system": "microsoft_sentinel",
            "results": [],
            "total_count": 0,
            "execution_time": datetime.now().isoformat(),
            "note": "Query execution not fully implemented for Microsoft Sentinel"
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get SIEM integration metrics"""
        return {
            **self.metrics,
            "configured_siem_systems": len(self.siem_configs),
            "enabled_siem_systems": len([c for c in self.siem_configs.values() if c.enabled]),
            "event_queue_size": self.event_queue.qsize(),
            "query_cache_size": len(self.query_results)
        }

    def create_siem_query(self, name: str, description: str, query: str,
                        hours_back: int = 24, fields: List[str] = None,
                        limit: int = 1000) -> SIEMQuery:
        """Create SIEM query"""
        return SIEMQuery(
            query_id=str(uuid.uuid4()),
            name=name,
            description=description,
            query=query,
            start_time=datetime.now() - timedelta(hours=hours_back),
            end_time=datetime.now(),
            fields=fields or [],
            limit=limit
        )

    async def correlate_events(self, siem_name: str, hours: int = 24) -> Dict[str, Any]:
        """Correlate events across different sources"""
        query = self.create_siem_query(
            name="Event Correlation",
            description="Correlate security events across sources",
            query="* | stats count by source, event_type, severity | sort -count",
            hours_back=hours,
            limit=100
        )

        try:
            results = await self.execute_query(siem_name, query)

            # Analyze correlation patterns
            correlation_analysis = {
                "total_events": results.get("total_count", 0),
                "source_distribution": {},
                "event_type_distribution": {},
                "severity_distribution": {},
                "correlation_patterns": []
            }

            # Process results to identify patterns
            for result in results.get("results", []):
                # This would analyze the actual results to find correlations
                # For now, return basic structure
                pass

            return correlation_analysis

        except Exception as e:
            logger.error(f"Error correlating events: {str(e)}")
            return {"error": str(e)}

    async def export_events(self, siem_name: str, start_time: datetime, end_time: datetime,
                           format_type: str = "json") -> str:
        """Export events from SIEM system"""
        query = self.create_siem_query(
            name="Event Export",
            description=f"Export events from {start_time} to {end_time}",
            query="*",
            hours_back=int((end_time - start_time).total_seconds() / 3600),
            limit=10000
        )

        query.start_time = start_time
        query.end_time = end_time

        try:
            results = await self.execute_query(siem_name, query)

            if format_type == "json":
                return json.dumps(results, indent=2)
            elif format_type == "csv":
                # Convert to CSV format
                return self._convert_to_csv(results.get("results", []))
            else:
                raise ValueError(f"Unsupported export format: {format_type}")

        except Exception as e:
            logger.error(f"Error exporting events: {str(e)}")
            raise

    def _convert_to_csv(self, results: List[Dict]) -> str:
        """Convert results to CSV format"""
        if not results:
            return ""

        # Extract headers from first result
        headers = set()
        for result in results:
            if isinstance(result, dict):
                headers.update(result.keys())

        headers = sorted(list(headers))

        # Generate CSV
        csv_lines = [",".join(headers)]

        for result in results:
            row = []
            for header in headers:
                value = result.get(header, "")
                # Escape CSV special characters
                if isinstance(value, str):
                    value = f'"{value.replace('"', '""')}"'
                else:
                    value = str(value)
                row.append(value)
            csv_lines.append(",".join(row))

        return "\n".join(csv_lines)


# Import required modules
import base64
import hmac
import hashlib