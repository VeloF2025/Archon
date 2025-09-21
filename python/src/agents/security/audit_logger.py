"""
Advanced Audit Logging System
Comprehensive audit trail with tamper-proof logging and compliance features
"""

import asyncio
import logging
import json
import hashlib
import hmac
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import gzip
import base64
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import os

logger = logging.getLogger(__name__)


class AuditLevel(Enum):
    """Audit log levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"
    COMPLIANCE = "compliance"


class AuditCategory(Enum):
    """Categories of audit events"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    USER_ACTION = "user_action"
    API_CALL = "api_call"
    FILE_ACCESS = "file_access"
    NETWORK = "network"
    ERROR = "error"


class ComplianceFramework(Enum):
    """Compliance frameworks"""
    SOX = "sox"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NIST = "nist"
    SOC2 = "soc2"
    FERPA = "ferpa"


@dataclass
class AuditEvent:
    """Immutable audit event record"""
    event_id: str
    timestamp: datetime
    level: AuditLevel
    category: AuditCategory
    event_type: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    result: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    compliance_tags: Set[ComplianceFramework] = field(default_factory=set)
    severity_score: float = 0.0
    risk_score: float = 0.0
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    hash_chain_previous: Optional[str] = None
    hash_chain_current: Optional[str] = None
    signature: Optional[str] = None
    
    def __post_init__(self):
        """Calculate hash chain and signature after initialization"""
        if not self.hash_chain_current:
            self.hash_chain_current = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of audit event"""
        # Create deterministic string representation
        data = {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "category": self.category.value,
            "event_type": self.event_type,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "source_ip": self.source_ip,
            "resource": self.resource,
            "action": self.action,
            "result": self.result,
            "details": json.dumps(self.details, sort_keys=True),
            "hash_chain_previous": self.hash_chain_previous
        }
        
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "category": self.category.value,
            "event_type": self.event_type,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "source_ip": self.source_ip,
            "user_agent": self.user_agent,
            "resource": self.resource,
            "action": self.action,
            "result": self.result,
            "details": self.details,
            "compliance_tags": [tag.value for tag in self.compliance_tags],
            "severity_score": self.severity_score,
            "risk_score": self.risk_score,
            "correlation_id": self.correlation_id,
            "parent_event_id": self.parent_event_id,
            "hash_chain_previous": self.hash_chain_previous,
            "hash_chain_current": self.hash_chain_current,
            "signature": self.signature
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create from dictionary"""
        compliance_tags = {ComplianceFramework(tag) for tag in data.get("compliance_tags", [])}
        
        return cls(
            event_id=data["event_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            level=AuditLevel(data["level"]),
            category=AuditCategory(data["category"]),
            event_type=data["event_type"],
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            source_ip=data.get("source_ip"),
            user_agent=data.get("user_agent"),
            resource=data.get("resource"),
            action=data.get("action"),
            result=data.get("result"),
            details=data.get("details", {}),
            compliance_tags=compliance_tags,
            severity_score=data.get("severity_score", 0.0),
            risk_score=data.get("risk_score", 0.0),
            correlation_id=data.get("correlation_id"),
            parent_event_id=data.get("parent_event_id"),
            hash_chain_previous=data.get("hash_chain_previous"),
            hash_chain_current=data.get("hash_chain_current"),
            signature=data.get("signature")
        )


@dataclass
class AuditFilter:
    """Filter criteria for audit log queries"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    levels: List[AuditLevel] = field(default_factory=list)
    categories: List[AuditCategory] = field(default_factory=list)
    user_ids: List[str] = field(default_factory=list)
    source_ips: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    minimum_severity: float = 0.0
    minimum_risk: float = 0.0
    search_terms: List[str] = field(default_factory=list)
    correlation_id: Optional[str] = None
    limit: int = 1000


class AuditStorage:
    """Abstract base class for audit log storage"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    async def store_event(self, event: AuditEvent) -> None:
        """Store audit event"""
        raise NotImplementedError
        
    async def query_events(self, audit_filter: AuditFilter) -> List[AuditEvent]:
        """Query audit events"""
        raise NotImplementedError
        
    async def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """Get specific audit event"""
        raise NotImplementedError
        
    async def verify_integrity(self, start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Verify audit log integrity"""
        raise NotImplementedError


class FileAuditStorage(AuditStorage):
    """File-based audit storage with rotation and compression"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.log_dir = config.get("log_directory", "/var/log/audit")
        self.max_file_size = config.get("max_file_size", 100 * 1024 * 1024)  # 100MB
        self.max_files = config.get("max_files", 100)
        self.compress_after_days = config.get("compress_after_days", 7)
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.current_file = None
        self.current_file_size = 0
        self.lock = threading.Lock()
        
    async def store_event(self, event: AuditEvent) -> None:
        """Store audit event to file"""
        event_json = json.dumps(event.to_dict()) + '\n'
        
        with self.lock:
            # Check if we need to rotate file
            if (self.current_file is None or 
                self.current_file_size + len(event_json) > self.max_file_size):
                await self._rotate_file()
                
            # Write to current file
            with open(self.current_file, 'a', encoding='utf-8') as f:
                f.write(event_json)
                self.current_file_size += len(event_json)
                
    async def _rotate_file(self) -> None:
        """Rotate log file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_file = os.path.join(self.log_dir, f"audit_{timestamp}.log")
        
        self.current_file = new_file
        self.current_file_size = 0
        
        # Clean up old files
        await self._cleanup_old_files()
        
    async def _cleanup_old_files(self) -> None:
        """Clean up old audit files"""
        files = []
        for filename in os.listdir(self.log_dir):
            if filename.startswith("audit_") and filename.endswith(".log"):
                filepath = os.path.join(self.log_dir, filename)
                stat = os.stat(filepath)
                files.append((filepath, stat.st_mtime))
                
        # Sort by modification time (oldest first)
        files.sort(key=lambda x: x[1])
        
        # Compress old files
        cutoff_time = time.time() - (self.compress_after_days * 24 * 3600)
        for filepath, mtime in files:
            if mtime < cutoff_time and not filepath.endswith('.gz'):
                await self._compress_file(filepath)
                
        # Remove excess files
        all_files = [f[0] for f in files]
        if len(all_files) > self.max_files:
            for filepath in all_files[:len(all_files) - self.max_files]:
                os.remove(filepath)
                
    async def _compress_file(self, filepath: str) -> None:
        """Compress audit log file"""
        compressed_path = filepath + '.gz'
        
        with open(filepath, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                f_out.writelines(f_in)
                
        os.remove(filepath)
        
    async def query_events(self, audit_filter: AuditFilter) -> List[AuditEvent]:
        """Query events from files"""
        events = []
        
        # Get all log files
        log_files = []
        for filename in os.listdir(self.log_dir):
            if filename.startswith("audit_"):
                log_files.append(os.path.join(self.log_dir, filename))
                
        log_files.sort()
        
        for filepath in log_files:
            file_events = await self._read_events_from_file(filepath, audit_filter)
            events.extend(file_events)
            
            if len(events) >= audit_filter.limit:
                break
                
        return events[:audit_filter.limit]
        
    async def _read_events_from_file(self, filepath: str, audit_filter: AuditFilter) -> List[AuditEvent]:
        """Read events from a single file"""
        events = []
        
        try:
            if filepath.endswith('.gz'):
                file_obj = gzip.open(filepath, 'rt', encoding='utf-8')
            else:
                file_obj = open(filepath, 'r', encoding='utf-8')
                
            with file_obj as f:
                for line in f:
                    try:
                        event_data = json.loads(line.strip())
                        event = AuditEvent.from_dict(event_data)
                        
                        if self._matches_filter(event, audit_filter):
                            events.append(event)
                            
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading audit file {filepath}: {str(e)}")
            
        return events
        
    def _matches_filter(self, event: AuditEvent, audit_filter: AuditFilter) -> bool:
        """Check if event matches filter criteria"""
        # Time range
        if audit_filter.start_time and event.timestamp < audit_filter.start_time:
            return False
        if audit_filter.end_time and event.timestamp > audit_filter.end_time:
            return False
            
        # Levels
        if audit_filter.levels and event.level not in audit_filter.levels:
            return False
            
        # Categories
        if audit_filter.categories and event.category not in audit_filter.categories:
            return False
            
        # User IDs
        if audit_filter.user_ids and event.user_id not in audit_filter.user_ids:
            return False
            
        # Source IPs
        if audit_filter.source_ips and event.source_ip not in audit_filter.source_ips:
            return False
            
        # Resources
        if audit_filter.resources:
            if not event.resource or not any(res in event.resource for res in audit_filter.resources):
                return False
                
        # Actions
        if audit_filter.actions:
            if not event.action or not any(act in event.action for act in audit_filter.actions):
                return False
                
        # Compliance frameworks
        if audit_filter.compliance_frameworks:
            if not any(cf in event.compliance_tags for cf in audit_filter.compliance_frameworks):
                return False
                
        # Severity and risk thresholds
        if event.severity_score < audit_filter.minimum_severity:
            return False
        if event.risk_score < audit_filter.minimum_risk:
            return False
            
        # Search terms
        if audit_filter.search_terms:
            event_text = json.dumps(event.to_dict()).lower()
            if not any(term.lower() in event_text for term in audit_filter.search_terms):
                return False
                
        # Correlation ID
        if audit_filter.correlation_id and event.correlation_id != audit_filter.correlation_id:
            return False
            
        return True
        
    async def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """Get specific event by ID"""
        audit_filter = AuditFilter(limit=1)
        events = await self.query_events(audit_filter)
        
        for event in events:
            if event.event_id == event_id:
                return event
                
        return None
        
    async def verify_integrity(self, start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Verify hash chain integrity"""
        audit_filter = AuditFilter(start_time=start_time, end_time=end_time, limit=100000)
        events = await self.query_events(audit_filter)
        
        if not events:
            return {"valid": True, "total_events": 0}
            
        # Sort events by timestamp
        events.sort(key=lambda e: e.timestamp)
        
        integrity_issues = []
        previous_hash = None
        
        for i, event in enumerate(events):
            # Verify hash chain
            if i > 0 and event.hash_chain_previous != previous_hash:
                integrity_issues.append({
                    "event_id": event.event_id,
                    "issue": "hash_chain_broken",
                    "expected": previous_hash,
                    "actual": event.hash_chain_previous
                })
                
            # Verify event hash
            expected_hash = event._calculate_hash()
            if event.hash_chain_current != expected_hash:
                integrity_issues.append({
                    "event_id": event.event_id,
                    "issue": "hash_mismatch",
                    "expected": expected_hash,
                    "actual": event.hash_chain_current
                })
                
            previous_hash = event.hash_chain_current
            
        return {
            "valid": len(integrity_issues) == 0,
            "total_events": len(events),
            "integrity_issues": integrity_issues
        }


class AuditLogger:
    """Main audit logging system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.storage = self._initialize_storage()
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self.processing = False
        self.workers: List[asyncio.Task] = []
        
        # Hash chain tracking
        self.last_hash = None
        self.hash_lock = asyncio.Lock()
        
        # Metrics
        self.metrics: Dict[str, int] = defaultdict(int)
        
        # Event handlers
        self.event_handlers: Dict[AuditLevel, List[Callable]] = defaultdict(list)
        
        # Real-time alerting
        self.alert_rules: List[Dict[str, Any]] = []
        
        # Load default alert rules
        self._load_default_alert_rules()
        
    def _initialize_storage(self) -> AuditStorage:
        """Initialize audit storage backend"""
        storage_type = self.config.get("storage_type", "file")
        
        if storage_type == "file":
            return FileAuditStorage(self.config.get("storage_config", {}))
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
            
    def _load_default_alert_rules(self) -> None:
        """Load default alerting rules"""
        self.alert_rules = [
            {
                "name": "Multiple Failed Logins",
                "condition": {
                    "category": AuditCategory.AUTHENTICATION,
                    "result": "failure",
                    "threshold": 5,
                    "time_window": 300  # 5 minutes
                },
                "severity": AuditLevel.SECURITY
            },
            {
                "name": "Administrative Action",
                "condition": {
                    "category": AuditCategory.CONFIGURATION,
                    "action": "admin"
                },
                "severity": AuditLevel.SECURITY
            },
            {
                "name": "Suspicious Data Access",
                "condition": {
                    "category": AuditCategory.DATA_ACCESS,
                    "risk_threshold": 0.8
                },
                "severity": AuditLevel.SECURITY
            }
        ]
        
    async def start(self, num_workers: int = 3) -> None:
        """Start audit processing workers"""
        if self.processing:
            return
            
        self.processing = True
        logger.info(f"Starting audit logger with {num_workers} workers")
        
        # Start worker tasks
        for i in range(num_workers):
            worker = asyncio.create_task(self._process_events(i))
            self.workers.append(worker)
            
    async def stop(self) -> None:
        """Stop audit processing"""
        self.processing = False
        
        # Process remaining events
        while not self.event_queue.empty():
            await asyncio.sleep(0.1)
            
        # Cancel workers
        for worker in self.workers:
            worker.cancel()
            
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        logger.info("Audit logger stopped")
        
    async def log(self, level: AuditLevel, category: AuditCategory, event_type: str,
                 user_id: Optional[str] = None, session_id: Optional[str] = None,
                 source_ip: Optional[str] = None, user_agent: Optional[str] = None,
                 resource: Optional[str] = None, action: Optional[str] = None,
                 result: Optional[str] = None, details: Dict[str, Any] = None,
                 compliance_tags: Set[ComplianceFramework] = None,
                 correlation_id: Optional[str] = None) -> str:
        """Log audit event"""
        
        event_id = str(uuid.uuid4())
        
        # Calculate severity and risk scores
        severity_score = self._calculate_severity_score(level, category, details or {})
        risk_score = self._calculate_risk_score(category, action, details or {})
        
        # Create audit event
        async with self.hash_lock:
            event = AuditEvent(
                event_id=event_id,
                timestamp=datetime.utcnow(),
                level=level,
                category=category,
                event_type=event_type,
                user_id=user_id,
                session_id=session_id,
                source_ip=source_ip,
                user_agent=user_agent,
                resource=resource,
                action=action,
                result=result,
                details=details or {},
                compliance_tags=compliance_tags or set(),
                severity_score=severity_score,
                risk_score=risk_score,
                correlation_id=correlation_id,
                hash_chain_previous=self.last_hash
            )
            
            # Update hash chain
            self.last_hash = event.hash_chain_current
            
        # Queue for processing
        if not self.event_queue.full():
            await self.event_queue.put(event)
        else:
            logger.error("Audit event queue is full - dropping event")
            
        return event_id
        
    async def _process_events(self, worker_id: int) -> None:
        """Worker to process audit events"""
        logger.info(f"Audit worker {worker_id} started")
        
        while self.processing or not self.event_queue.empty():
            try:
                # Get event from queue
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Store event
                await self.storage.store_event(event)
                
                # Update metrics
                self.metrics[f"events_{event.level.value}"] += 1
                self.metrics[f"category_{event.category.value}"] += 1
                self.metrics["events_total"] += 1
                
                # Check alert rules
                await self._check_alert_rules(event)
                
                # Trigger event handlers
                handlers = self.event_handlers.get(event.level, [])
                for handler in handlers:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(f"Error in audit event handler: {str(e)}")
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Audit worker {worker_id} error: {str(e)}")
                
        logger.info(f"Audit worker {worker_id} stopped")
        
    def _calculate_severity_score(self, level: AuditLevel, category: AuditCategory,
                                 details: Dict[str, Any]) -> float:
        """Calculate severity score (0.0 to 1.0)"""
        base_scores = {
            AuditLevel.DEBUG: 0.1,
            AuditLevel.INFO: 0.2,
            AuditLevel.WARNING: 0.5,
            AuditLevel.ERROR: 0.7,
            AuditLevel.CRITICAL: 0.9,
            AuditLevel.SECURITY: 0.8,
            AuditLevel.COMPLIANCE: 0.6
        }
        
        category_modifiers = {
            AuditCategory.SECURITY: 0.3,
            AuditCategory.AUTHENTICATION: 0.2,
            AuditCategory.AUTHORIZATION: 0.2,
            AuditCategory.DATA_MODIFICATION: 0.2,
            AuditCategory.CONFIGURATION: 0.1
        }
        
        base_score = base_scores.get(level, 0.5)
        modifier = category_modifiers.get(category, 0.0)
        
        return min(1.0, base_score + modifier)
        
    def _calculate_risk_score(self, category: AuditCategory, action: Optional[str],
                             details: Dict[str, Any]) -> float:
        """Calculate risk score (0.0 to 1.0)"""
        base_risk = {
            AuditCategory.SECURITY: 0.8,
            AuditCategory.DATA_MODIFICATION: 0.7,
            AuditCategory.CONFIGURATION: 0.6,
            AuditCategory.AUTHENTICATION: 0.5,
            AuditCategory.DATA_ACCESS: 0.4
        }.get(category, 0.3)
        
        # Adjust based on action
        if action:
            if "delete" in action.lower():
                base_risk += 0.2
            elif "admin" in action.lower():
                base_risk += 0.3
            elif "root" in action.lower():
                base_risk += 0.4
                
        # Adjust based on details
        if details.get("failed_attempts", 0) > 5:
            base_risk += 0.3
        if details.get("privileged_operation"):
            base_risk += 0.2
            
        return min(1.0, base_risk)
        
    async def _check_alert_rules(self, event: AuditEvent) -> None:
        """Check if event triggers any alert rules"""
        for rule in self.alert_rules:
            if await self._evaluate_alert_rule(rule, event):
                await self._trigger_alert(rule, event)
                
    async def _evaluate_alert_rule(self, rule: Dict[str, Any], event: AuditEvent) -> bool:
        """Evaluate if event matches alert rule"""
        condition = rule.get("condition", {})
        
        # Check category
        if "category" in condition and event.category != condition["category"]:
            return False
            
        # Check result
        if "result" in condition and event.result != condition["result"]:
            return False
            
        # Check action
        if "action" in condition:
            if not event.action or condition["action"] not in event.action:
                return False
                
        # Check risk threshold
        if "risk_threshold" in condition:
            if event.risk_score < condition["risk_threshold"]:
                return False
                
        # Check threshold-based rules (e.g., multiple failed logins)
        if "threshold" in condition and "time_window" in condition:
            return await self._check_threshold_rule(condition, event)
            
        return True
        
    async def _check_threshold_rule(self, condition: Dict[str, Any], event: AuditEvent) -> bool:
        """Check threshold-based alert rule"""
        threshold = condition["threshold"]
        time_window = condition["time_window"]
        
        # Query recent events
        start_time = datetime.utcnow() - timedelta(seconds=time_window)
        audit_filter = AuditFilter(
            start_time=start_time,
            categories=[condition["category"]],
            user_ids=[event.user_id] if event.user_id else [],
            limit=threshold * 2
        )
        
        recent_events = await self.storage.query_events(audit_filter)
        
        # Count matching events
        matching_count = 0
        for recent_event in recent_events:
            if ("result" not in condition or 
                recent_event.result == condition["result"]):
                matching_count += 1
                
        return matching_count >= threshold
        
    async def _trigger_alert(self, rule: Dict[str, Any], event: AuditEvent) -> None:
        """Trigger security alert"""
        alert_event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            level=rule.get("severity", AuditLevel.WARNING),
            category=AuditCategory.SECURITY,
            event_type="security_alert",
            details={
                "rule_name": rule["name"],
                "triggered_by": event.event_id,
                "original_event": event.to_dict()
            },
            correlation_id=event.event_id
        )
        
        await self.event_queue.put(alert_event)
        logger.warning(f"Security alert triggered: {rule['name']}")
        
    def register_event_handler(self, level: AuditLevel, handler: Callable) -> None:
        """Register handler for audit events"""
        self.event_handlers[level].append(handler)
        
    async def query(self, audit_filter: AuditFilter) -> List[AuditEvent]:
        """Query audit events"""
        return await self.storage.query_events(audit_filter)
        
    async def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """Get specific audit event"""
        return await self.storage.get_event(event_id)
        
    async def verify_integrity(self, start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Verify audit log integrity"""
        return await self.storage.verify_integrity(start_time, end_time)
        
    def get_audit_metrics(self) -> Dict[str, Any]:
        """Get audit logging metrics"""
        return {
            "events_total": self.metrics.get("events_total", 0),
            "events_by_level": {
                level.value: self.metrics.get(f"events_{level.value}", 0)
                for level in AuditLevel
            },
            "events_by_category": {
                category.value: self.metrics.get(f"category_{category.value}", 0)
                for category in AuditCategory
            },
            "queue_size": self.event_queue.qsize(),
            "workers_active": len(self.workers),
            "alert_rules_count": len(self.alert_rules)
        }
        
    # Convenience methods for common audit events
    async def log_authentication(self, user_id: str, result: str, source_ip: str = None,
                               details: Dict[str, Any] = None) -> str:
        """Log authentication event"""
        return await self.log(
            level=AuditLevel.INFO if result == "success" else AuditLevel.WARNING,
            category=AuditCategory.AUTHENTICATION,
            event_type="login_attempt",
            user_id=user_id,
            source_ip=source_ip,
            result=result,
            details=details,
            compliance_tags={ComplianceFramework.SOX, ComplianceFramework.ISO_27001}
        )
        
    async def log_data_access(self, user_id: str, resource: str, action: str,
                             result: str = "success", details: Dict[str, Any] = None) -> str:
        """Log data access event"""
        return await self.log(
            level=AuditLevel.INFO,
            category=AuditCategory.DATA_ACCESS,
            event_type="data_access",
            user_id=user_id,
            resource=resource,
            action=action,
            result=result,
            details=details,
            compliance_tags={ComplianceFramework.GDPR, ComplianceFramework.HIPAA}
        )
        
    async def log_configuration_change(self, user_id: str, resource: str,
                                     action: str, details: Dict[str, Any] = None) -> str:
        """Log configuration change event"""
        return await self.log(
            level=AuditLevel.WARNING,
            category=AuditCategory.CONFIGURATION,
            event_type="configuration_change",
            user_id=user_id,
            resource=resource,
            action=action,
            result="success",
            details=details,
            compliance_tags={ComplianceFramework.SOX, ComplianceFramework.SOC2}
        )
        
    async def log_security_event(self, event_type: str, severity: AuditLevel,
                               details: Dict[str, Any] = None) -> str:
        """Log security event"""
        return await self.log(
            level=severity,
            category=AuditCategory.SECURITY,
            event_type=event_type,
            details=details,
            compliance_tags={ComplianceFramework.ISO_27001, ComplianceFramework.NIST}
        )