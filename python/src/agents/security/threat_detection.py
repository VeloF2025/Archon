"""
Advanced Threat Detection System
AI-powered threat detection with machine learning and behavioral analysis
"""

import asyncio
import logging
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import re
import hashlib
from collections import defaultdict, deque
import ipaddress
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import joblib
import pickle

logger = logging.getLogger(__name__)


class ThreatType(Enum):
    """Types of security threats"""
    MALWARE = "malware"
    PHISHING = "phishing"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    BRUTE_FORCE = "brute_force"
    DDoS = "ddos"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    INSIDER_THREAT = "insider_threat"
    APT = "apt"  # Advanced Persistent Threat
    ZERO_DAY = "zero_day"
    RANSOMWARE = "ransomware"


class ThreatSeverity(Enum):
    """Threat severity levels"""
    INFO = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


class ThreatStatus(Enum):
    """Threat investigation status"""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"


@dataclass
class ThreatIndicator:
    """Indicator of Compromise (IoC)"""
    indicator_id: str
    indicator_type: str  # ip, domain, hash, url, email, etc.
    value: str
    threat_types: List[ThreatType]
    confidence: float  # 0.0 to 1.0
    first_seen: datetime
    last_seen: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_match(self, data: str) -> bool:
        """Check if data matches this indicator"""
        if self.indicator_type == "ip":
            return self._match_ip(data)
        elif self.indicator_type == "domain":
            return self._match_domain(data)
        elif self.indicator_type == "hash":
            return self.value.lower() == data.lower()
        elif self.indicator_type == "url":
            return self.value in data
        elif self.indicator_type == "regex":
            return bool(re.search(self.value, data, re.IGNORECASE))
        else:
            return self.value.lower() in data.lower()
    
    def _match_ip(self, data: str) -> bool:
        """Match IP address or range"""
        try:
            if "/" in self.value:  # CIDR range
                network = ipaddress.ip_network(self.value, strict=False)
                return ipaddress.ip_address(data) in network
            else:
                return self.value == data
        except ValueError:
            return False
    
    def _match_domain(self, data: str) -> bool:
        """Match domain name"""
        return data.endswith(self.value) or self.value in data


@dataclass
class ThreatEvent:
    """Detected threat event"""
    event_id: str
    threat_type: ThreatType
    severity: ThreatSeverity
    status: ThreatStatus = ThreatStatus.DETECTED
    source_ip: Optional[str] = None
    target_ip: Optional[str] = None
    user_id: Optional[str] = None
    description: str = ""
    raw_data: Dict[str, Any] = field(default_factory=dict)
    indicators: List[ThreatIndicator] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 0.5
    false_positive_probability: float = 0.0
    mitigation_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "threat_type": self.threat_type.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "source_ip": self.source_ip,
            "target_ip": self.target_ip,
            "user_id": self.user_id,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "false_positive_probability": self.false_positive_probability,
            "mitigation_actions": self.mitigation_actions
        }


@dataclass
class ThreatHunting:
    """Threat hunting rules and queries"""
    hunt_id: str
    name: str
    description: str
    threat_types: List[ThreatType]
    query: str  # Query to search for threats
    indicators: List[ThreatIndicator]
    severity_threshold: ThreatSeverity
    enabled: bool = True
    last_run: Optional[datetime] = None
    
    def execute(self, data_source: Any) -> List[ThreatEvent]:
        """Execute threat hunting query"""
        # Simplified implementation - would integrate with SIEM/log analysis
        threats = []
        
        # Parse query and search data
        # This would be much more sophisticated in real implementation
        
        return threats


class AnomalyDetector:
    """ML-based anomaly detection for threat identification"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, training_data: np.ndarray) -> None:
        """Train anomaly detection models"""
        # Scale features
        scaled_data = self.scaler.fit_transform(training_data)
        
        # Train Isolation Forest
        self.isolation_forest.fit(scaled_data)
        
        # Train DBSCAN
        self.dbscan.fit(scaled_data)
        
        self.is_trained = True
        logger.info("Anomaly detection models trained")
        
    def detect_anomalies(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies in data"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Scale data
        scaled_data = self.scaler.transform(data)
        
        # Isolation Forest predictions (-1 for anomalies, 1 for normal)
        iso_predictions = self.isolation_forest.predict(scaled_data)
        iso_scores = self.isolation_forest.decision_function(scaled_data)
        
        # DBSCAN predictions (-1 for anomalies/noise)
        dbscan_labels = self.dbscan.fit_predict(scaled_data)
        
        # Combine results
        anomalies = (iso_predictions == -1) | (dbscan_labels == -1)
        
        return anomalies, iso_scores
        
    def save_model(self, filepath: str) -> None:
        """Save trained model"""
        model_data = {
            'isolation_forest': self.isolation_forest,
            'dbscan': self.dbscan,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        
    def load_model(self, filepath: str) -> None:
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.isolation_forest = model_data['isolation_forest']
        self.dbscan = model_data['dbscan']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']


class BehavioralAnalyzer:
    """Analyze user and system behavior for threats"""
    
    def __init__(self):
        self.user_baselines: Dict[str, Dict] = {}
        self.system_baselines: Dict[str, Dict] = {}
        
    def create_user_baseline(self, user_id: str, activities: List[Dict[str, Any]]) -> None:
        """Create baseline behavior profile for user"""
        if not activities:
            return
            
        # Analyze typical activity patterns
        hours = [datetime.fromisoformat(act['timestamp']).hour for act in activities]
        actions = [act.get('action', 'unknown') for act in activities]
        
        baseline = {
            'typical_hours': list(set(hours)),
            'common_actions': list(set(actions)),
            'activity_frequency': len(activities) / 7,  # per day
            'created_at': datetime.now(),
            'sample_size': len(activities)
        }
        
        self.user_baselines[user_id] = baseline
        logger.info(f"Created baseline for user {user_id}")
        
    def detect_behavioral_anomaly(self, user_id: str, activity: Dict[str, Any]) -> Dict[str, Any]:
        """Detect if activity is anomalous for user"""
        baseline = self.user_baselines.get(user_id)
        if not baseline:
            return {"anomalous": False, "reason": "No baseline available"}
            
        anomalies = []
        
        # Check time anomaly
        activity_hour = datetime.fromisoformat(activity['timestamp']).hour
        if activity_hour not in baseline['typical_hours']:
            anomalies.append("unusual_time")
            
        # Check action anomaly
        action = activity.get('action', 'unknown')
        if action not in baseline['common_actions']:
            anomalies.append("unusual_action")
            
        # Check volume anomaly (simplified)
        if activity.get('data_volume', 0) > baseline.get('avg_data_volume', 1000000) * 10:
            anomalies.append("unusual_volume")
            
        return {
            "anomalous": len(anomalies) > 0,
            "anomalies": anomalies,
            "confidence": min(1.0, len(anomalies) * 0.3)
        }


class ThreatIntelligence:
    """Threat intelligence feeds and IoC management"""
    
    def __init__(self):
        self.indicators: Dict[str, ThreatIndicator] = {}
        self.feeds: Dict[str, Dict] = {}
        self.last_update: Dict[str, datetime] = {}
        
    def add_indicator(self, indicator: ThreatIndicator) -> None:
        """Add threat indicator"""
        self.indicators[indicator.indicator_id] = indicator
        
    def check_indicators(self, data: Dict[str, Any]) -> List[ThreatIndicator]:
        """Check data against threat indicators"""
        matches = []
        
        for indicator in self.indicators.values():
            # Check various data fields
            for field, value in data.items():
                if isinstance(value, str) and indicator.is_match(value):
                    matches.append(indicator)
                    break
                    
        return matches
        
    def update_threat_feed(self, feed_name: str, indicators: List[Dict[str, Any]]) -> None:
        """Update threat intelligence feed"""
        updated_count = 0
        
        for indicator_data in indicators:
            indicator = ThreatIndicator(
                indicator_id=str(uuid.uuid4()),
                indicator_type=indicator_data.get('type', 'unknown'),
                value=indicator_data.get('value', ''),
                threat_types=[ThreatType(t) for t in indicator_data.get('threat_types', [])],
                confidence=indicator_data.get('confidence', 0.5),
                first_seen=datetime.fromisoformat(indicator_data.get('first_seen', datetime.now().isoformat())),
                last_seen=datetime.fromisoformat(indicator_data.get('last_seen', datetime.now().isoformat())),
                source=feed_name
            )
            
            self.add_indicator(indicator)
            updated_count += 1
            
        self.last_update[feed_name] = datetime.now()
        logger.info(f"Updated {feed_name} feed with {updated_count} indicators")


class ThreatDetectionSystem:
    """Main threat detection system orchestrating all components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.threat_events: deque = deque(maxlen=100000)
        self.active_threats: Dict[str, ThreatEvent] = {}
        self.hunting_rules: Dict[str, ThreatHunting] = {}
        self.detection_rules: List[Dict[str, Any]] = []
        
        # Initialize components
        self.anomaly_detector = AnomalyDetector()
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.threat_intelligence = ThreatIntelligence()
        
        # Metrics
        self.metrics: Dict[str, int] = defaultdict(int)
        
        # Event handlers
        self.event_handlers: Dict[ThreatType, List[Callable]] = defaultdict(list)
        
        # Load default detection rules
        self._load_default_rules()
        
        # Load threat intelligence
        self._load_default_indicators()
        
    def _load_default_rules(self) -> None:
        """Load default threat detection rules"""
        # SQL Injection detection
        sql_injection_rule = {
            'name': 'SQL Injection Detection',
            'threat_type': ThreatType.SQL_INJECTION,
            'patterns': [
                r'(\bUNION\b.*\bSELECT\b)|(\bSELECT\b.*\bFROM\b.*\bWHERE\b.*[\'"]\s*(OR|AND)\s*[\'"]\s*=\s*[\'"]\s*)',
                r'[\'"]\s*;\s*(DROP|DELETE|INSERT|UPDATE)\b',
                r'\b(EXEC|EXECUTE)\s*\(',
                r'[\'"]\s*(OR|AND)\s*[\'"]\d+[\'"]\s*=\s*[\'"]\d+[\'"]\s*',
            ],
            'severity': ThreatSeverity.HIGH,
            'confidence_threshold': 0.7
        }
        self.detection_rules.append(sql_injection_rule)
        
        # XSS detection
        xss_rule = {
            'name': 'Cross-Site Scripting Detection',
            'threat_type': ThreatType.XSS,
            'patterns': [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'on\w+\s*=',
                r'<iframe[^>]*>',
                r'eval\s*\(',
            ],
            'severity': ThreatSeverity.MEDIUM,
            'confidence_threshold': 0.6
        }
        self.detection_rules.append(xss_rule)
        
        # Brute force detection
        brute_force_rule = {
            'name': 'Brute Force Attack Detection',
            'threat_type': ThreatType.BRUTE_FORCE,
            'conditions': [
                {'field': 'failed_attempts', 'operator': '>', 'value': 10},
                {'field': 'time_window', 'operator': '<', 'value': 300},  # 5 minutes
            ],
            'severity': ThreatSeverity.HIGH,
            'confidence_threshold': 0.8
        }
        self.detection_rules.append(brute_force_rule)
        
    def _load_default_indicators(self) -> None:
        """Load default threat indicators"""
        # Known malicious IPs (examples)
        malicious_ips = [
            "198.51.100.1",  # TEST-NET-2
            "203.0.113.1",   # TEST-NET-3
        ]
        
        for ip in malicious_ips:
            indicator = ThreatIndicator(
                indicator_id=str(uuid.uuid4()),
                indicator_type="ip",
                value=ip,
                threat_types=[ThreatType.MALWARE, ThreatType.APT],
                confidence=0.9,
                first_seen=datetime.now() - timedelta(days=30),
                last_seen=datetime.now(),
                source="default_feed"
            )
            self.threat_intelligence.add_indicator(indicator)
            
        # Known malicious domains
        malicious_domains = [
            "evil.example.com",
            "malware.test.net"
        ]
        
        for domain in malicious_domains:
            indicator = ThreatIndicator(
                indicator_id=str(uuid.uuid4()),
                indicator_type="domain",
                value=domain,
                threat_types=[ThreatType.PHISHING, ThreatType.MALWARE],
                confidence=0.8,
                first_seen=datetime.now() - timedelta(days=15),
                last_seen=datetime.now(),
                source="default_feed"
            )
            self.threat_intelligence.add_indicator(indicator)
            
    async def analyze_event(self, event_data: Dict[str, Any]) -> Optional[ThreatEvent]:
        """Analyze event for potential threats"""
        threats_detected = []
        
        # Check against threat intelligence
        matching_indicators = self.threat_intelligence.check_indicators(event_data)
        if matching_indicators:
            for indicator in matching_indicators:
                threat_event = ThreatEvent(
                    event_id=str(uuid.uuid4()),
                    threat_type=indicator.threat_types[0] if indicator.threat_types else ThreatType.MALWARE,
                    severity=self._calculate_severity(indicator.confidence),
                    source_ip=event_data.get('source_ip'),
                    target_ip=event_data.get('target_ip'),
                    user_id=event_data.get('user_id'),
                    description=f"Threat indicator match: {indicator.indicator_type} {indicator.value}",
                    raw_data=event_data,
                    indicators=[indicator],
                    confidence=indicator.confidence
                )
                threats_detected.append(threat_event)
        
        # Check against detection rules
        for rule in self.detection_rules:
            if await self._evaluate_rule(rule, event_data):
                threat_event = ThreatEvent(
                    event_id=str(uuid.uuid4()),
                    threat_type=rule['threat_type'],
                    severity=rule['severity'],
                    source_ip=event_data.get('source_ip'),
                    target_ip=event_data.get('target_ip'),
                    user_id=event_data.get('user_id'),
                    description=f"Detection rule triggered: {rule['name']}",
                    raw_data=event_data,
                    confidence=rule.get('confidence_threshold', 0.7)
                )
                threats_detected.append(threat_event)
        
        # Behavioral analysis
        user_id = event_data.get('user_id')
        if user_id:
            behavioral_result = self.behavioral_analyzer.detect_behavioral_anomaly(user_id, event_data)
            if behavioral_result['anomalous']:
                threat_event = ThreatEvent(
                    event_id=str(uuid.uuid4()),
                    threat_type=ThreatType.INSIDER_THREAT,
                    severity=ThreatSeverity.MEDIUM,
                    source_ip=event_data.get('source_ip'),
                    user_id=user_id,
                    description=f"Behavioral anomaly detected: {', '.join(behavioral_result['anomalies'])}",
                    raw_data=event_data,
                    confidence=behavioral_result['confidence']
                )
                threats_detected.append(threat_event)
        
        # Process detected threats
        highest_severity_threat = None
        if threats_detected:
            # Sort by severity and confidence
            threats_detected.sort(key=lambda t: (t.severity.value, t.confidence), reverse=True)
            highest_severity_threat = threats_detected[0]
            
            # Store threat event
            await self._process_threat_event(highest_severity_threat)
            
        return highest_severity_threat
        
    async def _evaluate_rule(self, rule: Dict[str, Any], event_data: Dict[str, Any]) -> bool:
        """Evaluate detection rule against event data"""
        rule_name = rule.get('name', 'Unknown')
        
        # Pattern-based rules
        if 'patterns' in rule:
            for pattern in rule['patterns']:
                for field, value in event_data.items():
                    if isinstance(value, str) and re.search(pattern, value, re.IGNORECASE):
                        logger.debug(f"Pattern match in rule {rule_name}: {pattern}")
                        return True
        
        # Condition-based rules
        if 'conditions' in rule:
            for condition in rule['conditions']:
                field = condition['field']
                operator = condition['operator']
                expected_value = condition['value']
                
                if field not in event_data:
                    continue
                    
                actual_value = event_data[field]
                
                if operator == '>' and actual_value > expected_value:
                    return True
                elif operator == '<' and actual_value < expected_value:
                    return True
                elif operator == '==' and actual_value == expected_value:
                    return True
                elif operator == '!=' and actual_value != expected_value:
                    return True
        
        return False
        
    def _calculate_severity(self, confidence: float) -> ThreatSeverity:
        """Calculate threat severity based on confidence"""
        if confidence >= 0.9:
            return ThreatSeverity.CRITICAL
        elif confidence >= 0.7:
            return ThreatSeverity.HIGH
        elif confidence >= 0.5:
            return ThreatSeverity.MEDIUM
        elif confidence >= 0.3:
            return ThreatSeverity.LOW
        else:
            return ThreatSeverity.INFO
            
    async def _process_threat_event(self, threat_event: ThreatEvent) -> None:
        """Process detected threat event"""
        # Store event
        self.threat_events.append(threat_event)
        
        # Add to active threats if high severity
        if threat_event.severity.value >= ThreatSeverity.HIGH.value:
            self.active_threats[threat_event.event_id] = threat_event
            
        # Update metrics
        self.metrics[f"threats_{threat_event.threat_type.value}"] += 1
        self.metrics[f"severity_{threat_event.severity.name.lower()}"] += 1
        
        # Trigger event handlers
        handlers = self.event_handlers.get(threat_event.threat_type, [])
        for handler in handlers:
            try:
                await handler(threat_event)
            except Exception as e:
                logger.error(f"Error in threat event handler: {str(e)}")
                
        # Auto-mitigation for critical threats
        if threat_event.severity == ThreatSeverity.CRITICAL:
            await self._auto_mitigate_threat(threat_event)
            
        logger.info(f"Threat detected: {threat_event.threat_type.value} - {threat_event.description}")
        
    async def _auto_mitigate_threat(self, threat_event: ThreatEvent) -> None:
        """Automatically mitigate critical threats"""
        mitigation_actions = []
        
        # Block IP address
        if threat_event.source_ip:
            mitigation_actions.append(f"block_ip:{threat_event.source_ip}")
            
        # Lock user account for insider threats
        if threat_event.threat_type == ThreatType.INSIDER_THREAT and threat_event.user_id:
            mitigation_actions.append(f"lock_user:{threat_event.user_id}")
            
        # Isolate system for malware
        if threat_event.threat_type == ThreatType.MALWARE and threat_event.target_ip:
            mitigation_actions.append(f"isolate_system:{threat_event.target_ip}")
            
        threat_event.mitigation_actions = mitigation_actions
        threat_event.status = ThreatStatus.MITIGATED
        
        logger.warning(f"Auto-mitigated critical threat {threat_event.event_id}: {mitigation_actions}")
        
    def register_threat_handler(self, threat_type: ThreatType, handler: Callable) -> None:
        """Register handler for specific threat type"""
        self.event_handlers[threat_type].append(handler)
        
    def add_detection_rule(self, rule: Dict[str, Any]) -> None:
        """Add custom detection rule"""
        self.detection_rules.append(rule)
        logger.info(f"Added detection rule: {rule.get('name', 'Unknown')}")
        
    def train_anomaly_detector(self, training_data: List[Dict[str, Any]]) -> None:
        """Train anomaly detection model with historical data"""
        if not training_data:
            logger.warning("No training data provided for anomaly detector")
            return
            
        # Convert to numerical features
        features = self._extract_features(training_data)
        
        if features.size > 0:
            self.anomaly_detector.train(features)
            logger.info(f"Trained anomaly detector with {len(training_data)} samples")
        else:
            logger.warning("Could not extract features from training data")
            
    def _extract_features(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """Extract numerical features from event data"""
        features = []
        
        for event in data:
            feature_vector = []
            
            # Time-based features
            timestamp = datetime.fromisoformat(event.get('timestamp', datetime.now().isoformat()))
            feature_vector.extend([
                timestamp.hour,
                timestamp.weekday(),
                timestamp.day
            ])
            
            # Numerical features
            feature_vector.extend([
                event.get('request_size', 0),
                event.get('response_size', 0),
                event.get('response_time', 0),
                len(event.get('user_agent', '')),
                len(event.get('url', ''))
            ])
            
            # Categorical features (simplified encoding)
            method = event.get('method', 'GET')
            feature_vector.append(hash(method) % 1000)
            
            features.append(feature_vector)
            
        return np.array(features)
        
    def get_threat_metrics(self) -> Dict[str, Any]:
        """Get threat detection metrics"""
        total_threats = len(self.threat_events)
        active_threats = len(self.active_threats)
        
        # Calculate threat distribution
        threat_distribution = defaultdict(int)
        severity_distribution = defaultdict(int)
        
        for threat in self.threat_events:
            threat_distribution[threat.threat_type.value] += 1
            severity_distribution[threat.severity.name.lower()] += 1
            
        return {
            "total_threats_detected": total_threats,
            "active_threats": active_threats,
            "threat_distribution": dict(threat_distribution),
            "severity_distribution": dict(severity_distribution),
            "detection_rules_count": len(self.detection_rules),
            "threat_indicators_count": len(self.threat_intelligence.indicators),
            "false_positive_rate": self._calculate_false_positive_rate()
        }
        
    def _calculate_false_positive_rate(self) -> float:
        """Calculate false positive rate"""
        total_events = len(self.threat_events)
        if total_events == 0:
            return 0.0
            
        false_positives = sum(1 for event in self.threat_events 
                            if event.status == ThreatStatus.FALSE_POSITIVE)
        
        return false_positives / total_events
        
    def get_recent_threats(self, hours: int = 24, severity: ThreatSeverity = None) -> List[Dict[str, Any]]:
        """Get recent threat events"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_threats = [
            threat for threat in self.threat_events
            if threat.timestamp > cutoff_time
        ]
        
        if severity:
            recent_threats = [
                threat for threat in recent_threats
                if threat.severity == severity
            ]
            
        return [threat.to_dict() for threat in recent_threats[-100:]]  # Last 100
        
    async def hunt_threats(self, hunt_rule: ThreatHunting) -> List[ThreatEvent]:
        """Execute threat hunting rule"""
        # Simplified implementation
        # In real system, would query SIEM/logs with hunt_rule.query
        
        found_threats = []
        
        # Check recent events against hunting indicators
        recent_events = list(self.threat_events)[-1000:]  # Last 1000 events
        
        for event in recent_events:
            for indicator in hunt_rule.indicators:
                if self.threat_intelligence.check_indicators(event.raw_data):
                    found_threats.append(event)
                    break
                    
        hunt_rule.last_run = datetime.now()
        
        logger.info(f"Threat hunting '{hunt_rule.name}' found {len(found_threats)} threats")
        
        return found_threats
        
    def update_threat_status(self, event_id: str, status: ThreatStatus,
                           notes: str = "") -> bool:
        """Update threat event status"""
        # Find in active threats
        if event_id in self.active_threats:
            self.active_threats[event_id].status = status
            if status in [ThreatStatus.FALSE_POSITIVE, ThreatStatus.RESOLVED]:
                del self.active_threats[event_id]
            return True
            
        # Find in threat events
        for threat in self.threat_events:
            if threat.event_id == event_id:
                threat.status = status
                return True
                
        return False