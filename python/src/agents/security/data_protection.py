"""
Data Protection Service
Enterprise data protection with DLP, classification, and privacy controls
"""

import asyncio
import logging
import json
import re
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor

from .encryption_service import EncryptionService
from .access_control import AccessControlManager, AccessDecision

logger = logging.getLogger(__name__)


class DataSensitivity(Enum):
    """Data sensitivity levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    HIGHLY_RESTRICTED = "highly_restricted"


class DataClassification(Enum):
    """Data classification types"""
    PERSONAL_IDENTIFIABLE_INFORMATION = "pii"
    PROTECTED_HEALTH_INFORMATION = "phi"
    PAYMENT_CARD_DATA = "pci"
    INTELLECTUAL_PROPERTY = "ip"
    FINANCIAL_DATA = "financial"
    LEGAL_DATA = "legal"
    HR_DATA = "hr"
    CUSTOMER_DATA = "customer"
    SYSTEM_DATA = "system"
    PUBLIC_DATA = "public"


class DataProtectionAction(Enum):
    """Data protection actions"""
    ALLOW = "allow"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    ENCRYPT = "encrypt"
    MASK = "mask"
    REDACT = "redact"
    ALERT = "alert"
    LOG = "log"


@dataclass
class DataPattern:
    """Data pattern for classification"""
    pattern_id: str
    name: str
    description: str
    classification: DataClassification
    pattern: str  # Regex pattern
    confidence: float  # 0.0 to 1.0
    context_patterns: List[str] = field(default_factory=list)  # Additional context keywords
    enabled: bool = True
    priority: int = 1

    def matches(self, text: str) -> Tuple[bool, float]:
        """Check if pattern matches text"""
        match = re.search(self.pattern, text, re.IGNORECASE)
        if not match:
            return False, 0.0

        # Calculate confidence based on match quality and context
        base_confidence = self.confidence

        # Boost confidence if context patterns are found
        context_boost = 0.0
        for context_pattern in self.context_patterns:
            if re.search(context_pattern, text, re.IGNORECASE):
                context_boost += 0.1

        final_confidence = min(1.0, base_confidence + context_boost)
        return True, final_confidence


@dataclass
class DataClassificationResult:
    """Result of data classification"""
    classification_id: str
    data_id: str
    classification: DataClassification
    sensitivity: DataSensitivity
    confidence: float
    patterns_found: List[str]
    location: str  # file_path, database_table, api_endpoint, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    classified_at: datetime = field(default_factory=datetime.now)
    classifier: str = "system"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classification_id": self.classification_id,
            "data_id": self.data_id,
            "classification": self.classification.value,
            "sensitivity": self.sensitivity.value,
            "confidence": self.confidence,
            "patterns_found": self.patterns_found,
            "location": self.location,
            "metadata": self.metadata,
            "classified_at": self.classified_at.isoformat(),
            "classifier": self.classifier
        }


@dataclass
class DataProtectionPolicy:
    """Data protection policy"""
    policy_id: str
    name: str
    description: str
    classification: DataClassification
    actions: List[DataProtectionAction]
    conditions: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)

    def applies_to(self, classification_result: DataClassificationResult,
                   context: Dict[str, Any]) -> bool:
        """Check if policy applies to classification result"""
        if not self.enabled:
            return False

        if classification_result.classification != self.classification:
            return False

        # Check conditions
        for condition_key, condition_value in self.conditions.items():
            if condition_key == "min_confidence":
                if classification_result.confidence < condition_value:
                    return False
            elif condition_key == "sensitivity":
                if classification_result.sensitivity.value != condition_value:
                    return False
            elif condition_key == "location":
                if condition_value not in classification_result.location:
                    return False

        return True


@dataclass
class DataProtectionEvent:
    """Data protection event"""
    event_id: str
    event_type: str  # classification, policy_violation, data_access, etc.
    data_id: str
    classification: Optional[DataClassification] = None
    sensitivity: Optional[DataSensitivity] = None
    action: Optional[DataProtectionAction] = None
    user_id: Optional[str] = None
    resource: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    severity: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "data_id": self.data_id,
            "classification": self.classification.value if self.classification else None,
            "sensitivity": self.sensitivity.value if self.sensitivity else None,
            "action": self.action.value if self.action else None,
            "user_id": self.user_id,
            "resource": self.resource,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity
        }


class DataClassifier:
    """Data classification engine"""

    def __init__(self):
        self.patterns: Dict[str, DataPattern] = {}
        self.classification_cache: Dict[str, List[DataClassificationResult]] = {}
        self.cache_ttl = timedelta(hours=24)

        self._initialize_default_patterns()

    def _initialize_default_patterns(self) -> None:
        """Initialize default data classification patterns"""

        # PII Patterns
        ssn_pattern = DataPattern(
            pattern_id="ssn",
            name="Social Security Number",
            description="US Social Security Number pattern",
            classification=DataClassification.PERSONAL_IDENTIFIABLE_INFORMATION,
            pattern=r'\b\d{3}-\d{2}-\d{4}\b|\b\d{3}\s\d{2}\s\d{4}\b|\b\d{9}\b',
            confidence=0.9,
            context_patterns=["ssn", "social security", "tax id", "taxpayer"],
            priority=10
        )
        self.patterns[ssn_pattern.pattern_id] = ssn_pattern

        email_pattern = DataPattern(
            pattern_id="email",
            name="Email Address",
            description="Email address pattern",
            classification=DataClassification.PERSONAL_IDENTIFIABLE_INFORMATION,
            pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            confidence=0.95,
            context_patterns=["email", "e-mail", "contact"],
            priority=5
        )
        self.patterns[email_pattern.pattern_id] = email_pattern

        phone_pattern = DataPattern(
            pattern_id="phone",
            name="Phone Number",
            description="Phone number pattern",
            classification=DataClassification.PERSONAL_IDENTIFIABLE_INFORMATION,
            pattern=r'\b(\+\d{1,3}\s?)?(\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b',
            confidence=0.85,
            context_patterns=["phone", "tel", "mobile", "cell"],
            priority=5
        )
        self.patterns[phone_pattern.pattern_id] = phone_pattern

        # Credit Card Pattern
        credit_card_pattern = DataPattern(
            pattern_id="credit_card",
            name="Credit Card Number",
            description="Credit card number pattern",
            classification=DataClassification.PAYMENT_CARD_DATA,
            pattern=r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
            confidence=0.95,
            context_patterns=["card", "credit", "visa", "mastercard", "amex"],
            priority=15
        )
        self.patterns[credit_card_pattern.pattern_id] = credit_card_pattern

        # PHI Pattern
        medical_record_pattern = DataPattern(
            pattern_id="medical_record",
            name="Medical Record Number",
            description="Medical record number pattern",
            classification=DataClassification.PROTECTED_HEALTH_INFORMATION,
            pattern=r'\b(MR|MRN|Medical\s*Record)\s*:?\s*\d+\b',
            confidence=0.8,
            context_patterns=["patient", "medical", "health", "diagnosis"],
            priority=12
        )
        self.patterns[medical_record_pattern.pattern_id] = medical_record_pattern

        # Financial Pattern
        bank_account_pattern = DataPattern(
            pattern_id="bank_account",
            name="Bank Account Number",
            description="Bank account number pattern",
            classification=DataClassification.FINANCIAL_DATA,
            pattern=r'\b(account|acct|account\s*number)\s*:?\s*\d{8,17}\b',
            confidence=0.85,
            context_patterns=["bank", "account", "routing", "aba"],
            priority=10
        )
        self.patterns[bank_account_pattern.pattern_id] = bank_account_pattern

        # IP Address Pattern
        ip_address_pattern = DataPattern(
            pattern_id="ip_address",
            name="IP Address",
            description="IP address pattern",
            classification=DataClassification.SYSTEM_DATA,
            pattern=r'\b(?:\d{1,3}\.){3}\d{1,3}\b|\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
            confidence=0.9,
            context_patterns=["ip", "address", "server", "host"],
            priority=3
        )
        self.patterns[ip_address_pattern.pattern_id] = ip_address_pattern

    def classify_data(self, data: str, data_id: str, location: str = "unknown") -> List[DataClassificationResult]:
        """Classify data using pattern matching"""
        results = []

        # Check cache first
        cache_key = f"{data_id}_{hash(data) % 10000}"
        if cache_key in self.classification_cache:
            cached_time = self.classification_cache[cache_key][0].classified_at
            if datetime.now() - cached_time < self.cache_ttl:
                return self.classification_cache[cache_key]

        for pattern in self.patterns.values():
            if not pattern.enabled:
                continue

            matches, confidence = pattern.matches(data)
            if matches and confidence >= 0.7:  # Minimum confidence threshold
                sensitivity = self._determine_sensitivity(pattern.classification)

                result = DataClassificationResult(
                    classification_id=str(uuid.uuid4()),
                    data_id=data_id,
                    classification=pattern.classification,
                    sensitivity=sensitivity,
                    confidence=confidence,
                    patterns_found=[pattern.pattern_id],
                    location=location,
                    metadata={
                        "pattern_name": pattern.name,
                        "match_quality": confidence
                    }
                )
                results.append(result)

        # Sort by confidence and priority
        results.sort(key=lambda x: (x.confidence, self.patterns.get(x.patterns_found[0], DataPattern("", "", "", DataClassification.PUBLIC, "", 0)).priority), reverse=True)

        # Cache results
        if results:
            self.classification_cache[cache_key] = results

        return results

    def _determine_sensitivity(self, classification: DataClassification) -> DataSensitivity:
        """Determine sensitivity level based on classification"""
        sensitivity_mapping = {
            DataClassification.PERSONAL_IDENTIFIABLE_INFORMATION: DataSensitivity.RESTRICTED,
            DataClassification.PROTECTED_HEALTH_INFORMATION: DataSensitivity.HIGHLY_RESTRICTED,
            DataClassification.PAYMENT_CARD_DATA: DataSensitivity.HIGHLY_RESTRICTED,
            DataClassification.INTELLECTUAL_PROPERTY: DataSensitivity.RESTRICTED,
            DataClassification.FINANCIAL_DATA: DataSensitivity.CONFIDENTIAL,
            DataClassification.LEGAL_DATA: DataSensitivity.CONFIDENTIAL,
            DataClassification.HR_DATA: DataSensitivity.CONFIDENTIAL,
            DataClassification.CUSTOMER_DATA: DataSensitivity.CONFIDENTIAL,
            DataClassification.SYSTEM_DATA: DataSensitivity.INTERNAL,
            DataClassification.PUBLIC_DATA: DataSensitivity.PUBLIC
        }

        return sensitivity_mapping.get(classification, DataSensitivity.INTERNAL)

    def add_pattern(self, pattern: DataPattern) -> None:
        """Add custom classification pattern"""
        self.patterns[pattern.pattern_id] = pattern
        # Clear cache when patterns change
        self.classification_cache.clear()

    def remove_pattern(self, pattern_id: str) -> bool:
        """Remove classification pattern"""
        if pattern_id in self.patterns:
            del self.patterns[pattern_id]
            self.classification_cache.clear()
            return True
        return False

    def get_patterns(self) -> List[Dict[str, Any]]:
        """Get all classification patterns"""
        return [
            {
                "pattern_id": p.pattern_id,
                "name": p.name,
                "description": p.description,
                "classification": p.classification.value,
                "confidence": p.confidence,
                "enabled": p.enabled,
                "priority": p.priority
            }
            for p in self.patterns.values()
        ]


class DataProtectionEngine:
    """Main data protection engine"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.classifier = DataClassifier()
        self.encryption_service = EncryptionService(config.get('encryption', {}))
        self.access_control = AccessControlManager(config.get('access_control', {}))

        self.policies: Dict[str, DataProtectionPolicy] = {}
        self.protection_events: List[DataProtectionEvent] = []
        self.quarantined_data: Dict[str, Dict[str, Any]] = {}

        # DLP settings
        self.dlp_enabled = self.config.get('dlp_enabled', True)
        self.auto_encrypt = self.config.get('auto_encrypt', True)
        self.block_high_risk = self.config.get('block_high_risk', True)

        # Metrics
        self.metrics: Dict[str, int] = {
            "data_classified": 0,
            "policies_violated": 0,
            "data_encrypted": 0,
            "data_blocked": 0,
            "alerts_generated": 0
        }

        # Initialize default policies
        self._initialize_default_policies()

    def _initialize_default_policies(self) -> None:
        """Initialize default data protection policies"""

        # PII Protection Policy
        pii_policy = DataProtectionPolicy(
            policy_id="pii_protection",
            name="PII Data Protection",
            description="Protect personally identifiable information",
            classification=DataClassification.PERSONAL_IDENTIFIABLE_INFORMATION,
            actions=[DataProtectionAction.ENCRYPT, DataProtectionAction.ALERT],
            conditions={
                "min_confidence": 0.8,
                "sensitivity": "restricted"
            },
            priority=10
        )
        self.policies[pii_policy.policy_id] = pii_policy

        # PHI Protection Policy
        phi_policy = DataProtectionPolicy(
            policy_id="phi_protection",
            name="PHI Data Protection",
            description="Protect protected health information",
            classification=DataClassification.PROTECTED_HEALTH_INFORMATION,
            actions=[DataProtectionAction.ENCRYPT, DataProtectionAction.ALERT, DataProtectionAction.LOG],
            conditions={
                "min_confidence": 0.8,
                "sensitivity": "highly_restricted"
            },
            priority=15
        )
        self.policies[phi_policy.policy_id] = phi_policy

        # Payment Card Data Policy
        pci_policy = DataProtectionPolicy(
            policy_id="pci_protection",
            name="PCI Data Protection",
            description="Protect payment card industry data",
            classification=DataClassification.PAYMENT_CARD_DATA,
            actions=[DataProtectionAction.BLOCK, DataProtectionAction.ALERT, DataProtectionAction.QUARANTINE],
            conditions={
                "min_confidence": 0.9,
                "sensitivity": "highly_restricted"
            },
            priority=20
        )
        self.policies[pci_policy.policy_id] = pci_policy

        # Intellectual Property Policy
        ip_policy = DataProtectionPolicy(
            policy_id="ip_protection",
            name="Intellectual Property Protection",
            description="Protect intellectual property",
            classification=DataClassification.INTELLECTUAL_PROPERTY,
            actions=[DataProtectionAction.ENCRYPT, DataProtectionAction.LOG],
            conditions={
                "min_confidence": 0.7,
                "sensitivity": "restricted"
            },
            priority=8
        )
        self.policies[ip_policy.policy_id] = ip_policy

        logger.info("Initialized default data protection policies")

    async def classify_and_protect_data(self, data: str, data_id: str, location: str = "unknown",
                                      user_id: Optional[str] = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Classify data and apply protection policies"""
        context = context or {}

        # Classify data
        classification_results = self.classifier.classify_data(data, data_id, location)
        self.metrics["data_classified"] += 1

        # Log classification event
        if classification_results:
            for result in classification_results:
                event = DataProtectionEvent(
                    event_id=str(uuid.uuid4()),
                    event_type="classification",
                    data_id=data_id,
                    classification=result.classification,
                    sensitivity=result.sensitivity,
                    user_id=user_id,
                    resource=location,
                    details={
                        "confidence": result.confidence,
                        "patterns_found": result.patterns_found
                    }
                )
                self.protection_events.append(event)

        # Apply protection policies
        protection_actions = []
        protected_data = data

        for result in classification_results:
            applicable_policies = self._get_applicable_policies(result, context)

            for policy in applicable_policies:
                self.metrics["policies_violated"] += 1

                # Apply policy actions
                for action in policy.actions:
                    protected_data = await self._apply_protection_action(
                        action, protected_data, result, user_id, context
                    )
                    protection_actions.append(action)

                # Log policy violation
                event = DataProtectionEvent(
                    event_id=str(uuid.uuid4()),
                    event_type="policy_violation",
                    data_id=data_id,
                    classification=result.classification,
                    sensitivity=result.sensitivity,
                    action=policy.actions[0] if policy.actions else None,
                    user_id=user_id,
                    resource=location,
                    details={
                        "policy_id": policy.policy_id,
                        "policy_name": policy.name,
                        "actions": [action.value for action in policy.actions],
                        "confidence": result.confidence
                    },
                    severity="high" if result.sensitivity in [DataSensitivity.HIGHLY_RESTRICTED, DataSensitivity.RESTRICTED] else "medium"
                )
                self.protection_events.append(event)

        return {
            "original_data": data,
            "protected_data": protected_data,
            "classification_results": [result.to_dict() for result in classification_results],
            "protection_actions": [action.value for action in protection_actions],
            "data_id": data_id
        }

    def _get_applicable_policies(self, classification_result: DataClassificationResult,
                                context: Dict[str, Any]) -> List[DataProtectionPolicy]:
        """Get applicable protection policies for classification result"""
        applicable = []

        for policy in self.policies.values():
            if policy.applies_to(classification_result, context):
                applicable.append(policy)

        # Sort by priority
        applicable.sort(key=lambda p: p.priority, reverse=True)
        return applicable

    async def _apply_protection_action(self, action: DataProtectionAction, data: str,
                                   classification_result: DataClassificationResult,
                                   user_id: Optional[str], context: Dict[str, Any]) -> str:
        """Apply data protection action"""

        if action == DataProtectionAction.ALLOW:
            return data

        elif action == DataProtectionAction.ENCRYPT:
            if self.auto_encrypt:
                encrypted_data = self.encryption_service.encrypt_data(data)
                self.metrics["data_encrypted"] += 1
                return encrypted_data
            return data

        elif action == DataProtectionAction.MASK:
            return self._mask_sensitive_data(data, classification_result)

        elif action == DataProtectionAction.REDACT:
            return self._redact_sensitive_data(data, classification_result)

        elif action == DataProtectionAction.BLOCK:
            if self.block_high_risk:
                self.metrics["data_blocked"] += 1
                return "[DATA BLOCKED - CONTAINS SENSITIVE INFORMATION]"

        elif action == DataProtectionAction.QUARANTINE:
            self.quarantined_data[classification_result.data_id] = {
                "data": data,
                "classification": classification_result.to_dict(),
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "context": context
            }
            return "[DATA QUARANTINED - PENDING REVIEW]"

        elif action == DataProtectionAction.ALERT:
            self.metrics["alerts_generated"] += 1
            # Alert would be sent to monitoring system
            logger.warning(f"Data protection alert for {classification_result.data_id}")

        elif action == DataProtectionAction.LOG:
            # Event is already logged
            pass

        return data

    def _mask_sensitive_data(self, data: str, classification_result: DataClassificationResult) -> str:
        """Mask sensitive data patterns"""
        masked_data = data

        for pattern_id in classification_result.patterns_found:
            pattern = self.classifier.patterns.get(pattern_id)
            if pattern:
                # Replace matches with masked version
                matches = re.finditer(pattern.pattern, data, re.IGNORECASE)
                for match in matches:
                    if pattern.classification == DataClassification.PERSONAL_IDENTIFIABLE_INFORMATION:
                        # Mask PII with asterisks
                        masked_value = '*' * len(match.group())
                        masked_data = masked_data.replace(match.group(), masked_value)
                    elif pattern.classification == DataClassification.PAYMENT_CARD_DATA:
                        # Mask credit card showing only last 4 digits
                        card_number = match.group().replace(' ', '').replace('-', '')
                        if len(card_number) >= 4:
                            masked_value = '*' * (len(card_number) - 4) + card_number[-4:]
                            # Format back to original spacing
                            if '-' in match.group():
                                # Add dashes back
                                formatted_masked = ''
                                dash_positions = [i for i, char in enumerate(match.group()) if char == '-']
                                masked_iter = iter(masked_value)
                                for i, char in enumerate(match.group()):
                                    if char == '-':
                                        formatted_masked += '-'
                                    else:
                                        formatted_masked += next(masked_iter, '*')
                                masked_data = masked_data.replace(match.group(), formatted_masked)
                            else:
                                masked_data = masked_data.replace(match.group(), masked_value)

        return masked_data

    def _redact_sensitive_data(self, data: str, classification_result: DataClassificationResult) -> str:
        """Redact (completely remove) sensitive data patterns"""
        redacted_data = data

        for pattern_id in classification_result.patterns_found:
            pattern = self.classifier.patterns.get(pattern_id)
            if pattern:
                # Remove all matches
                redacted_data = re.sub(pattern.pattern, "[REDACTED]", redacted_data, flags=re.IGNORECASE)

        return redacted_data

    async def check_data_access(self, user_id: str, data_id: str, data: str,
                              action: str = "read", context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check if user can access sensitive data"""
        context = context or {}

        # Classify data
        classification_results = self.classifier.classify_data(data, data_id, "access_check")

        if not classification_results:
            # No sensitive data found, allow access
            return {"allowed": True, "reason": "No sensitive data detected"}

        # Get most sensitive classification
        most_sensitive = max(classification_results, key=lambda x: x.sensitivity.value)

        # Check access control
        access_response = await self.access_control.check_access(
            user_id, f"data://{data_id}", action, context
        )

        if access_response.decision != AccessDecision.PERMIT:
            return {
                "allowed": False,
                "reason": f"Access control denied: {access_response.reason}",
                "classification": most_sensitive.to_dict()
            }

        # Check if user has appropriate clearance for data sensitivity
        user_clearance = self._get_user_clearance(user_id)
        data_clearance_required = self._get_clearance_for_sensitivity(most_sensitive.sensitivity)

        if user_clearance < data_clearance_required:
            return {
                "allowed": False,
                "reason": f"Insufficient clearance for {most_sensitive.sensitivity.value} data",
                "classification": most_sensitive.to_dict()
            }

        return {
            "allowed": True,
            "reason": "Access permitted with appropriate clearance",
            "classification": most_sensitive.to_dict(),
            "protection_applied": self.auto_encrypt
        }

    def _get_user_clearance(self, user_id: str) -> int:
        """Get user clearance level (1-5)"""
        # This would integrate with user management system
        # For now, return default clearance
        return 3  # Medium clearance

    def _get_clearance_for_sensitivity(self, sensitivity: DataSensitivity) -> int:
        """Get required clearance level for data sensitivity"""
        clearance_mapping = {
            DataSensitivity.PUBLIC: 1,
            DataSensitivity.INTERNAL: 2,
            DataSensitivity.CONFIDENTIAL: 3,
            DataSensitivity.RESTRICTED: 4,
            DataSensitivity.HIGHLY_RESTRICTED: 5
        }
        return clearance_mapping.get(sensitivity, 1)

    def scan_file_for_sensitive_data(self, file_path: str, file_id: str = None) -> Dict[str, Any]:
        """Scan file for sensitive data"""
        if not file_id:
            file_id = str(uuid.uuid4())

        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()

            # Classify content
            classification_results = self.classifier.classify_data(content, file_id, file_path)

            # Get file metadata
            file_stat = os.stat(file_path)
            file_info = {
                "size": file_stat.st_size,
                "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                "mime_type": mimetypes.guess_type(file_path)[0] or "unknown"
            }

            return {
                "file_id": file_id,
                "file_path": file_path,
                "file_info": file_info,
                "classification_results": [result.to_dict() for result in classification_results],
                "risk_score": self._calculate_file_risk_score(classification_results),
                "recommendations": self._generate_file_recommendations(classification_results)
            }

        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {str(e)}")
            return {
                "file_id": file_id,
                "file_path": file_path,
                "error": str(e)
            }

    def _calculate_file_risk_score(self, classification_results: List[DataClassificationResult]) -> float:
        """Calculate risk score for file based on classifications"""
        if not classification_results:
            return 0.0

        base_score = 0.0

        for result in classification_results:
            # Base score from classification
            classification_scores = {
                DataClassification.PERSONAL_IDENTIFIABLE_INFORMATION: 30,
                DataClassification.PROTECTED_HEALTH_INFORMATION: 40,
                DataClassification.PAYMENT_CARD_DATA: 50,
                DataClassification.INTELLECTUAL_PROPERTY: 35,
                DataClassification.FINANCIAL_DATA: 25,
                DataClassification.LEGAL_DATA: 20,
                DataClassification.HR_DATA: 15,
                DataClassification.CUSTOMER_DATA: 10,
                DataClassification.SYSTEM_DATA: 5,
                DataClassification.PUBLIC_DATA: 0
            }

            base_score += classification_scores.get(result.classification, 0)

            # Weight by confidence
            base_score *= result.confidence

        # Normalize to 0-100 scale
        return min(100.0, base_score)

    def _generate_file_recommendations(self, classification_results: List[DataClassificationResult]) -> List[str]:
        """Generate protection recommendations for file"""
        recommendations = []

        if not classification_results:
            recommendations.append("No sensitive data detected - standard protection recommended")
            return recommendations

        # Analyze classifications
        has_pii = any(r.classification == DataClassification.PERSONAL_IDENTIFIABLE_INFORMATION for r in classification_results)
        has_phi = any(r.classification == DataClassification.PROTECTED_HEALTH_INFORMATION for r in classification_results)
        has_pci = any(r.classification == DataClassification.PAYMENT_CARD_DATA for r in classification_results)
        has_ip = any(r.classification == DataClassification.INTELLECTUAL_PROPERTY for r in classification_results)

        # Generate recommendations
        if has_pci:
            recommendations.append("CRITICAL: File contains payment card data - immediate encryption required")
            recommendations.append("Consider removing or tokenizing credit card numbers")

        if has_phi:
            recommendations.append("HIGH: File contains protected health information - encryption and access controls required")
            recommendations.append("Ensure HIPAA compliance measures are in place")

        if has_pii:
            recommendations.append("MEDIUM: File contains personally identifiable information - consider encryption")

        if has_ip:
            recommendations.append("MEDIUM: File contains intellectual property - apply access controls and encryption")

        # General recommendations
        high_confidence_classifications = [r for r in classification_results if r.confidence >= 0.9]
        if high_confidence_classifications:
            recommendations.append("High-confidence sensitive data detected - implement strict protection measures")

        return recommendations

    def get_data_protection_metrics(self) -> Dict[str, Any]:
        """Get data protection metrics"""
        return {
            "classification_metrics": {
                "total_classifications": self.metrics["data_classified"],
                "patterns_available": len(self.classifier.patterns),
                "cache_size": len(self.classifier.classification_cache)
            },
            "protection_metrics": {
                "policies_violated": self.metrics["policies_violated"],
                "data_encrypted": self.metrics["data_encrypted"],
                "data_blocked": self.metrics["data_blocked"],
                "alerts_generated": self.metrics["alerts_generated"]
            },
            "quarantine_metrics": {
                "quarantined_items": len(self.quarantined_data),
                "quarantined_by_classification": self._get_quarantine_by_classification()
            },
            "event_summary": {
                "total_events": len(self.protection_events),
                "events_by_type": self._get_events_by_type(),
                "recent_events": len([e for e in self.protection_events if datetime.now() - e.timestamp < timedelta(hours=24)])
            }
        }

    def _get_quarantine_by_classification(self) -> Dict[str, int]:
        """Get quarantine statistics by classification"""
        classification_counts = defaultdict(int)
        for item in self.quarantined_data.values():
            if "classification" in item:
                classification = item["classification"]["classification"]
                classification_counts[classification] += 1
        return dict(classification_counts)

    def _get_events_by_type(self) -> Dict[str, int]:
        """Get event statistics by type"""
        event_counts = defaultdict(int)
        for event in self.protection_events:
            event_counts[event.event_type] += 1
        return dict(event_counts)

    def create_protection_policy(self, name: str, description: str, classification: DataClassification,
                               actions: List[DataProtectionAction], conditions: Dict[str, Any] = None,
                               priority: int = 1) -> DataProtectionPolicy:
        """Create new data protection policy"""
        policy = DataProtectionPolicy(
            policy_id=str(uuid.uuid4()),
            name=name,
            description=description,
            classification=classification,
            actions=actions,
            conditions=conditions or {},
            priority=priority
        )

        self.policies[policy.policy_id] = policy
        logger.info(f"Created data protection policy: {policy.name}")
        return policy

    def get_policy(self, policy_id: str) -> Optional[DataProtectionPolicy]:
        """Get protection policy by ID"""
        return self.policies.get(policy_id)

    def list_policies(self) -> List[Dict[str, Any]]:
        """List all protection policies"""
        return [
            {
                "policy_id": p.policy_id,
                "name": p.name,
                "description": p.description,
                "classification": p.classification.value,
                "actions": [action.value for action in p.actions],
                "enabled": p.enabled,
                "priority": p.priority
            }
            for p in self.policies.values()
        ]

    def get_quarantined_items(self) -> List[Dict[str, Any]]:
        """Get quarantined data items"""
        return [
            {
                "data_id": data_id,
                "classification": item["classification"]["classification"],
                "sensitivity": item["classification"]["sensitivity"],
                "user_id": item["user_id"],
                "timestamp": item["timestamp"]
            }
            for data_id, item in self.quarantined_data.items()
        ]

    def release_quarantined_data(self, data_id: str, approved_by: str, reason: str) -> bool:
        """Release quarantined data"""
        if data_id in self.quarantined_data:
            # Log release event
            item = self.quarantined_data[data_id]
            event = DataProtectionEvent(
                event_id=str(uuid.uuid4()),
                event_type="quarantine_release",
                data_id=data_id,
                details={
                    "approved_by": approved_by,
                    "reason": reason,
                    "original_classification": item["classification"]
                }
            )
            self.protection_events.append(event)

            # Remove from quarantine
            del self.quarantined_data[data_id]

            logger.info(f"Released quarantined data {data_id} approved by {approved_by}")
            return True

        return False