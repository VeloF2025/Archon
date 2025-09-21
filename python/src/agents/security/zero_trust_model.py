"""
Zero Trust Security Model Implementation
Never trust, always verify - comprehensive zero trust architecture
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import ipaddress
import hashlib
import hmac
from collections import defaultdict, deque
import json
import re

logger = logging.getLogger(__name__)


class TrustLevel(Enum):
    """Trust levels in zero trust model"""
    UNTRUSTED = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERIFIED = 4


class VerificationStatus(Enum):
    """Verification status for entities"""
    UNVERIFIED = "unverified"
    PENDING = "pending"
    VERIFIED = "verified"
    EXPIRED = "expired"
    REVOKED = "revoked"


class EntityType(Enum):
    """Types of entities in zero trust model"""
    USER = "user"
    DEVICE = "device"
    APPLICATION = "application"
    SERVICE = "service"
    NETWORK = "network"
    DATA = "data"


class RiskLevel(Enum):
    """Risk assessment levels"""
    MINIMAL = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    CRITICAL = 5


@dataclass
class TrustEntity:
    """Entity in zero trust model"""
    entity_id: str
    entity_type: EntityType
    name: str
    trust_level: TrustLevel = TrustLevel.UNTRUSTED
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    risk_score: float = 0.0
    last_verified: Optional[datetime] = None
    verification_expiry: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    behaviors: List[Dict[str, Any]] = field(default_factory=list)
    violations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def needs_verification(self) -> bool:
        """Check if entity needs verification"""
        if self.verification_status == VerificationStatus.UNVERIFIED:
            return True
        if self.verification_expiry and datetime.now() > self.verification_expiry:
            return True
        return False
    
    def is_trusted(self, minimum_level: TrustLevel = TrustLevel.MEDIUM) -> bool:
        """Check if entity meets minimum trust level"""
        return (self.trust_level.value >= minimum_level.value and 
                self.verification_status == VerificationStatus.VERIFIED)
    
    def add_behavior(self, action: str, context: Dict[str, Any]) -> None:
        """Add behavior record"""
        behavior = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "context": context,
            "risk_score": self._calculate_behavior_risk(action, context)
        }
        self.behaviors.append(behavior)
        
        # Keep only last 1000 behaviors
        if len(self.behaviors) > 1000:
            self.behaviors = self.behaviors[-1000:]
    
    def _calculate_behavior_risk(self, action: str, context: Dict[str, Any]) -> float:
        """Calculate risk score for behavior"""
        base_risk = {
            "login": 0.1,
            "data_access": 0.3,
            "configuration_change": 0.8,
            "admin_action": 0.9,
            "suspicious_activity": 1.0
        }.get(action, 0.5)
        
        # Adjust based on context
        if context.get("unusual_time"):
            base_risk += 0.2
        if context.get("unusual_location"):
            base_risk += 0.3
        if context.get("failed_attempt"):
            base_risk += 0.5
            
        return min(base_risk, 1.0)


@dataclass
class TrustPolicy:
    """Trust policy for zero trust decisions"""
    policy_id: str
    name: str
    description: str
    entity_types: List[EntityType]
    minimum_trust_level: TrustLevel
    verification_requirements: List[str]
    risk_threshold: float = 0.7
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True
    
    def evaluate(self, entity: TrustEntity, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate trust policy against entity"""
        result = {
            "allowed": False,
            "trust_decision": "deny",
            "reasons": [],
            "required_actions": []
        }
        
        if not self.enabled:
            result["allowed"] = True
            result["trust_decision"] = "allow"
            return result
        
        # Check entity type
        if entity.entity_type not in self.entity_types:
            result["reasons"].append("Entity type not covered by policy")
            return result
        
        # Check trust level
        if not entity.is_trusted(self.minimum_trust_level):
            result["reasons"].append(f"Trust level {entity.trust_level.name} below required {self.minimum_trust_level.name}")
            result["required_actions"].append("increase_trust_level")
        
        # Check risk score
        if entity.risk_score > self.risk_threshold:
            result["reasons"].append(f"Risk score {entity.risk_score} exceeds threshold {self.risk_threshold}")
            result["required_actions"].append("reduce_risk")
        
        # Check verification status
        if entity.needs_verification():
            result["reasons"].append("Entity verification required")
            result["required_actions"].append("verify_entity")
        
        # Evaluate conditions
        for condition in self.conditions:
            if not self._evaluate_condition(condition, entity, context):
                result["reasons"].append(f"Condition failed: {condition.get('description', 'Unknown')}")
        
        # Determine final decision
        if not result["reasons"]:
            result["allowed"] = True
            result["trust_decision"] = "allow"
        elif result["required_actions"]:
            result["trust_decision"] = "challenge"
        else:
            result["trust_decision"] = "deny"
        
        return result
    
    def _evaluate_condition(self, condition: Dict[str, Any], 
                          entity: TrustEntity, context: Dict[str, Any]) -> bool:
        """Evaluate a single condition"""
        condition_type = condition.get("type")
        
        if condition_type == "time_based":
            return self._evaluate_time_condition(condition, context)
        elif condition_type == "location_based":
            return self._evaluate_location_condition(condition, context)
        elif condition_type == "behavior_based":
            return self._evaluate_behavior_condition(condition, entity)
        elif condition_type == "attribute_based":
            return self._evaluate_attribute_condition(condition, entity)
        
        return True
    
    def _evaluate_time_condition(self, condition: Dict[str, Any], 
                               context: Dict[str, Any]) -> bool:
        """Evaluate time-based condition"""
        allowed_hours = condition.get("allowed_hours", [])
        if not allowed_hours:
            return True
        
        current_hour = datetime.now().hour
        return current_hour in allowed_hours
    
    def _evaluate_location_condition(self, condition: Dict[str, Any], 
                                   context: Dict[str, Any]) -> bool:
        """Evaluate location-based condition"""
        allowed_locations = condition.get("allowed_locations", [])
        if not allowed_locations:
            return True
        
        client_ip = context.get("client_ip")
        if not client_ip:
            return False
        
        try:
            client_addr = ipaddress.ip_address(client_ip)
            for location in allowed_locations:
                if client_addr in ipaddress.ip_network(location, strict=False):
                    return True
        except ValueError:
            return False
        
        return False
    
    def _evaluate_behavior_condition(self, condition: Dict[str, Any], 
                                   entity: TrustEntity) -> bool:
        """Evaluate behavior-based condition"""
        max_risk_score = condition.get("max_risk_score", 1.0)
        return entity.risk_score <= max_risk_score
    
    def _evaluate_attribute_condition(self, condition: Dict[str, Any], 
                                    entity: TrustEntity) -> bool:
        """Evaluate attribute-based condition"""
        required_attributes = condition.get("required_attributes", {})
        
        for attr_name, expected_value in required_attributes.items():
            if attr_name not in entity.attributes:
                return False
            if entity.attributes[attr_name] != expected_value:
                return False
        
        return True


@dataclass
class VerificationMethod:
    """Method for entity verification"""
    method_id: str
    name: str
    description: str
    entity_types: List[EntityType]
    trust_level_granted: TrustLevel
    expiry_duration: timedelta
    verification_steps: List[Dict[str, Any]]
    enabled: bool = True
    
    async def verify(self, entity: TrustEntity, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform verification for entity"""
        result = {
            "success": False,
            "trust_level": entity.trust_level,
            "verification_status": VerificationStatus.PENDING,
            "errors": []
        }
        
        if not self.enabled:
            result["errors"].append("Verification method disabled")
            return result
        
        if entity.entity_type not in self.entity_types:
            result["errors"].append("Entity type not supported")
            return result
        
        # Execute verification steps
        for step in self.verification_steps:
            step_result = await self._execute_verification_step(step, entity, context)
            if not step_result["success"]:
                result["errors"].extend(step_result.get("errors", []))
                return result
        
        # Verification successful
        result["success"] = True
        result["trust_level"] = self.trust_level_granted
        result["verification_status"] = VerificationStatus.VERIFIED
        
        return result
    
    async def _execute_verification_step(self, step: Dict[str, Any], 
                                       entity: TrustEntity, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single verification step"""
        step_type = step.get("type")
        
        if step_type == "password_check":
            return await self._verify_password(step, entity, context)
        elif step_type == "mfa_check":
            return await self._verify_mfa(step, entity, context)
        elif step_type == "certificate_check":
            return await self._verify_certificate(step, entity, context)
        elif step_type == "biometric_check":
            return await self._verify_biometric(step, entity, context)
        elif step_type == "device_trust":
            return await self._verify_device_trust(step, entity, context)
        
        return {"success": True}
    
    async def _verify_password(self, step: Dict[str, Any], 
                             entity: TrustEntity, context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify password"""
        # Simplified implementation
        password = context.get("password")
        if not password:
            return {"success": False, "errors": ["Password required"]}
        
        # In real implementation, would check against password store
        if len(password) >= 8:
            return {"success": True}
        else:
            return {"success": False, "errors": ["Invalid password"]}
    
    async def _verify_mfa(self, step: Dict[str, Any], 
                         entity: TrustEntity, context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify MFA token"""
        # Simplified implementation
        mfa_token = context.get("mfa_token")
        if not mfa_token:
            return {"success": False, "errors": ["MFA token required"]}
        
        # In real implementation, would verify against MFA provider
        if len(mfa_token) == 6 and mfa_token.isdigit():
            return {"success": True}
        else:
            return {"success": False, "errors": ["Invalid MFA token"]}
    
    async def _verify_certificate(self, step: Dict[str, Any], 
                                 entity: TrustEntity, context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify certificate"""
        certificate = context.get("certificate")
        if not certificate:
            return {"success": False, "errors": ["Certificate required"]}
        
        # Simplified certificate validation
        return {"success": True}
    
    async def _verify_biometric(self, step: Dict[str, Any], 
                              entity: TrustEntity, context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify biometric data"""
        biometric_data = context.get("biometric_data")
        if not biometric_data:
            return {"success": False, "errors": ["Biometric data required"]}
        
        # Simplified biometric validation
        return {"success": True}
    
    async def _verify_device_trust(self, step: Dict[str, Any], 
                                 entity: TrustEntity, context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify device trust"""
        device_id = context.get("device_id")
        if not device_id:
            return {"success": False, "errors": ["Device ID required"]}
        
        # Check if device is in trusted list
        trusted_devices = entity.attributes.get("trusted_devices", [])
        if device_id in trusted_devices:
            return {"success": True}
        else:
            return {"success": False, "errors": ["Device not trusted"]}


class ZeroTrustModel:
    """Main zero trust security model implementation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.entities: Dict[str, TrustEntity] = {}
        self.policies: Dict[str, TrustPolicy] = {}
        self.verification_methods: Dict[str, VerificationMethod] = {}
        self.access_attempts: deque = deque(maxlen=100000)
        self.risk_calculator = RiskCalculator()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.metrics: Dict[str, int] = defaultdict(int)
        
        # Initialize default components
        self._initialize_default_policies()
        self._initialize_verification_methods()
        
    def _initialize_default_policies(self) -> None:
        """Initialize default trust policies"""
        # High security policy
        high_security_policy = TrustPolicy(
            policy_id="high_security",
            name="High Security Policy",
            description="Policy for high-security resources",
            entity_types=[EntityType.USER, EntityType.DEVICE],
            minimum_trust_level=TrustLevel.HIGH,
            verification_requirements=["mfa", "certificate"],
            risk_threshold=0.3,
            conditions=[
                {
                    "type": "time_based",
                    "allowed_hours": list(range(8, 18)),  # 8 AM to 6 PM
                    "description": "Business hours only"
                },
                {
                    "type": "location_based",
                    "allowed_locations": ["10.0.0.0/8", "192.168.0.0/16"],
                    "description": "Internal networks only"
                }
            ]
        )
        self.register_policy(high_security_policy)
        
        # Standard policy
        standard_policy = TrustPolicy(
            policy_id="standard",
            name="Standard Security Policy", 
            description="Standard access policy",
            entity_types=[EntityType.USER, EntityType.DEVICE],
            minimum_trust_level=TrustLevel.MEDIUM,
            verification_requirements=["password"],
            risk_threshold=0.7
        )
        self.register_policy(standard_policy)
        
    def _initialize_verification_methods(self) -> None:
        """Initialize verification methods"""
        # Password verification
        password_method = VerificationMethod(
            method_id="password",
            name="Password Verification",
            description="Password-based authentication",
            entity_types=[EntityType.USER],
            trust_level_granted=TrustLevel.LOW,
            expiry_duration=timedelta(hours=8),
            verification_steps=[
                {"type": "password_check"}
            ]
        )
        self.register_verification_method(password_method)
        
        # MFA verification
        mfa_method = VerificationMethod(
            method_id="mfa",
            name="Multi-Factor Authentication",
            description="Password + MFA token verification",
            entity_types=[EntityType.USER],
            trust_level_granted=TrustLevel.HIGH,
            expiry_duration=timedelta(hours=4),
            verification_steps=[
                {"type": "password_check"},
                {"type": "mfa_check"}
            ]
        )
        self.register_verification_method(mfa_method)
        
    def register_entity(self, entity: TrustEntity) -> None:
        """Register entity in zero trust model"""
        self.entities[entity.entity_id] = entity
        logger.info(f"Registered entity {entity.name} ({entity.entity_type.value})")
        
    def register_policy(self, policy: TrustPolicy) -> None:
        """Register trust policy"""
        self.policies[policy.policy_id] = policy
        logger.info(f"Registered trust policy: {policy.name}")
        
    def register_verification_method(self, method: VerificationMethod) -> None:
        """Register verification method"""
        self.verification_methods[method.method_id] = method
        logger.info(f"Registered verification method: {method.name}")
        
    async def evaluate_trust(self, entity_id: str, resource: str, action: str,
                           context: Dict[str, Any], policy_id: str = "standard") -> Dict[str, Any]:
        """Evaluate trust for access request"""
        entity = self.entities.get(entity_id)
        if not entity:
            return {
                "allowed": False,
                "trust_decision": "deny",
                "reason": "Entity not found",
                "entity_id": entity_id
            }
        
        policy = self.policies.get(policy_id)
        if not policy:
            return {
                "allowed": False,
                "trust_decision": "deny",
                "reason": "Policy not found",
                "policy_id": policy_id
            }
        
        # Update risk score based on current context
        await self._update_entity_risk(entity, context)
        
        # Record behavior
        entity.add_behavior(action, context)
        
        # Evaluate policy
        policy_result = policy.evaluate(entity, context)
        
        # Log access attempt
        self._log_access_attempt(entity, resource, action, context, policy_result)
        
        # Update metrics
        self.metrics[f"access_{policy_result['trust_decision']}"] += 1
        
        return {
            **policy_result,
            "entity_id": entity_id,
            "resource": resource,
            "action": action,
            "trust_level": entity.trust_level.name,
            "risk_score": entity.risk_score
        }
        
    async def verify_entity(self, entity_id: str, method_id: str,
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify entity using specified method"""
        entity = self.entities.get(entity_id)
        if not entity:
            return {
                "success": False,
                "error": "Entity not found"
            }
        
        method = self.verification_methods.get(method_id)
        if not method:
            return {
                "success": False,
                "error": "Verification method not found"
            }
        
        # Perform verification
        result = await method.verify(entity, context)
        
        if result["success"]:
            # Update entity trust level and verification status
            entity.trust_level = result["trust_level"]
            entity.verification_status = result["verification_status"]
            entity.last_verified = datetime.now()
            entity.verification_expiry = datetime.now() + method.expiry_duration
            
            logger.info(f"Entity {entity.name} verified with trust level {entity.trust_level.name}")
            self.metrics["verifications_successful"] += 1
        else:
            self.metrics["verifications_failed"] += 1
            
        return result
        
    async def _update_entity_risk(self, entity: TrustEntity, context: Dict[str, Any]) -> None:
        """Update entity risk score based on current context"""
        risk_factors = []
        
        # Check for unusual access patterns
        if await self.behavior_analyzer.is_unusual_behavior(entity, context):
            risk_factors.append(0.3)
        
        # Check for suspicious IP addresses
        client_ip = context.get("client_ip")
        if client_ip and await self._is_suspicious_ip(client_ip):
            risk_factors.append(0.4)
        
        # Check for unusual time access
        if self._is_unusual_time(context):
            risk_factors.append(0.2)
        
        # Calculate new risk score
        if risk_factors:
            new_risk = sum(risk_factors) / len(risk_factors)
            # Exponential moving average for risk score
            alpha = 0.3
            entity.risk_score = alpha * new_risk + (1 - alpha) * entity.risk_score
        else:
            # Decay risk score over time if no risk factors
            entity.risk_score *= 0.95
            
        entity.risk_score = max(0.0, min(1.0, entity.risk_score))
        
    async def _is_suspicious_ip(self, ip_address: str) -> bool:
        """Check if IP address is suspicious"""
        # Simplified implementation - would integrate with threat intelligence
        try:
            ip = ipaddress.ip_address(ip_address)
            # Consider certain ranges suspicious
            suspicious_ranges = [
                "192.0.2.0/24",  # TEST-NET-1
                "198.51.100.0/24",  # TEST-NET-2
                "203.0.113.0/24"   # TEST-NET-3
            ]
            
            for range_str in suspicious_ranges:
                if ip in ipaddress.ip_network(range_str, strict=False):
                    return True
        except ValueError:
            return True  # Invalid IP is suspicious
        
        return False
        
    def _is_unusual_time(self, context: Dict[str, Any]) -> bool:
        """Check if access time is unusual"""
        current_hour = datetime.now().hour
        # Consider access outside business hours unusual
        return current_hour < 8 or current_hour > 18
        
    def _log_access_attempt(self, entity: TrustEntity, resource: str, action: str,
                           context: Dict[str, Any], policy_result: Dict[str, Any]) -> None:
        """Log access attempt"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "entity_id": entity.entity_id,
            "entity_type": entity.entity_type.value,
            "resource": resource,
            "action": action,
            "trust_decision": policy_result["trust_decision"],
            "trust_level": entity.trust_level.name,
            "risk_score": entity.risk_score,
            "context": context
        }
        self.access_attempts.append(log_entry)
        
    def get_entity_trust_level(self, entity_id: str) -> Optional[TrustLevel]:
        """Get current trust level for entity"""
        entity = self.entities.get(entity_id)
        return entity.trust_level if entity else None
        
    def get_zero_trust_metrics(self) -> Dict[str, Any]:
        """Get zero trust model metrics"""
        total_entities = len(self.entities)
        verified_entities = sum(1 for e in self.entities.values() 
                              if e.verification_status == VerificationStatus.VERIFIED)
        
        trust_distribution = defaultdict(int)
        for entity in self.entities.values():
            trust_distribution[entity.trust_level.name] += 1
        
        return {
            "total_entities": total_entities,
            "verified_entities": verified_entities,
            "verification_rate": verified_entities / total_entities if total_entities else 0,
            "trust_distribution": dict(trust_distribution),
            "access_attempts_total": len(self.access_attempts),
            "access_allowed": self.metrics.get("access_allow", 0),
            "access_challenged": self.metrics.get("access_challenge", 0),
            "access_denied": self.metrics.get("access_deny", 0),
            "verifications_successful": self.metrics.get("verifications_successful", 0),
            "verifications_failed": self.metrics.get("verifications_failed", 0)
        }
        
    async def continuous_monitoring(self) -> None:
        """Continuous monitoring of entities and risk assessment"""
        while True:
            try:
                # Re-evaluate risk for all entities
                for entity in self.entities.values():
                    if entity.behaviors:
                        # Analyze recent behaviors
                        recent_behaviors = [
                            b for b in entity.behaviors[-10:]  # Last 10 behaviors
                            if datetime.fromisoformat(b["timestamp"]) > datetime.now() - timedelta(hours=1)
                        ]
                        
                        if recent_behaviors:
                            avg_risk = sum(b["risk_score"] for b in recent_behaviors) / len(recent_behaviors)
                            # Update risk score
                            alpha = 0.1
                            entity.risk_score = alpha * avg_risk + (1 - alpha) * entity.risk_score
                
                # Clean up expired verifications
                await self._cleanup_expired_verifications()
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {str(e)}")
                await asyncio.sleep(60)
                
    async def _cleanup_expired_verifications(self) -> None:
        """Clean up expired entity verifications"""
        now = datetime.now()
        expired_count = 0
        
        for entity in self.entities.values():
            if (entity.verification_expiry and now > entity.verification_expiry and
                entity.verification_status == VerificationStatus.VERIFIED):
                entity.verification_status = VerificationStatus.EXPIRED
                entity.trust_level = TrustLevel.UNTRUSTED
                expired_count += 1
                
        if expired_count > 0:
            logger.info(f"Expired {expired_count} entity verifications")


class RiskCalculator:
    """Calculate risk scores for entities and behaviors"""
    
    def __init__(self):
        self.risk_weights = {
            "failed_attempts": 0.4,
            "unusual_location": 0.3,
            "unusual_time": 0.2,
            "privilege_escalation": 0.8,
            "data_exfiltration": 0.9
        }
    
    def calculate_risk(self, factors: Dict[str, float]) -> float:
        """Calculate weighted risk score"""
        total_risk = 0.0
        total_weight = 0.0
        
        for factor, value in factors.items():
            if factor in self.risk_weights:
                weight = self.risk_weights[factor]
                total_risk += value * weight
                total_weight += weight
        
        return total_risk / total_weight if total_weight > 0 else 0.0


class BehaviorAnalyzer:
    """Analyze entity behavior patterns"""
    
    def __init__(self):
        self.behavior_models: Dict[str, Dict] = {}
    
    async def is_unusual_behavior(self, entity: TrustEntity, context: Dict[str, Any]) -> bool:
        """Check if current behavior is unusual for entity"""
        if len(entity.behaviors) < 10:
            return False  # Not enough data
        
        # Simplified behavior analysis
        recent_behaviors = entity.behaviors[-50:]  # Last 50 behaviors
        
        # Check for unusual patterns
        current_hour = datetime.now().hour
        usual_hours = set()
        for behavior in recent_behaviors:
            behavior_time = datetime.fromisoformat(behavior["timestamp"])
            usual_hours.add(behavior_time.hour)
        
        # If current hour is not in usual hours, it's unusual
        if current_hour not in usual_hours and len(usual_hours) > 5:
            return True
        
        return False