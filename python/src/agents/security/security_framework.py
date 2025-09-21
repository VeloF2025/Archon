"""
Advanced Security Framework for Archon Enhancement 2025
Comprehensive security management with defense-in-depth approach
"""

import asyncio
import logging
import hashlib
import hmac
import secrets
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import ipaddress
import re
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security clearance levels"""
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    SECRET = 3
    TOP_SECRET = 4


class ThreatLevel(Enum):
    """Threat severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EXTREME = 5


class AuthenticationMethod(Enum):
    """Authentication methods"""
    PASSWORD = "password"
    MFA = "mfa"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"
    OAUTH = "oauth"
    SAML = "saml"
    LDAP = "ldap"


class SecurityEvent(Enum):
    """Security event types"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    AUTHORIZATION_DENIED = "authorization_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"
    THREAT_DETECTED = "threat_detected"
    SECURITY_VIOLATION = "security_violation"


@dataclass
class SecurityContext:
    """Security context for requests"""
    user_id: str
    session_id: str
    ip_address: str
    user_agent: str
    permissions: Set[str] = field(default_factory=set)
    roles: Set[str] = field(default_factory=set)
    security_level: SecurityLevel = SecurityLevel.PUBLIC
    authentication_method: AuthenticationMethod = AuthenticationMethod.PASSWORD
    authenticated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if security context is expired"""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False
    
    def has_permission(self, permission: str) -> bool:
        """Check if context has specific permission"""
        return permission in self.permissions
    
    def has_role(self, role: str) -> bool:
        """Check if context has specific role"""
        return role in self.roles
    
    def meets_security_level(self, required_level: SecurityLevel) -> bool:
        """Check if security level meets requirement"""
        return self.security_level.value >= required_level.value


@dataclass
class SecurityPolicy:
    """Security policy definition"""
    policy_id: str
    name: str
    description: str
    rules: List[Dict[str, Any]]
    required_security_level: SecurityLevel = SecurityLevel.INTERNAL
    allowed_ip_ranges: List[str] = field(default_factory=list)
    blocked_ip_ranges: List[str] = field(default_factory=list)
    required_authentication: List[AuthenticationMethod] = field(default_factory=list)
    session_timeout: timedelta = timedelta(hours=8)
    max_failed_attempts: int = 5
    lockout_duration: timedelta = timedelta(minutes=30)
    enabled: bool = True
    
    def evaluate(self, context: SecurityContext, resource: str, action: str) -> bool:
        """Evaluate policy against security context"""
        if not self.enabled:
            return True
            
        # Check security level
        if not context.meets_security_level(self.required_security_level):
            return False
            
        # Check IP restrictions
        if not self._check_ip_restrictions(context.ip_address):
            return False
            
        # Check authentication requirements
        if not self._check_authentication_requirements(context):
            return False
            
        # Evaluate rules
        return self._evaluate_rules(context, resource, action)
    
    def _check_ip_restrictions(self, ip_address: str) -> bool:
        """Check IP address restrictions"""
        ip = ipaddress.ip_address(ip_address)
        
        # Check blocked ranges
        for blocked_range in self.blocked_ip_ranges:
            if ip in ipaddress.ip_network(blocked_range, strict=False):
                return False
                
        # Check allowed ranges (if specified)
        if self.allowed_ip_ranges:
            allowed = False
            for allowed_range in self.allowed_ip_ranges:
                if ip in ipaddress.ip_network(allowed_range, strict=False):
                    allowed = True
                    break
            if not allowed:
                return False
                
        return True
    
    def _check_authentication_requirements(self, context: SecurityContext) -> bool:
        """Check authentication method requirements"""
        if not self.required_authentication:
            return True
            
        return context.authentication_method in self.required_authentication
    
    def _evaluate_rules(self, context: SecurityContext, resource: str, action: str) -> bool:
        """Evaluate policy rules"""
        for rule in self.rules:
            if not self._evaluate_rule(rule, context, resource, action):
                return False
        return True
    
    def _evaluate_rule(self, rule: Dict[str, Any], context: SecurityContext,
                      resource: str, action: str) -> bool:
        """Evaluate a single rule"""
        rule_type = rule.get("type")
        
        if rule_type == "permission":
            required_permission = rule.get("permission")
            return context.has_permission(required_permission)
            
        elif rule_type == "role":
            required_role = rule.get("role")
            return context.has_role(required_role)
            
        elif rule_type == "resource_pattern":
            pattern = rule.get("pattern")
            return re.match(pattern, resource) is not None
            
        elif rule_type == "time_restriction":
            start_time = rule.get("start_time")
            end_time = rule.get("end_time")
            current_time = datetime.now().time()
            return start_time <= current_time <= end_time
            
        elif rule_type == "rate_limit":
            # Rate limiting would be handled by external rate limiter
            return True
            
        return True


@dataclass
class SecurityAlert:
    """Security alert/incident"""
    alert_id: str
    event_type: SecurityEvent
    threat_level: ThreatLevel
    user_id: Optional[str]
    ip_address: Optional[str]
    resource: Optional[str]
    description: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    investigated: bool = False
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "event_type": self.event_type.value,
            "threat_level": self.threat_level.value,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "resource": self.resource,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "investigated": self.investigated,
            "resolved": self.resolved
        }


class SecurityFramework:
    """Main security framework orchestrating all security components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.policies: Dict[str, SecurityPolicy] = {}
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.locked_accounts: Dict[str, datetime] = {}
        self.security_alerts: deque = deque(maxlen=10000)
        self.event_handlers: Dict[SecurityEvent, List[Callable]] = defaultdict(list)
        self.encryption_keys: Dict[str, bytes] = {}
        self.access_logs: deque = deque(maxlen=50000)
        
        # Initialize default security keys
        self._initialize_encryption()
        
        # Load default policies
        self._load_default_policies()
        
    def _initialize_encryption(self) -> None:
        """Initialize encryption keys"""
        # Generate master key if not provided
        master_key = self.config.get("master_key")
        if not master_key:
            master_key = Fernet.generate_key()
            
        self.encryption_keys["master"] = master_key
        self.fernet = Fernet(master_key)
        
        # Generate RSA key pair for asymmetric encryption
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.private_key = private_key
        self.public_key = private_key.public_key()
        
    def _load_default_policies(self) -> None:
        """Load default security policies"""
        # Default admin policy
        admin_policy = SecurityPolicy(
            policy_id="admin_policy",
            name="Administrator Policy",
            description="Full access policy for administrators",
            rules=[
                {"type": "role", "role": "admin"}
            ],
            required_security_level=SecurityLevel.SECRET,
            required_authentication=[AuthenticationMethod.MFA]
        )
        self.register_policy(admin_policy)
        
        # Default user policy
        user_policy = SecurityPolicy(
            policy_id="user_policy",
            name="Standard User Policy", 
            description="Standard access policy for regular users",
            rules=[
                {"type": "role", "role": "user"}
            ],
            required_security_level=SecurityLevel.INTERNAL
        )
        self.register_policy(user_policy)
        
    def register_policy(self, policy: SecurityPolicy) -> None:
        """Register a security policy"""
        self.policies[policy.policy_id] = policy
        logger.info(f"Registered security policy: {policy.name}")
        
    def authenticate_user(self, username: str, credentials: Dict[str, Any],
                         ip_address: str, user_agent: str) -> Optional[SecurityContext]:
        """Authenticate a user and create security context"""
        # Check if account is locked
        if self.is_account_locked(username):
            self._log_security_event(
                SecurityEvent.LOGIN_FAILURE,
                user_id=username,
                ip_address=ip_address,
                description="Login attempt on locked account"
            )
            return None
            
        # Verify credentials (simplified - would integrate with actual auth provider)
        if not self._verify_credentials(username, credentials):
            self._record_failed_attempt(username)
            self._log_security_event(
                SecurityEvent.LOGIN_FAILURE,
                user_id=username,
                ip_address=ip_address,
                description="Invalid credentials"
            )
            return None
            
        # Create security context
        context = SecurityContext(
            user_id=username,
            session_id=str(uuid.uuid4()),
            ip_address=ip_address,
            user_agent=user_agent,
            permissions=self._get_user_permissions(username),
            roles=self._get_user_roles(username),
            security_level=self._get_user_security_level(username),
            authentication_method=self._determine_auth_method(credentials),
            expires_at=datetime.now() + timedelta(hours=8)
        )
        
        # Store session
        self.active_sessions[context.session_id] = context
        
        self._log_security_event(
            SecurityEvent.LOGIN_SUCCESS,
            user_id=username,
            ip_address=ip_address,
            description="Successful authentication"
        )
        
        return context
        
    def authorize_access(self, context: SecurityContext, resource: str,
                        action: str, policy_id: str = "user_policy") -> bool:
        """Authorize access to a resource"""
        # Check if session is expired
        if context.is_expired():
            self._log_security_event(
                SecurityEvent.AUTHORIZATION_DENIED,
                user_id=context.user_id,
                ip_address=context.ip_address,
                description="Session expired"
            )
            return False
            
        # Get applicable policy
        policy = self.policies.get(policy_id)
        if not policy:
            logger.warning(f"Policy {policy_id} not found")
            return False
            
        # Evaluate policy
        authorized = policy.evaluate(context, resource, action)
        
        # Log access attempt
        self._log_access_attempt(context, resource, action, authorized)
        
        if not authorized:
            self._log_security_event(
                SecurityEvent.AUTHORIZATION_DENIED,
                user_id=context.user_id,
                ip_address=context.ip_address,
                description=f"Access denied to {resource} for action {action}"
            )
            
        return authorized
        
    def is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to failed attempts"""
        if username in self.locked_accounts:
            lock_time = self.locked_accounts[username]
            if datetime.now() - lock_time < timedelta(minutes=30):
                return True
            else:
                # Unlock account
                del self.locked_accounts[username]
                
        return False
        
    def _record_failed_attempt(self, username: str) -> None:
        """Record failed authentication attempt"""
        now = datetime.now()
        self.failed_attempts[username].append(now)
        
        # Clean old attempts (last hour)
        cutoff = now - timedelta(hours=1)
        self.failed_attempts[username] = [
            attempt for attempt in self.failed_attempts[username]
            if attempt > cutoff
        ]
        
        # Lock account if too many failures
        if len(self.failed_attempts[username]) >= 5:
            self.locked_accounts[username] = now
            logger.warning(f"Account {username} locked due to multiple failed attempts")
            
    def _verify_credentials(self, username: str, credentials: Dict[str, Any]) -> bool:
        """Verify user credentials (simplified implementation)"""
        # In real implementation, this would integrate with auth providers
        password = credentials.get("password")
        if not password:
            return False
            
        # Simple validation for demo
        return len(password) >= 8
        
    def _get_user_permissions(self, username: str) -> Set[str]:
        """Get user permissions from user store"""
        # Simplified - would query actual user database
        if username == "admin":
            return {"read", "write", "delete", "admin"}
        return {"read", "write"}
        
    def _get_user_roles(self, username: str) -> Set[str]:
        """Get user roles from user store"""
        # Simplified - would query actual user database
        if username == "admin":
            return {"admin", "user"}
        return {"user"}
        
    def _get_user_security_level(self, username: str) -> SecurityLevel:
        """Get user security clearance level"""
        # Simplified - would query actual user database
        if username == "admin":
            return SecurityLevel.SECRET
        return SecurityLevel.INTERNAL
        
    def _determine_auth_method(self, credentials: Dict[str, Any]) -> AuthenticationMethod:
        """Determine authentication method used"""
        if "mfa_token" in credentials:
            return AuthenticationMethod.MFA
        elif "certificate" in credentials:
            return AuthenticationMethod.CERTIFICATE
        else:
            return AuthenticationMethod.PASSWORD
            
    def _log_security_event(self, event_type: SecurityEvent, user_id: str = None,
                           ip_address: str = None, description: str = "",
                           details: Dict[str, Any] = None) -> None:
        """Log security event"""
        alert = SecurityAlert(
            alert_id=str(uuid.uuid4()),
            event_type=event_type,
            threat_level=self._assess_threat_level(event_type),
            user_id=user_id,
            ip_address=ip_address,
            description=description,
            timestamp=datetime.now(),
            details=details or {}
        )
        
        self.security_alerts.append(alert)
        
        # Trigger event handlers
        for handler in self.event_handlers[event_type]:
            try:
                asyncio.create_task(handler(alert))
            except Exception as e:
                logger.error(f"Error in security event handler: {str(e)}")
                
    def _assess_threat_level(self, event_type: SecurityEvent) -> ThreatLevel:
        """Assess threat level for security event"""
        threat_mapping = {
            SecurityEvent.LOGIN_SUCCESS: ThreatLevel.LOW,
            SecurityEvent.LOGIN_FAILURE: ThreatLevel.LOW,
            SecurityEvent.AUTHORIZATION_DENIED: ThreatLevel.MEDIUM,
            SecurityEvent.SUSPICIOUS_ACTIVITY: ThreatLevel.HIGH,
            SecurityEvent.THREAT_DETECTED: ThreatLevel.CRITICAL,
            SecurityEvent.SECURITY_VIOLATION: ThreatLevel.EXTREME
        }
        return threat_mapping.get(event_type, ThreatLevel.MEDIUM)
        
    def _log_access_attempt(self, context: SecurityContext, resource: str,
                           action: str, authorized: bool) -> None:
        """Log access attempt"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": context.user_id,
            "session_id": context.session_id,
            "ip_address": context.ip_address,
            "resource": resource,
            "action": action,
            "authorized": authorized
        }
        self.access_logs.append(log_entry)
        
    def encrypt_data(self, data: Union[str, bytes], key_id: str = "master") -> bytes:
        """Encrypt sensitive data"""
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        if key_id not in self.encryption_keys:
            raise ValueError(f"Encryption key {key_id} not found")
            
        return self.fernet.encrypt(data)
        
    def decrypt_data(self, encrypted_data: bytes, key_id: str = "master") -> bytes:
        """Decrypt sensitive data"""
        if key_id not in self.encryption_keys:
            raise ValueError(f"Encryption key {key_id} not found")
            
        return self.fernet.decrypt(encrypted_data)
        
    def generate_secure_token(self, payload: Dict[str, Any],
                             expires_in: timedelta = timedelta(hours=1)) -> str:
        """Generate secure JWT token"""
        payload["exp"] = datetime.utcnow() + expires_in
        payload["iat"] = datetime.utcnow()
        payload["jti"] = str(uuid.uuid4())
        
        return jwt.encode(payload, self.encryption_keys["master"], algorithm="HS256")
        
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.encryption_keys["master"], algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
        return None
        
    def register_event_handler(self, event_type: SecurityEvent,
                              handler: Callable) -> None:
        """Register handler for security events"""
        self.event_handlers[event_type].append(handler)
        
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics and statistics"""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        recent_alerts = [
            alert for alert in self.security_alerts
            if alert.timestamp > last_24h
        ]
        
        return {
            "active_sessions": len(self.active_sessions),
            "locked_accounts": len(self.locked_accounts),
            "failed_attempts_24h": sum(
                len([attempt for attempt in attempts if attempt > last_24h])
                for attempts in self.failed_attempts.values()
            ),
            "security_alerts_24h": len(recent_alerts),
            "high_threat_alerts_24h": len([
                alert for alert in recent_alerts
                if alert.threat_level.value >= ThreatLevel.HIGH.value
            ]),
            "policies_active": len([p for p in self.policies.values() if p.enabled])
        }
        
    def get_security_alerts(self, limit: int = 100,
                           threat_level: ThreatLevel = None) -> List[Dict[str, Any]]:
        """Get recent security alerts"""
        alerts = list(self.security_alerts)
        
        if threat_level:
            alerts = [alert for alert in alerts if alert.threat_level == threat_level]
            
        alerts = alerts[-limit:]
        return [alert.to_dict() for alert in alerts]
        
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a user session"""
        if session_id in self.active_sessions:
            context = self.active_sessions[session_id]
            del self.active_sessions[session_id]
            
            self._log_security_event(
                SecurityEvent.LOGIN_FAILURE,  # Using as logout event
                user_id=context.user_id,
                ip_address=context.ip_address,
                description="Session invalidated"
            )
            return True
        return False
        
    async def cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions"""
        now = datetime.now()
        expired_sessions = [
            session_id for session_id, context in self.active_sessions.items()
            if context.is_expired()
        ]
        
        for session_id in expired_sessions:
            self.invalidate_session(session_id)
            
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            
    def hash_password(self, password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_bytes(32)
            
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        password_hash = kdf.derive(password.encode('utf-8'))
        return password_hash, salt
        
    def verify_password(self, password: str, password_hash: bytes, salt: bytes) -> bool:
        """Verify password against hash"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        try:
            kdf.verify(password.encode('utf-8'), password_hash)
            return True
        except Exception:
            return False