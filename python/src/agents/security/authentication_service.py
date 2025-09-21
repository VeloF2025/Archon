"""
Multi-Factor Authentication Service
Enterprise-grade authentication with multiple authentication methods
"""

import asyncio
import logging
import secrets
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import hmac
import hashlib
import base64
import pyotp
import qrcode
from io import BytesIO
from PIL import Image
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption

logger = logging.getLogger(__name__)


class AuthenticationMethod(Enum):
    """Authentication methods"""
    PASSWORD = "password"
    MFA_TOTP = "mfa_totp"
    MFA_SMS = "mfa_sms"
    MFA_EMAIL = "mfa_email"
    MFA_PUSH = "mfa_push"
    BIOMETRIC = "biometric"
    CERTIFICATE = "certificate"
    SAML = "saml"
    OAUTH = "oauth"
    LDAP = "ldap"
    API_KEY = "api_key"
    HARDWARE_TOKEN = "hardware_token"


class MFAStatus(Enum):
    """MFA enrollment status"""
    NOT_ENROLLED = "not_enrolled"
    ENROLLED = "enrolled"
    REQUIRED = "required"
    PENDING_SETUP = "pending_setup"
    TEMPORARILY_DISABLED = "temporarily_disabled"


@dataclass
class MFAEnrollment:
    """MFA enrollment information"""
    enrollment_id: str
    user_id: str
    method: AuthenticationMethod
    secret_key: str  # TOTP secret or encrypted credentials
    device_name: str
    backup_codes: List[str] = field(default_factory=list)
    last_used: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    is_primary: bool = False
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enrollment_id": self.enrollment_id,
            "user_id": self.user_id,
            "method": self.method.value,
            "device_name": self.device_name,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_primary": self.is_primary,
            "is_active": self.is_active
        }


@dataclass
class AuthenticationSession:
    """Authentication session tracking"""
    session_id: str
    user_id: str
    ip_address: str
    user_agent: str
    authentication_methods_used: List[AuthenticationMethod] = field(default_factory=list)
    mfa_required: bool = False
    mfa_completed: bool = False
    risk_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=1))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at

    def is_complete(self) -> bool:
        if self.mfa_required:
            return self.mfa_completed and len(self.authentication_methods_used) > 0
        return len(self.authentication_methods_used) > 0


@dataclass
class AuthenticationResult:
    """Authentication result"""
    success: bool
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    message: str = ""
    requires_mfa: bool = False
    mfa_methods_available: List[AuthenticationMethod] = field(default_factory=list)
    risk_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AuthenticationService:
    """Enterprise authentication service with MFA support"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.sessions: Dict[str, AuthenticationSession] = {}
        self.mfa_enrollments: Dict[str, List[MFAEnrollment]] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.locked_accounts: Dict[str, datetime] = {}

        # Security settings
        self.max_failed_attempts = self.config.get("max_failed_attempts", 5)
        self.lockout_duration = timedelta(minutes=self.config.get("lockout_minutes", 30))
        self.session_timeout = timedelta(hours=self.config.get("session_timeout_hours", 1))
        self.mfa_required_for_admin = self.config.get("mfa_required_for_admin", True)
        self.risk_threshold = self.config.get("risk_threshold", 0.7)

        # Email/SMS settings
        self.smtp_config = self.config.get("smtp", {})
        self.sms_config = self.config.get("sms", {})

        # Initialize user store (in production, this would be a database)
        self.user_store = self._initialize_user_store()

    def _initialize_user_store(self) -> Dict[str, Dict[str, Any]]:
        """Initialize user store with default users"""
        return {
            "admin": {
                "password_hash": self._hash_password("admin123", b"admin_salt")[0],
                "salt": b"admin_salt",
                "roles": ["admin", "user"],
                "mfa_required": True,
                "is_active": True,
                "created_at": datetime.now()
            },
            "user": {
                "password_hash": self._hash_password("user123", b"user_salt")[0],
                "salt": b"user_salt",
                "roles": ["user"],
                "mfa_required": False,
                "is_active": True,
                "created_at": datetime.now()
            }
        }

    def _hash_password(self, password: str, salt: bytes) -> Tuple[bytes, bytes]:
        """Hash password with salt using PBKDF2"""
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

    def _verify_password(self, password: str, password_hash: bytes, salt: bytes) -> bool:
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

    def _assess_risk(self, user_id: str, ip_address: str, user_agent: str) -> float:
        """Assess authentication risk score"""
        risk_score = 0.0

        # Check for failed attempts
        recent_failures = self.failed_attempts.get(user_id, [])
        recent_failures = [f for f in recent_failures if datetime.now() - f < timedelta(hours=1)]

        if len(recent_failures) > 3:
            risk_score += 0.3

        if len(recent_failures) > 5:
            risk_score += 0.2

        # Check IP address reputation (simplified)
        suspicious_ips = ["192.168.1.1", "10.0.0.1"]  # Example suspicious IPs
        if ip_address in suspicious_ips:
            risk_score += 0.4

        # Check user agent anomalies
        if "bot" in user_agent.lower() or "crawler" in user_agent.lower():
            risk_score += 0.3

        # Check time of day (outside business hours)
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:
            risk_score += 0.1

        return min(risk_score, 1.0)

    def _is_account_locked(self, user_id: str) -> bool:
        """Check if account is locked"""
        if user_id in self.locked_accounts:
            lock_time = self.locked_accounts[user_id]
            if datetime.now() - lock_time < self.lockout_duration:
                return True
            else:
                # Unlock account
                del self.locked_accounts[user_id]
                self.failed_attempts[user_id] = []
        return False

    def _record_failed_attempt(self, user_id: str) -> None:
        """Record failed authentication attempt"""
        now = datetime.now()
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []

        self.failed_attempts[user_id].append(now)

        # Clean old attempts (last hour)
        cutoff = now - timedelta(hours=1)
        self.failed_attempts[user_id] = [
            attempt for attempt in self.failed_attempts[user_id]
            if attempt > cutoff
        ]

        # Lock account if too many failures
        if len(self.failed_attempts[user_id]) >= self.max_failed_attempts:
            self.locked_accounts[user_id] = now
            logger.warning(f"Account {user_id} locked due to multiple failed attempts")

    async def authenticate_password(self, user_id: str, password: str,
                                  ip_address: str, user_agent: str) -> AuthenticationResult:
        """Authenticate user with password"""
        # Check if account is locked
        if self._is_account_locked(user_id):
            return AuthenticationResult(
                success=False,
                message="Account is temporarily locked due to multiple failed attempts"
            )

        # Get user from store
        user = self.user_store.get(user_id)
        if not user or not user.get("is_active", False):
            return AuthenticationResult(
                success=False,
                message="Invalid username or password"
            )

        # Verify password
        if not self._verify_password(password, user["password_hash"], user["salt"]):
            self._record_failed_attempt(user_id)
            return AuthenticationResult(
                success=False,
                message="Invalid username or password"
            )

        # Assess risk
        risk_score = self._assess_risk(user_id, ip_address, user_agent)

        # Create authentication session
        session_id = str(uuid.uuid4())
        session = AuthenticationSession(
            session_id=session_id,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            risk_score=risk_score,
            mfa_required=user.get("mfa_required", False) or risk_score > self.risk_threshold
        )

        self.sessions[session_id] = session

        # Clear failed attempts on successful authentication
        self.failed_attempts.pop(user_id, None)

        # Check if MFA is required
        if session.mfa_required:
            mfa_methods = await self.get_available_mfa_methods(user_id)
            return AuthenticationResult(
                success=True,
                user_id=user_id,
                session_id=session_id,
                message="Password authenticated. MFA required.",
                requires_mfa=True,
                mfa_methods_available=mfa_methods,
                risk_score=risk_score
            )

        return AuthenticationResult(
            success=True,
            user_id=user_id,
            session_id=session_id,
            message="Authentication successful",
            risk_score=risk_score
        )

    async def generate_totp_secret(self, user_id: str, device_name: str) -> Dict[str, Any]:
        """Generate TOTP secret for MFA enrollment"""
        secret = pyotp.random_base32()
        enrollment_id = str(uuid.uuid4())

        # Generate provisioning URI
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(user_id, issuer_name="Archon Security")

        # Generate QR code
        qr_img = qrcode.make(provisioning_uri)
        qr_buffer = BytesIO()
        qr_img.save(qr_buffer, format='PNG')
        qr_base64 = base64.b64encode(qr_buffer.getvalue()).decode()

        # Generate backup codes
        backup_codes = [secrets.token_urlsafe(8) for _ in range(10)]

        # Create enrollment (not yet active)
        enrollment = MFAEnrollment(
            enrollment_id=enrollment_id,
            user_id=user_id,
            method=AuthenticationMethod.MFA_TOTP,
            secret_key=secret,
            backup_codes=backup_codes,
            device_name=device_name,
            is_active=False  # Will be activated after verification
        )

        # Store enrollment
        if user_id not in self.mfa_enrollments:
            self.mfa_enrollments[user_id] = []
        self.mfa_enrollments[user_id].append(enrollment)

        return {
            "enrollment_id": enrollment_id,
            "secret_key": secret,
            "provisioning_uri": provisioning_uri,
            "qr_code": f"data:image/png;base64,{qr_base64}",
            "backup_codes": backup_codes
        }

    async def verify_totp_setup(self, user_id: str, enrollment_id: str, totp_code: str) -> bool:
        """Verify TOTP setup and activate enrollment"""
        if user_id not in self.mfa_enrollments:
            return False

        enrollment = None
        for enr in self.mfa_enrollments[user_id]:
            if enr.enrollment_id == enrollment_id and not enr.is_active:
                enrollment = enr
                break

        if not enrollment:
            return False

        # Verify TOTP code
        totp = pyotp.TOTP(enrollment.secret_key)
        if totp.verify(totp_code, valid_window=1):  # Allow 1 step window
            enrollment.is_active = True
            enrollment.created_at = datetime.now()

            # Set as primary if this is the first MFA method
            active_enrollments = [e for e in self.mfa_enrollments[user_id] if e.is_active]
            if len(active_enrollments) == 1:
                enrollment.is_primary = True

            logger.info(f"TOTP setup verified for user {user_id}")
            return True

        return False

    async def verify_totp(self, user_id: str, totp_code: str, session_id: str) -> bool:
        """Verify TOTP code for authentication"""
        session = self.sessions.get(session_id)
        if not session or session.user_id != user_id:
            return False

        # Get active TOTP enrollments
        enrollments = self.mfa_enrollments.get(user_id, [])
        active_totp_enrollments = [e for e in enrollments if e.is_active and e.method == AuthenticationMethod.MFA_TOTP]

        for enrollment in active_totp_enrollments:
            totp = pyotp.TOTP(enrollment.secret_key)
            if totp.verify(totp_code, valid_window=1):
                # Update enrollment
                enrollment.last_used = datetime.now()

                # Update session
                session.mfa_completed = True
                session.authentication_methods_used.append(AuthenticationMethod.MFA_TOTP)

                logger.info(f"TOTP verification successful for user {user_id}")
                return True

        return False

    async def send_sms_code(self, user_id: str, phone_number: str) -> str:
        """Send SMS verification code"""
        code = str(secrets.randbelow(900000) + 100000)  # 6-digit code

        # In production, integrate with SMS service
        # For demo, just log the code
        logger.info(f"SMS code for {user_id}: {code}")

        # Store code for verification (in production, use secure storage)
        # This is a simplified implementation
        enrollment_id = f"sms_{user_id}_{int(time.time())}"

        enrollment = MFAEnrollment(
            enrollment_id=enrollment_id,
            user_id=user_id,
            method=AuthenticationMethod.MFA_SMS,
            secret_key=code,  # In production, encrypt this
            device_name=f"SMS: {phone_number}",
            expires_at=datetime.now() + timedelta(minutes=5)  # Code expires in 5 minutes
        )

        if user_id not in self.mfa_enrollments:
            self.mfa_enrollments[user_id] = []
        self.mfa_enrollments[user_id].append(enrollment)

        return enrollment_id

    async def verify_sms_code(self, user_id: str, enrollment_id: str, code: str, session_id: str) -> bool:
        """Verify SMS code"""
        session = self.sessions.get(session_id)
        if not session or session.user_id != user_id:
            return False

        enrollment = None
        for enr in self.mfa_enrollments.get(user_id, []):
            if enr.enrollment_id == enrollment_id and enr.method == AuthenticationMethod.MFA_SMS:
                enrollment = enr
                break

        if not enrollment or enrollment.is_active:
            return False

        # Check if code is expired
        if enrollment.expires_at and datetime.now() > enrollment.expires_at:
            return False

        if enrollment.secret_key == code:
            enrollment.is_active = True
            enrollment.last_used = datetime.now()

            session.mfa_completed = True
            session.authentication_methods_used.append(AuthenticationMethod.MFA_SMS)

            logger.info(f"SMS verification successful for user {user_id}")
            return True

        return False

    async def get_available_mfa_methods(self, user_id: str) -> List[AuthenticationMethod]:
        """Get available MFA methods for user"""
        methods = []
        enrollments = self.mfa_enrollments.get(user_id, [])

        for enrollment in enrollments:
            if enrollment.is_active:
                methods.append(enrollment.method)

        # Always allow SMS as backup
        methods.append(AuthenticationMethod.MFA_SMS)

        return methods

    async def get_mfa_enrollments(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's MFA enrollments"""
        enrollments = self.mfa_enrollments.get(user_id, [])
        return [enrollment.to_dict() for enrollment in enrollments if enrollment.is_active]

    async def disable_mfa_method(self, user_id: str, enrollment_id: str) -> bool:
        """Disable MFA method for user"""
        if user_id not in self.mfa_enrollments:
            return False

        for enrollment in self.mfa_enrollments[user_id]:
            if enrollment.enrollment_id == enrollment_id:
                enrollment.is_active = False

                # If this was primary, assign another as primary
                if enrollment.is_primary:
                    active_enrollments = [e for e in self.mfa_enrollments[user_id] if e.is_active and e.enrollment_id != enrollment_id]
                    if active_enrollments:
                        active_enrollments[0].is_primary = True

                logger.info(f"Disabled MFA method {enrollment_id} for user {user_id}")
                return True

        return False

    async def validate_session(self, session_id: str) -> Optional[AuthenticationSession]:
        """Validate authentication session"""
        session = self.sessions.get(session_id)
        if not session:
            return None

        if session.is_expired():
            del self.sessions[session_id]
            return None

        return session

    async def logout(self, session_id: str) -> bool:
        """Logout user and invalidate session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Session {session_id} invalidated")
            return True
        return False

    async def cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions"""
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session.is_expired()
        ]

        for session_id in expired_sessions:
            del self.sessions[session_id]

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    async def get_user_security_status(self, user_id: str) -> Dict[str, Any]:
        """Get user's security status"""
        user = self.user_store.get(user_id)
        if not user:
            return {}

        enrollments = self.mfa_enrollments.get(user_id, [])
        active_enrollments = [e for e in enrollments if e.is_active]

        return {
            "user_id": user_id,
            "is_active": user.get("is_active", False),
            "roles": user.get("roles", []),
            "mfa_required": user.get("mfa_required", False),
            "mfa_enrolled": len(active_enrollments) > 0,
            "mfa_methods": [e.method.value for e in active_enrollments],
            "account_locked": user_id in self.locked_accounts,
            "failed_attempts": len(self.failed_attempts.get(user_id, [])),
            "last_failed_attempt": self.failed_attempts.get(user_id, [-1])[-1] if user_id in self.failed_attempts else None
        }

    async def enforce_password_policy(self, password: str) -> Dict[str, Any]:
        """Enforce password policy requirements"""
        issues = []
        score = 0

        # Length requirement
        if len(password) < 12:
            issues.append("Password must be at least 12 characters long")
        else:
            score += 20

        # Complexity requirements
        if not any(c.isupper() for c in password):
            issues.append("Password must contain at least one uppercase letter")
        else:
            score += 15

        if not any(c.islower() for c in password):
            issues.append("Password must contain at least one lowercase letter")
        else:
            score += 15

        if not any(c.isdigit() for c in password):
            issues.append("Password must contain at least one digit")
        else:
            score += 15

        if not any(not c.isalnum() for c in password):
            issues.append("Password must contain at least one special character")
        else:
            score += 15

        # Check for common passwords
        common_passwords = ["password", "123456", "qwerty", "admin", "letmein"]
        if password.lower() in common_passwords:
            issues.append("Password is too common")
            score -= 20

        # Check for user information (simplified)
        if "user" in password.lower():
            issues.append("Password should not contain user information")
            score -= 10

        return {
            "valid": len(issues) == 0,
            "score": max(0, score),
            "issues": issues,
            "strength": "weak" if score < 50 else "medium" if score < 80 else "strong"
        }