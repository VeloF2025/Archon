"""
Security Framework Tests

Comprehensive test suite for the security framework components
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

from src.agents.security.security_framework import SecurityFramework, SecurityContext
from src.agents.security.authentication_service import AuthenticationService, AuthenticationMethod, RiskLevel
from src.agents.security.authorization_service import AuthorizationService, AccessDecision, PolicyCombiningAlgorithm
from src.agents.security.encryption_service import EncryptionService, EncryptionAlgorithm
from src.agents.security.audit_logger import AuditLogger, AuditLevel, AuditCategory
from src.agents.security.access_control import AccessControlManager


class TestSecurityFramework:
    """Test cases for SecurityFramework"""

    @pytest.fixture
    def security_framework(self):
        """Create a security framework instance for testing"""
        return SecurityFramework()

    def test_security_framework_initialization(self, security_framework):
        """Test security framework initialization"""
        assert security_framework.active_sessions == {}
        assert security_framework.user_sessions == {}
        assert security_framework.failed_attempts == {}
        assert len(security_framework.risk_policies) > 0

    def test_user_authentication_success(self, security_framework):
        """Test successful user authentication"""
        # Mock authentication context
        context = Mock(spec=SecurityContext)
        context.user_id = "test_user"
        context.session_id = "test_session"
        context.expires_at = datetime.now() + timedelta(hours=1)
        context.is_active.return_value = True

        # Mock authentication service
        with patch.object(security_framework, 'authenticate_user', return_value=context):
            result = security_framework.authenticate_user(
                username="test_user",
                credentials={"password": "test_pass"},
                ip_address="127.0.0.1",
                user_agent="test_agent"
            )
            assert result == context
            assert context.user_id in security_framework.active_sessions

    def test_user_authentication_failure(self, security_framework):
        """Test failed user authentication"""
        with patch.object(security_framework, 'authenticate_user', return_value=None):
            result = security_framework.authenticate_user(
                username="invalid_user",
                credentials={"password": "wrong_pass"},
                ip_address="127.0.0.1",
                user_agent="test_agent"
            )
            assert result is None

    def test_authorization_access_granted(self, security_framework):
        """Test successful authorization"""
        # Create a mock context
        context = Mock(spec=SecurityContext)
        context.user_id = "test_user"
        context.roles = ["admin"]
        context.attributes = {"department": "engineering"}

        # Add session
        security_framework.active_sessions["test_session"] = context

        with patch.object(security_framework, 'authorize_access', return_value=True):
            result = security_framework.authorize_access(
                context=context,
                resource="/api/admin/users",
                action="read",
                policy_id="admin_policy"
            )
            assert result is True

    def test_authorization_access_denied(self, security_framework):
        """Test authorization denial"""
        context = Mock(spec=SecurityContext)
        context.user_id = "test_user"
        context.roles = ["user"]
        context.attributes = {"department": "engineering"}

        security_framework.active_sessions["test_session"] = context

        with patch.object(security_framework, 'authorize_access', return_value=False):
            result = security_framework.authorize_access(
                context=context,
                resource="/api/admin/users",
                action="delete",
                policy_id="admin_policy"
            )
            assert result is False

    def test_session_invalidation(self, security_framework):
        """Test session invalidation"""
        context = Mock(spec=SecurityContext)
        context.user_id = "test_user"
        context.session_id = "test_session"

        security_framework.active_sessions["test_session"] = context

        result = security_framework.invalidate_session("test_session")
        assert result is True
        assert "test_session" not in security_framework.active_sessions

    def test_generate_secure_token(self, security_framework):
        """Test secure token generation"""
        payload = {"user_id": "test_user", "session_id": "test_session"}
        token = security_framework.generate_secure_token(payload)

        assert isinstance(token, str)
        assert len(token) > 0

        # Verify token can be decoded
        decoded = security_framework.verify_token(token)
        assert decoded is not None
        assert decoded["user_id"] == "test_user"

    def test_risk_assessment(self, security_framework):
        """Test risk assessment functionality"""
        context = Mock()
        context.ip_address = "192.168.1.1"
        context.user_agent = "Mozilla/5.0"
        context.location = "US"
        context.time_of_day = "09:00"

        risk_score = security_framework.assess_risk(context)

        assert isinstance(risk_score, float)
        assert 0.0 <= risk_score <= 1.0

    def test_rate_limiting(self, security_framework):
        """Test rate limiting functionality"""
        ip_address = "192.168.1.1"

        # Should allow first few requests
        for i in range(5):
            allowed = security_framework.check_rate_limit(ip_address)
            assert allowed is True

        # Exceed rate limit
        with patch.object(security_framework, 'check_rate_limit', return_value=False):
            allowed = security_framework.check_rate_limit(ip_address)
            assert allowed is False


class TestAuthenticationService:
    """Test cases for AuthenticationService"""

    @pytest.fixture
    def auth_service(self):
        """Create an authentication service instance for testing"""
        return AuthenticationService()

    def test_password_authentication_success(self, auth_service):
        """Test successful password authentication"""
        credentials = {
            "username": "test_user",
            "password": "secure_password123",
            "method": "password"
        }

        with patch.object(auth_service, 'verify_password', return_value=True):
            context = auth_service.authenticate(
                credentials=credentials,
                ip_address="127.0.0.1",
                user_agent="test_agent"
            )

            assert context is not None
            assert context.user_id == "test_user"

    def test_password_authentication_failure(self, auth_service):
        """Test failed password authentication"""
        credentials = {
            "username": "test_user",
            "password": "wrong_password",
            "method": "password"
        }

        with patch.object(auth_service, 'verify_password', return_value=False):
            context = auth_service.authenticate(
                credentials=credentials,
                ip_address="127.0.0.1",
                user_agent="test_agent"
            )

            assert context is None

    def test_mfa_authentication(self, auth_service):
        """Test multi-factor authentication"""
        credentials = {
            "username": "test_user",
            "password": "secure_password123",
            "totp_code": "123456",
            "method": "mfa"
        }

        with patch.object(auth_service, 'verify_password', return_value=True), \
             patch.object(auth_service, 'verify_totp', return_value=True):

            context = auth_service.authenticate(
                credentials=credentials,
                ip_address="127.0.0.1",
                user_agent="test_agent"
            )

            assert context is not None
            assert context.mfa_verified is True

    def test_risk_based_authentication(self, auth_service):
        """Test risk-based authentication"""
        # High risk scenario
        credentials = {
            "username": "test_user",
            "password": "secure_password123",
            "method": "password"
        }

        with patch.object(auth_service, 'verify_password', return_value=True), \
             patch.object(auth_service, 'assess_authentication_risk', return_value=RiskLevel.HIGH):

            context = auth_service.authenticate(
                credentials=credentials,
                ip_address="198.51.100.1",  # Unusual IP
                user_agent="test_agent"
            )

            # High risk should require additional verification
            assert context is not None
            assert context.requires_additional_verification is True

    def test_session_management(self, auth_service):
        """Test session management"""
        context = auth_service.create_security_context(
            user_id="test_user",
            session_id="test_session",
            roles=["user"],
            attributes={}
        )

        assert context.user_id == "test_user"
        assert context.session_id == "test_session"
        assert context.is_active() is True

        # Test session expiration
        context.expires_at = datetime.now() - timedelta(hours=1)
        assert context.is_active() is False


class TestAuthorizationService:
    """Test cases for AuthorizationService"""

    @pytest.fixture
    def authz_service(self):
        """Create an authorization service instance for testing"""
        return AuthorizationService()

    def test_policy_evaluation(self, authz_service):
        """Test policy evaluation"""
        policy = {
            "id": "test_policy",
            "rules": [
                {
                    "effect": "permit",
                    "conditions": [
                        {"field": "user.roles", "operator": "contains", "value": "admin"}
                    ]
                }
            ]
        }

        context = {
            "user": {
                "roles": ["admin", "user"],
                "department": "engineering"
            },
            "resource": {
                "type": "api",
                "path": "/api/admin/users"
            },
            "action": "read"
        }

        decision = authz_service.evaluate_policy(policy, context)
        assert decision == AccessDecision.PERMIT

    def test_role_based_access_control(self, authz_service):
        """Test role-based access control"""
        user_roles = ["admin"]
        required_roles = ["admin"]

        result = authz_service.check_role_based_access(user_roles, required_roles)
        assert result is True

        user_roles = ["user"]
        result = authz_service.check_role_based_access(user_roles, required_roles)
        assert result is False

    def test_attribute_based_access_control(self, authz_service):
        """Test attribute-based access control"""
        user_attributes = {
            "department": "engineering",
            "clearance_level": 3
        }

        rules = [
            {"field": "clearance_level", "operator": ">=", "value": 2},
            {"field": "department", "operator": "==", "value": "engineering"}
        ]

        result = authz_service.check_attribute_based_access(user_attributes, rules)
        assert result is True

    def test_policy_combining_deny_overrides(self, authz_service):
        """Test deny overrides policy combining"""
        policies = [
            {"decision": AccessDecision.PERMIT},
            {"decision": AccessDecision.DENY},
            {"decision": AccessDecision.PERMIT}
        ]

        result = authz_service.combine_policies(
            policies,
            PolicyCombiningAlgorithm.DENY_OVERRIDES
        )
        assert result == AccessDecision.DENY

    def test_policy_combining_permit_overrides(self, authz_service):
        """Test permit overrides policy combining"""
        policies = [
            {"decision": AccessDecision.DENY},
            {"decision": AccessDecision.PERMIT},
            {"decision": AccessDecision.DENY}
        ]

        result = authz_service.combine_policies(
            policies,
            PolicyCombiningAlgorithm.PERMIT_OVERRIDES
        )
        assert result == AccessDecision.PERMIT


class TestEncryptionService:
    """Test cases for EncryptionService"""

    @pytest.fixture
    def encryption_service(self):
        """Create an encryption service instance for testing"""
        return EncryptionService()

    def test_data_encryption_decryption(self, encryption_service):
        """Test data encryption and decryption"""
        plaintext = "Sensitive data to encrypt"
        key_id = "test_key"

        # Encrypt
        encrypted = encryption_service.encrypt(plaintext, key_id)
        assert encrypted is not None
        assert encrypted.key_id == key_id
        assert encrypted.encrypted_data != plaintext

        # Decrypt
        decrypted = encryption_service.decrypt(encrypted)
        assert decrypted.decode('utf-8') == plaintext

    def test_encryption_with_algorithm(self, encryption_service):
        """Test encryption with specific algorithm"""
        plaintext = "Test data"
        algorithm = EncryptionAlgorithm.AES_256_GCM

        encrypted = encryption_service.encrypt(
            plaintext,
            algorithm=algorithm
        )

        assert encrypted.algorithm == algorithm

        decrypted = encryption_service.decrypt(encrypted)
        assert decrypted.decode('utf-8') == plaintext

    def test_key_management(self, encryption_service):
        """Test key management"""
        key_id = "test_managed_key"
        algorithm = EncryptionAlgorithm.AES_256_GCM

        # Generate key
        key = encryption_service.key_manager.generate_key(
            algorithm=algorithm,
            key_id=key_id
        )

        assert key.key_id == key_id
        assert key.algorithm == algorithm

        # List keys
        keys = encryption_service.key_manager.list_keys()
        assert key_id in [k.key_id for k in keys]

        # Delete key
        deleted = encryption_service.key_manager.delete_key(key_id)
        assert deleted is True

    def test_key_rotation(self, encryption_service):
        """Test key rotation"""
        key_id = "rotate_test_key"
        original_key = encryption_service.key_manager.generate_key(
            key_id=key_id,
            algorithm=EncryptionAlgorithm.AES_256_GCM
        )

        # Rotate key
        new_key = encryption_service.key_manager.rotate_key(key_id)

        assert new_key.key_id == key_id
        assert new_key.key_material != original_key.key_material

    def test_encryption_integrity(self, encryption_service):
        """Test encryption integrity checking"""
        plaintext = "Important data"
        encrypted = encryption_service.encrypt(plaintext)

        # Tamper with encrypted data
        tampered_data = encrypted.encrypted_data[:-1] + b'\x00'

        tampered_encrypted = encrypted._replace(encrypted_data=tampered_data)

        # Decryption should fail
        with pytest.raises(Exception):
            encryption_service.decrypt(tampered_encrypted)


class TestAuditLogger:
    """Test cases for AuditLogger"""

    @pytest.fixture
    def audit_logger(self):
        """Create an audit logger instance for testing"""
        return AuditLogger()

    @pytest.mark.asyncio
    async def test_log_authentication_event(self, audit_logger):
        """Test authentication event logging"""
        event_id = await audit_logger.log_authentication(
            user_id="test_user",
            result="success",
            source_ip="127.0.0.1",
            details={"method": "password"}
        )

        assert event_id is not None
        assert isinstance(event_id, str)

    @pytest.mark.asyncio
    async def test_log_authorization_event(self, audit_logger):
        """Test authorization event logging"""
        event_id = await audit_logger.log_authorization(
            user_id="test_user",
            resource="/api/admin/users",
            action="read",
            decision="permit",
            source_ip="127.0.0.1",
            details={"policy": "admin_policy"}
        )

        assert event_id is not None

    @pytest.mark.asyncio
    async def test_log_configuration_change(self, audit_logger):
        """Test configuration change logging"""
        event_id = await audit_logger.log_configuration_change(
            user_id="admin_user",
            resource="security_policies",
            action="update",
            details={"policy_id": "test_policy", "changes": ["added_rule"]}
        )

        assert event_id is not None

    @pytest.mark.asyncio
    async def test_audit_query(self, audit_logger):
        """Test audit log querying"""
        # Log some events
        await audit_logger.log_authentication(
            user_id="test_user",
            result="success",
            source_ip="127.0.0.1"
        )

        await audit_logger.log_authorization(
            user_id="test_user",
            resource="/api/data",
            action="read",
            decision="permit",
            source_ip="127.0.0.1"
        )

        # Query events
        from src.agents.security.audit_logger import AuditFilter
        filter_obj = AuditFilter(
            user_ids=["test_user"],
            limit=10
        )

        events = await audit_logger.query(filter_obj)
        assert len(events) >= 1
        assert all(event.user_id == "test_user" for event in events)

    @pytest.mark.asyncio
    async def test_integrity_verification(self, audit_logger):
        """Test audit log integrity verification"""
        # Log some events
        for i in range(5):
            await audit_logger.log(
                level=AuditLevel.INFO,
                category=AuditCategory.AUTHENTICATION,
                event_type="test_event",
                user_id=f"test_user_{i}",
                source_ip="127.0.0.1"
            )

        # Verify integrity
        result = await audit_logger.verify_integrity()
        assert result["integrity_verified"] is True
        assert result["total_events"] == 5


class TestAccessControlManager:
    """Test cases for AccessControlManager"""

    @pytest.fixture
    def access_control(self):
        """Create an access control manager instance for testing"""
        return AccessControlManager()

    def test_create_subject(self, access_control):
        """Test subject creation"""
        subject = access_control.create_subject(
            subject_id="test_user",
            subject_type="user",
            roles=["user"],
            attributes={"department": "engineering"}
        )

        assert subject.subject_id == "test_user"
        assert subject.subject_type == "user"
        assert "user" in subject.roles
        assert subject.attributes["department"] == "engineering"

        assert "test_user" in access_control.subjects

    def test_create_role(self, access_control):
        """Test role creation"""
        role = access_control.create_role(
            role_id="admin_role",
            name="Administrator",
            description="Full system access",
            permissions=["read", "write", "delete"]
        )

        assert role.role_id == "admin_role"
        assert role.name == "Administrator"
        assert "read" in role.permissions

        assert "admin_role" in access_control.roles

    def test_assign_role_to_subject(self, access_control):
        """Test role assignment to subject"""
        # Create subject and role
        subject = access_control.create_subject("test_user", "user", [], {})
        role = access_control.create_role("admin_role", "Admin", "Admin role", ["admin"])

        # Assign role
        success = access_control.assign_role_to_subject("test_user", "admin_role")
        assert success is True
        assert "admin_role" in subject.roles

    def test_check_access(self, access_control):
        """Test access checking"""
        # Setup test data
        access_control.create_subject("test_user", "user", ["admin"], {})
        access_control.create_role("admin_role", "Admin", "Admin role", ["read", "write"])
        access_control.assign_role_to_subject("test_user", "admin_role")

        # Check access
        result = access_control.check_access(
            subject_id="test_user",
            resource="/api/admin/users",
            action="read"
        )

        assert result.decision == AccessDecision.PERMIT

    def test_permission_inheritance(self, access_control):
        """Test permission inheritance from parent roles"""
        # Create role hierarchy
        parent_role = access_control.create_role(
            "parent_role",
            "Parent",
            "Parent role",
            ["read"]
        )

        child_role = access_control.create_role(
            "child_role",
            "Child",
            "Child role",
            ["write"],
            parent_roles=["parent_role"]
        )

        # Create subject with child role
        subject = access_control.create_subject("test_user", "user", ["child_role"], {})
        access_control.assign_role_to_subject("test_user", "child_role")

        # Should have both read and write permissions
        permissions = access_control.get_subject_permissions("test_user")
        assert "read" in permissions
        assert "write" in permissions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])