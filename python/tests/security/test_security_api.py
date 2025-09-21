"""
Security API Tests

Integration tests for security and compliance API endpoints
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json
from fastapi.testclient import TestClient
from fastapi import FastAPI

from src.server.api_routes.security_api import router as security_router
from src.server.api_routes.compliance_api import router as compliance_router


# Create test FastAPI app
app = FastAPI()
app.include_router(security_router)
app.include_router(compliance_router)

client = TestClient(app)


class TestSecurityAPI:
    """Integration tests for Security API endpoints"""

    def test_security_health_check(self):
        """Test security system health check"""
        response = client.get("/api/security/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "components" in data
        assert "metrics" in data

    def test_authentication_endpoint(self):
        """Test authentication endpoint"""
        auth_request = {
            "username": "test_user",
            "credentials": {
                "password": "test_password",
                "method": "password"
            },
            "ip_address": "127.0.0.1",
            "user_agent": "test_agent"
        }

        with patch('src.server.api_routes.security_api.security_framework.authenticate_user') as mock_auth:
            mock_context = Mock()
            mock_context.user_id = "test_user"
            mock_context.session_id = "test_session"
            mock_context.expires_at = datetime.now() + timedelta(hours=1)
            mock_auth.return_value = mock_context

            response = client.post("/api/security/authenticate", json=auth_request)
            assert response.status_code == 200

            data = response.json()
            assert data["success"] is True
            assert "session_id" in data
            assert "token" in data

    def test_authentication_failure(self):
        """Test authentication failure handling"""
        auth_request = {
            "username": "invalid_user",
            "credentials": {
                "password": "wrong_password",
                "method": "password"
            },
            "ip_address": "127.0.0.1",
            "user_agent": "test_agent"
        }

        with patch('src.server.api_routes.security_api.security_framework.authenticate_user', return_value=None):
            response = client.post("/api/security/authenticate", json=auth_request)
            assert response.status_code == 200

            data = response.json()
            assert data["success"] is False
            assert data["message"] == "Authentication failed"

    def test_authorization_endpoint(self):
        """Test authorization endpoint"""
        # First authenticate to get token
        with patch('src.server.api_routes.security_api.security_framework.verify_token', return_value={"user_id": "test_user"}), \
             patch('src.server.api_routes.security_api.security_framework.active_sessions', {"test_session": Mock()}), \
             patch('src.server.api_routes.security_api.security_framework.authorize_access', return_value=True), \
             patch('src.server.api_routes.security_api.access_control.check_access', return_value=Mock(decision="permit")):

            auth_request = {
                "session_id": "test_session",
                "resource": "/api/data",
                "action": "read"
            }

            response = client.post(
                "/api/security/authorize",
                json=auth_request,
                headers={"Authorization": "Bearer test_token"}
            )
            assert response.status_code == 200

            data = response.json()
            assert data["authorized"] is True

    def test_threat_analysis_endpoint(self):
        """Test threat analysis endpoint"""
        threat_request = {
            "event_data": {
                "source_ip": "192.168.1.100",
                "user_agent": "Mozilla/5.0",
                "request_path": "/api/admin",
                "method": "GET"
            },
            "source_type": "api"
        }

        with patch('src.server.api_routes.security_api.threat_detection.analyze_event') as mock_analyze:
            mock_threat = Mock()
            mock_threat.threat_type.value = "suspicious_activity"
            mock_threat.severity.value = "medium"
            mock_threat.confidence = 0.8
            mock_threat.event_id = "threat_123"
            mock_threat.mitigation_actions = ["block_ip", "alert_admin"]
            mock_analyze.return_value = mock_threat

            response = client.post(
                "/api/security/threat/analyze",
                json=threat_request,
                headers={"Authorization": "Bearer test_token"}
            )
            assert response.status_code == 200

            data = response.json()
            assert data["threat_detected"] is True
            assert data["threat_type"] == "suspicious_activity"
            assert data["severity"] == "medium"

    def test_encryption_endpoint(self):
        """Test encryption endpoint"""
        encrypt_request = {
            "data": "sensitive_information",
            "algorithm": "AES_256_GCM"
        }

        with patch('src.server.api_routes.security_api.encryption_service.encrypt') as mock_encrypt:
            mock_result = Mock()
            mock_result.key_id = "test_key"
            mock_result.algorithm.value = "AES_256_GCM"
            mock_result.to_dict.return_value = {
                "encrypted_data": "encrypted_bytes_here",
                "iv": "iv_here",
                "tag": "tag_here"
            }
            mock_encrypt.return_value = mock_result

            response = client.post(
                "/api/security/encrypt",
                json=encrypt_request,
                headers={"Authorization": "Bearer test_token"}
            )
            assert response.status_code == 200

            data = response.json()
            assert data["success"] is True
            assert data["key_id"] == "test_key"

    def test_decryption_endpoint(self):
        """Test decryption endpoint"""
        decrypt_request = {
            "encrypted_data": {
                "encrypted_data": "encrypted_bytes_here",
                "iv": "iv_here",
                "tag": "tag_here",
                "key_id": "test_key",
                "algorithm": "AES_256_GCM"
            }
        }

        with patch('src.server.api_routes.security_api.encryption_service.decrypt', return_value=b"decrypted_data"):
            response = client.post(
                "/api/security/decrypt",
                json=decrypt_request,
                headers={"Authorization": "Bearer test_token"}
            )
            assert response.status_code == 200

            data = response.json()
            assert data["success"] is True
            assert data["decrypted_data"] == "decrypted_data"

    def test_audit_logging_endpoint(self):
        """Test audit logging endpoint"""
        audit_request = {
            "level": "INFO",
            "category": "AUTHENTICATION",
            "event_type": "user_login",
            "user_id": "test_user",
            "source_ip": "127.0.0.1",
            "resource": "/api/login",
            "action": "login",
            "result": "success",
            "details": {"method": "password"},
            "compliance_frameworks": ["GDPR", "SOC2"]
        }

        with patch('src.server.api_routes.security_api.audit_logger.log', return_value="event_123"):
            response = client.post(
                "/api/security/audit/log",
                json=audit_request,
                headers={"Authorization": "Bearer test_token"}
            )
            assert response.status_code == 200

            data = response.json()
            assert data["success"] is True
            assert data["event_id"] == "event_123"

    def test_access_control_check_endpoint(self):
        """Test access control check endpoint"""
        access_request = {
            "subject_id": "test_user",
            "resource": "/api/admin/users",
            "action": "read",
            "context": {"department": "engineering"}
        }

        with patch('src.server.api_routes.security_api.access_control.check_access') as mock_check:
            mock_result = Mock()
            mock_result.decision.value = "permit"
            mock_result.reason = "User has admin role"
            mock_result.processing_time_ms = 5.2
            mock_check.return_value = mock_result

            response = client.post(
                "/api/security/access/check",
                json=access_request,
                headers={"Authorization": "Bearer test_token"}
            )
            assert response.status_code == 200

            data = response.json()
            assert data["decision"] == "permit"

    def test_security_metrics_endpoint(self):
        """Test security metrics endpoint"""
        with patch('src.server.api_routes.security_api.security_framework.get_security_metrics', return_value={"active_sessions": 10}), \
             patch('src.server.api_routes.security_api.zero_trust_model.get_zero_trust_metrics', return_value={"verified_entities": 5}), \
             patch('src.server.api_routes.security_api.threat_detection.get_threat_metrics', return_value={"threats_detected": 2}), \
             patch('src.server.api_routes.security_api.encryption_service.get_encryption_metrics', return_value={"keys_managed": 3}), \
             patch('src.server.api_routes.security_api.audit_logger.get_audit_metrics', return_value={"audit_events": 100}), \
             patch('src.server.api_routes.security_api.access_control.get_access_control_metrics', return_value={"total_subjects": 20}):

            response = client.get(
                "/api/security/metrics",
                headers={"Authorization": "Bearer test_token"}
            )
            assert response.status_code == 200

            data = response.json()
            assert "security_framework" in data
            assert "zero_trust" in data
            assert "threat_detection" in data
            assert "encryption" in data
            assert "audit" in data
            assert "access_control" in data


class TestComplianceAPI:
    """Integration tests for Compliance API endpoints"""

    def test_gdpr_assessment_endpoint(self):
        """Test GDPR assessment endpoint"""
        assessment_request = {
            "framework": "GDPR",
            "scope": {"include_all": True},
            "include_recommendations": True
        }

        with patch('src.server.api_routes.compliance_api.gdpr_manager.conduct_compliance_assessment') as mock_assess:
            mock_assessment = Mock()
            mock_assessment.id = "gdpr_assessment_123"
            mock_assessment.overall_status.value = "compliant"
            mock_assessment.compliance_score = 85.0
            mock_assessment.findings = [{"description": "Test finding"}]
            mock_assessment.deficiencies = []
            mock_assessment.recommendations = ["Recommendation 1"]
            mock_assessment.assessment_date = datetime.now()
            mock_assess.return_value = mock_assessment

            response = client.post(
                "/api/compliance/gdpr/assessment",
                json=assessment_request,
                headers={"Authorization": "Bearer test_token"}
            )
            assert response.status_code == 200

            data = response.json()
            assert data["framework"] == "GDPR"
            assert data["overall_status"] == "compliant"
            assert data["compliance_score"] == 85.0

    def test_gdpr_dsar_endpoint(self):
        """Test GDPR DSAR endpoint"""
        dsar_request = {
            "request_type": "access",
            "subject_id": "user_123",
            "request_details": {
                "data_categories": ["personal_data"]
            },
            "identity_verification": {
                "method": "email",
                "verified": True
            }
        }

        with patch('src.server.api_routes.compliance_api.gdpr_manager.process_data_subject_request') as mock_dsar:
            mock_dsar_result = Mock()
            mock_dsar_result.request_id = "dsar_123"
            mock_dsar_result.status.value = "received"
            mock_dsar_result.estimated_completion_date = datetime.now() + timedelta(days=30)
            mock_dsar_result.processing_steps = ["Step 1", "Step 2"]
            mock_dsar_result.requirements_met = ["Requirement 1"]
            mock_dsar.return_value = mock_dsar_result

            response = client.post(
                "/api/compliance/gdpr/dsar",
                json=dsar_request,
                headers={"Authorization": "Bearer test_token"}
            )
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "received"
            assert data["request_id"] == "dsar_123"

    def test_soc2_assessment_endpoint(self):
        """Test SOC2 assessment endpoint"""
        assessment_request = {
            "framework": "SOC2",
            "scope": {"trust_services": ["security", "availability"]},
            "include_recommendations": True
        }

        with patch('src.server.api_routes.compliance_api.soc2_manager.conduct_compliance_assessment') as mock_assess:
            mock_assessment = Mock()
            mock_assessment.id = "soc2_assessment_123"
            mock_assessment.overall_status.value = "partially_compliant"
            mock_assessment.compliance_score = 75.0
            mock_assessment.assessment_date = datetime.now()
            mock_assess.return_value = mock_assessment

            response = client.post(
                "/api/compliance/soc2/assessment",
                json=assessment_request,
                headers={"Authorization": "Bearer test_token"}
            )
            assert response.status_code == 200

            data = response.json()
            assert data["framework"] == "SOC2"
            assert data["overall_status"] == "partially_compliant"

    def test_hipaa_assessment_endpoint(self):
        """Test HIPAA assessment endpoint"""
        assessment_request = {
            "framework": "HIPAA",
            "scope": {"include_bas": True},
            "include_recommendations": True
        }

        with patch('src.server.api_routes.compliance_api.hipaa_manager.conduct_compliance_assessment') as mock_assess:
            mock_assessment = Mock()
            mock_assessment.id = "hipaa_assessment_123"
            mock_assessment.overall_status.value = "compliant"
            mock_assessment.compliance_score = 90.0
            mock_assessment.assessment_date = datetime.now()
            mock_assess.return_value = mock_assessment

            response = client.post(
                "/api/compliance/hipaa/assessment",
                json=assessment_request,
                headers={"Authorization": "Bearer test_token"}
            )
            assert response.status_code == 200

            data = response.json()
            assert data["framework"] == "HIPAA"
            assert data["overall_status"] == "compliant"

    def test_hipaa_business_associate_endpoint(self):
        """Test HIPAA business associate endpoint"""
        ba_request = {
            "ba_name": "Medical Service Provider",
            "services_provided": ["medical_billing", "claims_processing"],
            "data_access_level": "limited",
            "risk_factors": {
                "data_sensitivity": 0.7,
                "security_maturity": 0.8
            }
        }

        with patch('src.server.api_routes.compliance_api.hipaa_manager.assess_business_associate') as mock_assess:
            mock_ba = Mock()
            mock_ba.id = "ba_123"
            mock_ba.risk_assessment_score = 75.0
            mock_ba.compliance_status.value = "compliant"
            mock_ba.control_requirements = ["Requirement 1", "Requirement 2"]
            mock_ba.recommended_actions = ["Action 1"]
            mock_assess.return_value = mock_ba

            response = client.post(
                "/api/compliance/hipaa/business-associate",
                json=ba_request,
                headers={"Authorization": "Bearer test_token"}
            )
            assert response.status_code == 200

            data = response.json()
            assert data["risk_assessment_score"] == 75.0
            assert data["compliance_status"] == "compliant"

    def test_compliance_report_generation_endpoint(self):
        """Test compliance report generation endpoint"""
        report_request = {
            "framework": "GDPR",
            "report_type": "detailed",
            "format": "pdf",
            "include_evidence": True
        }

        with patch('src.server.api_routes.compliance_api.reporting_service.generate_report') as mock_generate:
            mock_report = Mock()
            mock_report.id = "report_123"
            mock_report.download_url = "/api/compliance/reports/report_123.pdf"
            mock_report.generated_at = datetime.now()
            mock_report.format = "pdf"
            mock_generate.return_value = mock_report

            response = client.post(
                "/api/compliance/reports",
                json=report_request,
                headers={"Authorization": "Bearer test_token"}
            )
            assert response.status_code == 200

            data = response.json()
            assert data["report_id"] == "report_123"
            assert data["format"] == "pdf"

    @pytest.mark.asyncio
    async def test_api_error_handling(self):
        """Test API error handling"""
        # Test authentication required
        response = client.get("/api/security/metrics")
        assert response.status_code == 401

        # Test invalid token
        response = client.get(
            "/api/security/metrics",
            headers={"Authorization": "Bearer invalid_token"}
        )
        assert response.status_code == 401

        # Test service error
        with patch('src.server.api_routes.security_api.security_framework.get_security_metrics', side_effect=Exception("Service error")):
            response = client.get(
                "/api/security/metrics",
                headers={"Authorization": "Bearer test_token"}
            )
            assert response.status_code == 500


class TestSecurityPerformance:
    """Performance tests for security endpoints"""

    @pytest.mark.asyncio
    async def test_authentication_performance(self):
        """Test authentication endpoint performance"""
        import time

        auth_request = {
            "username": "test_user",
            "credentials": {"password": "test_password", "method": "password"},
            "ip_address": "127.0.0.1",
            "user_agent": "test_agent"
        }

        with patch('src.server.api_routes.security_api.security_framework.authenticate_user') as mock_auth:
            mock_context = Mock()
            mock_context.user_id = "test_user"
            mock_context.session_id = "test_session"
            mock_context.expires_at = datetime.now() + timedelta(hours=1)
            mock_auth.return_value = mock_context

            start_time = time.time()
            response = client.post("/api/security/authenticate", json=auth_request)
            end_time = time.time()

            assert response.status_code == 200
            assert (end_time - start_time) < 1.0  # Should complete within 1 second

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import asyncio

        async def make_request():
            with patch('src.server.api_routes.security_api.security_framework.verify_token', return_value={"user_id": "test_user"}):
                response = client.get(
                    "/api/security/metrics",
                    headers={"Authorization": "Bearer test_token"}
                )
                return response

        # Make 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        responses = await asyncio.gather(*tasks)

        assert all(response.status_code == 200 for response in responses)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])