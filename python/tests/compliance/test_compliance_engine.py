"""
Compliance Engine Tests

Comprehensive test suite for compliance management components
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

from src.agents.compliance.compliance_engine import (
    ComplianceEngine, ComplianceFramework, ComplianceControl, ComplianceStatus,
    ComplianceRiskLevel, ComplianceAssessment, ComplianceEvidence
)
from src.agents.compliance.gdpr_compliance import (
    GDPRComplianceManager, DataSubjectRequest, GDPRViolationType, DSARStatus
)
from src.agents.compliance.soc2_compliance import (
    SOC2ComplianceManager, SOC2ControlType, SOC2TrustServicesCriteria,
    SOC2TestResult, SOC2SystemDescription
)
from src.agents.compliance.hipaa_compliance import (
    HIPAAComplianceManager, PHIDataType, HIPAASection, HIPAAViolationType
)


class TestComplianceEngine:
    """Test cases for ComplianceEngine"""

    @pytest.fixture
    def compliance_engine(self):
        """Create a compliance engine instance for testing"""
        return ComplianceEngine()

    def test_engine_initialization(self, compliance_engine):
        """Test compliance engine initialization"""
        assert compliance_engine.frameworks == {}
        assert len(compliance_engine.assessments) == 0

    def test_register_framework(self, compliance_engine):
        """Test framework registration"""
        framework = Mock(spec=ComplianceFramework)
        framework.framework_id = "test_framework"
        framework.name = "Test Framework"

        compliance_engine.register_framework(framework)
        assert "test_framework" in compliance_engine.frameworks

    def test_conduct_assessment(self, compliance_engine):
        """Test compliance assessment"""
        # Mock framework
        framework = Mock(spec=ComplianceFramework)
        framework.framework_id = "test_framework"
        framework.conduct_assessment.return_value = ComplianceAssessment(
            id="test_assessment",
            framework_id="test_framework",
            framework_name="Test Framework",
            assessment_date=datetime.now(),
            assessor_id="system",
            overall_status=ComplianceStatus.COMPLIANT,
            compliance_score=85.0,
            findings=[],
            deficiencies=[],
            evidence=[],
            recommendations=[],
            next_review_date=datetime.now() + timedelta(days=365)
        )

        compliance_engine.register_framework(framework)

        # Conduct assessment
        assessment = compliance_engine.conduct_assessment(
            framework_id="test_framework",
            scope={"include_all": True}
        )

        assert assessment.framework_id == "test_framework"
        assert assessment.overall_status == ComplianceStatus.COMPLIANT
        assert assessment.compliance_score == 85.0

    def test_framework_not_found(self, compliance_engine):
        """Test assessment with non-existent framework"""
        with pytest.raises(ValueError, match="Framework not found"):
            compliance_engine.conduct_assessment(
                framework_id="non_existent_framework"
            )

    def test_get_assessment_history(self, compliance_engine):
        """Test assessment history retrieval"""
        # Create mock assessments
        for i in range(3):
            assessment = ComplianceAssessment(
                id=f"assessment_{i}",
                framework_id="test_framework",
                framework_name="Test Framework",
                assessment_date=datetime.now() - timedelta(days=i),
                assessor_id="system",
                overall_status=ComplianceStatus.COMPLIANT,
                compliance_score=80.0 + i * 5,
                findings=[],
                deficiencies=[],
                evidence=[],
                recommendations=[],
                next_review_date=datetime.now() + timedelta(days=365)
            )
            compliance_engine.assessments[assessment.id] = assessment

        history = compliance_engine.get_assessment_history("test_framework")
        assert len(history) == 3
        assert history[0].compliance_score == 85.0  # Most recent first

    def test_compliance_trend_analysis(self, compliance_engine):
        """Test compliance trend analysis"""
        # Create assessments with varying scores
        scores = [70.0, 75.0, 80.0, 85.0, 82.0]
        for i, score in enumerate(scores):
            assessment = ComplianceAssessment(
                id=f"assessment_{i}",
                framework_id="test_framework",
                framework_name="Test Framework",
                assessment_date=datetime.now() - timedelta(days=(len(scores) - i) * 30),
                assessor_id="system",
                overall_status=ComplianceStatus.COMPLIANT,
                compliance_score=score,
                findings=[],
                deficiencies=[],
                evidence=[],
                recommendations=[],
                next_review_date=datetime.now() + timedelta(days=365)
            )
            compliance_engine.assessments[assessment.id] = assessment

        trend = compliance_engine.analyze_compliance_trends("test_framework")
        assert "trend" in trend
        assert "improvement_rate" in trend
        assert "predicted_score" in trend


class TestGDPRComplianceManager:
    """Test cases for GDPRComplianceManager"""

    @pytest.fixture
    def gdpr_manager(self):
        """Create a GDPR compliance manager instance for testing"""
        return GDPRComplianceManager("Test Organization")

    def test_initialization(self, gdpr_manager):
        """Test GDPR manager initialization"""
        assert gdpr_manager.organization_name == "Test Organization"
        assert len(gdpr_manager.data_subject_requests) == 0
        assert len(gdpr_manager.data_processing_activities) == 0

    def test_data_subject_request_processing(self, gdpr_manager):
        """Test data subject request processing"""
        request_data = {
            "request_type": "access",
            "subject_id": "user_123",
            "request_details": {
                "data_categories": ["personal_data", "communications"]
            },
            "identity_verification": {
                "method": "email",
                "verified": True
            }
        }

        dsar = gdpr_manager.process_data_subject_request(**request_data)

        assert dsar.request_type == "access"
        assert dsar.subject_id == "user_123"
        assert dsar.status == DSARStatus.RECEIVED
        assert dsar.estimated_completion_date > datetime.now()

    def test_data_mapping(self, gdpr_manager):
        """Test data mapping functionality"""
        # Add data processing activity
        activity = {
            "activity_id": "user_management",
            "data_categories": ["personal_data", "contact_data"],
            "purposes": ["user_management", "service_provision"],
            "retention_period": 365,
            "data_processors": ["email_service"]
        }

        gdpr_manager.register_processing_activity(activity)

        # Query data mapping
        mapping = gdpr_manager.get_data_mapping("user_123")
        assert "user_management" in mapping

    def test_consent_management(self, gdpr_manager):
        """Test consent management"""
        consent_id = gdpr_manager.record_consent(
            subject_id="user_123",
            purpose_id="marketing_emails",
            consent_given=True,
            method="checkbox",
            timestamp=datetime.now()
        )

        assert consent_id is not None

        # Check consent status
        has_consent = gdpr_manager.check_consent("user_123", "marketing_emails")
        assert has_consent is True

    def test_breach_notification(self, gdpr_manager):
        """Test breach notification functionality"""
        breach_data = {
            "breach_type": GDPRViolationType.UNAUTHORIZED_ACCESS,
            "affected_individuals": 100,
            "data_categories": ["personal_data"],
            "description": "Unauthorized access to user database",
            "measures_taken": ["system_lockdown", "investigation"]
        }

        breach_id = gdpr_manager.report_breach(breach_data)
        assert breach_id is not None

        # Check if notification threshold is met (>72 hours)
        notification_required = gdpr_manager.check_notification_requirement(breach_id)
        assert notification_required is True

    def test_dpii_assessment(self, gdpr_manager):
        """Test DPIA (Data Protection Impact Assessment)"""
        dpia_data = {
            "processing_activity": "user_profiling",
            "data_categories": ["personal_data", "behavioral_data"],
            "risks": ["profiling_discrimination", "privacy_violation"],
            "measures": ["anonymization", "consent_mechanism"]
        }

        dpia = gdpr_manager.conduct_dpia(dpia_data)
        assert dpia is not None
        assert dpia.overall_risk_level in ["low", "medium", "high"]

    def test_compliance_assessment(self, gdpr_manager):
        """Test comprehensive GDPR compliance assessment"""
        assessment = gdpr_manager.conduct_compliance_assessment()

        assert assessment.framework_id == "GDPR"
        assert assessment.overall_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.PARTIALLY_COMPLIANT, ComplianceStatus.NON_COMPLIANT]
        assert 0 <= assessment.compliance_score <= 100

        # Should have findings related to GDPR articles
        assert len(assessment.findings) > 0


class TestSOC2ComplianceManager:
    """Test cases for SOC2ComplianceManager"""

    @pytest.fixture
    def soc2_manager(self):
        """Create a SOC2 compliance manager instance for testing"""
        return SOC2ComplianceManager("Test Organization")

    def test_initialization(self, soc2_manager):
        """Test SOC2 manager initialization"""
        assert soc2_manager.organization_name == "Test Organization"
        assert len(soc2_manager.controls) > 0  # Should have default controls
        assert len(soc2_manager.test_results) == 0

    def test_control_management(self, soc2_manager):
        """Test control management"""
        control = {
            "control_id": "CC-1",
            "name": "Access Control Management",
            "description": "Logical access controls for information systems",
            "type": SOC2ControlType.LOGICAL_ACCESS,
            "criteria": SOC2TrustServicesCriteria.SECURITY,
            "implementation_status": "implemented"
        }

        soc2_manager.add_control(control)
        assert "CC-1" in soc2_manager.controls

    def test_control_testing(self, soc2_manager):
        """Test control testing functionality"""
        test_data = {
            "control_id": "CC-1",
            "test_procedures": ["Review access logs", "Verify access rights"],
            "testing_period": {
                "start": datetime.now() - timedelta(days=30),
                "end": datetime.now()
            },
            "tester": "audit_team"
        }

        test_result = soc2_manager.test_control(test_data)
        assert test_result is not None
        assert test_result.control_id == "CC-1"
        assert test_result.tester == "audit_team"

    def test_system_description(self, soc2_manager):
        """Test system description management"""
        system_desc = {
            "system_name": "Customer Management System",
            "boundary": "Cloud-hosted CRM application",
            "components": ["database", "application_server", "web_interface"],
            "trust_service_categories": [SOC2TrustServicesCriteria.SECURITY, SOC2TrustServicesCriteria.AVAILABILITY]
        }

        description = soc2_manager.create_system_description(system_desc)
        assert description.system_name == "Customer Management System"
        assert len(description.trust_service_categories) == 2

    def test_risk_assessment(self, soc2_manager):
        """Test SOC2 risk assessment"""
        risk_data = {
            "risk_description": "Unauthorized access to sensitive data",
            "likelihood": "medium",
            "impact": "high",
            "existing_controls": ["encryption", "access_controls"],
            "risk_owner": "IT Security Manager"
        }

        risk = soc2_manager.assess_risk(risk_data)
        assert risk is not None
        assert risk.risk_level in ["low", "medium", "high"]

    def test_compliance_report_generation(self, soc2_manager):
        """Test SOC2 compliance report generation"""
        report = soc2_manager.generate_compliance_report()
        assert report is not None
        assert "organization" in report
        assert "controls_assessed" in report
        assert "compliance_status" in report

    def test_compliance_assessment(self, soc2_manager):
        """Test comprehensive SOC2 compliance assessment"""
        assessment = soc2_manager.conduct_compliance_assessment()

        assert assessment.framework_id == "SOC2"
        assert assessment.overall_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.PARTIALLY_COMPLIANT, ComplianceStatus.NON_COMPLIANT]
        assert 0 <= assessment.compliance_score <= 100

        # Should evaluate Trust Services Criteria
        assert len(assessment.findings) > 0


class TestHIPAAComplianceManager:
    """Test cases for HIPAAComplianceManager"""

    @pytest.fixture
    def hipaa_manager(self):
        """Create a HIPAA compliance manager instance for testing"""
        return HIPAAComplianceManager("Test Healthcare Organization")

    def test_initialization(self, hipaa_manager):
        """Test HIPAA manager initialization"""
        assert hipaa_manager.covered_entity_name == "Test Healthcare Organization"
        assert len(hipaa_manager.controls) > 0  # Should have default HIPAA controls
        assert len(hipaa_manager.business_associates) == 0

    def test_phi_element_management(self, hipaa_manager):
        """Test PHI element management"""
        phi_id = hipaa_manager.register_phi_element(
            data_type=PHIDataType.MEDICAL_RECORD,
            description="Patient medical records",
            location="electronic_health_records",
            retention_period=3650  # 10 years
        )

        assert phi_id is not None
        assert phi_id in hipaa_manager.phi_elements

    def test_phi_access_logging(self, hipaa_manager):
        """Test PHI access logging"""
        phi_id = hipaa_manager.register_phi_element(
            data_type=PHIDataType.DEMOGRAPHIC,
            description="Patient demographics",
            location="patient_database",
            retention_period=3650
        )

        # Log access
        access_logged = hipaa_manager.access_phi(
            phi_id=phi_id,
            user_id="doctor_123",
            purpose="patient_care"
        )

        assert access_logged is True
        phi_element = hipaa_manager.phi_elements[phi_id]
        assert phi_element.last_accessed is not None

    def test_business_associate_management(self, hipaa_manager):
        """Test business associate management"""
        ba_data = {
            "ba_name": "Medical Billing Service",
            "services_provided": ["medical_billing", "claims_processing"],
            "data_access_level": "limited",
            "risk_factors": {
                "data_sensitivity": 0.8,
                "security_maturity": 0.7
            }
        }

        business_associate = hipaa_manager.assess_business_associate(**ba_data)
        assert business_associate.name == "Medical Billing Service"
        assert business_associate.compliance_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.PARTIALLY_COMPLIANT, ComplianceStatus.NON_COMPLIANT]

    def test_breach_reporting(self, hipaa_manager):
        """Test HIPAA breach reporting"""
        breach_data = {
            "breach_type": HIPAAViolationType.UNAUTHORIZED_DISCLOSURE,
            "breach_date": datetime.now() - timedelta(days=1),
            "discovery_date": datetime.now(),
            "affected_individuals": 500,
            "data_types": [PHIDataType.MEDICAL_RECORD, PHIDataType.DEMOGRAPHIC],
            "breach_description": "Unauthorized disclosure of patient records",
            "cause": "hacking"
        }

        breach = hipaa_manager.report_breach(breach_data)
        assert breach is not None
        assert breach.affected_individuals == 500
        assert breach.breach_type == HIPAAViolationType.UNAUTHORIZED_DISCLOSURE

        # Should trigger notification requirements for large breach (>500)
        assert breach.media_notified is True

    def test_workforce_training_management(self, hipaa_manager):
        """Test workforce training management"""
        training_data = {
            "employee_id": "staff_123",
            "training_type": "privacy_rule",
            "training_date": datetime.now(),
            "training_content": "HIPAA Privacy Rule Requirements",
            "trainer_id": "compliance_officer",
            "score": 95.0,
            "topics_covered": ["patient_rights", "minimum_necessary", "safeguards"]
        }

        training = hipaa_manager.manage_workforce_training(training_data)
        assert training.employee_id == "staff_123"
        assert training.score == 95.0
        assert training.refresher_required is True

    def test_risk_assessment(self, hipaa_manager):
        """Test HIPAA risk assessment"""
        risk_data = {
            "asset_id": "ehr_system",
            "threat_description": "Malware infection",
            "vulnerability_description": "Unpatched operating system",
            "likelihood": "medium",
            "impact": "high",
            "existing_safeguards": ["antivirus", "firewall"],
            "risk_owner": "IT Director"
        }

        risk_assessment = hipaa_manager.conduct_risk_assessment(**risk_data)
        assert risk_assessment is not None
        assert risk_assessment.risk_level in ["low", "medium", "high", "critical"]

    def test_phi_deidentification(self, hipaa_manager):
        """Test PHI deidentification"""
        phi_data = {
            "data_type": PHIDataType.IDENTIFIERS,
            "original_value": "John Doe, DOB: 1990-01-15, SSN: 123-45-6789",
            "method": "generalization",
            "auditor_id": "data_analyst"
        }

        deidentified = hipaa_manager.deidentify_phi(phi_data)
        assert deidentified is not None
        assert deidentified.original_value != deidentified.deidentified_value
        assert deidentified.verification_result is True

    def test_compliance_assessment(self, hipaa_manager):
        """Test comprehensive HIPAA compliance assessment"""
        assessment = hipaa_manager.conduct_compliance_assessment()

        assert assessment.framework_id == "HIPAA"
        assert assessment.overall_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.PARTIALLY_COMPLIANT, ComplianceStatus.NON_COMPLIANT]
        assert 0 <= assessment.compliance_score <= 100

        # Should evaluate HIPAA rules
        assert len(assessment.findings) > 0

        # Should check business associate compliance
        assert len(assessment.deficiencies) >= 0


class TestComplianceIntegration:
    """Integration tests for compliance frameworks"""

    @pytest.fixture
    def compliance_engine(self):
        """Create a compliance engine with all frameworks"""
        engine = ComplianceEngine()

        # Register all compliance frameworks
        gdpr_manager = GDPRComplianceManager("Test Org")
        soc2_manager = SOC2ComplianceManager("Test Org")
        hipaa_manager = HIPAAManager("Test Org")

        engine.register_framework(gdpr_manager)
        engine.register_framework(soc2_manager)
        engine.register_framework(hipaa_manager)

        return engine

    def test_cross_framework_assessment(self, compliance_engine):
        """Test assessment across multiple frameworks"""
        # This would run assessments for GDPR, SOC2, and HIPAA
        assessments = {}

        for framework_id in ["gdpr", "soc2", "hipaa"]:
            try:
                assessment = compliance_engine.conduct_assessment(framework_id)
                assessments[framework_id] = assessment
            except Exception as e:
                pytest.fail(f"Assessment failed for {framework_id}: {str(e)}")

        assert len(assessments) == 3

        # Each assessment should have valid data
        for framework_id, assessment in assessments.items():
            assert assessment.framework_id == framework_id
            assert 0 <= assessment.compliance_score <= 100

    def test_compliance_correlation(self, compliance_engine):
        """Test correlation between different compliance frameworks"""
        # Run assessments
        assessments = {}
        for framework_id in ["gdpr", "soc2", "hipaa"]:
            assessments[framework_id] = compliance_engine.conduct_assessment(framework_id)

        # Analyze correlations
        correlation = compliance_engine.analyze_framework_correlations(assessments)

        assert "overall_correlation" in correlation
        assert "shared_controls" in correlation
        assert "conflicting_requirements" in correlation

    def test_compliance_reporting(self, compliance_engine):
        """Test comprehensive compliance reporting"""
        # Generate combined report
        report = compliance_engine.generate_comprehensive_report()

        assert "summary" in report
        assert "framework_scores" in report
        assert "recommendations" in report
        assert "next_steps" in report

        # Should include all frameworks
        assert len(report["framework_scores"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])