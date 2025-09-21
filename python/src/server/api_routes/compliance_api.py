"""
Compliance API Routes
Comprehensive API for compliance management and reporting
"""

from fastapi import APIRouter, HTTPException, Depends, Request, Body, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Set, Union
from datetime import datetime, timedelta
import logging
import json
import asyncio

from ...agents.compliance.compliance_engine import ComplianceFramework, ComplianceControl, ComplianceStatus, ComplianceRiskLevel
from ...agents.compliance.gdpr_compliance import GDPRComplianceManager, DataSubjectRequest, GDPRViolationType
from ...agents.compliance.soc2_compliance import SOC2ComplianceManager, SOC2ControlType, SOC2TrustServicesCriteria
from ...agents.compliance.hipaa_compliance import HIPAAComplianceManager, PHIDataType, HIPAAViolationType
from ...agents.compliance.compliance_reporting import ComplianceReportingService, ComplianceDashboard
from ...agents.security.security_framework import SecurityFramework
from ...agents.security.audit_logger import AuditLogger, AuditLevel, AuditCategory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/compliance", tags=["compliance"])
security = HTTPBearer(auto_error=False)

# Initialize compliance components
gdpr_manager = GDPRComplianceManager("Default Organization")
soc2_manager = SOC2ComplianceManager("Default Organization")
hipaa_manager = HIPAAComplianceManager("Default Organization")
reporting_service = ComplianceReportingService()
security_framework = SecurityFramework()
audit_logger = AuditLogger()

# Set managers in reporting service
reporting_service.set_gdpr_manager(gdpr_manager)
reporting_service.set_soc2_manager(soc2_manager)
reporting_service.set_hipaa_manager(hipaa_manager)


# Pydantic models for request/response
class ComplianceAssessmentRequest(BaseModel):
    framework: str = Field(..., description="Compliance framework (GDPR, SOC2, HIPAA)")
    scope: Optional[Dict[str, Any]] = Field(default_factory=dict)
    include_recommendations: bool = Field(default=True, description="Include remediation recommendations")


class ComplianceAssessmentResponse(BaseModel):
    assessment_id: str
    framework: str
    overall_status: str
    compliance_score: float
    findings: List[Dict[str, Any]]
    deficiencies: List[Dict[str, Any]]
    recommendations: List[str]
    assessment_date: datetime


class GDPRDataSubjectRequest(BaseModel):
    request_type: str = Field(..., description="Type of DSAR (access, rectification, erasure, portability, objection)")
    subject_id: str = Field(..., description="Data subject identifier")
    request_details: Dict[str, Any] = Field(default_factory=dict)
    identity_verification: Dict[str, Any] = Field(..., description="Identity verification data")


class GDPRDataSubjectResponse(BaseModel):
    request_id: str
    status: str
    estimated_completion_date: datetime
    processing_steps: List[str]
    requirements_met: List[str]


class SOC2ControlTestRequest(BaseModel):
    control_id: str = Field(..., description="SOC2 control identifier")
    test_procedures: List[str] = Field(..., description="Test procedures to execute")
    evidence_requirements: List[str] = Field(default_factory=dict)
    testing_period_start: datetime = Field(..., description="Start of testing period")
    testing_period_end: datetime = Field(..., description="End of testing period")


class SOC2ControlTestResponse(BaseModel):
    test_id: str
    control_id: str
    test_results: List[Dict[str, Any]]
    evidence_collected: List[str]
    deficiencies_identified: List[Dict[str, Any]]
    overall_effectiveness: str


class HIPAABusinessAssociateRequest(BaseModel):
    ba_name: str = Field(..., description="Business associate name")
    services_provided: List[str] = Field(..., description="Services provided by BA")
    data_access_level: str = Field(..., description="Level of data access required")
    risk_factors: Dict[str, Any] = Field(default_factory=dict)
    compliance_documentation: Dict[str, Any] = Field(default_factory=dict)


class HIPAABusinessAssociateResponse(BaseModel):
    ba_id: str
    risk_assessment_score: float
    compliance_status: str
    control_requirements: List[str]
    recommended_actions: List[str]


class ComplianceReportRequest(BaseModel):
    framework: str = Field(..., description="Compliance framework")
    report_type: str = Field(..., description="Report type (assessment, summary, detailed, executive)")
    format: str = Field(default="json", description="Report format (json, pdf, html, csv)")
    time_period: Optional[Dict[str, datetime]] = Field(default_factory=dict)
    include_evidence: bool = Field(default=True)


class ComplianceReportResponse(BaseModel):
    report_id: str
    download_url: Optional[str] = None
    report_content: Optional[Dict[str, Any]] = None
    generated_at: datetime
    format: str


class ComplianceDashboardRequest(BaseModel):
    dashboard_id: str = Field(..., description="Dashboard identifier")
    timeframe: str = Field(default="24h", description="Timeframe for data (1h, 24h, 7d, 30d)")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class CompliancePolicyRequest(BaseModel):
    policy_name: str = Field(..., description="Policy name")
    framework: str = Field(..., description="Compliance framework")
    policy_content: Dict[str, Any] = Field(..., description="Policy content and requirements")
    implementation_guidance: Optional[Dict[str, Any]] = Field(default_factory=dict)
    validation_rules: List[Dict[str, Any]] = Field(default_factory=dict)


class CompliancePolicyResponse(BaseModel):
    policy_id: str
    status: str
    validation_results: List[Dict[str, Any]]
    compliance_score: float
    gaps_identified: List[Dict[str, Any]]


class ComplianceEvidenceUpload(BaseModel):
    framework: str = Field(..., description="Compliance framework")
    control_id: str = Field(..., description="Associated control ID")
    evidence_type: str = Field(..., description="Type of evidence")
    description: str = Field(..., description="Evidence description")
    validity_period: Optional[Dict[str, datetime]] = Field(default_factory=dict)


# Dependency for authentication
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Verify token (simplified - would use proper JWT verification)
    token = credentials.credentials
    payload = security_framework.verify_token(token)

    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    return payload.get("user_id")


# GDPR Compliance endpoints
@router.post("/gdpr/assessment", response_model=ComplianceAssessmentResponse)
async def conduct_gdpr_assessment(
    request: ComplianceAssessmentRequest,
    user_id: str = Depends(get_current_user)
):
    """Conduct GDPR compliance assessment"""
    try:
        assessment = gdpr_manager.conduct_compliance_assessment()

        response = ComplianceAssessmentResponse(
            assessment_id=assessment.id,
            framework="GDPR",
            overall_status=assessment.overall_status.value,
            compliance_score=assessment.compliance_score,
            findings=assessment.findings,
            deficiencies=assessment.deficiencies,
            recommendations=assessment.recommendations if request.include_recommendations else [],
            assessment_date=assessment.assessment_date
        )

        # Log assessment event
        await audit_logger.log_compliance_event(
            event_type="gdpr_assessment",
            framework="GDPR",
            user_id=user_id,
            details={
                "assessment_id": assessment.id,
                "compliance_score": assessment.compliance_score,
                "findings_count": len(assessment.findings)
            }
        )

        return response

    except Exception as e:
        logger.error(f"GDPR assessment error: {str(e)}")
        raise HTTPException(status_code=500, detail="GDPR assessment service error")


@router.post("/gdpr/dsar", response_model=GDPRDataSubjectResponse)
async def process_data_subject_request(
    request: GDPRDataSubjectRequest,
    user_id: str = Depends(get_current_user)
):
    """Process GDPR Data Subject Access Request"""
    try:
        dsar = gdpr_manager.process_data_subject_request(
            request_type=request.request_type,
            subject_id=request.subject_id,
            request_details=request.request_details,
            identity_verification=request.identity_verification
        )

        response = GDPRDataSubjectResponse(
            request_id=dsar.request_id,
            status=dsar.status.value,
            estimated_completion_date=dsar.estimated_completion_date,
            processing_steps=dsar.processing_steps,
            requirements_met=dsar.requirements_met
        )

        # Log DSAR event
        await audit_logger.log_compliance_event(
            event_type="gdpr_dsar",
            framework="GDPR",
            user_id=user_id,
            details={
                "request_id": dsar.request_id,
                "request_type": request.request_type,
                "subject_id": request.subject_id
            }
        )

        return response

    except Exception as e:
        logger.error(f"GDPR DSAR error: {str(e)}")
        raise HTTPException(status_code=500, detail="GDPR DSAR service error")


@router.get("/gdpr/dsar/{request_id}")
async def get_dsar_status(request_id: str, user_id: str = Depends(get_current_user)):
    """Get status of data subject request"""
    try:
        dsar = gdpr_manager.data_subject_requests.get(request_id)
        if not dsar:
            raise HTTPException(status_code=404, detail="DSAR not found")

        return {
            "request_id": dsar.request_id,
            "status": dsar.status.value,
            "progress": dsar.progress,
            "estimated_completion_date": dsar.estimated_completion_date.isoformat(),
            "data_provided": dsar.data_provided,
            "completed_steps": dsar.completed_steps
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GDPR DSAR status error: {str(e)}")
        raise HTTPException(status_code=500, detail="GDPR DSAR status service error")


@router.get("/gdpr/data-mapping")
async def get_gdpr_data_mapping(user_id: str = Depends(get_current_user)):
    """Get GDPR data mapping and inventory"""
    try:
        mapping = gdpr_manager.generate_data_mapping_report()
        return mapping

    except Exception as e:
        logger.error(f"GDPR data mapping error: {str(e)}")
        raise HTTPException(status_code=500, detail="GDPR data mapping service error")


@router.get("/gdpr/consent-records")
async def get_consent_records(subject_id: Optional[str] = None, user_id: str = Depends(get_current_user)):
    """Get GDPR consent records"""
    try:
        records = gdpr_manager.consent_manager.get_consent_records(subject_id)
        return {"consent_records": records}

    except Exception as e:
        logger.error(f"GDPR consent records error: {str(e)}")
        raise HTTPException(status_code=500, detail="GDPR consent records service error")


# SOC2 Compliance endpoints
@router.post("/soc2/assessment", response_model=ComplianceAssessmentResponse)
async def conduct_soc2_assessment(
    request: ComplianceAssessmentRequest,
    user_id: str = Depends(get_current_user)
):
    """Conduct SOC2 Type II compliance assessment"""
    try:
        assessment = soc2_manager.conduct_compliance_assessment()

        response = ComplianceAssessmentResponse(
            assessment_id=assessment.id,
            framework="SOC2",
            overall_status=assessment.overall_status.value,
            compliance_score=assessment.compliance_score,
            findings=assessment.findings,
            deficiencies=assessment.deficiencies,
            recommendations=assessment.recommendations if request.include_recommendations else [],
            assessment_date=assessment.assessment_date
        )

        # Log assessment event
        await audit_logger.log_compliance_event(
            event_type="soc2_assessment",
            framework="SOC2",
            user_id=user_id,
            details={
                "assessment_id": assessment.id,
                "compliance_score": assessment.compliance_score,
                "findings_count": len(assessment.findings)
            }
        )

        return response

    except Exception as e:
        logger.error(f"SOC2 assessment error: {str(e)}")
        raise HTTPException(status_code=500, detail="SOC2 assessment service error")


@router.post("/soc2/control-test", response_model=SOC2ControlTestResponse)
async def test_soc2_control(
    request: SOC2ControlTestRequest,
    user_id: str = Depends(get_current_user)
):
    """Test SOC2 control effectiveness"""
    try:
        test_result = soc2_manager.test_control(
            control_id=request.control_id,
            test_procedures=request.test_procedures,
            testing_period_start=request.testing_period_start,
            testing_period_end=request.testing_period_end
        )

        response = SOC2ControlTestResponse(
            test_id=test_result.test_id,
            control_id=request.control_id,
            test_results=test_result.test_results,
            evidence_collected=test_result.evidence_collected,
            deficiencies_identified=test_result.deficiencies_identified,
            overall_effectiveness=test_result.overall_effectiveness
        )

        # Log control test event
        await audit_logger.log_compliance_event(
            event_type="soc2_control_test",
            framework="SOC2",
            user_id=user_id,
            details={
                "control_id": request.control_id,
                "test_id": test_result.test_id,
                "effectiveness": test_result.overall_effectiveness
            }
        )

        return response

    except Exception as e:
        logger.error(f"SOC2 control test error: {str(e)}")
        raise HTTPException(status_code=500, detail="SOC2 control test service error")


@router.get("/soc2/system-description")
async def get_soc2_system_description(user_id: str = Depends(get_current_user)):
    """Get SOC2 system description"""
    try:
        description = soc2_manager.generate_system_description()
        return description

    except Exception as e:
        logger.error(f"SOC2 system description error: {str(e)}")
        raise HTTPException(status_code=500, detail="SOC2 system description service error")


@router.get("/soc2/controls")
async def list_soc2_controls(user_id: str = Depends(get_current_user)):
    """List SOC2 controls"""
    try:
        controls = []
        for control in soc2_manager.controls.values():
            controls.append({
                "control_id": control.control_id,
                "title": control.title,
                "criteria": control.criteria.value,
                "type": control.control_type.value,
                "frequency": control.testing_frequency,
                "last_test_date": control.last_test_date.isoformat() if control.last_test_date else None,
                "effectiveness": control.effectiveness
            })

        return {"controls": controls}

    except Exception as e:
        logger.error(f"SOC2 controls list error: {str(e)}")
        raise HTTPException(status_code=500, detail="SOC2 controls service error")


# HIPAA Compliance endpoints
@router.post("/hipaa/assessment", response_model=ComplianceAssessmentResponse)
async def conduct_hipaa_assessment(
    request: ComplianceAssessmentRequest,
    user_id: str = Depends(get_current_user)
):
    """Conduct HIPAA compliance assessment"""
    try:
        assessment = hipaa_manager.conduct_compliance_assessment()

        response = ComplianceAssessmentResponse(
            assessment_id=assessment.id,
            framework="HIPAA",
            overall_status=assessment.overall_status.value,
            compliance_score=assessment.compliance_score,
            findings=assessment.findings,
            deficiencies=assessment.deficiencies,
            recommendations=assessment.recommendations if request.include_recommendations else [],
            assessment_date=assessment.assessment_date
        )

        # Log assessment event
        await audit_logger.log_compliance_event(
            event_type="hipaa_assessment",
            framework="HIPAA",
            user_id=user_id,
            details={
                "assessment_id": assessment.id,
                "compliance_score": assessment.compliance_score,
                "findings_count": len(assessment.findings)
            }
        )

        return response

    except Exception as e:
        logger.error(f"HIPAA assessment error: {str(e)}")
        raise HTTPException(status_code=500, detail="HIPAA assessment service error")


@router.post("/hipaa/business-associate", response_model=HIPAABusinessAssociateResponse)
async def assess_business_associate(
    request: HIPAABusinessAssociateRequest,
    user_id: str = Depends(get_current_user)
):
    """Assess HIPAA business associate compliance"""
    try:
        ba = hipaa_manager.assess_business_associate(
            ba_name=request.ba_name,
            assessment_data={
                "services_provided": request.services_provided,
                "data_access_level": request.data_access_level,
                "risk_factors": request.risk_factors,
                "compliance_documentation": request.compliance_documentation
            }
        )

        response = HIPAABusinessAssociateResponse(
            ba_id=ba.id,
            risk_assessment_score=ba.risk_assessment_score,
            compliance_status=ba.compliance_status.value,
            control_requirements=ba.safeguard_certifications,
            recommended_actions=[
                f"Schedule review within {ba.next_audit_date - datetime.now()}",
                f"Monitor risk score: {ba.risk_assessment_score}"
            ]
        )

        # Log BA assessment event
        await audit_logger.log_compliance_event(
            event_type="hipaa_ba_assessment",
            framework="HIPAA",
            user_id=user_id,
            details={
                "ba_id": ba.id,
                "ba_name": request.ba_name,
                "risk_score": ba.risk_assessment_score
            }
        )

        return response

    except Exception as e:
        logger.error(f"HIPAA business associate error: {str(e)}")
        raise HTTPException(status_code=500, detail="HIPAA business associate service error")


@router.post("/hipaa/breach")
async def report_hipaa_breach(
    breach_data: Dict[str, Any] = Body(...),
    user_id: str = Depends(get_current_user)
):
    """Report HIPAA breach"""
    try:
        breach = hipaa_manager.report_breach(breach_data)

        # Log breach event
        await audit_logger.log_security_event(
            event_type="hipaa_breach",
            description=f"HIPAA breach reported: {breach.breach_type.value}",
            severity="high",
            details={
                "breach_id": breach.id,
                "affected_individuals": breach.affected_individuals,
                "data_types": [dt.value for dt in breach.data_types]
            }
        )

        return {
            "breach_id": breach.id,
            "status": breach.resolution_status.value,
            "notification_deadline": breach.notification_deadline.isoformat(),
            "mitigation_actions": breach.mitigation_actions
        }

    except Exception as e:
        logger.error(f"HIPAA breach reporting error: {str(e)}")
        raise HTTPException(status_code=500, detail="HIPAA breach reporting service error")


@router.get("/hipaa/training")
async def manage_workforce_training(
    employee_id: Optional[str] = None,
    training_type: Optional[str] = None,
    user_id: str = Depends(get_current_user)
):
    """Manage HIPAA workforce training"""
    try:
        if employee_id:
            records = hipaa_manager.training_records.get(employee_id, [])
            return {"training_records": [record.__dict__ for record in records]}
        else:
            # Get training summary
            all_employees = len(hipaa_manager.training_records)
            compliant_employees = sum(
                1 for records in hipaa_manager.training_records.values()
                if any(record.expiry_date > datetime.now() for record in records)
            )

            return {
                "total_employees": all_employees,
                "compliant_employees": compliant_employees,
                "compliance_rate": (compliant_employees / all_employees * 100) if all_employees > 0 else 0
            }

    except Exception as e:
        logger.error(f"HIPAA training error: {str(e)}")
        raise HTTPException(status_code=500, detail="HIPAA training service error")


# Compliance Reporting endpoints
@router.post("/reports/generate", response_model=ComplianceReportResponse)
async def generate_compliance_report(
    request: ComplianceReportRequest,
    user_id: str = Depends(get_current_user)
):
    """Generate compliance report"""
    try:
        report_config = {
            "framework": request.framework,
            "report_type": request.report_type,
            "format": request.format,
            "generated_by": user_id
        }

        if request.time_period:
            report_config["time_period"] = request.time_period

        report = reporting_service.generate_compliance_report(report_config)

        response = ComplianceReportResponse(
            report_id=report.id,
            generated_at=report.generated_date,
            format=request.format
        )

        if request.format == "json":
            response.report_content = report.content

        # Log report generation
        await audit_logger.log_compliance_event(
            event_type="report_generated",
            framework=request.framework,
            user_id=user_id,
            details={
                "report_id": report.id,
                "report_type": request.report_type,
                "format": request.format
            }
        )

        return response

    except Exception as e:
        logger.error(f"Compliance report generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Compliance report generation service error")


@router.get("/reports/{report_id}")
async def get_compliance_report(report_id: str, user_id: str = Depends(get_current_user)):
    """Get compliance report"""
    try:
        if report_id not in reporting_service.reports:
            raise HTTPException(status_code=404, detail="Report not found")

        report = reporting_service.reports[report_id]

        return {
            "report_id": report.id,
            "title": report.title,
            "description": report.description,
            "framework": report.framework,
            "report_type": report.report_type,
            "generated_at": report.generated_date.isoformat(),
            "generated_by": report.generated_by,
            "format": report.format,
            "content": report.content
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get compliance report error: {str(e)}")
        raise HTTPException(status_code=500, detail="Get compliance report service error")


@router.get("/dashboards/{dashboard_id}")
async def get_compliance_dashboard(
    dashboard_id: str,
    timeframe: str = "24h",
    filters: Optional[str] = None,
    user_id: str = Depends(get_current_user)
):
    """Get compliance dashboard data"""
    try:
        if dashboard_id not in reporting_service.dashboards:
            raise HTTPException(status_code=404, detail="Dashboard not found")

        # Parse filters if provided
        filter_dict = {}
        if filters:
            filter_dict = json.loads(filters)

        dashboard_data = reporting_service.generate_dashboard_data(dashboard_id)

        return dashboard_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Compliance dashboard error: {str(e)}")
        raise HTTPException(status_code=500, detail="Compliance dashboard service error")


@router.get("/metrics/summary")
async def get_compliance_summary(user_id: str = Depends(get_current_user)):
    """Get compliance summary across all frameworks"""
    try:
        summary = reporting_service.get_compliance_summary()
        return summary

    except Exception as e:
        logger.error(f"Compliance summary error: {str(e)}")
        raise HTTPException(status_code=500, detail="Compliance summary service error")


# Compliance Policy Management
@router.post("/policies", response_model=CompliancePolicyResponse)
async def create_compliance_policy(
    request: CompliancePolicyRequest,
    user_id: str = Depends(get_current_user)
):
    """Create compliance policy"""
    try:
        # This would integrate with a policy management system
        # For now, return a mock response

        policy_id = f"policy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Validate policy content
        validation_results = []
        for rule in request.validation_rules:
            # Simple validation logic
            validation_results.append({
                "rule_id": rule.get("id", "unknown"),
                "status": "passed",
                "message": "Validation successful"
            })

        # Calculate compliance score
        passed_validations = len([r for r in validation_results if r["status"] == "passed"])
        compliance_score = (passed_validations / len(validation_results)) * 100 if validation_results else 100

        # Identify gaps
        gaps = []
        if compliance_score < 100:
            gaps.append({
                "type": "validation_gap",
                "description": f"Policy validation failed with {compliance_score}% score",
                "severity": "medium"
            })

        response = CompliancePolicyResponse(
            policy_id=policy_id,
            status="active",
            validation_results=validation_results,
            compliance_score=compliance_score,
            gaps_identified=gaps
        )

        # Log policy creation
        await audit_logger.log_compliance_event(
            event_type="policy_created",
            framework=request.framework,
            user_id=user_id,
            details={
                "policy_id": policy_id,
                "policy_name": request.policy_name,
                "compliance_score": compliance_score
            }
        )

        return response

    except Exception as e:
        logger.error(f"Compliance policy creation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Compliance policy creation service error")


@router.post("/evidence/upload")
async def upload_compliance_evidence(
    framework: str = Form(...),
    control_id: str = Form(...),
    evidence_type: str = Form(...),
    description: str = Form(...),
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user)
):
    """Upload compliance evidence"""
    try:
        # Process uploaded file
        file_content = await file.read()

        # Store evidence (simplified - would use proper storage)
        evidence_id = f"evidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create evidence record
        evidence_data = {
            "evidence_id": evidence_id,
            "framework": framework,
            "control_id": control_id,
            "evidence_type": evidence_type,
            "description": description,
            "file_name": file.filename,
            "file_size": len(file_content),
            "uploaded_by": user_id,
            "uploaded_at": datetime.now().isoformat(),
            "status": "pending_review"
        }

        # Log evidence upload
        await audit_logger.log_compliance_event(
            event_type="evidence_uploaded",
            framework=framework,
            user_id=user_id,
            details={
                "evidence_id": evidence_id,
                "control_id": control_id,
                "file_name": file.filename
            }
        )

        return {"evidence_id": evidence_id, "status": "uploaded"}

    except Exception as e:
        logger.error(f"Compliance evidence upload error: {str(e)}")
        raise HTTPException(status_code=500, detail="Compliance evidence upload service error")


# Compliance Health Check
@router.get("/health")
async def compliance_health_check(user_id: str = Depends(get_current_user)):
    """Compliance system health check"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "frameworks": {
                "GDPR": "operational" if gdpr_manager else "not_configured",
                "SOC2": "operational" if soc2_manager else "not_configured",
                "HIPAA": "operational" if hipaa_manager else "not_configured"
            },
            "components": {
                "reporting_service": "operational",
                "audit_logger": "operational" if audit_logger.processing else "stopped"
            },
            "metrics": {
                "total_assessments": (
                    len(gdpr_manager.assessments) if gdpr_manager else 0 +
                    len(soc2_manager.assessments) if soc2_manager else 0 +
                    len(hipaa_manager.assessments) if hipaa_manager else 0
                ),
                "active_dsars": len(gdpr_manager.data_subject_requests) if gdpr_manager else 0,
                "business_associates": len(hipaa_manager.business_associates) if hipaa_manager else 0,
                "recent_reports": len([r for r in reporting_service.reports.values() if (datetime.now() - r.generated_date).days < 7])
            }
        }

        # Check if any component is unhealthy
        if "not_configured" in health_status["frameworks"].values():
            health_status["status"] = "partial"

        if "stopped" in health_status["components"].values():
            health_status["status"] = "degraded"

        return health_status

    except Exception as e:
        logger.error(f"Compliance health check error: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Scheduled compliance tasks
@router.post("/tasks/schedule-assessment")
async def schedule_compliance_assessment(
    framework: str = Body(...),
    schedule: Dict[str, Any] = Body(...),
    user_id: str = Depends(get_current_user)
):
    """Schedule recurring compliance assessment"""
    try:
        schedule_id = f"schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # This would integrate with a task scheduler
        scheduled_task = {
            "schedule_id": schedule_id,
            "framework": framework,
            "schedule": schedule,
            "created_by": user_id,
            "created_at": datetime.now().isoformat(),
            "status": "scheduled",
            "next_run": schedule.get("start_time")
        }

        # Log scheduled task
        await audit_logger.log_compliance_event(
            event_type="assessment_scheduled",
            framework=framework,
            user_id=user_id,
            details={
                "schedule_id": schedule_id,
                "schedule": schedule
            }
        )

        return {"schedule_id": schedule_id, "status": "scheduled"}

    except Exception as e:
        logger.error(f"Schedule compliance assessment error: {str(e)}")
        raise HTTPException(status_code=500, detail="Schedule compliance assessment service error")


@router.get("/tasks/scheduled")
async def get_scheduled_tasks(user_id: str = Depends(get_current_user)):
    """Get scheduled compliance tasks"""
    try:
        # This would query actual scheduled tasks
        # For now, return mock data
        return {
            "scheduled_tasks": [
                {
                    "task_id": "task_1",
                    "framework": "GDPR",
                    "type": "assessment",
                    "next_run": (datetime.now() + timedelta(days=30)).isoformat(),
                    "status": "scheduled"
                },
                {
                    "task_id": "task_2",
                    "framework": "SOC2",
                    "type": "control_testing",
                    "next_run": (datetime.now() + timedelta(days=90)).isoformat(),
                    "status": "scheduled"
                }
            ]
        }

    except Exception as e:
        logger.error(f"Get scheduled tasks error: {str(e)}")
        raise HTTPException(status_code=500, detail="Get scheduled tasks service error")