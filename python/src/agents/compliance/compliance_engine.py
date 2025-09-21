"""
Compliance Engine Core
Centralized compliance monitoring and enforcement system
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import yaml
from pathlib import Path
import hashlib
import re

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    GDPR = "gdpr"
    SOC2 = "soc2"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"
    NIST_800_53 = "nist_800_53"
    CCPA = "ccpa"
    FERPA = "ferpa"
    GLBA = "glba"


class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_ASSESSED = "not_assessed"
    EXEMPT = "exempt"


class RiskLevel(Enum):
    """Risk levels for compliance findings"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ControlType(Enum):
    """Types of compliance controls"""
    TECHNICAL = "technical"
    ADMINISTRATIVE = "administrative"
    PHYSICAL = "physical"


@dataclass
class ComplianceControl:
    """Individual compliance control"""
    control_id: str
    framework: ComplianceFramework
    name: str
    description: str
    control_type: ControlType
    requirement: str
    implementation_guidance: str
    evidence_requirements: List[str]
    testing_procedures: List[str]
    frequency: str  # daily, weekly, monthly, quarterly, annually
    owner: str
    status: ComplianceStatus = ComplianceStatus.NOT_ASSESSED
    last_assessed: Optional[datetime] = None
    next_assessment: Optional[datetime] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "control_id": self.control_id,
            "framework": self.framework.value,
            "name": self.name,
            "description": self.description,
            "control_type": self.control_type.value,
            "requirement": self.requirement,
            "implementation_guidance": self.implementation_guidance,
            "evidence_requirements": self.evidence_requirements,
            "testing_procedures": self.testing_procedures,
            "frequency": self.frequency,
            "owner": self.owner,
            "status": self.status.value,
            "last_assessed": self.last_assessed.isoformat() if self.last_assessed else None,
            "next_assessment": self.next_assessment.isoformat() if self.next_assessment else None,
            "evidence": self.evidence,
            "findings": self.findings,
            "metadata": self.metadata
        }


@dataclass
class ComplianceFinding:
    """Compliance finding or violation"""
    finding_id: str
    control_id: str
    framework: ComplianceFramework
    title: str
    description: str
    risk_level: RiskLevel
    status: ComplianceStatus
    finding_type: str  # violation, observation, opportunity_for_improvement
    identified_date: datetime
    due_date: Optional[datetime] = None
    remediation_plan: Optional[str] = None
    remediation_status: str = "open"  # open, in_progress, resolved, deferred
    assigned_to: Optional[str] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    impact: str = ""
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "control_id": self.control_id,
            "framework": self.framework.value,
            "title": self.title,
            "description": self.description,
            "risk_level": self.risk_level.value,
            "status": self.status.value,
            "finding_type": self.finding_type,
            "identified_date": self.identified_date.isoformat(),
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "remediation_plan": self.remediation_plan,
            "remediation_status": self.remediation_status,
            "assigned_to": self.assigned_to,
            "evidence": self.evidence,
            "impact": self.impact,
            "recommendations": self.recommendations,
            "metadata": self.metadata
        }


@dataclass
class ComplianceAssessment:
    """Compliance assessment results"""
    assessment_id: str
    framework: ComplianceFramework
    scope: str
    assessment_date: datetime
    assessor: str
    overall_status: ComplianceStatus
    controls_assessed: int
    controls_compliant: int
    controls_non_compliant: int
    controls_partially_compliant: int
    findings: List[ComplianceFinding] = field(default_factory=list)
    score: float = 0.0  # 0-100
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "framework": self.framework.value,
            "scope": self.scope,
            "assessment_date": self.assessment_date.isoformat(),
            "assessor": self.assessor,
            "overall_status": self.overall_status.value,
            "controls_assessed": self.controls_assessed,
            "controls_compliant": self.controls_compliant,
            "controls_non_compliant": self.controls_non_compliant,
            "controls_partially_compliant": self.controls_partially_compliant,
            "findings": [finding.to_dict() for finding in self.findings],
            "score": self.score,
            "recommendations": self.recommendations,
            "metadata": self.metadata
        }


@dataclass
class CompliancePolicy:
    """Compliance policy definition"""
    policy_id: str
    framework: ComplianceFramework
    name: str
    description: str
    category: str  # data_protection, access_control, security, etc.
    requirements: List[str]
    controls: List[str]  # Control IDs
    implementation_date: Optional[datetime] = None
    review_frequency: str = "annually"
    last_reviewed: Optional[datetime] = None
    next_review: Optional[datetime] = None
    version: str = "1.0"
    status: str = "active"  # active, draft, retired
    owner: str = ""
    approvers: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "framework": self.framework.value,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "requirements": self.requirements,
            "controls": self.controls,
            "implementation_date": self.implementation_date.isoformat() if self.implementation_date else None,
            "review_frequency": self.review_frequency,
            "last_reviewed": self.last_reviewed.isoformat() if self.last_reviewed else None,
            "next_review": self.next_review.isoformat() if self.next_review else None,
            "version": self.version,
            "status": self.status,
            "owner": self.owner,
            "approvers": self.approvers,
            "metadata": self.metadata
        }


class ComplianceEngine:
    """Central compliance management engine"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.controls: Dict[str, ComplianceControl] = {}
        self.findings: Dict[str, ComplianceFinding] = {}
        self.assessments: Dict[str, ComplianceAssessment] = {}
        self.policies: Dict[str, CompliancePolicy] = {}
        self.framework_managers: Dict[ComplianceFramework, Any] = {}

        # Configuration
        self.auto_assessment_enabled = self.config.get("auto_assessment", True)
        self.alert_threshold = self.config.get("alert_threshold", 0.8)  # Alert if score below 80%
        self.remediation_deadline_days = self.config.get("remediation_deadline_days", 30)

        # Initialize with standard controls
        self._initialize_standard_controls()

    def _initialize_standard_controls(self) -> None:
        """Initialize standard compliance controls"""
        # This would typically load from a database or configuration files
        # For now, we'll create some essential controls
        standard_controls = [
            # Security Controls
            ComplianceControl(
                control_id="SEC-001",
                framework=ComplianceFramework.SOC2,
                name="Access Control Management",
                description="Implement comprehensive access control policies and procedures",
                control_type=ControlType.TECHNICAL,
                requirement="CC6.1 - Logical access controls for information systems",
                implementation_guidance="Implement role-based access control with regular reviews",
                evidence_requirements=["Access control policies", "User access reviews", "Access logs"],
                testing_procedures=["Review access control policies", "Test user access provisioning/deprovisioning"],
                frequency="quarterly",
                owner="Security Team"
            ),
            ComplianceControl(
                control_id="SEC-002",
                framework=ComplianceFramework.SOC2,
                name="Change Management",
                description="Implement formal change management processes",
                control_type=ControlType.ADMINISTRATIVE,
                requirement="CC6.7 - Change management procedures",
                implementation_guidance="Document and track all system changes",
                evidence_requirements=["Change management procedures", "Change requests", "Approval records"],
                testing_procedures=["Review change management logs", "Verify change approvals"],
                frequency="quarterly",
                owner="IT Operations"
            ),
            # GDPR Controls
            ComplianceControl(
                control_id="GDPR-001",
                framework=ComplianceFramework.GDPR,
                name="Lawful Basis for Processing",
                description="Establish lawful basis for all personal data processing",
                control_type=ControlType.ADMINISTRATIVE,
                requirement="Article 6 - Lawfulness of processing",
                implementation_guidance="Document lawful basis for each data processing activity",
                evidence_requirements=["Data processing records", "Privacy policies", "Consent records"],
                testing_procedures=["Review data processing documentation", "Verify consent mechanisms"],
                frequency="annually",
                owner="Data Protection Officer"
            ),
            ComplianceControl(
                control_id="GDPR-002",
                framework=ComplianceFramework.GDPR,
                name="Data Subject Rights",
                description="Implement procedures to handle data subject requests",
                control_type=ControlType.ADMINISTRATIVE,
                requirement="Articles 12-15 - Data subject rights",
                implementation_guidance="Establish processes for DSAR handling within 30 days",
                evidence_requirements=["DSAR procedures", "Response templates", "Handling logs"],
                testing_procedures=["Test DSAR process", "Review response times"],
                frequency="quarterly",
                owner="Privacy Team"
            ),
            # HIPAA Controls
            ComplianceControl(
                control_id="HIPAA-001",
                framework=ComplianceFramework.HIPAA,
                name="Safeguards for PHI",
                description="Implement administrative, physical, and technical safeguards",
                control_type=ControlType.TECHNICAL,
                requirement="§164.306 - Security standards: General rules",
                implementation_guidance="Implement comprehensive security controls for PHI",
                evidence_requirements=["Security policies", "Risk assessment", "Audit logs"],
                testing_procedures=["Review security controls", "Test access controls", "Verify encryption"],
                frequency="annually",
                owner="Security Officer"
            ),
        ]

        for control in standard_controls:
            self.controls[control.control_id] = control

    def register_framework_manager(self, framework: ComplianceFramework, manager: Any) -> None:
        """Register a framework-specific manager"""
        self.framework_managers[framework] = manager
        logger.info(f"Registered framework manager for {framework.value}")

    async def assess_compliance(self, framework: ComplianceFramework,
                              scope: str = "full",
                              assessor: str = "system") -> ComplianceAssessment:
        """Perform compliance assessment for a framework"""
        assessment_id = str(uuid.uuid4())
        assessment_date = datetime.now()

        # Get controls for the framework
        framework_controls = [
            control for control in self.controls.values()
            if control.framework == framework
        ]

        if not framework_controls:
            logger.warning(f"No controls found for framework {framework.value}")
            return ComplianceAssessment(
                assessment_id=assessment_id,
                framework=framework,
                scope=scope,
                assessment_date=assessment_date,
                assessor=assessor,
                overall_status=ComplianceStatus.NOT_ASSESSED,
                controls_assessed=0,
                controls_compliant=0,
                controls_non_compliant=0,
                controls_partially_compliant=0
            )

        # Assess each control
        findings = []
        compliant_count = 0
        non_compliant_count = 0
        partially_compliant_count = 0

        for control in framework_controls:
            control_result = await self._assess_control(control)
            findings.extend(control_result.findings)

            if control_result.status == ComplianceStatus.COMPLIANT:
                compliant_count += 1
            elif control_result.status == ComplianceStatus.NON_COMPLIANT:
                non_compliant_count += 1
            elif control_result.status == ComplianceStatus.PARTIALLY_COMPLIANT:
                partially_compliant_count += 1

            # Update control status
            control.status = control_result.status
            control.last_assessed = assessment_date
            control.evidence.extend(control_result.evidence)

        # Calculate overall status and score
        total_controls = len(framework_controls)
        score = (compliant_count / total_controls) * 100 if total_controls > 0 else 0

        overall_status = self._determine_overall_status(
            compliant_count, non_compliant_count, partially_compliant_count, total_controls
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(framework, findings)

        assessment = ComplianceAssessment(
            assessment_id=assessment_id,
            framework=framework,
            scope=scope,
            assessment_date=assessment_date,
            assessor=assessor,
            overall_status=overall_status,
            controls_assessed=total_controls,
            controls_compliant=compliant_count,
            controls_non_compliant=non_compliant_count,
            controls_partially_compliant=partially_compliant,
            findings=findings,
            score=score,
            recommendations=recommendations
        )

        # Store assessment
        self.assessments[assessment_id] = assessment

        # Log assessment results
        logger.info(f"Compliance assessment completed for {framework.value}: "
                   f"Score={score:.1f}%, Status={overall_status.value}")

        # Check if alert threshold is breached
        if score < (self.alert_threshold * 100):
            await self._send_compliance_alert(assessment)

        return assessment

    async def _assess_control(self, control: ComplianceControl) -> ComplianceControl:
        """Assess an individual compliance control"""
        # In a real implementation, this would perform actual testing
        # For now, we'll simulate the assessment
        try:
            # Use framework manager if available
            if control.framework in self.framework_managers:
                return await self.framework_managers[control.framework].assess_control(control)
            else:
                return await self._simulate_control_assessment(control)
        except Exception as e:
            logger.error(f"Error assessing control {control.control_id}: {str(e)}")
            # Create a finding for the assessment error
            finding = ComplianceFinding(
                finding_id=str(uuid.uuid4()),
                control_id=control.control_id,
                framework=control.framework,
                title=f"Assessment Error for {control.name}",
                description=f"Error occurred during compliance assessment: {str(e)}",
                risk_level=RiskLevel.HIGH,
                status=ComplianceStatus.NON_COMPLIANT,
                finding_type="violation",
                identified_date=datetime.now(),
                recommendations=["Retry assessment", "Review assessment configuration"]
            )
            control.findings.append(finding.to_dict())
            control.status = ComplianceStatus.NON_COMPLIANT
            return control

    async def _simulate_control_assessment(self, control: ComplianceControl) -> ComplianceControl:
        """Simulate control assessment for demonstration"""
        # Simulate different outcomes based on control type
        import random

        # Higher chance of compliance for security controls
        if control.control_type == ControlType.TECHNICAL:
            compliance_probability = 0.8
        else:
            compliance_probability = 0.6

        if random.random() < compliance_probability:
            control.status = ComplianceStatus.COMPLIANT
            evidence = [{
                "type": "automated_check",
                "result": "passed",
                "timestamp": datetime.now().isoformat(),
                "details": f"Automated compliance check passed for {control.name}"
            }]
        else:
            control.status = ComplianceStatus.NON_COMPLIANT
            finding = ComplianceFinding(
                finding_id=str(uuid.uuid4()),
                control_id=control.control_id,
                framework=control.framework,
                title=f"Compliance Gap: {control.name}",
                description=f"Control assessment identified compliance gaps",
                risk_level=RiskLevel.MEDIUM,
                status=ComplianceStatus.NON_COMPLIANT,
                finding_type="violation",
                identified_date=datetime.now(),
                due_date=datetime.now() + timedelta(days=self.remediation_deadline_days),
                recommendations=[
                    "Review control implementation",
                    "Update policies and procedures",
                    "Implement required controls"
                ]
            )
            control.findings.append(finding.to_dict())
            evidence = [{
                "type": "automated_check",
                "result": "failed",
                "timestamp": datetime.now().isoformat(),
                "details": f"Automated compliance check failed for {control.name}"
            }]

        control.evidence.extend(evidence)
        return control

    def _determine_overall_status(self, compliant: int, non_compliant: int,
                                partially_compliant: int, total: int) -> ComplianceStatus:
        """Determine overall compliance status"""
        if total == 0:
            return ComplianceStatus.NOT_ASSESSED

        compliant_percentage = (compliant / total) * 100

        if compliant_percentage >= 95:
            return ComplianceStatus.COMPLIANT
        elif compliant_percentage >= 80:
            return ComplianceStatus.PARTIALLY_COMPLIANT
        elif non_compliant == 0:
            return ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            return ComplianceStatus.NON_COMPLIANT

    def _generate_recommendations(self, framework: ComplianceFramework,
                                findings: List[ComplianceFinding]) -> List[str]:
        """Generate compliance recommendations based on findings"""
        recommendations = []

        # Group findings by risk level
        critical_findings = [f for f in findings if f.risk_level == RiskLevel.CRITICAL]
        high_findings = [f for f in findings if f.risk_level == RiskLevel.HIGH]

        if critical_findings:
            recommendations.append(f"Address {len(critical_findings)} critical findings immediately")
            recommendations.append("Implement emergency remediation procedures")

        if high_findings:
            recommendations.append(f"Prioritize {len(high_findings)} high-risk findings")
            recommendations.append("Develop detailed remediation plans")

        # Framework-specific recommendations
        if framework == ComplianceFramework.GDPR:
            recommendations.append("Review data processing activities and lawful basis documentation")
            recommendations.append("Update data subject rights handling procedures")
        elif framework == ComplianceFramework.SOC2:
            recommendations.append("Enhance control documentation and evidence collection")
            recommendations.append("Strengthen change management processes")
        elif framework == ComplianceFramework.HIPAA:
            recommendations.append("Review PHI safeguard implementations")
            recommendations.append("Update risk assessment documentation")

        return recommendations

    async def _send_compliance_alert(self, assessment: ComplianceAssessment) -> None:
        """Send compliance alert for low scores"""
        alert_data = {
            "type": "compliance_alert",
            "framework": assessment.framework.value,
            "score": assessment.score,
            "status": assessment.overall_status.value,
            "findings_count": len(assessment.findings),
            "timestamp": datetime.now().isoformat(),
            "message": f"Compliance score below threshold: {assessment.score:.1f}%"
        }

        # In a real implementation, this would send alerts via email, Slack, etc.
        logger.warning(f"COMPLIANCE ALERT: {alert_data}")

    def create_finding(self, control_id: str, title: str, description: str,
                      risk_level: RiskLevel, finding_type: str = "violation",
                      assigned_to: str = None) -> ComplianceFinding:
        """Create a new compliance finding"""
        control = self.controls.get(control_id)
        if not control:
            raise ValueError(f"Control {control_id} not found")

        finding = ComplianceFinding(
            finding_id=str(uuid.uuid4()),
            control_id=control_id,
            framework=control.framework,
            title=title,
            description=description,
            risk_level=risk_level,
            status=ComplianceStatus.NON_COMPLIANT,
            finding_type=finding_type,
            identified_date=datetime.now(),
            due_date=datetime.now() + timedelta(days=self.remediation_deadline_days),
            assigned_to=assigned_to or control.owner
        )

        self.findings[finding.finding_id] = finding
        control.findings.append(finding.to_dict())

        logger.info(f"Created compliance finding: {finding.title}")
        return finding

    def update_finding(self, finding_id: str, **kwargs) -> bool:
        """Update a compliance finding"""
        finding = self.findings.get(finding_id)
        if not finding:
            return False

        for key, value in kwargs.items():
            if hasattr(finding, key):
                setattr(finding, key, value)

        # Update control findings list
        control = self.controls.get(finding.control_id)
        if control:
            # Remove old finding data
            control.findings = [f for f in control.findings if f.get("finding_id") != finding_id]
            # Add updated finding data
            control.findings.append(finding.to_dict())

        logger.info(f"Updated compliance finding: {finding.title}")
        return True

    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get compliance dashboard data"""
        dashboard_data = {
            "overall_status": self._calculate_overall_status(),
            "frameworks": {},
            "recent_findings": [],
            "upcoming_assessments": [],
            "risk_summary": {}
        }

        # Framework status
        for framework in ComplianceFramework:
            framework_controls = [c for c in self.controls.values() if c.framework == framework]
            if framework_controls:
                compliant = len([c for c in framework_controls if c.status == ComplianceStatus.COMPLIANT])
                total = len(framework_controls)
                score = (compliant / total) * 100 if total > 0 else 0

                dashboard_data["frameworks"][framework.value] = {
                    "total_controls": total,
                    "compliant_controls": compliant,
                    "score": round(score, 1),
                    "status": self._determine_overall_status(compliant, 0, 0, total).value
                }

        # Recent findings
        recent_findings = sorted(
            self.findings.values(),
            key=lambda f: f.identified_date,
            reverse=True
        )[:10]

        dashboard_data["recent_findings"] = [
            {
                "finding_id": f.finding_id,
                "title": f.title,
                "risk_level": f.risk_level.value,
                "framework": f.framework.value,
                "identified_date": f.identified_date.isoformat(),
                "remediation_status": f.remediation_status
            }
            for f in recent_findings
        ]

        # Risk summary
        risk_counts = {}
        for finding in self.findings.values():
            level = finding.risk_level.value
            risk_counts[level] = risk_counts.get(level, 0) + 1

        dashboard_data["risk_summary"] = risk_counts

        # Upcoming assessments
        upcoming_assessments = []
        for control in self.controls.values():
            if control.next_assessment and control.next_assessment > datetime.now():
                upcoming_assessments.append({
                    "control_id": control.control_id,
                    "control_name": control.name,
                    "framework": control.framework.value,
                    "next_assessment": control.next_assessment.isoformat()
                })

        dashboard_data["upcoming_assessments"] = sorted(
            upcoming_assessments,
            key=lambda x: x["next_assessment"]
        )[:10]

        return dashboard_data

    def _calculate_overall_status(self) -> Dict[str, Any]:
        """Calculate overall compliance status across all frameworks"""
        all_controls = list(self.controls.values())
        if not all_controls:
            return {"status": "not_assessed", "score": 0.0}

        compliant = len([c for c in all_controls if c.status == ComplianceStatus.COMPLIANT])
        total = len(all_controls)
        score = (compliant / total) * 100 if total > 0 else 0

        return {
            "status": self._determine_overall_status(compliant, 0, 0, total).value,
            "score": round(score, 1),
            "total_controls": total,
            "compliant_controls": compliant
        }

    async def generate_compliance_report(self, framework: ComplianceFramework,
                                      report_type: str = "detailed") -> Dict[str, Any]:
        """Generate compliance report"""
        # Get latest assessment for framework
        framework_assessments = [
            a for a in self.assessments.values()
            if a.framework == framework
        ]

        if not framework_assessments:
            return {"error": "No assessments found for framework"}

        latest_assessment = max(framework_assessments, key=lambda a: a.assessment_date)

        report = {
            "report_id": str(uuid.uuid4()),
            "framework": framework.value,
            "report_type": report_type,
            "generated_at": datetime.now().isoformat(),
            "assessment": latest_assessment.to_dict(),
            "trend_analysis": self._analyze_compliance_trends(framework),
            "recommendations": latest_assessment.recommendations
        }

        if report_type == "detailed":
            report["control_details"] = [
                control.to_dict()
                for control in self.controls.values()
                if control.framework == framework
            ]

        return report

    def _analyze_compliance_trends(self, framework: ComplianceFramework) -> Dict[str, Any]:
        """Analyze compliance trends over time"""
        framework_assessments = [
            a for a in self.assessments.values()
            if a.framework == framework
        ]

        if len(framework_assessments) < 2:
            return {"insufficient_data": True}

        # Sort by date
        sorted_assessments = sorted(framework_assessments, key=lambda a: a.assessment_date)

        trends = {
            "score_trend": [],
            "compliant_controls_trend": [],
            "non_compliant_controls_trend": []
        }

        for assessment in sorted_assessments:
            trends["score_trend"].append({
                "date": assessment.assessment_date.isoformat(),
                "score": assessment.score
            })
            trends["compliant_controls_trend"].append({
                "date": assessment.assessment_date.isoformat(),
                "count": assessment.controls_compliant
            })
            trends["non_compliant_controls_trend"].append({
                "date": assessment.assessment_date.isoformat(),
                "count": assessment.controls_non_compliant
            })

        return trends

    def schedule_assessments(self) -> None:
        """Schedule recurring compliance assessments"""
        now = datetime.now()

        for control in self.controls.values():
            if not control.next_assessment or control.next_assessment <= now:
                # Schedule next assessment based on frequency
                frequency_map = {
                    "daily": timedelta(days=1),
                    "weekly": timedelta(weeks=1),
                    "monthly": timedelta(days=30),
                    "quarterly": timedelta(days=90),
                    "annually": timedelta(days=365)
                }

                frequency = frequency_map.get(control.frequency, timedelta(days=30))
                control.next_assessment = now + frequency

    async def run_scheduled_assessments(self) -> List[ComplianceAssessment]:
        """Run all scheduled compliance assessments"""
        self.schedule_assessments()

        now = datetime.now()
        scheduled_controls = [
            control for control in self.controls.values()
            if control.next_assessment and control.next_assessment <= now
        ]

        if not scheduled_controls:
            logger.info("No scheduled assessments to run")
            return []

        logger.info(f"Running {len(scheduled_controls)} scheduled assessments")

        # Group by framework
        frameworks_to_assess = set(control.framework for control in scheduled_controls)
        assessments = []

        for framework in frameworks_to_assess:
            assessment = await self.assess_compliance(framework)
            assessments.append(assessment)

        return assessments

    def export_compliance_data(self, format: str = "json") -> Any:
        """Export compliance data"""
        data = {
            "controls": [control.to_dict() for control in self.controls.values()],
            "findings": [finding.to_dict() for finding in self.findings.values()],
            "assessments": [assessment.to_dict() for assessment in self.assessments.values()],
            "policies": [policy.to_dict() for policy in self.policies.values()],
            "exported_at": datetime.now().isoformat()
        }

        if format == "json":
            return json.dumps(data, indent=2)
        elif format == "yaml":
            return yaml.dump(data, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def import_compliance_data(self, data: Union[str, Dict], format: str = "json") -> None:
        """Import compliance data"""
        if format == "json":
            if isinstance(data, str):
                data = json.loads(data)
        elif format == "yaml":
            if isinstance(data, str):
                data = yaml.safe_load(data)
        else:
            raise ValueError(f"Unsupported import format: {format}")

        # Import controls
        for control_data in data.get("controls", []):
            control = ComplianceControl(
                control_id=control_data["control_id"],
                framework=ComplianceFramework(control_data["framework"]),
                name=control_data["name"],
                description=control_data["description"],
                control_type=ControlType(control_data["control_type"]),
                requirement=control_data["requirement"],
                implementation_guidance=control_data["implementation_guidance"],
                evidence_requirements=control_data["evidence_requirements"],
                testing_procedures=control_data["testing_procedures"],
                frequency=control_data["frequency"],
                owner=control_data["owner"],
                status=ComplianceStatus(control_data["status"]),
                last_assessed=datetime.fromisoformat(control_data["last_assessment"]) if control_data["last_assessment"] else None,
                next_assessment=datetime.fromisoformat(control_data["next_assessment"]) if control_data["next_assessment"] else None,
                evidence=control_data.get("evidence", []),
                findings=control_data.get("findings", []),
                metadata=control_data.get("metadata", {})
            )
            self.controls[control.control_id] = control

        # Import other data types as needed
        logger.info("Compliance data imported successfully")