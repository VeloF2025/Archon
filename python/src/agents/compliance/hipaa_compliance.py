"""
HIPAA Compliance Management System
Comprehensive HIPAA compliance management for healthcare data
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass, field
import hashlib
import uuid

from .compliance_engine import (
    ComplianceFramework,
    ComplianceControl,
    ComplianceStatus,
    RiskLevel,
    ComplianceAssessment,
    )

logger = logging.getLogger(__name__)


@dataclass
class ComplianceEvidence:
    """Evidence of compliance control implementation"""
    id: str
    control_id: str
    evidence_type: str
    evidence_content: str
    collection_date: datetime
    collected_by: str = "system"
    verified: bool = False


class HIPAASection(Enum):
    """HIPAA Privacy Rule sections"""
    PRIVACY_RULE = "privacy_rule"
    SECURITY_RULE = "security_rule"
    BREACH_NOTIFICATION = "breach_notification"
    ENFORCEMENT_RULE = "enforcement_rule"
    HITECH_ACT = "hitech_act"
    OMNIBUS_RULE = "omnibus_rule"


class PHIDataType(Enum):
    """Protected Health Information data types"""
    DEMOGRAPHIC = "demographic"
    MEDICAL_RECORD = "medical_record"
    BILLING = "billing"
    INSURANCE = "insurance"
    TREATMENT = "treatment"
    PAYMENT = "payment"
    HEALTHCARE_OPERATIONS = "healthcare_operations"
    IDENTIFIERS = "identifiers"


class HIPAAViolationType(Enum):
    """Types of HIPAA violations"""
    UNAUTHORIZED_DISCLOSURE = "unauthorized_disclosure"
    INSUFFICIENT_SAFEGUARDS = "insufficient_safeguards"
    IMPROPER_ACCESS = "improper_access"
    MISSING_BREACH_NOTIFICATION = "missing_breach_notification"
    IMPROPER_DISPOSAL = "imper_disposal"
    LACK_OF_TRAINING = "lack_of_training"
    BA_AGREEMENT_VIOLATION = "ba_agreement_violation"
    MINIMUM_NECESSARY_VIOLATION = "minimum_necessary_violation"


@dataclass
class PHIDeclassification:
    """PHI declassification and anonymization"""
    data_type: PHIDataType
    original_value: str
    deidentified_value: str
    deidentification_method: str
    verification_result: bool
    deidentified_date: datetime
    auditor_id: str
    compliance_score: float


@dataclass
class BusinessAssociate:
    """Business Associate management"""
    id: str
    name: str
    contract_date: datetime
    contract_expiry: datetime
    services_provided: List[str]
    data_access_level: str
    compliance_status: ComplianceStatus
    risk_assessment_score: float
    last_audit_date: datetime
    next_audit_date: datetime
    breach_history: List[Dict[str, Any]] = field(default_factory=list)
    safeguard_certifications: List[str] = field(default_factory=list)


@dataclass
class HIPAABreach:
    """HIPAA breach notification management"""
    id: str
    breach_type: HIPAAViolationType
    discovery_date: datetime
    breach_date: datetime
    affected_individuals: int
    data_types: List[PHIDataType]
    breach_description: str
    cause: str
    mitigating_actions: List[str]
    notification_deadline: datetime
    notifications_sent: bool
    oic_reported: bool
    individuals_notified: bool
    media_notified: bool
    resolution_status: ComplianceStatus
    remediation_plan: str
    lessons_learned: str


@dataclass
class HIPAAControl(ComplianceControl):
    """HIPAA-specific compliance control"""
    hipaa_section: Optional[HIPAASection] = None
    covered_entity_responsibility: str = ""
    business_associate_responsibility: str = ""
    safeguard_type: str = ""  # administrative, technical, physical
    data_protection_scope: Optional[List[PHIDataType]] = None
    implementation_specifications: Optional[List[str]] = None
    audit_procedures: Optional[List[str]] = None
    minimum_required: bool = True
    addressable: bool = False


@dataclass
class HIPAATraining:
    """HIPAA workforce training management"""
    id: str
    employee_id: str
    training_type: str  # privacy, security, breach notification
    training_date: datetime
    completion_date: datetime
    training_content: str
    trainer_id: str
    score: float
    certificate_id: str
    expiry_date: datetime
    refresher_required: bool
    topics_covered: List[str]
    assessment_questions: List[Dict[str, Any]]


class HIPAAComplianceManager:
    """HIPAA compliance management system"""

    def __init__(self, covered_entity_name: str):
        self.covered_entity_name = covered_entity_name
        self.controls: Dict[str, HIPAAControl] = {}
        self.business_associates: Dict[str, BusinessAssociate] = {}
        self.breaches: Dict[str, HIPAABreach] = {}
        self.training_records: Dict[str, List[HIPAATraining]] = {}
        self.phi_classifications: Dict[str, PHIDeclassification] = {}
        self.risk_assessments: Dict[str, Any] = {}

        self._initialize_hipaa_controls()

    def _initialize_hipaa_controls(self):
        """Initialize HIPAA compliance controls"""

        # Privacy Rule Controls
        self.controls["PR-001"] = HIPAAControl(
            id="PR-001",
            title="Notice of Privacy Practices",
            description="Provide notice of privacy practices to individuals",
            hipaa_section=HIPAASection.PRIVACY_RULE,
            covered_entity_responsibility="Develop and distribute notice",
            business_associate_responsibility="Follow CE privacy practices",
            safeguard_type="administrative",
            data_protection_scope=[PHIDataType.IDENTIFIERS, PHIDataType.DEMOGRAPHIC],
            implementation_specifications=[
                "Written notice available to individuals",
                "Notice posted in prominent locations",
                "Electronic notice available",
                "Notice provided at first service delivery"
            ],
            audit_procedures=[
                "Review notice content and availability",
                "Verify distribution records",
                "Check posting locations",
                "Assess electronic access"
            ],
            status=ComplianceStatus.COMPLIANT,
            risk_level=RiskLevel.LOW
        )

        self.controls["PR-002"] = HIPAAControl(
            id="PR-002",
            title="Authorization for Uses and Disclosures",
            description="Obtain valid authorization for uses and disclosures not otherwise permitted",
            hipaa_section=HIPAASection.PRIVACY_RULE,
            covered_entity_responsibility="Implement authorization process",
            business_associate_responsibility="Follow authorization requirements",
            safeguard_type="administrative",
            data_protection_scope=list(PHIDataType),
            implementation_specifications=[
                "Authorization request form",
                "Valid authorization elements",
                "Documentation and retention",
                "Revocation process"
            ],
            audit_procedures=[
                "Review authorization forms",
                "Check documentation completeness",
                "Verify revocation process",
                "Assess retention practices"
            ],
            status=ComplianceStatus.COMPLIANT,
            risk_level=RiskLevel.MEDIUM
        )

        # Security Rule Controls
        self.controls["SR-001"] = HIPAAControl(
            id="SR-001",
            title="Risk Analysis",
            description="Conduct accurate and thorough risk analysis",
            hipaa_section=HIPAASection.SECURITY_RULE,
            covered_entity_responsibility="Perform risk analysis",
            business_associate_responsibility="Conduct own risk analysis",
            safeguard_type="administrative",
            data_protection_scope=list(PHIDataType),
            implementation_specifications=[
                "Identify e-PHI risks and vulnerabilities",
                "Assess current security measures",
                "Document risk analysis process",
                "Update analysis as needed"
            ],
            audit_procedures=[
                "Review risk analysis documentation",
                "Verify risk identification process",
                "Assess risk assessment methodology",
                "Check update frequency"
            ],
            status=ComplianceStatus.COMPLIANT,
            risk_level=RiskLevel.HIGH,
            minimum_required=True
        )

        self.controls["SR-002"] = HIPAAControl(
            id="SR-002",
            title="Access Control",
            description="Implement technical policies and procedures for electronic information systems",
            hipaa_section=HIPAASection.SECURITY_RULE,
            covered_entity_responsibility="Implement access controls",
            business_associate_responsibility="Follow access control policies",
            safeguard_type="technical",
            data_protection_scope=list(PHIDataType),
            implementation_specifications=[
                "Unique user identification",
                "Emergency access procedure",
                "Automatic logoff",
                "Encryption and decryption"
            ],
            audit_procedures=[
                "Review access control policies",
                "Test authentication mechanisms",
                "Verify automatic logoff",
                "Assess encryption implementation"
            ],
            status=ComplianceStatus.COMPLIANT,
            risk_level=RiskLevel.HIGH,
            minimum_required=True
        )

        self.controls["SR-003"] = HIPAAControl(
            id="SR-003",
            title="Audit Controls",
            description="Implement hardware, software, and procedural mechanisms to record and examine activity",
            hipaa_section=HIPAASection.SECURITY_RULE,
            covered_entity_responsibility="Implement audit controls",
            business_associate_responsibility="Maintain audit logs",
            safeguard_type="technical",
            data_protection_scope=list(PHIDataType),
            implementation_specifications=[
                "Audit log collection",
                "Log analysis and review",
                "Retention and protection",
                "Monitoring and alerting"
            ],
            audit_procedures=[
                "Review audit log policies",
                "Verify log collection",
                "Check review procedures",
                "Assess monitoring capabilities"
            ],
            status=ComplianceStatus.COMPLIANT,
            risk_level=RiskLevel.MEDIUM,
            addressable=True
        )

        # Breach Notification Controls
        self.controls["BN-001"] = HIPAAControl(
            id="BN-001",
            title="Breach Notification",
            description="Notify individuals of breaches of unsecured PHI",
            hipaa_section=HIPAASection.BREACH_NOTIFICATION,
            covered_entity_responsibility="Implement breach notification process",
            business_associate_responsibility="Report breaches to covered entity",
            safeguard_type="administrative",
            data_protection_scope=list(PHIDataType),
            implementation_specifications=[
                "Breach detection procedures",
                "Risk assessment process",
                "Notification procedures",
                "Documentation requirements"
            ],
            audit_procedures=[
                "Review breach notification procedures",
                "Test breach detection",
                "Verify notification timelines",
                "Check documentation practices"
            ],
            status=ComplianceStatus.COMPLIANT,
            risk_level=RiskLevel.HIGH,
            minimum_required=True
        )

    def assess_business_associate(self, ba_name: str, assessment_data: Dict[str, Any]) -> BusinessAssociate:
        """Assess business associate compliance"""

        ba_id = str(uuid.uuid4())
        contract_date = datetime.now()
        contract_expiry = contract_date + timedelta(days=365)

        # Calculate risk assessment score
        risk_factors = assessment_data.get('risk_factors', {})
        risk_score = self._calculate_ba_risk_score(risk_factors)

        # Determine compliance status
        compliance_score = assessment_data.get('compliance_score', 0)
        if compliance_score >= 90:
            status = ComplianceStatus.COMPLIANT
        elif compliance_score >= 70:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT

        ba = BusinessAssociate(
            id=ba_id,
            name=ba_name,
            contract_date=contract_date,
            contract_expiry=contract_expiry,
            services_provided=assessment_data.get('services_provided', []),
            data_access_level=assessment_data.get('data_access_level', 'limited'),
            compliance_status=status,
            risk_assessment_score=risk_score,
            last_audit_date=datetime.now(),
            next_audit_date=datetime.now() + timedelta(days=180),
            breach_history=assessment_data.get('breach_history', []),
            safeguard_certifications=assessment_data.get('certifications', [])
        )

        self.business_associates[ba_id] = ba

        logger.info(f"Assessed business associate {ba_name} with risk score {risk_score}")
        return ba

    def _calculate_ba_risk_score(self, risk_factors: Dict[str, Any]) -> float:
        """Calculate business associate risk score"""

        weights = {
            'data_sensitivity': 0.3,
            'access_level': 0.25,
            'security_maturity': 0.2,
            'compliance_history': 0.15,
            'geographic_risk': 0.1
        }

        scores = {
            'data_sensitivity': risk_factors.get('data_sensitivity', 0),
            'access_level': risk_factors.get('access_level', 0),
            'security_maturity': risk_factors.get('security_maturity', 0),
            'compliance_history': risk_factors.get('compliance_history', 0),
            'geographic_risk': risk_factors.get('geographic_risk', 0)
        }

        total_score = sum(scores[metric] * weights[metric] for metric in scores)
        return round(total_score, 2)

    def report_breach(self, breach_data: Dict[str, Any]) -> HIPAABreach:
        """Report and manage HIPAA breach"""

        breach_id = str(uuid.uuid4())
        discovery_date = datetime.now()
        breach_date = breach_data.get('breach_date', discovery_date)

        # Calculate notification deadline
        notification_deadline = discovery_date
        if breach_data.get('affected_individuals', 0) > 500:
            # Large breach - notify within 60 days
            notification_deadline += timedelta(days=60)
        else:
            # Small breach - notify without unreasonable delay
            notification_deadline += timedelta(days=30)

        breach = HIPAABreach(
            id=breach_id,
            breach_type=breach_data['breach_type'],
            discovery_date=discovery_date,
            breach_date=breach_date,
            affected_individuals=breach_data['affected_individuals'],
            data_types=breach_data.get('data_types', []),
            breach_description=breach_data['breach_description'],
            cause=breach_data['cause'],
            mitigating_actions=breach_data.get('mitigating_actions', []),
            notification_deadline=notification_deadline,
            notifications_sent=False,
            oic_reported=False,
            individuals_notified=False,
            media_notified=False,
            resolution_status=ComplianceStatus.NON_COMPLIANT,
            remediation_plan=breach_data.get('remediation_plan', ''),
            lessons_learned=''
        )

        self.breaches[breach_id] = breach

        # Trigger breach response
        self._initiate_breach_response(breach_id)

        logger.warning(f"HIPAA breach reported: {breach_id} affecting {breach.affected_individuals} individuals")
        return breach

    def _initiate_breach_response(self, breach_id: str):
        """Initiate breach response procedures"""

        breach = self.breaches.get(breach_id)
        if not breach:
            return

        # Determine if breach requires notification
        risk_assessment = self._assess_breach_notification_requirement(breach)

        if risk_assessment['requires_notification']:
            # Check if large breach (>500 individuals)
            if breach.affected_individuals > 500:
                # Large breach - notify media, HHS, and individuals
                breach.media_notified = True
                breach.oic_reported = True
                logger.info(f"Large breach detected - media and HHS notification required")

        # Create remediation plan if not provided
        if not breach.remediation_plan:
            breach.remediation_plan = self._generate_remediation_plan(breach)

    def _assess_breach_notification_requirement(self, breach: HIPAABreach) -> Dict[str, Any]:
        """Assess if breach requires notification"""

        # Factors that make notification more likely
        high_risk_factors = 0

        # Data type sensitivity
        if PHIDataType.MEDICAL_RECORD in breach.data_types:
            high_risk_factors += 2
        if PHIDataType.IDENTIFIERS in breach.data_types:
            high_risk_factors += 1

        # Cause of breach
        if "hacking" in breach.cause.lower() or "theft" in breach.cause.lower():
            high_risk_factors += 2

        # Duration of breach
        breach_duration = breach.discovery_date - breach.breach_date
        if breach_duration.days > 30:
            high_risk_factors += 1

        # Number of individuals
        if breach.affected_individuals > 100:
            high_risk_factors += 1

        # Calculate probability of compromise
        compromise_probability = min(high_risk_factors / 6.0, 1.0)

        return {
            'requires_notification': compromise_probability > 0.5,
            'compromise_probability': compromise_probability,
            'risk_factors': high_risk_factors
        }

    def _generate_remediation_plan(self, breach: HIPAABreach) -> str:
        """Generate breach remediation plan"""

        plan = []

        # Immediate actions
        plan.append("1. Contain breach and secure systems")
        plan.append("2. Investigate root cause and scope")
        plan.append("3. Notify appropriate personnel and authorities")

        # Medium-term actions
        plan.append("4. Enhance security controls to prevent recurrence")
        plan.append("5. Review and update policies and procedures")
        plan.append("6. Provide additional training to workforce")

        # Long-term actions
        plan.append("7. Implement continuous monitoring")
        plan.append("8. Conduct regular security assessments")
        plan.append("9. Update incident response plan")

        return "\n".join(plan)

    def manage_workforce_training(self, training_data: Dict[str, Any]) -> HIPAATraining:
        """Manage HIPAA workforce training"""

        training_id = str(uuid.uuid4())
        expiry_date = datetime.now() + timedelta(days=365)

        training = HIPAATraining(
            id=training_id,
            employee_id=training_data['employee_id'],
            training_type=training_data['training_type'],
            training_date=training_data['training_date'],
            completion_date=training_data.get('completion_date', datetime.now()),
            training_content=training_data['training_content'],
            trainer_id=training_data['trainer_id'],
            score=training_data.get('score', 0),
            certificate_id=str(uuid.uuid4()),
            expiry_date=expiry_date,
            refresher_required=True,
            topics_covered=training_data.get('topics_covered', []),
            assessment_questions=training_data.get('assessment_questions', [])
        )

        # Store training record
        if training.employee_id not in self.training_records:
            self.training_records[training.employee_id] = []

        self.training_records[training.employee_id].append(training)

        logger.info(f"Recorded HIPAA training for employee {training.employee_id}")
        return training

    def deidentify_phi(self, phi_data: Dict[str, Any]) -> PHIDeclassification:
        """Deidentify protected health information"""

        data_type = PHIDataType(phi_data['data_type'])
        original_value = phi_data['original_value']
        method = phi_data['method']

        # Apply deidentification
        deidentified_value = self._apply_deidentification(original_value, method)

        # Verify deidentification
        verification_result = self._verify_deidentification(original_value, deidentified_value)

        declassification = PHIDeclassification(
            data_type=data_type,
            original_value=original_value,
            deidentified_value=deidentified_value,
            deidentification_method=method,
            verification_result=verification_result,
            deidentified_date=datetime.now(),
            auditor_id=phi_data['auditor_id'],
            compliance_score=self._calculate_compliance_score(verification_result, method)
        )

        self.phi_classifications[declassification.original_value] = declassification

        logger.info(f"Deidentified PHI data using method: {method}")
        return declassification

    def _apply_deidentification(self, value: str, method: str) -> str:
        """Apply deidentification method"""

        if method == "tokenization":
            return f"TOKEN_{hashlib.sha256(value.encode()).hexdigest()[:16]}"
        elif method == "generalization":
            # Replace specific values with general categories
            if "@" in value:  # Email
                return "EMAIL_ADDRESS"
            elif any(char.isdigit() for char in value):  # Phone/ID
                return "IDENTIFIER"
            else:
                return "TEXT_DATA"
        elif method == "pseudonymization":
            return f"PS_{uuid.uuid4().hex[:12]}"
        elif method == "masking":
            return "*" * len(value)
        else:
            return "DEIDENTIFIED"

    def _verify_deidentification(self, original: str, deidentified: str) -> bool:
        """Verify deidentification effectiveness"""

        # Check if deidentified value contains any original identifiers
        original_lower = original.lower()
        deidentified_lower = deidentified.lower()

        # Check for common identifiers
        identifiers = ['ssn', 'dob', 'address', 'phone', 'email', 'name']
        for identifier in identifiers:
            if identifier in original_lower and identifier in deidentified_lower:
                return False

        # Check for numeric patterns
        if any(char.isdigit() for char in original):
            if any(char.isdigit() for char in deidentified):
                return False

        return True

    def _calculate_compliance_score(self, verification_result: bool, method: str) -> float:
        """Calculate deidentification compliance score"""

        base_score = 100 if verification_result else 0

        # Method effectiveness modifiers
        method_scores = {
            "tokenization": 95,
            "pseudonymization": 90,
            "generalization": 85,
            "masking": 75
        }

        method_bonus = method_scores.get(method, 70)

        return min(base_score + (method_bonus / 100) * 10, 100)

    def conduct_compliance_assessment(self) -> ComplianceAssessment:
        """Conduct comprehensive HIPAA compliance assessment"""

        findings = []
        deficiencies = []
        evidence = []

        # Assess all controls
        for control_id, control in self.controls.items():
            # Check control implementation
            control_evidence = self._assess_control_implementation(control)
            evidence.extend(control_evidence)

            # Determine control status
            if control_evidence and all(e['status'] == 'pass' for e in control_evidence):
                control.status = ComplianceStatus.COMPLIANT
            elif any(e['status'] == 'fail' for e in control_evidence):
                control.status = ComplianceStatus.NON_COMPLIANT
                deficiencies.append({
                    'control_id': control_id,
                    'issue': 'Control implementation issues detected',
                    'severity': 'high' if control.risk_level == RiskLevel.HIGH else 'medium'
                })
            else:
                control.status = ComplianceStatus.PARTIALLY_COMPLIANT

        # Assess business associates
        ba_compliance_issues = []
        for ba_id, ba in self.business_associates.items():
            if ba.compliance_status != ComplianceStatus.COMPLIANT:
                ba_compliance_issues.append({
                    'ba_name': ba.name,
                    'issue': f'Business associate not compliant - {ba.compliance_status}',
                    'severity': 'high' if ba.risk_assessment_score > 0.7 else 'medium'
                })

        # Assess breach management
        active_breaches = [b for b in self.breaches.values()
                          if b.resolution_status != ComplianceStatus.COMPLIANT]
        if active_breaches:
            deficiencies.append({
                'control_id': 'BN-001',
                'issue': f'Active breaches requiring attention: {len(active_breaches)}',
                'severity': 'high'
            })

        # Calculate overall compliance score
        compliant_controls = len([c for c in self.controls.values() if c.status == ComplianceStatus.COMPLIANT])
        total_controls = len(self.controls)
        compliance_score = (compliant_controls / total_controls) * 100 if total_controls > 0 else 0

        # Determine overall status
        if compliance_score >= 90 and not deficiencies:
            overall_status = ComplianceStatus.COMPLIANT
        elif compliance_score >= 70:
            overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            overall_status = ComplianceStatus.NON_COMPLIANT

        assessment = ComplianceAssessment(
            id=str(uuid.uuid4()),
            framework_id="HIPAA",
            framework_name="HIPAA Compliance",
            assessment_date=datetime.now(),
            assessor_id="system",
            overall_status=overall_status,
            compliance_score=compliance_score,
            findings=findings,
            deficiencies=deficiencies,
            evidence=evidence,
            recommendations=self._generate_recommendations(deficiencies, ba_compliance_issues),
            next_review_date=datetime.now() + timedelta(days=365)
        )

        logger.info(f"HIPAA compliance assessment completed: {compliance_score:.1f}% compliant")
        return assessment

    def _assess_control_implementation(self, control: HIPAAControl) -> List[ComplianceEvidence]:
        """Assess control implementation effectiveness"""

        evidence = []

        # For each audit procedure, generate evidence
        for procedure in control.audit_procedures:
            evidence_item = ComplianceEvidence(
                id=str(uuid.uuid4()),
                control_id=control.id,
                evidence_type="audit_check",
                evidence_content=f"Audit procedure: {procedure}",
                collection_date=datetime.now(),
                validity_period=365,
                status="pass" if control.status == ComplianceStatus.COMPLIANT else "fail",
                reviewer_id="system",
                confidence_level=0.8
            )
            evidence.append(evidence_item)

        return evidence

    def _generate_recommendations(self, deficiencies: List[Dict], ba_issues: List[Dict]) -> List[str]:
        """Generate remediation recommendations"""

        recommendations = []

        # Control-based recommendations
        for deficiency in deficiencies:
            control_id = deficiency['control_id']
            severity = deficiency['severity']

            if control_id == 'SR-001':
                recommendations.append(f"{'Critical' if severity == 'high' else 'High'}: Conduct comprehensive risk analysis with documented methodology")
            elif control_id == 'SR-002':
                recommendations.append(f"{'Critical' if severity == 'high' else 'High'}: Implement robust access controls with multi-factor authentication")
            elif control_id == 'BN-001':
                recommendations.append(f"{'Critical' if severity == 'high' else 'High'}: Improve breach detection and notification procedures")

        # Business associate recommendations
        for ba_issue in ba_issues:
            recommendations.append(f"High: Address business associate compliance issues for {ba_issue['ba_name']}")

        # General recommendations
        if len(deficiencies) > 3:
            recommendations.append("High: Implement comprehensive HIPAA compliance program with regular monitoring")

        if len(self.breaches) > 0:
            recommendations.append("Medium: Enhance security controls to prevent future breaches")

        return recommendations

    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive HIPAA compliance report"""

        assessment = self.conduct_compliance_assessment()

        report = {
            'report_metadata': {
                'generated_date': datetime.now().isoformat(),
                'covered_entity': self.covered_entity_name,
                'report_type': 'HIPAA Compliance Assessment',
                'framework_version': 'HIPAA 1996/2009/HITECH'
            },
            'compliance_overview': {
                'overall_status': assessment.overall_status.value,
                'compliance_score': assessment.compliance_score,
                'total_controls': len(self.controls),
                'compliant_controls': len([c for c in self.controls.values() if c.status == ComplianceStatus.COMPLIANT]),
                'non_compliant_controls': len([c for c in self.controls.values() if c.status == ComplianceStatus.NON_COMPLIANT]),
                'partially_compliant_controls': len([c for c in self.controls.values() if c.status == ComplianceStatus.PARTIALLY_COMPLIANT])
            },
            'business_associate_status': {
                'total_associates': len(self.business_associates),
                'compliant_associates': len([ba for ba in self.business_associates.values() if ba.compliance_status == ComplianceStatus.COMPLIANT]),
                'high_risk_associates': len([ba for ba in self.business_associates.values() if ba.risk_assessment_score > 0.7]),
                'contracts_expiring_soon': len([ba for ba in self.business_associates.values() if (ba.contract_expiry - datetime.now()).days < 90])
            },
            'breach_status': {
                'total_breaches': len(self.breaches),
                'active_breaches': len([b for b in self.breaches.values() if b.resolution_status != ComplianceStatus.COMPLIANT]),
                'large_breaches': len([b for b in self.breaches.values() if b.affected_individuals > 500]),
                'breaches_pending_notification': len([b for b in self.breaches.values() if not b.notifications_sent and datetime.now() > b.notification_deadline])
            },
            'training_status': {
                'total_employees_trained': len(self.training_records),
                'training_compliance_rate': self._calculate_training_compliance_rate(),
                'certifications_expiring_soon': len([t for employee_trainings in self.training_records.values() for t in employee_trainings if (t.expiry_date - datetime.now()).days < 30])
            },
            'risk_assessment': {
                'overall_risk_score': self._calculate_overall_risk_score(),
                'key_risk_areas': self._identify_key_risk_areas(),
                'risk_trend': 'stable'  # Could be calculated from historical data
            },
            'deficiencies': assessment.deficiencies,
            'recommendations': assessment.recommendations,
            'next_steps': [
                'Address all high-priority deficiencies within 30 days',
                'Schedule business associate audits',
                'Update risk analysis documentation',
                'Conduct workforce training refreshers',
                'Review and update incident response plan'
            ]
        }

        return report

    def _calculate_training_compliance_rate(self) -> float:
        """Calculate workforce training compliance rate"""

        if not self.training_records:
            return 0.0

        current_date = datetime.now()
        compliant_employees = 0
        total_employees = len(self.training_records)

        for employee_trainings in self.training_records.values():
            # Check if employee has valid, non-expired training
            valid_training = any(
                t.completion_date <= current_date and t.expiry_date > current_date
                for t in employee_trainings
            )

            if valid_training:
                compliant_employees += 1

        return (compliant_employees / total_employees) * 100 if total_employees > 0 else 0.0

    def _calculate_overall_risk_score(self) -> float:
        """Calculate overall HIPAA risk score"""

        # Control risk
        control_risks = sum(1 for c in self.controls.values() if c.status != ComplianceStatus.COMPLIANT)
        control_risk_score = (control_risks / len(self.controls)) * 100 if self.controls else 0

        # Business associate risk
        ba_risk_score = sum(ba.risk_assessment_score for ba in self.business_associates.values())
        ba_risk_score = (ba_risk_score / len(self.business_associates)) if self.business_associates else 0

        # Breach risk
        breach_risk_score = min(len(self.breaches) * 10, 100)

        # Training risk
        training_risk_score = 100 - self._calculate_training_compliance_rate()

        # Weighted overall risk
        overall_risk = (
            control_risk_score * 0.4 +
            ba_risk_score * 0.3 +
            breach_risk_score * 0.2 +
            training_risk_score * 0.1
        )

        return round(overall_risk, 2)

    def _identify_key_risk_areas(self) -> List[str]:
        """Identify key risk areas"""

        risk_areas = []

        # Check control risks
        non_compliant_controls = [c for c in self.controls.values() if c.status != ComplianceStatus.COMPLIANT]
        if non_compliant_controls:
            risk_areas.append("Control implementation gaps")

        # Check business associate risks
        high_risk_bas = [ba for ba in self.business_associates.values() if ba.risk_assessment_score > 0.7]
        if high_risk_bas:
            risk_areas.append("Business associate compliance")

        # Check breach risks
        if self.breaches:
            risk_areas.append("Breach management and prevention")

        # Check training risks
        if self._calculate_training_compliance_rate() < 80:
            risk_areas.append("Workforce training compliance")

        return risk_areas