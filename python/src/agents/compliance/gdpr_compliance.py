"""
GDPR Compliance Manager
Specialized compliance management for General Data Protection Regulation
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import re
import hashlib

logger = logging.getLogger(__name__)


class GDPRRight(Enum):
    """GDPR data subject rights"""
    ACCESS = "right_of_access"
    RECTIFICATION = "right_of_rectification"
    ERASURE = "right_to_erasure"
    RESTRICT_PROCESSING = "right_to_restrict_processing"
    DATA_PORTABILITY = "right_to_data_portability"
    OBJECT = "right_to_object"
    NOT_CONSENT_AUTOMATED = "right_not_to_subject_automated"


class DataCategory(Enum):
    """Categories of personal data under GDPR"""
    IDENTIFICATION = "identification_data"
    CONTACT = "contact_data"
    FINANCIAL = "financial_data"
    TECHNICAL = "technical_data"
    PROFILE = "profile_data"
    HEALTH = "health_data"
    BIOMETRIC = "biometric_data"
    GENETIC = "genetic_data"
    RACIAL_ETHNIC = "racial_ethnic_data"
    POLITICAL_OPINION = "political_opinion_data"
    RELIGIOUS_PHILOSOPHICAL = "religious_philosophical_data"
    SEXUAL_ORIENTATION = "sexual_orientation_data"


class LawfulBasis(Enum):
    """Lawful bases for processing under GDPR"""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


class DataTransferMechanism(Enum):
    """Mechanisms for international data transfers"""
    ADEQUACY_DECISION = "adequacy_decision"
    SCCS = "standard_contractual_clauses"
    BCRS = "binding_corporate_rules"
    DEROGATIONS = "derogations"


@dataclass
class DataProcessingActivity:
    """Data processing activity under GDPR"""
    activity_id: str
    name: str
    description: str
    data_categories: List[DataCategory]
    lawful_basis: LawfulBasis
    purpose: str
    data_subjects: List[str]  # Types of data subjects
    retention_period: str
    controller: str
    third_parties: List[str] = field(default_factory=list)
    international_transfers: bool = False
    transfer_mechanism: Optional[DataTransferMechanism] = None
    DPIA_required: bool = False
    DPIA_completed: bool = False
    automated_decision_making: bool = False
    profiling: bool = False
    security_measures: List[str] = field(default_factory=list)
    representatives: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    documentation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activity_id": self.activity_id,
            "name": self.name,
            "description": self.description,
            "data_categories": [dc.value for dc in self.data_categories],
            "lawful_basis": self.lawful_basis.value,
            "purpose": self.purpose,
            "data_subjects": self.data_subjects,
            "retention_period": self.retention_period,
            "third_parties": self.third_parties,
            "international_transfers": self.international_transfers,
            "transfer_mechanism": self.transfer_mechanism.value if self.transfer_mechanism else None,
            "DPIA_required": self.DPIA_required,
            "DPIA_completed": self.DPIA_completed,
            "automated_decision_making": self.automated_decision_making,
            "profiling": self.profiling,
            "security_measures": self.security_measures,
            "controller": self.controller,
            "representatives": self.representatives,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "documentation": self.documentation
        }


@dataclass
class DataSubjectRequest:
    """Data Subject Access Request (DSAR) under GDPR"""
    request_id: str
    request_type: GDPRRight
    subject_id: str
    subject_type: str  # customer, employee, etc.
    request_details: str
    status: str = "received"  # received, in_progress, completed, rejected
    priority: str = "normal"  # low, normal, high, urgent
    assigned_to: Optional[str] = None
    due_date: Optional[datetime] = None
    completed_date: Optional[datetime] = None
    response_data: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    documents: List[str] = field(default_factory=list)  # Document IDs
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "request_type": self.request_type.value,
            "subject_id": self.subject_id,
            "subject_type": self.subject_type,
            "request_details": self.request_details,
            "status": self.status,
            "priority": self.priority,
            "assigned_to": self.assigned_to,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "completed_date": self.completed_date.isoformat() if self.completed_date else None,
            "response_data": self.response_data,
            "notes": self.notes,
            "documents": self.documents,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class DataBreachRecord:
    """Personal data breach record under GDPR"""
    breach_id: str
    description: str
    data_categories_affected: List[DataCategory]
    data_subjects_affected: int
    nature_of_breach: str  # confidentiality, integrity, availability
    causes: List[str]
    consequences: List[str]
    mitigation_measures: List[str]
    controller_representative: str
    DPO_contacted: bool
    supervisory_authority_notified: bool
    notification_date: Optional[datetime] = None
    data_subjects_notified: bool = False
    high_risk: bool = False
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    discovered_date: datetime = field(default_factory=datetime.now)
    containment_date: Optional[datetime] = None
    resolution_date: Optional[datetime] = None
    status: str = "open"  # open, contained, resolved, closed
    severity: str = "medium"  # low, medium, high, critical

    def to_dict(self) -> Dict[str, Any]:
        return {
            "breach_id": self.breach_id,
            "description": self.description,
            "data_categories_affected": [dc.value for dc in self.data_categories_affected],
            "data_subjects_affected": self.data_subjects_affected,
            "nature_of_breach": self.nature_of_breach,
            "causes": self.causes,
            "consequences": self.consequences,
            "mitigation_measures": self.mitigation_measures,
            "controller_representative": self.controller_representative,
            "DPO_contacted": self.DPO_contacted,
            "supervisory_authority_notified": self.supervisory_authority_notified,
            "notification_date": self.notification_date.isoformat() if self.notification_date else None,
            "data_subjects_notified": self.data_subjects_notified,
            "high_risk": self.high_risk,
            "risk_assessment": self.risk_assessment,
            "discovered_date": self.discovered_date.isoformat(),
            "containment_date": self.containment_date.isoformat() if self.containment_date else None,
            "resolution_date": self.resolution_date.isoformat() if self.resolution_date else None,
            "status": self.status,
            "severity": self.severity
        }


@dataclass
class ConsentRecord:
    """Consent record under GDPR"""
    consent_id: str
    subject_id: str
    processing_activity_id: str
    purpose: str
    data_categories: List[DataCategory]
    consent_given: bool
    timestamp: datetime
    method: str  # web_form, email, phone, paper
    withdrawal_timestamp: Optional[datetime] = None
    version: str = "1.0"
    privacy_policy_version: str = ""
    granular: bool = True
    specific: bool = True
    informed: bool = True
    unambiguous: bool = True
    documentation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consent_id": self.consent_id,
            "subject_id": self.subject_id,
            "processing_activity_id": self.processing_activity_id,
            "purpose": self.purpose,
            "data_categories": [dc.value for dc in self.data_categories],
            "consent_given": self.consent_given,
            "timestamp": self.timestamp.isoformat(),
            "withdrawal_timestamp": self.withdrawal_timestamp.isoformat() if self.withdrawal_timestamp else None,
            "method": self.method,
            "version": self.version,
            "privacy_policy_version": self.privacy_policy_version,
            "granular": self.granular,
            "specific": self.specific,
            "informed": self.informed,
            "unambiguous": self.unambiguous,
            "documentation": self.documentation
        }


class GDPRComplianceManager:
    """GDPR compliance management system"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.processing_activities: Dict[str, DataProcessingActivity] = {}
        self.data_subject_requests: Dict[str, DataSubjectRequest] = {}
        self.breach_records: Dict[str, DataBreachRecord] = {}
        self.consent_records: Dict[str, ConsentRecord] = {}

        # Configuration
        self.dsar_response_deadline = timedelta(days=self.config.get("dsar_response_days", 30))
        self.breach_notification_threshold = timedelta(hours=self.config.get("breach_notification_hours", 72))
        self.data_retention_policies = self.config.get("retention_policies", {})
        self.DPO_contact = self.config.get("dpo_contact", "dpo@organization.com")

        # Initialize with sample processing activities
        self._initialize_sample_activities()

    def _initialize_sample_activities(self) -> None:
        """Initialize sample data processing activities"""
        # User registration processing
        user_registration = DataProcessingActivity(
            activity_id="user_registration",
            name="User Registration",
            description="Processing of personal data during user registration",
            data_categories=[DataCategory.IDENTIFICATION, DataCategory.CONTACT],
            lawful_basis=LawfulBasis.CONSENT,
            purpose="Account creation and service access",
            data_subjects=["customers"],
            retention_period="2 years after account deletion",
            security_measures=["encryption_at_rest", "access_controls", "audit_logging"],
            controller="Organization Ltd."
        )
        self.processing_activities[user_registration.activity_id] = user_registration

        # Marketing communications
        marketing_comms = DataProcessingActivity(
            activity_id="marketing_communications",
            name="Marketing Communications",
            description="Sending marketing emails and promotional content",
            data_categories=[DataCategory.CONTACT, DataCategory.PROFILE],
            lawful_basis=LawfulBasis.CONSENT,
            purpose="Marketing and promotional activities",
            data_subjects=["customers"],
            retention_period="6 months after unsubscribe",
            security_measures=["encryption_in_transit", "consent_management"],
            controller="Marketing Department"
        )
        self.processing_activities[marketing_comms.activity_id] = marketing_comms

    async def register_processing_activity(self, activity: DataProcessingActivity) -> bool:
        """Register a new data processing activity"""
        # Validate required fields
        if not activity.name or not activity.lawful_basis or not activity.purpose:
            logger.error("Missing required fields for processing activity")
            return False

        # Check if DPIA is required for high-risk processing
        if self._requires_dpia(activity):
            activity.DPIA_required = True

        activity.updated_at = datetime.now()
        self.processing_activities[activity.activity_id] = activity

        logger.info(f"Registered processing activity: {activity.name}")
        return True

    def _requires_dpia(self, activity: DataProcessingActivity) -> bool:
        """Determine if DPIA is required for processing activity"""
        # High-risk processing requiring DPIA
        high_risk_categories = [
            DataCategory.HEALTH,
            DataCategory.BIOMETRIC,
            DataCategory.GENETIC,
            DataCategory.RACIAL_ETHNIC,
            DataCategory.POLITICAL_OPINION,
            DataCategory.RELIGIOUS_PHILOSOPHICAL,
            DataCategory.SEXUAL_ORIENTATION
        ]

        # Check if processing special categories
        if any(category in high_risk_categories for category in activity.data_categories):
            return True

        # Check for systematic profiling
        if activity.profiling and activity.automated_decision_making:
            return True

        # Check for large-scale processing
        if "large_scale" in activity.description.lower():
            return True

        # Check for public monitoring
        if "public" in activity.description.lower() and "monitoring" in activity.description.lower():
            return True

        return False

    async def create_data_subject_request(self, request_type: GDPRRight,
                                       subject_id: str, subject_type: str,
                                       request_details: str,
                                       priority: str = "normal") -> DataSubjectRequest:
        """Create a new data subject request"""
        request_id = str(uuid.uuid4())
        due_date = datetime.now() + self.dsar_response_deadline

        request = DataSubjectRequest(
            request_id=request_id,
            request_type=request_type,
            subject_id=subject_id,
            subject_type=subject_type,
            request_details=request_details,
            priority=priority,
            due_date=due_date
        )

        self.data_subject_requests[request_id] = request

        # Log the request
        logger.info(f"Created data subject request: {request_type.value} for {subject_id}")

        # If urgent, send notification
        if priority in ["high", "urgent"]:
            await self._send_urgent_request_notification(request)

        return request

    async def process_data_subject_request(self, request_id: str,
                                         processor_id: str) -> bool:
        """Process a data subject request"""
        request = self.data_subject_requests.get(request_id)
        if not request:
            logger.error(f"Request not found: {request_id}")
            return False

        # Update request status
        request.status = "in_progress"
        request.assigned_to = processor_id
        request.updated_at = datetime.now()

        try:
            # Process based on request type
            if request.request_type == GDPRRight.ACCESS:
                await self._process_access_request(request)
            elif request.request_type == GDPRRight.ERASURE:
                await self._process_erasure_request(request)
            elif request.request_type == GDPRRight.RECTIFICATION:
                await self._process_rectification_request(request)
            elif request.request_type == GDPRRight.DATA_PORTABILITY:
                await self._process_portability_request(request)
            else:
                await self._process_generic_request(request)

            # Mark as completed
            request.status = "completed"
            request.completed_date = datetime.now()
            request.updated_at = datetime.now()

            logger.info(f"Completed data subject request: {request_id}")
            return True

        except Exception as e:
            logger.error(f"Error processing request {request_id}: {str(e)}")
            request.status = "rejected"
            request.notes.append(f"Processing error: {str(e)}")
            request.updated_at = datetime.now()
            return False

    async def _process_access_request(self, request: DataSubjectRequest) -> None:
        """Process right of access request"""
        # Collect all personal data related to the subject
        personal_data = await self._collect_personal_data(request.subject_id)

        request.response_data = {
            "personal_data": personal_data,
            "processing_activities": [
                activity.to_dict() for activity in self.processing_activities.values()
            ],
            "data_sources": ["user_database", "analytics_system", "marketing_platform"],
            "retention_periods": self._get_retention_periods(request.subject_id),
            "third_party_disclosures": self._get_third_party_disclosures(request.subject_id)
        }

        # Generate data package
        data_package_id = await self._generate_data_package(request.subject_id, personal_data)
        if data_package_id:
            request.documents.append(data_package_id)

    async def _process_erasure_request(self, request: DataSubjectRequest) -> None:
        """Process right to erasure (right to be forgotten) request"""
        # Identify all personal data to be deleted
        data_to_delete = await self._identify_personal_data(request.subject_id)

        # Check for legal grounds to refuse erasure
        refusal_reasons = await self._check_erasure_refusal_grounds(request.subject_id)

        if refusal_reasons:
            request.response_data = {
                "action": "refused",
                "reasons": refusal_reasons,
                "legal_basis": "Legal obligation to retain data"
            }
        else:
            # Execute data deletion
            deletion_results = await self._execute_data_deletion(request.subject_id, data_to_delete)

            request.response_data = {
                "action": "completed",
                "deleted_data": deletion_results,
                "verification_timestamp": datetime.now().isoformat()
            }

    async def _process_rectification_request(self, request: DataSubjectRequest) -> None:
        """Process right to rectification request"""
        # Parse rectification details from request
        rectification_details = self._parse_rectification_request(request.request_details)

        # Validate rectification request
        validation_result = await self._validate_rectification_request(rectification_details)

        if validation_result["valid"]:
            # Execute data correction
            correction_results = await self._execute_data_correction(rectification_details)

            request.response_data = {
                "action": "completed",
                "corrections_made": correction_results,
                "verification_timestamp": datetime.now().isoformat()
            }
        else:
            request.response_data = {
                "action": "rejected",
                "reasons": validation_result["errors"]
            }

    async def _process_portability_request(self, request: DataSubjectRequest) -> None:
        """Process right to data portability request"""
        # Collect data for export
        personal_data = await self._collect_personal_data(request.subject_id)

        # Convert to portable format
        portable_data = await self._convert_to_portable_format(personal_data)

        # Generate portable package
        package_id = await self._generate_portable_package(request.subject_id, portable_data)

        request.response_data = {
            "action": "completed",
            "data_format": "json",  # or xml, csv
            "package_id": package_id,
            "generation_timestamp": datetime.now().isoformat()
        }

        if package_id:
            request.documents.append(package_id)

    async def _process_generic_request(self, request: DataSubjectRequest) -> None:
        """Process generic data subject request"""
        # Handle other request types
        request.response_data = {
            "action": "acknowledged",
            "message": f"Request type {request.request_type.value} has been received and is being processed",
            "estimated_completion_date": (datetime.now() + timedelta(days=15)).isoformat()
        }

    async def record_data_breach(self, breach: DataBreachRecord) -> bool:
        """Record a personal data breach"""
        # Validate breach record
        if not breach.description or not breach.data_categories_affected:
            logger.error("Missing required breach information")
            return False

        # Assess risk level
        breach.high_risk = self._assess_breach_risk(breach)
        breach.severity = self._determine_breach_severity(breach)

        # Set due date for notification if high risk
        if breach.high_risk:
            breach.notification_due = breach.discovered_date + self.breach_notification_threshold

        self.breach_records[breach.breach_id] = breach

        # Notify DPO
        await self._notify_dpo_of_breach(breach)

        logger.warning(f"Recorded data breach: {breach.breach_id} - {breach.severity} severity")
        return True

    def _assess_breach_risk(self, breach: DataBreachRecord) -> bool:
        """Assess if breach poses high risk to data subjects"""
        high_risk_indicators = 0

        # Check sensitive data categories
        sensitive_categories = [
            DataCategory.HEALTH, DataCategory.BIOMETRIC, DataCategory.GENETIC,
            DataCategory.RACIAL_ETHNIC, DataCategory.FINANCIAL
        ]

        if any(category in sensitive_categories for category in breach.data_categories_affected):
            high_risk_indicators += 1

        # Check number of affected subjects
        if breach.data_subjects_affected > 1000:
            high_risk_indicators += 1

        # Check breach nature
        if breach.nature_of_breach in ["confidentiality", "multiple"]:
            high_risk_indicators += 1

        # Check consequences
        severe_consequences = ["identity_theft", "financial_loss", "discrimination", "reputational_damage"]
        if any(consequence in severe_consequences for consequence in breach.consequences):
            high_risk_indicators += 1

        return high_risk_indicators >= 2

    def _determine_breach_severity(self, breach: DataBreachRecord) -> str:
        """Determine breach severity level"""
        severity_score = 0

        # Data sensitivity
        if any(cat in [DataCategory.HEALTH, DataCategory.BIOMETRIC, DataCategory.FINANCIAL]
               for cat in breach.data_categories_affected):
            severity_score += 3
        elif any(cat in [DataCategory.GENETIC, DataCategory.RACIAL_ETHNIC, DataCategory.POLITICAL_OPINION]
                 for cat in breach.data_categories_affected):
            severity_score += 2

        # Scale of breach
        if breach.data_subjects_affected > 10000:
            severity_score += 3
        elif breach.data_subjects_affected > 1000:
            severity_score += 2
        elif breach.data_subjects_affected > 100:
            severity_score += 1

        # Impact
        if "identity_theft" in breach.consequences:
            severity_score += 3
        elif "financial_loss" in breach.consequences:
            severity_score += 2

        if severity_score >= 6:
            return "critical"
        elif severity_score >= 4:
            return "high"
        elif severity_score >= 2:
            return "medium"
        else:
            return "low"

    async def notify_supervisory_authority(self, breach_id: str) -> bool:
        """Notify supervisory authority of data breach"""
        breach = self.breach_records.get(breach_id)
        if not breach:
            logger.error(f"Breach not found: {breach_id}")
            return False

        # Check if notification is required
        if not self._requires_authority_notification(breach):
            logger.info("Authority notification not required for this breach")
            return True

        # Prepare notification data
        notification_data = {
            "breach_id": breach.breach_id,
            "controller": breach.controller_representative,
            "description": breach.description,
            "data_categories": [cat.value for cat in breach.data_categories_affected],
            "subjects_affected": breach.data_subjects_affected,
            "consequences": breach.consequences,
            "measures_taken": breach.mitigation_measures,
            "DPO_contact": self.DPO_contact,
            "notification_timestamp": datetime.now().isoformat()
        }

        # Send notification (simulated)
        await self._send_authority_notification(notification_data)

        breach.supervisory_authority_notified = True
        breach.notification_date = datetime.now()

        logger.info(f"Notified supervisory authority of breach: {breach_id}")
        return True

    def _requires_authority_notification(self, breach: DataBreachRecord) -> bool:
        """Determine if breach requires supervisory authority notification"""
        # Notify if high risk to rights and freedoms
        if breach.high_risk:
            return True

        # Notify for certain breach types regardless of risk
        if breach.data_subjects_affected > 1000:
            return True

        # Notify for sensitive data categories
        sensitive_categories = [
            DataCategory.HEALTH, DataCategory.BIOMETRIC, DataCategory.GENETIC,
            DataCategory.RACIAL_ETHNIC, DataCategory.POLITICAL_OPINION
        ]
        if any(category in sensitive_categories for category in breach.data_categories_affected):
            return True

        return False

    async def record_consent(self, consent: ConsentRecord) -> bool:
        """Record consent given by data subject"""
        # Validate consent record
        if not consent.subject_id or not consent.processing_activity_id:
            logger.error("Missing required consent information")
            return False

        # Validate consent quality
        consent.informed = await self._validate_informed_consent(consent)
        consent.granular = await self._validate_granular_consent(consent)
        consent.specific = await self._validate_specific_consent(consent)
        consent.unambiguous = await self._validate_unambiguous_consent(consent)

        consent.timestamp = datetime.now()
        self.consent_records[consent.consent_id] = consent

        logger.info(f"Recorded consent: {consent.consent_id} for subject {consent.subject_id}")
        return True

    async def withdraw_consent(self, consent_id: str, subject_id: str) -> bool:
        """Withdraw previously given consent"""
        consent = self.consent_records.get(consent_id)
        if not consent:
            logger.error(f"Consent not found: {consent_id}")
            return False

        # Verify subject ownership
        if consent.subject_id != subject_id:
            logger.error("Unauthorized consent withdrawal attempt")
            return False

        # Record withdrawal
        consent.consent_given = False
        consent.withdrawal_timestamp = datetime.now()

        # Initiate data processing stop
        await self._stop_processing_based_on_consent(consent)

        logger.info(f"Withdrew consent: {consent_id}")
        return True

    async def generate_gdpr_report(self, report_type: str = "comprehensive") -> Dict[str, Any]:
        """Generate GDPR compliance report"""
        report = {
            "report_id": str(uuid.uuid4()),
            "framework": "GDPR",
            "report_type": report_type,
            "generated_at": datetime.now().isoformat(),
            "period": {
                "start": (datetime.now() - timedelta(days=90)).isoformat(),
                "end": datetime.now().isoformat()
            }
        }

        if report_type == "comprehensive":
            report.update({
                "processing_activities": [
                    activity.to_dict() for activity in self.processing_activities.values()
                ],
                "data_subject_requests": self._get_dsar_summary(),
                "data_breaches": self._get_breach_summary(),
                "consent_records": self._get_consent_summary(),
                "compliance_score": self._calculate_gdpr_compliance_score(),
                "recommendations": self._generate_gdpr_recommendations()
            })

        return report

    def _get_dsar_summary(self) -> Dict[str, Any]:
        """Get data subject request summary"""
        if not self.data_subject_requests:
            return {"total": 0, "by_status": {}, "by_type": {}}

        total_requests = len(self.data_subject_requests)
        by_status = {}
        by_type = {}

        for request in self.data_subject_requests.values():
            by_status[request.status] = by_status.get(request.status, 0) + 1
            by_type[request.request_type.value] = by_type.get(request.request_type.value, 0) + 1

        # Calculate average response time
        completed_requests = [r for r in self.data_subject_requests.values() if r.completed_date]
        avg_response_time = None
        if completed_requests:
            response_times = [
                (r.completed_date - r.created_at).total_seconds() / 3600  # hours
                for r in completed_requests
            ]
            avg_response_time = sum(response_times) / len(response_times)

        return {
            "total": total_requests,
            "by_status": by_status,
            "by_type": by_type,
            "average_response_time_hours": avg_response_time,
            "within_deadline_percentage": self._calculate_deadline_compliance()
        }

    def _get_breach_summary(self) -> Dict[str, Any]:
        """Get data breach summary"""
        if not self.breach_records:
            return {"total": 0, "by_severity": {}, "notification_metrics": {}}

        total_breaches = len(self.breach_records)
        by_severity = {}
        notification_metrics = {
            "authority_notified": 0,
            "subjects_notified": 0,
            "average_notification_time_hours": None
        }

        notification_times = []

        for breach in self.breach_records.values():
            by_severity[breach.severity] = by_severity.get(breach.severity, 0) + 1

            if breach.supervisory_authority_notified and breach.notification_date:
                notification_metrics["authority_notified"] += 1
                if breach.discovered_date:
                    notification_time = (breach.notification_date - breach.discovered_date).total_seconds() / 3600
                    notification_times.append(notification_time)

            if breach.data_subjects_notified:
                notification_metrics["subjects_notified"] += 1

        if notification_times:
            notification_metrics["average_notification_time_hours"] = sum(notification_times) / len(notification_times)

        return {
            "total": total_breaches,
            "by_severity": by_severity,
            "notification_metrics": notification_metrics
        }

    def _get_consent_summary(self) -> Dict[str, Any]:
        """Get consent record summary"""
        if not self.consent_records:
            return {"total": 0, "active_consents": 0, "withdrawn_consents": 0}

        total_consents = len(self.consent_records)
        active_consents = len([c for c in self.consent_records.values() if c.consent_given])
        withdrawn_consents = total_consents - active_consents

        return {
            "total": total_consents,
            "active_consents": active_consents,
            "withdrawn_consents": withdrawn_consents,
            "compliance_rate": (active_consents / total_consents * 100) if total_consents > 0 else 0
        }

    def _calculate_gdpr_compliance_score(self) -> float:
        """Calculate overall GDPR compliance score"""
        # Score based on multiple factors
        scores = []

        # Processing activities documentation (25%)
        if self.processing_activities:
            documented_activities = len([a for a in self.processing_activities.values() if a.DPIA_required == a.DPIA_completed])
            activities_score = (documented_activities / len(self.processing_activities)) * 100
            scores.append(activities_score * 0.25)

        # DSAR compliance (25%)
        if self.data_subject_requests:
            completed_requests = [r for r in self.data_subject_requests.values() if r.status == "completed"]
            completed_in_time = [
                r for r in completed_requests
                if r.completed_date and r.due_date and r.completed_date <= r.due_date
            ]
            dsar_score = (len(completed_in_time) / len(self.data_subject_requests)) * 100 if self.data_subject_requests else 100
            scores.append(dsar_score * 0.25)

        # Breach notification compliance (25%)
        if self.breach_records:
            notifiable_breaches = [b for b in self.breach_records.values() if self._requires_authority_notification(b)]
            notified_breaches = [b for b in notifiable_breaches if b.supervisory_authority_notified]
            breach_score = (len(notified_breaches) / len(notifiable_breaches)) * 100 if notifiable_breaches else 100
            scores.append(breach_score * 0.25)

        # Consent management (25%)
        if self.consent_records:
            valid_consents = [
                c for c in self.consent_records.values()
                if c.informed and c.granular and c.specific and c.unambiguous
            ]
            consent_score = (len(valid_consents) / len(self.consent_records)) * 100
            scores.append(consent_score * 0.25)

        return sum(scores) if scores else 0.0

    def _generate_gdpr_recommendations(self) -> List[str]:
        """Generate GDPR compliance recommendations"""
        recommendations = []

        # Check processing activities
        activities_without_dpia = [
            a for a in self.processing_activities.values()
            if a.DPIA_required and not a.DPIA_completed
        ]
        if activities_without_dpia:
            recommendations.append(f"Complete DPIA for {len(activities_without_dpia)} high-risk processing activities")

        # Check DSAR response times
        overdue_requests = [
            r for r in self.data_subject_requests.values()
            if r.status not in ["completed", "rejected"] and r.due_date and datetime.now() > r.due_date
        ]
        if overdue_requests:
            recommendations.append(f"Address {len(overdue_requests)} overdue data subject requests")

        # Check breach notifications
        unnotified_breaches = [
            b for b in self.breach_records.values()
            if self._requires_authority_notification(b) and not b.supervisory_authority_notified
        ]
        if unnotified_breaches:
            recommendations.append(f"Notify supervisory authority for {len(unnotified_breaches)} breaches")

        # Check consent quality
        invalid_consents = [
            c for c in self.consent_records.values()
            if not (c.informed and c.granular and c.specific and c.unambiguous)
        ]
        if invalid_consents:
            recommendations.append(f"Improve consent quality for {len(invalid_consents)} consent records")

        return recommendations

    # Helper methods (simplified implementations)
    async def _collect_personal_data(self, subject_id: str) -> Dict[str, Any]:
        """Collect personal data for a subject"""
        # In real implementation, this would query various systems
        return {
            "subject_id": subject_id,
            "data_found": True,
            "data_sources": ["user_profile", "order_history", "communication_log"],
            "data_categories": ["identification", "contact", "financial"],
            "last_updated": datetime.now().isoformat()
        }

    async def _send_urgent_request_notification(self, request: DataSubjectRequest) -> None:
        """Send notification for urgent DSAR"""
        logger.warning(f"Urgent DSAR notification: {request.request_id}")

    async def _notify_dpo_of_breach(self, breach: DataBreachRecord) -> None:
        """Notify DPO of data breach"""
        logger.warning(f"DPO breach notification: {breach.breach_id}")

    async def _send_authority_notification(self, notification_data: Dict[str, Any]) -> None:
        """Send notification to supervisory authority"""
        logger.info(f"Authority notification sent for breach: {notification_data['breach_id']}")

    async def _validate_informed_consent(self, consent: ConsentRecord) -> bool:
        """Validate if consent is informed"""
        return True  # Simplified validation

    async def _validate_granular_consent(self, consent: ConsentRecord) -> bool:
        """Validate if consent is granular"""
        return True  # Simplified validation

    async def _validate_specific_consent(self, consent: ConsentRecord) -> bool:
        """Validate if consent is specific"""
        return True  # Simplified validation

    async def _validate_unambiguous_consent(self, consent: ConsentRecord) -> bool:
        """Validate if consent is unambiguous"""
        return True  # Simplified validation

    async def _stop_processing_based_on_consent(self, consent: ConsentRecord) -> None:
        """Stop processing based on withdrawn consent"""
        logger.info(f"Stopping processing for activity: {consent.processing_activity_id}")

    def _calculate_deadline_compliance(self) -> float:
        """Calculate percentage of DSARs completed within deadline"""
        if not self.data_subject_requests:
            return 100.0

        completed_requests = [r for r in self.data_subject_requests.values() if r.completed_date]
        if not completed_requests:
            return 100.0

        on_time = [
            r for r in completed_requests
            if r.completed_date and r.due_date and r.completed_date <= r.due_date
        ]

        return (len(on_time) / len(completed_requests)) * 100

    def _parse_rectification_request(self, request_details: str) -> Dict[str, Any]:
        """Parse rectification request details"""
        return {"data_field": "email", "current_value": "old@email.com", "correct_value": "new@email.com"}

    async def _validate_rectification_request(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Validate rectification request"""
        return {"valid": True, "errors": []}

    async def _execute_data_correction(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data correction"""
        return {"corrected_fields": ["email"], "timestamp": datetime.now().isoformat()}

    async def _identify_personal_data(self, subject_id: str) -> List[str]:
        """Identify personal data to be deleted"""
        return ["user_profile", "order_history", "communications"]

    async def _check_erasure_refusal_grounds(self, subject_id: str) -> List[str]:
        """Check legal grounds for refusing erasure"""
        return []  # No grounds to refuse in this case

    async def _execute_data_deletion(self, subject_id: str, data_to_delete: List[str]) -> Dict[str, Any]:
        """Execute data deletion"""
        return {"deleted_data": data_to_delete, "timestamp": datetime.now().isoformat()}

    async def _convert_to_portable_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert data to portable format"""
        return {"format": "json", "data": data, "schema_version": "1.0"}

    async def _generate_data_package(self, subject_id: str, data: Dict[str, Any]) -> str:
        """Generate data package for subject access request"""
        return f"package_{subject_id}_{uuid.uuid4().hex[:8]}"

    async def _generate_portable_package(self, subject_id: str, data: Dict[str, Any]) -> str:
        """Generate portable data package"""
        return f"portable_{subject_id}_{uuid.uuid4().hex[:8]}"

    def _get_retention_periods(self, subject_id: str) -> Dict[str, str]:
        """Get retention periods for subject's data"""
        return {"user_profile": "2 years", "order_history": "7 years", "communications": "1 year"}

    def _get_third_party_disclosures(self, subject_id: str) -> List[Dict[str, Any]]:
        """Get third party disclosures for subject"""
        return [{"recipient": "payment_processor", "purpose": "payment_processing", "date": "2023-01-01"}]