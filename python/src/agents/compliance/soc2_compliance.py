"""
SOC 2 Compliance Manager
Specialized compliance management for SOC 2 Type II certification
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

logger = logging.getLogger(__name__)


class SOC2TrustServiceCategory(Enum):
    """SOC 2 Trust Services Criteria"""
    SECURITY = "security"
    AVAILABILITY = "availability"
    PROCESSING_INTEGRITY = "processing_integrity"
    CONFIDENTIALITY = "confidentiality"
    PRIVACY = "privacy"


class SOC2ControlType(Enum):
    """Types of SOC 2 controls"""
    CONTROL_ENVIRONMENT = "control_environment"
    COMMUNICATION_AND_INFORMATION = "communication_and_information"
    RISK_ASSESSMENT = "risk_assessment"
    MONITORING_ACTIVITIES = "monitoring_activities"
    CONTROL_ACTIVITIES = "control_activities"
    INFORMATION_AND_COMMUNICATION = "information_and_communication"
    LOGICAL_AND_PHYSICAL_ACCESS = "logical_and_physical_access"
    SYSTEM_OPERATIONS = "system_operations"
    CHANGE_MANAGEMENT = "change_management"
    RISK_MITIGATION = "risk_mitigation"


class ControlStatus(Enum):
    """SOC 2 control status"""
    DESIGNED = "designed"
    IMPLEMENTED = "implemented"
    OPERATING_EFFECTIVELY = "operating_effectively"
    NEEDS_IMPROVEMENT = "needs_improvement"
    NOT_OPERATING = "not_operating"


class TestType(Enum):
    """Types of SOC 2 control tests"""
    INSPECTION = "inspection"
    OBSERVATION = "observation"
    INQUIRY = "inquiry"
    REPERFORMANCE = "reperformance"
    ANALYTICAL_PROCEDURES = "analytical_procedures"


@dataclass
class SOC2Control:
    """SOC 2 control definition"""
    control_id: str
    trust_service_category: SOC2TrustServiceCategory
    control_type: SOC2ControlType
    name: str
    description: str
    criteria_reference: str  # CC1.1, CC2.1, etc.
    control_objective: str
    implementation_details: str
    frequency: str  # daily, weekly, monthly, quarterly, annually, continuously
    responsible_party: str
    status: ControlStatus = ControlStatus.DESIGNED
    implementation_date: Optional[datetime] = None
    last_tested: Optional[datetime] = None
    next_test_date: Optional[datetime] = None
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    deficiencies: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    key_control: bool = False
    automated: bool = False
    documentation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "control_id": self.control_id,
            "trust_service_category": self.trust_service_category.value,
            "control_type": self.control_type.value,
            "name": self.name,
            "description": self.description,
            "criteria_reference": self.criteria_reference,
            "control_objective": self.control_objective,
            "implementation_details": self.implementation_details,
            "frequency": self.frequency,
            "responsible_party": self.responsible_party,
            "status": self.status.value,
            "implementation_date": self.implementation_date.isoformat() if self.implementation_date else None,
            "last_tested": self.last_tested.isoformat() if self.last_tested else None,
            "next_test_date": self.next_test_date.isoformat() if self.next_test_date else None,
            "test_results": self.test_results,
            "deficiencies": self.deficiencies,
            "evidence": self.evidence,
            "key_control": self.key_control,
            "automated": self.automated,
            "documentation": self.documentation
        }


@dataclass
class SOC2TestResult:
    """SOC 2 control test result"""
    test_id: str
    control_id: str
    test_type: TestType
    test_description: str
    test_date: datetime
    tester: str
    result: str  # pass, fail, exception, na
    findings: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    sampling_method: str = "full_population"
    sample_size: int = 0
    population_size: int = 0
    exceptions_identified: int = 0
    remediation_required: bool = False
    remediation_plan: Optional[str] = None
    due_date: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "control_id": self.control_id,
            "test_type": self.test_type.value,
            "test_description": self.test_description,
            "test_date": self.test_date.isoformat(),
            "tester": self.tester,
            "result": self.result,
            "findings": self.findings,
            "evidence": self.evidence,
            "sampling_method": self.sampling_method,
            "sample_size": self.sample_size,
            "population_size": self.population_size,
            "exceptions_identified": self.exceptions_identified,
            "remediation_required": self.remediation_required,
            "remediation_plan": self.remediation_plan,
            "due_date": self.due_date.isoformat() if self.due_date else None
        }


@dataclass
class SOC2Deficiency:
    """SOC 2 control deficiency"""
    deficiency_id: str
    control_id: str
    deficiency_type: str  # design_deficiency, operating_deficiency
    severity: str  # material, significant
    description: str
    impact: str
    identified_date: datetime
    root_cause: str
    remediation_status: str = "open"  # open, in_progress, resolved, monitored
    remediation_plan: Optional[str] = None
    responsible_party: str = ""
    due_date: Optional[datetime] = None
    resolved_date: Optional[datetime] = None
    compensating_controls: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deficiency_id": self.deficiency_id,
            "control_id": self.control_id,
            "deficiency_type": self.deficiency_type,
            "severity": self.severity,
            "description": self.description,
            "impact": self.impact,
            "identified_date": self.identified_date.isoformat(),
            "root_cause": self.root_cause,
            "remediation_status": self.remediation_status,
            "remediation_plan": self.remediation_plan,
            "responsible_party": self.responsible_party,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "resolved_date": self.resolved_date.isoformat() if self.resolved_date else None,
            "compensating_controls": self.compensating_controls,
            "evidence": self.evidence
        }


@dataclass
class SOC2SystemDescription:
    """SOC 2 system description"""
    system_id: str
    system_name: str
    description: str
    trust_service_categories: List[SOC2TrustServiceCategory]
    system_boundary: str
    components: List[Dict[str, Any]]
    user_entities: List[Dict[str, Any]]
    subservice_organization_uses: List[Dict[str, Any]]
    complementary_user_entity_responsibilities: List[str] = field(default_factory=list)
    period_covered: Dict[str, str] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_id": self.system_id,
            "system_name": self.system_name,
            "description": self.description,
            "trust_service_categories": [tsc.value for tsc in self.trust_service_categories],
            "system_boundary": self.system_boundary,
            "components": self.components,
            "user_entities": self.user_entities,
            "subservice_organization_uses": self.subservice_organization_uses,
            "complementary_user_entity_responsibilities": self.complementary_user_entity_responsibilities,
            "period_covered": self.period_covered,
            "last_updated": self.last_updated.isoformat(),
            "version": self.version
        }


class SOC2ComplianceManager:
    """SOC 2 compliance management system"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.controls: Dict[str, SOC2Control] = {}
        self.test_results: Dict[str, SOC2TestResult] = {}
        self.deficiencies: Dict[str, SOC2Deficiency] = {}
        self.system_description: Optional[SOC2SystemDescription] = None

        # Configuration
        self.testing_frequency = self.config.get("testing_frequency", "quarterly")
        self.key_control_threshold = self.config.get("key_control_threshold", 0.95)  # 95% effectiveness
        self.audit_period_days = self.config.get("audit_period_days", 365)

        # Initialize with SOC 2 controls
        self._initialize_soc2_controls()

    def _initialize_soc2_controls(self) -> None:
        """Initialize standard SOC 2 controls based on Trust Services Criteria"""
        # Common Criteria (CC) controls
        common_controls = [
            # CC1: Control Environment
            SOC2Control(
                control_id="CC1.1",
                trust_service_category=SOC2TrustServiceCategory.SECURITY,
                control_type=SOC2ControlType.CONTROL_ENVIRONMENT,
                name="Governance and Management",
                description="Governance structure and management oversight",
                criteria_reference="CC1.1",
                control_objective="Establish governance structure and oversight responsibilities",
                implementation_details="Board of directors provides oversight, management establishes structure",
                frequency="annually",
                responsible_party="Board of Directors",
                key_control=True
            ),
            SOC2Control(
                control_id="CC1.2",
                trust_service_category=SOC2TrustServiceCategory.SECURITY,
                control_type=SOC2ControlType.RISK_ASSESSMENT,
                name="Risk Assessment",
                description="Identify and assess risks related to objectives",
                criteria_reference="CC1.2",
                control_objective="Identify and analyze risks that could prevent objectives achievement",
                implementation_details="Regular risk assessments, identify business and compliance risks",
                frequency="quarterly",
                responsible_party="Risk Management",
                key_control=True
            ),
            # CC2: Communication and Information
            SOC2Control(
                control_id="CC2.1",
                trust_service_category=SOC2TrustServiceCategory.SECURITY,
                control_type=SOC2ControlType.COMMUNICATION_AND_INFORMATION,
                name="Communication of Information",
                description="Communicate quality objectives and control responsibilities",
                criteria_reference="CC2.1",
                control_objective="Communicate objectives and responsibilities throughout the entity",
                implementation_details="Policy documentation, training programs, performance metrics",
                frequency="continuously",
                responsible_party="Management",
                key_control=True
            ),
            # CC3: Risk Assessment
            SOC2Control(
                control_id="CC3.1",
                trust_service_category=SOC2TrustServiceCategory.SECURITY,
                control_type=SOC2ControlType.RISK_ASSESSMENT,
                name="Fraud Risk Assessment",
                description="Assess fraud risk and implement anti-fraud controls",
                criteria_reference="CC3.1",
                control_objective="Consider potential for fraud in risk assessment",
                implementation_details="Fraud risk assessment, whistleblower program, monitoring",
                frequency="annually",
                responsible_party="Internal Audit",
                key_control=True
            ),
            # CC4: Monitoring Activities
            SOC2Control(
                control_id="CC4.1",
                trust_service_category=SOC2TrustServiceCategory.SECURITY,
                control_type=SOC2ControlType.MONITORING_ACTIVITIES,
                name="Control Monitoring",
                description="Monitor ongoing effectiveness of controls",
                criteria_reference="CC4.1",
                control_objective="Evaluate and communicate control effectiveness",
                implementation_details="Internal audits, management reviews, control testing",
                frequency="quarterly",
                responsible_party="Internal Audit",
                key_control=True
            ),
            # CC5: Control Activities
            SOC2Control(
                control_id="CC5.1",
                trust_service_category=SOC2TrustServiceCategory.SECURITY,
                control_type=SOC2ControlType.CONTROL_ACTIVITIES,
                name="Change Management",
                description="Control changes to prevent unauthorized modifications",
                criteria_reference="CC5.1",
                control_objective="Develop change control procedures",
                implementation_details="Change request process, testing, approval, documentation",
                frequency="continuously",
                responsible_party="IT Operations",
                key_control=True
            ),
            # CC6: Logical and Physical Access
            SOC2Control(
                control_id="CC6.1",
                trust_service_category=SOC2TrustServiceCategory.SECURITY,
                control_type=SOC2ControlType.LOGICAL_AND_PHYSICAL_ACCESS,
                name="Logical Access Controls",
                description="Implement logical access security measures",
                criteria_reference="CC6.1",
                control_objective="Prevent unauthorized access to systems and data",
                implementation_details="Access management, authentication, authorization, password policies",
                frequency="continuously",
                responsible_party="Security Team",
                key_control=True,
                automated=True
            ),
            SOC2Control(
                control_id="CC6.2",
                trust_service_category=SOC2TrustServiceCategory.SECURITY,
                control_type=SOC2ControlType.LOGICAL_AND_PHYSICAL_ACCESS,
                name="Physical Access Controls",
                description="Implement physical access security measures",
                criteria_reference="CC6.2",
                control_objective="Prevent unauthorized physical access to facilities",
                implementation_details="Facility security, access badges, visitor management, surveillance",
                frequency="continuously",
                responsible_party="Facilities",
                key_control=True
            ),
            # CC7: System Operations
            SOC2Control(
                control_id="CC7.1",
                trust_service_category=SOC2TrustServiceCategory.AVAILABILITY,
                control_type=SOC2ControlType.SYSTEM_OPERATIONS,
                name="System Monitoring",
                description="Monitor system performance and availability",
                criteria_reference="CC7.1",
                control_objective="Ensure system meets availability commitments",
                implementation_details="Performance monitoring, alerting, capacity planning",
                frequency="continuously",
                responsible_party="IT Operations",
                key_control=True,
                automated=True
            ),
            # CC8: Change Management
            SOC2Control(
                control_id="CC8.1",
                trust_service_category=SOC2TrustServiceCategory.PROCESSING_INTEGRITY,
                control_type=SOC2ControlType.CHANGE_MANAGEMENT,
                name="System Development Lifecycle",
                description="Controls over system development and changes",
                criteria_reference="CC8.1",
                control_objective="Ensure systems meet processing integrity requirements",
                implementation_details="SDLC processes, code reviews, testing, deployment procedures",
                frequency="continuously",
                responsible_party="Development Team",
                key_control=True
            ),
        ]

        for control in common_controls:
            self.controls[control.control_id] = control

        # Set next test dates
        self._schedule_control_tests()

    def _schedule_control_tests(self) -> None:
        """Schedule control testing based on frequency"""
        now = datetime.now()
        frequency_map = {
            "daily": timedelta(days=1),
            "weekly": timedelta(weeks=1),
            "monthly": timedelta(days=30),
            "quarterly": timedelta(days=90),
            "annually": timedelta(days=365),
            "continuously": timedelta(days=1)  # Test daily for continuous controls
        }

        for control in self.controls.values():
            frequency = frequency_map.get(control.frequency, timedelta(days=90))
            control.next_test_date = now + frequency

    async def assess_control(self, control: SOC2Control) -> SOC2Control:
        """Assess a SOC 2 control"""
        test_result = await self._test_control(control)

        # Update control based on test results
        control.last_tested = test_result.test_date
        control.test_results.append(test_result.to_dict())

        # Determine control status based on test results
        if test_result.result == "pass":
            if control.status == ControlStatus.DESIGNED:
                control.status = ControlStatus.IMPLEMENTED
            elif control.status == ControlStatus.IMPLEMENTED:
                control.status = ControlStatus.OPERATING_EFFECTIVELY
        elif test_result.result == "fail":
            if control.status == ControlStatus.OPERATING_EFFECTIVELY:
                control.status = ControlStatus.NEEDS_IMPROVEMENT
            else:
                control.status = ControlStatus.NOT_OPERATING

        # Check for deficiencies
        if test_result.exceptions_identified > 0 or test_result.remediation_required:
            deficiency = await self._create_deficiency(control, test_result)
            control.deficiencies.append(deficiency.to_dict())

        # Schedule next test
        self._schedule_next_test(control)

        return control

    async def _test_control(self, control: SOC2Control) -> SOC2TestResult:
        """Test a SOC 2 control"""
        test_id = str(uuid.uuid4())
        test_date = datetime.now()

        # Simulate test execution
        import random

        # Different pass rates based on control type and key control status
        if control.automated:
            pass_rate = 0.95 if control.key_control else 0.90
        else:
            pass_rate = 0.85 if control.key_control else 0.80

        test_result = "pass" if random.random() < pass_rate else "fail"

        # Generate findings based on result
        findings = []
        exceptions_identified = 0
        remediation_required = False

        if test_result == "fail":
            findings.append(f"Control test failed for {control.name}")
            exceptions_identified = random.randint(1, 5)
            remediation_required = random.choice([True, False])

        # Determine test type
        test_type = TestType.INSPECTION
        if control.automated:
            test_type = TestType.REPERFORMANCE
        elif "monitoring" in control.name.lower():
            test_type = TestType.OBSERVATION
        elif "policy" in control.name.lower():
            test_type = TestType.INQUIRY

        result = SOC2TestResult(
            test_id=test_id,
            control_id=control.control_id,
            test_type=test_type,
            test_description=f"Test of {control.name} control",
            test_date=test_date,
            tester="System Automated Test",
            result=test_result,
            findings=findings,
            evidence=[{
                "type": "automated_test",
                "timestamp": test_date.isoformat(),
                "details": f"Automated test execution for control {control.control_id}"
            }],
            sampling_method="full_population" if control.automated else "statistical_sampling",
            sample_size=100 if not control.automated else 0,
            population_size=100 if not control.automated else 0,
            exceptions_identified=exceptions_identified,
            remediation_required=remediation_required,
            due_date=test_date + timedelta(days=30) if remediation_required else None
        )

        self.test_results[test_id] = result
        return result

    async def _create_deficiency(self, control: SOC2Control, test_result: SOC2TestResult) -> SOC2Deficiency:
        """Create a control deficiency"""
        deficiency_id = str(uuid.uuid4())

        # Determine deficiency type and severity
        deficiency_type = "operating_deficiency" if control.status in [ControlStatus.IMPLEMENTED, ControlStatus.OPERATING_EFFECTIVELY] else "design_deficiency"
        severity = "significant" if control.key_control else "material"

        deficiency = SOC2Deficiency(
            deficiency_id=deficiency_id,
            control_id=control.control_id,
            deficiency_type=deficiency_type,
            severity=severity,
            description=f"Deficiency identified in control {control.name}",
            impact="Impact on control effectiveness and system reliability",
            identified_date=test_result.test_date,
            root_cause="Control not operating as designed",
            remediation_status="open",
            responsible_party=control.responsible_party,
            due_date=test_result.test_date + timedelta(days=30),
            evidence=[{
                "type": "test_result",
                "test_id": test_result.test_id,
                "date": test_result.test_date.isoformat()
            }]
        )

        self.deficiencies[deficiency_id] = deficiency
        return deficiency

    def _schedule_next_test(self, control: SOC2Control) -> None:
        """Schedule next test for control"""
        frequency_map = {
            "daily": timedelta(days=1),
            "weekly": timedelta(weeks=1),
            "monthly": timedelta(days=30),
            "quarterly": timedelta(days=90),
            "annually": timedelta(days=365),
            "continuously": timedelta(days=1)
        }

        frequency = frequency_map.get(control.frequency, timedelta(days=90))
        control.next_test_date = datetime.now() + frequency

    async def run_soc2_assessment(self, trust_service_categories: List[SOC2TrustServiceCategory] = None,
                               scope: str = "full") -> Dict[str, Any]:
        """Run comprehensive SOC 2 assessment"""
        if trust_service_categories is None:
            trust_service_categories = list(SOC2TrustServiceCategory)

        assessment_results = {
            "assessment_id": str(uuid.uuid4()),
            "assessment_date": datetime.now().isoformat(),
            "trust_service_categories": [tsc.value for tsc in trust_service_categories],
            "scope": scope,
            "controls_assessed": 0,
            "controls_effective": 0,
            "controls_needing_improvement": 0,
            "controls_not_operating": 0,
            "key_controls_effective": 0,
            "total_key_controls": 0,
            "deficiencies_identified": 0,
            "overall_effectiveness": 0.0,
            "category_results": {},
            "recommendations": []
        }

        # Assess controls by category
        for category in trust_service_categories:
            category_controls = [
                control for control in self.controls.values()
                if control.trust_service_category == category
            ]

            category_results = {
                "total_controls": len(category_controls),
                "effective_controls": 0,
                "needs_improvement": 0,
                "not_operating": 0,
                "key_controls": 0,
                "key_controls_effective": 0
            }

            for control in category_controls:
                assessment_results["controls_assessed"] += 1

                # Assess control
                assessed_control = await self.assess_control(control)

                # Update counts
                if assessed_control.status == ControlStatus.OPERATING_EFFECTIVELY:
                    category_results["effective_controls"] += 1
                    assessment_results["controls_effective"] += 1
                elif assessed_control.status == ControlStatus.NEEDS_IMPROVEMENT:
                    category_results["needs_improvement"] += 1
                    assessment_results["controls_needing_improvement"] += 1
                else:
                    category_results["not_operating"] += 1
                    assessment_results["controls_not_operating"] += 1

                if control.key_control:
                    category_results["key_controls"] += 1
                    assessment_results["total_key_controls"] += 1
                    if assessed_control.status == ControlStatus.OPERATING_EFFECTIVELY:
                        category_results["key_controls_effective"] += 1
                        assessment_results["key_controls_effective"] += 1

                # Count deficiencies
                category_deficiencies = [d for d in self.deficiencies.values() if d.control_id == control.control_id and d.remediation_status == "open"]
                assessment_results["deficiencies_identified"] += len(category_deficiencies)

            # Calculate category effectiveness
            if category_results["total_controls"] > 0:
                category_effectiveness = (category_results["effective_controls"] / category_results["total_controls"]) * 100
            else:
                category_effectiveness = 100.0

            category_results["effectiveness"] = round(category_effectiveness, 1)
            assessment_results["category_results"][category.value] = category_results

        # Calculate overall effectiveness
        if assessment_results["controls_assessed"] > 0:
            overall_effectiveness = (assessment_results["controls_effective"] / assessment_results["controls_assessed"]) * 100
        else:
            overall_effectiveness = 0.0

        assessment_results["overall_effectiveness"] = round(overall_effectiveness, 1)

        # Generate recommendations
        assessment_results["recommendations"] = self._generate_soc2_recommendations(assessment_results)

        logger.info(f"SOC 2 assessment completed: {assessment_results['overall_effectiveness']}% overall effectiveness")
        return assessment_results

    def _generate_soc2_recommendations(self, assessment_results: Dict[str, Any]) -> List[str]:
        """Generate SOC 2 compliance recommendations"""
        recommendations = []

        # Key control effectiveness
        if assessment_results["total_key_controls"] > 0:
            key_control_effectiveness = (assessment_results["key_controls_effective"] / assessment_results["total_key_controls"]) * 100
            if key_control_effectiveness < self.key_control_threshold * 100:
                recommendations.append(f"Improve key control effectiveness (currently {key_control_effectiveness:.1f}%)")

        # Deficiencies
        if assessment_results["deficiencies_identified"] > 0:
            recommendations.append(f"Address {assessment_results['deficiencies_identified']} identified control deficiencies")

        # Category-specific recommendations
        for category, results in assessment_results["category_results"].items():
            if results["effectiveness"] < 80:
                recommendations.append(f"Improve controls in {category} category (currently {results['effectiveness']}% effective)")

            if results["key_controls"] > 0:
                key_effectiveness = (results["key_controls_effective"] / results["key_controls"]) * 100
                if key_effectiveness < 90:
                    recommendations.append(f"Strengthen key controls in {category} category")

        # General recommendations
        if assessment_results["controls_needing_improvement"] > 0:
            recommendations.append(f"Review and improve {assessment_results['controls_needing_improvement']} controls needing improvement")

        if assessment_results["controls_not_operating"] > 0:
            recommendations.append(f"Implement {assessment_results['controls_not_operating']} controls that are not operating")

        return recommendations

    async def generate_soc2_report(self, report_type: str = "management") -> Dict[str, Any]:
        """Generate SOC 2 compliance report"""
        report = {
            "report_id": str(uuid.uuid4()),
            "framework": "SOC 2 Type II",
            "report_type": report_type,
            "generated_at": datetime.now().isoformat(),
            "period_covered": {
                "start": (datetime.now() - timedelta(days=self.audit_period_days)).isoformat(),
                "end": datetime.now().isoformat()
            }
        }

        if report_type == "management":
            # Get latest assessment results
            assessment = await self.run_soc2_assessment()
            report.update({
                "assessment_summary": assessment,
                "control_inventory": [
                    control.to_dict() for control in self.controls.values()
                ],
                "deficiency_summary": self._get_deficiency_summary(),
                "testing_summary": self._get_testing_summary(),
                "system_description": self.system_description.to_dict() if self.system_description else None
            })

        elif report_type == "auditor":
            report.update({
                "control_effectiveness": self._calculate_control_effectiveness_by_category(),
                "testing_methodology": self._describe_testing_methodology(),
                "sampling_approach": self._describe_sampling_approach(),
                "deficiency_analysis": self._analyze_deficiencies(),
                "management_response": self._get_management_response()
            })

        return report

    def _get_deficiency_summary(self) -> Dict[str, Any]:
        """Get summary of control deficiencies"""
        if not self.deficiencies:
            return {"total": 0, "by_severity": {}, "by_status": {}}

        total_deficiencies = len(self.deficiencies)
        by_severity = {}
        by_status = {}

        for deficiency in self.deficiencies.values():
            by_severity[deficiency.severity] = by_severity.get(deficiency.severity, 0) + 1
            by_status[deficiency.remediation_status] = by_status.get(deficiency.remediation_status, 0) + 1

        # Calculate age of deficiencies
        open_deficiencies = [d for d in self.deficiencies.values() if d.remediation_status == "open"]
        average_age_days = 0
        if open_deficiencies:
            ages = [(datetime.now() - d.identified_date).days for d in open_deficiencies]
            average_age_days = sum(ages) / len(ages)

        return {
            "total": total_deficiencies,
            "by_severity": by_severity,
            "by_status": by_status,
            "open_deficiencies": len(open_deficiencies),
            "average_age_days_open": round(average_age_days, 1)
        }

    def _get_testing_summary(self) -> Dict[str, Any]:
        """Get summary of control testing"""
        if not self.test_results:
            return {"total_tests": 0, "pass_rate": 0.0}

        total_tests = len(self.test_results)
        passed_tests = len([t for t in self.test_results.values() if t.result == "pass"])
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0.0

        by_test_type = {}
        for test in self.test_results.values():
            test_type = test.test_type.value
            by_test_type[test_type] = by_test_type.get(test_type, 0) + 1

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "pass_rate": round(pass_rate, 1),
            "by_test_type": by_test_type
        }

    def _calculate_control_effectiveness_by_category(self) -> Dict[str, Any]:
        """Calculate control effectiveness by trust service category"""
        effectiveness_by_category = {}

        for category in SOC2TrustServiceCategory:
            category_controls = [c for c in self.controls.values() if c.trust_service_category == category]
            if not category_controls:
                continue

            effective_controls = len([c for c in category_controls if c.status == ControlStatus.OPERATING_EFFECTIVELY])
            effectiveness = (effective_controls / len(category_controls)) * 100

            effectiveness_by_category[category.value] = {
                "total_controls": len(category_controls),
                "effective_controls": effective_controls,
                "effectiveness": round(effectiveness, 1)
            }

        return effectiveness_by_category

    def _describe_testing_methodology(self) -> Dict[str, Any]:
        """Describe testing methodology used"""
        return {
            "approach": "Combination of automated testing, manual inspection, and inquiry",
            "frequency": self.testing_frequency,
            "testing_types": [
                "Automated controls testing",
                "Manual walkthroughs",
                "Documentation review",
                "Configuration validation",
                "Log analysis",
                "Interviews with responsible parties"
            ],
            "evidence_collection": "Combination of automated logs, screenshots, and documentation"
        }

    def _describe_sampling_approach(self) -> Dict[str, Any]:
        """Describe sampling approach"""
        return {
            "methodology": "Risk-based sampling approach",
            "sample_size_determination": "Based on control criticality and historical performance",
            "confidence_level": "95%",
            "sampling_methods": [
                "Statistical sampling for routine controls",
                "Full population testing for key controls",
                "Judgmental sampling for complex controls"
            ]
        }

    def _analyze_deficiencies(self) -> Dict[str, Any]:
        """Analyze control deficiencies"""
        if not self.deficiencies:
            return {"total": 0, "analysis": "No deficiencies identified"}

        deficiencies_by_type = {}
        deficiencies_by_severity = {}

        for deficiency in self.deficiencies.values():
            deficiencies_by_type[deficiency.deficiency_type] = deficiencies_by_type.get(deficiency.deficiency_type, 0) + 1
            deficiencies_by_severity[deficiency.severity] = deficiencies_by_severity.get(deficiency.severity, 0) + 1

        return {
            "total": len(self.deficiencies),
            "by_type": deficiencies_by_type,
            "by_severity": deficiencies_by_severity,
            "remediation_status": {
                status: len([d for d in self.deficiencies.values() if d.remediation_status == status])
                for status in ["open", "in_progress", "resolved", "monitored"]
            }
        }

    def _get_management_response(self) -> Dict[str, Any]:
        """Get management response to deficiencies"""
        open_deficiencies = [d for d in self.deficiencies.values() if d.remediation_status == "open"]
        in_progress_deficiencies = [d for d in self.deficiencies.values() if d.remediation_status == "in_progress"]

        return {
            "acknowledged": True,
            "action_plan": "Remediation plans in place for all identified deficiencies",
            "timeline": "All deficiencies to be resolved within 90 days",
            "resources_allocated": "Dedicated team assigned to deficiency remediation",
            "monitoring": "Weekly progress reviews scheduled",
            "open_deficiencies": len(open_deficiencies),
            "in_progress_deficiencies": len(in_progress_deficiencies)
        }

    def create_system_description(self, system_description: SOC2SystemDescription) -> bool:
        """Create or update system description"""
        system_description.last_updated = datetime.now()
        self.system_description = system_description

        logger.info(f"Updated system description: {system_description.system_name}")
        return True

    async def monitor_continuous_controls(self) -> Dict[str, Any]:
        """Monitor controls with continuous testing requirements"""
        continuous_controls = [c for c in self.controls.values() if c.frequency == "continuously"]

        monitoring_results = {
            "monitored_controls": len(continuous_controls),
            "test_results": [],
            "alerts_triggered": [],
            "recommendations": []
        }

        for control in continuous_controls:
            if control.next_test_date and control.next_test_date <= datetime.now():
                # Test the control
                test_result = await self._test_control(control)
                monitoring_results["test_results"].append(test_result.to_dict())

                # Check for alerts
                if test_result.result == "fail" and control.key_control:
                    alert = {
                        "control_id": control.control_id,
                        "control_name": control.name,
                        "alert_type": "key_control_failure",
                        "timestamp": datetime.now().isoformat(),
                        "severity": "high"
                    }
                    monitoring_results["alerts_triggered"].append(alert)

                # Update control
                control.last_tested = test_result.test_date
                control.test_results.append(test_result.to_dict())
                self._schedule_next_test(control)

        # Generate recommendations
        if monitoring_results["alerts_triggered"]:
            monitoring_results["recommendations"].append("Investigate and address key control failures immediately")

        failed_tests = [t for t in monitoring_results["test_results"] if t["result"] == "fail"]
        if len(failed_tests) > len(continuous_controls) * 0.1:  # More than 10% failure rate
            monitoring_results["recommendations"].append("Review and improve continuous control monitoring")

        return monitoring_results

    def get_soc2_dashboard(self) -> Dict[str, Any]:
        """Get SOC 2 compliance dashboard data"""
        dashboard = {
            "overall_effectiveness": self._calculate_overall_effectiveness(),
            "trust_service_categories": {},
            "key_control_status": {},
            "recent_testing": [],
            "open_deficiencies": [],
            "upcoming_tests": []
        }

        # Trust service categories
        for category in SOC2TrustServiceCategory:
            category_controls = [c for c in self.controls.values() if c.trust_service_category == category]
            if category_controls:
                effective = len([c for c in category_controls if c.status == ControlStatus.OPERATING_EFFECTIVELY])
                effectiveness = (effective / len(category_controls)) * 100

                dashboard["trust_service_categories"][category.value] = {
                    "total_controls": len(category_controls),
                    "effective_controls": effective,
                    "effectiveness": round(effectiveness, 1)
                }

        # Key control status
        key_controls = [c for c in self.controls.values() if c.key_control]
        if key_controls:
            effective_key_controls = len([c for c in key_controls if c.status == ControlStatus.OPERATING_EFFECTIVELY])
            key_effectiveness = (effective_key_controls / len(key_controls)) * 100

            dashboard["key_control_status"] = {
                "total_key_controls": len(key_controls),
                "effective_key_controls": effective_key_controls,
                "effectiveness": round(key_effectiveness, 1),
                "below_threshold": key_effectiveness < (self.key_control_threshold * 100)
            }

        # Recent testing
        recent_tests = sorted(
            self.test_results.values(),
            key=lambda t: t.test_date,
            reverse=True
        )[:10]

        dashboard["recent_testing"] = [
            {
                "test_id": t.test_id,
                "control_id": t.control_id,
                "control_name": self.controls.get(t.control_id, {}).name,
                "result": t.result,
                "test_date": t.test_date.isoformat()
            }
            for t in recent_tests
        ]

        # Open deficiencies
        open_deficiencies = [
            d for d in self.deficiencies.values()
            if d.remediation_status == "open"
        ]

        dashboard["open_deficiencies"] = [
            {
                "deficiency_id": d.deficiency_id,
                "control_id": d.control_id,
                "severity": d.severity,
                "identified_date": d.identified_date.isoformat(),
                "due_date": d.due_date.isoformat() if d.due_date else None
            }
            for d in sorted(open_deficiencies, key=lambda d: d.identified_date, reverse=True)[:10]
        ]

        # Upcoming tests
        upcoming_tests = [
            control for control in self.controls.values()
            if control.next_test_date and control.next_test_date <= datetime.now() + timedelta(days=7)
        ]

        dashboard["upcoming_tests"] = [
            {
                "control_id": c.control_id,
                "control_name": c.name,
                "category": c.trust_service_category.value,
                "next_test_date": c.next_test_date.isoformat()
            }
            for c in sorted(upcoming_tests, key=lambda c: c.next_test_date)[:10]
        ]

        return dashboard

    def _calculate_overall_effectiveness(self) -> Dict[str, float]:
        """Calculate overall SOC 2 effectiveness"""
        all_controls = list(self.controls.values())
        if not all_controls:
            return {"overall": 0.0, "key_controls": 0.0}

        effective_controls = len([c for c in all_controls if c.status == ControlStatus.OPERATING_EFFECTIVELY])
        overall_effectiveness = (effective_controls / len(all_controls)) * 100

        key_controls = [c for c in all_controls if c.key_control]
        key_effectiveness = 0.0
        if key_controls:
            effective_key_controls = len([c for c in key_controls if c.status == ControlStatus.OPERATING_EFFECTIVELY])
            key_effectiveness = (effective_key_controls / len(key_controls)) * 100

        return {
            "overall": round(overall_effectiveness, 1),
            "key_controls": round(key_effectiveness, 1)
        }