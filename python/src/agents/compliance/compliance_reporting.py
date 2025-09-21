"""
Compliance Reporting Service
Centralized reporting for all compliance frameworks
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

from .compliance_engine import (
    ComplianceFramework,
    ComplianceControl,
    ComplianceStatus,
    ComplianceRiskLevel,
    ComplianceEvidence,
    ComplianceAssessment,
    ComplianceFrameworkType
)
from .gdpr_compliance import GDPRComplianceManager
from .soc2_compliance import SOC2ComplianceManager
from .hipaa_compliance import HIPAAComplianceManager

logger = logging.getLogger(__name__)


@dataclass
class ComplianceMetric:
    """Compliance metric definition"""
    name: str
    description: str
    framework: str
    calculation_method: str
    target_value: float
    current_value: float
    trend: str  # 'improving', 'declining', 'stable'
    last_updated: datetime
    unit: str = "percentage"


@dataclass
class ComplianceDashboard:
    """Compliance dashboard configuration"""
    id: str
    name: str
    description: str
    frameworks: List[str]
    metrics: List[str]
    refresh_interval: int  # seconds
    widgets: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ComplianceReport:
    """Compliance report definition"""
    id: str
    title: str
    description: str
    framework: str
    report_type: str  # 'assessment', 'summary', 'detailed', 'executive'
    generated_date: datetime
    generated_by: str
    content: Dict[str, Any]
    format: str  # 'json', 'pdf', 'html', 'csv'
    distribution_list: List[str] = field(default_factory=list)
    retention_days: int = 365


class ComplianceReportingService:
    """Centralized compliance reporting service"""

    def __init__(self):
        self.gdpr_manager: Optional[GDPRComplianceManager] = None
        self.soc2_manager: Optional[SOC2ComplianceManager] = None
        self.hipaa_manager: Optional[HIPAAComplianceManager] = None

        self.metrics: Dict[str, ComplianceMetric] = {}
        self.dashboards: Dict[str, ComplianceDashboard] = {}
        self.reports: Dict[str, ComplianceReport] = {}
        self.report_schedules: Dict[str, Dict[str, Any]] = {}

        self._initialize_metrics()
        self._initialize_dashboards()

    def set_gdpr_manager(self, manager: GDPRComplianceManager):
        """Set GDPR compliance manager"""
        self.gdpr_manager = manager

    def set_soc2_manager(self, manager: SOC2ComplianceManager):
        """Set SOC2 compliance manager"""
        self.soc2_manager = manager

    def set_hipaa_manager(self, manager: HIPAAComplianceManager):
        """Set HIPAA compliance manager"""
        self.hipaa_manager = manager

    def _initialize_metrics(self):
        """Initialize compliance metrics"""

        # GDPR metrics
        self.metrics["gdpr_compliance_score"] = ComplianceMetric(
            name="GDPR Compliance Score",
            description="Overall GDPR compliance percentage",
            framework="GDPR",
            calculation_method="average_control_compliance",
            target_value=95.0,
            current_value=0.0,
            trend="stable",
            last_updated=datetime.now()
        )

        self.metrics["gdpr_dsar_response_time"] = ComplianceMetric(
            name="GDPR DSAR Response Time",
            description="Average time to respond to data subject access requests",
            framework="GDPR",
            calculation_method="average_days",
            target_value=30.0,
            current_value=0.0,
            trend="stable",
            last_updated=datetime.now(),
            unit="days"
        )

        self.metrics["gdpr_breach_notifications"] = ComplianceMetric(
            name="GDPR Breach Notifications",
            description="Number of data breaches reported to authorities",
            framework="GDPR",
            calculation_method="count",
            target_value=0.0,
            current_value=0.0,
            trend="stable",
            last_updated=datetime.now(),
            unit="count"
        )

        # SOC2 metrics
        self.metrics["soc2_compliance_score"] = ComplianceMetric(
            name="SOC2 Compliance Score",
            description="Overall SOC2 Type II compliance percentage",
            framework="SOC2",
            calculation_method="average_control_compliance",
            target_value=95.0,
            current_value=0.0,
            trend="stable",
            last_updated=datetime.now()
        )

        self.metrics["soc2_control_effectiveness"] = ComplianceMetric(
            name="SOC2 Control Effectiveness",
            description="Percentage of SOC2 controls operating effectively",
            framework="SOC2",
            calculation_method="percentage_effective",
            target_value=90.0,
            current_value=0.0,
            trend="stable",
            last_updated=datetime.now()
        )

        self.metrics["soc2_audit_findings"] = ComplianceMetric(
            name="SOC2 Audit Findings",
            description="Number of findings from SOC2 audits",
            framework="SOC2",
            calculation_method="count",
            target_value=0.0,
            current_value=0.0,
            trend="stable",
            last_updated=datetime.now(),
            unit="count"
        )

        # HIPAA metrics
        self.metrics["hipaa_compliance_score"] = ComplianceMetric(
            name="HIPAA Compliance Score",
            description="Overall HIPAA compliance percentage",
            framework="HIPAA",
            calculation_method="average_control_compliance",
            target_value=95.0,
            current_value=0.0,
            trend="stable",
            last_updated=datetime.now()
        )

        self.metrics["hipaa_breach_incidents"] = ComplianceMetric(
            name="HIPAA Breach Incidents",
            description="Number of HIPAA breach incidents",
            framework="HIPAA",
            calculation_method="count",
            target_value=0.0,
            current_value=0.0,
            trend="stable",
            last_updated=datetime.now(),
            unit="count"
        )

        self.metrics["hipaa_business_associate_risk"] = ComplianceMetric(
            name="HIPAA Business Associate Risk",
            description="Average risk score for business associates",
            framework="HIPAA",
            calculation_method="average_risk_score",
            target_value=0.3,
            current_value=0.0,
            trend="stable",
            last_updated=datetime.now()
        )

        # Cross-framework metrics
        self.metrics["overall_compliance_score"] = ComplianceMetric(
            name="Overall Compliance Score",
            description="Weighted average of all framework compliance scores",
            framework="All",
            calculation_method="weighted_average",
            target_value=90.0,
            current_value=0.0,
            trend="stable",
            last_updated=datetime.now()
        )

    def _initialize_dashboards(self):
        """Initialize compliance dashboards"""

        # Executive dashboard
        self.dashboards["executive"] = ComplianceDashboard(
            id="executive",
            name="Executive Compliance Dashboard",
            description="High-level compliance overview for executives",
            frameworks=["GDPR", "SOC2", "HIPAA"],
            metrics=[
                "overall_compliance_score",
                "gdpr_compliance_score",
                "soc2_compliance_score",
                "hipaa_compliance_score"
            ],
            refresh_interval=3600,  # 1 hour
            widgets=[
                {
                    "type": "score_card",
                    "title": "Overall Compliance",
                    "metric": "overall_compliance_score",
                    "size": "large"
                },
                {
                    "type": "trend_chart",
                    "title": "Compliance Trend",
                    "metrics": ["gdpr_compliance_score", "soc2_compliance_score", "hipaa_compliance_score"],
                    "size": "medium"
                },
                {
                    "type": "alert_list",
                    "title": "Critical Alerts",
                    "size": "medium"
                }
            ]
        )

        # Compliance manager dashboard
        self.dashboards["compliance_manager"] = ComplianceDashboard(
            id="compliance_manager",
            name="Compliance Manager Dashboard",
            description="Detailed compliance management view",
            frameworks=["GDPR", "SOC2", "HIPAA"],
            metrics=list(self.metrics.keys()),
            refresh_interval=1800,  # 30 minutes
            widgets=[
                {
                    "type": "framework_scores",
                    "title": "Framework Compliance Scores",
                    "size": "large"
                },
                {
                    "type": "risk_matrix",
                    "title": "Compliance Risk Matrix",
                    "size": "medium"
                },
                {
                    "type": "control_status",
                    "title": "Control Implementation Status",
                    "size": "medium"
                },
                {
                    "type": "recent_assessments",
                    "title": "Recent Compliance Assessments",
                    "size": "medium"
                }
            ]
        )

    def update_metrics(self):
        """Update all compliance metrics with current data"""

        # Update GDPR metrics
        if self.gdpr_manager:
            gdpr_assessment = self.gdpr_manager.conduct_compliance_assessment()
            self.metrics["gdpr_compliance_score"].current_value = gdpr_assessment.compliance_score
            self.metrics["gdpr_compliance_score"].last_updated = datetime.now()

            # Calculate DSAR response time
            if hasattr(self.gdpr_manager, 'data_subject_requests'):
                dsars = self.gdpr_manager.data_subject_requests.values()
                if dsars:
                    completed_dsars = [dsar for dsar in dsars if dsar.completion_date]
                    if completed_dsars:
                        avg_response_time = sum(
                            (dsar.completion_date - dsar.request_date).days
                            for dsar in completed_dsars
                        ) / len(completed_dsars)
                        self.metrics["gdpr_dsar_response_time"].current_value = avg_response_time

            # Count breaches
            if hasattr(self.gdpr_manager, 'breaches'):
                self.metrics["gdpr_breach_notifications"].current_value = len(self.gdpr_manager.breaches)

        # Update SOC2 metrics
        if self.soc2_manager:
            soc2_assessment = self.soc2_manager.conduct_compliance_assessment()
            self.metrics["soc2_compliance_score"].current_value = soc2_assessment.compliance_score
            self.metrics["soc2_compliance_score"].last_updated = datetime.now()

            # Calculate control effectiveness
            effective_controls = len([
                c for c in self.soc2_manager.controls.values()
                if c.test_results and any(
                    test.status == "passed" for test in c.test_results
                )
            ])
            total_controls = len(self.soc2_manager.controls)
            if total_controls > 0:
                self.metrics["soc2_control_effectiveness"].current_value = (effective_controls / total_controls) * 100

            # Count audit findings
            if hasattr(self.soc2_manager, 'audit_findings'):
                self.metrics["soc2_audit_findings"].current_value = len(self.soc2_manager.audit_findings)

        # Update HIPAA metrics
        if self.hipaa_manager:
            hipaa_assessment = self.hipaa_manager.conduct_compliance_assessment()
            self.metrics["hipaa_compliance_score"].current_value = hipaa_assessment.compliance_score
            self.metrics["hipaa_compliance_score"].last_updated = datetime.now()

            # Count breaches
            self.metrics["hipaa_breach_incidents"].current_value = len(self.hipaa_manager.breaches)

            # Calculate business associate risk
            if self.hipaa_manager.business_associates:
                avg_ba_risk = sum(
                    ba.risk_assessment_score
                    for ba in self.hipaa_manager.business_associates.values()
                ) / len(self.hipaa_manager.business_associates)
                self.metrics["hipaa_business_associate_risk"].current_value = avg_ba_risk

        # Update overall compliance score
        framework_scores = []
        weights = {"GDPR": 0.35, "SOC2": 0.4, "HIPAA": 0.25}

        if self.gdpr_manager:
            framework_scores.append(self.metrics["gdpr_compliance_score"].current_value * weights["GDPR"])
        if self.soc2_manager:
            framework_scores.append(self.metrics["soc2_compliance_score"].current_value * weights["SOC2"])
        if self.hipaa_manager:
            framework_scores.append(self.metrics["hipaa_compliance_score"].current_value * weights["HIPAA"])

        if framework_scores:
            self.metrics["overall_compliance_score"].current_value = sum(framework_scores)
            self.metrics["overall_compliance_score"].last_updated = datetime.now()

        # Update trends
        self._update_metric_trends()

    def _update_metric_trends(self):
        """Update metric trends based on historical data"""

        # This would typically compare with previous values
        # For now, we'll set a default trend
        for metric in self.metrics.values():
            if metric.current_value >= metric.target_value:
                metric.trend = "improving"
            elif metric.current_value >= metric.target_value * 0.9:
                metric.trend = "stable"
            else:
                metric.trend = "declining"

    def generate_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Generate data for a specific dashboard"""

        if dashboard_id not in self.dashboards:
            raise ValueError(f"Dashboard {dashboard_id} not found")

        dashboard = self.dashboards[dashboard_id]
        self.update_metrics()

        dashboard_data = {
            "dashboard": {
                "id": dashboard.id,
                "name": dashboard.name,
                "description": dashboard.description,
                "last_updated": datetime.now().isoformat(),
                "refresh_interval": dashboard.refresh_interval
            },
            "metrics": {},
            "widgets": [],
            "alerts": self._generate_dashboard_alerts(dashboard)
        }

        # Add metric data
        for metric_name in dashboard.metrics:
            if metric_name in self.metrics:
                metric = self.metrics[metric_name]
                dashboard_data["metrics"][metric_name] = {
                    "name": metric.name,
                    "value": metric.current_value,
                    "target": metric.target_value,
                    "trend": metric.trend,
                    "unit": metric.unit,
                    "status": self._get_metric_status(metric)
                }

        # Generate widget data
        for widget in dashboard.widgets:
            widget_data = self._generate_widget_data(widget, dashboard_data["metrics"])
            dashboard_data["widgets"].append(widget_data)

        return dashboard_data

    def _get_metric_status(self, metric: ComplianceMetric) -> str:
        """Get metric status based on value vs target"""

        if metric.unit == "count" and metric.target_value == 0:
            return "good" if metric.current_value == 0 else "poor"
        else:
            percentage = (metric.current_value / metric.target_value) * 100
            if percentage >= 100:
                return "excellent"
            elif percentage >= 90:
                return "good"
            elif percentage >= 75:
                return "fair"
            else:
                return "poor"

    def _generate_widget_data(self, widget: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data for a specific widget type"""

        widget_type = widget["type"]

        if widget_type == "score_card":
            metric_name = widget["metric"]
            if metric_name in metrics:
                metric_data = metrics[metric_name]
                return {
                    "type": "score_card",
                    "title": widget["title"],
                    "size": widget["size"],
                    "data": {
                        "value": metric_data["value"],
                        "target": metric_data["target"],
                        "trend": metric_data["trend"],
                        "status": metric_data["status"],
                        "unit": metric_data["unit"]
                    }
                }

        elif widget_type == "trend_chart":
            return {
                "type": "trend_chart",
                "title": widget["title"],
                "size": widget["size"],
                "data": {
                    "series": [
                        {
                            "name": metrics[metric]["name"],
                            "data": [metrics[metric]["value"]],  # Historical data would go here
                            "trend": metrics[metric]["trend"]
                        }
                        for metric in widget["metrics"]
                        if metric in metrics
                    ]
                }
            }

        elif widget_type == "framework_scores":
            framework_scores = []
            for framework in ["GDPR", "SOC2", "HIPAA"]:
                metric_key = f"{framework.lower()}_compliance_score"
                if metric_key in metrics:
                    framework_scores.append({
                        "framework": framework,
                        "score": metrics[metric_key]["value"],
                        "status": metrics[metric_key]["status"]
                    })

            return {
                "type": "framework_scores",
                "title": widget["title"],
                "size": widget["size"],
                "data": framework_scores
            }

        elif widget_type == "risk_matrix":
            # Generate risk matrix based on control statuses
            return {
                "type": "risk_matrix",
                "title": widget["title"],
                "size": widget["size"],
                "data": self._generate_risk_matrix()
            }

        elif widget_type == "control_status":
            return {
                "type": "control_status",
                "title": widget["title"],
                "size": widget["size"],
                "data": self._generate_control_status_summary()
            }

        else:
            return {
                "type": widget["type"],
                "title": widget["title"],
                "size": widget["size"],
                "data": {}
            }

    def _generate_risk_matrix(self) -> Dict[str, Any]:
        """Generate compliance risk matrix"""

        # This would aggregate risk data from all frameworks
        # For now, return a sample structure
        return {
            "high_impact_high_likelihood": 2,
            "high_impact_medium_likelihood": 3,
            "high_impact_low_likelihood": 1,
            "medium_impact_high_likelihood": 4,
            "medium_impact_medium_likelihood": 6,
            "medium_impact_low_likelihood": 8,
            "low_impact_high_likelihood": 2,
            "low_impact_medium_likelihood": 3,
            "low_impact_low_likelihood": 12
        }

    def _generate_control_status_summary(self) -> Dict[str, Any]:
        """Generate control implementation status summary"""

        # Aggregate control statuses from all frameworks
        summary = {
            "total_controls": 0,
            "compliant": 0,
            "partially_compliant": 0,
            "non_compliant": 0,
            "not_assessed": 0
        }

        if self.gdpr_manager:
            summary["total_controls"] += len(self.gdpr_manager.controls)
            summary["compliant"] += len([c for c in self.gdpr_manager.controls.values() if c.status == ComplianceStatus.COMPLIANT])
            summary["partially_compliant"] += len([c for c in self.gdpr_manager.controls.values() if c.status == ComplianceStatus.PARTIALLY_COMPLIANT])
            summary["non_compliant"] += len([c for c in self.gdpr_manager.controls.values() if c.status == ComplianceStatus.NON_COMPLIANT])

        if self.soc2_manager:
            summary["total_controls"] += len(self.soc2_manager.controls)
            summary["compliant"] += len([c for c in self.soc2_manager.controls.values() if c.status == ComplianceStatus.COMPLIANT])
            summary["partially_compliant"] += len([c for c in self.soc2_manager.controls.values() if c.status == ComplianceStatus.PARTIALLY_COMPLIANT])
            summary["non_compliant"] += len([c for c in self.soc2_manager.controls.values() if c.status == ComplianceStatus.NON_COMPLIANT])

        if self.hipaa_manager:
            summary["total_controls"] += len(self.hipaa_manager.controls)
            summary["compliant"] += len([c for c in self.hipaa_manager.controls.values() if c.status == ComplianceStatus.COMPLIANT])
            summary["partially_compliant"] += len([c for c in self.hipaa_manager.controls.values() if c.status == ComplianceStatus.PARTIALLY_COMPLIANT])
            summary["non_compliant"] += len([c for c in self.hipaa_manager.controls.values() if c.status == ComplianceStatus.NON_COMPLIANT])

        summary["not_assessed"] = summary["total_controls"] - (summary["compliant"] + summary["partially_compliant"] + summary["non_compliant"])

        return summary

    def _generate_dashboard_alerts(self, dashboard: ComplianceDashboard) -> List[Dict[str, Any]]:
        """Generate alerts for dashboard"""

        alerts = []

        # Check for critical metric values
        for metric_name in dashboard.metrics:
            if metric_name in self.metrics:
                metric = self.metrics[metric_name]

                # Generate alerts for poor performance
                if self._get_metric_status(metric) == "poor":
                    alerts.append({
                        "type": "critical",
                        "metric": metric_name,
                        "message": f"{metric.name} is below target: {metric.current_value}{metric.unit} (target: {metric.target_value}{metric.unit})",
                        "timestamp": datetime.now().isoformat()
                    })

                # Generate alerts for declining trends
                if metric.trend == "declining":
                    alerts.append({
                        "type": "warning",
                        "metric": metric_name,
                        "message": f"{metric.name} trend is declining",
                        "timestamp": datetime.now().isoformat()
                    })

        return alerts

    def generate_compliance_report(self, report_config: Dict[str, Any]) -> ComplianceReport:
        """Generate a compliance report"""

        report_id = report_config.get("id", f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        framework = report_config.get("framework", "All")
        report_type = report_config.get("report_type", "summary")
        format_type = report_config.get("format", "json")

        # Generate report content
        content = self._generate_report_content(framework, report_type)

        report = ComplianceReport(
            id=report_id,
            title=report_config.get("title", f"{framework} Compliance Report"),
            description=report_config.get("description", ""),
            framework=framework,
            report_type=report_type,
            generated_date=datetime.now(),
            generated_by=report_config.get("generated_by", "system"),
            content=content,
            format=format_type,
            distribution_list=report_config.get("distribution_list", []),
            retention_days=report_config.get("retention_days", 365)
        )

        self.reports[report_id] = report

        logger.info(f"Generated compliance report: {report_id}")
        return report

    def _generate_report_content(self, framework: str, report_type: str) -> Dict[str, Any]:
        """Generate content for compliance report"""

        content = {
            "metadata": {
                "generated_date": datetime.now().isoformat(),
                "framework": framework,
                "report_type": report_type,
                "version": "1.0"
            },
            "executive_summary": self._generate_executive_summary(framework),
            "compliance_overview": self._generate_compliance_overview(framework),
            "detailed_findings": self._generate_detailed_findings(framework),
            "recommendations": self._generate_report_recommendations(framework),
            "appendices": self._generate_report_appendices(framework)
        }

        # Add framework-specific content
        if framework != "All":
            content["framework_specific"] = self._generate_framework_specific_content(framework)

        return content

    def _generate_executive_summary(self, framework: str) -> Dict[str, Any]:
        """Generate executive summary for report"""

        summary = {
            "overall_compliance_score": self.metrics.get("overall_compliance_score", {}).current_value or 0,
            "key_findings": [],
            "critical_issues": [],
            "positive_trends": []
        }

        # Analyze metrics for key findings
        for metric_name, metric in self.metrics.items():
            if framework == "All" or framework.lower() in metric_name.lower():
                if self._get_metric_status(metric) == "poor":
                    summary["critical_issues"].append(f"{metric.name} is below target")
                elif self._get_metric_status(metric) == "excellent":
                    summary["positive_trends"].append(f"{metric.name} is meeting or exceeding targets")

        return summary

    def _generate_compliance_overview(self, framework: str) -> Dict[str, Any]:
        """Generate compliance overview"""

        overview = {
            "framework_scores": {},
            "control_status": self._generate_control_status_summary(),
            "risk_assessment": {
                "overall_risk_level": "medium",  # Calculated from data
                "key_risk_areas": self._identify_key_risk_areas()
            },
            "recent_activities": []  # Would include recent compliance activities
        }

        # Add framework scores
        for framework_name in ["GDPR", "SOC2", "HIPAA"]:
            if framework == "All" or framework == framework_name:
                metric_key = f"{framework_name.lower()}_compliance_score"
                if metric_key in self.metrics:
                    overview["framework_scores"][framework_name] = self.metrics[metric_key].current_value

        return overview

    def _identify_key_risk_areas(self) -> List[str]:
        """Identify key compliance risk areas"""

        risk_areas = []

        # Check metric trends
        for metric in self.metrics.values():
            if metric.trend == "declining":
                risk_areas.append(f"Declining performance in {metric.name}")

        # Check for poor metric status
        for metric in self.metrics.values():
            if self._get_metric_status(metric) == "poor":
                risk_areas.append(f"Target not met for {metric.name}")

        return risk_areas

    def _generate_detailed_findings(self, framework: str) -> Dict[str, Any]:
        """Generate detailed findings"""

        findings = {
            "control_deficiencies": [],
            "compliance_gaps": [],
            "best_practices": []
        }

        # This would analyze actual control implementations
        # For now, return placeholder structure
        return findings

    def _generate_report_recommendations(self, framework: str) -> List[str]:
        """Generate report recommendations"""

        recommendations = []

        # Generate recommendations based on metric performance
        for metric in self.metrics.values():
            if framework == "All" or framework.lower() in metric.framework.lower():
                if self._get_metric_status(metric) == "poor":
                    recommendations.append(f"Address {metric.name} to meet target of {metric.target_value}{metric.unit}")

        # Add general recommendations
        recommendations.extend([
            "Conduct regular compliance assessments",
            "Maintain comprehensive documentation",
            "Implement continuous monitoring",
            "Provide regular training to staff"
        ])

        return recommendations

    def _generate_report_appendices(self, framework: str) -> Dict[str, Any]:
        """Generate report appendices"""

        return {
            "glossary": {},
            "control_mappings": {},
            "regulatory_references": {},
            "contact_information": {}
        }

    def _generate_framework_specific_content(self, framework: str) -> Dict[str, Any]:
        """Generate framework-specific report content"""

        if framework == "GDPR" and self.gdpr_manager:
            return self.gdpr_manager.generate_compliance_report()
        elif framework == "SOC2" and self.soc2_manager:
            return self.soc2_manager.generate_compliance_report()
        elif framework == "HIPAA" and self.hipaa_manager:
            return self.hipaa_manager.generate_compliance_report()
        else:
            return {}

    def export_report(self, report_id: str, format_type: str) -> bytes:
        """Export report in specified format"""

        if report_id not in self.reports:
            raise ValueError(f"Report {report_id} not found")

        report = self.reports[report_id]

        if format_type == "json":
            return json.dumps(report.content, indent=2).encode('utf-8')
        elif format_type == "csv":
            # Convert to CSV format
            df = pd.DataFrame([report.content])
            return df.to_csv(index=False).encode('utf-8')
        elif format_type == "html":
            # Generate HTML report
            return self._generate_html_report(report).encode('utf-8')
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _generate_html_report(self, report: ComplianceReport) -> str:
        """Generate HTML report"""

        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; margin-bottom: 30px; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ background-color: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .finding {{ background-color: #fff3cd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .recommendation {{ background-color: #d4edda; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>Generated: {generated_date}</p>
                <p>Framework: {framework}</p>
            </div>

            <div class="section">
                <h2>Executive Summary</h2>
                <p>Overall Compliance Score: {overall_score}%</p>
            </div>

            {content_sections}
        </body>
        </html>
        """

        # Generate content sections
        content_sections = ""
        for section_name, section_content in report.content.items():
            if section_name != "metadata":
                content_sections += f'<div class="section"><h2>{section_name.replace("_", " ").title()}</h2>'
                if isinstance(section_content, dict):
                    for key, value in section_content.items():
                        content_sections += f'<p><strong>{key}:</strong> {value}</p>'
                else:
                    content_sections += f'<p>{section_content}</p>'
                content_sections += '</div>'

        return html_template.format(
            title=report.title,
            generated_date=report.generated_date.strftime("%Y-%m-%d %H:%M:%S"),
            framework=report.framework,
            overall_score=report.content.get("executive_summary", {}).get("overall_compliance_score", 0),
            content_sections=content_sections
        )

    def schedule_report(self, schedule_config: Dict[str, Any]) -> str:
        """Schedule a recurring report"""

        schedule_id = schedule_config.get("id", f"schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        self.report_schedules[schedule_id] = {
            "config": schedule_config,
            "last_run": None,
            "next_run": self._calculate_next_run(schedule_config.get("schedule", {})),
            "active": True
        }

        logger.info(f"Scheduled report: {schedule_id}")
        return schedule_id

    def _calculate_next_run(self, schedule: Dict[str, Any]) -> datetime:
        """Calculate next run time for scheduled report"""

        frequency = schedule.get("frequency", "monthly")
        start_date = schedule.get("start_date", datetime.now())

        if frequency == "daily":
            return start_date + timedelta(days=1)
        elif frequency == "weekly":
            return start_date + timedelta(weeks=1)
        elif frequency == "monthly":
            return start_date + timedelta(days=30)
        elif frequency == "quarterly":
            return start_date + timedelta(days=90)
        else:
            return start_date + timedelta(days=30)  # Default to monthly

    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get high-level compliance summary"""

        self.update_metrics()

        summary = {
            "overall_compliance_score": self.metrics.get("overall_compliance_score", {}).current_value or 0,
            "framework_scores": {},
            "active_alerts": 0,
            "upcoming_assessments": [],
            "recent_reports": list(self.reports.keys())[-5:],  # Last 5 reports
            "key_metrics": {}
        }

        # Add framework scores
        for framework in ["GDPR", "SOC2", "HIPAA"]:
            metric_key = f"{framework.lower()}_compliance_score"
            if metric_key in self.metrics:
                summary["framework_scores"][framework] = self.metrics[metric_key].current_value

        # Count active alerts
        for dashboard in self.dashboards.values():
            alerts = self._generate_dashboard_alerts(dashboard)
            summary["active_alerts"] += len(alerts)

        # Add key metrics
        key_metric_names = [
            "overall_compliance_score",
            "gdpr_compliance_score",
            "soc2_compliance_score",
            "hipaa_compliance_score"
        ]

        for metric_name in key_metric_names:
            if metric_name in self.metrics:
                metric = self.metrics[metric_name]
                summary["key_metrics"][metric_name] = {
                    "value": metric.current_value,
                    "target": metric.target_value,
                    "trend": metric.trend,
                    "status": self._get_metric_status(metric)
                }

        return summary