"""
Report Builder Module
Automated report generation with templates and scheduling
"""

import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from string import Template
import markdown
import pdfkit
from jinja2 import Environment, FileSystemLoader, Template as JinjaTemplate

from .analytics_engine import AnalyticsEngine
from .visualization_engine import VisualizationEngine
from .dashboard_generator import DashboardGenerator


class ReportFormat(Enum):
    """Report output formats"""
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    DOCX = "docx"
    EXCEL = "excel"
    JSON = "json"
    CSV = "csv"
    XML = "xml"


class ReportFrequency(Enum):
    """Report generation frequency"""
    ONCE = "once"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    ON_DEMAND = "on_demand"
    CUSTOM = "custom"


class DataSourceType(Enum):
    """Types of data sources for reports"""
    METRICS = "metrics"
    LOGS = "logs"
    DATABASE = "database"
    API = "api"
    FILE = "file"
    DASHBOARD = "dashboard"
    CUSTOM = "custom"


@dataclass
class ReportSection:
    """Report section configuration"""
    section_id: str
    title: str
    type: str  # text, chart, table, metric, image
    data_source: Optional[str] = None
    query: Optional[str] = None
    template: Optional[str] = None
    content: Optional[Any] = None
    order: int = 0
    visible: bool = True
    page_break: bool = False
    styling: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportTemplate:
    """Report template configuration"""
    template_id: str
    name: str
    description: str
    category: str
    sections: List[ReportSection]
    header_template: Optional[str] = None
    footer_template: Optional[str] = None
    styling: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class ReportSchedule:
    """Report scheduling configuration"""
    schedule_id: str
    report_id: str
    frequency: ReportFrequency
    next_run: datetime
    last_run: Optional[datetime] = None
    recipients: List[str] = field(default_factory=list)
    formats: List[ReportFormat] = field(default_factory=list)
    enabled: bool = True
    custom_schedule: Optional[str] = None  # Cron expression
    retry_on_failure: bool = True
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    """Report configuration"""
    report_id: str
    name: str
    description: str
    template_id: Optional[str]
    owner: str
    created_at: datetime
    updated_at: datetime
    sections: List[ReportSection]
    parameters: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    schedule: Optional[ReportSchedule] = None
    tags: List[str] = field(default_factory=list)
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedReport:
    """Generated report instance"""
    instance_id: str
    report_id: str
    generated_at: datetime
    format: ReportFormat
    content: Union[str, bytes]
    file_path: Optional[str] = None
    size_bytes: int = 0
    generation_time_ms: int = 0
    parameters_used: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReportBuilder:
    """
    Advanced report generation and management system
    """
    
    def __init__(self, analytics_engine: AnalyticsEngine,
                 visualization_engine: VisualizationEngine,
                 dashboard_generator: DashboardGenerator):
        self.analytics_engine = analytics_engine
        self.visualization_engine = visualization_engine
        self.dashboard_generator = dashboard_generator
        self.reports: Dict[str, Report] = {}
        self.templates: Dict[str, ReportTemplate] = {}
        self.schedules: Dict[str, ReportSchedule] = {}
        self.generated_reports: Dict[str, GeneratedReport] = {}
        self.jinja_env = Environment(loader=FileSystemLoader('templates'))
        self._init_default_templates()
        self._start_scheduler()
    
    def _init_default_templates(self):
        """Initialize default report templates"""
        # Executive Summary Template
        self.templates["executive_summary"] = ReportTemplate(
            template_id="executive_summary",
            name="Executive Summary",
            description="High-level overview for executives",
            category="management",
            sections=[
                ReportSection(
                    section_id="header",
                    title="Executive Summary",
                    type="text",
                    template="# Executive Summary\n\n**Report Date:** {{date}}\n**Period:** {{period}}",
                    order=1
                ),
                ReportSection(
                    section_id="kpi_metrics",
                    title="Key Performance Indicators",
                    type="metric",
                    data_source="kpi.all",
                    order=2
                ),
                ReportSection(
                    section_id="performance_chart",
                    title="Performance Trends",
                    type="chart",
                    data_source="performance.trends",
                    order=3
                ),
                ReportSection(
                    section_id="issues",
                    title="Critical Issues",
                    type="table",
                    data_source="issues.critical",
                    order=4
                ),
                ReportSection(
                    section_id="recommendations",
                    title="Recommendations",
                    type="text",
                    template="## Recommendations\n\n{{recommendations}}",
                    order=5,
                    page_break=True
                )
            ],
            tags=["executive", "summary", "management"]
        )
        
        # Technical Report Template
        self.templates["technical_report"] = ReportTemplate(
            template_id="technical_report",
            name="Technical Report",
            description="Detailed technical analysis",
            category="technical",
            sections=[
                ReportSection(
                    section_id="header",
                    title="Technical Report",
                    type="text",
                    template="# Technical Report\n\n**Generated:** {{timestamp}}",
                    order=1
                ),
                ReportSection(
                    section_id="system_metrics",
                    title="System Metrics",
                    type="chart",
                    data_source="system.all_metrics",
                    order=2
                ),
                ReportSection(
                    section_id="error_analysis",
                    title="Error Analysis",
                    type="table",
                    data_source="errors.detailed",
                    order=3
                ),
                ReportSection(
                    section_id="performance_analysis",
                    title="Performance Analysis",
                    type="chart",
                    data_source="performance.detailed",
                    order=4
                ),
                ReportSection(
                    section_id="code_quality",
                    title="Code Quality Metrics",
                    type="table",
                    data_source="quality.metrics",
                    order=5
                ),
                ReportSection(
                    section_id="recommendations",
                    title="Technical Recommendations",
                    type="text",
                    template="## Technical Recommendations\n\n{{tech_recommendations}}",
                    order=6
                )
            ],
            tags=["technical", "detailed", "analysis"]
        )
        
        # Compliance Report Template
        self.templates["compliance_report"] = ReportTemplate(
            template_id="compliance_report",
            name="Compliance Report",
            description="Security and compliance audit report",
            category="compliance",
            sections=[
                ReportSection(
                    section_id="header",
                    title="Compliance Report",
                    type="text",
                    template="# Compliance & Security Report\n\n**Audit Date:** {{date}}",
                    order=1
                ),
                ReportSection(
                    section_id="compliance_status",
                    title="Compliance Status",
                    type="metric",
                    data_source="compliance.status",
                    order=2
                ),
                ReportSection(
                    section_id="security_issues",
                    title="Security Issues",
                    type="table",
                    data_source="security.issues",
                    order=3
                ),
                ReportSection(
                    section_id="audit_findings",
                    title="Audit Findings",
                    type="table",
                    data_source="audit.findings",
                    order=4
                ),
                ReportSection(
                    section_id="remediation",
                    title="Remediation Actions",
                    type="text",
                    template="## Required Remediation\n\n{{remediation_plan}}",
                    order=5
                )
            ],
            tags=["compliance", "security", "audit"]
        )
    
    def _start_scheduler(self):
        """Start the report scheduler"""
        asyncio.create_task(self._schedule_runner())
    
    async def _schedule_runner(self):
        """Background task to run scheduled reports"""
        while True:
            try:
                current_time = datetime.now()
                
                for schedule in self.schedules.values():
                    if not schedule.enabled:
                        continue
                    
                    if schedule.next_run <= current_time:
                        # Generate the report
                        asyncio.create_task(self._run_scheduled_report(schedule))
                        
                        # Calculate next run time
                        schedule.last_run = current_time
                        schedule.next_run = self._calculate_next_run(schedule)
                
                # Check every minute
                await asyncio.sleep(60)
                
            except Exception as e:
                print(f"Scheduler error: {e}")
                await asyncio.sleep(60)
    
    async def create_report(self, name: str, description: str, owner: str,
                          template_id: Optional[str] = None) -> Report:
        """Create a new report"""
        report_id = str(uuid.uuid4())
        
        sections = []
        if template_id and template_id in self.templates:
            template = self.templates[template_id]
            sections = template.sections.copy()
        
        report = Report(
            report_id=report_id,
            name=name,
            description=description,
            template_id=template_id,
            owner=owner,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            sections=sections
        )
        
        self.reports[report_id] = report
        return report
    
    async def generate_report(self, report_id: str, format: ReportFormat,
                            parameters: Optional[Dict[str, Any]] = None) -> GeneratedReport:
        """Generate a report"""
        if report_id not in self.reports:
            raise ValueError(f"Report {report_id} not found")
        
        start_time = datetime.now()
        report = self.reports[report_id]
        
        # Merge parameters
        effective_params = {**report.parameters, **(parameters or {})}
        
        # Build report content
        content_sections = []
        for section in report.sections:
            if not section.visible:
                continue
            
            section_content = await self._render_section(section, effective_params)
            content_sections.append(section_content)
        
        # Format the report
        formatted_content = await self._format_report(
            content_sections, format, report
        )
        
        # Create generated report record
        generation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        generated = GeneratedReport(
            instance_id=str(uuid.uuid4()),
            report_id=report_id,
            generated_at=datetime.now(),
            format=format,
            content=formatted_content,
            size_bytes=len(formatted_content) if isinstance(formatted_content, (str, bytes)) else 0,
            generation_time_ms=int(generation_time),
            parameters_used=effective_params
        )
        
        self.generated_reports[generated.instance_id] = generated
        return generated
    
    async def _render_section(self, section: ReportSection,
                            parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Render a report section"""
        rendered = {
            "title": section.title,
            "type": section.type,
            "content": None
        }
        
        if section.type == "text":
            if section.template:
                template = Template(section.template)
                rendered["content"] = template.safe_substitute(**parameters)
            elif section.content:
                rendered["content"] = section.content
        
        elif section.type == "chart":
            if section.data_source:
                # Fetch data and create chart
                data = await self._fetch_data(section.data_source, parameters)
                chart = await self.visualization_engine.create_chart(
                    "line",  # Should be configurable
                    data
                )
                rendered["content"] = chart
        
        elif section.type == "table":
            if section.data_source:
                data = await self._fetch_data(section.data_source, parameters)
                rendered["content"] = self._format_table(data)
        
        elif section.type == "metric":
            if section.data_source:
                metrics = await self.analytics_engine.get_metrics(
                    metric_names=[section.data_source]
                )
                rendered["content"] = metrics
        
        elif section.type == "image":
            rendered["content"] = section.content
        
        return rendered
    
    async def _format_report(self, sections: List[Dict[str, Any]],
                           format: ReportFormat, report: Report) -> Union[str, bytes]:
        """Format report in specified format"""
        if format == ReportFormat.JSON:
            return json.dumps(sections, indent=2)
        
        elif format == ReportFormat.MARKDOWN:
            md_content = []
            for section in sections:
                md_content.append(f"## {section['title']}\n")
                
                if section['type'] == 'text':
                    md_content.append(section['content'])
                elif section['type'] == 'table':
                    md_content.append(self._table_to_markdown(section['content']))
                elif section['type'] == 'metric':
                    md_content.append(self._metrics_to_markdown(section['content']))
                
                md_content.append("\n")
            
            return "\n".join(md_content)
        
        elif format == ReportFormat.HTML:
            html_sections = []
            for section in sections:
                html_sections.append(f"<h2>{section['title']}</h2>")
                
                if section['type'] == 'text':
                    html_sections.append(markdown.markdown(section['content']))
                elif section['type'] == 'table':
                    html_sections.append(self._table_to_html(section['content']))
                elif section['type'] == 'chart':
                    html_sections.append(f"<div class='chart'>{section['content']}</div>")
                elif section['type'] == 'metric':
                    html_sections.append(self._metrics_to_html(section['content']))
            
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>{{title}}</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    h2 { color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }
                    table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    .metric { display: inline-block; margin: 10px; padding: 20px; 
                             border: 1px solid #ddd; border-radius: 5px; }
                </style>
            </head>
            <body>
                <h1>{{title}}</h1>
                {{content}}
            </body>
            </html>
            """
            
            template = Template(html_template)
            return template.safe_substitute(
                title=report.name,
                content="\n".join(html_sections)
            )
        
        elif format == ReportFormat.PDF:
            # Generate HTML first, then convert to PDF
            html_content = await self._format_report(sections, ReportFormat.HTML, report)
            
            # Use pdfkit to convert HTML to PDF
            # Note: Requires wkhtmltopdf to be installed
            try:
                pdf_content = pdfkit.from_string(html_content, False)
                return pdf_content
            except Exception as e:
                print(f"PDF generation failed: {e}")
                return b""
        
        elif format == ReportFormat.CSV:
            # Convert data to CSV format
            csv_lines = []
            for section in sections:
                if section['type'] == 'table' and section['content']:
                    csv_lines.append(f"\n{section['title']}")
                    csv_lines.extend(self._table_to_csv(section['content']))
            
            return "\n".join(csv_lines)
        
        return ""
    
    async def schedule_report(self, report_id: str, frequency: ReportFrequency,
                            recipients: List[str], formats: List[ReportFormat]) -> ReportSchedule:
        """Schedule a report for automatic generation"""
        if report_id not in self.reports:
            raise ValueError(f"Report {report_id} not found")
        
        schedule_id = str(uuid.uuid4())
        
        schedule = ReportSchedule(
            schedule_id=schedule_id,
            report_id=report_id,
            frequency=frequency,
            next_run=self._calculate_next_run_from_frequency(frequency),
            recipients=recipients,
            formats=formats
        )
        
        self.schedules[schedule_id] = schedule
        
        # Update report
        self.reports[report_id].schedule = schedule
        
        return schedule
    
    async def _run_scheduled_report(self, schedule: ReportSchedule):
        """Run a scheduled report"""
        try:
            for format in schedule.formats:
                # Generate report
                generated = await self.generate_report(
                    schedule.report_id, format
                )
                
                # Distribute to recipients
                await self._distribute_report(generated, schedule.recipients)
                
        except Exception as e:
            print(f"Failed to run scheduled report: {e}")
            
            if schedule.retry_on_failure:
                # Implement retry logic
                pass
    
    async def _distribute_report(self, report: GeneratedReport, recipients: List[str]):
        """Distribute generated report to recipients"""
        # Placeholder for distribution logic
        # Would integrate with email, messaging, or file sharing services
        for recipient in recipients:
            print(f"Distributing report {report.instance_id} to {recipient}")
    
    def _calculate_next_run(self, schedule: ReportSchedule) -> datetime:
        """Calculate next run time for schedule"""
        return self._calculate_next_run_from_frequency(schedule.frequency)
    
    def _calculate_next_run_from_frequency(self, frequency: ReportFrequency) -> datetime:
        """Calculate next run time from frequency"""
        now = datetime.now()
        
        if frequency == ReportFrequency.DAILY:
            return now + timedelta(days=1)
        elif frequency == ReportFrequency.WEEKLY:
            return now + timedelta(weeks=1)
        elif frequency == ReportFrequency.MONTHLY:
            return now + timedelta(days=30)
        elif frequency == ReportFrequency.QUARTERLY:
            return now + timedelta(days=90)
        elif frequency == ReportFrequency.YEARLY:
            return now + timedelta(days=365)
        else:
            return now + timedelta(hours=1)
    
    async def _fetch_data(self, data_source: str,
                        parameters: Dict[str, Any]) -> Any:
        """Fetch data from specified source"""
        # Placeholder for data fetching
        # Would integrate with actual data sources
        return []
    
    def _format_table(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format data as table"""
        return data
    
    def _table_to_markdown(self, table: List[Dict[str, Any]]) -> str:
        """Convert table to markdown format"""
        if not table:
            return ""
        
        headers = list(table[0].keys())
        md_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |"
        ]
        
        for row in table:
            md_lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        
        return "\n".join(md_lines)
    
    def _table_to_html(self, table: List[Dict[str, Any]]) -> str:
        """Convert table to HTML format"""
        if not table:
            return ""
        
        headers = list(table[0].keys())
        html = ["<table>", "<thead><tr>"]
        
        for header in headers:
            html.append(f"<th>{header}</th>")
        
        html.append("</tr></thead><tbody>")
        
        for row in table:
            html.append("<tr>")
            for header in headers:
                html.append(f"<td>{row[header]}</td>")
            html.append("</tr>")
        
        html.append("</tbody></table>")
        
        return "\n".join(html)
    
    def _table_to_csv(self, table: List[Dict[str, Any]]) -> List[str]:
        """Convert table to CSV format"""
        if not table:
            return []
        
        headers = list(table[0].keys())
        csv_lines = [",".join(headers)]
        
        for row in table:
            csv_lines.append(",".join(str(row[h]) for h in headers))
        
        return csv_lines
    
    def _metrics_to_markdown(self, metrics: List[Dict[str, Any]]) -> str:
        """Convert metrics to markdown format"""
        md_lines = []
        for metric in metrics:
            md_lines.append(f"- **{metric.get('name', 'Unknown')}**: {metric.get('value', 'N/A')}")
        
        return "\n".join(md_lines)
    
    def _metrics_to_html(self, metrics: List[Dict[str, Any]]) -> str:
        """Convert metrics to HTML format"""
        html_parts = []
        for metric in metrics:
            html_parts.append(
                f"<div class='metric'>"
                f"<strong>{metric.get('name', 'Unknown')}</strong>: "
                f"{metric.get('value', 'N/A')}"
                f"</div>"
            )
        
        return "\n".join(html_parts)
    
    async def export_report_template(self, report_id: str) -> str:
        """Export report as template"""
        if report_id not in self.reports:
            return ""
        
        report = self.reports[report_id]
        
        template = ReportTemplate(
            template_id=str(uuid.uuid4()),
            name=f"{report.name} Template",
            description=f"Template based on {report.name}",
            category="custom",
            sections=report.sections.copy(),
            variables=report.parameters.copy(),
            tags=report.tags.copy()
        )
        
        return json.dumps({
            "template_id": template.template_id,
            "name": template.name,
            "description": template.description,
            "category": template.category,
            "sections": [
                {
                    "section_id": s.section_id,
                    "title": s.title,
                    "type": s.type,
                    "data_source": s.data_source,
                    "query": s.query,
                    "template": s.template,
                    "order": s.order
                }
                for s in template.sections
            ],
            "variables": template.variables,
            "tags": template.tags
        }, indent=2)
    
    def get_report_list(self, owner: Optional[str] = None,
                       tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get list of reports"""
        reports = []
        
        for report in self.reports.values():
            if owner and report.owner != owner:
                continue
            
            if tags and not any(tag in report.tags for tag in tags):
                continue
            
            reports.append({
                "report_id": report.report_id,
                "name": report.name,
                "description": report.description,
                "owner": report.owner,
                "created_at": report.created_at.isoformat(),
                "updated_at": report.updated_at.isoformat(),
                "scheduled": report.schedule is not None,
                "tags": report.tags
            })
        
        return reports