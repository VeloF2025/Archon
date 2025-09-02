#!/usr/bin/env python3

"""
Security Report Generator for Archon Phase 6 Authentication System
Aggregates and analyzes security scan results from multiple tools
"""

import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

@dataclass
class SecurityIssue:
    """Represents a security issue found by scanning tools"""
    severity: str
    tool: str
    category: str
    title: str
    description: str
    file_path: str
    line_number: int
    cve_id: str = ""
    cvss_score: float = 0.0
    remediation: str = ""

@dataclass
class SecurityReport:
    """Aggregated security report"""
    scan_timestamp: str
    total_issues: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    info_count: int
    tools_used: List[str]
    issues: List[SecurityIssue]
    summary: Dict[str, Any]

class SecurityReportGenerator:
    """Generates comprehensive security reports from multiple scan tools"""
    
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        self.issues: List[SecurityIssue] = []
        self.tools_used: set = set()
    
    def process_bandit_report(self, file_path: Path) -> None:
        """Process Bandit SAST scan results"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            self.tools_used.add("bandit")
            
            for result in data.get('results', []):
                issue = SecurityIssue(
                    severity=result.get('issue_severity', 'UNKNOWN').lower(),
                    tool="bandit",
                    category="sast",
                    title=result.get('test_name', 'Unknown'),
                    description=result.get('issue_text', ''),
                    file_path=result.get('filename', ''),
                    line_number=result.get('line_number', 0),
                    cve_id=result.get('issue_cwe', {}).get('id', ''),
                    cvss_score=result.get('issue_severity_score', 0.0)
                )
                self.issues.append(issue)
                
        except Exception as e:
            print(f"Error processing Bandit report {file_path}: {e}")
    
    def process_trivy_report(self, file_path: Path) -> None:
        """Process Trivy vulnerability scan results"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            self.tools_used.add("trivy")
            
            for run in data.get('runs', []):
                for result in run.get('results', []):
                    for rule in result.get('ruleIndex', []):
                        issue = SecurityIssue(
                            severity=rule.get('properties', {}).get('severity', 'unknown').lower(),
                            tool="trivy",
                            category="container",
                            title=rule.get('name', 'Unknown'),
                            description=rule.get('fullDescription', {}).get('text', ''),
                            file_path=result.get('ruleId', ''),
                            line_number=0,
                            cve_id=rule.get('properties', {}).get('cve', ''),
                            cvss_score=rule.get('properties', {}).get('cvss', 0.0)
                        )
                        self.issues.append(issue)
                        
        except Exception as e:
            print(f"Error processing Trivy report {file_path}: {e}")
    
    def process_npm_audit_report(self, file_path: Path) -> None:
        """Process npm audit results"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            self.tools_used.add("npm-audit")
            
            for vuln_id, vuln in data.get('vulnerabilities', {}).items():
                issue = SecurityIssue(
                    severity=vuln.get('severity', 'unknown').lower(),
                    tool="npm-audit",
                    category="dependency",
                    title=vuln.get('name', vuln_id),
                    description=vuln.get('overview', ''),
                    file_path=', '.join(vuln.get('via', [])),
                    line_number=0,
                    cve_id=', '.join([ref.get('url', '') for ref in vuln.get('references', []) if 'CVE' in ref.get('url', '')]),
                    cvss_score=vuln.get('range', 0)
                )
                self.issues.append(issue)
                
        except Exception as e:
            print(f"Error processing npm audit report {file_path}: {e}")
    
    def process_safety_report(self, file_path: Path) -> None:
        """Process Safety dependency scan results"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            self.tools_used.add("safety")
            
            for vuln in data.get('vulnerabilities', []):
                issue = SecurityIssue(
                    severity="high",  # Safety typically reports high-severity issues
                    tool="safety",
                    category="dependency",
                    title=f"{vuln.get('package_name', 'Unknown')} - {vuln.get('vulnerability_id', '')}",
                    description=vuln.get('advisory', ''),
                    file_path=vuln.get('package_name', ''),
                    line_number=0,
                    cve_id=vuln.get('vulnerability_id', ''),
                    cvss_score=vuln.get('cvss', 0.0)
                )
                self.issues.append(issue)
                
        except Exception as e:
            print(f"Error processing Safety report {file_path}: {e}")
    
    def process_all_reports(self) -> None:
        """Process all security reports in the input directory"""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory {self.input_dir} does not exist")
        
        # Process all report files
        for file_path in self.input_dir.rglob("*.json"):
            file_name = file_path.name.lower()
            
            if "bandit" in file_name:
                self.process_bandit_report(file_path)
            elif "trivy" in file_name:
                self.process_trivy_report(file_path)
            elif "npm-audit" in file_name:
                self.process_npm_audit_report(file_path)
            elif "safety" in file_name:
                self.process_safety_report(file_path)
            else:
                print(f"Unknown report type: {file_name}")
    
    def calculate_risk_score(self) -> float:
        """Calculate overall risk score based on issues"""
        score = 0.0
        
        for issue in self.issues:
            if issue.severity == "critical":
                score += 10.0
            elif issue.severity == "high":
                score += 5.0
            elif issue.severity == "medium":
                score += 2.0
            elif issue.severity == "low":
                score += 1.0
        
        return min(score, 100.0)  # Cap at 100
    
    def get_severity_counts(self) -> Dict[str, int]:
        """Get count of issues by severity"""
        counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0
        }
        
        for issue in self.issues:
            if issue.severity in counts:
                counts[issue.severity] += 1
            else:
                counts["info"] += 1
        
        return counts
    
    def generate_report(self) -> SecurityReport:
        """Generate comprehensive security report"""
        self.process_all_reports()
        
        severity_counts = self.get_severity_counts()
        risk_score = self.calculate_risk_score()
        
        # Determine overall status
        if severity_counts["critical"] > 0:
            status = "CRITICAL - Immediate action required"
        elif severity_counts["high"] > 5:
            status = "HIGH RISK - Address high-severity issues"
        elif severity_counts["medium"] > 20:
            status = "MEDIUM RISK - Review medium-severity issues"
        else:
            status = "LOW RISK - Acceptable security posture"
        
        report = SecurityReport(
            scan_timestamp=datetime.now().isoformat(),
            total_issues=len(self.issues),
            critical_count=severity_counts["critical"],
            high_count=severity_counts["high"],
            medium_count=severity_counts["medium"],
            low_count=severity_counts["low"],
            info_count=severity_counts["info"],
            tools_used=list(self.tools_used),
            issues=self.issues,
            summary={
                "risk_score": risk_score,
                "status": status,
                "total_scanned_files": len(set(issue.file_path for issue in self.issues)),
                "tools_count": len(self.tools_used),
                "recommendations": self.get_recommendations(severity_counts)
            }
        )
        
        return report
    
    def get_recommendations(self, severity_counts: Dict[str, int]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        if severity_counts["critical"] > 0:
            recommendations.append("🚨 URGENT: Address all critical vulnerabilities immediately")
            recommendations.append("Consider emergency rollback if issues affect production")
        
        if severity_counts["high"] > 5:
            recommendations.append("⚠️ High number of high-severity issues detected")
            recommendations.append("Prioritize patching dependencies and fixing code issues")
        
        if severity_counts["medium"] > 20:
            recommendations.append("📋 Medium-severity issues require attention")
            recommendations.append("Schedule security hardening sprint")
        
        recommendations.extend([
            "🔄 Regular security scanning should be automated",
            "📊 Monitor security metrics in CI/CD pipeline",
            "🛡️ Implement security-first development practices",
            "📚 Provide security training for development team"
        ])
        
        return recommendations

def save_json_report(report: SecurityReport, output_file: str) -> None:
    """Save report as JSON"""
    with open(output_file, 'w') as f:
        json.dump(asdict(report), f, indent=2, default=str)

def save_html_report(report: SecurityReport, output_file: str) -> None:
    """Save report as HTML"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Archon Security Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
            .card {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
            .critical {{ background-color: #fee; border-left: 4px solid #dc3545; }}
            .high {{ background-color: #fff3cd; border-left: 4px solid #ffc107; }}
            .medium {{ background-color: #e2f3ff; border-left: 4px solid #17a2b8; }}
            .low {{ background-color: #d4edda; border-left: 4px solid #28a745; }}
            .issue {{ margin-bottom: 15px; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }}
            .issue-title {{ font-weight: bold; margin-bottom: 5px; }}
            .issue-details {{ font-size: 0.9em; color: #666; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔒 Archon Security Scan Report</h1>
                <p>Generated on {report.scan_timestamp}</p>
                <p><strong>Status:</strong> {report.summary['status']}</p>
            </div>
            
            <div class="summary">
                <div class="card critical">
                    <h3>Critical</h3>
                    <h2>{report.critical_count}</h2>
                </div>
                <div class="card high">
                    <h3>High</h3>
                    <h2>{report.high_count}</h2>
                </div>
                <div class="card medium">
                    <h3>Medium</h3>
                    <h2>{report.medium_count}</h2>
                </div>
                <div class="card low">
                    <h3>Low</h3>
                    <h2>{report.low_count}</h2>
                </div>
            </div>
            
            <div class="recommendations">
                <h2>📋 Recommendations</h2>
                <ul>
    """
    
    for rec in report.summary['recommendations']:
        html_content += f"                    <li>{rec}</li>\n"
    
    html_content += f"""
                </ul>
            </div>
            
            <div class="tools">
                <h2>🔧 Tools Used</h2>
                <p>{', '.join(report.tools_used)}</p>
            </div>
            
            <div class="issues">
                <h2>🐛 Issues Found ({report.total_issues} total)</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Severity</th>
                            <th>Tool</th>
                            <th>Category</th>
                            <th>Title</th>
                            <th>File</th>
                            <th>Line</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    for issue in sorted(report.issues, key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x.severity, 4)):
        severity_class = issue.severity
        html_content += f"""
                        <tr class="{severity_class}">
                            <td><span class="badge {severity_class}">{issue.severity.upper()}</span></td>
                            <td>{issue.tool}</td>
                            <td>{issue.category}</td>
                            <td>{issue.title}</td>
                            <td>{issue.file_path}</td>
                            <td>{issue.line_number if issue.line_number > 0 else 'N/A'}</td>
                        </tr>
        """
    
    html_content += """
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive security report")
    parser.add_argument("--input-dir", required=True, help="Directory containing security scan results")
    parser.add_argument("--output", required=True, help="Output file path (without extension)")
    parser.add_argument("--format", choices=["json", "html", "both"], default="both", help="Output format")
    
    args = parser.parse_args()
    
    try:
        generator = SecurityReportGenerator(args.input_dir)
        report = generator.generate_report()
        
        print(f"Security scan completed:")
        print(f"  Total issues: {report.total_issues}")
        print(f"  Critical: {report.critical_count}")
        print(f"  High: {report.high_count}")
        print(f"  Medium: {report.medium_count}")
        print(f"  Low: {report.low_count}")
        print(f"  Risk Score: {report.summary['risk_score']:.1f}/100")
        print(f"  Status: {report.summary['status']}")
        
        # Save reports
        if args.format in ["json", "both"]:
            json_file = f"{args.output}.json"
            save_json_report(report, json_file)
            print(f"JSON report saved: {json_file}")
        
        if args.format in ["html", "both"]:
            html_file = f"{args.output}.html"
            save_html_report(report, html_file)
            print(f"HTML report saved: {html_file}")
        
        # Set exit code based on severity
        if report.critical_count > 0:
            print("⚠️ Critical issues found - consider blocking deployment")
            sys.exit(2)
        elif report.high_count > 5:
            print("⚠️ High number of high-severity issues")
            sys.exit(1)
        else:
            print("✅ Security scan completed with acceptable risk level")
            sys.exit(0)
            
    except Exception as e:
        print(f"Error generating security report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()