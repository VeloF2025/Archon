#!/usr/bin/env python3

"""
Security Threshold Checker for Archon Phase 6 Authentication System
Validates security scan results against defined thresholds
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

class SecurityThresholdChecker:
    """Checks security scan results against defined thresholds"""
    
    def __init__(self, report_path: str):
        self.report_path = Path(report_path)
        self.report = self.load_report()
    
    def load_report(self) -> Dict[str, Any]:
        """Load security report from JSON file"""
        try:
            with open(self.report_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Report file {self.report_path} not found")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in report file: {e}")
            sys.exit(1)
    
    def check_thresholds(self, 
                        fail_on_critical: bool = True,
                        max_high: int = 5,
                        max_medium: int = 20,
                        max_low: int = 50) -> bool:
        """Check if security results exceed thresholds"""
        
        critical_count = self.report.get('critical_count', 0)
        high_count = self.report.get('high_count', 0)
        medium_count = self.report.get('medium_count', 0)
        low_count = self.report.get('low_count', 0)
        
        print("Security Threshold Check Results")
        print("=" * 40)
        print(f"Critical Issues: {critical_count} (threshold: {'0 (fail)' if fail_on_critical else 'N/A'})")
        print(f"High Issues:     {high_count} (threshold: {max_high})")
        print(f"Medium Issues:   {medium_count} (threshold: {max_medium})")
        print(f"Low Issues:      {low_count} (threshold: {max_low})")
        print()
        
        failures = []
        
        # Check critical threshold
        if fail_on_critical and critical_count > 0:
            failures.append(f"❌ CRITICAL: {critical_count} critical issues found (blocking)")
        
        # Check high threshold
        if high_count > max_high:
            failures.append(f"❌ HIGH: {high_count} high issues exceed threshold of {max_high}")
        
        # Check medium threshold
        if medium_count > max_medium:
            failures.append(f"⚠️ MEDIUM: {medium_count} medium issues exceed threshold of {max_medium}")
        
        # Check low threshold (warning only)
        if low_count > max_low:
            print(f"ℹ️ INFO: {low_count} low issues exceed threshold of {max_low} (non-blocking)")
        
        if failures:
            print("Threshold Violations:")
            for failure in failures:
                print(f"  {failure}")
            print()
            
            # Show top critical/high issues
            self.show_top_issues()
            
            return False
        else:
            print("✅ All security thresholds passed!")
            return True
    
    def show_top_issues(self, limit: int = 10) -> None:
        """Show the most critical security issues"""
        issues = self.report.get('issues', [])
        
        # Sort by severity priority
        severity_priority = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        sorted_issues = sorted(issues, 
                             key=lambda x: (severity_priority.get(x.get('severity', 'low'), 4), 
                                          x.get('cvss_score', 0)), 
                             reverse=True)
        
        print(f"Top {min(limit, len(sorted_issues))} Security Issues:")
        print("-" * 60)
        
        for i, issue in enumerate(sorted_issues[:limit], 1):
            severity = issue.get('severity', 'unknown').upper()
            tool = issue.get('tool', 'unknown')
            title = issue.get('title', 'No title')
            file_path = issue.get('file_path', 'Unknown file')
            
            print(f"{i:2}. [{severity}] {title}")
            print(f"    Tool: {tool}")
            print(f"    File: {file_path}")
            print(f"    Description: {issue.get('description', 'No description')[:100]}...")
            print()
    
    def get_security_score(self) -> float:
        """Calculate overall security score (0-100, higher is better)"""
        total_issues = self.report.get('total_issues', 0)
        critical_count = self.report.get('critical_count', 0)
        high_count = self.report.get('high_count', 0)
        medium_count = self.report.get('medium_count', 0)
        
        if total_issues == 0:
            return 100.0
        
        # Weighted penalty system
        penalty = (critical_count * 20) + (high_count * 5) + (medium_count * 2)
        score = max(0, 100 - penalty)
        
        return score
    
    def generate_gate_decision(self, 
                              fail_on_critical: bool = True,
                              max_high: int = 5,
                              max_medium: int = 20) -> Dict[str, Any]:
        """Generate quality gate decision"""
        
        passed = self.check_thresholds(fail_on_critical, max_high, max_medium, 999)
        security_score = self.get_security_score()
        
        decision = {
            "gate_status": "PASS" if passed else "FAIL",
            "security_score": security_score,
            "deployment_allowed": passed,
            "requires_approval": not passed and security_score > 70,
            "risk_level": self.get_risk_level(security_score),
            "timestamp": self.report.get('scan_timestamp'),
            "summary": self.report.get('summary', {}),
            "recommendations": self.get_gate_recommendations(passed, security_score)
        }
        
        return decision
    
    def get_risk_level(self, score: float) -> str:
        """Determine risk level based on security score"""
        if score >= 90:
            return "LOW"
        elif score >= 70:
            return "MEDIUM"
        elif score >= 50:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def get_gate_recommendations(self, passed: bool, score: float) -> List[str]:
        """Get quality gate recommendations"""
        recommendations = []
        
        if not passed:
            recommendations.append("🚫 DEPLOYMENT BLOCKED - Address security issues before proceeding")
            
            if self.report.get('critical_count', 0) > 0:
                recommendations.append("🚨 Critical vulnerabilities must be fixed immediately")
            
            if self.report.get('high_count', 0) > 5:
                recommendations.append("⚠️ Too many high-severity issues - prioritize fixes")
        
        if score < 70:
            recommendations.append("📈 Security score below 70 - comprehensive security review needed")
        
        if score < 90:
            recommendations.append("🔧 Consider implementing additional security controls")
        
        recommendations.extend([
            "🔄 Run security scans regularly in CI/CD pipeline",
            "📚 Provide security training for development team",
            "🛡️ Implement shift-left security practices"
        ])
        
        return recommendations

def main():
    parser = argparse.ArgumentParser(description="Check security scan results against thresholds")
    parser.add_argument("--report", required=True, help="Path to security report JSON file")
    parser.add_argument("--fail-on-critical", action="store_true", help="Fail if any critical issues found")
    parser.add_argument("--max-high", type=int, default=5, help="Maximum high-severity issues allowed")
    parser.add_argument("--max-medium", type=int, default=20, help="Maximum medium-severity issues allowed")
    parser.add_argument("--max-low", type=int, default=50, help="Maximum low-severity issues allowed")
    parser.add_argument("--output-decision", help="Output file for gate decision JSON")
    parser.add_argument("--quiet", action="store_true", help="Suppress output except for failures")
    
    args = parser.parse_args()
    
    try:
        checker = SecurityThresholdChecker(args.report)
        
        # Check thresholds
        passed = checker.check_thresholds(
            fail_on_critical=args.fail_on_critical,
            max_high=args.max_high,
            max_medium=args.max_medium,
            max_low=args.max_low
        )
        
        # Generate gate decision
        decision = checker.generate_gate_decision(
            fail_on_critical=args.fail_on_critical,
            max_high=args.max_high,
            max_medium=args.max_medium
        )
        
        # Save gate decision if requested
        if args.output_decision:
            with open(args.output_decision, 'w') as f:
                json.dump(decision, f, indent=2)
            
            if not args.quiet:
                print(f"Gate decision saved to: {args.output_decision}")
        
        # Print decision summary
        if not args.quiet:
            print("\nQuality Gate Decision:")
            print(f"Status: {decision['gate_status']}")
            print(f"Security Score: {decision['security_score']:.1f}/100")
            print(f"Risk Level: {decision['risk_level']}")
            print(f"Deployment Allowed: {decision['deployment_allowed']}")
            
            if decision['requires_approval']:
                print("⚠️ Manual approval required for deployment")
        
        # Exit with appropriate code
        if passed:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"Error checking security thresholds: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()