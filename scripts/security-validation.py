#!/usr/bin/env python3
"""
Security Validation Script for Archon
=====================================

This script scans the codebase for sensitive information and security vulnerabilities
before allowing commits to GitHub. It ensures no credentials, API keys, or personal
information is exposed in the public repository.

Usage:
    python scripts/security-validation.py
    python scripts/security-validation.py --fix  # Automatically fix issues where possible
"""

import os
import re
import sys
import argparse
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
import fnmatch

class SecurityValidator:
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.violations = []
        self.warnings = []
        
        # Patterns for sensitive information
        self.credential_patterns = {
            'api_key': [
                r'sk-[a-zA-Z0-9\-_]{20,}',  # OpenAI/DeepSeek API keys
                r'API_KEY\s*=\s*["\'][a-zA-Z0-9\-_]{20,}["\']',
                r'api[_-]?key["\']?\s*[:=]\s*["\'][^"\']{20,}["\']'
            ],
            'jwt_token': [
                r'eyJ[a-zA-Z0-9_\-]+\.eyJ[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+',  # JWT tokens
            ],
            'database_url': [
                r'https://[a-zA-Z0-9\-_]+\.supabase\.co',
                r'postgresql://[^@]+@[^/]+/[^?\s]+',
                r'mongodb://[^@]+@[^/]+/[^?\s]+',
            ],
            'personal_paths': [
                r'/mnt/c/[Jj]arvis',
                r'C:\\\\[Jj]arvis',
                r'/home/[a-zA-Z][a-zA-Z0-9_\-]+',
                r'/Users/[a-zA-Z][a-zA-Z0-9_\-]+',
            ],
            'passwords': [
                r'password\s*[:=]\s*["\'][^"\']{8,}["\']',
                r'PASSWORD\s*=\s*["\'][^"\']{8,}["\']',
            ],
            'secrets': [
                r'secret\s*[:=]\s*["\'][^"\']{20,}["\']',
                r'SECRET\s*=\s*["\'][^"\']{20,}["\']',
            ]
        }
        
        # File patterns to ignore
        self.ignore_patterns = [
            '*.git*',
            '__pycache__*',
            'node_modules*',
            '*.pyc',
            '*.pyo',
            '*.egg-info*',
            'dist/*',
            'build/*',
            '.venv/*',
            'venv/*',
            '*.log',
            'test-results/*',
            'playwright-report/*',
        ]
        
        # Files that are allowed to contain example credentials
        self.allowed_example_files = [
            '.env.example',
            'docker-compose.yml',
            'README.md',
            'getting-started.mdx',
        ]

    def should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored during security scan."""
        relative_path = file_path.relative_to(self.repo_path)
        
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(str(relative_path), pattern):
                return True
                
        return False

    def is_example_file(self, file_path: Path) -> bool:
        """Check if file is allowed to contain example credentials."""
        return any(example in file_path.name for example in self.allowed_example_files)

    def scan_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Scan a single file for security violations."""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            for violation_type, patterns in self.credential_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # Skip if it's an example file and contains placeholder text
                        if self.is_example_file(file_path):
                            if any(placeholder in match.group().lower() for placeholder in 
                                   ['your-', 'example', 'placeholder', 'demo', 'test']):
                                continue
                        
                        line_num = content[:match.start()].count('\n') + 1
                        violations.append({
                            'type': violation_type,
                            'file': str(file_path.relative_to(self.repo_path)),
                            'line': line_num,
                            'match': match.group(),
                            'pattern': pattern,
                            'severity': 'CRITICAL' if violation_type in ['api_key', 'jwt_token', 'database_url'] else 'HIGH'
                        })
                        
        except (UnicodeDecodeError, PermissionError):
            # Skip binary files or files we can't read
            pass
            
        return violations

    def scan_repository(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Scan entire repository for security violations."""
        violations = []
        warnings = []
        
        print("🔍 Scanning repository for security violations...")
        
        # Scan all files recursively
        for file_path in self.repo_path.rglob('*'):
            if not file_path.is_file():
                continue
                
            if self.should_ignore_file(file_path):
                continue
                
            file_violations = self.scan_file(file_path)
            violations.extend(file_violations)
            
        # Check for missing .gitignore patterns
        gitignore_path = self.repo_path / '.gitignore'
        if gitignore_path.exists():
            gitignore_content = gitignore_path.read_text()
            
            required_patterns = ['.env', '*.env', '*api*key*', '*secret*', '*token*']
            for pattern in required_patterns:
                if pattern not in gitignore_content:
                    warnings.append({
                        'type': 'missing_gitignore_pattern',
                        'message': f"Missing .gitignore pattern: {pattern}",
                        'severity': 'MEDIUM'
                    })
        else:
            warnings.append({
                'type': 'missing_gitignore',
                'message': "No .gitignore file found",
                'severity': 'HIGH'
            })
            
        return violations, warnings

    def generate_report(self, violations: List[Dict[str, Any]], warnings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        
        # Group violations by severity
        critical = [v for v in violations if v.get('severity') == 'CRITICAL']
        high = [v for v in violations if v.get('severity') == 'HIGH']
        medium = [v for v in violations if v.get('severity') == 'MEDIUM']
        
        # Group by violation type
        by_type = {}
        for violation in violations:
            vtype = violation['type']
            if vtype not in by_type:
                by_type[vtype] = []
            by_type[vtype].append(violation)
        
        report = {
            'timestamp': '2025-01-31T20:00:00Z',
            'status': 'FAIL' if violations else 'PASS',
            'summary': {
                'total_violations': len(violations),
                'critical': len(critical),
                'high': len(high),
                'medium': len(medium),
                'warnings': len(warnings)
            },
            'violations_by_type': by_type,
            'violations': violations,
            'warnings': warnings,
            'recommendations': self.get_recommendations(violations, warnings)
        }
        
        return report

    def get_recommendations(self, violations: List[Dict[str, Any]], warnings: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        if violations:
            recommendations.extend([
                "🚨 IMMEDIATE ACTION REQUIRED: Remove all exposed credentials from repository",
                "🔄 Rotate ALL exposed API keys and tokens immediately",
                "🔐 Use environment variables for all sensitive configuration",
                "📝 Update .env.example with placeholder values only",
            ])
            
        if any(v['type'] == 'api_key' for v in violations):
            recommendations.extend([
                "🔑 Generate new API keys from provider dashboards",
                "⚠️ Consider exposed keys compromised and rotate immediately",
            ])
            
        if any(v['type'] == 'database_url' for v in violations):
            recommendations.extend([
                "🗄️ Reset database connection strings",
                "🔒 Enable row-level security if using Supabase",
            ])
            
        if any(w['type'] == 'missing_gitignore_pattern' for w in warnings):
            recommendations.append("📋 Update .gitignore with comprehensive security patterns")
            
        recommendations.extend([
            "✅ Run this security validation script before every commit",
            "🔄 Set up pre-commit hooks to prevent credential exposure",
            "📚 Train team members on secure development practices",
        ])
        
        return recommendations

    def print_report(self, report: Dict[str, Any]):
        """Print security report to console."""
        
        print("\n" + "="*80)
        print("🛡️  ARCHON SECURITY AUDIT REPORT")
        print("="*80)
        
        print(f"\n📊 SUMMARY:")
        print(f"   Status: {'❌ FAIL' if report['status'] == 'FAIL' else '✅ PASS'}")
        print(f"   Total Violations: {report['summary']['total_violations']}")
        print(f"   Critical: {report['summary']['critical']}")
        print(f"   High: {report['summary']['high']}")
        print(f"   Medium: {report['summary']['medium']}")
        print(f"   Warnings: {report['summary']['warnings']}")
        
        if report['violations']:
            print(f"\n🚨 VIOLATIONS:")
            for violation in report['violations']:
                severity_icon = "🔴" if violation['severity'] == 'CRITICAL' else "🟠" if violation['severity'] == 'HIGH' else "🟡"
                print(f"   {severity_icon} {violation['severity']}: {violation['type']}")
                print(f"      File: {violation['file']}:{violation['line']}")
                print(f"      Match: {violation['match'][:50]}...")
                print()
                
        if report['warnings']:
            print(f"\n⚠️  WARNINGS:")
            for warning in report['warnings']:
                print(f"   🟡 {warning['severity']}: {warning['message']}")
            print()
            
        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"   {rec}")
            print()
            
        print("="*80)
        
        if report['status'] == 'FAIL':
            print("❌ SECURITY VALIDATION FAILED - Repository is not safe for public release")
            print("   Fix all violations before committing to GitHub")
        else:
            print("✅ SECURITY VALIDATION PASSED - Repository is safe for public release")

def main():
    parser = argparse.ArgumentParser(description='Archon Security Validation')
    parser.add_argument('--repo-path', default='.', help='Repository path to scan')
    parser.add_argument('--output', help='Output JSON report to file')
    parser.add_argument('--fail-on-violations', action='store_true', 
                       help='Exit with error code if violations found')
    
    args = parser.parse_args()
    
    validator = SecurityValidator(args.repo_path)
    violations, warnings = validator.scan_repository()
    report = validator.generate_report(violations, warnings)
    
    validator.print_report(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved to: {args.output}")
    
    if args.fail_on_violations and violations:
        sys.exit(1)
    
    sys.exit(0 if report['status'] == 'PASS' else 1)

if __name__ == '__main__':
    main()