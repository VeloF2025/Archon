#!/usr/bin/env python3
"""
Enhanced Pre-commit QA Hook
Integrates with Standard Archon Quality Assurance workflow
"""

import sys
import os
import asyncio
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python" / "src"))

from agents.quality.qa_orchestrator import (
    QAWorkflowOrchestrator,
    WorkflowRequest,
    WorkflowTrigger,
    ValidationStage,
    create_qa_workflow_orchestrator
)


class PrecommitQAChecker:
    """Enhanced pre-commit QA checker"""

    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.orchestrator = create_qa_workflow_orchestrator(self.config.get('qa', {}))
        self.output_dir = Path(self.config.get('output_dir', '.qa-reports'))
        self.output_dir.mkdir(exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from file"""
        default_config = {
            'qa': {
                'parallel_execution': True,
                'git_commit': {
                    'max_commit_size_mb': 5,
                    'forbidden_patterns': [
                        r'password',
                        r'secret.*key',
                        r'api_key',
                        r'token.*=',
                        r'credential'
                    ]
                }
            },
            'output_dir': '.qa-reports'
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge with default config
                for section, values in user_config.items():
                    if section in default_config and isinstance(default_config[section], dict):
                        default_config[section].update(values)
                    else:
                        default_config[section] = values
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")

        return default_config

    async def run_precommit_validation(self) -> bool:
        """Run pre-commit QA validation"""
        print("🔍 Running Archon Quality Assurance pre-commit checks...")
        print("=" * 60)

        try:
            # Get staged files
            staged_files = self._get_staged_files()
            if not staged_files:
                print("✅ No staged files to validate")
                return True

            print(f"📁 Found {len(staged_files)} staged files to validate:")
            for file_path in staged_files:
                print(f"   - {file_path}")

            # Create workflow request
            request = self.orchestrator.create_workflow_request(
                trigger=WorkflowTrigger.GIT_COMMIT,
                stage=ValidationStage.GIT_COMMIT,
                target=staged_files,
                requester='pre-commit-hook',
                metadata={
                    'hook_type': 'pre-commit',
                    'timestamp': datetime.now().isoformat()
                }
            )

            # Execute QA workflow
            print("\n🚀 Executing QA validation...")
            result = await self.orchestrator.execute_workflow(request)

            # Save report
            report_path = self._save_precommit_report(result)

            # Display results
            self._display_results(result, report_path)

            return result.passed

        except Exception as e:
            print(f"\n❌ Pre-commit QA validation failed with error:")
            print(f"   {str(e)}")
            return False

    def _get_staged_files(self) -> list:
        """Get list of staged files"""
        try:
            import subprocess
            # Get staged files (both added and modified)
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM'],
                capture_output=True,
                text=True,
                check=True
            )
            staged_files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
            return staged_files
        except subprocess.CalledProcessError as e:
            print(f"Error getting staged files: {e}")
            return []
        except FileNotFoundError:
            print("Error: git command not found. Are you in a git repository?")
            return []

    def _save_precommit_report(self, result) -> str:
        """Save pre-commit report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"precommit_qa_report_{timestamp}.json"
        report_path = self.output_dir / report_filename

        report_data = {
            'precommit_validation': True,
            'timestamp': datetime.now().isoformat(),
            'passed': result.passed,
            'request_id': result.request_id,
            'stage': result.stage.value,
            'trigger': result.trigger.value,
            'execution_summary': result.execution_summary,
            'violations_count': len(result.all_violations),
            'artifacts': result.artifacts
        }

        # Add detailed violation information
        if result.all_violations:
            report_data['violations'] = [
                {
                    'rule': v.rule,
                    'message': v.message,
                    'severity': v.severity.value,
                    'file_path': v.file_path,
                    'line_number': v.line_number,
                    'suggestion': v.suggestion
                }
                for v in result.all_violations
            ]

        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        return str(report_path)

    def _display_results(self, result, report_path: str):
        """Display validation results"""
        print("\n" + "=" * 60)
        print("📊 PRE-COMMIT QA VALIDATION RESULTS")
        print("=" * 60)

        # Overall status
        if result.passed:
            print("✅ PRE-COMMIT VALIDATION PASSED")
            print("   Your changes meet all quality standards!")
        else:
            print("❌ PRE-COMMIT VALIDATION FAILED")
            print("   Please address the violations below before committing.")

        print(f"\n📈 Execution Summary:")
        print(f"   Request ID: {result.request_id}")
        print(f"   Execution Time: {result.execution_summary.get('execution_time_ms', 0)}ms")
        print(f"   Files Validated: {result.execution_summary.get('commits_validated', 0)}")
        print(f"   Total Violations: {len(result.all_violations)}")

        # Violations by severity
        critical_violations = [v for v in result.all_violations if v.severity.value == 'critical']
        error_violations = [v for v in result.all_violations if v.severity.value == 'error']
        warning_violations = [v for v in result.all_violations if v.severity.value == 'warning']
        info_violations = [v for v in result.all_violations if v.severity.value == 'info']

        if critical_violations:
            print(f"\n🚨 CRITICAL VIOLATIONS ({len(critical_violations)}):")
            for violation in critical_violations[:5]:  # Show first 5
                print(f"   ❌ {violation.rule}: {violation.message}")
                if violation.file_path:
                    print(f"      📁 {violation.file_path}")
                    if violation.line_number:
                        print(f"      📍 Line {violation.line_number}")
                if violation.suggestion:
                    print(f"      💡 {violation.suggestion}")
            if len(critical_violations) > 5:
                print(f"   ... and {len(critical_violations) - 5} more critical violations")

        if error_violations:
            print(f"\n⚠️  ERROR VIOLATIONS ({len(error_violations)}):")
            for violation in error_violations[:5]:  # Show first 5
                print(f"   ❌ {violation.rule}: {violation.message}")
                if violation.file_path:
                    print(f"      📁 {violation.file_path}")
                    if violation.line_number:
                        print(f"      📍 Line {violation.line_number}")
                if violation.suggestion:
                    print(f"      💡 {violation.suggestion}")
            if len(error_violations) > 5:
                print(f"   ... and {len(error_violations) - 5} more error violations")

        if warning_violations:
            print(f"\n⚡ WARNING VIOLATIONS ({len(warning_violations)}):")
            for violation in warning_violations[:3]:  # Show first 3
                print(f"   ⚠️  {violation.rule}: {violation.message}")
                if violation.file_path:
                    print(f"      📁 {violation.file_path}")
                if violation.suggestion:
                    print(f"      💡 {violation.suggestion}")
            if len(warning_violations) > 3:
                print(f"   ... and {len(warning_violations) - 3} more warnings")

        # Report location
        print(f"\n📄 Detailed report saved to:")
        print(f"   {report_path}")

        # Help message
        if not result.passed:
            print(f"\n🛠️  To fix violations:")
            print(f"   1. Review the detailed report for specific guidance")
            print(f"   2. Fix the issues in your code")
            print(f"   3. Stage the fixes again")
            print(f"   4. Run git commit again")

        print("\n" + "=" * 60)


async def main():
    """Main pre-commit hook entry point"""
    parser = argparse.ArgumentParser(description='Archon QA Pre-commit Hook')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Create and run pre-commit checker
    checker = PrecommitQAChecker(args.config)
    success = await checker.run_precommit_validation()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    asyncio.run(main())