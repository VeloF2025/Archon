"""
Sprint Completion Quality Assurance Validator
Implements QA validation for sprint completion stage
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
import json
import re
from datetime import datetime, timedelta

from .qa_framework import (
    QAOrchestrator,
    ValidationStage,
    ValidationResult,
    ValidationViolation,
    ValidationSeverity,
    create_qa_orchestrator
)


@dataclass
class SprintGoal:
    """Represents a sprint goal"""
    id: str
    title: str
    description: str
    acceptance_criteria: List[str] = field(default_factory=list)
    user_stories: List[str] = field(default_factory=list)
    completed: bool = False
    completion_percentage: float = 0.0


@dataclass
class SprintDefinition:
    """Represents a sprint definition"""
    sprint_id: str
    name: str
    start_date: datetime
    end_date: datetime
    goals: List[SprintGoal] = field(default_factory=list)
    team_members: List[str] = field(default_factory=list)
    definition_of_done: List[str] = field(default_factory=list)


@dataclass
class SprintCompletionReport:
    """Represents a sprint completion report"""
    sprint: SprintDefinition
    completed_files: List[str] = field(default_factory=list)
    test_results: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    documentation_links: List[str] = field(default_factory=list)
    deployment_info: Dict[str, Any] = field(default_factory=dict)
    completion_metadata: Dict[str, Any] = field(default_factory=dict)


class SprintCompletionValidator:
    """Validates sprint completion using QA framework"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.qa_orchestrator = create_qa_orchestrator(config)
        self.min_test_coverage = self.config.get('min_test_coverage', 0.95)
        self.min_goal_completion = self.config.get('min_goal_completion', 0.90)
        self.required_documentation = self.config.get('required_documentation', ['PRD', 'PRP', 'ADR'])

    async def validate_sprint_completion(self, report: SprintCompletionReport) -> ValidationResult:
        """Validate sprint completion"""
        start_time = datetime.now()
        all_violations = []
        all_metrics = {}

        # Validate sprint goals completion
        goal_violations = self._validate_sprint_goals(report.sprint)
        all_violations.extend(goal_violations)

        # Validate completed files
        file_violations = await self._validate_completed_files(report.completed_files)
        all_violations.extend(file_violations)

        # Validate test coverage and results
        test_violations = self._validate_test_results(report.test_results)
        all_violations.extend(test_violations)

        # Validate documentation
        doc_violations = self._validate_documentation(report)
        all_violations.extend(doc_violations)

        # Validate quality metrics
        metrics_violations = self._validate_quality_metrics(report.quality_metrics)
        all_violations.extend(metrics_violations)

        # Validate deployment readiness
        deployment_violations = self._validate_deployment_readiness(report)
        all_violations.extend(deployment_violations)

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        # Calculate overall metrics
        overall_metrics = self._calculate_sprint_metrics(report, all_violations)

        passed = len([v for v in all_violations if v.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]) == 0

        return ValidationResult(
            stage=ValidationStage.SPRINT_COMPLETION,
            passed=passed,
            violations=all_violations,
            metrics={
                'execution_time_ms': int(execution_time),
                'sprint_id': report.sprint.sprint_id,
                'sprint_name': report.sprint.name,
                'total_violations': len(all_violations),
                'critical_violations': len([v for v in all_violations if v.severity == ValidationSeverity.CRITICAL]),
                'error_violations': len([v for v in all_violations if v.severity == ValidationSeverity.ERROR]),
                'files_validated': len(report.completed_files),
                **overall_metrics
            }
        )

    def _validate_sprint_goals(self, sprint: SprintDefinition) -> List[ValidationViolation]:
        """Validate sprint goals completion"""
        violations = []

        if not sprint.goals:
            violations.append(ValidationViolation(
                rule='no_sprint_goals',
                message='No sprint goals defined',
                severity=ValidationSeverity.ERROR,
                suggestion='Define clear sprint goals before completion'
            ))
            return violations

        # Calculate goal completion
        total_goals = len(sprint.goals)
        completed_goals = len([g for g in sprint.goals if g.completed])
        overall_completion = completed_goals / total_goals if total_goals > 0 else 0

        if overall_completion < self.min_goal_completion:
            violations.append(ValidationViolation(
                rule='insufficient_goal_completion',
                message=f'Goal completion {overall_completion:.1%} below required {self.min_goal_completion:.1%}',
                severity=ValidationSeverity.ERROR,
                suggestion=f'Complete at least {self.min_goal_completion:.1%} of sprint goals'
            ))

        # Validate individual goals
        for goal in sprint.goals:
            if not goal.acceptance_criteria:
                violations.append(ValidationViolation(
                    rule='no_acceptance_criteria',
                    message=f'Goal "{goal.title}" has no acceptance criteria',
                    severity=ValidationSeverity.WARNING,
                    suggestion=f'Add acceptance criteria for goal: {goal.title}'
                ))

            if goal.completed and goal.completion_percentage < 1.0:
                violations.append(ValidationViolation(
                    rule='inconsistent_completion_status',
                    message=f'Goal "{goal.title}" marked complete but completion percentage is {goal.completion_percentage:.1%}',
                    severity=ValidationSeverity.WARNING,
                    suggestion=f'Update completion status or percentage for goal: {goal.title}'
                ))

        return violations

    async def _validate_completed_files(self, file_paths: List[str]) -> List[ValidationViolation]:
        """Validate completed files"""
        violations = []

        if not file_paths:
            violations.append(ValidationViolation(
                rule='no_completed_files',
                message='No completed files provided for sprint',
                severity=ValidationSeverity.ERROR,
                suggestion='Add completed files to sprint completion report'
            ))
            return violations

        # Run QA validation on all files
        qa_result = self.qa_orchestrator.validate(
            file_paths,
            ValidationStage.SPRINT_COMPLETION
        )

        violations.extend(qa_result.violations)

        # Check for file existence
        for file_path in file_paths:
            if not Path(file_path).exists():
                violations.append(ValidationViolation(
                    rule='missing_completed_file',
                    message=f'Completed file not found: {file_path}',
                    severity=ValidationSeverity.ERROR,
                    file_path=file_path
                ))

        return violations

    def _validate_test_results(self, test_results: Dict[str, Any]) -> List[ValidationViolation]:
        """Validate test results"""
        violations = []

        if not test_results:
            violations.append(ValidationViolation(
                rule='no_test_results',
                message='No test results provided for sprint completion',
                severity=ValidationSeverity.CRITICAL,
                suggestion='Run all tests and include results in completion report'
            ))
            return violations

        # Check test coverage
        coverage = test_results.get('coverage', 0)
        if coverage < self.min_test_coverage:
            violations.append(ValidationViolation(
                rule='insufficient_test_coverage',
                message=f'Test coverage {coverage:.1%} below required {self.min_test_coverage:.1%}',
                severity=ValidationSeverity.ERROR,
                suggestion=f'Increase test coverage to at least {self.min_test_coverage:.1%}'
            ))

        # Check for test failures
        total_tests = test_results.get('total', 0)
        failed_tests = test_results.get('failed', 0)
        if failed_tests > 0:
            violations.append(ValidationViolation(
                rule='failed_tests',
                message=f'{failed_tests} out of {total_tests} tests failed',
                severity=ValidationSeverity.CRITICAL,
                suggestion='Fix all failing tests before sprint completion'
            ))

        # Check for skipped tests
        skipped_tests = test_results.get('skipped', 0)
        if skipped_tests > total_tests * 0.1:  # More than 10% skipped
            violations.append(ValidationViolation(
                rule='too_many_skipped_tests',
                message=f'{skipped_tests} tests skipped ({skipped_tests/total_tests:.1%})',
                severity=ValidationSeverity.WARNING,
                suggestion='Review and implement skipped tests'
            ))

        return violations

    def _validate_documentation(self, report: SprintCompletionReport) -> List[ValidationViolation]:
        """Validate documentation"""
        violations = []

        if not report.documentation_links:
            violations.append(ValidationViolation(
                rule='no_documentation',
                message='No documentation links provided',
                severity=ValidationSeverity.ERROR,
                suggestion='Add documentation links for all completed work'
            ))
            return violations

        # Check for required documentation types
        missing_docs = []
        for doc_type in self.required_documentation:
            if not any(doc_type.lower() in doc.lower() for doc in report.documentation_links):
                missing_docs.append(doc_type)

        if missing_docs:
            violations.append(ValidationViolation(
                rule='missing_required_documentation',
                message=f'Missing required documentation: {", ".join(missing_docs)}',
                severity=ValidationSeverity.ERROR,
                suggestion=f'Add {", ".join(missing_docs)} documentation'
            ))

        return violations

    def _validate_quality_metrics(self, quality_metrics: Dict[str, float]) -> List[ValidationViolation]:
        """Validate quality metrics"""
        violations = []

        required_metrics = ['code_coverage', 'performance_score', 'security_score']
        for metric in required_metrics:
            if metric not in quality_metrics:
                violations.append(ValidationViolation(
                    rule='missing_quality_metric',
                    message=f'Missing quality metric: {metric}',
                    severity=ValidationSeverity.WARNING,
                    suggestion=f'Include {metric} in quality metrics'
                ))

        # Check minimum quality thresholds
        if 'code_coverage' in quality_metrics and quality_metrics['code_coverage'] < self.min_test_coverage:
            violations.append(ValidationViolation(
                rule='low_code_coverage',
                message=f'Code coverage {quality_metrics["code_coverage"]:.1%} below threshold',
                severity=ValidationSeverity.ERROR,
                suggestion='Improve test coverage'
            ))

        return violations

    def _validate_deployment_readiness(self, report: SprintCompletionReport) -> List[ValidationViolation]:
        """Validate deployment readiness"""
        violations = []

        if not report.deployment_info:
            violations.append(ValidationViolation(
                rule='no_deployment_info',
                message='No deployment information provided',
                severity=ValidationSeverity.WARNING,
                suggestion='Include deployment configuration and status'
            ))

        # Check deployment prerequisites
        deployment_info = report.deployment_info
        if not deployment_info.get('environment_configured'):
            violations.append(ValidationViolation(
                rule='environment_not_configured',
                message='Target environment not properly configured',
                severity=ValidationSeverity.ERROR,
                suggestion='Configure target environment before deployment'
            ))

        if not deployment_info.get('rollback_plan'):
            violations.append(ValidationViolation(
                rule='no_rollback_plan',
                message='No rollback plan provided',
                severity=ValidationSeverity.WARNING,
                suggestion='Create rollback plan for deployment'
            ))

        return violations

    def _calculate_sprint_metrics(self, report: SprintCompletionReport, violations: List[ValidationViolation]) -> Dict[str, Any]:
        """Calculate overall sprint metrics"""
        sprint = report.sprint

        # Goal completion metrics
        total_goals = len(sprint.goals)
        completed_goals = len([g for g in sprint.goals if g.completed])
        goal_completion_rate = completed_goals / total_goals if total_goals > 0 else 0

        # Quality metrics
        critical_violations = len([v for v in violations if v.severity == ValidationSeverity.CRITICAL])
        error_violations = len([v for v in violations if v.severity == ValidationSeverity.ERROR])
        quality_score = 1.0 - ((critical_violations * 0.5 + error_violations * 0.2) / max(len(violations), 1))

        # Test metrics
        test_metrics = report.test_results
        test_pass_rate = (test_metrics.get('passed', 0) / max(test_metrics.get('total', 1), 1))

        # Documentation metrics
        doc_coverage = len([d for d in self.required_documentation
                           if any(d.lower() in doc.lower() for doc in report.documentation_links)]) / len(self.required_documentation)

        return {
            'goal_completion_rate': goal_completion_rate,
            'quality_score': max(0, quality_score),
            'test_pass_rate': test_pass_rate,
            'documentation_coverage': doc_coverage,
            'total_goals': total_goals,
            'completed_goals': completed_goals,
            'files_completed': len(report.completed_files),
            'sprint_duration_days': (sprint.end_date - sprint.start_date).days
        }

    def create_sprint_definition(self, sprint_data: Dict[str, Any]) -> SprintDefinition:
        """Create sprint definition from data"""
        goals = []
        for goal_data in sprint_data.get('goals', []):
            goals.append(SprintGoal(
                id=goal_data.get('id', ''),
                title=goal_data.get('title', ''),
                description=goal_data.get('description', ''),
                acceptance_criteria=goal_data.get('acceptance_criteria', []),
                user_stories=goal_data.get('user_stories', []),
                completed=goal_data.get('completed', False),
                completion_percentage=goal_data.get('completion_percentage', 0.0)
            ))

        return SprintDefinition(
            sprint_id=sprint_data.get('sprint_id', ''),
            name=sprint_data.get('name', ''),
            start_date=datetime.fromisoformat(sprint_data.get('start_date', datetime.now().isoformat())),
            end_date=datetime.fromisoformat(sprint_data.get('end_date', datetime.now().isoformat())),
            goals=goals,
            team_members=sprint_data.get('team_members', []),
            definition_of_done=sprint_data.get('definition_of_done', [])
        )

    def create_completion_report(self, sprint_def: SprintDefinition, report_data: Dict[str, Any]) -> SprintCompletionReport:
        """Create completion report from data"""
        return SprintCompletionReport(
            sprint=sprint_def,
            completed_files=report_data.get('completed_files', []),
            test_results=report_data.get('test_results', {}),
            quality_metrics=report_data.get('quality_metrics', {}),
            documentation_links=report_data.get('documentation_links', []),
            deployment_info=report_data.get('deployment_info', {}),
            completion_metadata=report_data.get('completion_metadata', {})
        )


# CLI interface for standalone usage
async def main():
    """CLI interface for sprint completion validator"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Validate sprint completion with QA')
    parser.add_argument('--sprint-config', required=True, help='Sprint configuration JSON file')
    parser.add_argument('--completion-report', required=True, help='Sprint completion report JSON file')
    parser.add_argument('--output', help='Output JSON report file')
    parser.add_argument('--config', help='QA configuration file')

    args = parser.parse_args()

    # Load configuration
    config = {}
    if args.config and Path(args.config).exists():
        config = json.loads(Path(args.config).read_text())

    # Load sprint definition and completion report
    sprint_data = json.loads(Path(args.sprint_config).read_text())
    report_data = json.loads(Path(args.completion_report).read_text())

    # Create validator and run validation
    validator = SprintCompletionValidator(config)
    sprint_def = validator.create_sprint_definition(sprint_data)
    completion_report = validator.create_completion_report(sprint_def, report_data)

    result = await validator.validate_sprint_completion(completion_report)

    # Export report
    report_json = validator.qa_orchestrator.export_validation_report(result, args.output)

    # Print summary
    print(f"Sprint Completion Validation: {'PASSED' if result.passed else 'FAILED'}")
    print(f"Total Violations: {len(result.violations)}")
    print(f"Execution Time: {result.execution_time_ms}ms")
    print(f"Sprint: {result.metrics.get('sprint_name', 'Unknown')}")
    print(f"Goal Completion: {result.metrics.get('goal_completion_rate', 0):.1%}")

    if args.output:
        print(f"Report saved to: {args.output}")

    # Exit with appropriate code
    sys.exit(0 if result.passed else 1)


if __name__ == '__main__':
    asyncio.run(main())