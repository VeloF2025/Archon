"""
QA Workflow Orchestration System
Coordinates and manages quality assurance workflows across all stages
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime, timedelta

from .qa_framework import (
    QAOrchestrator as CoreQAOrchestrator,
    ValidationStage,
    ValidationResult,
    ValidationViolation,
    ValidationSeverity
)
from .file_submission_validator import FileSubmissionValidator, SubmissionBatch
from .sprint_completion_validator import SprintCompletionValidator, SprintCompletionReport
from .git_commit_validator import GitCommitValidator, CommitValidationRequest


class WorkflowTrigger(Enum):
    """Triggers for QA workflow execution"""
    FILE_SUBMISSION = "file_submission"
    SPRINT_COMPLETION = "sprint_completion"
    GIT_COMMIT = "git_commit"
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    CI_PIPELINE = "ci_pipeline"


@dataclass
class WorkflowRequest:
    """Represents a QA workflow request"""
    request_id: str
    trigger: WorkflowTrigger
    stage: ValidationStage
    target: Any
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    requester: str = ""
    priority: int = 5  # 1-10, 1 is highest


@dataclass
class WorkflowResult:
    """Represents a QA workflow result"""
    request_id: str
    stage: ValidationStage
    trigger: WorkflowTrigger
    passed: bool
    validation_results: List[ValidationResult] = field(default_factory=list)
    all_violations: List[ValidationViolation] = field(default_factory=list)
    execution_summary: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    artifacts: Dict[str, str] = field(default_factory=dict)  # artifact_type -> file_path


class QAWorkflowOrchestrator:
    """Main orchestrator for QA workflows"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize validators
        self.core_qa = CoreQAOrchestrator(config)
        self.file_validator = FileSubmissionValidator(config.get('file_submission', {}))
        self.sprint_validator = SprintCompletionValidator(config.get('sprint_completion', {}))
        self.commit_validator = GitCommitValidator(config.get('git_commit', {}))

        # Workflow configuration
        self.parallel_execution = self.config.get('parallel_execution', True)
        self.max_concurrent_workflows = self.config.get('max_concurrent_workflows', 3)
        self.default_timeout_seconds = self.config.get('default_timeout_seconds', 300)

        # Workflow state
        self.active_workflows: Dict[str, WorkflowRequest] = {}
        self.workflow_history: List[WorkflowResult] = []
        self.workflow_hooks: Dict[str, List[Callable]] = {}

        # Performance tracking
        self.performance_metrics = {
            'total_workflows': 0,
            'successful_workflows': 0,
            'failed_workflows': 0,
            'average_execution_time_ms': 0,
            'last_execution': None
        }

    async def execute_workflow(self, request: WorkflowRequest) -> WorkflowResult:
        """Execute a QA workflow"""
        start_time = datetime.now()

        # Check if workflow already exists
        if request.request_id in self.active_workflows:
            raise ValueError(f"Workflow {request.request_id} is already active")

        # Add to active workflows
        self.active_workflows[request.request_id] = request

        try:
            # Execute pre-workflow hooks
            await self._execute_hooks('pre_workflow', request)

            # Execute the appropriate workflow
            if request.stage == ValidationStage.FILE_SUBMISSION:
                result = await self._execute_file_submission_workflow(request)
            elif request.stage == ValidationStage.SPRINT_COMPLETION:
                result = await self._execute_sprint_completion_workflow(request)
            elif request.stage == ValidationStage.GIT_COMMIT:
                result = await self._execute_git_commit_workflow(request)
            else:
                raise ValueError(f"Unsupported validation stage: {request.stage}")

            # Execute post-workflow hooks
            await self._execute_hooks('post_workflow', request, result)

            # Update performance metrics
            self._update_performance_metrics(result)

            return result

        except Exception as e:
            self.logger.error(f"Workflow {request.request_id} failed: {str(e)}")

            # Create failed result
            failed_result = WorkflowResult(
                request_id=request.request_id,
                stage=request.stage,
                trigger=request.trigger,
                passed=False,
                execution_summary={
                    'error': str(e),
                    'error_type': type(e).__name__
                },
                start_time=start_time,
                end_time=datetime.now()
            )

            # Execute error hooks
            await self._execute_hooks('workflow_error', request, failed_result)

            return failed_result

        finally:
            # Remove from active workflows
            self.active_workflows.pop(request.request_id, None)

    async def _execute_file_submission_workflow(self, request: WorkflowRequest) -> WorkflowResult:
        """Execute file submission workflow"""
        self.logger.info(f"Executing file submission workflow: {request.request_id}")

        validation_results = []
        all_violations = []
        artifacts = {}

        # Determine target type
        if isinstance(request.target, list):
            # List of file paths
            batch = self.file_validator.create_submission_batch(
                request.target,
                request.metadata.get('submitter', ''),
                request.metadata.get('context', {})
            )
            result = await self.file_validator.validate_batch(batch)
        elif isinstance(request.target, SubmissionBatch):
            # Pre-created batch
            result = await self.file_validator.validate_batch(request.target)
        else:
            # Single submission
            from .file_submission_validator import FileSubmission
            if isinstance(request.target, FileSubmission):
                result = await self.file_validator.validate_submission(request.target)
            else:
                # Assume file path
                from .file_submission_validator import FileSubmission
                submission = FileSubmission(file_path=str(request.target))
                result = await self.file_validator.validate_submission(submission)

        validation_results.append(result)
        all_violations.extend(result.violations)

        # Generate artifacts
        artifacts['validation_report'] = self._save_artifact(
            request.request_id, 'validation_report',
            self.core_qa.export_validation_report(result)
        )

        workflow_result = WorkflowResult(
            request_id=request.request_id,
            stage=ValidationStage.FILE_SUBMISSION,
            trigger=request.trigger,
            passed=result.passed,
            validation_results=validation_results,
            all_violations=all_violations,
            execution_summary={
                'files_validated': result.metrics.get('total_files', 0),
                'total_violations': len(all_violations),
                'execution_time_ms': result.execution_time_ms
            },
            artifacts=artifacts
        )

        self.workflow_history.append(workflow_result)
        return workflow_result

    async def _execute_sprint_completion_workflow(self, request: WorkflowRequest) -> WorkflowResult:
        """Execute sprint completion workflow"""
        self.logger.info(f"Executing sprint completion workflow: {request.request_id}")

        validation_results = []
        all_violations = []
        artifacts = {}

        # Validate sprint completion
        result = await self.sprint_validator.validate_sprint_completion(request.target)
        validation_results.append(result)
        all_violations.extend(result.violations)

        # Generate artifacts
        artifacts['sprint_report'] = self._save_artifact(
            request.request_id, 'sprint_report',
            self.core_qa.export_validation_report(result)
        )

        # Generate sprint summary
        sprint_summary = self._generate_sprint_summary(request.target, result)
        artifacts['sprint_summary'] = self._save_artifact(
            request.request_id, 'sprint_summary',
            json.dumps(sprint_summary, indent=2)
        )

        workflow_result = WorkflowResult(
            request_id=request.request_id,
            stage=ValidationStage.SPRINT_COMPLETION,
            trigger=request.trigger,
            passed=result.passed,
            validation_results=validation_results,
            all_violations=all_violations,
            execution_summary={
                'sprint_id': result.metrics.get('sprint_id', ''),
                'goal_completion_rate': result.metrics.get('goal_completion_rate', 0),
                'quality_score': result.metrics.get('quality_score', 0),
                'total_violations': len(all_violations)
            },
            artifacts=artifacts
        )

        self.workflow_history.append(workflow_result)
        return workflow_result

    async def _execute_git_commit_workflow(self, request: WorkflowRequest) -> WorkflowResult:
        """Execute git commit workflow"""
        self.logger.info(f"Executing git commit workflow: {request.request_id}")

        validation_results = []
        all_violations = []
        artifacts = {}

        # Validate commits
        result = await self.commit_validator.validate_commits(request.target)
        validation_results.append(result)
        all_violations.extend(result.violations)

        # Generate artifacts
        artifacts['commit_report'] = self._save_artifact(
            request.request_id, 'commit_report',
            self.commit_validator.qa_orchestrator.export_validation_report(result)
        )

        workflow_result = WorkflowResult(
            request_id=request.request_id,
            stage=ValidationStage.GIT_COMMIT,
            trigger=request.trigger,
            passed=result.passed,
            validation_results=validation_results,
            all_violations=all_violations,
            execution_summary={
                'commits_validated': result.metrics.get('total_commits', 0),
                'validation_scope': result.metrics.get('validation_scope', ''),
                'total_violations': len(all_violations)
            },
            artifacts=artifacts
        )

        self.workflow_history.append(workflow_result)
        return workflow_result

    async def execute_parallel_workflows(self, requests: List[WorkflowRequest]) -> List[WorkflowResult]:
        """Execute multiple workflows in parallel"""
        if not self.parallel_execution:
            # Execute sequentially
            results = []
            for request in requests:
                result = await self.execute_workflow(request)
                results.append(result)
            return results

        # Execute in parallel with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent_workflows)

        async def execute_with_semaphore(request):
            async with semaphore:
                return await self.execute_workflow(request)

        tasks = [execute_with_semaphore(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Workflow {requests[i].request_id} failed with exception: {result}")
                processed_results.append(WorkflowResult(
                    request_id=requests[i].request_id,
                    stage=requests[i].stage,
                    trigger=requests[i].trigger,
                    passed=False,
                    execution_summary={'error': str(result)}
                ))
            else:
                processed_results.append(result)

        return processed_results

    def register_hook(self, event: str, callback: Callable):
        """Register a workflow hook"""
        if event not in self.workflow_hooks:
            self.workflow_hooks[event] = []
        self.workflow_hooks[event].append(callback)

    async def _execute_hooks(self, event: str, request: WorkflowRequest, result: WorkflowResult = None):
        """Execute workflow hooks"""
        if event in self.workflow_hooks:
            for hook in self.workflow_hooks[event]:
                try:
                    if result:
                        await hook(request, result)
                    else:
                        await hook(request)
                except Exception as e:
                    self.logger.error(f"Hook {event} failed: {str(e)}")

    def _save_artifact(self, request_id: str, artifact_type: str, content: str) -> str:
        """Save workflow artifact to file"""
        artifacts_dir = Path("qa_artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{request_id}_{artifact_type}_{timestamp}.json"
        file_path = artifacts_dir / filename

        file_path.write_text(content, encoding='utf-8')
        return str(file_path)

    def _generate_sprint_summary(self, report: SprintCompletionReport, result: ValidationResult) -> Dict[str, Any]:
        """Generate sprint summary"""
        return {
            'sprint_info': {
                'id': report.sprint.sprint_id,
                'name': report.sprint.name,
                'duration_days': (report.sprint.end_date - report.sprint.start_date).days
            },
            'completion_status': {
                'passed': result.passed,
                'goals_completed': result.metrics.get('completed_goals', 0),
                'total_goals': result.metrics.get('total_goals', 0),
                'completion_rate': result.metrics.get('goal_completion_rate', 0)
            },
            'quality_metrics': {
                'quality_score': result.metrics.get('quality_score', 0),
                'test_pass_rate': result.metrics.get('test_pass_rate', 0),
                'documentation_coverage': result.metrics.get('documentation_coverage', 0)
            },
            'violations_summary': {
                'total_violations': len(result.violations),
                'critical_violations': len([v for v in result.violations if v.severity == ValidationSeverity.CRITICAL]),
                'error_violations': len([v for v in result.violations if v.severity == ValidationSeverity.ERROR])
            }
        }

    def _update_performance_metrics(self, result: WorkflowResult):
        """Update performance metrics"""
        self.performance_metrics['total_workflows'] += 1

        if result.passed:
            self.performance_metrics['successful_workflows'] += 1
        else:
            self.performance_metrics['failed_workflows'] += 1

        if result.end_time:
            execution_time = (result.end_time - result.start_time).total_seconds() * 1000
            current_avg = self.performance_metrics['average_execution_time_ms']
            total_workflows = self.performance_metrics['total_workflows']

            # Calculate new average
            self.performance_metrics['average_execution_time_ms'] = (
                (current_avg * (total_workflows - 1) + execution_time) / total_workflows
            )

        self.performance_metrics['last_execution'] = datetime.now()

    def get_workflow_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific workflow"""
        # Check active workflows
        if request_id in self.active_workflows:
            request = self.active_workflows[request_id]
            return {
                'status': 'running',
                'request_id': request_id,
                'stage': request.stage.value,
                'trigger': request.trigger.value,
                'start_time': request.timestamp,
                'requester': request.requester
            }

        # Check workflow history
        for result in self.workflow_history:
            if result.request_id == request_id:
                return {
                    'status': 'completed',
                    'request_id': request_id,
                    'stage': result.stage.value,
                    'trigger': result.trigger.value,
                    'passed': result.passed,
                    'start_time': result.start_time,
                    'end_time': result.end_time,
                    'total_violations': len(result.all_violations)
                }

        return None

    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """Get all active workflows"""
        return [
            {
                'request_id': request_id,
                'stage': request.stage.value,
                'trigger': request.trigger.value,
                'start_time': request.timestamp,
                'requester': request.requester,
                'priority': request.priority
            }
            for request_id, request in self.active_workflows.items()
        ]

    def get_workflow_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get workflow history"""
        history = sorted(self.workflow_history, key=lambda x: x.start_time, reverse=True)
        return [
            {
                'request_id': result.request_id,
                'stage': result.stage.value,
                'trigger': result.trigger.value,
                'passed': result.passed,
                'start_time': result.start_time,
                'end_time': result.end_time,
                'total_violations': len(result.all_violations),
                'execution_time_ms': result.execution_summary.get('execution_time_ms', 0)
            }
            for result in history[:limit]
        ]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.performance_metrics.copy()

    def create_workflow_request(
        self,
        trigger: WorkflowTrigger,
        stage: ValidationStage,
        target: Any,
        requester: str = "",
        config: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> WorkflowRequest:
        """Create a workflow request"""
        import uuid

        return WorkflowRequest(
            request_id=str(uuid.uuid4()),
            trigger=trigger,
            stage=stage,
            target=target,
            config=config or {},
            metadata=metadata or {},
            requester=requester
        )


# Factory function
def create_qa_workflow_orchestrator(config: Dict[str, Any] = None) -> QAWorkflowOrchestrator:
    """Create a QA workflow orchestrator with default configuration"""
    default_config = {
        'parallel_execution': True,
        'max_concurrent_workflows': 3,
        'default_timeout_seconds': 300,
        'file_submission': {
            'max_file_size_mb': 10,
            'max_files_per_batch': 50
        },
        'sprint_completion': {
            'min_test_coverage': 0.95,
            'min_goal_completion': 0.90
        },
        'git_commit': {
            'max_commit_size_mb': 5,
            'forbidden_patterns': [
                r'password',
                r'secret.*key',
                r'api_key'
            ]
        }
    }

    if config:
        for section, values in config.items():
            if section in default_config and isinstance(default_config[section], dict):
                default_config[section].update(values)
            else:
                default_config[section] = values

    return QAWorkflowOrchestrator(default_config)