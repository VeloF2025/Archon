"""
File Submission Quality Assurance Validator
Implements QA validation for file submission stage
"""

import asyncio
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
import json
import mimetypes
from datetime import datetime

from .qa_framework import (
    QAOrchestrator,
    ValidationStage,
    ValidationResult,
    ValidationViolation,
    ValidationSeverity,
    create_qa_orchestrator
)


@dataclass
class FileSubmission:
    """Represents a file submission for QA validation"""
    file_path: str
    content: str = ""
    file_hash: str = ""
    file_type: str = ""
    size_bytes: int = 0
    submission_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.content and Path(self.file_path).exists():
            self.content = Path(self.file_path).read_text(encoding='utf-8')

        if self.content:
            self.file_hash = hashlib.sha256(self.content.encode()).hexdigest()
            self.size_bytes = len(self.content.encode('utf-8'))

        self.file_type = mimetypes.guess_type(self.file_path)[0] or "unknown"


@dataclass
class SubmissionBatch:
    """Represents a batch of file submissions"""
    submissions: List[FileSubmission] = field(default_factory=list)
    batch_id: str = ""
    submitter: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    submission_context: Dict[str, Any] = field(default_factory=dict)


class FileSubmissionValidator:
    """Validates file submissions using QA framework"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.qa_orchestrator = create_qa_orchestrator(config)
        self.supported_file_types = {
            '.py': 'python',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.md': 'markdown',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml'
        }
        self.max_file_size_mb = self.config.get('max_file_size_mb', 10)
        self.forbidden_files = self.config.get('forbidden_files', [
            '.env', '.env.local', '.env.production', 'id_rsa', '.pem'
        ])

    async def validate_submission(self, submission: FileSubmission) -> ValidationResult:
        """Validate a single file submission"""
        # Pre-validation checks
        pre_violations = self._pre_validate_submission(submission)
        if pre_violations:
            return ValidationResult(
                stage=ValidationStage.FILE_SUBMISSION,
                passed=False,
                violations=pre_violations,
                metrics={'pre_validation_failures': len(pre_violations)}
            )

        # Run QA validation
        qa_result = self.qa_orchestrator.validate(
            submission.file_path,
            ValidationStage.FILE_SUBMISSION
        )

        # Add file-specific validation
        file_violations = self._validate_file_specific_rules(submission)
        qa_result.violations.extend(file_violations)

        # Update metrics
        qa_result.metrics.update({
            'file_size_bytes': submission.size_bytes,
            'file_type': submission.file_type,
            'file_hash': submission.file_hash,
            'file_specific_violations': len(file_violations)
        })

        return qa_result

    async def validate_batch(self, batch: SubmissionBatch) -> ValidationResult:
        """Validate a batch of file submissions"""
        start_time = datetime.now()
        all_violations = []
        all_metrics = {}
        passed = True

        # Validate each submission
        for submission in batch.submissions:
            try:
                result = await self.validate_submission(submission)
                all_violations.extend(result.violations)
                all_metrics[submission.file_path] = result.metrics

                if not result.passed:
                    passed = False

            except Exception as e:
                all_violations.append(ValidationViolation(
                    rule='submission_validation_error',
                    message=f'Error validating submission {submission.file_path}: {str(e)}',
                    severity=ValidationSeverity.CRITICAL,
                    file_path=submission.file_path
                ))
                passed = False

        # Batch-level validation
        batch_violations = self._validate_batch_rules(batch)
        all_violations.extend(batch_violations)

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        return ValidationResult(
            stage=ValidationStage.FILE_SUBMISSION,
            passed=passed and len(batch_violations) == 0,
            violations=all_violations,
            metrics={
                'execution_time_ms': int(execution_time),
                'batch_id': batch.batch_id,
                'total_files': len(batch.submissions),
                'total_violations': len(all_violations),
                'critical_violations': len([v for v in all_violations if v.severity == ValidationSeverity.CRITICAL]),
                'submission_context': batch.submission_context,
                **all_metrics
            }
        )

    def _pre_validate_submission(self, submission: FileSubmission) -> List[ValidationViolation]:
        """Pre-validate submission before QA checks"""
        violations = []
        file_path = Path(submission.file_path)

        # Check file existence
        if not file_path.exists():
            violations.append(ValidationViolation(
                rule='file_not_found',
                message=f'File does not exist: {submission.file_path}',
                severity=ValidationSeverity.CRITICAL,
                file_path=submission.file_path
            ))
            return violations

        # Check file size
        size_mb = submission.size_bytes / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            violations.append(ValidationViolation(
                rule='file_too_large',
                message=f'File size {size_mb:.2f}MB exceeds maximum {self.max_file_size_mb}MB',
                severity=ValidationSeverity.ERROR,
                file_path=submission.file_path,
                suggestion='Compress file or split into smaller files'
            ))

        # Check forbidden files
        if any(forbidden in file_path.name for forbidden in self.forbidden_files):
            violations.append(ValidationViolation(
                rule='forbidden_file_type',
                message=f'Submission of forbidden file type: {file_path.name}',
                severity=ValidationSeverity.CRITICAL,
                file_path=submission.file_path,
                suggestion='Remove sensitive or forbidden files from submission'
            ))

        # Check file type support
        file_ext = file_path.suffix.lower()
        if file_ext not in self.supported_file_types:
            violations.append(ValidationViolation(
                rule='unsupported_file_type',
                message=f'Unsupported file type: {file_ext}',
                severity=ValidationSeverity.WARNING,
                file_path=submission.file_path,
                suggestion=f'Supported types: {", ".join(self.supported_file_types.keys())}'
            ))

        return violations

    def _validate_file_specific_rules(self, submission: FileSubmission) -> List[ValidationViolation]:
        """Validate file-specific rules"""
        violations = []
        file_path = Path(submission.file_path)
        file_ext = file_path.suffix.lower()

        # Language-specific validations
        if file_ext in ['.py']:
            violations.extend(self._validate_python_file(submission))
        elif file_ext in ['.ts', '.tsx', '.js', '.jsx']:
            violations.extend(self._validate_javascript_typescript_file(submission))
        elif file_ext == '.md':
            violations.extend(self._validate_markdown_file(submission))

        return violations

    def _validate_python_file(self, submission: FileSubmission) -> List[ValidationViolation]:
        """Validate Python-specific rules"""
        violations = []
        lines = submission.content.split('\n')

        # Check for problematic imports
        problematic_imports = ['eval', 'exec', 'compile', '__import__']
        for i, line in enumerate(lines, 1):
            for imp in problematic_imports:
                if f'import {imp}' in line or f'from {imp}' in line:
                    violations.append(ValidationViolation(
                        rule='dangerous_import',
                        message=f'Dangerous import detected: {imp}',
                        severity=ValidationSeverity.WARNING,
                        file_path=submission.file_path,
                        line_number=i,
                        code_snippet=line.strip(),
                        suggestion='Avoid dangerous imports or provide proper validation'
                    ))

        # Check for hardcoded credentials
        credential_patterns = ['password', 'secret', 'token', 'key', 'api_key']
        for i, line in enumerate(lines, 1):
            line_lower = line.lower()
            for pattern in credential_patterns:
                if f'{pattern}=' in line_lower and '"' in line:
                    violations.append(ValidationViolation(
                        rule='hardcoded_credentials',
                        message=f'Potential hardcoded credential detected: {pattern}',
                        severity=ValidationSeverity.CRITICAL,
                        file_path=submission.file_path,
                        line_number=i,
                        code_snippet=line.strip(),
                        suggestion='Use environment variables or secure credential management'
                    ))

        return violations

    def _validate_javascript_typescript_file(self, submission: FileSubmission) -> List[ValidationViolation]:
        """Validate JavaScript/TypeScript-specific rules"""
        violations = []
        lines = submission.content.split('\n')

        # Check for eval usage
        for i, line in enumerate(lines, 1):
            if 'eval(' in line:
                violations.append(ValidationViolation(
                    rule='eval_usage',
                    message='eval() usage detected - potential security risk',
                    severity=ValidationSeverity.ERROR,
                    file_path=submission.file_path,
                    line_number=i,
                    code_snippet=line.strip(),
                    suggestion='Avoid eval() or provide proper input sanitization'
                ))

        # Check for innerHTML usage
        for i, line in enumerate(lines, 1):
            if 'innerHTML' in line:
                violations.append(ValidationViolation(
                    rule='inner_html_usage',
                    message='innerHTML usage detected - potential XSS risk',
                    severity=ValidationSeverity.WARNING,
                    file_path=submission.file_path,
                    line_number=i,
                    code_snippet=line.strip(),
                    suggestion='Use textContent or proper HTML sanitization'
                ))

        return violations

    def _validate_markdown_file(self, submission: FileSubmission) -> List[ValidationViolation]:
        """Validate Markdown-specific rules"""
        violations = []
        lines = submission.content.split('\n')

        # Check for broken links
        for i, line in enumerate(lines, 1):
            if '[text](url)' in line and 'http' not in line:
                violations.append(ValidationViolation(
                    rule='broken_link_format',
                    message='Potential broken link in markdown',
                    severity=ValidationSeverity.INFO,
                    file_path=submission.file_path,
                    line_number=i,
                    code_snippet=line.strip(),
                    suggestion='Verify link format and accessibility'
                ))

        return violations

    def _validate_batch_rules(self, batch: SubmissionBatch) -> List[ValidationViolation]:
        """Validate batch-level rules"""
        violations = []

        # Check for duplicate files
        file_paths = [s.file_path for s in batch.submissions]
        if len(file_paths) != len(set(file_paths)):
            violations.append(ValidationViolation(
                rule='duplicate_files',
                message='Duplicate files detected in batch',
                severity=ValidationSeverity.ERROR,
                suggestion='Remove duplicate files from submission'
            ))

        # Check batch size limits
        max_files_per_batch = self.config.get('max_files_per_batch', 50)
        if len(batch.submissions) > max_files_per_batch:
            violations.append(ValidationViolation(
                rule='batch_too_large',
                message=f'Batch size {len(batch.submissions)} exceeds maximum {max_files_per_batch}',
                severity=ValidationSeverity.WARNING,
                suggestion='Split large submissions into smaller batches'
            ))

        return violations

    def create_submission_batch(self, file_paths: List[str], submitter: str = "", context: Dict[str, Any] = None) -> SubmissionBatch:
        """Create a submission batch from file paths"""
        submissions = []

        for file_path in file_paths:
            if Path(file_path).exists():
                submissions.append(FileSubmission(file_path=file_path))

        batch_id = hashlib.md5(
            f"{submitter}_{datetime.now().isoformat()}_{len(submissions)}".encode()
        ).hexdigest()[:12]

        return SubmissionBatch(
            submissions=submissions,
            batch_id=batch_id,
            submitter=submitter,
            timestamp=datetime.now(),
            submission_context=context or {}
        )

    async def validate_paths(self, file_paths: List[str], submitter: str = "", context: Dict[str, Any] = None) -> ValidationResult:
        """Convenience method to validate file paths directly"""
        batch = self.create_submission_batch(file_paths, submitter, context)
        return await self.validate_batch(batch)


# CLI interface for standalone usage
async def main():
    """CLI interface for file submission validator"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Validate file submissions with QA')
    parser.add_argument('files', nargs='+', help='Files to validate')
    parser.add_argument('--submitter', default='', help='Submitter name')
    parser.add_argument('--output', help='Output JSON report file')
    parser.add_argument('--config', help='Configuration file')

    args = parser.parse_args()

    # Load configuration
    config = {}
    if args.config and Path(args.config).exists():
        config = json.loads(Path(args.config).read_text())

    # Create validator and run validation
    validator = FileSubmissionValidator(config)
    result = await validator.validate_paths(args.files, args.submitter)

    # Export report
    report = validator.qa_orchestrator.export_validation_report(result, args.output)

    # Print summary
    print(f"Validation Result: {'PASSED' if result.passed else 'FAILED'}")
    print(f"Total Violations: {len(result.violations)}")
    print(f"Execution Time: {result.execution_time_ms}ms")

    if args.output:
        print(f"Report saved to: {args.output}")

    # Exit with appropriate code
    sys.exit(0 if result.passed else 1)


if __name__ == '__main__':
    asyncio.run(main())