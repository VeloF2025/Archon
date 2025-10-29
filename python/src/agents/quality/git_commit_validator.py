"""
Git Commit Quality Assurance Validator
Enhanced commit validation with QA framework integration
"""

import asyncio
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
import json
from datetime import datetime
import git

from .qa_framework import (
    QAOrchestrator,
    ValidationStage,
    ValidationResult,
    ValidationViolation,
    ValidationSeverity,
    create_qa_orchestrator
)


@dataclass
class GitCommit:
    """Represents a git commit"""
    commit_hash: str
    author: str
    email: str
    message: str
    timestamp: datetime
    files_changed: List[str] = field(default_factory=list)
    file_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)  # filename -> {additions, deletions}
    branch: str = ""
    parent_commits: List[str] = field(default_factory=list)


@dataclass
class CommitValidationRequest:
    """Represents a commit validation request"""
    commits: List[GitCommit] = field(default_factory=list)
    validation_scope: str = "all"  # "all", "staged", "last_commit", "commit_range"
    target_branch: str = "main"
    validation_config: Dict[str, Any] = field(default_factory=dict)


class GitCommitValidator:
    """Validates git commits using QA framework"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.qa_orchestrator = create_qa_orchestrator(config)
        self.repo_path = Path(self.config.get('repo_path', '.'))
        self.forbidden_patterns = self.config.get('forbidden_patterns', [
            r'password',
            r'secret.*key',
            r'api_key',
            r'token.*=',
            r'credential'
        ])
        self.required_commit_sections = self.config.get('required_commit_sections', [])
        self.max_commit_size_mb = self.config.get('max_commit_size_mb', 5)

    async def validate_commits(self, request: CommitValidationRequest) -> ValidationResult:
        """Validate git commits"""
        start_time = datetime.now()
        all_violations = []
        all_metrics = {}

        try:
            # Get commits based on validation scope
            commits = self._get_commits_to_validate(request)
            request.commits = commits

            if not commits:
                all_violations.append(ValidationViolation(
                    rule='no_commits_to_validate',
                    message='No commits found to validate',
                    severity=ValidationSeverity.WARNING
                ))
            else:
                # Validate each commit
                for commit in commits:
                    commit_violations = await self._validate_single_commit(commit)
                    all_violations.extend(commit_violations)

                # Validate commit batch
                batch_violations = self._validate_commit_batch(commits)
                all_violations.extend(batch_violations)

        except Exception as e:
            all_violations.append(ValidationViolation(
                rule='commit_validation_error',
                message=f'Error during commit validation: {str(e)}',
                severity=ValidationSeverity.CRITICAL
            ))

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        passed = len([v for v in all_violations if v.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]) == 0

        return ValidationResult(
            stage=ValidationStage.GIT_COMMIT,
            passed=passed,
            violations=all_violations,
            metrics={
                'execution_time_ms': int(execution_time),
                'total_commits': len(commits),
                'total_violations': len(all_violations),
                'critical_violations': len([v for v in all_violations if v.severity == ValidationSeverity.CRITICAL]),
                'error_violations': len([v for v in all_violations if v.severity == ValidationSeverity.ERROR]),
                'validation_scope': request.validation_scope,
                'target_branch': request.target_branch
            }
        )

    async def _validate_single_commit(self, commit: GitCommit) -> List[ValidationViolation]:
        """Validate a single commit"""
        violations = []

        # Validate commit message
        message_violations = self._validate_commit_message(commit)
        violations.extend(message_violations)

        # Validate commit files
        if commit.files_changed:
            file_violations = await self._validate_commit_files(commit.files_changed)
            violations.extend(file_violations)

        # Validate commit size
        size_violations = self._validate_commit_size(commit)
        violations.extend(size_violations)

        # Validate commit metadata
        metadata_violations = self._validate_commit_metadata(commit)
        violations.extend(metadata_violations)

        return violations

    def _validate_commit_message(self, commit: GitCommit) -> List[ValidationViolation]:
        """Validate commit message format and content"""
        violations = []
        message = commit.message.strip()

        if not message:
            violations.append(ValidationViolation(
                rule='empty_commit_message',
                message='Commit message cannot be empty',
                severity=ValidationSeverity.ERROR,
                suggestion='Add a descriptive commit message'
            ))
            return violations

        # Check message length
        if len(message) > 2000:
            violations.append(ValidationViolation(
                rule='commit_message_too_long',
                message=f'Commit message too long: {len(message)} characters (max 2000)',
                severity=ValidationSeverity.WARNING,
                suggestion='Keep commit messages concise and detailed'
            ))

        # Check for proper subject line (first line)
        lines = message.split('\n')
        subject = lines[0] if lines else ""

        if len(subject) > 72:
            violations.append(ValidationViolation(
                rule='commit_subject_too_long',
                message=f'Commit subject too long: {len(subject)} characters (max 72)',
                severity=ValidationSeverity.WARNING,
                suggestion='Keep commit subject under 72 characters'
            ))

        # Check for forbidden patterns in commit message
        for pattern in self.forbidden_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                violations.append(ValidationViolation(
                    rule='sensitive_data_in_message',
                    message=f'Potential sensitive data in commit message: {pattern}',
                    severity=ValidationSeverity.CRITICAL,
                    suggestion='Remove sensitive information from commit message'
                ))

        # Check for merge commit indicators
        if message.startswith('Merge') or 'Merge branch' in message:
            violations.append(ValidationViolation(
                rule='merge_commit_detected',
                message='Merge commit detected - should be validated separately',
                severity=ValidationSeverity.INFO,
                suggestion='Consider using pull requests for merges'
            ))

        return violations

    async def _validate_commit_files(self, file_paths: List[str]) -> List[ValidationViolation]:
        """Validate files in commit"""
        violations = []

        # Run QA validation on all files
        qa_result = self.qa_orchestrator.validate(
            file_paths,
            ValidationStage.GIT_COMMIT
        )

        violations.extend(qa_result.violations)

        # Additional commit-specific file validations
        for file_path in file_paths:
            path = Path(file_path)

            # Check for sensitive files
            if self._is_sensitive_file(path):
                violations.append(ValidationViolation(
                    rule='sensitive_file_committed',
                    message=f'Sensitive file committed: {path.name}',
                    severity=ValidationSeverity.CRITICAL,
                    file_path=str(path),
                    suggestion='Remove sensitive files from commits'
                ))

            # Check file size in commit
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                if size_mb > self.max_commit_size_mb:
                    violations.append(ValidationViolation(
                        rule='large_file_committed',
                        message=f'Large file committed: {path.name} ({size_mb:.2f}MB)',
                        severity=ValidationSeverity.WARNING,
                        file_path=str(path),
                        suggestion='Consider using Git LFS for large files'
                    ))

        return violations

    def _validate_commit_size(self, commit: GitCommit) -> List[ValidationViolation]:
        """Validate commit size and complexity"""
        violations = []

        # Calculate total changes
        total_additions = sum(stats.get('additions', 0) for stats in commit.file_stats.values())
        total_deletions = sum(stats.get('deletions', 0) for stats in commit.file_stats.values())
        total_changes = total_additions + total_deletions

        # Check for unusually large commits
        if total_changes > 5000:
            violations.append(ValidationViolation(
                rule='large_commit',
                message=f'Large commit: {total_changes} changes (max recommended: 5000)',
                severity=ValidationSeverity.WARNING,
                suggestion='Break large commits into smaller, focused commits'
            ))

        # Check for commits with too many files
        if len(commit.files_changed) > 50:
            violations.append(ValidationViolation(
                rule='many_files_changed',
                message=f'Too many files changed: {len(commit.files_changed)} (max recommended: 50)',
                severity=ValidationSeverity.WARNING,
                suggestion='Group related changes in separate commits'
            ))

        return violations

    def _validate_commit_metadata(self, commit: GitCommit) -> List[ValidationViolation]:
        """Validate commit metadata"""
        violations = []

        # Check author email
        if not commit.email or '@' not in commit.email:
            violations.append(ValidationViolation(
                rule='invalid_author_email',
                message=f'Invalid author email: {commit.email}',
                severity=ValidationSeverity.WARNING,
                suggestion='Configure proper git user email'
            ))

        # Check for commits too far in future/past
        now = datetime.now()
        time_diff = abs((now - commit.timestamp).total_seconds())

        if time_diff > 86400:  # More than 1 day
            violations.append(ValidationViolation(
                rule='unusual_commit_timestamp',
                message=f'Unusual commit timestamp: {commit.timestamp}',
                severity=ValidationSeverity.INFO,
                suggestion='Check system clock and git configuration'
            ))

        return violations

    def _validate_commit_batch(self, commits: List[GitCommit]) -> List[ValidationViolation]:
        """Validate commit batch as a whole"""
        violations = []

        if len(commits) == 0:
            return violations

        # Check for duplicate commits
        commit_hashes = [c.commit_hash for c in commits]
        if len(commit_hashes) != len(set(commit_hashes)):
            violations.append(ValidationViolation(
                rule='duplicate_commits',
                message='Duplicate commits found in validation scope',
                severity=ValidationSeverity.WARNING
            ))

        # Check commit sequence
        for i in range(1, len(commits)):
            prev_commit = commits[i-1]
            curr_commit = commits[i]

            if curr_commit.timestamp < prev_commit.timestamp:
                violations.append(ValidationViolation(
                    rule='commit_sequence_error',
                    message=f'Commit sequence error: {curr_commit.commit_hash} appears before {prev_commit.commit_hash}',
                    severity=ValidationSeverity.WARNING
                ))

        return violations

    def _is_sensitive_file(self, file_path: Path) -> bool:
        """Check if file is sensitive"""
        sensitive_patterns = [
            r'\.env',
            r'\.key$',
            r'\.pem$',
            r'\.p12$',
            r'\.pfx$',
            r'id_rsa',
            r'id_dsa',
            r'\.pgp$',
            r'secret',
            r'password',
            r'credential'
        ]

        file_name = file_path.name.lower()
        return any(re.search(pattern, file_name) for pattern in sensitive_patterns)

    def _get_commits_to_validate(self, request: CommitValidationRequest) -> List[GitCommit]:
        """Get commits to validate based on scope"""
        try:
            repo = git.Repo(self.repo_path)

            if request.validation_scope == "staged":
                return self._get_staged_commits(repo)
            elif request.validation_scope == "last_commit":
                return [self._get_last_commit(repo)]
            elif request.validation_scope == "commit_range":
                return self._get_commit_range(repo, request)
            else:  # "all"
                return self._get_all_commits(repo, request)

        except Exception as e:
            raise ValueError(f"Error accessing git repository: {str(e)}")

    def _get_staged_commits(self, repo: git.Repo) -> List[GitCommit]:
        """Get staged changes as a virtual commit"""
        # For staged changes, we create a virtual commit representation
        staged_files = [item.a_path for item in repo.index.diff(None)]
        staged_files.extend([item.a_path for item in repo.index.diff(repo.head.commit)])

        virtual_commit = GitCommit(
            commit_hash="STAGED",
            author=repo.config.reader().get_value("user", "name", "Unknown"),
            email=repo.config.reader().get_value("user", "email", "unknown@example.com"),
            message="Staged changes",
            timestamp=datetime.now(),
            files_changed=staged_files,
            branch=repo.active_branch.name,
            parent_commits=[repo.head.commit.hexsha]
        )

        return [virtual_commit]

    def _get_last_commit(self, repo: git.Repo) -> GitCommit:
        """Get the last commit"""
        commit = repo.head.commit
        return self._convert_git_commit(commit)

    def _get_commit_range(self, repo: git.Repo, request: CommitValidationRequest) -> List[GitCommit]:
        """Get commits in a range"""
        # This would need to be implemented based on specific range requirements
        # For now, return last 10 commits
        commits = []
        for commit in list(repo.iter_commits('HEAD', max_count=10)):
            commits.append(self._convert_git_commit(commit))
        return commits

    def _get_all_commits(self, repo: git.Repo, request: CommitValidationRequest) -> List[GitCommit]:
        """Get all commits (limited to recent history for performance)"""
        commits = []
        for commit in list(repo.iter_commits('HEAD', max_count=50)):
            commits.append(self._convert_git_commit(commit))
        return commits

    def _convert_git_commit(self, commit) -> GitCommit:
        """Convert git.Commit to GitCommit"""
        files_changed = []
        file_stats = {}

        try:
            # Get changed files and stats
            if commit.parents:
                diff = commit.parents[0].diff(commit, create_patch=True)
                for diff_item in diff:
                    file_path = diff_item.a_path or diff_item.b_path
                    files_changed.append(file_path)
                    file_stats[file_path] = {
                        'additions': diff_item.diff.count(b'\n+') if diff_item.diff else 0,
                        'deletions': diff_item.diff.count(b'\n-') if diff_item.diff else 0
                    }
        except Exception:
            # Fallback for cases where diff calculation fails
            pass

        return GitCommit(
            commit_hash=commit.hexsha,
            author=commit.author.name,
            email=commit.author.email,
            message=commit.message,
            timestamp=datetime.fromtimestamp(commit.committed_date),
            files_changed=files_changed,
            file_stats=file_stats,
            branch=str(commit.repo.active_branch) if hasattr(commit, 'repo') else "",
            parent_commits=[p.hexsha for p in commit.parents]
        )

    async def validate_staged_changes(self, repo_path: str = ".") -> ValidationResult:
        """Validate staged changes - common use case"""
        request = CommitValidationRequest(
            validation_scope="staged"
        )
        self.repo_path = Path(repo_path)
        return await self.validate_commits(request)

    async def validate_last_commit(self, repo_path: str = ".") -> ValidationResult:
        """Validate last commit - common use case"""
        request = CommitValidationRequest(
            validation_scope="last_commit"
        )
        self.repo_path = Path(repo_path)
        return await self.validate_commits(request)


# CLI interface for standalone usage
async def main():
    """CLI interface for git commit validator"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Validate git commits with QA')
    parser.add_argument('--scope', choices=['all', 'staged', 'last_commit', 'commit_range'],
                       default='staged', help='Validation scope')
    parser.add_argument('--repo-path', default='.', help='Git repository path')
    parser.add_argument('--output', help='Output JSON report file')
    parser.add_argument('--config', help='QA configuration file')

    args = parser.parse_args()

    # Load configuration
    config = {}
    if args.config and Path(args.config).exists():
        config = json.loads(Path(args.config).read_text())

    # Create validator and run validation
    validator = GitCommitValidator({**config, 'repo_path': args.repo_path})

    if args.scope == 'staged':
        result = await validator.validate_staged_changes(args.repo_path)
    elif args.scope == 'last_commit':
        result = await validator.validate_last_commit(args.repo_path)
    else:
        request = CommitValidationRequest(validation_scope=args.scope)
        result = await validator.validate_commits(request)

    # Export report
    report = validator.qa_orchestrator.export_validation_report(result, args.output)

    # Print summary
    print(f"Git Commit Validation: {'PASSED' if result.passed else 'FAILED'}")
    print(f"Total Violations: {len(result.violations)}")
    print(f"Execution Time: {result.execution_time_ms}ms")
    print(f"Commits Validated: {result.metrics.get('total_commits', 0)}")

    if args.output:
        print(f"Report saved to: {args.output}")

    # Exit with appropriate code
    sys.exit(0 if result.passed else 1)


if __name__ == '__main__':
    asyncio.run(main())