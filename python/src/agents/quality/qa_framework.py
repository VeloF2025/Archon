"""
Standard Archon Quality Assurance Framework
Implements unified ZT, DGTS, and NLNH validation workflows
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationStage(Enum):
    """QA validation stages"""
    FILE_SUBMISSION = "file_submission"
    SPRINT_COMPLETION = "sprint_completion"
    GIT_COMMIT = "git_commit"
    CI_PIPELINE = "ci_pipeline"


class ValidationSeverity(Enum):
    """Validation violation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationViolation:
    """Represents a QA validation violation"""
    rule: str
    message: str
    severity: ValidationSeverity
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of a QA validation"""
    stage: ValidationStage
    passed: bool
    violations: List[ValidationViolation] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QualityGate:
    """Defines a quality gate for validation"""
    name: str
    description: str
    validator_class: str
    enabled: bool = True
    blocking: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


class QualityAssuranceAgent(ABC):
    """Abstract base class for QA validation agents"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def validate(self, target: Any, stage: ValidationStage) -> ValidationResult:
        """Perform validation for the given target and stage"""
        pass

    @abstractmethod
    def get_quality_gates(self) -> List[QualityGate]:
        """Return the quality gates this agent enforces"""
        pass


class ZTValidator(QualityAssuranceAgent):
    """Zero Tolerance validation agent"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.zero_tolerance_rules = {
            'typescript_errors': 0,
            'eslint_errors': 0,
            'eslint_warnings': 0,
            'console_log_statements': 0,
            'undefined_error_refs': 0,
            'build_failures': 0
        }

    def validate(self, target: Any, stage: ValidationStage) -> ValidationResult:
        """Perform zero tolerance validation"""
        start_time = datetime.now()
        violations = []

        if isinstance(target, (str, Path)) and Path(target).exists():
            violations.extend(self._validate_file(target))
        elif isinstance(target, list):
            for item in target:
                if isinstance(item, (str, Path)) and Path(item).exists():
                    violations.extend(self._validate_file(item))

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        return ValidationResult(
            stage=stage,
            passed=len([v for v in violations if v.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]) == 0,
            violations=violations,
            execution_time_ms=int(execution_time),
            metrics={
                'total_violations': len(violations),
                'critical_violations': len([v for v in violations if v.severity == ValidationSeverity.CRITICAL]),
                'error_violations': len([v for v in violations if v.severity == ValidationSeverity.ERROR])
            }
        )

    def _validate_file(self, file_path: str) -> List[ValidationViolation]:
        """Validate a single file for zero tolerance violations"""
        violations = []
        path = Path(file_path)

        try:
            content = path.read_text(encoding='utf-8')
            lines = content.split('\n')

            # Check for console.log statements
            for i, line in enumerate(lines, 1):
                if 'console.log' in line:
                    violations.append(ValidationViolation(
                        rule='no_console_log',
                        message='Console.log statements are not allowed in production code',
                        severity=ValidationSeverity.CRITICAL,
                        file_path=str(path),
                        line_number=i,
                        code_snippet=line.strip(),
                        suggestion='Remove console.log or use proper logging system'
                    ))

            # Check for undefined error references in catch blocks
            for i, line in enumerate(lines, 1):
                if 'catch (' in line and ')' not in line.split('catch (')[1].strip():
                    violations.append(ValidationViolation(
                        rule='defined_error_parameters',
                        message='Catch blocks must have defined error parameters',
                        severity=ValidationSeverity.ERROR,
                        file_path=str(path),
                        line_number=i,
                        code_snippet=line.strip(),
                        suggestion='Define error parameter: catch (error: unknown)'
                    ))

            # Check for TypeScript files
            if path.suffix in ['.ts', '.tsx']:
                violations.extend(self._validate_typescript_file(path, content))

        except Exception as e:
            self.logger.error(f"Error validating file {file_path}: {e}")
            violations.append(ValidationViolation(
                rule='file_access_error',
                message=f'Could not read or parse file: {str(e)}',
                severity=ValidationSeverity.ERROR,
                file_path=str(path)
            ))

        return violations

    def _validate_typescript_file(self, path: Path, content: str) -> List[ValidationViolation]:
        """Validate TypeScript-specific rules"""
        violations = []

        # Check for 'any' types
        if ': any' in content or ':any' in content:
            violations.append(ValidationViolation(
                rule='no_any_types',
                message="'any' types are not allowed - use proper TypeScript typing",
                severity=ValidationSeverity.ERROR,
                file_path=str(path),
                suggestion='Replace with proper TypeScript types'
            ))

        # Check for type assertions without proper validation
        if ' as ' in content and ' as any' in content:
            violations.append(ValidationViolation(
                rule='no_type_assertions',
                message='Type assertions should be avoided, especially "as any"',
                severity=ValidationSeverity.WARNING,
                file_path=str(path),
                suggestion='Use proper type guards or type-safe alternatives'
            ))

        return violations

    def get_quality_gates(self) -> List[QualityGate]:
        """Return zero tolerance quality gates"""
        return [
            QualityGate(
                name="no_console_log",
                description="Zero console.log statements allowed",
                validator_class="ZTValidator",
                blocking=True
            ),
            QualityGate(
                name="no_undefined_errors",
                description="All catch blocks must have defined error parameters",
                validator_class="ZTValidator",
                blocking=True
            ),
            QualityGate(
                name="no_any_types",
                description="No 'any' types allowed in TypeScript",
                validator_class="ZTValidator",
                blocking=True
            )
        ]


class DGTSValidator(QualityAssuranceAgent):
    """Don't Game The System validation agent"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.gaming_patterns = {
            'fake_implementations': [
                r'return\s+"mock"',
                r'return\s+"fake"',
                r'return\s+"stub"',
                r'#\s*TODO:\s*implement',
                r'pass\s*#\s*placeholder'
            ],
            'test_gaming': [
                r'assert\s+True',
                r'skip\s*=\s*True',
                r'@pytest\.mark\.skip',
                r'#\s*validation.*disabled'
            ],
            'validation_bypass': [
                r'#\s*validation.*skip',
                r'#\s*bypass.*check',
                r'if\s+False:\s*#.*disabled'
            ]
        }
        self.gaming_threshold = 0.3

    def validate(self, target: Any, stage: ValidationStage) -> ValidationResult:
        """Perform DGTS validation"""
        start_time = datetime.now()
        violations = []
        gaming_score = 0.0

        if isinstance(target, (str, Path)) and Path(target).exists():
            file_violations, file_score = self._validate_file_for_gaming(target)
            violations.extend(file_violations)
            gaming_score = max(gaming_score, file_score)
        elif isinstance(target, list):
            for item in target:
                if isinstance(item, (str, Path)) and Path(item).exists():
                    file_violations, file_score = self._validate_file_for_gaming(item)
                    violations.extend(file_violations)
                    gaming_score = max(gaming_score, file_score)

        # Block if gaming score exceeds threshold
        passed = gaming_score <= self.gaming_threshold
        if not passed:
            violations.append(ValidationViolation(
                rule='gaming_threshold_exceeded',
                message=f'Gaming score {gaming_score:.2f} exceeds threshold {self.gaming_threshold}',
                severity=ValidationSeverity.CRITICAL,
                suggestion='Review and remove gaming patterns before proceeding'
            ))

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        return ValidationResult(
            stage=stage,
            passed=passed,
            violations=violations,
            execution_time_ms=int(execution_time),
            metrics={
                'gaming_score': gaming_score,
                'gaming_threshold': self.gaming_threshold,
                'total_violations': len(violations),
                'gaming_patterns_detected': len([v for v in violations if 'gaming' in v.rule])
            }
        )

    def _validate_file_for_gaming(self, file_path: str) -> Tuple[List[ValidationViolation], float]:
        """Validate a file for gaming patterns"""
        violations = []
        path = Path(file_path)
        gaming_score = 0.0

        try:
            content = path.read_text(encoding='utf-8')
            lines = content.split('\n')

            # Check for gaming patterns
            for pattern_type, patterns in self.gaming_patterns.items():
                for pattern in patterns:
                    import re
                    for i, line in enumerate(lines, 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            violations.append(ValidationViolation(
                                rule=f'gaming_pattern_{pattern_type}',
                                message=f'Potential gaming pattern detected: {pattern_type}',
                                severity=ValidationSeverity.CRITICAL,
                                file_path=str(path),
                                line_number=i,
                                code_snippet=line.strip(),
                                suggestion='Replace with proper implementation'
                            ))
                            gaming_score += 0.1

            # Additional checks for suspicious patterns
            for i, line in enumerate(lines, 1):
                # Check for placeholder comments
                if any(placeholder in line.lower() for placeholder in ['todo', 'fixme', 'placeholder', 'implement']):
                    violations.append(ValidationViolation(
                        rule='placeholder_comment',
                        message='Placeholder comment detected - incomplete implementation',
                        severity=ValidationSeverity.WARNING,
                        file_path=str(path),
                        line_number=i,
                        code_snippet=line.strip(),
                        suggestion='Complete implementation or remove placeholder'
                    ))
                    gaming_score += 0.05

        except Exception as e:
            self.logger.error(f"Error validating file for gaming patterns {file_path}: {e}")

        return violations, min(gaming_score, 1.0)

    def get_quality_gates(self) -> List[QualityGate]:
        """Return DGTS quality gates"""
        return [
            QualityGate(
                name="no_gaming_patterns",
                description="No gaming or fake implementation patterns allowed",
                validator_class="DGTSValidator",
                blocking=True,
                config={'gaming_threshold': self.gaming_threshold}
            ),
            QualityGate(
                name="no_placeholder_comments",
                description="No TODO/FIXME placeholders in committed code",
                validator_class="DGTSValidator",
                blocking=False
            )
        ]


class NLNHValidator(QualityAssuranceAgent):
    """No Lies No Hallucination validation agent"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

    def validate(self, target: Any, stage: ValidationStage) -> ValidationResult:
        """Perform NLNH validation"""
        start_time = datetime.now()
        violations = []

        # NLNH validation focuses on truthfulness and documentation compliance
        if isinstance(target, dict) and 'documentation' in target:
            violations.extend(self._validate_documentation_compliance(target))
        elif isinstance(target, (str, Path)):
            violations.extend(self._validate_file_truthfulness(target))

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        return ValidationResult(
            stage=stage,
            passed=len([v for v in violations if v.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]) == 0,
            violations=violations,
            execution_time_ms=int(execution_time),
            metrics={
                'total_violations': len(violations),
                'documentation_violations': len([v for v in violations if 'documentation' in v.rule])
            }
        )

    def _validate_documentation_compliance(self, target: Dict[str, Any]) -> List[ValidationViolation]:
        """Validate documentation compliance"""
        violations = []

        required_docs = ['PRD', 'PRP', 'ADR']
        provided_docs = target.get('documentation', {})

        for doc_type in required_docs:
            if doc_type not in provided_docs:
                violations.append(ValidationViolation(
                    rule=f'missing_documentation_{doc_type.lower()}',
                    message=f'Missing required documentation: {doc_type}',
                    severity=ValidationSeverity.ERROR,
                    suggestion=f'Create {doc_type} document before proceeding'
                ))

        # Validate that tests match documentation
        if 'tests' in target and 'documentation' in target:
            test_coverage = self._validate_test_documentation_alignment(target['tests'], target['documentation'])
            if test_coverage < 0.95:
                violations.append(ValidationViolation(
                    rule='test_documentation_mismatch',
                    message=f'Test coverage ({test_coverage:.2%}) does not match documentation requirements',
                    severity=ValidationSeverity.ERROR,
                    suggestion='Ensure all documented requirements have corresponding tests'
                ))

        return violations

    def _validate_test_documentation_alignment(self, tests: List[str], documentation: Dict[str, Any]) -> float:
        """Calculate how well tests align with documentation"""
        # Simplified implementation - in real system would use NLP/semantic analysis
        documented_requirements = len(documentation.get('requirements', []))
        tested_requirements = len(tests)

        if documented_requirements == 0:
            return 1.0 if tested_requirements == 0 else 0.0

        return min(tested_requirements / documented_requirements, 1.0)

    def _validate_file_truthfulness(self, file_path: str) -> List[ValidationViolation]:
        """Validate file for truthfulness claims"""
        violations = []
        path = Path(file_path)

        try:
            content = path.read_text(encoding='utf-8')

            # Check for potentially false claims in comments
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                line_lower = line.lower().strip()
                if any(claim in line_lower for claim in ['fully implemented', 'complete', 'perfect', 'bug-free']):
                    violations.append(ValidationViolation(
                        rule='questionable_claim',
                        message='Potentially untrue claim found in code comments',
                        severity=ValidationSeverity.WARNING,
                        file_path=str(path),
                        line_number=i,
                        code_snippet=line.strip(),
                        suggestion='Remove absolute claims or provide evidence'
                    ))

        except Exception as e:
            self.logger.error(f"Error validating file truthfulness {file_path}: {e}")

        return violations

    def get_quality_gates(self) -> List[QualityGate]:
        """Return NLNH quality gates"""
        return [
            QualityGate(
                name="documentation_compliance",
                description="All required documentation must exist and be complete",
                validator_class="NLNHValidator",
                blocking=True
            ),
            QualityGate(
                name="test_documentation_alignment",
                description="Tests must align with documented requirements",
                validator_class="NLNHValidator",
                blocking=True
            ),
            QualityGate(
                name="no_questionable_claims",
                description="No absolute or unverified claims in code",
                validator_class="NLNHValidator",
                blocking=False
            )
        ]


class QAOrchestrator:
    """Orchestrates the QA validation process"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.agents = {
            'zt': ZTValidator(config.get('zt', {})),
            'dgts': DGTSValidator(config.get('dgts', {})),
            'nlnh': NLNHValidator(config.get('nlnh', {}))
        }
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate(self, target: Any, stage: ValidationStage, agents: List[str] = None) -> ValidationResult:
        """Run validation through specified agents"""
        if agents is None:
            agents = list(self.agents.keys())

        start_time = datetime.now()
        all_violations = []
        all_metrics = {}
        passed = True

        for agent_name in agents:
            if agent_name not in self.agents:
                self.logger.warning(f"Unknown QA agent: {agent_name}")
                continue

            agent = self.agents[agent_name]
            try:
                result = agent.validate(target, stage)
                all_violations.extend(result.violations)
                all_metrics[agent_name] = result.metrics

                if not result.passed:
                    passed = False

            except Exception as e:
                self.logger.error(f"Error in {agent_name} agent: {e}")
                all_violations.append(ValidationViolation(
                    rule='agent_execution_error',
                    message=f'QA agent {agent_name} failed: {str(e)}',
                    severity=ValidationSeverity.CRITICAL
                ))
                passed = False

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        return ValidationResult(
            stage=stage,
            passed=passed,
            violations=all_violations,
            metrics={
                'execution_time_ms': int(execution_time),
                'agents_run': agents,
                'total_violations': len(all_violations),
                'critical_violations': len([v for v in all_violations if v.severity == ValidationSeverity.CRITICAL]),
                'error_violations': len([v for v in all_violations if v.severity == ValidationSeverity.ERROR]),
                **all_metrics
            },
            execution_time_ms=int(execution_time)
        )

    def get_quality_gates(self) -> Dict[str, List[QualityGate]]:
        """Get all quality gates from all agents"""
        gates = {}
        for agent_name, agent in self.agents.items():
            gates[agent_name] = agent.get_quality_gates()
        return gates

    def export_validation_report(self, result: ValidationResult, output_path: str = None) -> str:
        """Export validation result to JSON report"""
        report = {
            'stage': result.stage.value,
            'passed': result.passed,
            'timestamp': result.timestamp.isoformat(),
            'execution_time_ms': result.execution_time_ms,
            'metrics': result.metrics,
            'violations': [
                {
                    'rule': v.rule,
                    'message': v.message,
                    'severity': v.severity.value,
                    'file_path': v.file_path,
                    'line_number': v.line_number,
                    'code_snippet': v.code_snippet,
                    'suggestion': v.suggestion
                }
                for v in result.violations
            ],
            'summary': {
                'total_violations': len(result.violations),
                'critical': len([v for v in result.violations if v.severity == ValidationSeverity.CRITICAL]),
                'errors': len([v for v in result.violations if v.severity == ValidationSeverity.ERROR]),
                'warnings': len([v for v in result.violations if v.severity == ValidationSeverity.WARNING]),
                'info': len([v for v in result.violations if v.severity == ValidationSeverity.INFO])
            }
        }

        json_report = json.dumps(report, indent=2)

        if output_path:
            Path(output_path).write_text(json_report, encoding='utf-8')
            self.logger.info(f"Validation report exported to {output_path}")

        return json_report


# Factory function for easy instantiation
def create_qa_orchestrator(config: Dict[str, Any] = None) -> QAOrchestrator:
    """Create a QA orchestrator with default configuration"""
    default_config = {
        'zt': {
            'console_log_forbidden': True,
            'any_types_forbidden': True,
            'undefined_errors_forbidden': True
        },
        'dgts': {
            'gaming_threshold': 0.3,
            'block_on_gaming': True
        },
        'nlnh': {
            'require_documentation': True,
            'test_coverage_threshold': 0.95
        }
    }

    if config:
        for section, values in config.items():
            if section in default_config:
                default_config[section].update(values)
            else:
                default_config[section] = values

    return QAOrchestrator(default_config)