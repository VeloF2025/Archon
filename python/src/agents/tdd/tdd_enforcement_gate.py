#!/usr/bin/env python3
"""
TDD Enforcement Gate - Mandatory Test-First Validation

This gate ensures NO feature development can proceed without tests first.
Blocks all code changes, commits, and deployments until TDD compliance is verified.

CRITICAL: This is a BLOCKING gate that cannot be bypassed or disabled.
All feature development must follow strict TDD principles.
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

from .stagehand_test_engine import StagehandTestEngine, TestGenerationResult
from .browserbase_executor import BrowserbaseExecutor, ExecutionResult
from .enhanced_dgts_validator import EnhancedDGTSValidator
from ..validation.dgts_validator import validate_dgts_compliance

logger = logging.getLogger(__name__)

class EnforcementLevel(Enum):
    """TDD enforcement strictness levels"""
    STRICT = "strict"      # No bypasses allowed
    MODERATE = "moderate"  # Limited bypasses for urgent fixes
    DEVELOPMENT = "development"  # More flexible for dev environments

class ViolationType(Enum):
    """Types of TDD violations"""
    NO_TESTS = "no_tests"
    FAILING_TESTS = "failing_tests"
    TESTS_AFTER_CODE = "tests_after_code"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    GAMING_DETECTED = "gaming_detected"
    MOCK_ABUSE = "mock_abuse"
    BYPASS_ATTEMPT = "bypass_attempt"

class FeatureStatus(Enum):
    """Feature development status"""
    NOT_STARTED = "not_started"
    TESTS_REQUIRED = "tests_required"
    TESTS_CREATED = "tests_created"
    TESTS_FAILING = "tests_failing"
    TESTS_PASSING = "tests_passing"
    IMPLEMENTATION_ALLOWED = "implementation_allowed"
    IMPLEMENTATION_BLOCKED = "implementation_blocked"
    READY_FOR_REVIEW = "ready_for_review"
    APPROVED = "approved"

@dataclass
class TDDViolation:
    """TDD violation details"""
    violation_type: ViolationType
    feature_name: str
    file_path: str
    line_number: int
    severity: str  # critical, error, warning
    description: str
    remediation: str
    detected_at: datetime
    blocking: bool

@dataclass
class FeatureValidation:
    """Feature TDD validation result"""
    feature_name: str
    status: FeatureStatus
    tests_exist: bool
    tests_passing: bool
    coverage_percentage: float
    violations: List[TDDViolation]
    implementation_files: List[str]
    test_files: List[str]
    last_validated: datetime
    validation_duration_ms: int

@dataclass
class EnforcementResult:
    """TDD enforcement gate result"""
    allowed: bool
    message: str
    feature_validations: List[FeatureValidation]
    total_violations: int
    critical_violations: int
    blocked_features: List[str]
    gaming_score: float
    enforcement_level: EnforcementLevel
    bypass_tokens: List[str]
    errors: List[str]
    warnings: List[str]
    enforcement_time_ms: int

class TDDEnforcementGate:
    """
    Mandatory TDD enforcement gate
    
    This gate implements zero-tolerance TDD enforcement, ensuring that:
    1. Tests are created BEFORE any implementation
    2. All tests pass before implementation is approved
    3. Coverage requirements are met
    4. No gaming or bypassing is allowed
    5. All changes follow TDD principles
    """

    def __init__(
        self,
        project_path: str = ".",
        enforcement_level: EnforcementLevel = EnforcementLevel.STRICT,
        min_coverage_percentage: float = 95.0,
        enable_gaming_detection: bool = True,
        **kwargs
    ):
        self.project_path = Path(project_path)
        self.enforcement_level = enforcement_level
        self.min_coverage_percentage = min_coverage_percentage
        self.enable_gaming_detection = enable_gaming_detection
        
        # Initialize components
        self.test_engine = StagehandTestEngine(project_path=str(self.project_path))
        self.executor = BrowserbaseExecutor(**kwargs)
        
        if enable_gaming_detection:
            self.dgts_validator = EnhancedDGTSValidator(str(self.project_path))
        else:
            self.dgts_validator = None
        
        # Load enforcement configuration
        self.config = self._load_enforcement_config()
        
        # Violation tracking
        self.violation_history: List[TDDViolation] = []
        self.feature_status: Dict[str, FeatureStatus] = {}
        self.bypass_tokens: Set[str] = set()
        
        # Performance tracking
        self.enforcement_stats = {
            "total_validations": 0,
            "blocked_features": 0,
            "allowed_features": 0,
            "gaming_attempts": 0,
            "bypass_attempts": 0
        }

    def _load_enforcement_config(self) -> Dict[str, Any]:
        """Load TDD enforcement configuration"""
        config_path = self.project_path / "tdd_config.yaml"
        
        default_config = {
            "enforcement": {
                "level": self.enforcement_level.value,
                "min_coverage": self.min_coverage_percentage,
                "require_tests_first": True,
                "block_on_failure": True,
                "enable_bypass": self.enforcement_level != EnforcementLevel.STRICT
            },
            "test_requirements": {
                "min_tests_per_feature": 5,
                "require_unit_tests": True,
                "require_integration_tests": True,
                "require_e2e_tests": True,
                "require_accessibility_tests": True
            },
            "coverage_requirements": {
                "statement_coverage": 95.0,
                "branch_coverage": 90.0,
                "function_coverage": 100.0,
                "critical_path_coverage": 100.0
            },
            "gaming_detection": {
                "enabled": self.enable_gaming_detection,
                "max_gaming_score": 0.3,
                "block_on_gaming": True,
                "monitor_patterns": True
            },
            "validation_rules": {
                "tests_before_implementation": True,
                "no_commented_tests": True,
                "no_fake_assertions": True,
                "no_mock_abuse": True,
                "require_real_data": True
            }
        }
        
        if config_path.exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    default_config.update(file_config)
            except Exception as e:
                logger.warning(f"Could not load TDD config: {e}")
        
        return default_config

    async def validate_feature_development(
        self,
        feature_name: str,
        implementation_files: List[str] = None,
        test_files: List[str] = None,
        force_validation: bool = False
    ) -> EnforcementResult:
        """
        Validate feature development for TDD compliance
        
        Args:
            feature_name: Name of the feature being developed
            implementation_files: List of implementation files (optional)
            test_files: List of test files (optional) 
            force_validation: Force validation even if recently validated
            
        Returns:
            EnforcementResult with validation details and enforcement decision
        """
        start_time = datetime.now()
        
        try:
            # Auto-discover files if not provided
            if implementation_files is None:
                implementation_files = self._discover_implementation_files(feature_name)
            if test_files is None:
                test_files = self._discover_test_files(feature_name)
            
            logger.info(f"Validating TDD compliance for feature: {feature_name}")
            
            # Run comprehensive TDD validation
            feature_validation = await self._validate_single_feature(
                feature_name, implementation_files, test_files
            )
            
            # Check for gaming patterns if enabled
            gaming_score = 0.0
            if self.dgts_validator:
                dgts_result = await self._run_gaming_detection(feature_name)
                gaming_score = dgts_result.get("gaming_score", 0.0)
                
                # Add gaming violations
                if dgts_result.get("is_gaming", False):
                    gaming_violation = TDDViolation(
                        violation_type=ViolationType.GAMING_DETECTED,
                        feature_name=feature_name,
                        file_path="",
                        line_number=0,
                        severity="critical",
                        description="Gaming patterns detected in test implementation",
                        remediation="Remove gaming patterns and implement genuine tests",
                        detected_at=datetime.now(),
                        blocking=True
                    )
                    feature_validation.violations.append(gaming_violation)
            
            # Make enforcement decision
            enforcement_decision = self._make_enforcement_decision([feature_validation], gaming_score)
            
            # Update feature status tracking
            self.feature_status[feature_name] = feature_validation.status
            
            # Update statistics
            self.enforcement_stats["total_validations"] += 1
            if enforcement_decision["allowed"]:
                self.enforcement_stats["allowed_features"] += 1
            else:
                self.enforcement_stats["blocked_features"] += 1
            
            # Record violations
            for violation in feature_validation.violations:
                self.violation_history.append(violation)
            
            end_time = datetime.now()
            enforcement_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return EnforcementResult(
                allowed=enforcement_decision["allowed"],
                message=enforcement_decision["message"],
                feature_validations=[feature_validation],
                total_violations=len(feature_validation.violations),
                critical_violations=len([v for v in feature_validation.violations if v.severity == "critical"]),
                blocked_features=enforcement_decision["blocked_features"],
                gaming_score=gaming_score,
                enforcement_level=self.enforcement_level,
                bypass_tokens=list(self.bypass_tokens),
                errors=enforcement_decision.get("errors", []),
                warnings=enforcement_decision.get("warnings", []),
                enforcement_time_ms=enforcement_time_ms
            )
            
        except Exception as e:
            error_msg = f"TDD enforcement validation failed for {feature_name}: {str(e)}"
            logger.error(error_msg)
            
            return EnforcementResult(
                allowed=False,
                message=error_msg,
                feature_validations=[],
                total_violations=1,
                critical_violations=1,
                blocked_features=[feature_name],
                gaming_score=1.0,  # Assume gaming on error
                enforcement_level=self.enforcement_level,
                bypass_tokens=[],
                errors=[error_msg],
                warnings=[],
                enforcement_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )

    async def _validate_single_feature(
        self,
        feature_name: str,
        implementation_files: List[str],
        test_files: List[str]
    ) -> FeatureValidation:
        """Validate a single feature for TDD compliance"""
        
        start_time = datetime.now()
        violations = []
        
        # Check if tests exist
        tests_exist = len(test_files) > 0 and all(Path(f).exists() for f in test_files)
        
        if not tests_exist:
            violations.append(TDDViolation(
                violation_type=ViolationType.NO_TESTS,
                feature_name=feature_name,
                file_path="",
                line_number=0,
                severity="critical",
                description=f"No tests found for feature {feature_name}",
                remediation="Create tests before implementing the feature",
                detected_at=datetime.now(),
                blocking=True
            ))
        
        # Check test timing (tests should exist before implementation)
        if tests_exist and implementation_files:
            violation = await self._check_test_first_principle(
                feature_name, implementation_files, test_files
            )
            if violation:
                violations.append(violation)
        
        # Run tests and check if they pass
        tests_passing = False
        coverage_percentage = 0.0
        
        if tests_exist:
            try:
                execution_result = await self.executor.execute_test_suite(test_files)
                tests_passing = execution_result.success
                
                if execution_result.coverage_report:
                    coverage_percentage = execution_result.coverage_report.get("statement_coverage", 0.0)
                
                # Check for failing tests
                if not tests_passing:
                    violations.append(TDDViolation(
                        violation_type=ViolationType.FAILING_TESTS,
                        feature_name=feature_name,
                        file_path="",
                        line_number=0,
                        severity="critical",
                        description=f"Tests are failing for feature {feature_name}",
                        remediation="Fix failing tests before proceeding",
                        detected_at=datetime.now(),
                        blocking=True
                    ))
                
                # Check coverage requirements
                if coverage_percentage < self.min_coverage_percentage:
                    violations.append(TDDViolation(
                        violation_type=ViolationType.INSUFFICIENT_COVERAGE,
                        feature_name=feature_name,
                        file_path="",
                        line_number=0,
                        severity="error",
                        description=f"Coverage {coverage_percentage:.1f}% below required {self.min_coverage_percentage}%",
                        remediation="Add more tests to meet coverage requirements",
                        detected_at=datetime.now(),
                        blocking=True
                    ))
                    
            except Exception as e:
                logger.error(f"Test execution failed for {feature_name}: {str(e)}")
                violations.append(TDDViolation(
                    violation_type=ViolationType.FAILING_TESTS,
                    feature_name=feature_name,
                    file_path="",
                    line_number=0,
                    severity="critical",
                    description=f"Test execution failed: {str(e)}",
                    remediation="Fix test execution issues",
                    detected_at=datetime.now(),
                    blocking=True
                ))
        
        # Determine feature status
        status = self._determine_feature_status(
            tests_exist, tests_passing, coverage_percentage, violations
        )
        
        end_time = datetime.now()
        validation_duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        return FeatureValidation(
            feature_name=feature_name,
            status=status,
            tests_exist=tests_exist,
            tests_passing=tests_passing,
            coverage_percentage=coverage_percentage,
            violations=violations,
            implementation_files=implementation_files,
            test_files=test_files,
            last_validated=end_time,
            validation_duration_ms=validation_duration_ms
        )

    async def _check_test_first_principle(
        self,
        feature_name: str,
        implementation_files: List[str],
        test_files: List[str]
    ) -> Optional[TDDViolation]:
        """Check if tests were created before implementation"""
        
        try:
            # Get file creation/modification times
            impl_times = []
            for impl_file in implementation_files:
                if Path(impl_file).exists():
                    impl_times.append(Path(impl_file).stat().st_mtime)
            
            test_times = []
            for test_file in test_files:
                if Path(test_file).exists():
                    test_times.append(Path(test_file).stat().st_mtime)
            
            if impl_times and test_times:
                earliest_impl = min(impl_times)
                earliest_test = min(test_times)
                
                # Allow 5 minute buffer for file system timing
                buffer_seconds = 300
                
                if earliest_impl < (earliest_test - buffer_seconds):
                    return TDDViolation(
                        violation_type=ViolationType.TESTS_AFTER_CODE,
                        feature_name=feature_name,
                        file_path="",
                        line_number=0,
                        severity="critical",
                        description="Implementation files created before tests (violates TDD)",
                        remediation="Delete implementation and create tests first",
                        detected_at=datetime.now(),
                        blocking=True
                    )
            
        except Exception as e:
            logger.warning(f"Could not check test-first principle for {feature_name}: {str(e)}")
        
        return None

    async def _run_gaming_detection(self, feature_name: str) -> Dict[str, Any]:
        """Run enhanced gaming detection"""
        
        if not self.dgts_validator:
            return {"is_gaming": False, "gaming_score": 0.0}
        
        try:
            # Run enhanced DGTS validation
            dgts_result = await asyncio.to_thread(
                self.dgts_validator.validate_stagehand_compliance
            )
            
            # Update gaming attempt statistics
            if dgts_result.get("is_gaming", False):
                self.enforcement_stats["gaming_attempts"] += 1
            
            return dgts_result
            
        except Exception as e:
            logger.error(f"Gaming detection failed for {feature_name}: {str(e)}")
            return {"is_gaming": False, "gaming_score": 0.0, "error": str(e)}

    def _determine_feature_status(
        self,
        tests_exist: bool,
        tests_passing: bool,
        coverage_percentage: float,
        violations: List[TDDViolation]
    ) -> FeatureStatus:
        """Determine current feature status based on validation results"""
        
        # Check for blocking violations
        blocking_violations = [v for v in violations if v.blocking]
        
        if not tests_exist:
            return FeatureStatus.TESTS_REQUIRED
        
        if tests_exist and not tests_passing:
            return FeatureStatus.TESTS_FAILING
        
        if tests_exist and tests_passing and coverage_percentage < self.min_coverage_percentage:
            return FeatureStatus.IMPLEMENTATION_BLOCKED
        
        if blocking_violations:
            return FeatureStatus.IMPLEMENTATION_BLOCKED
        
        if tests_exist and tests_passing and coverage_percentage >= self.min_coverage_percentage:
            return FeatureStatus.IMPLEMENTATION_ALLOWED
        
        return FeatureStatus.NOT_STARTED

    def _make_enforcement_decision(
        self,
        feature_validations: List[FeatureValidation],
        gaming_score: float
    ) -> Dict[str, Any]:
        """Make final enforcement decision"""
        
        blocked_features = []
        critical_violations = []
        errors = []
        warnings = []
        
        # Check each feature
        for validation in feature_validations:
            # Block features with critical violations
            critical_viols = [v for v in validation.violations if v.severity == "critical"]
            if critical_viols:
                blocked_features.append(validation.feature_name)
                critical_violations.extend(critical_viols)
            
            # Block features that aren't in allowed status
            if validation.status not in [FeatureStatus.IMPLEMENTATION_ALLOWED, FeatureStatus.APPROVED]:
                blocked_features.append(validation.feature_name)
        
        # Check gaming score
        max_gaming_score = self.config["gaming_detection"]["max_gaming_score"]
        if gaming_score > max_gaming_score:
            blocked_features = [v.feature_name for v in feature_validations]
            errors.append(f"Gaming score {gaming_score:.2f} exceeds limit {max_gaming_score}")
        
        # Check for bypass attempts (strict enforcement only)
        if self.enforcement_level == EnforcementLevel.STRICT:
            # No bypasses allowed in strict mode
            bypass_attempts = self._detect_bypass_attempts()
            if bypass_attempts:
                blocked_features = [v.feature_name for v in feature_validations]
                errors.append(f"Bypass attempts detected: {', '.join(bypass_attempts)}")
                self.enforcement_stats["bypass_attempts"] += len(bypass_attempts)
        
        # Make final decision
        allowed = len(blocked_features) == 0 and len(critical_violations) == 0
        
        if allowed:
            message = "TDD compliance validated - implementation allowed"
        else:
            reasons = []
            if blocked_features:
                reasons.append(f"Blocked features: {', '.join(blocked_features)}")
            if critical_violations:
                reasons.append(f"Critical violations: {len(critical_violations)}")
            if gaming_score > max_gaming_score:
                reasons.append(f"Gaming detected: {gaming_score:.2f}")
            
            message = f"TDD compliance failed - {'; '.join(reasons)}"
        
        return {
            "allowed": allowed,
            "message": message,
            "blocked_features": blocked_features,
            "critical_violations": critical_violations,
            "errors": errors,
            "warnings": warnings
        }

    def _detect_bypass_attempts(self) -> List[str]:
        """Detect attempts to bypass TDD enforcement"""
        
        bypass_patterns = [
            "skip_tdd",
            "bypass_tests", 
            "no_tests",
            "disable_validation",
            "emergency_deploy",
            "hotfix_bypass"
        ]
        
        detected_attempts = []
        
        # Check for bypass patterns in code comments
        code_files = list(self.project_path.glob("**/*.py")) + \
                    list(self.project_path.glob("**/*.js")) + \
                    list(self.project_path.glob("**/*.ts"))
        
        for code_file in code_files:
            try:
                content = code_file.read_text(encoding='utf-8')
                for pattern in bypass_patterns:
                    if pattern.lower() in content.lower():
                        detected_attempts.append(f"{pattern} in {code_file}")
            except Exception:
                continue
        
        return detected_attempts

    def _discover_implementation_files(self, feature_name: str) -> List[str]:
        """Auto-discover implementation files for a feature"""
        
        patterns = [
            f"**/{feature_name.lower()}*.py",
            f"**/{feature_name.lower()}*.js",
            f"**/{feature_name.lower()}*.ts",
            f"**/{feature_name.lower()}*.tsx",
            f"**/src/**/{feature_name.lower()}*",
            f"**/lib/**/{feature_name.lower()}*"
        ]
        
        implementation_files = []
        
        for pattern in patterns:
            for file_path in self.project_path.glob(pattern):
                # Exclude test files
                if not any(test_indicator in str(file_path).lower() 
                          for test_indicator in ["test", "spec"]):
                    implementation_files.append(str(file_path))
        
        return implementation_files

    def _discover_test_files(self, feature_name: str) -> List[str]:
        """Auto-discover test files for a feature"""
        
        patterns = [
            f"**/test*{feature_name.lower()}*.py",
            f"**/test*{feature_name.lower()}*.js",
            f"**/test*{feature_name.lower()}*.ts",
            f"**/{feature_name.lower()}*.test.*",
            f"**/{feature_name.lower()}*.spec.*",
            f"**/tests/**/{feature_name.lower()}*"
        ]
        
        test_files = []
        
        for pattern in patterns:
            for file_path in self.project_path.glob(pattern):
                test_files.append(str(file_path))
        
        return test_files

    async def generate_tests_for_feature(
        self,
        feature_name: str,
        requirements: List[str],
        acceptance_criteria: List[str],
        **kwargs
    ) -> TestGenerationResult:
        """Generate tests for a feature using the test engine"""
        
        return await self.test_engine.generate_tests_from_requirements(
            requirements=requirements,
            feature_name=feature_name,
            acceptance_criteria=acceptance_criteria,
            **kwargs
        )

    async def create_enforcement_report(self) -> Dict[str, Any]:
        """Create comprehensive enforcement report"""
        
        return {
            "enforcement_summary": {
                "level": self.enforcement_level.value,
                "total_validations": self.enforcement_stats["total_validations"],
                "allowed_features": self.enforcement_stats["allowed_features"],
                "blocked_features": self.enforcement_stats["blocked_features"],
                "success_rate": (
                    self.enforcement_stats["allowed_features"] / 
                    max(self.enforcement_stats["total_validations"], 1)
                ) * 100
            },
            "gaming_detection": {
                "attempts_detected": self.enforcement_stats["gaming_attempts"],
                "bypass_attempts": self.enforcement_stats["bypass_attempts"]
            },
            "feature_status": {
                feature: status.value 
                for feature, status in self.feature_status.items()
            },
            "recent_violations": [
                {
                    "type": v.violation_type.value,
                    "feature": v.feature_name,
                    "severity": v.severity,
                    "description": v.description,
                    "detected_at": v.detected_at.isoformat()
                }
                for v in self.violation_history[-10:]  # Last 10 violations
            ],
            "configuration": {
                "min_coverage": self.min_coverage_percentage,
                "gaming_detection_enabled": self.enable_gaming_detection,
                "enforcement_rules": self.config
            },
            "generated_at": datetime.now().isoformat()
        }

    def get_enforcement_stats(self) -> Dict[str, Any]:
        """Get current enforcement statistics"""
        return {
            **self.enforcement_stats,
            "violation_count": len(self.violation_history),
            "active_features": len(self.feature_status),
            "bypass_tokens_used": len(self.bypass_tokens)
        }

# Standalone functions for easy integration
async def enforce_tdd_compliance(
    feature_name: str,
    project_path: str = ".",
    implementation_files: List[str] = None,
    test_files: List[str] = None,
    **kwargs
) -> EnforcementResult:
    """
    Enforce TDD compliance for a feature
    
    Args:
        feature_name: Name of the feature being validated
        project_path: Path to the project root
        implementation_files: List of implementation files (optional)
        test_files: List of test files (optional)
        **kwargs: Additional configuration options
        
    Returns:
        EnforcementResult with validation details and enforcement decision
    """
    
    gate = TDDEnforcementGate(project_path=project_path, **kwargs)
    
    return await gate.validate_feature_development(
        feature_name=feature_name,
        implementation_files=implementation_files,
        test_files=test_files
    )

def is_feature_implementation_allowed(
    feature_name: str,
    project_path: str = ".",
    **kwargs
) -> bool:
    """
    Quick check if feature implementation is allowed
    
    Args:
        feature_name: Name of the feature
        project_path: Path to the project root
        **kwargs: Additional configuration options
        
    Returns:
        True if implementation is allowed, False otherwise
    """
    
    gate = TDDEnforcementGate(project_path=project_path, **kwargs)
    
    # Run synchronous validation (simplified)
    implementation_files = gate._discover_implementation_files(feature_name)
    test_files = gate._discover_test_files(feature_name)
    
    # Basic checks
    tests_exist = len(test_files) > 0 and all(Path(f).exists() for f in test_files)
    
    if not tests_exist:
        return False
    
    # Check for gaming patterns
    if gate.dgts_validator:
        dgts_result = validate_dgts_compliance(project_path)
        if dgts_result.get("dgts_result", {}).get("is_gaming", False):
            return False
    
    return True