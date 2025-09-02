#!/usr/bin/env python3
"""
ARCHON GLOBAL VALIDATION RULES
Immutable validation rules that apply to all projects and agents

These rules cannot be overridden or bypassed by any agent or system.
Violation of these rules is a critical system failure.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any

class ValidationRule(Enum):
    """Immutable validation rules for all Archon projects"""
    
    # CORE RULES - Cannot be disabled or bypassed
    TESTS_MANDATORY = "All code must have comprehensive tests - NO EXCEPTIONS"
    TESTS_MUST_PASS = "All tests must pass before code is valid - NO EXCEPTIONS"  
    NO_SYNTAX_ERRORS = "Zero syntax/compilation errors allowed - NO EXCEPTIONS"
    NO_SECURITY_VULNS = "Zero security vulnerabilities allowed - NO EXCEPTIONS"
    
    # DEVELOPMENT RULES
    TDD_REQUIRED = "Tests must be written BEFORE development begins"
    DOC_DRIVEN_TDD = "Tests must be created from PRD/PRP/ADR docs BEFORE any code implementation"
    PRD_PRP_COMPLIANCE = "All PRD/PRP features must have corresponding tests"
    NO_SCOPE_CREEP = "Features not in PRD/PRP require updated documentation and tests"
    TEST_SPEC_VALIDATION = "Test specs must validate against planned requirements from documentation"
    
    # QUALITY RULES
    MIN_COVERAGE = "Minimum 95% test coverage required"
    NO_DEAD_CODE = "No unreachable or unused code allowed"
    ERROR_HANDLING = "All error conditions must be handled and tested"
    
    # REPORTING RULES
    HONEST_REPORTING = "Never game or manipulate validation metrics"
    TRANSPARENT_FAILURES = "All validation failures must be reported immediately"
    NO_FALSE_POSITIVES = "Validation must be accurate, not permissive"
    
    # ANTI-GAMING RULES (DGTS - DON'T GAME THE SYSTEM)
    NO_TEST_GAMING = "Tests must validate real functionality, not mocked/stubbed implementations"
    NO_CODE_GAMING = "Code must not be commented out or disabled to pass validation"
    NO_FEATURE_FAKING = "Features marked as complete must actually work, not return fake data"
    NO_VALIDATION_BYPASS = "Validation rules cannot be disabled, commented out, or circumvented"
    REAL_IMPLEMENTATION_REQUIRED = "All completed features must have genuine working implementations"

@dataclass
class ValidationEnforcement:
    """Enforcement configuration for validation rules"""
    rule: ValidationRule
    severity: str  # critical, error, warning, info
    blocking: bool  # Does this block development?
    bypassable: bool  # Can this rule ever be bypassed?
    remediation: str  # What to do when rule is violated

# IMMUTABLE RULE ENFORCEMENT
ARCHON_VALIDATION_ENFORCEMENT = [
    ValidationEnforcement(
        rule=ValidationRule.TESTS_MANDATORY,
        severity="critical",
        blocking=True,
        bypassable=False,
        remediation="Write comprehensive tests for all code before proceeding"
    ),
    ValidationEnforcement(
        rule=ValidationRule.TESTS_MUST_PASS,
        severity="critical", 
        blocking=True,
        bypassable=False,
        remediation="Fix all failing tests before proceeding"
    ),
    ValidationEnforcement(
        rule=ValidationRule.NO_SYNTAX_ERRORS,
        severity="critical",
        blocking=True, 
        bypassable=False,
        remediation="Fix all syntax and compilation errors"
    ),
    ValidationEnforcement(
        rule=ValidationRule.NO_SECURITY_VULNS,
        severity="critical",
        blocking=True,
        bypassable=False,
        remediation="Fix all security vulnerabilities immediately"
    ),
    ValidationEnforcement(
        rule=ValidationRule.TDD_REQUIRED,
        severity="error",
        blocking=True,
        bypassable=False,
        remediation="Write tests before implementing features"
    ),
    ValidationEnforcement(
        rule=ValidationRule.DOC_DRIVEN_TDD,
        severity="critical",
        blocking=True,
        bypassable=False,
        remediation="Create test specifications from PRD/PRP/ADR documents before writing any implementation code"
    ),
    ValidationEnforcement(
        rule=ValidationRule.PRD_PRP_COMPLIANCE,
        severity="error",
        blocking=True,
        bypassable=False,
        remediation="Ensure all PRD/PRP requirements have corresponding tests"
    ),
    ValidationEnforcement(
        rule=ValidationRule.TEST_SPEC_VALIDATION,
        severity="error",
        blocking=True,
        bypassable=False,
        remediation="Validate that test specifications match documented requirements from PRDs/PRPs/ADRs"
    ),
    ValidationEnforcement(
        rule=ValidationRule.MIN_COVERAGE,
        severity="error",
        blocking=True,
        bypassable=False,
        remediation="Increase test coverage to minimum 95%"
    ),
    ValidationEnforcement(
        rule=ValidationRule.HONEST_REPORTING,
        severity="critical",
        blocking=True,
        bypassable=False,
        remediation="Report accurate validation results, do not manipulate metrics"
    ),
    # DGTS ENFORCEMENT - DON'T GAME THE SYSTEM
    ValidationEnforcement(
        rule=ValidationRule.NO_TEST_GAMING,
        severity="critical",
        blocking=True,
        bypassable=False,
        remediation="Remove mocked/stubbed test implementations and create tests for real functionality"
    ),
    ValidationEnforcement(
        rule=ValidationRule.NO_CODE_GAMING,
        severity="critical", 
        blocking=True,
        bypassable=False,
        remediation="Uncomment and properly implement all disabled code, do not bypass validation"
    ),
    ValidationEnforcement(
        rule=ValidationRule.NO_FEATURE_FAKING,
        severity="critical",
        blocking=True,
        bypassable=False,
        remediation="Implement genuine working feature instead of returning fake/mock data"
    ),
    ValidationEnforcement(
        rule=ValidationRule.NO_VALIDATION_BYPASS,
        severity="critical",
        blocking=True,
        bypassable=False,
        remediation="Restore disabled validation rules and fix underlying issues properly"
    ),
    ValidationEnforcement(
        rule=ValidationRule.REAL_IMPLEMENTATION_REQUIRED,
        severity="critical",
        blocking=True,
        bypassable=False,
        remediation="Replace stub/mock implementations with real working code"
    )
]

def get_validation_rules() -> List[ValidationEnforcement]:
    """Get immutable validation rules for all Archon projects"""
    return ARCHON_VALIDATION_ENFORCEMENT.copy()

def is_rule_bypassable(rule: ValidationRule) -> bool:
    """Check if a validation rule can ever be bypassed (answer: NO for critical rules)"""
    for enforcement in ARCHON_VALIDATION_ENFORCEMENT:
        if enforcement.rule == rule:
            return enforcement.bypassable
    return False

def get_rule_remediation(rule: ValidationRule) -> str:
    """Get remediation steps for a validation rule violation"""
    for enforcement in ARCHON_VALIDATION_ENFORCEMENT:
        if enforcement.rule == rule:
            return enforcement.remediation
    return "Fix the validation issue and retry"

def validate_rule_compliance(validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that results comply with Archon validation rules"""
    violations = []
    
    # Check for documentation-driven test development
    if validation_results.get("has_implementation", False) and not validation_results.get("tests_from_docs", False):
        violations.append({
            "rule": ValidationRule.DOC_DRIVEN_TDD.value,
            "severity": "critical",
            "message": "Code implementation found without tests derived from documentation (PRD/PRP/ADR)",
            "remediation": get_rule_remediation(ValidationRule.DOC_DRIVEN_TDD)
        })
    
    # Check for test existence
    if not validation_results.get("tests_exist", False):
        violations.append({
            "rule": ValidationRule.TESTS_MANDATORY.value,
            "severity": "critical",
            "message": "No tests found - This is a CRITICAL validation failure",
            "remediation": get_rule_remediation(ValidationRule.TESTS_MANDATORY)
        })
    
    # Check for test-spec validation against docs
    if validation_results.get("tests_exist", False) and not validation_results.get("tests_match_specs", True):
        violations.append({
            "rule": ValidationRule.TEST_SPEC_VALIDATION.value,
            "severity": "error",
            "message": "Tests do not match documented requirements from PRD/PRP/ADR",
            "remediation": get_rule_remediation(ValidationRule.TEST_SPEC_VALIDATION)
        })
    
    # Check for test passing
    if validation_results.get("tests_exist", False) and not validation_results.get("tests_pass", False):
        violations.append({
            "rule": ValidationRule.TESTS_MUST_PASS.value,
            "severity": "critical", 
            "message": "Tests are failing - This is a CRITICAL validation failure",
            "remediation": get_rule_remediation(ValidationRule.TESTS_MUST_PASS)
        })
    
    # Check for syntax errors
    if validation_results.get("syntax_errors", 0) > 0:
        violations.append({
            "rule": ValidationRule.NO_SYNTAX_ERRORS.value,
            "severity": "critical",
            "message": f"Found {validation_results.get('syntax_errors', 0)} syntax errors",
            "remediation": get_rule_remediation(ValidationRule.NO_SYNTAX_ERRORS)
        })
    
    # Check coverage
    coverage = validation_results.get("coverage", 0.0)
    if coverage < 0.95:
        violations.append({
            "rule": ValidationRule.MIN_COVERAGE.value,
            "severity": "error",
            "message": f"Coverage {coverage:.1%} below minimum 95%",
            "remediation": get_rule_remediation(ValidationRule.MIN_COVERAGE)
        })
    
    return {
        "compliant": len(violations) == 0,
        "violations": violations,
        "critical_violations": len([v for v in violations if v["severity"] == "critical"]),
        "blocking_violations": len([v for v in violations if v["severity"] in ["critical", "error"]])
    }

# SYSTEM INTEGRITY CHECK
def verify_system_integrity():
    """Verify that validation system hasn't been compromised"""
    
    # Check that critical rules are still non-bypassable
    critical_rules = [
        ValidationRule.TESTS_MANDATORY,
        ValidationRule.TESTS_MUST_PASS,
        ValidationRule.NO_SYNTAX_ERRORS,
        ValidationRule.NO_SECURITY_VULNS,
        ValidationRule.DOC_DRIVEN_TDD,
        ValidationRule.HONEST_REPORTING
    ]
    
    for rule in critical_rules:
        if is_rule_bypassable(rule):
            raise RuntimeError(f"CRITICAL: Rule {rule.value} has been compromised - system integrity failure")
    
    return True

# Auto-verify on module import
try:
    verify_system_integrity()
except RuntimeError as e:
    print(f"VALIDATION SYSTEM COMPROMISED: {e}")
    raise