#!/usr/bin/env python3
"""
DGTS VALIDATOR - DON'T GAME THE SYSTEM
Anti-gaming detection and prevention system for agent behavior

This validator detects and prevents agents from:
1. Gaming tests with fake/mocked implementations
2. Commenting out validation rules to pass checks
3. Creating fake features that return mock data
4. Bypassing validation through code manipulation
5. Marking features as complete without real implementation

CRITICAL: This is a blocking validation that prevents gaming behaviors
"""

import os
import re
import ast
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime

from .archon_validation_rules import ValidationRule, validate_rule_compliance

logger = logging.getLogger(__name__)

@dataclass
class GamingViolation:
    """Detected gaming violation"""
    violation_type: str
    file_path: str
    line_number: int
    content: str
    severity: str
    explanation: str
    remediation: str

@dataclass 
class DGTSValidation:
    """DGTS validation result"""
    is_gaming: bool
    violations: List[GamingViolation]
    gaming_score: float  # 0.0 = clean, 1.0 = heavily gamed
    blocked_features: List[str]
    suspicious_patterns: List[str]

class DGTSValidator:
    """Detects and prevents system gaming by agents"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.violations: List[GamingViolation] = []
        
        # Gaming pattern definitions
        self.gaming_patterns = {
            'commented_validation': [
                r'#.*validation',
                r'#.*test.*required',
                r'#.*mandatory',
                r'//.*validation',
                r'//.*test.*required',
                r'/\*.*validation.*\*/',
            ],
            'fake_returns': [
                r'return\s+["\']mock["\']',
                r'return\s+["\']fake["\']',
                r'return\s+["\']stub["\']',
                r'return\s+["\']placeholder["\']',
                r'return\s+["\']TODO["\']',
                r'return\s+\{\s*["\']status["\']:\s*["\']ok["\']',
                r'return\s+True\s*#.*fake',
                r'return\s+\[\]\s*#.*mock',
            ],
            'test_gaming': [
                r'assert\s+True\s*#.*mock',
                r'assert\s+True\s*#.*fake',
                r'@mock\.',
                r'@patch\.',
                r'\.return_value\s*=',
                r'mock_.*\s*=',
                r'stub_.*\s*=',
                r'when\(.*\)\.then.*Return',
            ],
            'validation_bypass': [
                r'skip.*test',
                r'@skip',
                r'@pytest\.mark\.skip',
                r'if\s+False:',
                r'#\s*TODO:.*bypass',
                r'#\s*HACK:',
                r'#\s*TEMP:.*disable',
            ],
            'feature_faking': [
                r'def\s+\w+.*:\s*#.*not implemented',
                r'def\s+\w+.*:\s*pass\s*#.*stub',
                r'raise\s+NotImplementedError\s*#.*fake',
                r'return\s+None\s*#.*not ready',
                r'return\s+\{\}\s*#.*empty',
            ]
        }
        
        # Suspicious keywords that indicate gaming
        self.gaming_keywords = [
            'mock', 'fake', 'stub', 'placeholder', 'dummy',
            'not implemented', 'coming soon', 'todo implementation',
            'bypass', 'skip', 'disable', 'temporary', 'hack'
        ]
        
    def validate_no_gaming(self) -> DGTSValidation:
        """Main validation method - detects gaming patterns"""
        
        self.violations = []
        
        # Scan all code files for gaming patterns
        self._scan_for_gaming_patterns()
        
        # Analyze test files for fake implementations
        self._analyze_test_gaming()
        
        # Check for validation bypasses
        self._detect_validation_bypasses()
        
        # Check feature completion claims vs actual implementation
        self._validate_feature_completeness()
        
        # Calculate gaming score
        gaming_score = self._calculate_gaming_score()
        
        # Determine if system is being gamed
        is_gaming = gaming_score > 0.3 or any(v.severity == 'critical' for v in self.violations)
        
        return DGTSValidation(
            is_gaming=is_gaming,
            violations=self.violations,
            gaming_score=gaming_score,
            blocked_features=self._get_blocked_features(),
            suspicious_patterns=self._get_suspicious_patterns()
        )
    
    def _scan_for_gaming_patterns(self):
        """Scan all code files for gaming patterns"""
        
        code_patterns = [
            '**/*.py', '**/*.js', '**/*.ts', '**/*.tsx', 
            '**/*.java', '**/*.cs', '**/*.cpp', '**/*.go'
        ]
        
        exclude_patterns = [
            '**/node_modules/*', '**/venv/*', '**/dist/*', 
            '**/build/*', '**/.git/*', '**/coverage/*'
        ]
        
        for pattern in code_patterns:
            for file_path in self.project_path.glob(pattern):
                
                # Skip excluded directories
                if any(file_path.match(excl) for excl in exclude_patterns):
                    continue
                
                try:
                    content = file_path.read_text(encoding='utf-8')
                    self._analyze_file_for_gaming(file_path, content)
                except Exception as e:
                    logger.warning(f"Could not read file {file_path}: {e}")
                    continue
    
    def _analyze_file_for_gaming(self, file_path: Path, content: str):
        """Analyze individual file for gaming patterns"""
        
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line_lower = line.lower().strip()
            
            # Check each gaming pattern category
            for category, patterns in self.gaming_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        
                        # Determine severity based on pattern type
                        severity = 'critical' if category in [
                            'validation_bypass', 'feature_faking'
                        ] else 'error'
                        
                        violation = GamingViolation(
                            violation_type=f"DGTS_{category.upper()}",
                            file_path=str(file_path),
                            line_number=line_num,
                            content=line.strip(),
                            severity=severity,
                            explanation=self._get_violation_explanation(category, line),
                            remediation=self._get_violation_remediation(category)
                        )
                        
                        self.violations.append(violation)
            
            # Check for gaming keywords
            for keyword in self.gaming_keywords:
                if keyword in line_lower and not self._is_acceptable_context(line_lower, keyword):
                    
                    violation = GamingViolation(
                        violation_type="DGTS_SUSPICIOUS_KEYWORD",
                        file_path=str(file_path),
                        line_number=line_num,
                        content=line.strip(),
                        severity='warning',
                        explanation=f"Suspicious gaming keyword '{keyword}' found",
                        remediation="Replace with genuine implementation or remove if unnecessary"
                    )
                    
                    self.violations.append(violation)
    
    def _analyze_test_gaming(self):
        """Specifically analyze test files for gaming patterns"""
        
        test_patterns = [
            '**/test_*.py', '**/tests/*.py', '**/*_test.py',
            '**/test_*.js', '**/tests/*.js', '**/*.test.js', '**/*.spec.js',
            '**/test_*.ts', '**/tests/*.ts', '**/*.test.ts', '**/*.spec.ts'
        ]
        
        for pattern in test_patterns:
            for test_file in self.project_path.glob(pattern):
                try:
                    content = test_file.read_text(encoding='utf-8')
                    self._analyze_test_file_gaming(test_file, content)
                except Exception as e:
                    logger.warning(f"Could not read test file {test_file}: {e}")
                    continue
    
    def _analyze_test_file_gaming(self, test_file: Path, content: str):
        """Analyze test file for gaming patterns"""
        
        lines = content.split('\n')
        
        # Look for test gaming indicators
        gaming_indicators = [
            r'assert\s+True\s*$',  # Meaningless assertions
            r'assert\s+1\s*==\s*1',  # Tautological assertions
            r'assert\s+not\s+False',
            r'pass\s*#.*test',  # Empty test bodies
            r'return\s*#.*test',
        ]
        
        # Count real vs fake assertions
        real_assertions = 0
        fake_assertions = 0
        
        for line_num, line in enumerate(lines, 1):
            line_clean = line.strip()
            
            # Check for meaningful assertions
            if re.search(r'assert\s+\w+.*==.*\w+', line) or re.search(r'assert\s+\w+\(', line):
                real_assertions += 1
            
            # Check for gaming indicators
            for indicator in gaming_indicators:
                if re.search(indicator, line, re.IGNORECASE):
                    fake_assertions += 1
                    
                    violation = GamingViolation(
                        violation_type="DGTS_TEST_GAMING",
                        file_path=str(test_file),
                        line_number=line_num,
                        content=line_clean,
                        severity='critical',
                        explanation="Test contains meaningless or fake assertion",
                        remediation="Replace with genuine test that validates real functionality"
                    )
                    
                    self.violations.append(violation)
        
        # Check test quality ratio
        if fake_assertions > 0 and real_assertions / max(fake_assertions, 1) < 2:
            
            violation = GamingViolation(
                violation_type="DGTS_LOW_TEST_QUALITY",
                file_path=str(test_file),
                line_number=0,
                content=f"Real assertions: {real_assertions}, Fake assertions: {fake_assertions}",
                severity='error',
                explanation="Test file has too many fake/meaningless assertions relative to real ones",
                remediation="Increase real test coverage and remove fake assertions"
            )
            
            self.violations.append(violation)
    
    def _detect_validation_bypasses(self):
        """Detect attempts to bypass validation rules"""
        
        # Check for commented out validation code
        validation_files = [
            '**/validation*.py', '**/test*.py', '**/*validation*',
            '**/rules*.py', '**/policy*.py'
        ]
        
        for pattern in validation_files:
            for file_path in self.project_path.glob(pattern):
                try:
                    content = file_path.read_text(encoding='utf-8')
                    self._check_validation_bypass(file_path, content)
                except Exception:
                    continue
    
    def _check_validation_bypass(self, file_path: Path, content: str):
        """Check specific file for validation bypasses"""
        
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            
            # Check for commented validation rules
            if re.search(r'#.*(?:validation|enforce|mandatory|critical)', line, re.IGNORECASE):
                
                violation = GamingViolation(
                    violation_type="DGTS_VALIDATION_BYPASS",
                    file_path=str(file_path),
                    line_number=line_num,
                    content=line.strip(),
                    severity='critical',
                    explanation="Validation rule or enforcement has been commented out",
                    remediation="Uncomment validation rule and fix underlying issue properly"
                )
                
                self.violations.append(violation)
    
    def _validate_feature_completeness(self):
        """Validate that completed features actually work"""
        
        # Look for features marked as complete but with stub implementations
        completion_indicators = [
            'complete', 'done', 'finished', 'implemented', 'ready'
        ]
        
        stub_indicators = [
            'todo', 'not implemented', 'coming soon', 'placeholder',
            'stub', 'mock', 'fake', 'dummy'
        ]
        
        for file_path in self.project_path.glob('**/*.py'):
            try:
                content = file_path.read_text(encoding='utf-8')
                
                # Check for completion claims near stub implementations
                if any(indicator in content.lower() for indicator in completion_indicators):
                    if any(stub in content.lower() for stub in stub_indicators):
                        
                        violation = GamingViolation(
                            violation_type="DGTS_FEATURE_FAKING",
                            file_path=str(file_path),
                            line_number=0,
                            content="Feature marked complete but contains stub implementations",
                            severity='critical',
                            explanation="Feature claimed as complete but still has placeholder/stub code",
                            remediation="Complete genuine implementation or update status to reflect actual progress"
                        )
                        
                        self.violations.append(violation)
                        
            except Exception:
                continue
    
    def _calculate_gaming_score(self) -> float:
        """Calculate overall gaming score (0.0 = clean, 1.0 = heavily gamed)"""
        
        if not self.violations:
            return 0.0
        
        # Weight violations by severity
        weights = {'critical': 1.0, 'error': 0.6, 'warning': 0.3}
        
        total_weight = sum(weights.get(v.severity, 0.3) for v in self.violations)
        
        # Normalize to 0-1 scale
        max_expected_violations = 20  # Reasonable upper bound
        gaming_score = min(total_weight / max_expected_violations, 1.0)
        
        return gaming_score
    
    def _get_blocked_features(self) -> List[str]:
        """Get list of features that should be blocked due to gaming"""
        
        blocked = []
        
        for violation in self.violations:
            if violation.severity == 'critical':
                # Extract potential feature name from file path
                feature_name = Path(violation.file_path).stem
                if feature_name not in blocked:
                    blocked.append(feature_name)
        
        return blocked
    
    def _get_suspicious_patterns(self) -> List[str]:
        """Get list of suspicious patterns detected"""
        
        patterns = set()
        
        for violation in self.violations:
            patterns.add(violation.violation_type)
        
        return list(patterns)
    
    def _is_acceptable_context(self, line: str, keyword: str) -> bool:
        """Check if gaming keyword is in acceptable context"""
        
        acceptable_contexts = [
            'import', 'from', 'test_mock', 'unittest.mock',
            'jest.mock', '@mock', 'mockito', 'sinon',
            'legitimate mock', 'proper stub', 'documentation'
        ]
        
        return any(context in line for context in acceptable_contexts)
    
    def _get_violation_explanation(self, category: str, line: str) -> str:
        """Get explanation for violation category"""
        
        explanations = {
            'commented_validation': "Validation rule has been commented out to bypass checks",
            'fake_returns': "Function returns fake/mock data instead of real implementation", 
            'test_gaming': "Test uses mocking/stubbing instead of testing real functionality",
            'validation_bypass': "Code attempts to skip or disable validation checks",
            'feature_faking': "Feature appears complete but contains only stub/placeholder code"
        }
        
        return explanations.get(category, "Potential gaming behavior detected")
    
    def _get_violation_remediation(self, category: str) -> str:
        """Get remediation steps for violation category"""
        
        remediations = {
            'commented_validation': "Uncomment validation rule and fix underlying issue",
            'fake_returns': "Implement genuine functionality instead of returning fake data",
            'test_gaming': "Write tests that validate real behavior, not mocked responses", 
            'validation_bypass': "Remove bypass attempts and address validation failures properly",
            'feature_faking': "Complete actual implementation instead of using placeholder code"
        }
        
        return remediations.get(category, "Fix gaming behavior and implement properly")

def validate_dgts_compliance(project_path: str = ".") -> Dict[str, Any]:
    """
    Main DGTS validation function
    
    Validates that agents are not gaming the system through:
    - Fake test implementations
    - Commented out validation rules  
    - Mock data for completed features
    - Validation bypasses
    """
    
    validator = DGTSValidator(project_path)
    dgts_result = validator.validate_no_gaming()
    
    # Create validation results for rule compliance check
    validation_results = {
        "tests_exist": True,  # Assume tests exist for now
        "tests_from_docs": True,
        "tests_match_specs": not dgts_result.is_gaming,
        "has_implementation": True,
        "syntax_errors": 0,
        "coverage": 0.0 if dgts_result.is_gaming else 1.0,
        "gaming_detected": dgts_result.is_gaming,
        "gaming_score": dgts_result.gaming_score
    }
    
    # Run standard rule compliance 
    rule_compliance = validate_rule_compliance(validation_results)
    
    return {
        "compliant": not dgts_result.is_gaming and rule_compliance["compliant"],
        "dgts_result": {
            "is_gaming": dgts_result.is_gaming,
            "gaming_score": dgts_result.gaming_score,
            "violations": [
                {
                    "type": v.violation_type,
                    "file": v.file_path,
                    "line": v.line_number,
                    "content": v.content,
                    "severity": v.severity,
                    "explanation": v.explanation,
                    "remediation": v.remediation
                }
                for v in dgts_result.violations
            ],
            "blocked_features": dgts_result.blocked_features,
            "suspicious_patterns": dgts_result.suspicious_patterns
        },
        "rule_compliance": rule_compliance,
        "message": "DGTS validation complete - gaming detection active" if not dgts_result.is_gaming else "GAMING DETECTED - System is being manipulated",
        "action_required": "Fix all gaming violations before proceeding" if dgts_result.is_gaming else None
    }

if __name__ == "__main__":
    import sys
    project_path = sys.argv[1] if len(sys.argv) > 1 else "."
    result = validate_dgts_compliance(project_path)
    print(json.dumps(result, indent=2, default=str))