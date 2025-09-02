#!/usr/bin/env python3
"""
Enhanced DGTS Validator - Extended Gaming Detection for Stagehand Integration

This enhanced validator extends the base DGTS system to detect Stagehand-specific
gaming patterns and natural language test manipulation attempts.

CRITICAL: Prevents sophisticated gaming attempts in AI-generated browser automation tests.
"""

import os
import re
import ast
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..validation.dgts_validator import DGTSValidator, GamingViolation, DGTSValidation

logger = logging.getLogger(__name__)

class StagehandGamingType(Enum):
    """Stagehand-specific gaming violation types"""
    FAKE_ACTIONS = "fake_actions"
    MOCK_OBSERVATIONS = "mock_observations"
    DISABLED_ASSERTIONS = "disabled_assertions"
    HARDCODED_EXTRACTIONS = "hardcoded_extractions"
    BYPASSED_WAITS = "bypassed_waits"
    FAKE_NATURAL_LANGUAGE = "fake_natural_language"
    EMPTY_TEST_BLOCKS = "empty_test_blocks"
    ALWAYS_PASS_CONDITIONS = "always_pass_conditions"
    MOCKED_BROWSERBASE = "mocked_browserbase"
    DISABLED_HEADLESS = "disabled_headless"

@dataclass
class StagehandGamingViolation(GamingViolation):
    """Enhanced gaming violation with Stagehand context"""
    stagehand_context: str
    natural_language_action: Optional[str]
    browser_context: str
    gaming_sophistication: str  # simple, moderate, sophisticated, advanced

class EnhancedDGTSValidator(DGTSValidator):
    """
    Enhanced DGTS validator for Stagehand browser automation gaming detection
    
    Extends base DGTS functionality to detect sophisticated gaming patterns
    specific to AI-generated browser tests and natural language automation.
    """

    def __init__(self, project_path: str):
        super().__init__(project_path)
        
        # Stagehand-specific gaming patterns
        self.stagehand_gaming_patterns = {
            'fake_stagehand_actions': [
                r'stagehand\.act\([\'"]mock[\'"]',
                r'stagehand\.act\([\'"]fake[\'"]',
                r'stagehand\.act\([\'"]dummy[\'"]',
                r'stagehand\.act\([\'"]stub[\'"]',
                r'stagehand\.act\([\'"]placeholder[\'"]',
                r'stagehand\.act\([\'"]return\s+true[\'"]',
                r'stagehand\.act\([\'"]do\s+nothing[\'"]',
                r'stagehand\.act\([\'"]skip[\'"]',
            ],
            'fake_observations': [
                r'stagehand\.observe\([\'"]always\s+true[\'"]',
                r'stagehand\.observe\([\'"]fake\s+success[\'"]',
                r'stagehand\.observe\([\'"]mock\s+result[\'"]',
                r'stagehand\.observe\([\'"]dummy\s+data[\'"]',
                r'stagehand\.observe\([\'"]return\s+\{\}[\'"]',
                r'\.mockReturnValue\(true\)',
                r'\.mockResolvedValue\(.*success.*\)',
            ],
            'hardcoded_extractions': [
                r'stagehand\.extract\([\'"]return\s+[\'"][^\'"]+[\'"][\'"]',
                r'stagehand\.extract\([\'"]hardcoded[\'"]',
                r'stagehand\.extract\([\'"]fixed\s+value[\'"]',
                r'stagehand\.extract\([\'"]mock\s+data[\'"]',
                r'\.extract\(.*\)\.mockReturnValue',
            ],
            'bypassed_waits': [
                r'waitForTimeout\(0\)',
                r'waitForTimeout\(1\)', 
                r'setTimeout\(.*,\s*0\)',
                r'sleep\(0\)',
                r'await\s+Promise\.resolve\(\)',
                r'#.*skip.*wait',
                r'#.*disable.*timeout',
            ],
            'disabled_assertions': [
                r'expect\.not\.toBeTruthy\(\)',
                r'expect\(true\)\.toBeTruthy\(\)',
                r'expect\(false\)\.toBeFalsy\(\)',
                r'expect\(1\)\.toBe\(1\)',
                r'expect\(.*\)\.resolves\.toBeTruthy\(\)',
                r'#.*expect.*disabled',
                r'//.*assertion.*skipped',
            ],
            'mocked_browserbase': [
                r'Browserbase.*mock',
                r'mockBrowserbase',
                r'BROWSERBASE_API_KEY.*fake',
                r'BROWSERBASE_API_KEY.*mock',
                r'browserbaseAPIKey:\s*[\'"]fake[\'"]',
                r'browserbaseAPIKey:\s*[\'"]test[\'"]',
            ],
            'empty_natural_language': [
                r'stagehand\.act\([\'"][\'"]',
                r'stagehand\.observe\([\'"][\'"]',
                r'stagehand\.extract\([\'"][\'"]',
                r'stagehand\.act\([\'"]TODO[\'"]',
                r'stagehand\.act\([\'"]placeholder[\'"]',
            ],
            'always_pass_conditions': [
                r'if\s*\(\s*true\s*\)',
                r'while\s*\(\s*false\s*\)',
                r'assert\s*\(\s*true\s*\)',
                r'expect\(true\)',
                r'\.toBe\(true\)',
                r'===\s*true',
            ]
        }
        
        # Sophisticated gaming detection patterns
        self.sophisticated_patterns = {
            'conditional_gaming': [
                r'if.*process\.env\.NODE_ENV.*mock',
                r'if.*testing.*return\s+true',
                r'NODE_ENV.*===.*test.*mock',
                r'CI.*===.*true.*skip',
            ],
            'dynamic_mocking': [
                r'jest\.mock.*dynamic',
                r'jest\.doMock',
                r'require\.cache.*delete',
                r'import\.meta\.hot.*mock',
            ],
            'stealth_bypasses': [
                r'stagehand\.act.*setTimeout.*0',
                r'page\.evaluate.*return\s+true',
                r'page\.addInitScript.*mock',
                r'context\.addInitScript.*bypass',
            ],
            'ai_generated_fakes': [
                r'\/\*\s*AI\s*generated.*mock',
                r'\/\*\s*Claude.*placeholder',
                r'\/\*\s*GPT.*fake',
                r'\/\*\s*Copilot.*stub',
                r'auto.*generated.*test.*mock',
            ]
        }
        
        # Natural language gaming keywords
        self.natural_language_gaming = [
            "click nothing", "do nothing", "fake click", "pretend to",
            "simulate success", "mock interaction", "dummy action",
            "return fake", "skip this step", "bypass this", "ignore error",
            "always succeed", "auto pass", "force success"
        ]

    async def validate_stagehand_compliance(self) -> Dict[str, Any]:
        """
        Enhanced validation including Stagehand-specific gaming detection
        
        Returns comprehensive validation result with gaming score and violations
        """
        
        # Run base DGTS validation
        base_result = self.validate_no_gaming()
        
        # Run Stagehand-specific validation
        stagehand_violations = await self._detect_stagehand_gaming()
        
        # Combine violations
        all_violations = base_result.violations + [
            self._convert_stagehand_violation(v) for v in stagehand_violations
        ]
        
        # Calculate enhanced gaming score
        enhanced_gaming_score = self._calculate_enhanced_gaming_score(
            base_result.gaming_score, stagehand_violations
        )
        
        # Check for sophisticated gaming patterns
        sophistication_level = self._assess_gaming_sophistication(stagehand_violations)
        
        # Determine if gaming is detected (lower threshold for Stagehand tests)
        is_gaming = enhanced_gaming_score > 0.2 or any(
            v.gaming_sophistication in ["sophisticated", "advanced"] 
            for v in stagehand_violations
        )
        
        return {
            "compliant": not is_gaming,
            "is_gaming": is_gaming,
            "gaming_score": enhanced_gaming_score,
            "sophistication_level": sophistication_level,
            "violations": [
                {
                    "type": v.violation_type,
                    "file": v.file_path,
                    "line": v.line_number,
                    "content": v.content,
                    "severity": v.severity,
                    "explanation": v.explanation,
                    "remediation": v.remediation,
                    "stagehand_context": getattr(v, 'stagehand_context', None),
                    "natural_language_action": getattr(v, 'natural_language_action', None),
                    "gaming_sophistication": getattr(v, 'gaming_sophistication', 'simple')
                }
                for v in all_violations
            ],
            "stagehand_specific_violations": len(stagehand_violations),
            "base_dgts_violations": len(base_result.violations),
            "blocked_features": base_result.blocked_features + [
                f"stagehand_{v.file_path}" for v in stagehand_violations
                if v.gaming_sophistication in ["sophisticated", "advanced"]
            ],
            "suspicious_patterns": base_result.suspicious_patterns + [
                v.violation_type for v in stagehand_violations
            ],
            "message": self._generate_enhanced_message(is_gaming, enhanced_gaming_score, sophistication_level),
            "action_required": "Fix all Stagehand gaming violations before proceeding" if is_gaming else None
        }

    async def _detect_stagehand_gaming(self) -> List[StagehandGamingViolation]:
        """Detect Stagehand-specific gaming patterns"""
        
        violations = []
        
        # Find test files that might contain Stagehand code
        test_patterns = [
            '**/*.spec.ts', '**/*.test.ts', '**/*.spec.js', '**/*.test.js',
            '**/e2e/**/*.ts', '**/e2e/**/*.js',
            '**/tests/**/*.ts', '**/tests/**/*.js'
        ]
        
        for pattern in test_patterns:
            for test_file in self.project_path.glob(pattern):
                try:
                    content = test_file.read_text(encoding='utf-8')
                    file_violations = await self._analyze_stagehand_file(test_file, content)
                    violations.extend(file_violations)
                except Exception as e:
                    logger.warning(f"Could not analyze Stagehand file {test_file}: {e}")
                    continue
        
        return violations

    async def _analyze_stagehand_file(
        self, 
        file_path: Path, 
        content: str
    ) -> List[StagehandGamingViolation]:
        """Analyze individual file for Stagehand gaming patterns"""
        
        violations = []
        lines = content.split('\n')
        
        # Check if file actually uses Stagehand
        uses_stagehand = 'stagehand' in content.lower() or 'browserbase' in content.lower()
        
        if not uses_stagehand:
            return violations
        
        for line_num, line in enumerate(lines, 1):
            line_clean = line.strip()
            line_lower = line_clean.lower()
            
            # Check each Stagehand gaming pattern category
            for category, patterns in self.stagehand_gaming_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        violation = self._create_stagehand_violation(
                            category, file_path, line_num, line_clean, pattern
                        )
                        violations.append(violation)
            
            # Check sophisticated patterns
            for category, patterns in self.sophisticated_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        violation = self._create_stagehand_violation(
                            category, file_path, line_num, line_clean, pattern,
                            sophistication="sophisticated"
                        )
                        violations.append(violation)
            
            # Check natural language gaming
            for nl_gaming in self.natural_language_gaming:
                if nl_gaming.lower() in line_lower:
                    violation = self._create_stagehand_violation(
                        "fake_natural_language", file_path, line_num, line_clean, nl_gaming,
                        natural_language_action=nl_gaming
                    )
                    violations.append(violation)
        
        # Check for structural gaming patterns
        structural_violations = self._detect_structural_gaming(file_path, content)
        violations.extend(structural_violations)
        
        return violations

    def _create_stagehand_violation(
        self,
        violation_type: str,
        file_path: Path,
        line_number: int,
        content: str,
        pattern: str,
        sophistication: str = "simple",
        natural_language_action: str = None
    ) -> StagehandGamingViolation:
        """Create Stagehand-specific gaming violation"""
        
        # Map violation types to Stagehand gaming types
        stagehand_type_map = {
            'fake_stagehand_actions': StagehandGamingType.FAKE_ACTIONS,
            'fake_observations': StagehandGamingType.MOCK_OBSERVATIONS,
            'hardcoded_extractions': StagehandGamingType.HARDCODED_EXTRACTIONS,
            'bypassed_waits': StagehandGamingType.BYPASSED_WAITS,
            'disabled_assertions': StagehandGamingType.DISABLED_ASSERTIONS,
            'mocked_browserbase': StagehandGamingType.MOCKED_BROWSERBASE,
            'empty_natural_language': StagehandGamingType.FAKE_NATURAL_LANGUAGE,
            'always_pass_conditions': StagehandGamingType.ALWAYS_PASS_CONDITIONS,
            'conditional_gaming': StagehandGamingType.ALWAYS_PASS_CONDITIONS,
            'dynamic_mocking': StagehandGamingType.MOCKED_BROWSERBASE,
            'stealth_bypasses': StagehandGamingType.BYPASSED_WAITS,
            'ai_generated_fakes': StagehandGamingType.FAKE_NATURAL_LANGUAGE,
            'fake_natural_language': StagehandGamingType.FAKE_NATURAL_LANGUAGE
        }
        
        stagehand_type = stagehand_type_map.get(violation_type, StagehandGamingType.FAKE_ACTIONS)
        
        # Determine severity based on sophistication and type
        if sophistication in ["sophisticated", "advanced"]:
            severity = "critical"
        elif stagehand_type in [
            StagehandGamingType.MOCKED_BROWSERBASE, 
            StagehandGamingType.BYPASSED_WAITS,
            StagehandGamingType.ALWAYS_PASS_CONDITIONS
        ]:
            severity = "critical"
        else:
            severity = "error"
        
        return StagehandGamingViolation(
            violation_type=f"STAGEHAND_{stagehand_type.value.upper()}",
            file_path=str(file_path),
            line_number=line_number,
            content=content,
            severity=severity,
            explanation=self._get_stagehand_explanation(stagehand_type, sophistication),
            remediation=self._get_stagehand_remediation(stagehand_type),
            stagehand_context=self._extract_stagehand_context(content),
            natural_language_action=natural_language_action,
            browser_context=self._extract_browser_context(content),
            gaming_sophistication=sophistication
        )

    def _detect_structural_gaming(self, file_path: Path, content: str) -> List[StagehandGamingViolation]:
        """Detect structural gaming patterns in test files"""
        
        violations = []
        
        # Count real vs fake Stagehand actions
        stagehand_actions = re.findall(r'stagehand\.act\([\'"]([^\'"]+)[\'"]', content)
        fake_actions = [action for action in stagehand_actions if any(
            fake_word in action.lower() 
            for fake_word in ['mock', 'fake', 'dummy', 'stub', 'placeholder', 'nothing']
        )]
        
        if len(fake_actions) > len(stagehand_actions) / 2:
            violations.append(StagehandGamingViolation(
                violation_type="STAGEHAND_STRUCTURAL_GAMING",
                file_path=str(file_path),
                line_number=0,
                content=f"Too many fake actions: {len(fake_actions)}/{len(stagehand_actions)}",
                severity="critical",
                explanation="Test file contains more fake Stagehand actions than real ones",
                remediation="Replace fake actions with genuine browser interactions",
                stagehand_context="structural_analysis",
                natural_language_action=None,
                browser_context="test_file_structure",
                gaming_sophistication="moderate"
            ))
        
        # Check for empty test blocks with Stagehand imports
        if 'stagehand' in content.lower() and content.count('test(') > 0:
            empty_tests = re.findall(r'test\([^{]*\{[^}]*\}', content, re.DOTALL)
            for empty_test in empty_tests:
                if len(empty_test.strip()) < 100:  # Very short test
                    violations.append(StagehandGamingViolation(
                        violation_type="STAGEHAND_EMPTY_TEST_BLOCK",
                        file_path=str(file_path),
                        line_number=0,
                        content=empty_test.strip(),
                        severity="error", 
                        explanation="Empty or minimal test block with Stagehand import",
                        remediation="Implement proper test logic with meaningful assertions",
                        stagehand_context="empty_test_detection",
                        natural_language_action=None,
                        browser_context="test_structure",
                        gaming_sophistication="simple"
                    ))
        
        return violations

    def _extract_stagehand_context(self, content: str) -> str:
        """Extract Stagehand-specific context from code line"""
        
        if 'stagehand.act' in content:
            return "action_execution"
        elif 'stagehand.observe' in content:
            return "observation_check"
        elif 'stagehand.extract' in content:
            return "data_extraction"
        elif 'Stagehand' in content:
            return "stagehand_initialization"
        else:
            return "unknown_context"

    def _extract_browser_context(self, content: str) -> str:
        """Extract browser automation context"""
        
        if 'page.' in content:
            return "playwright_page"
        elif 'browser.' in content:
            return "playwright_browser" 
        elif 'context.' in content:
            return "playwright_context"
        elif 'browserbase' in content.lower():
            return "browserbase_cloud"
        else:
            return "unknown_browser"

    def _calculate_enhanced_gaming_score(
        self, 
        base_score: float, 
        stagehand_violations: List[StagehandGamingViolation]
    ) -> float:
        """Calculate enhanced gaming score including Stagehand violations"""
        
        if not stagehand_violations:
            return base_score
        
        # Weight Stagehand violations more heavily
        stagehand_weights = {
            'simple': 0.3,
            'moderate': 0.6,
            'sophisticated': 1.0,
            'advanced': 1.5
        }
        
        stagehand_score = sum(
            stagehand_weights.get(v.gaming_sophistication, 0.3) 
            for v in stagehand_violations
        )
        
        # Normalize to 0-1 scale
        max_expected_stagehand_violations = 10
        normalized_stagehand_score = min(stagehand_score / max_expected_stagehand_violations, 1.0)
        
        # Combine with base score (weighted average)
        combined_score = (base_score * 0.4) + (normalized_stagehand_score * 0.6)
        
        return min(combined_score, 1.0)

    def _assess_gaming_sophistication(
        self, 
        violations: List[StagehandGamingViolation]
    ) -> str:
        """Assess overall gaming sophistication level"""
        
        if not violations:
            return "none"
        
        sophistication_levels = [v.gaming_sophistication for v in violations]
        
        if "advanced" in sophistication_levels:
            return "advanced"
        elif "sophisticated" in sophistication_levels:
            return "sophisticated" 
        elif "moderate" in sophistication_levels:
            return "moderate"
        else:
            return "simple"

    def _convert_stagehand_violation(self, violation: StagehandGamingViolation) -> GamingViolation:
        """Convert Stagehand violation to base gaming violation for compatibility"""
        
        return GamingViolation(
            violation_type=violation.violation_type,
            file_path=violation.file_path,
            line_number=violation.line_number,
            content=violation.content,
            severity=violation.severity,
            explanation=violation.explanation,
            remediation=violation.remediation
        )

    def _get_stagehand_explanation(self, gaming_type: StagehandGamingType, sophistication: str) -> str:
        """Get explanation for Stagehand gaming violation"""
        
        explanations = {
            StagehandGamingType.FAKE_ACTIONS: "Stagehand action uses fake/mock behavior instead of real browser interaction",
            StagehandGamingType.MOCK_OBSERVATIONS: "Stagehand observation is mocked to always return success",
            StagehandGamingType.HARDCODED_EXTRACTIONS: "Stagehand extraction returns hardcoded values instead of dynamic data",
            StagehandGamingType.BYPASSED_WAITS: "Wait mechanisms are bypassed or set to zero timeout",
            StagehandGamingType.DISABLED_ASSERTIONS: "Test assertions are disabled or made meaningless",
            StagehandGamingType.MOCKED_BROWSERBASE: "Browserbase API is mocked instead of using real cloud infrastructure",
            StagehandGamingType.FAKE_NATURAL_LANGUAGE: "Natural language actions are empty or fake",
            StagehandGamingType.ALWAYS_PASS_CONDITIONS: "Conditions are hardcoded to always pass",
            StagehandGamingType.EMPTY_TEST_BLOCKS: "Test blocks are empty or contain minimal code",
        }
        
        base_explanation = explanations.get(gaming_type, "Unknown Stagehand gaming pattern detected")
        
        if sophistication in ["sophisticated", "advanced"]:
            return f"{base_explanation} - {sophistication.capitalize()} gaming pattern detected"
        
        return base_explanation

    def _get_stagehand_remediation(self, gaming_type: StagehandGamingType) -> str:
        """Get remediation for Stagehand gaming violation"""
        
        remediations = {
            StagehandGamingType.FAKE_ACTIONS: "Replace with genuine browser interactions using proper Stagehand natural language",
            StagehandGamingType.MOCK_OBSERVATIONS: "Implement real observation checks that validate actual page state",
            StagehandGamingType.HARDCODED_EXTRACTIONS: "Use dynamic extraction that reads actual page content",
            StagehandGamingType.BYPASSED_WAITS: "Implement proper wait conditions for reliable test execution",
            StagehandGamingType.DISABLED_ASSERTIONS: "Enable proper assertions that validate expected behavior",
            StagehandGamingType.MOCKED_BROWSERBASE: "Use real Browserbase API with proper authentication",
            StagehandGamingType.FAKE_NATURAL_LANGUAGE: "Write meaningful natural language descriptions for actions",
            StagehandGamingType.ALWAYS_PASS_CONDITIONS: "Use dynamic conditions that reflect real application state",
            StagehandGamingType.EMPTY_TEST_BLOCKS: "Implement comprehensive test logic with proper validations",
        }
        
        return remediations.get(gaming_type, "Fix gaming behavior and implement proper Stagehand test")

    def _generate_enhanced_message(self, is_gaming: bool, gaming_score: float, sophistication: str) -> str:
        """Generate enhanced validation message"""
        
        if not is_gaming:
            return "Enhanced DGTS validation passed - No Stagehand gaming detected"
        
        message_parts = [
            f"Enhanced DGTS validation failed - Gaming score: {gaming_score:.2f}"
        ]
        
        if sophistication != "none":
            message_parts.append(f"Sophistication level: {sophistication}")
        
        if gaming_score > 0.5:
            message_parts.append("HIGH RISK - Extensive gaming patterns detected")
        elif gaming_score > 0.3:
            message_parts.append("MODERATE RISK - Multiple gaming patterns found")
        
        return " | ".join(message_parts)

# Standalone functions for easy integration
async def validate_stagehand_tests(project_path: str = ".") -> Dict[str, Any]:
    """
    Validate Stagehand tests for gaming patterns
    
    Args:
        project_path: Path to the project root
        
    Returns:
        Validation result with Stagehand-specific gaming detection
    """
    
    validator = EnhancedDGTSValidator(project_path)
    return await validator.validate_stagehand_compliance()

def detect_natural_language_gaming(test_content: str) -> List[Dict[str, Any]]:
    """
    Detect gaming in natural language test descriptions
    
    Args:
        test_content: Content of test file to analyze
        
    Returns:
        List of detected natural language gaming violations
    """
    
    validator = EnhancedDGTSValidator(".")
    violations = []
    
    lines = test_content.split('\n')
    for line_num, line in enumerate(lines, 1):
        for gaming_phrase in validator.natural_language_gaming:
            if gaming_phrase.lower() in line.lower():
                violations.append({
                    "line": line_num,
                    "content": line.strip(),
                    "gaming_phrase": gaming_phrase,
                    "severity": "error",
                    "explanation": f"Natural language gaming detected: '{gaming_phrase}'"
                })
    
    return violations

async def enforce_stagehand_quality(
    test_files: List[str],
    project_path: str = "."
) -> Dict[str, Any]:
    """
    Enforce quality standards for Stagehand tests
    
    Args:
        test_files: List of test files to validate
        project_path: Path to the project root
        
    Returns:
        Quality enforcement result with pass/fail decision
    """
    
    validator = EnhancedDGTSValidator(project_path)
    validation_result = await validator.validate_stagehand_compliance()
    
    # Check quality thresholds
    max_allowed_gaming_score = 0.2
    max_allowed_violations = 5
    
    quality_passed = (
        validation_result["gaming_score"] <= max_allowed_gaming_score and
        validation_result["stagehand_specific_violations"] <= max_allowed_violations and
        validation_result["sophistication_level"] not in ["sophisticated", "advanced"]
    )
    
    return {
        "quality_passed": quality_passed,
        "gaming_score": validation_result["gaming_score"],
        "violations_count": validation_result["stagehand_specific_violations"],
        "sophistication_level": validation_result["sophistication_level"],
        "validation_details": validation_result,
        "action_required": "Fix Stagehand gaming violations" if not quality_passed else None,
        "quality_threshold": {
            "max_gaming_score": max_allowed_gaming_score,
            "max_violations": max_allowed_violations
        }
    }