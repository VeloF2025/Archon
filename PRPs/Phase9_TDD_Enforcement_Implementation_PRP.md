# Project Requirements Prompt (PRP)
# Phase 9: Test-Driven Development Enforcement System Implementation

**Project**: Archon Phase 9 TDD Enforcement with browserbase-claude-code-stagehand  
**Version**: 9.0  
**Date**: January 31, 2025  
**PRD Reference**: Phase9_TDD_Enforcement_PRD.md  

## 1. Implementation Overview

This PRP provides comprehensive implementation requirements for Phase 9 TDD Enforcement System, integrating Stagehand's natural language testing capabilities with Browserbase cloud infrastructure to enforce mandatory Test-Driven Development across all Archon agents. This phase transforms Archon from a development assistant into a strict TDD enforcer, reducing errors by 90% through AI-powered test generation and cloud-based execution.

**Core Deliverables**:
- Stagehand AI integration for natural language test generation
- Browserbase cloud testing infrastructure setup
- Enhanced DGTS validator for test gaming prevention
- Test-first validation gates blocking non-TDD development
- Claude Code Task tool integration for test execution
- Real-time test result streaming via WebSocket
- PRD-to-test automatic generation pipeline

## 2. Core Implementation Tasks

### 2.1 Stagehand Test Engine Integration

**File**: `python/src/services/stagehand_service.py`

```python
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from playwright.async_api import Page, Browser
import stagehand
from stagehand import StagehandPage, StagehandConfig

@dataclass
class StagehandTestResult:
    """Result from Stagehand test execution"""
    test_name: str
    status: str  # 'passed', 'failed', 'skipped'
    duration_ms: int
    error_message: Optional[str] = None
    screenshot_path: Optional[str] = None
    video_path: Optional[str] = None
    
@dataclass
class NaturalLanguageTest:
    """Natural language test specification"""
    name: str
    description: str
    actions: List[str]  # Natural language actions
    assertions: List[str]  # Natural language validations
    tags: List[str] = None  # @smoke, @regression, etc.

class StagehandTestEngine:
    """
    Core Stagehand integration for natural language testing.
    Converts PRD requirements into executable tests without technical selectors.
    """
    
    def __init__(self, config: Optional[StagehandConfig] = None):
        self.config = config or StagehandConfig(
            model="gpt-4-turbo",
            api_key=os.getenv("STAGEHAND_API_KEY"),
            debug_mode=os.getenv("STAGEHAND_DEBUG", "false") == "true",
            strict_mode=True  # Enforce exact matching
        )
        self.stagehand_pages: Dict[str, StagehandPage] = {}
        
    async def initialize_page(self, page: Page, session_id: str) -> StagehandPage:
        """Initialize Stagehand on a Playwright page"""
        stagehand_page = await stagehand.init(page, self.config)
        self.stagehand_pages[session_id] = stagehand_page
        return stagehand_page
        
    async def generate_tests_from_prd(
        self, 
        prd_content: str,
        test_strategy: str = "comprehensive"
    ) -> List[NaturalLanguageTest]:
        """
        Generate natural language tests from PRD documentation.
        Uses AI to extract requirements and create test scenarios.
        """
        # Parse PRD for testable requirements
        requirements = await self._extract_requirements(prd_content)
        
        tests = []
        for req in requirements:
            # Generate test scenarios for each requirement
            scenarios = await self._generate_test_scenarios(req, test_strategy)
            
            for scenario in scenarios:
                test = NaturalLanguageTest(
                    name=scenario['name'],
                    description=scenario['description'],
                    actions=[
                        self._convert_to_natural_language(step) 
                        for step in scenario['steps']
                    ],
                    assertions=[
                        self._convert_to_natural_language(assertion)
                        for assertion in scenario['assertions']
                    ],
                    tags=scenario.get('tags', ['@generated'])
                )
                tests.append(test)
                
        return tests
        
    async def execute_natural_language_test(
        self,
        stagehand_page: StagehandPage,
        test: NaturalLanguageTest
    ) -> StagehandTestResult:
        """Execute a natural language test using Stagehand"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Execute actions
            for action in test.actions:
                await stagehand_page.act(action)
                
            # Validate assertions
            for assertion in test.assertions:
                result = await stagehand_page.observe(assertion)
                if not result:
                    raise AssertionError(f"Assertion failed: {assertion}")
                    
            duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            
            return StagehandTestResult(
                test_name=test.name,
                status='passed',
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            
            # Capture debugging information
            screenshot_path = await self._capture_failure_screenshot(
                stagehand_page, test.name
            )
            
            return StagehandTestResult(
                test_name=test.name,
                status='failed',
                duration_ms=duration_ms,
                error_message=str(e),
                screenshot_path=screenshot_path
            )
            
    async def convert_playwright_to_stagehand(
        self,
        playwright_test_path: str
    ) -> str:
        """
        Convert traditional Playwright tests to Stagehand natural language.
        Reduces code by 70% while maintaining functionality.
        """
        with open(playwright_test_path, 'r') as f:
            playwright_code = f.read()
            
        # Parse Playwright selectors and actions
        selectors = self._extract_selectors(playwright_code)
        actions = self._extract_actions(playwright_code)
        
        # Convert to natural language
        stagehand_code = []
        for action in actions:
            natural_action = await self._selector_to_natural_language(
                action['selector'],
                action['action'],
                action.get('value')
            )
            stagehand_code.append(f"await stagehand.act('{natural_action}');")
            
        return '\n'.join(stagehand_code)
        
    async def _selector_to_natural_language(
        self,
        selector: str,
        action: str,
        value: Optional[str] = None
    ) -> str:
        """Convert technical selector to natural language"""
        # Examples of conversion
        conversions = {
            "click": f"Click the {self._describe_element(selector)}",
            "fill": f"Enter '{value}' in the {self._describe_element(selector)}",
            "check": f"Check the {self._describe_element(selector)}",
            "select": f"Select '{value}' from the {self._describe_element(selector)}"
        }
        return conversions.get(action, f"Interact with {selector}")
        
    def _describe_element(self, selector: str) -> str:
        """Convert selector to human-readable description"""
        if 'button' in selector.lower():
            return "button"
        elif 'input' in selector.lower():
            return "input field"
        elif 'select' in selector.lower():
            return "dropdown"
        else:
            return "element"
```

### 2.2 Browserbase Cloud Infrastructure

**File**: `python/src/services/browserbase_service.py`

```python
import os
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from browserbase import Browserbase, Session, SessionConfig
import aiohttp

@dataclass
class CloudTestSession:
    """Browserbase cloud test session"""
    session_id: str
    browser_type: str  # 'chromium', 'firefox', 'webkit'
    region: str  # 'us-west', 'eu-central', 'asia-pacific'
    status: str  # 'running', 'completed', 'failed'
    video_url: Optional[str] = None
    logs_url: Optional[str] = None
    
@dataclass
class CloudTestResult:
    """Aggregated results from cloud test execution"""
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration_ms: int
    browser_results: Dict[str, Any]
    video_recordings: List[str]
    
class BrowserbaseCloudExecutor:
    """
    Manages test execution in Browserbase cloud infrastructure.
    Provides parallel execution across multiple browsers and regions.
    """
    
    def __init__(self):
        self.client = Browserbase(
            api_key=os.getenv("BROWSERBASE_API_KEY"),
            project_id=os.getenv("BROWSERBASE_PROJECT_ID")
        )
        self.max_parallel_sessions = int(
            os.getenv("BROWSERBASE_PARALLEL_SESSIONS", "5")
        )
        self.timeout_ms = int(
            os.getenv("BROWSERBASE_TIMEOUT", "120000")
        )
        
    async def create_test_session(
        self,
        browser_type: str = "chromium",
        region: str = "us-west",
        record_video: bool = True
    ) -> CloudTestSession:
        """Create a new Browserbase test session"""
        config = SessionConfig(
            browser=browser_type,
            region=region,
            viewport={"width": 1920, "height": 1080},
            record_video=record_video,
            timeout=self.timeout_ms,
            # Enable debugging features
            capture_console=True,
            capture_network=True,
            capture_screenshots=True
        )
        
        session = await self.client.create_session(config)
        
        return CloudTestSession(
            session_id=session.id,
            browser_type=browser_type,
            region=region,
            status='running'
        )
        
    async def execute_test_suite_in_cloud(
        self,
        test_suite: List[NaturalLanguageTest],
        browsers: List[str] = None,
        regions: List[str] = None
    ) -> CloudTestResult:
        """
        Execute test suite across multiple browsers and regions.
        Provides parallel execution for faster feedback.
        """
        browsers = browsers or ['chromium', 'firefox', 'webkit']
        regions = regions or ['us-west']
        
        # Create sessions for each browser/region combination
        sessions = []
        for browser in browsers:
            for region in regions:
                session = await self.create_test_session(
                    browser_type=browser,
                    region=region,
                    record_video=True
                )
                sessions.append(session)
                
        # Execute tests in parallel
        tasks = []
        for session in sessions:
            task = asyncio.create_task(
                self._execute_tests_in_session(session, test_suite)
            )
            tasks.append(task)
            
        # Wait for all sessions to complete
        results = await asyncio.gather(*tasks)
        
        # Aggregate results
        return self._aggregate_results(results, sessions)
        
    async def _execute_tests_in_session(
        self,
        session: CloudTestSession,
        tests: List[NaturalLanguageTest]
    ) -> Dict[str, Any]:
        """Execute tests in a specific cloud session"""
        # Connect to session
        browser = await self.client.connect_to_session(session.session_id)
        page = await browser.new_page()
        
        # Initialize Stagehand
        stagehand_engine = StagehandTestEngine()
        stagehand_page = await stagehand_engine.initialize_page(
            page, session.session_id
        )
        
        results = []
        for test in tests:
            result = await stagehand_engine.execute_natural_language_test(
                stagehand_page, test
            )
            results.append(result)
            
        # Close session
        await browser.close()
        
        # Get video and logs
        session_details = await self.client.get_session(session.session_id)
        
        return {
            'session_id': session.session_id,
            'browser': session.browser_type,
            'region': session.region,
            'results': results,
            'video_url': session_details.video_url,
            'logs_url': session_details.logs_url
        }
        
    def _aggregate_results(
        self,
        results: List[Dict[str, Any]],
        sessions: List[CloudTestSession]
    ) -> CloudTestResult:
        """Aggregate results from multiple test sessions"""
        total_tests = 0
        passed = 0
        failed = 0
        skipped = 0
        total_duration = 0
        browser_results = {}
        video_recordings = []
        
        for result in results:
            test_results = result['results']
            browser = result['browser']
            
            browser_stats = {
                'passed': 0,
                'failed': 0,
                'skipped': 0
            }
            
            for test_result in test_results:
                total_tests += 1
                total_duration += test_result.duration_ms
                
                if test_result.status == 'passed':
                    passed += 1
                    browser_stats['passed'] += 1
                elif test_result.status == 'failed':
                    failed += 1
                    browser_stats['failed'] += 1
                else:
                    skipped += 1
                    browser_stats['skipped'] += 1
                    
            browser_results[browser] = browser_stats
            
            if result.get('video_url'):
                video_recordings.append(result['video_url'])
                
        return CloudTestResult(
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            skipped=skipped,
            duration_ms=total_duration,
            browser_results=browser_results,
            video_recordings=video_recordings
        )
        
    async def stream_test_progress(
        self,
        session_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream real-time test progress from cloud session"""
        while True:
            session = await self.client.get_session(session_id)
            
            yield {
                'session_id': session_id,
                'status': session.status,
                'progress': session.progress,
                'current_test': session.current_test,
                'logs': session.recent_logs
            }
            
            if session.status in ['completed', 'failed']:
                break
                
            await asyncio.sleep(1)  # Poll every second
```

### 2.3 Enhanced DGTS Test Gaming Prevention

**File**: `python/src/agents/validation/enhanced_dgts_validator.py`

```python
import re
import ast
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TestGamingViolation:
    """Represents a test gaming violation"""
    violation_type: str
    severity: str  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    file_path: str
    line_number: int
    code_snippet: str
    explanation: str
    remediation: str
    
@dataclass
class TestValidationResult:
    """Result of test validation"""
    is_valid: bool
    is_gaming: bool
    gaming_score: float  # 0.0 (clean) to 1.0 (heavily gamed)
    violations: List[TestGamingViolation]
    recommendations: List[str]

class EnhancedDGTSValidator:
    """
    Enhanced DGTS validator specifically for test gaming detection.
    Prevents fake tests, mocked assertions, and meaningless validations.
    """
    
    # Critical patterns that indicate test gaming
    CRITICAL_GAMING_PATTERNS = {
        'ALWAYS_TRUE': (
            r'expect\s*\(\s*(true|1|".*")\s*\)\s*\.\s*toBe\s*\(\s*\1\s*\)',
            "Tautological assertion that always passes"
        ),
        'EMPTY_TEST': (
            r'(it|test|describe)\s*\([\'"][^\'"]+[\'"]\s*,\s*\(\)\s*=>\s*\{\s*\}\)',
            "Empty test block with no assertions"
        ),
        'SKIP_ALL': (
            r'\.(skip|only)\s*\(',
            "Skipping or focusing tests inappropriately"
        ),
        'MOCK_EVERYTHING': (
            r'jest\.mock\s*\(\s*[\'"].*[\'"]\s*,\s*\(\)\s*=>\s*\(\s*\{\s*\}\s*\)\s*\)',
            "Mocking entire modules with empty implementations"
        ),
        'NO_ASSERTIONS': (
            r'^(?!.*expect\s*\().*$',
            "Test has no assertions to validate behavior"
        ),
        'COMMENTED_ASSERTIONS': (
            r'//\s*expect\s*\(',
            "Assertions are commented out"
        ),
        'FAKE_ASYNC': (
            r'return\s+Promise\s*\.\s*resolve\s*\(\s*\)',
            "Fake async resolution without actual testing"
        ),
        'CONSOLE_LOG_ONLY': (
            r'^[^/]*console\s*\.\s*log[^/]*$',
            "Test only contains console.log statements"
        )
    }
    
    # Stagehand-specific gaming patterns
    STAGEHAND_GAMING_PATTERNS = {
        'FAKE_ACTIONS': (
            r'stagehand\.act\s*\(\s*[\'"][\'"]\s*\)',
            "Empty Stagehand action"
        ),
        'FAKE_OBSERVATIONS': (
            r'stagehand\.observe\s*\(\s*[\'"][\'"]\s*\)',
            "Empty Stagehand observation"
        ),
        'HARDCODED_RESULTS': (
            r'stagehand\.(act|observe)\s*\(\s*[\'"][^\'"]*(true|false|success)[^\'"][\'"]\s*\)',
            "Hardcoded test results in natural language"
        ),
        'NO_REAL_INTERACTIONS': (
            r'//\s*TODO:\s*Add\s+real\s+interactions',
            "Placeholder for unimplemented interactions"
        )
    }
    
    def __init__(self):
        self.violation_history = []
        self.gaming_threshold = float(os.getenv("GAMING_DETECTION_THRESHOLD", "0.3"))
        
    async def validate_test_file(
        self,
        file_path: str,
        content: Optional[str] = None
    ) -> TestValidationResult:
        """Validate a test file for gaming patterns"""
        if content is None:
            with open(file_path, 'r') as f:
                content = f.read()
                
        violations = []
        
        # Check for critical gaming patterns
        for pattern_name, (pattern, explanation) in self.CRITICAL_GAMING_PATTERNS.items():
            matches = list(re.finditer(pattern, content, re.MULTILINE))
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                violations.append(TestGamingViolation(
                    violation_type=f"DGTS_{pattern_name}",
                    severity="CRITICAL",
                    file_path=file_path,
                    line_number=line_num,
                    code_snippet=match.group(0),
                    explanation=explanation,
                    remediation=self._get_remediation(pattern_name)
                ))
                
        # Check for Stagehand-specific gaming
        if 'stagehand' in content.lower():
            for pattern_name, (pattern, explanation) in self.STAGEHAND_GAMING_PATTERNS.items():
                matches = list(re.finditer(pattern, content, re.IGNORECASE))
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    violations.append(TestGamingViolation(
                        violation_type=f"DGTS_STAGEHAND_{pattern_name}",
                        severity="HIGH",
                        file_path=file_path,
                        line_number=line_num,
                        code_snippet=match.group(0),
                        explanation=explanation,
                        remediation=self._get_stagehand_remediation(pattern_name)
                    ))
                    
        # Calculate gaming score
        gaming_score = self._calculate_gaming_score(violations, content)
        
        # Determine if test is valid
        is_gaming = gaming_score >= self.gaming_threshold
        is_valid = not is_gaming and len([v for v in violations if v.severity == "CRITICAL"]) == 0
        
        return TestValidationResult(
            is_valid=is_valid,
            is_gaming=is_gaming,
            gaming_score=gaming_score,
            violations=violations,
            recommendations=self._generate_recommendations(violations)
        )
        
    async def validate_test_suite(
        self,
        test_directory: str
    ) -> Dict[str, TestValidationResult]:
        """Validate all test files in a directory"""
        results = {}
        test_path = Path(test_directory)
        
        # Find all test files
        test_files = list(test_path.rglob("*.spec.ts")) + \
                    list(test_path.rglob("*.spec.js")) + \
                    list(test_path.rglob("*.test.ts")) + \
                    list(test_path.rglob("*.test.js"))
                    
        for test_file in test_files:
            result = await self.validate_test_file(str(test_file))
            results[str(test_file)] = result
            
        return results
        
    def _calculate_gaming_score(
        self,
        violations: List[TestGamingViolation],
        content: str
    ) -> float:
        """Calculate gaming score from 0.0 (clean) to 1.0 (heavily gamed)"""
        score = 0.0
        
        # Weight violations by severity
        severity_weights = {
            'CRITICAL': 0.3,
            'HIGH': 0.2,
            'MEDIUM': 0.1,
            'LOW': 0.05
        }
        
        for violation in violations:
            score += severity_weights.get(violation.severity, 0.05)
            
        # Additional checks
        if 'expect' not in content:
            score += 0.3  # No assertions at all
            
        if content.count('expect') < content.count('it('):
            score += 0.2  # Less assertions than tests
            
        # Cap at 1.0
        return min(score, 1.0)
        
    def _get_remediation(self, pattern_name: str) -> str:
        """Get remediation advice for specific violation"""
        remediations = {
            'ALWAYS_TRUE': "Write assertions that validate actual behavior, not tautologies",
            'EMPTY_TEST': "Add meaningful test steps and assertions",
            'SKIP_ALL': "Remove skip/only modifiers before committing",
            'MOCK_EVERYTHING': "Mock only external dependencies, test real implementations",
            'NO_ASSERTIONS': "Add expect() statements to validate behavior",
            'COMMENTED_ASSERTIONS': "Uncomment or remove commented assertions",
            'FAKE_ASYNC': "Implement real async operations with actual waiting",
            'CONSOLE_LOG_ONLY': "Replace console.log with proper assertions"
        }
        return remediations.get(pattern_name, "Fix the gaming pattern")
        
    def _get_stagehand_remediation(self, pattern_name: str) -> str:
        """Get Stagehand-specific remediation advice"""
        remediations = {
            'FAKE_ACTIONS': "Provide meaningful natural language actions",
            'FAKE_OBSERVATIONS': "Provide specific observations to validate",
            'HARDCODED_RESULTS': "Use dynamic validations based on actual state",
            'NO_REAL_INTERACTIONS': "Implement actual user interactions"
        }
        return remediations.get(pattern_name, "Fix the Stagehand gaming pattern")
        
    def _generate_recommendations(
        self,
        violations: List[TestGamingViolation]
    ) -> List[str]:
        """Generate recommendations based on violations"""
        recommendations = []
        
        if any(v.violation_type.endswith('NO_ASSERTIONS') for v in violations):
            recommendations.append(
                "Add meaningful assertions using expect() or Stagehand.observe()"
            )
            
        if any(v.violation_type.endswith('MOCK_EVERYTHING') for v in violations):
            recommendations.append(
                "Test real implementations, only mock external dependencies"
            )
            
        if any('STAGEHAND' in v.violation_type for v in violations):
            recommendations.append(
                "Use natural language that describes real user actions and validations"
            )
            
        if not violations:
            recommendations.append("Test file appears authentic and well-structured")
            
        return recommendations
```

### 2.4 Test-First Enforcement Gate

**File**: `python/src/services/tdd_enforcement.py`

```python
import os
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

@dataclass
class TDDEnforcementStatus:
    """Status of TDD enforcement for a feature"""
    feature_id: str
    tests_generated: bool
    tests_validated: bool
    tests_executed: bool
    tests_passing: bool
    implementation_unlocked: bool
    gaming_detected: bool
    bypass_token: Optional[str] = None
    
@dataclass
class TDDBypassToken:
    """Emergency bypass token for critical fixes"""
    token: str
    reason: str
    approver: str
    expires_at: datetime
    audit_logged: bool

class TDDEnforcementGate:
    """
    Core TDD enforcement system that blocks all non-test-first development.
    Integrates with Stagehand, Browserbase, and DGTS for complete validation.
    """
    
    def __init__(self):
        self.stagehand_engine = StagehandTestEngine()
        self.browserbase_executor = BrowserbaseCloudExecutor()
        self.dgts_validator = EnhancedDGTSValidator()
        self.enforcement_level = os.getenv("TDD_ENFORCEMENT_LEVEL", "strict")
        self.bypass_audit_log = []
        self.active_features = {}
        
    async def enforce_test_first_development(
        self,
        feature_request: Dict[str, Any],
        bypass_token: Optional[str] = None
    ) -> TDDEnforcementStatus:
        """
        Main enforcement gate that ensures tests exist and pass before development.
        This is the primary integration point with Claude Code Task tool.
        """
        feature_id = feature_request.get('id', str(uuid.uuid4()))
        
        # Check for emergency bypass
        if bypass_token and await self._validate_bypass_token(bypass_token):
            return TDDEnforcementStatus(
                feature_id=feature_id,
                tests_generated=False,
                tests_validated=False,
                tests_executed=False,
                tests_passing=False,
                implementation_unlocked=True,
                gaming_detected=False,
                bypass_token=bypass_token
            )
            
        # Step 1: Generate tests from requirements
        prd_content = feature_request.get('prd_content', '')
        if not prd_content:
            raise ValueError("PRD content required for test generation")
            
        tests = await self.stagehand_engine.generate_tests_from_prd(
            prd_content,
            test_strategy='comprehensive'
        )
        
        if not tests:
            raise ValueError("No tests generated from PRD")
            
        # Step 2: Validate tests aren't gamed
        test_content = self._serialize_tests(tests)
        validation_result = await self.dgts_validator.validate_test_file(
            f"generated_{feature_id}.spec.ts",
            test_content
        )
        
        if validation_result.is_gaming:
            return TDDEnforcementStatus(
                feature_id=feature_id,
                tests_generated=True,
                tests_validated=False,
                tests_executed=False,
                tests_passing=False,
                implementation_unlocked=False,
                gaming_detected=True
            )
            
        # Step 3: Execute tests (must fail initially - Red phase of TDD)
        initial_results = await self._execute_initial_tests(tests)
        
        if initial_results['all_passing']:
            raise ValueError(
                "Tests are passing without implementation. "
                "This indicates either the feature already exists or tests are invalid."
            )
            
        # Step 4: Create test files and unlock development
        await self._create_test_files(feature_id, tests)
        
        # Step 5: Register feature for continuous validation
        self.active_features[feature_id] = {
            'tests': tests,
            'status': 'development_unlocked',
            'created_at': datetime.now()
        }
        
        return TDDEnforcementStatus(
            feature_id=feature_id,
            tests_generated=True,
            tests_validated=True,
            tests_executed=True,
            tests_passing=False,  # Should be false initially (Red phase)
            implementation_unlocked=True,
            gaming_detected=False
        )
        
    async def validate_implementation_against_tests(
        self,
        feature_id: str,
        implementation_path: str
    ) -> Dict[str, Any]:
        """
        Validate that implementation passes all generated tests.
        Called during development to ensure TDD compliance.
        """
        if feature_id not in self.active_features:
            raise ValueError(f"Feature {feature_id} not found in active features")
            
        tests = self.active_features[feature_id]['tests']
        
        # Run tests against implementation
        test_results = await self.browserbase_executor.execute_test_suite_in_cloud(
            tests,
            browsers=['chromium', 'firefox', 'webkit']
        )
        
        # Check if all tests pass (Green phase of TDD)
        all_passing = test_results.passed == test_results.total_tests
        
        # Update feature status
        self.active_features[feature_id]['status'] = \
            'tests_passing' if all_passing else 'tests_failing'
            
        return {
            'feature_id': feature_id,
            'all_tests_passing': all_passing,
            'test_results': test_results,
            'coverage': await self._calculate_coverage(implementation_path),
            'can_deploy': all_passing and test_results.failed == 0
        }
        
    async def generate_emergency_bypass_token(
        self,
        reason: str,
        approver: str,
        duration_seconds: int = 3600
    ) -> TDDBypassToken:
        """
        Generate time-limited bypass token for emergency fixes.
        Requires VP-level approval and creates audit trail.
        """
        # Validate approver has authority
        if not self._validate_approver_authority(approver):
            raise PermissionError(f"Approver {approver} lacks bypass authority")
            
        token = TDDBypassToken(
            token=self._generate_secure_token(),
            reason=reason,
            approver=approver,
            expires_at=datetime.now() + timedelta(seconds=duration_seconds),
            audit_logged=False
        )
        
        # Log to audit trail
        await self._log_bypass_to_audit(token)
        token.audit_logged = True
        
        # Send notification to security team
        await self._notify_security_team(token)
        
        return token
        
    async def _execute_initial_tests(
        self,
        tests: List[NaturalLanguageTest]
    ) -> Dict[str, Any]:
        """Execute tests to verify they fail without implementation"""
        # Create temporary test environment
        temp_browser = await self._create_temp_browser()
        page = await temp_browser.new_page()
        
        # Navigate to application under test
        await page.goto(os.getenv("APP_URL", "http://localhost:3000"))
        
        # Initialize Stagehand
        stagehand_page = await self.stagehand_engine.initialize_page(
            page, "initial_test_run"
        )
        
        results = []
        for test in tests:
            result = await self.stagehand_engine.execute_natural_language_test(
                stagehand_page, test
            )
            results.append(result)
            
        await temp_browser.close()
        
        # Check if any tests are passing (they shouldn't be)
        passing_tests = [r for r in results if r.status == 'passed']
        
        return {
            'all_passing': len(passing_tests) == len(tests),
            'some_passing': len(passing_tests) > 0,
            'results': results
        }
        
    async def _create_test_files(
        self,
        feature_id: str,
        tests: List[NaturalLanguageTest]
    ) -> None:
        """Create test files for the feature"""
        test_dir = Path(f"tests/features/{feature_id}")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Stagehand test file
        stagehand_file = test_dir / "feature.stagehand.spec.ts"
        with open(stagehand_file, 'w') as f:
            f.write(self._generate_stagehand_test_file(tests))
            
        # Create traditional Playwright test file (for comparison)
        playwright_file = test_dir / "feature.playwright.spec.ts"
        with open(playwright_file, 'w') as f:
            f.write(await self._generate_playwright_test_file(tests))
            
    def _generate_stagehand_test_file(
        self,
        tests: List[NaturalLanguageTest]
    ) -> str:
        """Generate Stagehand test file content"""
        content = []
        content.append("import { test, expect } from '@playwright/test';")
        content.append("import { Stagehand } from '@browserbasehq/stagehand';")
        content.append("")
        content.append("test.describe('TDD Generated Tests', () => {")
        
        for test in tests:
            content.append(f"  test('{test.name}', async ({{ page }}) => {{")
            content.append("    const stagehand = new Stagehand(page);")
            content.append("")
            
            # Add actions
            for action in test.actions:
                content.append(f"    await stagehand.act('{action}');")
                
            content.append("")
            
            # Add assertions
            for assertion in test.assertions:
                content.append(f"    await stagehand.observe('{assertion}');")
                
            content.append("  });")
            content.append("")
            
        content.append("});")
        
        return '\n'.join(content)
        
    def _serialize_tests(
        self,
        tests: List[NaturalLanguageTest]
    ) -> str:
        """Serialize tests to string for validation"""
        return self._generate_stagehand_test_file(tests)
        
    async def _validate_bypass_token(
        self,
        token: str
    ) -> bool:
        """Validate emergency bypass token"""
        # Check token in database
        # Verify it hasn't expired
        # Log usage
        return False  # Placeholder
        
    def _validate_approver_authority(
        self,
        approver: str
    ) -> bool:
        """Check if approver has bypass authority"""
        authorized_approvers = os.getenv(
            "TDD_BYPASS_APPROVERS",
            "vp-engineering,cto,security-lead"
        ).split(',')
        return approver in authorized_approvers
        
    def _generate_secure_token(self) -> str:
        """Generate cryptographically secure bypass token"""
        import secrets
        return f"tdd-bypass-{secrets.token_urlsafe(32)}"
        
    async def _log_bypass_to_audit(
        self,
        token: TDDBypassToken
    ) -> None:
        """Log bypass usage to audit trail"""
        self.bypass_audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'token': token.token[:10] + '...',  # Partial token for security
            'reason': token.reason,
            'approver': token.approver,
            'expires_at': token.expires_at.isoformat()
        })
        
        # Also log to file
        with open('tdd_bypass_audit.json', 'a') as f:
            json.dump(self.bypass_audit_log[-1], f)
            f.write('\n')
            
    async def _notify_security_team(
        self,
        token: TDDBypassToken
    ) -> None:
        """Send notification to security team about bypass usage"""
        # Implement notification logic (email, Slack, etc.)
        pass
```

### 2.5 Claude Code Task Tool Integration

**File**: `python/src/mcp/tdd_enforcement_tool.py`

```python
from typing import Dict, Any, List, Optional
from mcp.server import MCPServer
from mcp.types import Tool, ToolResult
import asyncio

class TDDEnforcementTool:
    """
    MCP tool for TDD enforcement integration with Claude Code.
    Provides direct access to test generation and validation from Claude.
    """
    
    def __init__(self, server: MCPServer):
        self.server = server
        self.tdd_gate = TDDEnforcementGate()
        self.active_sessions = {}
        
        # Register tools with MCP server
        self._register_tools()
        
    def _register_tools(self):
        """Register TDD enforcement tools with MCP server"""
        
        # Test generation tool
        self.server.register_tool(Tool(
            name="generate_tests_from_prd",
            description="Generate natural language tests from PRD documentation",
            parameters={
                "type": "object",
                "properties": {
                    "prd_content": {
                        "type": "string",
                        "description": "PRD/PRP/ADR documentation content"
                    },
                    "test_strategy": {
                        "type": "string",
                        "enum": ["minimal", "standard", "comprehensive"],
                        "default": "comprehensive"
                    }
                },
                "required": ["prd_content"]
            },
            handler=self.generate_tests_handler
        ))
        
        # Test validation tool
        self.server.register_tool(Tool(
            name="validate_tests",
            description="Validate tests for gaming patterns using DGTS",
            parameters={
                "type": "object",
                "properties": {
                    "test_content": {
                        "type": "string",
                        "description": "Test file content to validate"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to test file"
                    }
                },
                "required": ["test_content"]
            },
            handler=self.validate_tests_handler
        ))
        
        # Test execution tool
        self.server.register_tool(Tool(
            name="execute_tests_cloud",
            description="Execute tests in Browserbase cloud infrastructure",
            parameters={
                "type": "object",
                "properties": {
                    "test_suite_path": {
                        "type": "string",
                        "description": "Path to test suite"
                    },
                    "browsers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": ["chromium", "firefox", "webkit"]
                    },
                    "parallel": {
                        "type": "boolean",
                        "default": True
                    }
                },
                "required": ["test_suite_path"]
            },
            handler=self.execute_tests_handler
        ))
        
        # TDD enforcement check tool
        self.server.register_tool(Tool(
            name="check_tdd_compliance",
            description="Check if feature development follows TDD principles",
            parameters={
                "type": "object",
                "properties": {
                    "feature_id": {
                        "type": "string",
                        "description": "Feature identifier"
                    },
                    "implementation_path": {
                        "type": "string",
                        "description": "Path to implementation code"
                    }
                },
                "required": ["feature_id"]
            },
            handler=self.check_compliance_handler
        ))
        
    async def generate_tests_handler(
        self,
        prd_content: str,
        test_strategy: str = "comprehensive"
    ) -> ToolResult:
        """Handle test generation from PRD"""
        try:
            tests = await self.tdd_gate.stagehand_engine.generate_tests_from_prd(
                prd_content,
                test_strategy
            )
            
            # Store in session
            session_id = str(uuid.uuid4())
            self.active_sessions[session_id] = {
                'tests': tests,
                'created_at': datetime.now()
            }
            
            return ToolResult(
                success=True,
                result={
                    'session_id': session_id,
                    'tests_generated': len(tests),
                    'test_names': [t.name for t in tests],
                    'natural_language_preview': tests[0].actions[:2] if tests else []
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
            
    async def validate_tests_handler(
        self,
        test_content: str,
        file_path: Optional[str] = None
    ) -> ToolResult:
        """Handle test validation for gaming patterns"""
        try:
            validation_result = await self.tdd_gate.dgts_validator.validate_test_file(
                file_path or "temp_test.spec.ts",
                test_content
            )
            
            return ToolResult(
                success=True,
                result={
                    'is_valid': validation_result.is_valid,
                    'is_gaming': validation_result.is_gaming,
                    'gaming_score': validation_result.gaming_score,
                    'violations': [
                        {
                            'type': v.violation_type,
                            'severity': v.severity,
                            'line': v.line_number,
                            'explanation': v.explanation
                        }
                        for v in validation_result.violations
                    ],
                    'recommendations': validation_result.recommendations
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
            
    async def execute_tests_handler(
        self,
        test_suite_path: str,
        browsers: List[str] = None,
        parallel: bool = True
    ) -> ToolResult:
        """Handle test execution in cloud"""
        try:
            # Load tests from path
            tests = await self._load_tests_from_path(test_suite_path)
            
            # Execute in cloud
            results = await self.tdd_gate.browserbase_executor.execute_test_suite_in_cloud(
                tests,
                browsers=browsers or ['chromium', 'firefox', 'webkit']
            )
            
            return ToolResult(
                success=True,
                result={
                    'total_tests': results.total_tests,
                    'passed': results.passed,
                    'failed': results.failed,
                    'duration_ms': results.duration_ms,
                    'browser_results': results.browser_results,
                    'video_recordings': results.video_recordings,
                    'all_passing': results.failed == 0
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
            
    async def check_compliance_handler(
        self,
        feature_id: str,
        implementation_path: Optional[str] = None
    ) -> ToolResult:
        """Handle TDD compliance checking"""
        try:
            if implementation_path:
                # Validate implementation against tests
                validation = await self.tdd_gate.validate_implementation_against_tests(
                    feature_id,
                    implementation_path
                )
                
                return ToolResult(
                    success=True,
                    result={
                        'tdd_compliant': validation['all_tests_passing'],
                        'test_results': validation['test_results'],
                        'coverage': validation['coverage'],
                        'can_deploy': validation['can_deploy']
                    }
                )
            else:
                # Check if feature has tests
                status = self.tdd_gate.active_features.get(feature_id)
                
                return ToolResult(
                    success=True,
                    result={
                        'has_tests': status is not None,
                        'status': status.get('status') if status else 'no_tests',
                        'tests_count': len(status.get('tests', [])) if status else 0
                    }
                )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
```

### 2.6 WebSocket Real-Time Communication

**File**: `python/src/server/tdd_websocket_handler.py`

```python
import asyncio
import json
from typing import Dict, Any, Optional
from aiohttp import web
import aiohttp_cors

class TDDWebSocketHandler:
    """
    WebSocket handler for real-time TDD test results streaming.
    Provides live updates to Claude Code during test execution.
    """
    
    def __init__(self):
        self.active_connections = {}
        self.test_sessions = {}
        self.browserbase_executor = BrowserbaseCloudExecutor()
        
    async def handle_websocket(self, request):
        """Main WebSocket connection handler"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        session_id = str(uuid.uuid4())
        self.active_connections[session_id] = ws
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self.handle_message(session_id, data, ws)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f'WebSocket error: {ws.exception()}')
                    
        finally:
            del self.active_connections[session_id]
            
        return ws
        
    async def handle_message(
        self,
        session_id: str,
        data: Dict[str, Any],
        ws: web.WebSocketResponse
    ):
        """Handle incoming WebSocket messages"""
        message_type = data.get('type')
        
        if message_type == 'start_test_execution':
            await self.start_test_execution(session_id, data, ws)
        elif message_type == 'get_test_status':
            await self.get_test_status(session_id, data, ws)
        elif message_type == 'cancel_test_execution':
            await self.cancel_test_execution(session_id, data, ws)
        elif message_type == 'subscribe_to_progress':
            await self.subscribe_to_progress(session_id, data, ws)
            
    async def start_test_execution(
        self,
        session_id: str,
        data: Dict[str, Any],
        ws: web.WebSocketResponse
    ):
        """Start test execution and stream results"""
        test_suite_path = data.get('test_suite_path')
        browsers = data.get('browsers', ['chromium'])
        
        # Create cloud session
        cloud_session = await self.browserbase_executor.create_test_session()
        self.test_sessions[session_id] = cloud_session
        
        # Start streaming progress
        asyncio.create_task(
            self.stream_test_progress(session_id, cloud_session.session_id, ws)
        )
        
        # Send initial response
        await ws.send_json({
            'type': 'test_execution_started',
            'session_id': session_id,
            'cloud_session_id': cloud_session.session_id,
            'browsers': browsers
        })
        
    async def stream_test_progress(
        self,
        session_id: str,
        cloud_session_id: str,
        ws: web.WebSocketResponse
    ):
        """Stream real-time test progress"""
        async for update in self.browserbase_executor.stream_test_progress(cloud_session_id):
            if session_id not in self.active_connections:
                break
                
            await ws.send_json({
                'type': 'test_progress_update',
                'session_id': session_id,
                'progress': update['progress'],
                'current_test': update['current_test'],
                'status': update['status'],
                'logs': update['logs'][-10:]  # Last 10 log lines
            })
            
            if update['status'] in ['completed', 'failed']:
                # Send final results
                await ws.send_json({
                    'type': 'test_execution_complete',
                    'session_id': session_id,
                    'status': update['status'],
                    'results': await self.get_final_results(cloud_session_id)
                })
                break
                
    async def get_final_results(
        self,
        cloud_session_id: str
    ) -> Dict[str, Any]:
        """Get final test results from cloud session"""
        # Fetch complete results from Browserbase
        session = await self.browserbase_executor.client.get_session(cloud_session_id)
        
        return {
            'total_tests': session.test_count,
            'passed': session.passed_count,
            'failed': session.failed_count,
            'duration_ms': session.duration,
            'video_url': session.video_url,
            'logs_url': session.logs_url,
            'artifacts': session.artifacts
        }
```

## 3. Architecture Diagrams

### 3.1 System Architecture

```mermaid
graph TB
    subgraph "Claude Code Environment"
        CC[Claude Code IDE]
        CT[Task Tool]
        MCP[MCP Server]
    end
    
    subgraph "Phase 9 TDD System"
        TG[Test Generator<br/>Stagehand AI]
        TV[Test Validator<br/>Enhanced DGTS]
        TE[Test Executor<br/>Browserbase]
        TDD[TDD Enforcement Gate]
    end
    
    subgraph "Archon Core"
        ORCH[Orchestrator]
        AGENTS[Agent System]
        KB[Knowledge Base]
        VAL[Validators]
    end
    
    subgraph "Cloud Infrastructure"
        BB[Browserbase Cloud]
        BROWSERS[Chrome/Firefox/Safari]
        VIDEO[Video Recording]
    end
    
    CC --> CT
    CT --> MCP
    MCP --> TDD
    
    TDD --> TG
    TDD --> TV
    TDD --> TE
    
    TG --> KB
    TV --> VAL
    TE --> BB
    
    BB --> BROWSERS
    BB --> VIDEO
    
    TDD --> ORCH
    ORCH --> AGENTS
    
    style TDD fill:#f9f,stroke:#333,stroke-width:4px
    style BB fill:#9ff,stroke:#333,stroke-width:2px
```

### 3.2 TDD Workflow

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant CC as Claude Code
    participant TDD as TDD Gate
    participant SH as Stagehand
    participant DGTS as DGTS Validator
    participant BB as Browserbase
    
    Dev->>CC: Request feature implementation
    CC->>TDD: Check TDD compliance
    TDD->>SH: Generate tests from PRD
    SH-->>TDD: Natural language tests
    
    TDD->>DGTS: Validate test authenticity
    DGTS-->>TDD: Gaming check result
    
    alt Tests are gamed
        TDD-->>CC: BLOCKED: Test gaming detected
        CC-->>Dev: Fix gaming patterns
    else Tests are valid
        TDD->>BB: Execute initial tests
        BB-->>TDD: Tests failing (Red phase)
        
        TDD-->>CC: Development unlocked
        CC-->>Dev: Implement feature
        
        loop During development
            Dev->>CC: Write code
            CC->>BB: Run tests
            BB-->>CC: Test results
            CC-->>Dev: Feedback
        end
        
        Dev->>CC: Implementation complete
        CC->>BB: Final test run
        BB-->>CC: All tests passing (Green)
        CC-->>Dev: Ready to deploy
    end
```

### 3.3 Data Flow

```mermaid
graph LR
    subgraph "Input"
        PRD[PRD/PRP/ADR]
        CODE[Source Code]
    end
    
    subgraph "Processing"
        PARSE[Requirement Parser]
        GEN[Test Generator]
        VAL[Validator]
        EXEC[Executor]
    end
    
    subgraph "Output"
        TESTS[Test Suite]
        RESULTS[Test Results]
        COVERAGE[Coverage Report]
        VIDEOS[Test Videos]
    end
    
    PRD --> PARSE
    PARSE --> GEN
    GEN --> TESTS
    
    TESTS --> VAL
    VAL --> EXEC
    
    CODE --> EXEC
    EXEC --> RESULTS
    EXEC --> COVERAGE
    EXEC --> VIDEOS
    
    style GEN fill:#ffd,stroke:#333,stroke-width:2px
    style VAL fill:#dff,stroke:#333,stroke-width:2px
    style EXEC fill:#fdf,stroke:#333,stroke-width:2px
```

## 4. Database Schema

### 4.1 Test Management Schema

```sql
-- Test suites generated from PRDs
CREATE TABLE test_suites (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_id VARCHAR(255) NOT NULL,
    prd_hash VARCHAR(64) NOT NULL,  -- Track PRD changes
    test_count INTEGER NOT NULL,
    natural_language_tests JSONB NOT NULL,
    gaming_score FLOAT DEFAULT 0.0,
    validation_status VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Test execution history
CREATE TABLE test_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    suite_id UUID REFERENCES test_suites(id),
    session_id VARCHAR(255) NOT NULL,  -- Browserbase session
    browser_type VARCHAR(50),
    region VARCHAR(50),
    status VARCHAR(50) NOT NULL,  -- running, passed, failed
    total_tests INTEGER,
    passed_tests INTEGER,
    failed_tests INTEGER,
    duration_ms INTEGER,
    video_url TEXT,
    logs_url TEXT,
    executed_at TIMESTAMP DEFAULT NOW()
);

-- Gaming violations tracking
CREATE TABLE gaming_violations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    suite_id UUID REFERENCES test_suites(id),
    violation_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    file_path TEXT,
    line_number INTEGER,
    code_snippet TEXT,
    explanation TEXT,
    remediation TEXT,
    detected_at TIMESTAMP DEFAULT NOW()
);

-- TDD bypass audit log
CREATE TABLE tdd_bypass_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    token_hash VARCHAR(64) NOT NULL,  -- Hashed token
    reason TEXT NOT NULL,
    approver VARCHAR(255) NOT NULL,
    feature_id VARCHAR(255),
    expires_at TIMESTAMP NOT NULL,
    used_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Feature development tracking
CREATE TABLE feature_development (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_id VARCHAR(255) UNIQUE NOT NULL,
    prd_content TEXT,
    tests_generated BOOLEAN DEFAULT FALSE,
    tests_validated BOOLEAN DEFAULT FALSE,
    tests_passing BOOLEAN DEFAULT FALSE,
    implementation_started_at TIMESTAMP,
    implementation_completed_at TIMESTAMP,
    coverage_percentage FLOAT,
    status VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_test_suites_feature ON test_suites(feature_id);
CREATE INDEX idx_test_executions_suite ON test_executions(suite_id);
CREATE INDEX idx_test_executions_session ON test_executions(session_id);
CREATE INDEX idx_gaming_violations_suite ON gaming_violations(suite_id);
CREATE INDEX idx_feature_development_status ON feature_development(status);
```

## 5. API Endpoints

### 5.1 REST API Specification

```python
# File: python/src/api/tdd_endpoints.py

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="TDD Enforcement API", version="9.0")

class TestGenerationRequest(BaseModel):
    prd_content: str
    test_strategy: str = "comprehensive"
    feature_id: Optional[str] = None
    
class TestValidationRequest(BaseModel):
    test_content: str
    file_path: Optional[str] = None
    
class TestExecutionRequest(BaseModel):
    suite_id: str
    browsers: List[str] = ["chromium", "firefox", "webkit"]
    parallel: bool = True
    
class BypassTokenRequest(BaseModel):
    reason: str
    approver: str
    duration_seconds: int = 3600
    ticket_id: Optional[str] = None

@app.post("/api/v1/tdd/generate-tests")
async def generate_tests(request: TestGenerationRequest):
    """Generate natural language tests from PRD"""
    tdd_gate = TDDEnforcementGate()
    tests = await tdd_gate.stagehand_engine.generate_tests_from_prd(
        request.prd_content,
        request.test_strategy
    )
    return {
        "feature_id": request.feature_id or str(uuid.uuid4()),
        "tests_generated": len(tests),
        "tests": [t.dict() for t in tests]
    }
    
@app.post("/api/v1/tdd/validate-tests")
async def validate_tests(request: TestValidationRequest):
    """Validate tests for gaming patterns"""
    validator = EnhancedDGTSValidator()
    result = await validator.validate_test_file(
        request.file_path or "temp.spec.ts",
        request.test_content
    )
    return {
        "is_valid": result.is_valid,
        "is_gaming": result.is_gaming,
        "gaming_score": result.gaming_score,
        "violations": result.violations,
        "recommendations": result.recommendations
    }
    
@app.post("/api/v1/tdd/execute-tests")
async def execute_tests(request: TestExecutionRequest):
    """Execute tests in Browserbase cloud"""
    executor = BrowserbaseCloudExecutor()
    # Load test suite from database
    suite = await load_test_suite(request.suite_id)
    
    results = await executor.execute_test_suite_in_cloud(
        suite.tests,
        browsers=request.browsers
    )
    
    return {
        "execution_id": str(uuid.uuid4()),
        "results": results.dict()
    }
    
@app.post("/api/v1/tdd/generate-bypass-token")
async def generate_bypass_token(request: BypassTokenRequest):
    """Generate emergency bypass token (requires authorization)"""
    tdd_gate = TDDEnforcementGate()
    
    # Verify authorization (implement your auth logic)
    # if not authorized:
    #     raise HTTPException(status_code=403, detail="Unauthorized")
    
    token = await tdd_gate.generate_emergency_bypass_token(
        request.reason,
        request.approver,
        request.duration_seconds
    )
    
    return {
        "token": token.token,
        "expires_at": token.expires_at.isoformat(),
        "audit_logged": token.audit_logged
    }
    
@app.get("/api/v1/tdd/feature/{feature_id}/status")
async def get_feature_status(feature_id: str):
    """Get TDD compliance status for a feature"""
    tdd_gate = TDDEnforcementGate()
    
    if feature_id not in tdd_gate.active_features:
        raise HTTPException(status_code=404, detail="Feature not found")
        
    feature = tdd_gate.active_features[feature_id]
    
    return {
        "feature_id": feature_id,
        "status": feature['status'],
        "tests_count": len(feature.get('tests', [])),
        "created_at": feature['created_at'].isoformat()
    }
    
@app.ws("/ws/tdd/test-execution")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time test execution updates"""
    handler = TDDWebSocketHandler()
    await handler.handle_websocket(websocket)
```

## 6. Configuration Files

### 6.1 Docker Configuration

```dockerfile
# File: docker/tdd-enforcement/Dockerfile

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for Stagehand
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Stagehand
RUN npm install -g @browserbasehq/stagehand playwright

# Copy application code
COPY python/src /app/src
COPY config /app/config

# Environment variables
ENV PYTHONPATH=/app
ENV TDD_ENFORCEMENT_LEVEL=strict
ENV STAGEHAND_MODEL=gpt-4-turbo

# Expose ports
EXPOSE 8054 8055

# Run the service
CMD ["python", "-m", "src.services.tdd_enforcement_server"]
```

### 6.2 Docker Compose

```yaml
# File: docker-compose.tdd.yml

version: '3.8'

services:
  tdd-enforcement:
    build:
      context: .
      dockerfile: docker/tdd-enforcement/Dockerfile
    container_name: archon-tdd-enforcement
    ports:
      - "8054:8054"  # REST API
      - "8055:8055"  # WebSocket
    environment:
      - TDD_ENFORCEMENT_LEVEL=strict
      - STAGEHAND_API_KEY=${STAGEHAND_API_KEY}
      - BROWSERBASE_API_KEY=${BROWSERBASE_API_KEY}
      - BROWSERBASE_PROJECT_ID=${BROWSERBASE_PROJECT_ID}
      - DATABASE_URL=postgresql://archon:password@postgres:5432/archon
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    volumes:
      - ./tests:/app/tests
      - ./logs:/app/logs
    networks:
      - archon-network
      
  postgres:
    image: postgres:15
    container_name: archon-postgres
    environment:
      - POSTGRES_DB=archon
      - POSTGRES_USER=archon
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./sql/schema.sql:/docker-entrypoint-initdb.d/01-schema.sql
    ports:
      - "5432:5432"
    networks:
      - archon-network
      
  redis:
    image: redis:7-alpine
    container_name: archon-redis
    ports:
      - "6379:6379"
    networks:
      - archon-network

volumes:
  postgres-data:

networks:
  archon-network:
    driver: bridge
```

### 6.3 Environment Configuration

```bash
# File: .env.tdd

# Stagehand Configuration
STAGEHAND_API_KEY=your-stagehand-api-key-here
STAGEHAND_MODEL=gpt-4-turbo
STAGEHAND_MODE=strict
STAGEHAND_DEBUG=false

# Browserbase Cloud Configuration
BROWSERBASE_API_KEY=bb_your_api_key_here
BROWSERBASE_PROJECT_ID=proj_your_project_id
BROWSERBASE_TIMEOUT=120000
BROWSERBASE_PARALLEL_SESSIONS=5
BROWSERBASE_REGIONS=us-west,eu-central,asia-pacific

# TDD Enforcement Settings
TDD_ENFORCEMENT_LEVEL=strict  # strict|moderate|advisory
TDD_BYPASS_APPROVERS=vp-engineering,cto,security-lead
TDD_BYPASS_MAX_DURATION=7200  # 2 hours max
TEST_FIRST_REQUIRED=true
MIN_TEST_COVERAGE=95

# DGTS Integration
DGTS_TEST_VALIDATION=enabled
GAMING_DETECTION_THRESHOLD=0.3
BLOCK_GAMED_TESTS=true
GAMING_PATTERN_UPDATE_URL=https://archon-updates.com/dgts-patterns.json

# Database Configuration
DATABASE_URL=postgresql://archon:password@localhost:5432/archon
REDIS_URL=redis://localhost:6379

# Application Settings
APP_URL=http://localhost:3000
API_PORT=8054
WEBSOCKET_PORT=8055
LOG_LEVEL=INFO

# Monitoring & Alerts
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your-webhook
AUDIT_EMAIL=security@company.com
```

## 7. Testing Strategy

### 7.1 Unit Tests for TDD Components

```python
# File: python/tests/test_tdd_enforcement.py

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.services.tdd_enforcement import TDDEnforcementGate
from src.services.stagehand_service import StagehandTestEngine

@pytest.mark.asyncio
async def test_generate_tests_from_prd():
    """Test that PRD generates appropriate tests"""
    engine = StagehandTestEngine()
    
    prd_content = """
    Feature: User Authentication
    Requirements:
    - Users can log in with email and password
    - Invalid credentials show error message
    - Successful login redirects to dashboard
    """
    
    tests = await engine.generate_tests_from_prd(prd_content)
    
    assert len(tests) >= 3
    assert any("log in" in t.name.lower() for t in tests)
    assert any("error" in t.name.lower() for t in tests)
    assert any("dashboard" in t.name.lower() for t in tests)
    
@pytest.mark.asyncio
async def test_dgts_detects_gaming():
    """Test that DGTS correctly identifies test gaming"""
    validator = EnhancedDGTSValidator()
    
    gamed_test = """
    test('user login', () => {
        expect(true).toBe(true);  // Gaming pattern
    });
    """
    
    result = await validator.validate_test_file("test.spec.ts", gamed_test)
    
    assert result.is_gaming is True
    assert result.gaming_score > 0.3
    assert len(result.violations) > 0
    assert any(v.violation_type == "DGTS_ALWAYS_TRUE" for v in result.violations)
    
@pytest.mark.asyncio
async def test_tdd_enforcement_blocks_without_tests():
    """Test that development is blocked without tests"""
    gate = TDDEnforcementGate()
    
    with pytest.raises(ValueError, match="PRD content required"):
        await gate.enforce_test_first_development({
            'id': 'feature-123',
            'prd_content': ''  # Empty PRD
        })
        
@pytest.mark.asyncio
async def test_bypass_token_requires_authorization():
    """Test that bypass tokens require proper authorization"""
    gate = TDDEnforcementGate()
    
    with pytest.raises(PermissionError):
        await gate.generate_emergency_bypass_token(
            reason="Emergency fix",
            approver="unauthorized-user",
            duration_seconds=3600
        )
        
@pytest.mark.asyncio
async def test_stagehand_conversion_reduces_code():
    """Test that Stagehand conversion reduces code by 70%"""
    engine = StagehandTestEngine()
    
    playwright_code = """
    await page.locator('[data-testid="email-input"]').fill('user@example.com');
    await page.locator('[data-testid="password-input"]').fill('password123');
    await page.locator('button:has-text("Sign In")').click();
    await expect(page.locator('.dashboard-title')).toBeVisible();
    """
    
    stagehand_code = await engine.convert_playwright_to_stagehand("test.spec.ts")
    
    # Stagehand should be significantly shorter
    assert len(stagehand_code) < len(playwright_code) * 0.4
```

### 7.2 Integration Tests

```python
# File: python/tests/test_integration_tdd.py

@pytest.mark.integration
async def test_full_tdd_workflow():
    """Test complete TDD workflow from PRD to deployment"""
    gate = TDDEnforcementGate()
    
    # Step 1: Generate tests from PRD
    prd_content = load_fixture("sample_prd.md")
    enforcement_status = await gate.enforce_test_first_development({
        'id': 'feature-test-123',
        'prd_content': prd_content
    })
    
    assert enforcement_status.tests_generated is True
    assert enforcement_status.tests_validated is True
    assert enforcement_status.implementation_unlocked is True
    
    # Step 2: Simulate implementation
    implementation_path = "src/features/test_feature.ts"
    create_test_implementation(implementation_path)
    
    # Step 3: Validate implementation against tests
    validation = await gate.validate_implementation_against_tests(
        'feature-test-123',
        implementation_path
    )
    
    assert validation['all_tests_passing'] is True
    assert validation['coverage'] > 95.0
    assert validation['can_deploy'] is True
```

## 8. Deployment Instructions

### 8.1 Local Development Setup

```bash
#!/bin/bash
# File: scripts/setup-tdd-local.sh

echo "Setting up Phase 9 TDD Enforcement System..."

# 1. Install dependencies
pip install -r requirements-tdd.txt
npm install -g @browserbasehq/stagehand playwright

# 2. Set up environment
cp .env.tdd.example .env.tdd
echo "Please edit .env.tdd with your API keys"

# 3. Initialize database
python scripts/init_database.py --schema tdd

# 4. Start services
docker-compose -f docker-compose.tdd.yml up -d

# 5. Run migrations
python scripts/migrate_database.py --phase 9

# 6. Verify installation
python scripts/verify_tdd_setup.py

echo "TDD Enforcement System ready!"
echo "API: http://localhost:8054"
echo "WebSocket: ws://localhost:8055"
```

### 8.2 Production Deployment

```yaml
# File: kubernetes/tdd-enforcement.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: archon-tdd-enforcement
  namespace: archon
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tdd-enforcement
  template:
    metadata:
      labels:
        app: tdd-enforcement
    spec:
      containers:
      - name: tdd-enforcement
        image: archon/tdd-enforcement:9.0
        ports:
        - containerPort: 8054
        - containerPort: 8055
        env:
        - name: TDD_ENFORCEMENT_LEVEL
          value: "strict"
        - name: STAGEHAND_API_KEY
          valueFrom:
            secretKeyRef:
              name: archon-secrets
              key: stagehand-api-key
        - name: BROWSERBASE_API_KEY
          valueFrom:
            secretKeyRef:
              name: archon-secrets
              key: browserbase-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: tdd-enforcement-service
  namespace: archon
spec:
  selector:
    app: tdd-enforcement
  ports:
  - name: api
    port: 8054
    targetPort: 8054
  - name: websocket
    port: 8055
    targetPort: 8055
  type: LoadBalancer
```

## 9. Migration from Phase 8

### 9.1 Migration Script

```python
# File: scripts/migrate_to_phase9.py

import asyncio
from pathlib import Path

async def migrate_to_phase9():
    """Migrate existing Archon installation to Phase 9"""
    
    print("Starting Phase 9 migration...")
    
    # 1. Backup existing data
    print("Backing up existing configuration...")
    backup_existing_config()
    
    # 2. Update agent configurations
    print("Updating agent configurations for TDD...")
    update_agent_configs()
    
    # 3. Install new dependencies
    print("Installing TDD dependencies...")
    install_tdd_dependencies()
    
    # 4. Create database tables
    print("Creating TDD database schema...")
    create_tdd_tables()
    
    # 5. Configure MCP tools
    print("Registering TDD tools with MCP...")
    register_mcp_tools()
    
    # 6. Update validation rules
    print("Updating DGTS with test gaming patterns...")
    update_dgts_patterns()
    
    # 7. Test the installation
    print("Running verification tests...")
    verify_installation()
    
    print("Phase 9 migration complete!")
    
if __name__ == "__main__":
    asyncio.run(migrate_to_phase9())
```

## 10. Monitoring and Metrics

### 10.1 Prometheus Metrics

```python
# File: python/src/monitoring/tdd_metrics.py

from prometheus_client import Counter, Histogram, Gauge

# Define metrics
tests_generated = Counter(
    'tdd_tests_generated_total',
    'Total number of tests generated from PRDs'
)

gaming_violations_detected = Counter(
    'tdd_gaming_violations_total',
    'Total number of test gaming violations detected',
    ['violation_type', 'severity']
)

test_execution_duration = Histogram(
    'tdd_test_execution_duration_seconds',
    'Test execution duration in seconds',
    ['browser', 'status']
)

tdd_compliance_rate = Gauge(
    'tdd_compliance_rate',
    'Current TDD compliance rate (0-100)'
)

bypass_tokens_generated = Counter(
    'tdd_bypass_tokens_total',
    'Total number of emergency bypass tokens generated'
)

cloud_test_sessions = Gauge(
    'tdd_cloud_sessions_active',
    'Number of active Browserbase test sessions'
)
```

## 11. Success Validation

### 11.1 Phase 9 Validation Tests

```python
# File: python/tests/test_phase9_validation.py

async def test_phase9_complete():
    """Validate Phase 9 implementation is complete"""
    
    # Test 1: Stagehand integration works
    engine = StagehandTestEngine()
    assert await engine.health_check()
    
    # Test 2: Browserbase connection established
    executor = BrowserbaseCloudExecutor()
    assert await executor.test_connection()
    
    # Test 3: DGTS enhanced with test patterns
    validator = EnhancedDGTSValidator()
    assert len(validator.CRITICAL_GAMING_PATTERNS) >= 8
    assert len(validator.STAGEHAND_GAMING_PATTERNS) >= 4
    
    # Test 4: TDD enforcement gate operational
    gate = TDDEnforcementGate()
    assert gate.enforcement_level == "strict"
    
    # Test 5: MCP tools registered
    mcp_server = get_mcp_server()
    tools = mcp_server.list_tools()
    assert "generate_tests_from_prd" in tools
    assert "validate_tests" in tools
    assert "execute_tests_cloud" in tools
    
    # Test 6: WebSocket handler ready
    ws_handler = TDDWebSocketHandler()
    assert await ws_handler.health_check()
    
    # Test 7: Database schema created
    db = get_database_connection()
    tables = await db.list_tables()
    assert "test_suites" in tables
    assert "test_executions" in tables
    
    print("✅ Phase 9 validation complete!")
    return True
```

---

## Appendix A: Quick Reference

### Commands

```bash
# Generate tests from PRD
archon-tdd generate --prd docs/feature.md

# Validate tests for gaming
archon-tdd validate --file tests/feature.spec.ts

# Execute tests in cloud
archon-tdd execute --suite tests/ --browsers all

# Check TDD compliance
archon-tdd check --feature feature-123

# Generate emergency bypass
archon-tdd bypass --reason "Critical fix" --duration 3600
```

### API Endpoints

- `POST /api/v1/tdd/generate-tests` - Generate tests from PRD
- `POST /api/v1/tdd/validate-tests` - Validate for gaming
- `POST /api/v1/tdd/execute-tests` - Run in cloud
- `GET /api/v1/tdd/feature/{id}/status` - Check compliance
- `WS /ws/tdd/test-execution` - Real-time updates

### Environment Variables

```bash
STAGEHAND_API_KEY       # Required: Stagehand AI key
BROWSERBASE_API_KEY     # Required: Browserbase cloud key
TDD_ENFORCEMENT_LEVEL   # strict|moderate|advisory
MIN_TEST_COVERAGE       # Default: 95
GAMING_DETECTION_THRESHOLD  # Default: 0.3
```

---

**Status**: Ready for Implementation  
**Estimated Timeline**: 8 weeks  
**Dependencies**: Phase 1-8 complete, API keys configured  
**ROI**: 90% error reduction, 75% debugging time saved  

*End of Phase 9 TDD Enforcement Implementation PRP*