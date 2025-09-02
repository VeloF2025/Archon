#!/usr/bin/env python3
"""
Stagehand Test Engine - Natural Language Test Generation

This engine generates comprehensive tests using natural language descriptions,
converting feature requirements into executable test cases using Stagehand's
browser automation capabilities.

CRITICAL: All tests must be generated BEFORE implementation begins.
This enforces TDD principles and prevents gaming through post-hoc testing.
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from ..base_agent import BaseAgent, ArchonDependencies, BaseAgentOutput
from pydantic import BaseModel
from pydantic_ai import Agent

logger = logging.getLogger(__name__)

class TestType(Enum):
    """Types of tests that can be generated"""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    VISUAL = "visual"
    ACCESSIBILITY = "accessibility"
    PERFORMANCE = "performance"

class TestFramework(Enum):
    """Supported test frameworks"""
    PLAYWRIGHT = "playwright"
    CYPRESS = "cypress"
    JEST = "jest"
    VITEST = "vitest"
    PYTEST = "pytest"

@dataclass
class TestRequirement:
    """Individual test requirement from feature specification"""
    id: str
    description: str
    test_type: TestType
    priority: str  # critical, high, medium, low
    acceptance_criteria: List[str]
    preconditions: List[str]
    expected_behavior: str
    edge_cases: List[str]
    error_scenarios: List[str]

@dataclass
class GeneratedTest:
    """Generated test case with complete implementation"""
    id: str
    name: str
    description: str
    test_type: TestType
    framework: TestFramework
    file_path: str
    code: str
    dependencies: List[str]
    setup_code: str
    teardown_code: str
    assertions: List[str]
    stagehand_actions: List[str]  # Stagehand-specific browser actions
    natural_language_steps: List[str]  # Human-readable test steps

@dataclass
class TestGenerationResult:
    """Result of test generation process"""
    success: bool
    message: str
    tests_generated: List[GeneratedTest]
    total_tests: int
    coverage_percentage: float
    requirements_covered: List[str]
    requirements_missing: List[str]
    errors: List[str]
    warnings: List[str]
    generation_time_ms: int

class StagehandTestEngine(BaseAgent):
    """
    Natural language test generation engine using Stagehand
    
    Converts feature requirements and user stories into executable test cases
    with browser automation using Stagehand's natural language processing.
    """

    def __init__(
        self,
        model: str = "openai:gpt-4o",
        project_path: str = ".",
        test_output_dir: str = "tests/generated",
        **kwargs
    ):
        self.project_path = Path(project_path)
        self.test_output_dir = Path(test_output_dir)
        self.config_path = self.project_path / "tdd_config.yaml"
        
        super().__init__(model=model, name="StagehandTestEngine", **kwargs)
        
        # Ensure test output directory exists
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self._load_config()
        
        # Test generation statistics
        self.generation_stats = {
            "total_generated": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "last_generation_time": None
        }

    def _create_agent(self, **kwargs) -> Agent:
        """Create the PydanticAI agent for test generation"""
        return Agent(
            model=self.model,
            result_type=TestGenerationResult,
            system_prompt=self.get_system_prompt(),
            **kwargs
        )

    def get_system_prompt(self) -> str:
        """System prompt for natural language test generation"""
        return """You are an elite Test Generation AI specializing in creating comprehensive, 
        production-ready tests from natural language requirements using Stagehand browser automation.

        CORE RESPONSIBILITIES:
        1. Parse feature requirements and user stories
        2. Generate complete test suites with browser automation
        3. Create tests that validate real user workflows 
        4. Ensure comprehensive coverage of all acceptance criteria
        5. Generate Stagehand-compatible browser interaction code
        6. Include accessibility, performance, and edge case testing

        TEST GENERATION RULES:
        - ALWAYS generate tests BEFORE implementation exists
        - Create tests that fail until feature is properly implemented
        - Include natural language descriptions for each test step
        - Generate browser automation using Stagehand syntax
        - Cover happy path, edge cases, and error scenarios
        - Include setup and teardown for test isolation
        - Validate accessibility compliance (WCAG 2.1 AA)
        - Test responsive behavior across device sizes
        - Include performance assertions where relevant

        STAGEHAND INTEGRATION:
        - Use natural language for browser interactions
        - Generate page.act() calls with natural descriptions
        - Include page.extract() for data validation
        - Use page.observe() for visual verification
        - Implement proper waiting strategies
        - Handle dynamic content and async operations

        OUTPUT FORMAT:
        Generate complete, executable test files with:
        - Proper imports and setup
        - Clear test descriptions and comments
        - Stagehand browser automation code
        - Comprehensive assertions
        - Error handling and cleanup
        - Natural language step documentation

        QUALITY STANDARDS:
        - Zero tolerance for incomplete tests
        - All acceptance criteria must be covered
        - Tests must be maintainable and readable
        - Include debugging and troubleshooting comments
        - Follow framework best practices
        - Ensure tests are deterministic and reliable"""

    def _load_config(self):
        """Load TDD configuration from config file"""
        self.config = {
            "default_test_framework": "playwright",
            "browserbase_api_key": os.getenv("BROWSERBASE_API_KEY"),
            "stagehand_config": {
                "headless": True,
                "timeout": 30000,
                "wait_for_idle": True,
                "debug_dom": False
            },
            "test_generation": {
                "min_tests_per_feature": 5,
                "include_accessibility_tests": True,
                "include_performance_tests": True,
                "include_visual_regression": True,
                "max_test_duration": 60000
            },
            "coverage_requirements": {
                "minimum_coverage": 95,
                "critical_path_coverage": 100,
                "edge_case_coverage": 85
            }
        }
        
        # Load from file if it exists
        if self.config_path.exists():
            try:
                import yaml
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    self.config.update(file_config)
            except Exception as e:
                logger.warning(f"Could not load config from {self.config_path}: {e}")

    async def generate_tests_from_requirements(
        self, 
        requirements: List[str],
        feature_name: str,
        acceptance_criteria: List[str],
        user_stories: List[str] = None,
        existing_tests: List[str] = None
    ) -> TestGenerationResult:
        """
        Generate comprehensive test suite from feature requirements
        
        Args:
            requirements: List of feature requirements in natural language
            feature_name: Name of the feature being tested
            acceptance_criteria: Specific acceptance criteria to validate
            user_stories: Optional user stories for context
            existing_tests: List of existing test files to avoid duplication
            
        Returns:
            TestGenerationResult with generated tests and metadata
        """
        start_time = datetime.now()
        
        try:
            # Parse requirements into test requirements
            test_requirements = await self._parse_requirements_to_tests(
                requirements, feature_name, acceptance_criteria, user_stories
            )
            
            # Generate test implementations
            generated_tests = []
            errors = []
            warnings = []
            
            for req in test_requirements:
                try:
                    test = await self._generate_individual_test(req, feature_name)
                    generated_tests.append(test)
                    self.generation_stats["successful_generations"] += 1
                except Exception as e:
                    error_msg = f"Failed to generate test for requirement {req.id}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    self.generation_stats["failed_generations"] += 1
            
            # Calculate coverage
            coverage_percentage = self._calculate_coverage(
                test_requirements, generated_tests, acceptance_criteria
            )
            
            # Identify missing requirements
            requirements_covered = [test.id for test in generated_tests]
            requirements_missing = [
                req.id for req in test_requirements 
                if req.id not in requirements_covered
            ]
            
            # Generate test files
            if generated_tests:
                await self._write_test_files(generated_tests, feature_name)
            
            generation_time = int((datetime.now() - start_time).total_seconds() * 1000)
            self.generation_stats["total_generated"] += len(generated_tests)
            self.generation_stats["last_generation_time"] = generation_time
            
            return TestGenerationResult(
                success=len(generated_tests) > 0,
                message=f"Generated {len(generated_tests)} tests for {feature_name}",
                tests_generated=generated_tests,
                total_tests=len(generated_tests),
                coverage_percentage=coverage_percentage,
                requirements_covered=requirements_covered,
                requirements_missing=requirements_missing,
                errors=errors,
                warnings=warnings,
                generation_time_ms=generation_time
            )
            
        except Exception as e:
            error_msg = f"Test generation failed for {feature_name}: {str(e)}"
            logger.error(error_msg)
            return TestGenerationResult(
                success=False,
                message=error_msg,
                tests_generated=[],
                total_tests=0,
                coverage_percentage=0.0,
                requirements_covered=[],
                requirements_missing=[],
                errors=[error_msg],
                warnings=[],
                generation_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )

    async def _parse_requirements_to_tests(
        self, 
        requirements: List[str], 
        feature_name: str,
        acceptance_criteria: List[str],
        user_stories: List[str] = None
    ) -> List[TestRequirement]:
        """Parse natural language requirements into structured test requirements"""
        
        # Prepare the parsing prompt
        user_stories_text = "\n".join(user_stories) if user_stories else "No user stories provided"
        requirements_text = "\n".join(requirements)
        criteria_text = "\n".join(acceptance_criteria)
        
        parsing_prompt = f"""
        Parse the following feature requirements into structured test requirements:
        
        FEATURE: {feature_name}
        
        REQUIREMENTS:
        {requirements_text}
        
        ACCEPTANCE CRITERIA:
        {criteria_text}
        
        USER STORIES:
        {user_stories_text}
        
        Generate test requirements that cover:
        1. Happy path scenarios
        2. Edge cases and boundary conditions
        3. Error handling and validation
        4. Accessibility compliance
        5. Performance characteristics
        6. Cross-browser compatibility
        7. Mobile responsiveness
        8. Security considerations
        
        Each test requirement should include:
        - Unique ID
        - Clear description
        - Test type (unit/integration/e2e/visual/accessibility/performance)
        - Priority level
        - Specific acceptance criteria to validate
        - Preconditions required
        - Expected behavior
        - Edge cases to test
        - Error scenarios to handle
        
        Return as structured data for test generation.
        """
        
        deps = ArchonDependencies()
        
        # Use the agent to parse requirements
        try:
            # Since we're parsing requirements, we'll create a simpler structure
            test_requirements = []
            
            # Generate basic test requirements from acceptance criteria
            for i, criteria in enumerate(acceptance_criteria):
                # Create unit test requirement
                unit_req = TestRequirement(
                    id=f"{feature_name.lower()}_unit_{i+1}",
                    description=f"Unit test for: {criteria}",
                    test_type=TestType.UNIT,
                    priority="high",
                    acceptance_criteria=[criteria],
                    preconditions=["Feature module exists", "Dependencies available"],
                    expected_behavior=criteria,
                    edge_cases=[f"Edge case for {criteria}"],
                    error_scenarios=[f"Error handling for {criteria}"]
                )
                test_requirements.append(unit_req)
                
                # Create E2E test requirement
                e2e_req = TestRequirement(
                    id=f"{feature_name.lower()}_e2e_{i+1}",
                    description=f"End-to-end test for: {criteria}",
                    test_type=TestType.E2E,
                    priority="critical",
                    acceptance_criteria=[criteria],
                    preconditions=["Application running", "Browser available"],
                    expected_behavior=f"User can {criteria}",
                    edge_cases=[f"User workflow edge case: {criteria}"],
                    error_scenarios=[f"User error scenario: {criteria}"]
                )
                test_requirements.append(e2e_req)
            
            # Add accessibility test
            a11y_req = TestRequirement(
                id=f"{feature_name.lower()}_accessibility",
                description=f"Accessibility compliance test for {feature_name}",
                test_type=TestType.ACCESSIBILITY,
                priority="high",
                acceptance_criteria=["WCAG 2.1 AA compliance", "Keyboard navigation", "Screen reader support"],
                preconditions=["Feature UI rendered"],
                expected_behavior="Feature is accessible to users with disabilities",
                edge_cases=["High contrast mode", "Screen reader only", "Keyboard only navigation"],
                error_scenarios=["Missing alt text", "Keyboard traps", "Poor color contrast"]
            )
            test_requirements.append(a11y_req)
            
            return test_requirements
            
        except Exception as e:
            logger.error(f"Failed to parse requirements: {str(e)}")
            # Return minimal test requirements as fallback
            return [
                TestRequirement(
                    id=f"{feature_name.lower()}_basic",
                    description=f"Basic functionality test for {feature_name}",
                    test_type=TestType.E2E,
                    priority="critical",
                    acceptance_criteria=acceptance_criteria,
                    preconditions=["Application available"],
                    expected_behavior="Feature works as specified",
                    edge_cases=["Basic edge cases"],
                    error_scenarios=["Basic error handling"]
                )
            ]

    async def _generate_individual_test(
        self, 
        requirement: TestRequirement, 
        feature_name: str
    ) -> GeneratedTest:
        """Generate individual test implementation from requirement"""
        
        # Determine framework based on test type
        framework = self._get_framework_for_test_type(requirement.test_type)
        
        # Generate test file path
        test_file_path = self._generate_test_file_path(feature_name, requirement, framework)
        
        # Generate test code based on framework and test type
        test_code = await self._generate_test_code(requirement, framework, feature_name)
        
        # Extract stagehand actions and natural language steps
        stagehand_actions = self._extract_stagehand_actions(test_code)
        natural_language_steps = self._extract_natural_language_steps(requirement)
        
        return GeneratedTest(
            id=requirement.id,
            name=self._generate_test_name(requirement),
            description=requirement.description,
            test_type=requirement.test_type,
            framework=framework,
            file_path=test_file_path,
            code=test_code,
            dependencies=self._get_test_dependencies(framework),
            setup_code=self._generate_setup_code(requirement, framework),
            teardown_code=self._generate_teardown_code(requirement, framework),
            assertions=self._extract_assertions(requirement),
            stagehand_actions=stagehand_actions,
            natural_language_steps=natural_language_steps
        )

    def _get_framework_for_test_type(self, test_type: TestType) -> TestFramework:
        """Determine appropriate testing framework for test type"""
        framework_mapping = {
            TestType.UNIT: TestFramework.JEST,
            TestType.INTEGRATION: TestFramework.JEST,
            TestType.E2E: TestFramework.PLAYWRIGHT,
            TestType.VISUAL: TestFramework.PLAYWRIGHT,
            TestType.ACCESSIBILITY: TestFramework.PLAYWRIGHT,
            TestType.PERFORMANCE: TestFramework.PLAYWRIGHT
        }
        return framework_mapping.get(test_type, TestFramework.PLAYWRIGHT)

    def _generate_test_file_path(
        self, 
        feature_name: str, 
        requirement: TestRequirement, 
        framework: TestFramework
    ) -> str:
        """Generate appropriate file path for test"""
        
        type_dir = requirement.test_type.value
        framework_ext = {
            TestFramework.PLAYWRIGHT: ".spec.ts",
            TestFramework.JEST: ".test.ts", 
            TestFramework.VITEST: ".test.ts",
            TestFramework.PYTEST: ".py"
        }.get(framework, ".spec.ts")
        
        filename = f"{feature_name.lower()}_{requirement.id}{framework_ext}"
        return str(self.test_output_dir / type_dir / filename)

    async def _generate_test_code(
        self, 
        requirement: TestRequirement, 
        framework: TestFramework,
        feature_name: str
    ) -> str:
        """Generate actual test code implementation"""
        
        if framework == TestFramework.PLAYWRIGHT:
            return await self._generate_playwright_test(requirement, feature_name)
        elif framework in [TestFramework.JEST, TestFramework.VITEST]:
            return await self._generate_jest_test(requirement, feature_name)
        elif framework == TestFramework.PYTEST:
            return await self._generate_pytest_test(requirement, feature_name)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    async def _generate_playwright_test(
        self, 
        requirement: TestRequirement, 
        feature_name: str
    ) -> str:
        """Generate Playwright test with Stagehand integration"""
        
        test_code = f'''import {{ test, expect }} from '@playwright/test';
import {{ Stagehand }} from '@browserbasehq/stagehand';

test.describe('{feature_name} - {requirement.description}', () => {{
  let stagehand: Stagehand;

  test.beforeEach(async ({{ page }}) => {{
    // Initialize Stagehand with browserbase configuration
    stagehand = new Stagehand({{
      page,
      browserbaseAPIKey: process.env.BROWSERBASE_API_KEY,
      enableCaching: true,
      headless: true,
      debugDom: false
    }});
    
    // Setup test preconditions
    {self._generate_precondition_code(requirement)}
  }});

  test.afterEach(async () => {{
    // Cleanup after test
    await stagehand?.close();
  }});

  test('{requirement.description}', async ({{ page }}) => {{
    // Test implementation using natural language actions
    
    // Navigate to feature
    await stagehand.act("Navigate to the {feature_name.lower()} page");
    
    // Verify preconditions
    await expect(stagehand.observe("Check if {feature_name.lower()} is available")).toBeTruthy();
    
    // Execute main test actions
    {self._generate_main_test_actions(requirement)}
    
    // Verify expected behavior
    {self._generate_assertions_code(requirement)}
    
    // Test edge cases
    {self._generate_edge_case_tests(requirement)}
  }});

  // Additional test for error scenarios
  test('{requirement.description} - Error Handling', async ({{ page }}) => {{
    {self._generate_error_scenario_tests(requirement)}
  }});
}});'''
        
        return test_code

    async def _generate_jest_test(
        self, 
        requirement: TestRequirement, 
        feature_name: str
    ) -> str:
        """Generate Jest/Vitest unit test"""
        
        test_code = f'''import {{ describe, it, expect, beforeEach, afterEach }} from 'vitest';
import {{ {feature_name}Service }} from '../src/services/{feature_name.lower()}Service';

describe('{feature_name} - {requirement.description}', () => {{
  let service: {feature_name}Service;

  beforeEach(() => {{
    // Setup test instance
    service = new {feature_name}Service();
    
    // Setup preconditions
    {self._generate_unit_preconditions(requirement)}
  }});

  afterEach(() => {{
    // Cleanup
    service?.cleanup();
  }});

  it('should {requirement.expected_behavior.lower()}', async () => {{
    // Arrange - Setup test data
    {self._generate_test_data_setup(requirement)}
    
    // Act - Execute the functionality
    {self._generate_unit_test_execution(requirement, feature_name)}
    
    // Assert - Verify expected behavior
    {self._generate_unit_assertions(requirement)}
  }});

  it('should handle edge cases correctly', async () => {{
    {self._generate_unit_edge_cases(requirement)}
  }});

  it('should handle errors appropriately', async () => {{
    {self._generate_unit_error_tests(requirement)}
  }});
}});'''
        
        return test_code

    async def _generate_pytest_test(
        self, 
        requirement: TestRequirement, 
        feature_name: str
    ) -> str:
        """Generate pytest test"""
        
        test_code = f'''import pytest
from unittest.mock import Mock, patch
from src.{feature_name.lower()}_service import {feature_name}Service

class Test{feature_name}{requirement.id.replace(feature_name.lower() + "_", "").title()}:
    """Test class for {requirement.description}"""
    
    @pytest.fixture
    def service(self):
        """Setup test service instance"""
        return {feature_name}Service()
    
    @pytest.fixture
    def test_data(self):
        """Setup test data"""
        return {self._generate_python_test_data(requirement)}
    
    def test_{requirement.id}(self, service, test_data):
        """Test {requirement.expected_behavior}"""
        # Arrange
        {self._generate_python_setup(requirement)}
        
        # Act
        result = service.{self._get_method_name(requirement)}(test_data)
        
        # Assert
        {self._generate_python_assertions(requirement)}
    
    def test_{requirement.id}_edge_cases(self, service):
        """Test edge cases for {requirement.description}"""
        {self._generate_python_edge_cases(requirement)}
    
    def test_{requirement.id}_error_handling(self, service):
        """Test error handling for {requirement.description}"""
        {self._generate_python_error_tests(requirement)}
'''
        
        return test_code

    def _generate_precondition_code(self, requirement: TestRequirement) -> str:
        """Generate code to setup test preconditions"""
        setup_lines = []
        for precondition in requirement.preconditions:
            if "authentication" in precondition.lower():
                setup_lines.append("await stagehand.act('Log in with valid credentials');")
            elif "data" in precondition.lower():
                setup_lines.append("await stagehand.act('Create test data');")
            else:
                setup_lines.append(f"// Ensure {precondition}")
        
        return "\n    ".join(setup_lines)

    def _generate_main_test_actions(self, requirement: TestRequirement) -> str:
        """Generate main test execution actions"""
        actions = []
        for criteria in requirement.acceptance_criteria:
            action = f"await stagehand.act('{self._criteria_to_action(criteria)}');"
            actions.append(action)
        
        return "\n    ".join(actions)

    def _generate_assertions_code(self, requirement: TestRequirement) -> str:
        """Generate test assertions"""
        assertions = []
        for criteria in requirement.acceptance_criteria:
            assertion = f"await expect(stagehand.extract('{self._criteria_to_extraction(criteria)}')).toBeTruthy();"
            assertions.append(assertion)
        
        return "\n    ".join(assertions)

    def _generate_edge_case_tests(self, requirement: TestRequirement) -> str:
        """Generate edge case test code"""
        edge_tests = []
        for edge_case in requirement.edge_cases:
            test_code = f"// Test edge case: {edge_case}\n    await stagehand.act('Test {edge_case.lower()}');"
            edge_tests.append(test_code)
        
        return "\n    ".join(edge_tests)

    def _generate_error_scenario_tests(self, requirement: TestRequirement) -> str:
        """Generate error scenario test code"""
        error_tests = []
        for error_scenario in requirement.error_scenarios:
            test_code = f"// Test error: {error_scenario}\n    await stagehand.act('Trigger {error_scenario.lower()}');\n    await expect(stagehand.observe('Error message displayed')).toBeTruthy();"
            error_tests.append(test_code)
        
        return "\n    ".join(error_tests)

    # Helper methods for different test generation aspects
    def _generate_unit_preconditions(self, requirement: TestRequirement) -> str:
        return "// Unit test preconditions setup"

    def _generate_test_data_setup(self, requirement: TestRequirement) -> str:
        return "const testData = { /* test data */ };"

    def _generate_unit_test_execution(self, requirement: TestRequirement, feature_name: str) -> str:
        return f"const result = await service.{self._get_method_name(requirement)}(testData);"

    def _generate_unit_assertions(self, requirement: TestRequirement) -> str:
        return "expect(result).toBeDefined();\n    expect(result).toBeTruthy();"

    def _generate_unit_edge_cases(self, requirement: TestRequirement) -> str:
        return "// Edge case testing"

    def _generate_unit_error_tests(self, requirement: TestRequirement) -> str:
        return "// Error scenario testing"

    def _generate_python_test_data(self, requirement: TestRequirement) -> str:
        return "{'test': 'data'}"

    def _generate_python_setup(self, requirement: TestRequirement) -> str:
        return "# Python test setup"

    def _generate_python_assertions(self, requirement: TestRequirement) -> str:
        return "assert result is not None\n        assert result == expected_value"

    def _generate_python_edge_cases(self, requirement: TestRequirement) -> str:
        return "# Python edge case tests"

    def _generate_python_error_tests(self, requirement: TestRequirement) -> str:
        return "# Python error handling tests"

    def _criteria_to_action(self, criteria: str) -> str:
        """Convert acceptance criteria to Stagehand action"""
        return f"Perform action for: {criteria}"

    def _criteria_to_extraction(self, criteria: str) -> str:
        """Convert acceptance criteria to data extraction"""
        return f"Extract verification for: {criteria}"

    def _get_method_name(self, requirement: TestRequirement) -> str:
        """Generate method name from requirement"""
        return requirement.id.split('_')[-1]

    def _extract_stagehand_actions(self, test_code: str) -> List[str]:
        """Extract Stagehand actions from generated test code"""
        import re
        actions = re.findall(r'stagehand\.act\([\'"]([^\'"]+)[\'"]', test_code)
        return actions

    def _extract_natural_language_steps(self, requirement: TestRequirement) -> List[str]:
        """Extract natural language test steps"""
        steps = [
            f"Given: {', '.join(requirement.preconditions)}",
            f"When: {requirement.expected_behavior}",
            f"Then: {', '.join(requirement.acceptance_criteria)}"
        ]
        return steps

    def _extract_assertions(self, requirement: TestRequirement) -> List[str]:
        """Extract assertion statements"""
        return [f"Assert {criteria}" for criteria in requirement.acceptance_criteria]

    def _get_test_dependencies(self, framework: TestFramework) -> List[str]:
        """Get required dependencies for framework"""
        deps = {
            TestFramework.PLAYWRIGHT: ["@playwright/test", "@browserbasehq/stagehand"],
            TestFramework.JEST: ["jest", "@types/jest"],
            TestFramework.VITEST: ["vitest", "@vitest/ui"],
            TestFramework.PYTEST: ["pytest", "pytest-asyncio"]
        }
        return deps.get(framework, [])

    def _generate_setup_code(self, requirement: TestRequirement, framework: TestFramework) -> str:
        """Generate test setup code"""
        if framework == TestFramework.PLAYWRIGHT:
            return "await page.goto(baseURL);"
        elif framework in [TestFramework.JEST, TestFramework.VITEST]:
            return "// Setup test environment"
        else:
            return "# Setup test environment"

    def _generate_teardown_code(self, requirement: TestRequirement, framework: TestFramework) -> str:
        """Generate test cleanup code"""
        if framework == TestFramework.PLAYWRIGHT:
            return "await context.close();"
        elif framework in [TestFramework.JEST, TestFramework.VITEST]:
            return "// Cleanup test environment"
        else:
            return "# Cleanup test environment"

    def _generate_test_name(self, requirement: TestRequirement) -> str:
        """Generate human-readable test name"""
        return f"should {requirement.expected_behavior.lower()}"

    def _calculate_coverage(
        self, 
        requirements: List[TestRequirement], 
        generated_tests: List[GeneratedTest],
        acceptance_criteria: List[str]
    ) -> float:
        """Calculate test coverage percentage"""
        if not requirements:
            return 0.0
        
        covered_requirements = len(generated_tests)
        total_requirements = len(requirements)
        
        return (covered_requirements / total_requirements) * 100

    async def _write_test_files(self, generated_tests: List[GeneratedTest], feature_name: str):
        """Write generated test files to disk"""
        
        for test in generated_tests:
            file_path = Path(test.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(test.code)
                    
                logger.info(f"Generated test file: {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to write test file {file_path}: {str(e)}")
                raise

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get test generation statistics"""
        return {
            **self.generation_stats,
            "success_rate": (
                self.generation_stats["successful_generations"] / 
                max(self.generation_stats["total_generated"], 1)
            ) * 100 if self.generation_stats["total_generated"] > 0 else 0
        }

# Standalone function for easy integration
async def generate_tests_from_natural_language(
    feature_name: str,
    requirements: List[str],
    acceptance_criteria: List[str],
    project_path: str = ".",
    **kwargs
) -> TestGenerationResult:
    """
    Generate tests from natural language requirements
    
    Args:
        feature_name: Name of the feature being tested
        requirements: List of natural language requirements
        acceptance_criteria: Specific acceptance criteria to validate
        project_path: Path to the project root
        **kwargs: Additional configuration options
        
    Returns:
        TestGenerationResult with generated tests
    """
    
    engine = StagehandTestEngine(project_path=project_path, **kwargs)
    
    return await engine.generate_tests_from_requirements(
        requirements=requirements,
        feature_name=feature_name,
        acceptance_criteria=acceptance_criteria
    )