"""
Tests for UI/UX Validation Agent

This test suite validates the functionality of the UI/UX Validator Agent
including its integration with the Browser MCP module and validation
capabilities.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.agents.ui_ux_validator_agent import (
    UIUXValidatorAgent,
    ValidationMode,
    ValidationSeverity,
    BrowserType,
    ViewportConfig,
    ValidationIssue,
    ValidationReport,
    UIUXValidationDependencies,
    UIUXValidatorOutput,
    validate_ui_ux
)


class TestUIUXValidatorAgent:
    """Test cases for UI/UX Validator Agent"""

    @pytest.fixture
    def agent(self):
        """Create a UI/UX validator agent instance for testing"""
        return UIUXValidatorAgent()

    @pytest.fixture
    def mock_mcp_client(self):
        """Mock MCP client for testing"""
        mock_client = AsyncMock()
        
        # Mock navigation result
        mock_client.call_tool.return_value = json.dumps({
            "success": True,
            "url": "https://example.com",
            "title": "Test Page",
            "status": 200
        })
        
        return mock_client

    @pytest.fixture
    def sample_dependencies(self):
        """Sample dependencies for testing"""
        return UIUXValidationDependencies(
            target_url="https://example.com",
            validation_mode=ValidationMode.COMPREHENSIVE
        )

    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent.name == "UIUXValidator"
        assert agent.model == "openai:gpt-4o"
        assert len(agent.default_viewports) == 5
        
        # Verify default viewports
        viewport_names = [v.name for v in agent.default_viewports]
        assert "Desktop" in viewport_names
        assert "Mobile" in viewport_names
        assert "Tablet" in viewport_names

    def test_viewport_config(self):
        """Test viewport configuration"""
        viewport = ViewportConfig(
            name="Test Viewport",
            width=1920,
            height=1080,
            device_type="desktop"
        )
        
        assert viewport.name == "Test Viewport"
        assert viewport.width == 1920
        assert viewport.height == 1080
        assert viewport.device_type == "desktop"

    def test_validation_issue_creation(self):
        """Test validation issue model"""
        issue = ValidationIssue(
            category="accessibility",
            severity=ValidationSeverity.CRITICAL,
            title="Missing Alt Text",
            description="Images without alt text found",
            element_selector="img:not([alt])",
            wcag_criteria="1.1.1",
            recommendations=["Add alt text to images"]
        )
        
        assert issue.category == "accessibility"
        assert issue.severity == ValidationSeverity.CRITICAL
        assert issue.wcag_criteria == "1.1.1"
        assert len(issue.recommendations) == 1

    def test_validation_report_structure(self):
        """Test validation report structure"""
        report = ValidationReport(
            url="https://example.com",
            validation_mode=ValidationMode.COMPREHENSIVE,
            timestamp=datetime.now(),
            summary={"total_issues": 5},
            issues=[],
            screenshots={},
            performance_metrics={},
            recommendations=[],
            browsers_tested=["chromium"],
            viewports_tested=["desktop"]
        )
        
        assert report.url == "https://example.com"
        assert report.validation_mode == ValidationMode.COMPREHENSIVE
        assert report.summary["total_issues"] == 5

    @pytest.mark.asyncio
    async def test_convenience_function(self):
        """Test the convenience validate_ui_ux function"""
        with patch('src.agents.ui_ux_validator_agent.UIUXValidatorAgent') as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run_validation.return_value = UIUXValidatorOutput(
                success=True,
                message="Validation completed",
                pass_fail_status="pass"
            )
            mock_agent_class.return_value = mock_agent
            
            result = await validate_ui_ux(
                target_url="https://example.com",
                mode=ValidationMode.QUICK_SCAN
            )
            
            assert result.success is True
            assert result.pass_fail_status == "pass"
            mock_agent.run_validation.assert_called_once()

    def test_accessibility_score_calculation(self, agent):
        """Test accessibility score calculation"""
        # Test with no issues - should return 100
        score = agent._calculate_accessibility_score([])
        assert score == 100.0
        
        # Test with various severity issues
        issues = [
            ValidationIssue(
                category="accessibility",
                severity=ValidationSeverity.CRITICAL,
                title="Critical Issue",
                description="Test"
            ),
            ValidationIssue(
                category="accessibility", 
                severity=ValidationSeverity.HIGH,
                title="High Issue",
                description="Test"
            ),
            ValidationIssue(
                category="accessibility",
                severity=ValidationSeverity.MEDIUM, 
                title="Medium Issue",
                description="Test"
            )
        ]
        
        score = agent._calculate_accessibility_score(issues)
        # Should be 100 - 20 (critical) - 10 (high) - 5 (medium) = 65
        assert score == 65.0

    def test_recommendation_generation(self, agent):
        """Test recommendation generation based on issues"""
        issues = [
            ValidationIssue(
                category="accessibility",
                severity=ValidationSeverity.CRITICAL,
                title="Critical A11y Issue",
                description="Test"
            ),
            ValidationIssue(
                category="performance",
                severity=ValidationSeverity.HIGH,
                title="Performance Issue", 
                description="Test"
            ),
            ValidationIssue(
                category="accessibility",
                severity=ValidationSeverity.MEDIUM,
                title="Medium A11y Issue",
                description="Test"
            )
        ]
        
        recommendations = agent._generate_recommendations(issues)
        
        assert len(recommendations) == 2  # One for accessibility, one for performance
        assert any("URGENT" in rec and "accessibility" in rec for rec in recommendations)
        assert any("HIGH PRIORITY" in rec and "performance" in rec for rec in recommendations)

    def test_validation_modes(self):
        """Test validation mode enumeration"""
        assert ValidationMode.QUICK_SCAN == "quick_scan"
        assert ValidationMode.COMPREHENSIVE == "comprehensive"
        assert ValidationMode.FOCUSED == "focused"
        assert ValidationMode.CONTINUOUS == "continuous"

    def test_severity_levels(self):
        """Test validation severity enumeration"""
        assert ValidationSeverity.CRITICAL == "critical"
        assert ValidationSeverity.HIGH == "high"
        assert ValidationSeverity.MEDIUM == "medium"
        assert ValidationSeverity.LOW == "low"
        assert ValidationSeverity.INFO == "info"

    def test_browser_types(self):
        """Test browser type enumeration"""
        assert BrowserType.CHROMIUM == "chromium"
        assert BrowserType.FIREFOX == "firefox"
        assert BrowserType.WEBKIT == "webkit"

    @pytest.mark.asyncio
    async def test_agent_tool_registration(self, agent):
        """Test that all required tools are registered with the agent"""
        # Access the underlying PydanticAI agent
        pydantic_agent = agent._agent
        
        # Check that tools are registered (this would require inspection of the agent's tools)
        # This is a basic check - in practice you'd verify specific tools exist
        assert pydantic_agent is not None
        assert hasattr(pydantic_agent, 'tool')

    def test_dependencies_configuration(self, sample_dependencies):
        """Test dependencies configuration"""
        assert sample_dependencies.target_url == "https://example.com"
        assert sample_dependencies.validation_mode == ValidationMode.COMPREHENSIVE
        assert sample_dependencies.wcag_level == "AA"
        assert len(sample_dependencies.browsers_to_test) == 1
        assert sample_dependencies.browsers_to_test[0] == BrowserType.CHROMIUM

    def test_custom_viewport_configuration(self):
        """Test custom viewport configuration"""
        custom_viewports = [
            ViewportConfig(name="Large Desktop", width=2560, height=1440, device_type="desktop"),
            ViewportConfig(name="iPhone X", width=375, height=812, device_type="mobile")
        ]
        
        deps = UIUXValidationDependencies(
            target_url="https://example.com",
            viewports_to_test=custom_viewports
        )
        
        assert len(deps.viewports_to_test) == 2
        assert deps.viewports_to_test[0].width == 2560
        assert deps.viewports_to_test[1].device_type == "mobile"

    @pytest.mark.asyncio 
    async def test_validation_with_custom_config(self, agent):
        """Test validation with custom configuration"""
        custom_viewports = [
            ViewportConfig(name="Desktop", width=1920, height=1080, device_type="desktop")
        ]
        
        # Mock the actual validation run
        with patch.object(agent, 'run') as mock_run:
            mock_run.return_value = UIUXValidatorOutput(
                success=True,
                message="Validation completed",
                pass_fail_status="pass"
            )
            
            result = await agent.run_validation(
                target_url="https://example.com",
                validation_mode=ValidationMode.FOCUSED,
                viewports_to_test=custom_viewports,
                focus_component="navigation menu"
            )
            
            assert result.success is True
            mock_run.assert_called_once()

    def test_system_prompt_generation(self, agent):
        """Test system prompt generation"""
        prompt = agent.get_system_prompt()
        
        # Check that key validation areas are mentioned
        assert "accessibility" in prompt.lower()
        assert "performance" in prompt.lower()
        assert "responsive design" in prompt.lower()
        assert "wcag" in prompt.lower()
        assert "core web vitals" in prompt.lower()

    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling in validation"""
        # Mock an error scenario
        with patch.object(agent, 'run') as mock_run:
            mock_run.side_effect = Exception("Navigation failed")
            
            with pytest.raises(Exception, match="Navigation failed"):
                await agent.run_validation(
                    target_url="https://invalid-url.com",
                    validation_mode=ValidationMode.QUICK_SCAN
                )

    def test_output_model_structure(self):
        """Test the output model structure"""
        output = UIUXValidatorOutput(
            success=True,
            message="Validation completed",
            issues_count={"critical": 2, "high": 5},
            pass_fail_status="fail"
        )
        
        assert output.success is True
        assert output.issues_count["critical"] == 2
        assert output.pass_fail_status == "fail"

    @pytest.mark.asyncio
    async def test_multiple_browser_support(self, agent):
        """Test validation with multiple browsers"""
        browsers = [BrowserType.CHROMIUM, BrowserType.FIREFOX]
        
        with patch.object(agent, 'run') as mock_run:
            mock_run.return_value = UIUXValidatorOutput(
                success=True,
                message="Multi-browser validation completed"
            )
            
            result = await agent.run_validation(
                target_url="https://example.com",
                browsers_to_test=browsers
            )
            
            assert result.success is True
            mock_run.assert_called_once()

    def test_wcag_level_configuration(self):
        """Test WCAG level configuration"""
        # Test different WCAG levels
        for level in ["A", "AA", "AAA"]:
            deps = UIUXValidationDependencies(
                target_url="https://example.com",
                wcag_level=level
            )
            assert deps.wcag_level == level

    @pytest.mark.asyncio
    async def test_baseline_comparison_config(self, agent):
        """Test visual regression baseline configuration"""
        baseline_path = "/tmp/baseline.png"
        
        with patch.object(agent, 'run') as mock_run:
            mock_run.return_value = UIUXValidatorOutput(
                success=True,
                message="Visual regression test completed"
            )
            
            result = await agent.run_validation(
                target_url="https://example.com",
                baseline_path=baseline_path
            )
            
            assert result.success is True


class TestValidationIntegration:
    """Integration tests for UI/UX validation"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_validation_flow(self):
        """Test complete validation flow (mocked)"""
        with patch('src.agents.ui_ux_validator_agent.get_mcp_client') as mock_get_client:
            # Mock MCP client responses
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            
            # Mock navigation response
            mock_client.call_tool.return_value = json.dumps({
                "success": True,
                "url": "https://example.com",
                "title": "Test Page"
            })
            
            mock_get_client.return_value = mock_client
            
            # Run validation
            result = await validate_ui_ux(
                target_url="https://example.com",
                mode=ValidationMode.QUICK_SCAN
            )
            
            # Verify the flow executed
            assert mock_client.call_tool.called


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])