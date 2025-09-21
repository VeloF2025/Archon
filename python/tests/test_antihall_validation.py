"""
Test suite for Anti-Hallucination Validation and 75% Confidence Rule

Tests the enhanced validation system that prevents AI agents from
hallucinating code references and enforces confidence thresholds.
"""

import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock

from src.agents.validation.enhanced_antihall_validator import (
    EnhancedAntiHallValidator,
    CodeReference,
    ValidationReport,
    ValidationResult,
    RealTimeValidator,
    AgentValidationWrapper
)
from src.agents.validation.confidence_based_responses import (
    ConfidenceBasedResponseSystem,
    ConfidenceAssessment,
    ConfidenceLevel,
    UncertaintyHandler,
    AgentConfidenceWrapper
)
from src.server.services.validation_service import (
    ValidationService,
    ValidationConfig,
    initialize_validation_service
)

class TestEnhancedAntiHallValidator:
    """Test suite for enhanced anti-hallucination validator"""
    
    @pytest.fixture
    def validator(self, tmp_path):
        """Create validator with test project"""
        # Create test project structure
        project_root = tmp_path / "test_project"
        project_root.mkdir()
        
        # Create test Python file
        test_file = project_root / "test_module.py"
        test_file.write_text("""
def real_function():
    return "This function exists"

class RealClass:
    def real_method(self):
        return "This method exists"
        
def another_function(param1, param2):
    return param1 + param2
""")
        
        # Create test TypeScript file
        ts_file = project_root / "test.ts"
        ts_file.write_text("""
export function realTypeScriptFunction() {
    return "TypeScript function";
}

export class RealTypeScriptClass {
    realMethod() {
        return "TypeScript method";
    }
}
""")
        
        return EnhancedAntiHallValidator(str(project_root))
    
    def test_validate_existing_function(self, validator):
        """Test validation of existing function"""
        ref = CodeReference("function", "real_function")
        report = validator.validate_reference(ref)
        
        assert report.result == ValidationResult.EXISTS
        assert report.confidence == 1.0
        assert "test_module.py" in report.actual_location
    
    def test_validate_nonexistent_function(self, validator):
        """Test validation of non-existent function"""
        ref = CodeReference("function", "fake_function")
        report = validator.validate_reference(ref)
        
        assert report.result == ValidationResult.NOT_FOUND
        assert report.confidence == 0.0
        assert report.actual_location is None
    
    def test_validate_similar_name_suggestion(self, validator):
        """Test that validator suggests similar names"""
        ref = CodeReference("function", "real_functon")  # Typo
        report = validator.validate_reference(ref)
        
        assert report.result == ValidationResult.NOT_FOUND
        assert report.similar_matches is not None
        assert "real_function" in report.similar_matches
        assert "Did you mean: real_function?" in report.suggestion
    
    def test_validate_class_method(self, validator):
        """Test validation of class method"""
        ref = CodeReference("method", "real_method", context="RealClass")
        report = validator.validate_reference(ref)
        
        assert report.result == ValidationResult.EXISTS
        assert report.confidence == 1.0
    
    def test_validate_code_snippet(self, validator):
        """Test validation of complete code snippet"""
        code = """
result = real_function()
obj = RealClass()
value = obj.real_method()
fake_result = nonexistent_function()  # This should fail
"""
        
        reports = validator.validate_code_snippet(code, "python")
        
        # Should have reports for all references
        assert len(reports) > 0
        
        # Check that fake function is detected
        fake_reports = [r for r in reports if r.reference.name == "nonexistent_function"]
        assert len(fake_reports) > 0
        assert fake_reports[0].result == ValidationResult.NOT_FOUND
    
    def test_enforce_validation(self, validator):
        """Test enforcement of validation with confidence threshold"""
        code = """
real_function()
fake_function()  # Will fail validation
"""
        
        is_valid, summary = validator.enforce_validation(code, "python", min_confidence=0.75)
        
        assert not is_valid  # Should fail due to fake_function
        assert summary["invalid_references"] > 0
        assert not summary["safe_to_proceed"]


class TestConfidenceBasedResponses:
    """Test suite for confidence-based response system"""
    
    @pytest.fixture
    def confidence_system(self):
        """Create confidence system with 75% threshold"""
        return ConfidenceBasedResponseSystem(min_confidence_threshold=0.75)
    
    def test_high_confidence_response(self, confidence_system):
        """Test response with high confidence"""
        context = {
            "code_validation": {"all_references_valid": True, "validation_rate": 1.0},
            "documentation_found": True,
            "tests_exist": True,
            "similar_patterns_found": True,
            "recently_used": True
        }
        
        assessment = confidence_system.assess_confidence(context)
        
        assert assessment.confidence_score >= 0.9
        assert assessment.confidence_level == ConfidenceLevel.HIGH
        assert len(assessment.uncertainties) == 0
    
    def test_low_confidence_response(self, confidence_system):
        """Test response with low confidence"""
        context = {
            "code_validation": {"all_references_valid": False, "validation_rate": 0.3},
            "documentation_found": False,
            "tests_exist": False,
            "similar_patterns_found": False,
            "recently_used": False
        }
        
        assessment = confidence_system.assess_confidence(context)
        
        assert assessment.confidence_score < 0.75
        assert assessment.confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]
        assert len(assessment.uncertainties) > 0
        assert len(assessment.suggestions) > 0
    
    def test_confidence_threshold_enforcement(self, confidence_system):
        """Test that 75% threshold is enforced"""
        # Context with exactly 74% confidence
        context = {
            "code_validation": {"all_references_valid": True, "validation_rate": 0.74},
            "documentation_found": True,
            "tests_exist": False,
            "similar_patterns_found": False,
            "recently_used": False
        }
        
        assessment = confidence_system.assess_confidence(context)
        response = confidence_system.generate_response(
            "Some content",
            assessment,
            "code_implementation"
        )
        
        # Should express uncertainty below 75%
        assert "not certain" in response.lower() or "don't know" in response.lower()
    
    def test_uncertainty_detection(self):
        """Test detection of uncertainty patterns in responses"""
        handler = UncertaintyHandler()
        
        uncertain_response = "I think this might work, but I'm not sure"
        has_uncertainty, patterns = handler.detect_uncertainty_in_response(uncertain_response)
        
        assert has_uncertainty
        assert "I think" in patterns
        assert "might" in patterns or "might be" in patterns
        assert "not sure" in patterns or "I'm not sure" in patterns
    
    def test_uncertainty_rewriting(self):
        """Test rewriting of uncertain responses"""
        handler = UncertaintyHandler()
        
        original = "This should probably work"
        rewritten = handler.rewrite_uncertain_response(original, confidence=0.6)
        
        assert "Low Confidence" in rewritten
        assert "60%" in rewritten
        assert "verify" in rewritten.lower()


class TestValidationService:
    """Test suite for validation service integration"""
    
    @pytest.fixture
    async def validation_service(self, tmp_path):
        """Create validation service with test configuration"""
        config = ValidationConfig(
            project_root=str(tmp_path),
            min_confidence_threshold=0.75,
            enable_real_time_validation=True,
            enable_auto_fix=True,
            cache_validation_results=True
        )
        return ValidationService(config)
    
    @pytest.mark.asyncio
    async def test_validate_code_reference(self, validation_service):
        """Test validation of single code reference"""
        # This will fail since no code exists in test project
        report = await validation_service.validate_code_reference("fake_function", "function")
        
        assert report.result == ValidationResult.NOT_FOUND
        assert report.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_validate_with_confidence(self, validation_service):
        """Test validation with confidence assessment"""
        context = {
            "code_validation": {"all_references_valid": False, "validation_rate": 0.5},
            "documentation_found": False,
            "question_type": "code_implementation"
        }
        
        result = await validation_service.validate_with_confidence(
            "Test content",
            context
        )
        
        assert not result["success"]  # Should fail due to low confidence
        assert result["confidence_too_low"]
        assert result["confidence_score"] < 0.75
        assert "don't" in result["response"].lower() or "not certain" in result["response"].lower()
    
    @pytest.mark.asyncio
    async def test_validate_agent_response(self, validation_service):
        """Test validation of AI agent response"""
        agent_response = "I think this code might work:\n```python\nresult = maybe_function()\n```"
        context = {"contains_code": True, "question_type": "code_implementation"}
        
        result = await validation_service.validate_agent_response(agent_response, context)
        
        assert result["uncertainty_detected"]
        assert "I think" in result["uncertainty_patterns"]
        assert len(result["validation_results"]) > 0
    
    @pytest.mark.asyncio
    async def test_real_time_validation(self, validation_service):
        """Test real-time line validation"""
        # Test suspicious pattern
        suspicious_line = "result = obj.do_something()"
        error = await validation_service.perform_real_time_validation(
            suspicious_line,
            {}
        )
        
        # Should detect suspicious generic method name
        assert error is not None
    
    @pytest.mark.asyncio
    async def test_validation_statistics(self, validation_service):
        """Test statistics tracking"""
        # Perform some validations
        await validation_service.validate_code_reference("test_func", "function")
        await validation_service.validate_code_reference("test_func", "function")  # Cache hit
        
        stats = validation_service.get_statistics()
        
        assert stats["total_validations"] >= 1
        assert stats["cache_hits"] >= 1
        assert "cache_hit_rate" in stats
        assert "validation_success_rate" in stats


class TestAgentValidationWrapper:
    """Test suite for agent validation wrapper"""
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent for testing"""
        agent = Mock()
        agent.generate_code = AsyncMock(return_value={
            "success": True,
            "code": "print('Hello World')"
        })
        return agent
    
    @pytest.fixture
    def wrapped_agent(self, mock_agent, tmp_path):
        """Create wrapped agent with validation"""
        validator = EnhancedAntiHallValidator(str(tmp_path))
        return AgentValidationWrapper(mock_agent, validator)
    
    @pytest.mark.asyncio
    async def test_agent_validation_wrapper(self, wrapped_agent):
        """Test that agent wrapper validates before generation"""
        # Request with non-existent reference
        prompt = "Use the `fake_function` to process data"
        context = {"language": "python"}
        
        result = await wrapped_agent.generate_code(prompt, context)
        
        # Should fail validation
        assert not result["success"]
        assert "invalid references" in result["error"].lower()
        assert "fake_function" in result["error"]
    
    @pytest.mark.asyncio
    async def test_agent_confidence_wrapper(self):
        """Test agent confidence wrapper"""
        mock_agent = Mock()
        mock_agent.process_request = AsyncMock(return_value={
            "response": "Test response"
        })
        
        wrapped = AgentConfidenceWrapper(mock_agent, min_confidence=0.75)
        
        # Low confidence context
        context = {
            "code_validation": {"all_references_valid": False, "validation_rate": 0.3},
            "question_type": "code_implementation"
        }
        
        result = await wrapped.process_request("Test request", context)
        
        assert not result["success"]
        assert result["confidence_too_low"]
        assert result["confidence_score"] < 0.75


class TestIntegration:
    """Integration tests for complete validation system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_validation_flow(self, tmp_path):
        """Test complete validation flow from API to agent"""
        # Initialize service
        service = await initialize_validation_service(str(tmp_path))
        
        # Validate code snippet
        code = """
import os
result = fake_function()  # This will fail
"""
        
        validation_result = await service.validate_code_snippet(code, "python")
        
        assert not validation_result["valid"]
        
        # Check confidence
        confidence_result = await service.validate_with_confidence(
            "Solution code",
            {"code_validation": {"all_references_valid": False, "validation_rate": 0.5}}
        )
        
        assert confidence_result["confidence_too_low"]
        
        # Get statistics
        stats = service.get_statistics()
        assert stats["total_validations"] > 0
        assert stats["hallucinations_prevented"] > 0
    
    @pytest.mark.asyncio
    async def test_75_percent_rule_enforcement(self, tmp_path):
        """Test that 75% confidence rule is properly enforced"""
        service = await initialize_validation_service(str(tmp_path))
        
        # Test various confidence levels
        test_cases = [
            (0.9, True, "high confidence should pass"),
            (0.8, True, "80% should pass"),
            (0.75, True, "exactly 75% should pass"),
            (0.74, False, "74% should fail"),
            (0.5, False, "50% should fail"),
            (0.3, False, "30% should fail")
        ]
        
        for confidence, should_pass, description in test_cases:
            context = {
                "code_validation": {
                    "all_references_valid": confidence > 0.8,
                    "validation_rate": confidence
                }
            }
            
            result = await service.validate_with_confidence("Test", context)
            
            if should_pass:
                assert result["success"], f"Failed: {description}"
                assert not result.get("confidence_too_low"), f"Failed: {description}"
            else:
                assert not result["success"], f"Failed: {description}"
                assert result.get("confidence_too_low"), f"Failed: {description}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])