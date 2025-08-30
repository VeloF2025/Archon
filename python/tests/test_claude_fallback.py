"""
Test Claude Code Fallback for External Validator
Verifies fallback behavior and guardrails
"""

import asyncio
import pytest
from typing import Dict, Any

from src.agents.external_validator.claude_fallback import ClaudeFallbackValidator
from src.agents.external_validator.llm_client import LLMClient
from src.agents.external_validator.config import ValidatorConfig


@pytest.mark.asyncio
async def test_claude_fallback_initialization():
    """Test that Claude fallback initializes when no API key is present"""
    
    # Create config with no API key
    config = ValidatorConfig()
    config.llm_config.api_key = None  # No external API key
    
    # Initialize LLM client
    client = LLMClient(config)
    await client.initialize()
    
    # Should be using fallback
    assert client.using_fallback is True
    assert client.claude_fallback is not None
    assert client.client is None  # No external client
    
    # Check connection should succeed with fallback
    connected = await client.check_connection()
    assert connected is True


@pytest.mark.asyncio
async def test_guardrails_applied():
    """Test that guardrails are properly applied"""
    
    fallback = ClaudeFallbackValidator()
    
    # Test validation with self-work
    content = "def test(): return True"
    
    # Track as self-work
    fallback.track_archon_output(content)
    
    # Validate
    result = await fallback.validate_with_guardrails(
        content=content,
        validation_type="code",
        context=None
    )
    
    # Check guardrails applied
    assert result["guardrails_applied"] is True
    assert result["self_work"] is True
    assert result["confidence"] <= fallback.MAX_CONFIDENCE_SELF_WORK
    assert len(result["issues"]) >= fallback.MIN_ISSUES_REQUIRED
    
    # Should find gaming pattern
    gaming_issues = [i for i in result["issues"] if i["type"] == "gaming"]
    assert len(gaming_issues) > 0, "Should detect 'return True' as gaming pattern"


@pytest.mark.asyncio
async def test_minimum_issues_enforced():
    """Test that minimum issues are always found"""
    
    fallback = ClaudeFallbackValidator()
    
    # Even good code should have at least one issue
    good_code = """
def calculate_sum(a: int, b: int) -> int:
    '''Calculate sum of two integers'''
    return a + b
"""
    
    result = await fallback.validate_with_guardrails(
        content=good_code,
        validation_type="code",
        context=None
    )
    
    # Should have at least minimum issues
    assert len(result["issues"]) >= fallback.MIN_ISSUES_REQUIRED
    assert len(result["suggestions"]) > 0


@pytest.mark.asyncio
async def test_self_work_detection():
    """Test that self-generated content is properly detected"""
    
    fallback = ClaudeFallbackValidator()
    
    # Track some content
    original_content = "This is Archon-generated content"
    hash1 = fallback.track_archon_output(original_content)
    
    # Check detection
    assert fallback.is_self_work(original_content) is True
    assert fallback.is_self_work("Different content") is False
    
    # Validate self-work
    result = await fallback.validate_with_guardrails(
        content=original_content,
        validation_type="documentation",
        context=None
    )
    
    assert result["self_work"] is True
    assert "self-generated content" in str(result["issues"]).lower()


@pytest.mark.asyncio
async def test_skepticism_level():
    """Test that skepticism is properly applied"""
    
    fallback = ClaudeFallbackValidator()
    
    content = "This function works perfectly with no issues"
    
    result = await fallback.validate_with_guardrails(
        content=content,
        validation_type="documentation",
        context=None
    )
    
    # Confidence should be reduced by skepticism
    assert result["confidence"] < 1.0
    assert result["guardrails"]["skepticism_applied"] == fallback.SKEPTICISM_LEVEL
    
    # Should flag overly confident language
    confident_issues = [
        i for i in result["issues"] 
        if "confident" in i.get("description", "").lower()
    ]
    assert len(confident_issues) > 0


@pytest.mark.asyncio
async def test_gaming_pattern_detection():
    """Test detection of gaming patterns"""
    
    fallback = ClaudeFallbackValidator()
    
    gaming_code = """
def test_feature():
    # TODO: implement
    assert True  # Always passes
    return 'mock_data'
"""
    
    result = await fallback.validate_with_guardrails(
        content=gaming_code,
        validation_type="code",
        context=None
    )
    
    # Should detect multiple gaming patterns
    gaming_issues = [i for i in result["issues"] if i["type"] == "gaming"]
    assert len(gaming_issues) >= 2  # assert True and mock_data
    
    # Should not be valid
    assert result["valid"] is False
    assert result["confidence"] < 0.5


@pytest.mark.asyncio
async def test_fallback_with_llm_client():
    """Test fallback through LLM client"""
    
    # Config with no API key
    config = ValidatorConfig()
    config.llm_config.api_key = None
    
    client = LLMClient(config)
    await client.initialize()
    
    # Validate something
    result = await client.validate_with_llm(
        prompt="def test(): assert True",
        context={"source": "archon"},  # Mark as Archon output
        validation_type="code"
    )
    
    # Should use fallback
    assert result["validator"] == "claude_fallback"
    assert result["guardrails_applied"] is True
    assert len(result["issues"]) > 0
    
    # Get metrics
    metrics = await client.get_metrics()
    assert metrics["using_fallback"] is True
    assert metrics["guardrails_active"] is True


@pytest.mark.asyncio
async def test_fallback_on_api_failure():
    """Test automatic fallback when external API fails"""
    
    # Config with invalid API key
    config = ValidatorConfig()
    config.llm_config.provider = "openai"
    config.llm_config.api_key = "invalid-key-12345"
    
    client = LLMClient(config)
    await client.initialize()
    
    # Should try external first, then fall back
    # Check connection will fail and trigger fallback
    connected = await client.check_connection()
    
    # Should have switched to fallback
    assert client.using_fallback is True
    assert connected is True  # Fallback is available


def test_metrics_tracking():
    """Test that metrics are properly tracked"""
    
    fallback = ClaudeFallbackValidator()
    
    # Initially no validations
    metrics = fallback.get_metrics()
    assert metrics["total_validations"] == 0
    
    # Track some validations
    asyncio.run(fallback.validate_with_guardrails(
        "test content 1", "code", None
    ))
    asyncio.run(fallback.validate_with_guardrails(
        "test content 2", "code", None
    ))
    
    metrics = fallback.get_metrics()
    assert metrics["total_validations"] == 2
    assert metrics["average_confidence"] <= fallback.SKEPTICISM_LEVEL
    assert metrics["average_issues"] >= fallback.MIN_ISSUES_REQUIRED


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_claude_fallback_initialization())
    print("[PASS] Fallback initialization test")
    
    asyncio.run(test_guardrails_applied())
    print("[PASS] Guardrails application test")
    
    asyncio.run(test_minimum_issues_enforced())
    print("[PASS] Minimum issues enforcement test")
    
    asyncio.run(test_self_work_detection())
    print("[PASS] Self-work detection test")
    
    asyncio.run(test_skepticism_level())
    print("[PASS] Skepticism level test")
    
    asyncio.run(test_gaming_pattern_detection())
    print("[PASS] Gaming pattern detection test")
    
    asyncio.run(test_fallback_with_llm_client())
    print("[PASS] LLM client fallback test")
    
    asyncio.run(test_fallback_on_api_failure())
    print("[PASS] API failure fallback test")
    
    test_metrics_tracking()
    print("[PASS] Metrics tracking test")
    
    print("\n=== All Claude Fallback Tests PASSED ===")
    print("Guardrails are working correctly!")
    print("- Self-work detection: Active")
    print("- Minimum issues enforcement: Active")
    print("- Skepticism level: 80%")
    print("- Gaming pattern detection: Active")