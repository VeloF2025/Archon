# Agent Anti-Hallucination Integration Guide

## Quick Start: Adding Validation to Your Agent

### Step 1: Import Required Components
```python
from src.agents.validation.enhanced_antihall_validator import (
    EnhancedAntiHallValidator,
    AgentValidationWrapper
)
from src.agents.validation.confidence_based_responses import (
    ConfidenceBasedResponseSystem,
    AgentConfidenceWrapper
)
```

### Step 2: Wrap Your Agent
```python
# Your original agent
my_agent = MyCustomAgent()

# Add validation wrapper
validator = EnhancedAntiHallValidator(project_root)
validated_agent = AgentValidationWrapper(my_agent, validator)

# Add confidence wrapper
confident_agent = AgentConfidenceWrapper(validated_agent, min_confidence=0.75)
```

### Step 3: Use Validated Agent
```python
# Agent now automatically validates and checks confidence
result = await confident_agent.process_request(prompt, context)
```

## Integration Patterns for Each Agent Type

### 1. Code Implementer Agent

**Required Validations**:
- All imports must exist
- All function/class references must be valid
- Method calls must reference real methods

**Implementation**:
```python
class CodeImplementerAgent(BaseSpecializedAgent):
    async def generate_code(self, task: str, context: Dict) -> Dict:
        # Step 1: Extract code references from task
        references = self.extract_references(task)
        
        # Step 2: Validate all references
        for ref in references:
            validation = await self.validate_before_execution(
                f"from {ref.module} import {ref.name}",
                "python"
            )
            if not validation["is_valid"]:
                return {
                    "success": False,
                    "error": f"Cannot find '{ref.name}'. {validation['suggestions'][0] if validation.get('suggestions') else ''}",
                    "confidence": 0.0
                }
        
        # Step 3: Check confidence
        confidence_result = await self.check_confidence(
            task,
            {"code_validation": {"all_references_valid": True, "validation_rate": 1.0}}
        )
        
        if confidence_result["confidence_score"] < 0.75:
            return {
                "success": False,
                "error": "I don't have enough confidence to implement this. Let's discuss the requirements.",
                "confidence": confidence_result["confidence_score"]
            }
        
        # Step 4: Generate code (only if validation passes)
        code = await self._generate_validated_code(task, context)
        
        # Step 5: Final validation of generated code
        final_validation = await self.validate_before_execution(code, "python")
        
        return {
            "success": final_validation["is_valid"],
            "code": code if final_validation["is_valid"] else None,
            "confidence": confidence_result["confidence_score"],
            "validation_summary": final_validation["summary"]
        }
```

### 2. System Architect Agent

**Required Validations**:
- Existing architecture components
- Technology stack availability
- Integration points exist

**Implementation**:
```python
class SystemArchitectAgent(BaseSpecializedAgent):
    async def design_architecture(self, requirements: str) -> Dict:
        # Check confidence based on codebase understanding
        codebase_analysis = await self.analyze_codebase()
        
        confidence_context = {
            "code_validation": {
                "all_references_valid": codebase_analysis["has_required_components"],
                "validation_rate": codebase_analysis["component_coverage"]
            },
            "documentation_found": codebase_analysis["has_architecture_docs"],
            "similar_patterns_found": codebase_analysis["has_similar_patterns"]
        }
        
        confidence_result = await self.check_confidence(
            "Designing architecture",
            confidence_context
        )
        
        if confidence_result["confidence_score"] < 0.75:
            return {
                "success": False,
                "message": f"I need more information about your existing architecture. "
                          f"Confidence: {confidence_result['confidence_score']:.0%}\n"
                          f"Missing: {', '.join(confidence_result['uncertainties'])}"
            }
        
        # Design with confidence
        return await self._create_architecture_design(requirements)
```

### 3. Test Coverage Validator Agent

**Required Validations**:
- Functions to test exist
- Test framework is available
- Existing test patterns

**Implementation**:
```python
class TestCoverageValidatorAgent(BaseSpecializedAgent):
    async def create_tests(self, code_file: str) -> Dict:
        # Validate the code file exists
        if not Path(code_file).exists():
            return {
                "success": False,
                "error": f"File '{code_file}' does not exist. Please provide a valid file path."
            }
        
        # Parse and validate all functions in the file
        functions = await self.parse_functions(code_file)
        validation_results = []
        
        for func in functions:
            validation = await self.validate_before_execution(
                f"def {func.name}({func.params}):",
                "python"
            )
            validation_results.append(validation)
        
        # Calculate confidence based on validation
        valid_count = sum(1 for v in validation_results if v["is_valid"])
        validation_rate = valid_count / len(validation_results) if validation_results else 0
        
        confidence_result = await self.check_confidence(
            "Creating tests",
            {"code_validation": {
                "all_references_valid": all(v["is_valid"] for v in validation_results),
                "validation_rate": validation_rate
            }}
        )
        
        if confidence_result["confidence_score"] < 0.75:
            return {
                "success": False,
                "error": f"Cannot create reliable tests. Confidence: {confidence_result['confidence_score']:.0%}"
            }
        
        # Create tests with validation
        return await self._generate_validated_tests(functions)
```

### 4. Security Auditor Agent

**Required Validations**:
- Security patterns in codebase
- Authentication methods exist
- Vulnerability patterns

**Implementation**:
```python
class SecurityAuditorAgent(BaseSpecializedAgent):
    async def audit_security(self, code: str) -> Dict:
        # Validate security-related references
        security_patterns = [
            "authentication", "authorization", "encryption",
            "validation", "sanitization", "jwt", "oauth"
        ]
        
        found_patterns = []
        for pattern in security_patterns:
            if pattern in code.lower():
                # Validate actual implementation exists
                validation = await self.validate_pattern_exists(pattern)
                if validation:
                    found_patterns.append(pattern)
        
        # Check confidence based on security understanding
        confidence_result = await self.check_confidence(
            "Security audit",
            {
                "security_patterns_found": len(found_patterns) > 0,
                "validation_rate": len(found_patterns) / len(security_patterns)
            }
        )
        
        if confidence_result["confidence_score"] < 0.75:
            return {
                "success": False,
                "message": "I don't have enough information to perform a comprehensive security audit. "
                          "Let's review your security requirements together."
            }
        
        return await self._perform_security_audit(code, found_patterns)
```

### 5. AntiHallucination Validator Agent

**Core Implementation**:
```python
class AntiHallucinationValidatorAgent(BaseSpecializedAgent):
    def __init__(self, project_root: str):
        super().__init__()
        self.validator = EnhancedAntiHallValidator(project_root)
        self.confidence_system = ConfidenceBasedResponseSystem()
    
    async def validate_code(self, code: str, language: str = "python") -> Dict:
        # Direct validation without additional checks
        is_valid, summary = self.validator.enforce_validation(
            code, language, min_confidence=0.75
        )
        
        return {
            "is_valid": is_valid,
            "summary": summary,
            "suggestions": summary.get("suggestions", []),
            "confidence": summary.get("average_confidence", 0.0),
            "message": self._format_validation_message(is_valid, summary)
        }
    
    def _format_validation_message(self, is_valid: bool, summary: Dict) -> str:
        if is_valid and summary["average_confidence"] >= 0.75:
            return "Code validation passed. All references exist and confidence is high."
        elif not is_valid:
            return f"Code contains {summary['invalid_references']} invalid references. " \
                   f"Critical errors: {'; '.join(summary['critical_errors'])}"
        else:
            return f"Confidence too low ({summary['average_confidence']:.0%}). " \
                   f"I don't know if this code will work. Let's review it together."
```

## Common Integration Patterns

### Pattern 1: Pre-Validation
```python
# Validate BEFORE any code generation
async def before_generation(self, task: str) -> bool:
    # Extract what we need to validate
    requirements = self.parse_requirements(task)
    
    # Validate each requirement
    for req in requirements:
        if not await self.validate_requirement_feasible(req):
            self.log_warning(f"Cannot fulfill requirement: {req}")
            return False
    
    return True
```

### Pattern 2: Confidence Gating
```python
# Gate operations based on confidence
async def confidence_gated_operation(self, operation: str, context: Dict) -> Dict:
    confidence = await self.check_confidence(operation, context)
    
    if confidence["confidence_score"] >= 0.90:
        # High confidence - proceed normally
        return await self.execute_operation(operation)
    elif confidence["confidence_score"] >= 0.75:
        # Moderate confidence - proceed with warnings
        result = await self.execute_operation(operation)
        result["warnings"] = confidence["uncertainties"]
        return result
    else:
        # Low confidence - don't proceed
        return {
            "success": False,
            "error": "I don't know how to proceed. Let's discuss this.",
            "confidence": confidence["confidence_score"],
            "uncertainties": confidence["uncertainties"]
        }
```

### Pattern 3: Progressive Validation
```python
# Validate progressively as we build
async def progressive_build(self, components: List[str]) -> Dict:
    built = []
    
    for component in components:
        # Validate before adding
        validation = await self.validate_component(component)
        
        if not validation["is_valid"]:
            # Stop and report what we've built so far
            return {
                "success": False,
                "partial_result": built,
                "failed_at": component,
                "error": validation["error"]
            }
        
        # Build and add
        result = await self.build_component(component)
        built.append(result)
    
    return {"success": True, "result": built}
```

### Pattern 4: Fallback Strategies
```python
# Provide fallbacks when validation fails
async def with_fallback(self, primary_approach: str) -> Dict:
    # Try primary approach
    validation = await self.validate_approach(primary_approach)
    
    if validation["is_valid"] and validation["confidence"] >= 0.75:
        return await self.execute_approach(primary_approach)
    
    # Try fallback approaches
    fallbacks = await self.generate_fallback_approaches(primary_approach)
    
    for fallback in fallbacks:
        validation = await self.validate_approach(fallback)
        if validation["is_valid"] and validation["confidence"] >= 0.75:
            return await self.execute_approach(fallback)
    
    # No valid approach found
    return {
        "success": False,
        "error": "I cannot find a valid approach. Let's work together to find a solution.",
        "attempted_approaches": [primary_approach] + fallbacks
    }
```

## Error Handling

### Validation Errors
```python
try:
    validation = await self.validate_before_execution(code, language)
except ValidationException as e:
    # Log but don't crash
    logger.error(f"Validation error: {e}")
    return {
        "success": False,
        "error": "Could not validate code. Proceeding with caution.",
        "validation_error": str(e)
    }
```

### Confidence Errors
```python
try:
    confidence = await self.check_confidence(task, context)
except ConfidenceException as e:
    # Default to low confidence
    logger.warning(f"Could not assess confidence: {e}")
    return {
        "success": False,
        "error": "Cannot determine confidence. I don't know if I can help with this."
    }
```

## Testing Your Integration

### Unit Tests
```python
import pytest
from unittest.mock import Mock, AsyncMock

@pytest.mark.asyncio
async def test_agent_validates_before_execution():
    # Mock validator
    validator = Mock()
    validator.enforce_validation = Mock(return_value=(False, {"invalid_references": 1}))
    
    # Create agent with validator
    agent = MyAgent()
    agent.validator = validator
    
    # Test that invalid code is rejected
    result = await agent.generate_code("use fake_function()")
    
    assert not result["success"]
    assert "Cannot find" in result["error"]
    validator.enforce_validation.assert_called_once()

@pytest.mark.asyncio
async def test_agent_respects_confidence_threshold():
    agent = MyAgent()
    
    # Mock low confidence
    agent.check_confidence = AsyncMock(return_value={
        "confidence_score": 0.6,
        "uncertainties": ["Missing documentation"]
    })
    
    result = await agent.process_request("complex task")
    
    assert not result["success"]
    assert "don't know" in result["error"].lower()
```

### Integration Tests
```python
@pytest.mark.asyncio
async def test_end_to_end_validation_flow():
    # Use real validator with test project
    validator = EnhancedAntiHallValidator("test_project/")
    agent = AgentValidationWrapper(MyAgent(), validator)
    
    # Test with code that references non-existent function
    result = await agent.generate_code(
        "Create function using imaginary_helper()"
    )
    
    assert not result["success"]
    assert result["validation_failed"]
    assert "imaginary_helper" in result["error"]
```

## Performance Considerations

### Caching
```python
# Cache validation results
class CachedValidationAgent(BaseSpecializedAgent):
    def __init__(self):
        super().__init__()
        self.validation_cache = {}
    
    async def validate_with_cache(self, reference: str) -> bool:
        if reference in self.validation_cache:
            return self.validation_cache[reference]
        
        result = await self.validate_before_execution(reference)
        self.validation_cache[reference] = result["is_valid"]
        return result["is_valid"]
```

### Batch Validation
```python
# Validate multiple references at once
async def batch_validate(self, references: List[str]) -> Dict:
    validation_tasks = [
        self.validate_before_execution(ref) 
        for ref in references
    ]
    
    results = await asyncio.gather(*validation_tasks)
    
    return {
        "all_valid": all(r["is_valid"] for r in results),
        "results": results,
        "invalid_count": sum(1 for r in results if not r["is_valid"])
    }
```

## Monitoring and Metrics

### Track Validation Statistics
```python
class MonitoredAgent(BaseSpecializedAgent):
    def __init__(self):
        super().__init__()
        self.stats = {
            "validations_performed": 0,
            "validations_passed": 0,
            "hallucinations_prevented": 0,
            "low_confidence_blocks": 0
        }
    
    async def validate_and_track(self, code: str) -> Dict:
        self.stats["validations_performed"] += 1
        
        result = await self.validate_before_execution(code)
        
        if result["is_valid"]:
            self.stats["validations_passed"] += 1
        else:
            self.stats["hallucinations_prevented"] += 1
        
        if result.get("confidence_score", 1.0) < 0.75:
            self.stats["low_confidence_blocks"] += 1
        
        return result
    
    def get_statistics(self) -> Dict:
        return {
            **self.stats,
            "success_rate": self.stats["validations_passed"] / max(self.stats["validations_performed"], 1),
            "prevention_rate": self.stats["hallucinations_prevented"] / max(self.stats["validations_performed"], 1)
        }
```

## Best Practices

1. **Always Validate First**: Never generate code without validation
2. **Respect Confidence Threshold**: Never proceed below 75% confidence
3. **Provide Clear Feedback**: Explain why validation failed
4. **Suggest Alternatives**: Use "Did you mean...?" suggestions
5. **Track Metrics**: Monitor hallucinations prevented
6. **Cache When Possible**: Avoid redundant validations
7. **Fail Gracefully**: Don't crash on validation errors
8. **Be Honest**: Say "I don't know" when uncertain

## Conclusion

Integrating the anti-hallucination system into your agents ensures:
- No hallucinated code references
- Honest communication when uncertain
- Higher quality, more reliable outputs
- Better user trust and experience

Remember: **It's always better to say "I don't know" than to hallucinate a solution.**