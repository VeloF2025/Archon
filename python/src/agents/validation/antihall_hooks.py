#!/usr/bin/env python3
"""
Anti-Hallucination Validation Hooks

Automated hooks that prevent code hallucinations and enforce the 75% confidence rule
across all agent operations.
"""

import asyncio
import functools
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

from .enhanced_antihall_validator import EnhancedAntiHallValidator, ValidationResult
from .confidence_based_responses import ConfidenceBasedResponseSystem

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AntiHallHooks:
    """Anti-hallucination validation hooks"""
    
    def __init__(self, project_root: str = ".", min_confidence: float = 0.75):
        self.project_root = project_root
        self.min_confidence = min_confidence
        self.validator = EnhancedAntiHallValidator(project_root)
        self.confidence_system = ConfidenceBasedResponseSystem(min_confidence)
        self.enabled = True
        self.stats = {
            "validations": 0,
            "passed": 0,
            "failed": 0,
            "hallucinations_blocked": 0,
            "low_confidence_blocks": 0
        }
    
    async def pre_code_generation(self, agent_id: str, task: str, context: Dict) -> Dict[str, Any]:
        """Hook before code generation - validates requirements are achievable"""
        if not self.enabled:
            return {"proceed": True}
        
        self.stats["validations"] += 1
        
        # Extract references from task description
        potential_refs = self._extract_task_references(task)
        
        invalid_refs = []
        for ref in potential_refs:
            report = self.validator.validate_reference(
                self.validator.create_reference("any", ref)
            )
            if report.result == ValidationResult.NOT_FOUND:
                invalid_refs.append({
                    "name": ref,
                    "suggestion": report.suggestion
                })
        
        if invalid_refs:
            self.stats["failed"] += 1
            self.stats["hallucinations_blocked"] += 1
            
            return {
                "proceed": False,
                "error": "Task references non-existent code",
                "invalid_references": invalid_refs,
                "message": f"I cannot find {invalid_refs[0]['name']} in the codebase. "
                          f"{invalid_refs[0].get('suggestion', 'Please clarify what you need.')}"
            }
        
        # Check confidence
        confidence_context = {
            "code_validation": {
                "all_references_valid": len(invalid_refs) == 0,
                "validation_rate": 1.0 if not potential_refs else 
                                  (len(potential_refs) - len(invalid_refs)) / len(potential_refs)
            },
            "task_complexity": self._assess_task_complexity(task)
        }
        
        assessment = self.confidence_system.assess_confidence(confidence_context)
        
        if assessment.confidence_score < self.min_confidence:
            self.stats["low_confidence_blocks"] += 1
            
            return {
                "proceed": False,
                "error": "Confidence too low",
                "confidence": assessment.confidence_score,
                "message": f"I'm only {assessment.confidence_score:.0%} confident about this task. "
                          f"I don't know how to proceed. Let's figure this out together.",
                "uncertainties": assessment.uncertainties
            }
        
        self.stats["passed"] += 1
        
        return {
            "proceed": True,
            "confidence": assessment.confidence_score,
            "validated_references": len(potential_refs) - len(invalid_refs)
        }
    
    async def post_code_generation(self, agent_id: str, code: str, language: str = "python") -> Dict[str, Any]:
        """Hook after code generation - validates generated code"""
        if not self.enabled:
            return {"valid": True}
        
        self.stats["validations"] += 1
        
        # Validate the generated code
        is_valid, summary = self.validator.enforce_validation(code, language, self.min_confidence)
        
        if not is_valid:
            self.stats["failed"] += 1
            self.stats["hallucinations_blocked"] += summary.get("invalid_references", 0)
            
            return {
                "valid": False,
                "error": "Generated code contains invalid references",
                "summary": summary,
                "message": f"The generated code has {summary['invalid_references']} invalid references. "
                          f"Errors: {'; '.join(summary['critical_errors'][:2])}",
                "suggestions": summary.get("suggestions", [])
            }
        
        # Check confidence of generated code
        if summary.get("average_confidence", 0) < self.min_confidence:
            self.stats["low_confidence_blocks"] += 1
            
            return {
                "valid": False,
                "error": "Low confidence in generated code",
                "confidence": summary["average_confidence"],
                "message": f"Confidence in generated code is only {summary['average_confidence']:.0%}. "
                          f"This code might not work as expected."
            }
        
        self.stats["passed"] += 1
        
        return {
            "valid": True,
            "confidence": summary.get("average_confidence", 1.0),
            "validated_references": summary.get("valid_references", 0)
        }
    
    async def validate_agent_response(self, agent_id: str, response: str, context: Dict) -> Dict[str, Any]:
        """Validate complete agent response including code and text"""
        if not self.enabled:
            return {"valid": True}
        
        self.stats["validations"] += 1
        
        # Check for uncertainty patterns in response
        uncertainty_detected = self._detect_uncertainty(response)
        
        # Extract and validate code blocks
        code_blocks = self._extract_code_blocks(response)
        code_validation_results = []
        
        for code, language in code_blocks:
            is_valid, summary = self.validator.enforce_validation(code, language)
            code_validation_results.append({
                "valid": is_valid,
                "summary": summary
            })
        
        # Calculate overall confidence
        all_code_valid = all(r["valid"] for r in code_validation_results)
        avg_code_confidence = (
            sum(r["summary"].get("average_confidence", 0) for r in code_validation_results) / 
            len(code_validation_results)
        ) if code_validation_results else 1.0
        
        # Assess response confidence
        response_confidence = self.confidence_system.assess_confidence({
            "code_validation": {
                "all_references_valid": all_code_valid,
                "validation_rate": avg_code_confidence
            },
            "uncertainty_detected": uncertainty_detected,
            "contains_code": len(code_blocks) > 0
        })
        
        if not all_code_valid:
            self.stats["failed"] += 1
            self.stats["hallucinations_blocked"] += sum(
                r["summary"].get("invalid_references", 0) 
                for r in code_validation_results if not r["valid"]
            )
            
            return {
                "valid": False,
                "error": "Response contains invalid code",
                "code_validation_results": code_validation_results,
                "message": "The response contains code with invalid references."
            }
        
        if response_confidence.confidence_score < self.min_confidence:
            self.stats["low_confidence_blocks"] += 1
            
            return {
                "valid": False,
                "error": "Low confidence response",
                "confidence": response_confidence.confidence_score,
                "message": f"Response confidence ({response_confidence.confidence_score:.0%}) "
                          f"is below threshold. Response should be revised."
            }
        
        self.stats["passed"] += 1
        
        return {
            "valid": True,
            "confidence": response_confidence.confidence_score,
            "code_blocks_validated": len(code_blocks)
        }
    
    def _extract_task_references(self, task: str) -> List[str]:
        """Extract potential code references from task description"""
        import re
        
        # Look for code-like references
        patterns = [
            r"\b([A-Z][a-zA-Z0-9]+(?:Service|Manager|Controller|Helper|Utils?))\b",  # Classes
            r"\b([a-z_][a-z0-9_]+)\(\)",  # Function calls
            r"from\s+(\S+)\s+import",  # Import statements
            r"\`([^\`]+)\`",  # Backtick references
        ]
        
        references = []
        for pattern in patterns:
            matches = re.findall(pattern, task)
            references.extend(matches)
        
        return list(set(references))
    
    def _assess_task_complexity(self, task: str) -> Dict[str, Any]:
        """Assess complexity of the task"""
        complexity_indicators = {
            "complex_words": ["integrate", "refactor", "optimize", "implement", "architect"],
            "simple_words": ["add", "fix", "update", "change", "modify"],
            "uncertainty_words": ["might", "maybe", "possibly", "could", "should"]
        }
        
        task_lower = task.lower()
        
        return {
            "has_complex_requirements": any(w in task_lower for w in complexity_indicators["complex_words"]),
            "has_simple_requirements": any(w in task_lower for w in complexity_indicators["simple_words"]),
            "has_uncertainty": any(w in task_lower for w in complexity_indicators["uncertainty_words"]),
            "length": len(task.split())
        }
    
    def _detect_uncertainty(self, response: str) -> bool:
        """Detect uncertainty patterns in response"""
        uncertainty_patterns = [
            r"\bI think\b", r"\bI believe\b", r"\bmight\b", r"\bmaybe\b",
            r"\bprobably\b", r"\bpossibly\b", r"\bnot sure\b", r"\bcould be\b"
        ]
        
        import re
        response_lower = response.lower()
        
        return any(re.search(pattern, response_lower) for pattern in uncertainty_patterns)
    
    def _extract_code_blocks(self, response: str) -> List[tuple]:
        """Extract code blocks from response"""
        import re
        
        # Match markdown code blocks
        pattern = r"```(\w+)?\n([^`]+)```"
        matches = re.findall(pattern, response)
        
        code_blocks = []
        for language, code in matches:
            if not language:
                language = "python"  # Default
            code_blocks.append((code.strip(), language))
        
        return code_blocks
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get hook statistics"""
        total = self.stats["validations"]
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            "success_rate": self.stats["passed"] / total,
            "hallucination_block_rate": self.stats["hallucinations_blocked"] / total,
            "confidence_block_rate": self.stats["low_confidence_blocks"] / total
        }


def require_validation(min_confidence: float = 0.75):
    """Decorator to require validation for agent methods"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs) -> T:
            # Ensure agent has validation hooks
            if not hasattr(self, '_antihall_hooks'):
                self._antihall_hooks = AntiHallHooks(
                    project_root=getattr(self, 'project_root', '.'),
                    min_confidence=min_confidence
                )
            
            # Pre-validation for code generation methods
            if func.__name__ in ['generate_code', 'create_code', 'implement']:
                task = args[0] if args else kwargs.get('task', '')
                context = kwargs.get('context', {})
                
                validation = await self._antihall_hooks.pre_code_generation(
                    getattr(self, 'agent_id', 'unknown'),
                    task,
                    context
                )
                
                if not validation["proceed"]:
                    return {
                        "success": False,
                        "error": validation["error"],
                        "message": validation["message"]
                    }
            
            # Execute function
            result = await func(self, *args, **kwargs)
            
            # Post-validation for results containing code
            if isinstance(result, dict) and 'code' in result:
                validation = await self._antihall_hooks.post_code_generation(
                    getattr(self, 'agent_id', 'unknown'),
                    result['code'],
                    result.get('language', 'python')
                )
                
                if not validation["valid"]:
                    result["success"] = False
                    result["validation_error"] = validation["error"]
                    result["validation_message"] = validation["message"]
            
            return result
        
        return wrapper
    
    return decorator


# Integration with existing validation hooks
def integrate_with_validation_hooks(existing_hooks):
    """Integrate anti-hallucination with existing validation hooks"""
    
    antihall_hooks = AntiHallHooks()
    
    # Wrap existing pre-execution hook
    original_pre_execution = existing_hooks.pre_execution_hook
    
    def enhanced_pre_execution(agent_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        # Run original validation
        result = original_pre_execution(agent_id, task)
        
        if not result["proceed"]:
            return result
        
        # Run anti-hallucination validation
        task_description = task.get("description", "")
        context = task.get("context", {})
        
        asyncio.run(
            antihall_validation := antihall_hooks.pre_code_generation(
                agent_id, task_description, context
            )
        )
        
        if not antihall_validation["proceed"]:
            result["proceed"] = False
            result["errors"].append(antihall_validation["error"])
            result["metadata"]["antihall_validation"] = antihall_validation
        
        return result
    
    existing_hooks.pre_execution_hook = enhanced_pre_execution
    
    return existing_hooks