"""
Validation Hooks for Agent Execution Pipeline

These hooks integrate DGTS and NLNH validation into the agent execution flow,
ensuring all agent actions are validated in real-time.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Callable
from functools import wraps

from .dgts_validator import DGTSValidator
from .agent_behavior_monitor import AgentBehaviorMonitor

logger = logging.getLogger(__name__)


class ValidationHooks:
    """
    Hooks for integrating validation into agent execution
    """
    
    def __init__(self):
        self.dgts_validator = DGTSValidator(project_path=".")  # Use current directory as project path
        self.behavior_monitor = AgentBehaviorMonitor(project_path=".")  # Use current directory as project path
        self.validation_enabled = True
        self.strict_mode = True  # Fail on validation errors
        
    def pre_execution_hook(self, agent_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook to run before agent execution
        Validates task and checks if agent is blocked
        """
        validation_result = {
            "proceed": True,
            "warnings": [],
            "errors": [],
            "metadata": {}
        }
        
        # Check if agent is blocked
        if self.behavior_monitor.is_agent_blocked(agent_id):
            validation_result["proceed"] = False
            validation_result["errors"].append(f"Agent {agent_id} is blocked due to gaming behavior")
            return validation_result
        
        # Validate task description for gaming patterns
        task_description = task.get("description", "")
        if task_description:
            gaming_score = self.dgts_validator.calculate_gaming_score(task_description)
            if gaming_score > 0.3:
                validation_result["warnings"].append(f"Task description contains gaming patterns (score: {gaming_score})")
                validation_result["metadata"]["gaming_score"] = gaming_score
        
        # Track agent starting task
        action = {
            "timestamp": time.time(),
            "action_type": "task_start",
            "target": task.get("id", "unknown"),
            "details": {"task": task}
        }
        self.behavior_monitor.track_agent_action(agent_id, action)
        
        return validation_result
    
    def post_execution_hook(self, agent_id: str, task: Dict[str, Any], result: Any) -> Dict[str, Any]:
        """
        Hook to run after agent execution
        Validates results and detects gaming
        """
        validation_result = {
            "valid": True,
            "modified_result": result,
            "warnings": [],
            "errors": [],
            "metadata": {}
        }
        
        # Convert result to string for validation
        result_str = str(result)
        
        # Check for gaming patterns in result
        violations = self.dgts_validator.validate_code(result_str)
        gaming_score = self.dgts_validator.calculate_gaming_score(result_str)
        
        if violations:
            validation_result["warnings"].append(f"Result contains {len(violations)} DGTS violations")
            validation_result["metadata"]["violations"] = violations
        
        if gaming_score > 0.3:
            validation_result["warnings"].append(f"High gaming score detected: {gaming_score}")
            validation_result["metadata"]["gaming_score"] = gaming_score
            
            # In strict mode, invalidate results with high gaming scores
            if self.strict_mode and gaming_score > 0.5:
                validation_result["valid"] = False
                validation_result["errors"].append("Result rejected due to excessive gaming patterns")
        
        # Track agent completing task
        action = {
            "timestamp": time.time(),
            "action_type": "task_complete",
            "target": task.get("id", "unknown"),
            "gaming_score": gaming_score,
            "violations": len(violations)
        }
        
        suspicious = self.behavior_monitor.track_agent_action(agent_id, action)
        
        # Block agent if suspicious behavior detected
        if suspicious and gaming_score > 0.7:
            self.behavior_monitor.block_agent(agent_id, reason="Repeated gaming behavior detected")
            validation_result["errors"].append(f"Agent {agent_id} has been blocked")
        
        return validation_result
    
    def code_modification_hook(self, agent_id: str, file_path: str, old_content: str, new_content: str) -> Dict[str, Any]:
        """
        Hook for validating code modifications
        """
        validation_result = {
            "allow_modification": True,
            "warnings": [],
            "errors": [],
            "suggestions": []
        }
        
        # Validate new content for gaming patterns
        violations = self.dgts_validator.validate_code(new_content)
        gaming_score = self.dgts_validator.calculate_gaming_score(new_content)
        
        if violations:
            for violation in violations:
                if violation["type"].startswith("DGTS_"):
                    validation_result["warnings"].append(f"{violation['type']}: {violation['message']}")
                    
                    # Suggest fixes
                    if "TEST_GAMING" in violation["type"]:
                        validation_result["suggestions"].append("Replace mock tests with real functionality validation")
                    elif "CODE_GAMING" in violation["type"]:
                        validation_result["suggestions"].append("Uncomment validation rules and implement properly")
                    elif "FEATURE_FAKING" in violation["type"]:
                        validation_result["suggestions"].append("Implement real functionality instead of mocks")
        
        # Block modification if gaming score is too high
        if gaming_score > 0.7:
            validation_result["allow_modification"] = False
            validation_result["errors"].append(f"Code modification blocked due to high gaming score: {gaming_score}")
        
        # Track modification
        action = {
            "timestamp": time.time(),
            "action_type": "code_modification",
            "target": file_path,
            "gaming_score": gaming_score,
            "violations": len(violations)
        }
        self.behavior_monitor.track_agent_action(agent_id, action)
        
        return validation_result
    
    def response_validation_hook(self, agent_id: str, response: Any) -> Dict[str, Any]:
        """
        Hook for validating agent responses (NLNH)
        """
        from ...server.middleware.validation_middleware import NLNHValidator
        
        nlnh_validator = NLNHValidator()
        nlnh_result = nlnh_validator.validate_response(response)
        
        validation_result = {
            "valid": nlnh_result["valid"],
            "confidence": nlnh_result["confidence"],
            "warnings": nlnh_result["issues"],
            "corrections": nlnh_result["corrections"],
            "metadata": {"agent_id": agent_id}
        }
        
        # Track low confidence responses
        if nlnh_result["confidence"] < 0.7:
            action = {
                "timestamp": time.time(),
                "action_type": "low_confidence_response",
                "confidence": nlnh_result["confidence"],
                "issues": nlnh_result["issues"]
            }
            self.behavior_monitor.track_agent_action(agent_id, action)
        
        return validation_result


def with_validation(agent_id: str):
    """
    Decorator to add validation to agent functions
    """
    hooks = ValidationHooks()
    
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Pre-execution validation
            task = kwargs.get("task", {})
            pre_result = hooks.pre_execution_hook(agent_id, task)
            
            if not pre_result["proceed"]:
                raise ValueError(f"Validation failed: {pre_result['errors']}")
            
            # Execute function
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                # Track execution failure
                action = {
                    "timestamp": time.time(),
                    "action_type": "execution_failure",
                    "error": str(e)
                }
                hooks.behavior_monitor.track_agent_action(agent_id, action)
                raise
            
            # Post-execution validation
            post_result = hooks.post_execution_hook(agent_id, task, result)
            
            if not post_result["valid"]:
                raise ValueError(f"Result validation failed: {post_result['errors']}")
            
            # Response validation (NLNH)
            response_result = hooks.response_validation_hook(agent_id, result)
            
            if not response_result["valid"]:
                logger.warning(f"Low confidence response from {agent_id}: {response_result['confidence']}")
                # Add warnings to result
                if isinstance(result, dict):
                    result["_validation_warnings"] = response_result["warnings"]
                    result["_validation_confidence"] = response_result["confidence"]
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Pre-execution validation
            task = kwargs.get("task", {})
            pre_result = hooks.pre_execution_hook(agent_id, task)
            
            if not pre_result["proceed"]:
                raise ValueError(f"Validation failed: {pre_result['errors']}")
            
            # Execute function
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                # Track execution failure
                action = {
                    "timestamp": time.time(),
                    "action_type": "execution_failure",
                    "error": str(e)
                }
                hooks.behavior_monitor.track_agent_action(agent_id, action)
                raise
            
            # Post-execution validation
            post_result = hooks.post_execution_hook(agent_id, task, result)
            
            if not post_result["valid"]:
                raise ValueError(f"Result validation failed: {post_result['errors']}")
            
            # Response validation (NLNH)
            response_result = hooks.response_validation_hook(agent_id, result)
            
            if not response_result["valid"]:
                logger.warning(f"Low confidence response from {agent_id}: {response_result['confidence']}")
                # Add warnings to result
                if isinstance(result, dict):
                    result["_validation_warnings"] = response_result["warnings"]
                    result["_validation_confidence"] = response_result["confidence"]
            
            return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global hooks instance for direct use
global_hooks = ValidationHooks()


def validate_agent_action(agent_id: str, action_type: str, data: Any) -> bool:
    """
    Validate a single agent action
    Returns True if action is allowed, False otherwise
    """
    # Check if agent is blocked
    if global_hooks.behavior_monitor.is_agent_blocked(agent_id):
        logger.warning(f"Blocked agent {agent_id} attempted action: {action_type}")
        return False
    
    # Track action
    action = {
        "timestamp": time.time(),
        "action_type": action_type,
        "data": data
    }
    
    suspicious = global_hooks.behavior_monitor.track_agent_action(agent_id, action)
    
    if suspicious:
        logger.warning(f"Suspicious behavior detected for agent {agent_id}")
    
    return True


def get_validation_status() -> Dict[str, Any]:
    """
    Get current validation system status
    """
    return {
        "validation_enabled": global_hooks.validation_enabled,
        "strict_mode": global_hooks.strict_mode,
        "blocked_agents": list(global_hooks.behavior_monitor.blocked_agents.keys()),
        "total_agents_monitored": len(global_hooks.behavior_monitor.agent_history)
    }