#!/usr/bin/env python3
"""
AGENT VALIDATION ENFORCER
Mandatory validation workflow that ALL specialized agents must execute

This script enforces the hardcoded global rule for documentation-driven development:
- Tests must be created from PRD/PRP/ADR docs BEFORE implementation
- All agents must call this enforcer before beginning any development work
- Development is BLOCKED if validation fails

CRITICAL: This cannot be bypassed by any agent or system
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Import validation modules
from ..validation.doc_driven_validator import validate_doc_driven_development
from ..validation.archon_validation_rules import verify_system_integrity
from ..validation.dgts_validator import validate_dgts_compliance
from ..validation.agent_behavior_monitor import check_agent_blocked, monitor_agent_action, ActionType

logger = logging.getLogger(__name__)

class AgentValidationEnforcer:
    """Enforces validation rules for all specialized agents"""
    
    def __init__(self, project_path: str = "."):
        self.project_path = project_path
        self.validation_state: Optional[Dict[str, Any]] = None
        
        # Verify system integrity on initialization
        try:
            verify_system_integrity()
        except RuntimeError as e:
            raise RuntimeError(f"VALIDATION SYSTEM COMPROMISED: {e}")
    
    def enforce_pre_development_validation(self, agent_name: str, task_description: str) -> Dict[str, Any]:
        """
        MANDATORY: Must be called by ALL agents before any development work
        
        Args:
            agent_name: Name of the calling agent
            task_description: Description of the task to be performed
            
        Returns:
            Validation result with compliance status
            
        Raises:
            RuntimeError: If validation fails and development must be blocked
        """
        
        logger.info(f"Agent {agent_name} requesting pre-development validation for: {task_description}")
        
        # Step 0: Check if agent is blocked due to previous gaming behavior
        block_status = check_agent_blocked(agent_name, self.project_path)
        if block_status.get("blocked", False):
            error_msg = f"""
            AGENT BLOCKED: {agent_name} is currently blocked due to gaming behavior
            
            Blocked at: {block_status.get('blocked_at', 'Unknown')}
            Reason: {block_status.get('reason', 'Gaming behavior detected')}
            Auto-unblock at: {block_status.get('auto_unblock_at', 'Manual unblock required')}
            
            Gaming indicators: {block_status.get('gaming_indicators', [])}
            
            Agent must correct behavior and be manually unblocked before proceeding.
            """
            
            logger.error(error_msg)
            raise RuntimeError(f"AGENT BLOCKED: {agent_name} blocked due to gaming behavior")
        
        # Step 1: Document-driven validation
        doc_validation = validate_doc_driven_development(self.project_path)
        
        # Step 2: DGTS (Don't Game The System) validation  
        dgts_validation = validate_dgts_compliance(self.project_path)
        
        # Step 3: Check for bypass attempts
        self._detect_bypass_attempts(agent_name, task_description)
        
        # Step 4: Enforce blocking validation
        all_compliant = (doc_validation.get("compliant", False) and 
                        dgts_validation.get("compliant", False))
        
        if not all_compliant:
            violations = []
            violations.extend(doc_validation.get("violations", []))
            
            # Add DGTS violations
            dgts_violations = dgts_validation.get("dgts_result", {}).get("violations", [])
            violations.extend(dgts_violations)
            
            critical_violations = (doc_validation.get("critical_violations", 0) + 
                                 len([v for v in dgts_violations if v.get("severity") == "critical"]))
            
            error_msg = f"""
            DEVELOPMENT BLOCKED: Validation failures detected
            
            Agent: {agent_name}
            Task: {task_description}
            Critical Violations: {critical_violations}
            
            Document-driven violations: {len(doc_validation.get("violations", []))}
            DGTS (gaming) violations: {len(dgts_violations)}
            Gaming score: {dgts_validation.get("dgts_result", {}).get("gaming_score", 0.0):.2f}
            
            Violations:
            {json.dumps(violations, indent=2)}
            
            REQUIRED ACTIONS:
            {json.dumps(doc_validation.get('remediation_steps', []), indent=2)}
            
            Development cannot proceed until ALL violations are resolved.
            """
            
            logger.error(error_msg)
            raise RuntimeError(f"VALIDATION BLOCKED: {agent_name} cannot proceed due to critical violations")
        
        # Step 5: Log successful validation
        logger.info(f"✅ VALIDATION PASSED: {agent_name} cleared for development")
        
        self.validation_state = {
            "doc_validation": doc_validation,
            "dgts_validation": dgts_validation,
            "combined_compliant": True
        }
        
        return {
            "validation_passed": True,
            "agent": agent_name,
            "task": task_description,
            "requirements_found": doc_validation.get("requirements_found", 0),
            "gaming_score": dgts_validation.get("dgts_result", {}).get("gaming_score", 0.0),
            "message": "Agent cleared for development - all validation rules satisfied including DGTS compliance"
        }
    
    def _detect_bypass_attempts(self, agent_name: str, task_description: str):
        """Detect attempts to bypass validation"""
        
        bypass_indicators = [
            "skip", "bypass", "ignore", "override", "disable",
            "quick", "dirty", "temp", "hack", "workaround"
        ]
        
        task_lower = task_description.lower()
        
        for indicator in bypass_indicators:
            if indicator in task_lower:
                logger.warning(f"⚠️  BYPASS ATTEMPT DETECTED: {agent_name} task contains '{indicator}'")
                # Allow with warning for now, but could be made blocking
        
        # Check for direct attempts to modify validation code
        if any(word in task_lower for word in ["validation", "enforce", "rule", "policy"]):
            if any(word in task_lower for word in ["modify", "change", "update", "disable", "remove"]):
                raise RuntimeError(f"BLOCKED: {agent_name} attempting to modify validation system")
    
    def validate_post_development(self, agent_name: str, changes_made: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that development followed documentation-driven approach
        
        Args:
            agent_name: Name of the agent that performed development
            changes_made: Dictionary describing what was implemented
                Expected format: {
                    "files_modified": [{"path": "...", "before": "...", "after": "..."}],
                    "explanation": "What was changed and why"
                }
            
        Returns:
            Post-development validation result
        """
        
        logger.info(f"Post-development validation for {agent_name}")
        
        # Step 1: Record agent actions for behavior monitoring
        files_modified = changes_made.get("files_modified", [])
        explanation = changes_made.get("explanation", "No explanation provided")
        
        for file_change in files_modified:
            file_path = file_change.get("path", "unknown")
            content_before = file_change.get("before", "")
            content_after = file_change.get("after", "")
            
            # Determine action type based on file content
            action_type = ActionType.CODE_CHANGE
            if "test" in file_path.lower():
                if not content_before:
                    action_type = ActionType.TEST_CREATION
                else:
                    action_type = ActionType.TEST_MODIFICATION
            elif not content_before:
                action_type = ActionType.FEATURE_IMPLEMENTATION
            
            # Record action with behavior monitor
            monitor_result = monitor_agent_action(
                agent_name=agent_name,
                action_type=action_type,
                file_path=file_path,
                content_before=content_before,
                content_after=content_after,
                explanation=explanation,
                project_path=self.project_path
            )
            
            # Check if agent was blocked due to this action
            if monitor_result.get("blocked", False):
                return {
                    "post_validation_passed": False,
                    "agent": agent_name,
                    "blocked": True,
                    "block_reason": monitor_result.get("message", "Gaming behavior detected"),
                    "gaming_indicators": monitor_result.get("gaming_indicators", []),
                    "message": f"Agent {agent_name} was blocked due to gaming behavior during development"
                }
        
        # Step 2: Re-run validation to check current state
        current_doc_validation = validate_doc_driven_development(self.project_path)
        current_dgts_validation = validate_dgts_compliance(self.project_path)
        
        # Step 3: Check combined compliance
        all_compliant = (current_doc_validation.get("compliant", False) and 
                        current_dgts_validation.get("compliant", False))
        
        # Check if agent followed the approved approach
        if not all_compliant:
            violations = []
            violations.extend(current_doc_validation.get("violations", []))
            violations.extend(current_dgts_validation.get("dgts_result", {}).get("violations", []))
            
            logger.error(f"❌ POST-VALIDATION FAILED: {agent_name} development resulted in non-compliance")
            
            return {
                "post_validation_passed": False,
                "agent": agent_name,
                "violations": violations,
                "gaming_score": current_dgts_validation.get("dgts_result", {}).get("gaming_score", 0.0),
                "remediation_required": True,
                "message": "Development must be corrected to meet validation requirements including DGTS compliance"
            }
        
        logger.info(f"✅ POST-VALIDATION PASSED: {agent_name} development maintains compliance")
        
        return {
            "post_validation_passed": True,
            "agent": agent_name,
            "changes_validated": True,
            "gaming_score": current_dgts_validation.get("dgts_result", {}).get("gaming_score", 0.0),
            "message": "Development successfully completed with full compliance and no gaming detected"
        }
    
    def get_validation_status(self) -> Dict[str, Any]:
        """Get current validation status"""
        
        current_validation = validate_doc_driven_development(self.project_path)
        
        return {
            "system_integrity": "OK",
            "validation_state": current_validation,
            "last_validation": self.validation_state,
            "project_path": str(self.project_path)
        }

def enforce_agent_validation(agent_name: str, task_description: str, project_path: str = ".") -> Dict[str, Any]:
    """
    MANDATORY FUNCTION: All specialized agents must call this before development
    
    This function enforces the hardcoded global rule for documentation-driven development.
    It cannot be bypassed or disabled.
    
    Usage in agents:
        from agents.workflow.agent_validation_enforcer import enforce_agent_validation
        
        def agent_task(self, task):
            # MANDATORY: Validate before any work
            validation_result = enforce_agent_validation("code-implementer", "Implement user auth")
            
            if not validation_result["validation_passed"]:
                raise RuntimeError("Cannot proceed - validation blocked development")
            
            # Now safe to proceed with development
            ...
    """
    
    enforcer = AgentValidationEnforcer(project_path)
    return enforcer.enforce_pre_development_validation(agent_name, task_description)

def validate_post_development(agent_name: str, changes_made: Dict[str, Any], project_path: str = ".") -> Dict[str, Any]:
    """
    MANDATORY: Validate after development is complete
    """
    
    enforcer = AgentValidationEnforcer(project_path)
    return enforcer.validate_post_development(agent_name, changes_made)

# CLI interface for direct validation
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python agent_validation_enforcer.py <agent_name> <task_description> [project_path]")
        sys.exit(1)
    
    agent_name = sys.argv[1]
    task_description = sys.argv[2]
    project_path = sys.argv[3] if len(sys.argv) > 3 else "."
    
    try:
        result = enforce_agent_validation(agent_name, task_description, project_path)
        print("✅ VALIDATION PASSED")
        print(json.dumps(result, indent=2))
    except RuntimeError as e:
        print("❌ VALIDATION BLOCKED")
        print(str(e))
        sys.exit(1)