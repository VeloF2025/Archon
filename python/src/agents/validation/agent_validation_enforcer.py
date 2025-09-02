#!/usr/bin/env python3
"""
AGENT VALIDATION ENFORCER
Mandatory workflow enforcement for all specialized agents

This enforcer ensures ALL agents follow documentation-driven development:
1. Check if agent is blocked for gaming behavior
2. Validate documentation exists (PRD/PRP/ADR)
3. Enforce test creation from docs BEFORE implementation
4. Monitor agent behavior for DGTS violations
5. Block agents that game the system

CRITICAL: This is a blocking system - agents cannot proceed if violations detected
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Import validation modules
try:
    from .doc_driven_validator import validate_doc_driven_development
    from .dgts_validator import DGTSValidator, GamingViolation
    from .agent_behavior_monitor import AgentBehaviorMonitor
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from agents.validation.doc_driven_validator import validate_doc_driven_development
    from agents.validation.dgts_validator import DGTSValidator, GamingViolation
    from agents.validation.agent_behavior_monitor import AgentBehaviorMonitor

logger = logging.getLogger(__name__)

@dataclass
class AgentValidationResult:
    """Result of agent validation check"""
    agent_name: str
    task_description: str
    validation_passed: bool
    blocked: bool
    errors: List[str]
    warnings: List[str]
    gaming_score: float
    doc_compliance: bool
    test_coverage: float
    remediation_steps: List[str]

class AgentValidationEnforcer:
    """Enforces validation rules for all specialized agents"""
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path)
        self.behavior_monitor = AgentBehaviorMonitor()
        self.dgts_validator = DGTSValidator(project_path)
        
    def enforce_agent_validation(self, agent_name: str, task_description: str) -> AgentValidationResult:
        """
        Main enforcement method - validates agent can proceed with task
        
        Args:
            agent_name: Name of the specialized agent
            task_description: Description of the task to be performed
            
        Returns:
            AgentValidationResult with validation status and remediation steps
        """
        logger.info(f"🔍 Enforcing validation for agent '{agent_name}' - Task: {task_description[:100]}...")
        
        errors = []
        warnings = []
        remediation_steps = []
        
        # Step 1: Check if agent is blocked for gaming behavior
        blocked_status = self.behavior_monitor.check_agent_blocked(agent_name)
        if blocked_status["blocked"]:
            return AgentValidationResult(
                agent_name=agent_name,
                task_description=task_description,
                validation_passed=False,
                blocked=True,
                errors=[f"Agent blocked for gaming behavior: {blocked_status['reason']}"],
                warnings=[],
                gaming_score=blocked_status["gaming_score"],
                doc_compliance=False,
                test_coverage=0.0,
                remediation_steps=[
                    f"Agent is blocked until {blocked_status['unblock_time']}",
                    "Review and fix gaming violations before proceeding",
                    "Wait for automatic unblock or manual review"
                ]
            )
        
        # Step 2: Validate documentation-driven development
        doc_validation = validate_doc_driven_development(str(self.project_path))
        doc_compliant = doc_validation.get("compliant", False)
        
        if not doc_compliant:
            errors.append("Documentation-driven development validation failed")
            remediation_steps.extend(doc_validation.get("remediation_steps", []))
        
        # Step 3: Check for existing gaming patterns before allowing development
        gaming_score = self.dgts_validator.calculate_gaming_score()
        if gaming_score > 0.3:
            errors.append(f"High gaming score detected: {gaming_score:.2f} (max allowed: 0.3)")
            remediation_steps.append("Remove gaming patterns from codebase before proceeding")
        
        # Step 4: Validate task is appropriate for agent
        task_validation = self._validate_task_for_agent(agent_name, task_description)
        if not task_validation["valid"]:
            warnings.append(f"Task may not be suitable for {agent_name}: {task_validation['reason']}")
        
        # Step 5: Check if tests exist for existing features
        test_coverage = self._calculate_test_coverage()
        if test_coverage < 0.8 and "implement" in task_description.lower():
            warnings.append(f"Low test coverage ({test_coverage:.1%}) - create tests before new implementation")
            remediation_steps.append("Increase test coverage to >80% before implementing new features")
        
        # Step 6: Register agent activity for monitoring
        self.behavior_monitor.track_agent_activity(
            agent_name=agent_name,
            action="validation_check",
            details={
                "task": task_description[:200],
                "doc_compliant": doc_compliant,
                "gaming_score": gaming_score,
                "test_coverage": test_coverage
            }
        )
        
        # Determine overall validation result
        validation_passed = len(errors) == 0
        
        result = AgentValidationResult(
            agent_name=agent_name,
            task_description=task_description,
            validation_passed=validation_passed,
            blocked=False,
            errors=errors,
            warnings=warnings,
            gaming_score=gaming_score,
            doc_compliance=doc_compliant,
            test_coverage=test_coverage,
            remediation_steps=remediation_steps
        )
        
        # Log validation result
        status = "✅ PASSED" if validation_passed else "❌ FAILED"
        logger.info(f"🔍 Agent validation {status} for '{agent_name}' - Gaming: {gaming_score:.2f}, Docs: {doc_compliant}, Tests: {test_coverage:.1%}")
        
        return result
    
    def _validate_task_for_agent(self, agent_name: str, task_description: str) -> Dict[str, Any]:
        """Validate that the task is appropriate for the agent"""
        
        # Define agent capabilities and task keywords
        agent_capabilities = {
            "code-implementer": ["implement", "code", "create", "build", "develop", "write"],
            "code-quality-reviewer": ["review", "check", "validate", "audit", "analyze"],
            "test-coverage-validator": ["test", "coverage", "validate", "verify", "spec"],
            "security-auditor": ["security", "vulnerability", "audit", "scan", "penetration"],
            "performance-optimizer": ["performance", "optimize", "speed", "memory", "benchmark"],
            "deployment-automation": ["deploy", "build", "ci", "cd", "pipeline", "release"],
            "database-architect": ["database", "schema", "migration", "query", "index"],
            "ui-ux-optimizer": ["ui", "ux", "interface", "design", "accessibility", "responsive"],
            "antihallucination-validator": ["validate", "verify", "exists", "check", "hallucination"],
            "system-architect": ["architecture", "design", "structure", "pattern", "framework"],
            "documentation-generator": ["documentation", "readme", "docs", "guide", "manual"],
            "api-design-architect": ["api", "endpoint", "rest", "graphql", "service", "integration"],
            "code-refactoring-optimizer": ["refactor", "optimize", "clean", "restructure", "improve"]
        }
        
        task_lower = task_description.lower()
        agent_keywords = agent_capabilities.get(agent_name, [])
        
        if not agent_keywords:
            return {"valid": True, "reason": "Unknown agent - allowing task"}
        
        # Check if task contains relevant keywords for the agent
        matches = sum(1 for keyword in agent_keywords if keyword in task_lower)
        
        if matches == 0:
            return {
                "valid": False, 
                "reason": f"Task '{task_description[:50]}...' does not match {agent_name} capabilities: {', '.join(agent_keywords)}"
            }
        
        return {"valid": True, "reason": f"Task matches {matches} capability keywords"}
    
    def _calculate_test_coverage(self) -> float:
        """Calculate approximate test coverage based on test files vs implementation files"""
        
        # Count implementation files
        impl_patterns = ['**/*.py', '**/*.js', '**/*.ts', '**/*.jsx', '**/*.tsx']
        exclude_patterns = ['**/test_*', '**/tests/*', '**/*_test.*', '**/*.test.*', '**/node_modules/*', '**/venv/*']
        
        impl_files = 0
        for pattern in impl_patterns:
            for file_path in self.project_path.glob(pattern):
                if not any(file_path.match(excl) for excl in exclude_patterns):
                    impl_files += 1
        
        # Count test files
        test_patterns = ['**/test_*.py', '**/tests/*.py', '**/*_test.py', '**/*.test.js', '**/*.test.ts', '**/*.spec.js', '**/*.spec.ts']
        test_files = 0
        for pattern in test_patterns:
            test_files += len(list(self.project_path.glob(pattern)))
        
        if impl_files == 0:
            return 1.0  # No implementation = 100% coverage
        
        # Simple heuristic: test coverage = (test_files / impl_files) capped at 1.0
        return min(1.0, test_files / impl_files)
    
    def post_development_validation(self, agent_name: str, files_modified: List[str]) -> Dict[str, Any]:
        """
        Validate agent behavior after development work is complete
        
        Args:
            agent_name: Name of the agent that performed work
            files_modified: List of files that were modified
            
        Returns:
            Validation result with gaming detection and recommendations
        """
        logger.info(f"🔍 Post-development validation for '{agent_name}' - {len(files_modified)} files modified")
        
        # Run DGTS validation on modified files
        violations = []
        for file_path in files_modified:
            file_violations = self.dgts_validator.scan_file_for_gaming(file_path)
            violations.extend(file_violations)
        
        # Calculate updated gaming score
        new_gaming_score = self.dgts_validator.calculate_gaming_score()
        
        # Track the development activity
        self.behavior_monitor.track_agent_activity(
            agent_name=agent_name,
            action="development_complete",
            details={
                "files_modified": len(files_modified),
                "violations_found": len(violations),
                "gaming_score": new_gaming_score,
                "files": files_modified[:10]  # Limit for logging
            }
        )
        
        # Check if agent should be blocked
        should_block = False
        block_reason = ""
        
        if new_gaming_score > 0.3:
            should_block = True
            block_reason = f"Gaming score {new_gaming_score:.2f} exceeds threshold 0.3"
        elif len(violations) >= 3:
            should_block = True
            block_reason = f"{len(violations)} gaming violations detected"
        
        if should_block:
            self.behavior_monitor.block_agent(agent_name, block_reason)
            logger.warning(f"🚫 Agent '{agent_name}' blocked: {block_reason}")
        
        return {
            "validation_passed": not should_block,
            "gaming_score": new_gaming_score,
            "violations": [v.to_dict() for v in violations],
            "agent_blocked": should_block,
            "block_reason": block_reason if should_block else None,
            "recommendations": [
                "Review gaming violations and fix before next development cycle",
                "Ensure tests validate real functionality, not mock data",
                "Avoid commenting out validation rules or enforcement code",
                "Implement genuine features, not placeholder/stub functions"
            ] if violations else []
        }

def enforce_agent_validation(agent_name: str, task_description: str, project_path: str = ".") -> AgentValidationResult:
    """
    Main enforcement function - validates that an agent can proceed with a task
    
    This is the primary entry point for the documentation-driven development enforcement.
    ALL specialized agents MUST call this before beginning any development work.
    
    Args:
        agent_name: Name of the specialized agent (e.g., "code-implementer")
        task_description: Description of the task to be performed
        project_path: Path to the project directory
        
    Returns:
        AgentValidationResult indicating if the agent can proceed
    """
    enforcer = AgentValidationEnforcer(project_path)
    return enforcer.enforce_agent_validation(agent_name, task_description)

def validate_post_development(agent_name: str, files_modified: List[str], project_path: str = ".") -> Dict[str, Any]:
    """
    Post-development validation function - checks for gaming behavior
    
    Args:
        agent_name: Name of the agent that performed work
        files_modified: List of files that were modified
        project_path: Path to the project directory
        
    Returns:
        Validation result with gaming detection and agent blocking decisions
    """
    enforcer = AgentValidationEnforcer(project_path)
    return enforcer.post_development_validation(agent_name, files_modified)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python agent_validation_enforcer.py <agent_name> <task_description> [project_path]")
        sys.exit(1)
    
    agent_name = sys.argv[1]
    task_description = sys.argv[2]
    project_path = sys.argv[3] if len(sys.argv) > 3 else "."
    
    result = enforce_agent_validation(agent_name, task_description, project_path)
    
    print(json.dumps({
        "agent_name": result.agent_name,
        "task_description": result.task_description,
        "validation_passed": result.validation_passed,
        "blocked": result.blocked,
        "errors": result.errors,
        "warnings": result.warnings,
        "gaming_score": result.gaming_score,
        "doc_compliance": result.doc_compliance,
        "test_coverage": result.test_coverage,
        "remediation_steps": result.remediation_steps
    }, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if result.validation_passed else 1)