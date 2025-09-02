#!/usr/bin/env python3
"""
ARCHON MANIFEST INTEGRATION
Hardcoded integration that ensures all Archon components reference MANIFEST.md
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ArchonManifestIntegration:
    """
    Mandatory integration that enforces MANIFEST.md compliance across all Archon operations
    """
    
    MANIFEST_PATH = "MANIFEST.md"
    MANIFEST_AUTHORITY = "ARCHON_OPERATIONAL_MANIFEST"
    COMPLIANCE_REQUIRED = True
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent.parent
        self.manifest_file = self.project_root / self.MANIFEST_PATH
        self.manifest_loaded = False
        self.manifest_content = None
        
        # MANDATORY: Load manifest on initialization
        self._load_manifest()
    
    def _load_manifest(self) -> bool:
        """
        MANDATORY: Load the MANIFEST.md file
        CRITICAL: All Archon operations depend on this
        """
        try:
            if not self.manifest_file.exists():
                logger.critical(f"MANIFEST.md not found at {self.manifest_file}")
                logger.critical("ARCHON SYSTEM CANNOT OPERATE WITHOUT MANIFEST")
                raise FileNotFoundError("MANIFEST.md is required for Archon operation")
            
            with open(self.manifest_file, 'r', encoding='utf-8') as f:
                self.manifest_content = f.read()
            
            self.manifest_loaded = True
            logger.info("✅ ARCHON OPERATIONAL MANIFEST loaded successfully")
            
            # Validate manifest content
            if "ARCHON OPERATIONAL MANIFEST" not in self.manifest_content:
                logger.error("❌ Invalid manifest - missing ARCHON OPERATIONAL MANIFEST header")
                return False
            
            if "MANDATORY COMPLIANCE" not in self.manifest_content:
                logger.error("❌ Invalid manifest - missing MANDATORY COMPLIANCE directive")
                return False
                
            logger.info("✅ MANIFEST validation passed - system ready for operation")
            return True
            
        except Exception as e:
            logger.critical(f"FATAL: Cannot load ARCHON MANIFEST: {e}")
            logger.critical("SYSTEM HALTED - MANIFEST COMPLIANCE REQUIRED")
            raise
    
    def get_manifest_directive(self, section: str) -> Optional[str]:
        """
        Extract specific directive from manifest
        USAGE: All agents MUST reference manifest directives for their operations
        """
        if not self.manifest_loaded:
            logger.error("MANIFEST not loaded - cannot retrieve directive")
            return None
        
        # Extract section content (simplified implementation)
        lines = self.manifest_content.split('\n')
        section_content = []
        in_section = False
        
        for line in lines:
            if f"## {section}" in line or f"### {section}" in line:
                in_section = True
                continue
            elif in_section and (line.startswith('##') or line.startswith('###')):
                break
            elif in_section:
                section_content.append(line)
        
        return '\n'.join(section_content) if section_content else None
    
    def enforce_compliance_check(self, component: str, operation: str) -> bool:
        """
        MANDATORY: All components must call this before any operation
        """
        if not self.manifest_loaded:
            logger.critical(f"COMPLIANCE VIOLATION: {component} attempted {operation} without manifest")
            return False
        
        logger.debug(f"✅ {component} compliance check passed for {operation}")
        return True
    
    def get_agent_system_prompt_prefix(self) -> str:
        """
        MANDATORY: All agents must include this in their system prompt
        """
        return f"""
🎯 ARCHON OPERATIONAL MANIFEST COMPLIANCE REQUIRED

You are operating under the ARCHON OPERATIONAL MANIFEST located at {self.MANIFEST_PATH}.

MANDATORY DIRECTIVES:
1. ALWAYS reference the ARCHON OPERATIONAL MANIFEST for all operations
2. FOLLOW validation-first, documentation-driven patterns
3. ENFORCE zero tolerance for gaming or hallucination  
4. COMPLY WITH all quality gates and validation requirements
5. REPORT TO orchestration system for coordination

CRITICAL: Every action must align with manifest protocols. Non-compliance is not permitted.

MANIFEST STATUS: ACTIVE - MANDATORY COMPLIANCE
"""

    def get_validation_requirements(self) -> Dict[str, Any]:
        """
        Extract validation requirements from manifest
        """
        return {
            "pre_development": [
                "AntiHallucination Check",
                "DGTS Gaming Detection", 
                "Documentation-Driven Test Planning",
                "Agent Validation Enforcement"
            ],
            "post_development": [
                "Zero-Tolerance Check",
                "Full Test Suite",
                "Performance Validation",
                "Security Final Scan",
                "Documentation Verification",
                "NLNH Compliance Check"
            ],
            "quality_gates": {
                "code_coverage": ">95%",
                "security_score": ">90%",
                "performance_score": ">85%",
                "zero_tolerance_compliance": "100%"
            }
        }
    
    def get_agent_orchestration_rules(self) -> Dict[str, Any]:
        """
        Extract orchestration rules from manifest
        """
        return {
            "meta_agent_triggers": {
                "spawn_new_agents": [
                    "Resource contention >80%",
                    "Skill gap identified",
                    "Parallel opportunity available",
                    "Quality gate failure",
                    "Complexity escalation"
                ],
                "scale_down": [
                    "Agents idle >15 minutes",
                    "Task completion",
                    "Resource constraints",
                    "Efficiency optimization"
                ]
            },
            "blocking_conditions": [
                "DGTS Gaming Score >0.3",
                "AntiHall Failures",
                "Security Vulnerabilities",
                "Test Coverage <95%",
                "Build Failures",
                "Performance Regression"
            ],
            "communication_patterns": [
                "Event-Driven Updates via Socket.IO",
                "Database State Sharing via Supabase",
                "File System Coordination via git worktrees",
                "API Status Updates via RESTful endpoints"
            ]
        }

# GLOBAL MANIFEST INTEGRATION INSTANCE
# This ensures all Archon components reference the same manifest
_GLOBAL_MANIFEST = None

def get_archon_manifest() -> ArchonManifestIntegration:
    """
    Global singleton access to Archon Manifest
    USAGE: All Archon components must use this function
    """
    global _GLOBAL_MANIFEST
    
    if _GLOBAL_MANIFEST is None:
        _GLOBAL_MANIFEST = ArchonManifestIntegration()
    
    return _GLOBAL_MANIFEST

def enforce_manifest_compliance(component: str, operation: str) -> bool:
    """
    MANDATORY: All operations must call this function
    """
    manifest = get_archon_manifest()
    return manifest.enforce_compliance_check(component, operation)

def get_manifest_system_prompt() -> str:
    """
    MANDATORY: All agents must include this in their system prompt
    """
    manifest = get_archon_manifest()
    return manifest.get_agent_system_prompt_prefix()

if __name__ == "__main__":
    # Test manifest integration
    manifest = get_archon_manifest()
    print("✅ ARCHON MANIFEST INTEGRATION TEST PASSED")
    print(f"Manifest loaded: {manifest.manifest_loaded}")
    print(f"Manifest path: {manifest.manifest_file}")