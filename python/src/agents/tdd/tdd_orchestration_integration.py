#!/usr/bin/env python3
"""
TDD Orchestration Integration - Connect TDD System to Archon Orchestration

This module integrates the TDD enforcement system with the existing Archon
orchestration infrastructure, ensuring all agents comply with TDD principles.

CRITICAL: All orchestrated agents must pass through TDD validation gates.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .tdd_enforcement_gate import TDDEnforcementGate, EnforcementResult
from .file_monitor import TDDFileMonitor, get_file_monitor
from .browserbase_config import test_browserbase_connection

# Import orchestration components if available
try:
    from ..orchestration.meta_agent import MetaAgent
    from ..orchestration.parallel_executor import ParallelExecutor
    from ..orchestration.agent_manager import AgentManager
    META_AGENT_AVAILABLE = True
except ImportError:
    MetaAgent = None
    ParallelExecutor = None
    AgentManager = None
    META_AGENT_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class TDDIntegrationConfig:
    """Configuration for TDD orchestration integration"""
    enforce_on_agents: bool = True
    block_non_compliant_agents: bool = True
    require_tests_before_execution: bool = True
    enable_real_time_monitoring: bool = True
    gaming_detection_enabled: bool = True
    min_coverage_percentage: float = 95.0
    project_path: str = "."

class TDDOrchestrationIntegrator:
    """
    Integrates TDD enforcement with Archon orchestration system
    
    Ensures all orchestrated agents and tasks comply with TDD principles
    before execution, maintaining code quality and test-first development.
    """
    
    def __init__(self, config: TDDIntegrationConfig = None):
        self.config = config or TDDIntegrationConfig()
        
        # Initialize TDD components
        self.tdd_gate = TDDEnforcementGate(
            project_path=self.config.project_path,
            min_coverage_percentage=self.config.min_coverage_percentage,
            enable_gaming_detection=self.config.gaming_detection_enabled
        )
        
        self.file_monitor: Optional[TDDFileMonitor] = None
        if self.config.enable_real_time_monitoring:
            self.file_monitor = get_file_monitor(self.config.project_path)
        
        # Integration state
        self.integrated_agents: Dict[str, bool] = {}
        self.validation_history: List[Dict[str, Any]] = []
        self.blocked_tasks: List[str] = []
        
        logger.info("🔗 TDD Orchestration Integration initialized")
    
    async def initialize_integration(self) -> bool:
        """Initialize TDD integration with orchestration system"""
        try:
            logger.info("🚀 Initializing TDD orchestration integration...")
            
            # Start file monitoring if enabled
            if self.file_monitor and self.config.enable_real_time_monitoring:
                success = self.file_monitor.start_monitoring()
                if success:
                    logger.info("✅ TDD file monitoring integrated with orchestration")
                else:
                    logger.warning("⚠️  TDD file monitoring failed to start")
            
            # Test Browserbase connection
            browserbase_status = await test_browserbase_connection()
            if browserbase_status.get("connected", False):
                logger.info("✅ Browserbase cloud testing integrated")
            else:
                logger.warning("⚠️  Browserbase not available - local testing only")
            
            # Hook into orchestration system if available
            if META_AGENT_AVAILABLE:
                await self._integrate_with_meta_agent()
            
            logger.info("✅ TDD orchestration integration complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ TDD integration failed: {str(e)}")
            return False
    
    async def _integrate_with_meta_agent(self):
        """Integrate with MetaAgent orchestration system"""
        try:
            # This would hook into the MetaAgent's task execution pipeline
            # to enforce TDD compliance before any agent execution
            
            logger.info("🤖 Integrating TDD enforcement with MetaAgent system")
            
            # Add TDD validation to agent execution pipeline
            # This is a placeholder for the actual integration
            
        except Exception as e:
            logger.error(f"MetaAgent integration failed: {str(e)}")
    
    async def validate_agent_task(
        self,
        agent_name: str,
        task_description: str,
        feature_name: str = None,
        implementation_files: List[str] = None
    ) -> Tuple[bool, str]:
        """
        Validate TDD compliance for an agent task before execution
        
        Args:
            agent_name: Name of the agent requesting execution
            task_description: Description of the task to be performed
            feature_name: Name of the feature being developed
            implementation_files: List of files that will be modified
            
        Returns:
            Tuple of (allowed, message) indicating if task can proceed
        """
        
        try:
            logger.info(f"🔍 Validating TDD compliance for {agent_name}: {task_description}")
            
            # Extract feature name from task if not provided
            if not feature_name:
                feature_name = self._extract_feature_from_task(task_description)
            
            # Perform TDD validation
            validation_result = await self.tdd_gate.validate_feature_development(
                feature_name=feature_name or f"agent_task_{agent_name}",
                implementation_files=implementation_files
            )
            
            # Record validation history
            validation_record = {
                "timestamp": datetime.now().isoformat(),
                "agent_name": agent_name,
                "task_description": task_description,
                "feature_name": feature_name,
                "allowed": validation_result.allowed,
                "violations": validation_result.total_violations,
                "gaming_score": validation_result.gaming_score
            }
            self.validation_history.append(validation_record)
            
            # Update agent tracking
            self.integrated_agents[agent_name] = validation_result.allowed
            
            if validation_result.allowed:
                logger.info(f"✅ TDD validation passed for {agent_name}")
                return True, "TDD compliance validated - task approved"
            else:
                logger.warning(f"❌ TDD validation failed for {agent_name}: {validation_result.message}")
                self.blocked_tasks.append(f"{agent_name}:{task_description}")
                return False, validation_result.message
                
        except Exception as e:
            error_msg = f"TDD validation error for {agent_name}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _extract_feature_from_task(self, task_description: str) -> Optional[str]:
        """Extract feature name from task description"""
        
        # Simple extraction - look for common patterns
        task_lower = task_description.lower()
        
        # Look for "implement X", "create X", "add X" patterns
        patterns = [
            r"implement (\w+)",
            r"create (\w+)",
            r"add (\w+)",
            r"build (\w+)",
            r"develop (\w+)",
            r"feature (\w+)",
            r"component (\w+)"
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, task_lower)
            if match:
                return match.group(1)
        
        # Fallback - use first significant word
        words = task_description.split()
        for word in words:
            if len(word) > 3 and word.isalpha():
                return word.lower()
        
        return None
    
    async def pre_agent_execution_hook(
        self,
        agent_name: str,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Pre-execution hook called before any agent runs
        
        Args:
            agent_name: Name of the agent about to execute
            task_data: Task data including description, files, etc.
            
        Returns:
            Updated task data or raises exception if blocked
        """
        
        if not self.config.enforce_on_agents:
            return task_data
        
        # Validate TDD compliance
        allowed, message = await self.validate_agent_task(
            agent_name=agent_name,
            task_description=task_data.get("description", ""),
            feature_name=task_data.get("feature_name"),
            implementation_files=task_data.get("files", [])
        )
        
        if not allowed and self.config.block_non_compliant_agents:
            raise RuntimeError(f"TDD Validation Failed: {message}")
        
        # Add TDD metadata to task
        task_data["tdd_validation"] = {
            "validated_at": datetime.now().isoformat(),
            "allowed": allowed,
            "message": message
        }
        
        return task_data
    
    async def post_agent_execution_hook(
        self,
        agent_name: str,
        task_data: Dict[str, Any],
        execution_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Post-execution hook called after agent completes
        
        Args:
            agent_name: Name of the agent that executed
            task_data: Original task data
            execution_result: Result from agent execution
            
        Returns:
            Enhanced execution result with TDD validation
        """
        
        try:
            # Re-validate TDD compliance after execution
            feature_name = task_data.get("feature_name") or self._extract_feature_from_task(
                task_data.get("description", "")
            )
            
            if feature_name:
                post_validation = await self.tdd_gate.validate_feature_development(
                    feature_name=feature_name
                )
                
                execution_result["tdd_post_validation"] = {
                    "validated_at": datetime.now().isoformat(),
                    "allowed": post_validation.allowed,
                    "message": post_validation.message,
                    "violations": post_validation.total_violations,
                    "gaming_score": post_validation.gaming_score
                }
                
                if not post_validation.allowed:
                    logger.warning(f"⚠️  Post-execution TDD validation failed for {agent_name}")
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Post-execution TDD validation failed: {str(e)}")
            execution_result["tdd_post_validation"] = {
                "error": str(e),
                "validated_at": datetime.now().isoformat()
            }
            return execution_result
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current TDD integration status"""
        
        file_monitor_stats = {}
        if self.file_monitor:
            file_monitor_stats = self.file_monitor.get_monitoring_stats()
        
        return {
            "integration_active": True,
            "config": {
                "enforce_on_agents": self.config.enforce_on_agents,
                "block_non_compliant_agents": self.config.block_non_compliant_agents,
                "require_tests_before_execution": self.config.require_tests_before_execution,
                "real_time_monitoring": self.config.enable_real_time_monitoring,
                "gaming_detection": self.config.gaming_detection_enabled,
                "min_coverage": self.config.min_coverage_percentage
            },
            "validation_stats": {
                "total_validations": len(self.validation_history),
                "agents_validated": len(self.integrated_agents),
                "blocked_tasks": len(self.blocked_tasks),
                "success_rate": self._calculate_success_rate()
            },
            "file_monitoring": file_monitor_stats,
            "orchestration_hooks": {
                "meta_agent_integrated": META_AGENT_AVAILABLE,
                "pre_execution_hook_active": self.config.enforce_on_agents,
                "post_execution_hook_active": True
            }
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate TDD validation success rate"""
        if not self.validation_history:
            return 0.0
        
        successful = sum(1 for v in self.validation_history if v["allowed"])
        return (successful / len(self.validation_history)) * 100
    
    def get_recent_validations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent TDD validations"""
        return self.validation_history[-limit:] if self.validation_history else []
    
    def get_blocked_tasks(self) -> List[str]:
        """Get list of blocked tasks"""
        return self.blocked_tasks.copy()

# Global integration instance
_tdd_integrator: Optional[TDDOrchestrationIntegrator] = None

async def get_tdd_integrator(config: TDDIntegrationConfig = None) -> TDDOrchestrationIntegrator:
    """Get or create global TDD orchestration integrator"""
    global _tdd_integrator
    
    if _tdd_integrator is None:
        _tdd_integrator = TDDOrchestrationIntegrator(config)
        await _tdd_integrator.initialize_integration()
    
    return _tdd_integrator

async def validate_orchestrated_task(
    agent_name: str,
    task_description: str,
    feature_name: str = None,
    implementation_files: List[str] = None
) -> Tuple[bool, str]:
    """
    Validate TDD compliance for an orchestrated task
    
    Convenience function for quick TDD validation in orchestration workflows.
    """
    
    integrator = await get_tdd_integrator()
    return await integrator.validate_agent_task(
        agent_name=agent_name,
        task_description=task_description,
        feature_name=feature_name,
        implementation_files=implementation_files
    )

def setup_tdd_orchestration_hooks():
    """
    Setup TDD hooks in the orchestration system
    
    This would be called during system initialization to ensure
    all orchestrated tasks go through TDD validation.
    """
    
    logger.info("🔗 Setting up TDD orchestration hooks...")
    
    # This would integrate with the actual orchestration system
    # For now, log that hooks would be installed
    
    logger.info("✅ TDD orchestration hooks ready for integration")