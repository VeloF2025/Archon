"""
Workflow Knowledge Capture Service

Automatically captures and processes knowledge during workflow execution,
creating a continuous learning system that improves over time.

Key Features:
- Real-time knowledge capture during workflow execution
- Performance pattern detection and analysis
- Error pattern recognition and learning
- Success pattern extraction and sharing
- Automated documentation generation
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import re

from ...database.workflow_models import WorkflowExecution, StepExecution, ExecutionStatus
from ...database.agent_models import AgentV3
from ...utils import get_supabase_client
from .knowledge_agent_bridge import KnowledgeAgentBridge, KnowledgeIntegrationType

logger = logging.getLogger(__name__)

@dataclass
class CaptureEvent:
    """Event that triggers knowledge capture"""
    event_type: str
    timestamp: datetime
    execution_id: str
    step_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformancePattern:
    """Performance pattern detected in workflow execution"""
    pattern_id: str
    pattern_type: str  # "slow_step", "bottleneck", "efficient", "inefficient"
    step_id: str
    execution_time: float
    resource_usage: Dict[str, float]
    context: Dict[str, Any]
    confidence: float
    frequency: int = 1
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)

@dataclass
class ErrorPattern:
    """Error pattern detected in workflow execution"""
    pattern_id: str
    error_type: str
    step_id: str
    error_message: str
    context: Dict[str, Any]
    frequency: int = 1
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    recovery_suggestions: List[str] = field(default_factory=list)

@dataclass
class SuccessPattern:
    """Success pattern detected in workflow execution"""
    pattern_id: str
    success_type: str  # "fast_completion", "high_quality", "optimal_resource", "reliable"
    step_id: str
    execution_time: float
    context: Dict[str, Any]
    confidence: float
    frequency: int = 1
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)

class WorkflowKnowledgeCapture:
    """
    Service for automatically capturing and learning from workflow executions

    This service monitors workflow executions and extracts valuable knowledge
    about performance patterns, error patterns, and success patterns.
    """

    def __init__(self, knowledge_bridge: KnowledgeAgentBridge = None):
        """Initialize the knowledge capture service"""
        self.supabase = get_supabase_client()
        self.knowledge_bridge = knowledge_bridge or KnowledgeAgentBridge(self.supabase)

        # Pattern storage
        self.performance_patterns: Dict[str, PerformancePattern] = {}
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.success_patterns: Dict[str, SuccessPattern] = {}

        # Execution monitoring
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.execution_metrics: Dict[str, Dict[str, Any]] = {}

        # Configuration
        self.capture_config = {
            "enable_performance_capture": True,
            "enable_error_capture": True,
            "enable_success_capture": True,
            "min_confidence_threshold": 0.7,
            "min_frequency_threshold": 2,
            "real_time_analysis": True
        }

        # Pattern detection rules
        self.performance_thresholds = {
            "slow_step_threshold": 300,  # 5 minutes
            "bottleneck_threshold": 0.8,  # 80% of total execution time
            "efficient_threshold": 30,  # 30 seconds
            "resource_usage_high": 0.8  # 80% resource usage
        }

        logger.info("Workflow Knowledge Capture service initialized")

    async def start_execution_monitoring(self,
                                        execution_id: str,
                                        workflow_id: str,
                                        workflow_definition: Dict[str, Any]) -> str:
        """
        Start monitoring a workflow execution for knowledge capture

        Args:
            execution_id: Execution ID to monitor
            workflow_id: Workflow ID
            workflow_definition: Workflow definition

        Returns:
            Session ID for knowledge capture
        """
        # Start knowledge capture session
        session_id = await self.knowledge_bridge.start_workflow_session(
            workflow_id, execution_id, workflow_definition
        )

        # Initialize execution monitoring
        self.active_executions[execution_id] = {
            "session_id": session_id,
            "workflow_id": workflow_id,
            "start_time": datetime.now(),
            "step_start_times": {},
            "step_metrics": {},
            "events": [],
            "completed_steps": set(),
            "failed_steps": set(),
            "total_execution_time": 0
        }

        # Capture initial execution insight
        await self.knowledge_bridge.capture_execution_insight(
            session_id,
            "execution_start",
            "workflow_start",
            f"Workflow {workflow_id} execution started",
            {"workflow_definition": workflow_definition}
        )

        logger.info(f"Started execution monitoring for: {execution_id}")
        return session_id

    async def capture_step_start(self,
                               execution_id: str,
                               step_id: str,
                               step_data: Dict[str, Any]) -> bool:
        """
        Capture step start event

        Args:
            execution_id: Execution ID
            step_id: Step ID
            step_data: Step configuration data

        Returns:
            Success status
        """
        if execution_id not in self.active_executions:
            logger.warning(f"Unknown execution ID: {execution_id}")
            return False

        execution = self.active_executions[execution_id]
        session_id = execution["session_id"]

        # Record step start time
        execution["step_start_times"][step_id] = datetime.now()

        # Initialize step metrics
        execution["step_metrics"][step_id] = {
            "start_time": datetime.now().isoformat(),
            "resource_usage": {},
            "events": [],
            "status": "running"
        }

        # Capture step start insight
        await self.knowledge_bridge.capture_execution_insight(
            session_id,
            step_id,
            "step_start",
            f"Step {step_id} started execution",
            {"step_data": step_data}
        )

        logger.debug(f"Captured step start: {step_id} in execution: {execution_id}")
        return True

    async def capture_step_completion(self,
                                   execution_id: str,
                                   step_id: str,
                                   result: Dict[str, Any],
                                   metrics: Dict[str, Any] = None) -> bool:
        """
        Capture step completion event

        Args:
            execution_id: Execution ID
            step_id: Step ID
            result: Step execution result
            metrics: Step execution metrics

        Returns:
            Success status
        """
        if execution_id not in self.active_executions:
            return False

        execution = self.active_executions[execution_id]
        session_id = execution["session_id"]

        # Calculate execution time
        start_time = execution["step_start_times"].get(step_id)
        execution_time = 0
        if start_time:
            execution_time = (datetime.now() - start_time).total_seconds()

        # Update step metrics
        execution["step_metrics"][step_id].update({
            "end_time": datetime.now().isoformat(),
            "execution_time": execution_time,
            "result": result,
            "status": "completed",
            "metrics": metrics or {}
        })

        execution["completed_steps"].add(step_id)

        # Capture step completion insight
        await self.knowledge_bridge.capture_execution_insight(
            session_id,
            step_id,
            "step_completion",
            f"Step {step_id} completed in {execution_time:.2f} seconds",
            {"result": result, "metrics": metrics or {}},
            {"execution_time": execution_time}
        )

        # Analyze performance patterns
        if self.capture_config["enable_performance_capture"]:
            await self._analyze_step_performance(execution_id, step_id, execution_time, metrics or {})

        # Analyze success patterns
        if self.capture_config["enable_success_capture"] and result.get("success"):
            await self._analyze_success_pattern(execution_id, step_id, execution_time, result)

        logger.debug(f"Captured step completion: {step_id} in {execution_time:.2f}s")
        return True

    async def capture_step_failure(self,
                                  execution_id: str,
                                  step_id: str,
                                  error: Dict[str, Any],
                                  context: Dict[str, Any] = None) -> bool:
        """
        Capture step failure event

        Args:
            execution_id: Execution ID
            step_id: Step ID
            error: Error information
            context: Error context

        Returns:
            Success status
        """
        if execution_id not in self.active_executions:
            return False

        execution = self.active_executions[execution_id]
        session_id = execution["session_id"]

        # Calculate execution time
        start_time = execution["step_start_times"].get(step_id)
        execution_time = 0
        if start_time:
            execution_time = (datetime.now() - start_time).total_seconds()

        # Update step metrics
        execution["step_metrics"][step_id].update({
            "end_time": datetime.now().isoformat(),
            "execution_time": execution_time,
            "error": error,
            "status": "failed"
        })

        execution["failed_steps"].add(step_id)

        # Capture step failure insight
        await self.knowledge_bridge.capture_execution_insight(
            session_id,
            step_id,
            "step_failure",
            f"Step {step_id} failed after {execution_time:.2f} seconds: {error.get('message', 'Unknown error')}",
            {"error": error, "context": context or {}},
            {"execution_time": execution_time, "error_type": error.get("type", "unknown")}
        )

        # Analyze error patterns
        if self.capture_config["enable_error_capture"]:
            await self._analyze_error_pattern(execution_id, step_id, error, context or {})

        logger.warning(f"Captured step failure: {step_id} - {error.get('message', 'Unknown error')}")
        return True

    async def capture_agent_interaction(self,
                                     execution_id: str,
                                     step_id: str,
                                     agent_id: str,
                                     interaction_type: str,
                                     message: str,
                                     response: str,
                                     context: Dict[str, Any] = None) -> bool:
        """
        Capture agent interaction during workflow execution

        Args:
            execution_id: Execution ID
            step_id: Step ID
            agent_id: Agent ID
            interaction_type: Type of interaction
            message: Agent message
            response: Agent response
            context: Interaction context

        Returns:
            Success status
        """
        if execution_id not in self.active_executions:
            return False

        execution = self.active_executions[execution_id]
        session_id = execution["session_id"]

        # Capture agent communication
        success = await self.knowledge_bridge.capture_agent_communication(
            session_id,
            agent_id,
            message,
            response,
            {
                "step_id": step_id,
                "interaction_type": interaction_type,
                "context": context or {}
            }
        )

        # Record interaction in step metrics
        if step_id not in execution["step_metrics"]:
            execution["step_metrics"][step_id] = {"events": []}

        execution["step_metrics"][step_id]["events"].append({
            "type": "agent_interaction",
            "agent_id": agent_id,
            "interaction_type": interaction_type,
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "response": response
        })

        return success

    async def complete_execution_monitoring(self,
                                          execution_id: str,
                                          final_results: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Complete execution monitoring and generate final insights

        Args:
            execution_id: Execution ID
            final_results: Final execution results

        Returns:
            Success status and list of captured insights
        """
        if execution_id not in self.active_executions:
            return False, []

        execution = self.active_executions[execution_id]
        session_id = execution["session_id"]

        # Calculate total execution time
        start_time = execution["start_time"]
        total_execution_time = (datetime.now() - start_time).total_seconds()
        execution["total_execution_time"] = total_execution_time

        # Add final execution metrics
        execution_metrics = {
            "total_execution_time": total_execution_time,
            "completed_steps": len(execution["completed_steps"]),
            "failed_steps": len(execution["failed_steps"]),
            "total_steps": len(execution["step_metrics"]),
            "success_rate": len(execution["completed_steps"]) / max(len(execution["step_metrics"]), 1),
            "final_results": final_results
        }

        # Capture completion insight
        await self.knowledge_bridge.capture_execution_insight(
            session_id,
            "execution_end",
            "workflow_completion",
            f"Workflow execution completed in {total_execution_time:.2f} seconds",
            execution_metrics,
            {"total_execution_time": total_execution_time}
        )

        # Analyze overall execution patterns
        await self._analyze_execution_patterns(execution_id, final_results)

        # Complete knowledge capture session
        success, captured_knowledge = await self.knowledge_bridge.complete_workflow_session(
            session_id, final_results
        )

        # Clean up monitoring data
        del self.active_executions[execution_id]

        logger.info(f"Completed execution monitoring for: {execution_id}")
        return success, captured_knowledge

    async def get_execution_insights(self, execution_id: str) -> Dict[str, Any]:
        """
        Get insights captured during a specific execution

        Args:
            execution_id: Execution ID

        Returns:
            Execution insights
        """
        if execution_id not in self.active_executions:
            return {"error": "Execution not found or monitoring completed"}

        execution = self.active_executions[execution_id]
        session_id = execution["session_id"]

        # Get contextual knowledge from the session
        contextual_knowledge = await self.knowledge_bridge.get_contextual_knowledge(
            session_id,
            f"execution insights for {execution_id}"
        )

        return {
            "execution_id": execution_id,
            "workflow_id": execution["workflow_id"],
            "start_time": execution["start_time"].isoformat(),
            "current_duration": (datetime.now() - execution["start_time"]).total_seconds(),
            "completed_steps": len(execution["completed_steps"]),
            "failed_steps": len(execution["failed_steps"]),
            "step_metrics": execution["step_metrics"],
            "contextual_knowledge": contextual_knowledge,
            "total_events": len(execution.get("events", []))
        }

    async def get_workflow_patterns(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get patterns discovered for a specific workflow

        Args:
            workflow_id: Workflow ID

        Returns:
            Workflow patterns summary
        """
        # Filter patterns for this workflow
        workflow_performance_patterns = {
            k: v for k, v in self.performance_patterns.items()
            if v.context.get("workflow_id") == workflow_id
        }

        workflow_error_patterns = {
            k: v for k, v in self.error_patterns.items()
            if v.context.get("workflow_id") == workflow_id
        }

        workflow_success_patterns = {
            k: v for k, v in self.success_patterns.items()
            if v.context.get("workflow_id") == workflow_id
        }

        return {
            "workflow_id": workflow_id,
            "performance_patterns": len(workflow_performance_patterns),
            "error_patterns": len(workflow_error_patterns),
            "success_patterns": len(workflow_success_patterns),
            "top_performance_issues": [
                {
                    "pattern_id": p.pattern_id,
                    "step_id": p.step_id,
                    "pattern_type": p.pattern_type,
                    "avg_execution_time": p.execution_time,
                    "frequency": p.frequency
                }
                for p in sorted(
                    workflow_performance_patterns.values(),
                    key=lambda x: x.execution_time,
                    reverse=True
                )[:5]
            ],
            "common_errors": [
                {
                    "pattern_id": p.pattern_id,
                    "step_id": p.step_id,
                    "error_type": p.error_type,
                    "frequency": p.frequency,
                    "recovery_suggestions": p.recovery_suggestions
                }
                for p in sorted(
                    workflow_error_patterns.values(),
                    key=lambda x: x.frequency,
                    reverse=True
                )[:5]
            ],
            "success_factors": [
                {
                    "pattern_id": p.pattern_id,
                    "step_id": p.step_id,
                    "success_type": p.success_type,
                    "confidence": p.confidence,
                    "frequency": p.frequency
                }
                for p in sorted(
                    workflow_success_patterns.values(),
                    key=lambda x: (x.confidence, x.frequency),
                    reverse=True
                )[:5]
            ]
        }

    # Private helper methods

    async def _analyze_step_performance(self,
                                       execution_id: str,
                                       step_id: str,
                                       execution_time: float,
                                       metrics: Dict[str, Any]):
        """Analyze step performance for patterns"""
        try:
            workflow_id = self.active_executions[execution_id]["workflow_id"]

            # Check for slow step pattern
            if execution_time > self.performance_thresholds["slow_step_threshold"]:
                pattern_id = f"slow_step_{step_id}_{workflow_id}"
                if pattern_id not in self.performance_patterns:
                    self.performance_patterns[pattern_id] = PerformancePattern(
                        pattern_id=pattern_id,
                        pattern_type="slow_step",
                        step_id=step_id,
                        execution_time=execution_time,
                        resource_usage=metrics.get("resource_usage", {}),
                        context={"workflow_id": workflow_id, "execution_id": execution_id},
                        confidence=min(execution_time / self.performance_thresholds["slow_step_threshold"], 1.0)
                    )
                else:
                    pattern = self.performance_patterns[pattern_id]
                    pattern.frequency += 1
                    pattern.last_seen = datetime.now()
                    pattern.execution_time = (pattern.execution_time + execution_time) / 2  # Moving average
                    pattern.confidence = min(pattern.confidence + 0.1, 1.0)

            # Check for efficient step pattern
            elif execution_time < self.performance_thresholds["efficient_threshold"]:
                pattern_id = f"efficient_step_{step_id}_{workflow_id}"
                if pattern_id not in self.performance_patterns:
                    self.performance_patterns[pattern_id] = PerformancePattern(
                        pattern_id=pattern_id,
                        pattern_type="efficient",
                        step_id=step_id,
                        execution_time=execution_time,
                        resource_usage=metrics.get("resource_usage", {}),
                        context={"workflow_id": workflow_id, "execution_id": execution_id},
                        confidence=0.8
                    )
                else:
                    pattern = self.performance_patterns[pattern_id]
                    pattern.frequency += 1
                    pattern.last_seen = datetime.now()

        except Exception as e:
            logger.error(f"Failed to analyze step performance: {e}")

    async def _analyze_error_pattern(self,
                                   execution_id: str,
                                   step_id: str,
                                   error: Dict[str, Any],
                                   context: Dict[str, Any]):
        """Analyze error patterns"""
        try:
            workflow_id = self.active_executions[execution_id]["workflow_id"]
            error_type = error.get("type", "unknown")
            error_message = error.get("message", "")

            # Create pattern ID based on error characteristics
            error_hash = self._generate_error_hash(error_type, error_message, step_id)
            pattern_id = f"error_{error_hash}_{workflow_id}"

            recovery_suggestions = self._generate_recovery_suggestions(error_type, error_message)

            if pattern_id not in self.error_patterns:
                self.error_patterns[pattern_id] = ErrorPattern(
                    pattern_id=pattern_id,
                    error_type=error_type,
                    step_id=step_id,
                    error_message=error_message,
                    context={
                        "workflow_id": workflow_id,
                        "execution_id": execution_id,
                        **context
                    },
                    recovery_suggestions=recovery_suggestions
                )
            else:
                pattern = self.error_patterns[pattern_id]
                pattern.frequency += 1
                pattern.last_seen = datetime.now()

        except Exception as e:
            logger.error(f"Failed to analyze error pattern: {e}")

    async def _analyze_success_pattern(self,
                                     execution_id: str,
                                     step_id: str,
                                     execution_time: float,
                                     result: Dict[str, Any]):
        """Analyze success patterns"""
        try:
            workflow_id = self.active_executions[execution_id]["workflow_id"]

            # Check for fast completion pattern
            if execution_time < self.performance_thresholds["efficient_threshold"]:
                pattern_id = f"fast_completion_{step_id}_{workflow_id}"
                if pattern_id not in self.success_patterns:
                    self.success_patterns[pattern_id] = SuccessPattern(
                        pattern_id=pattern_id,
                        success_type="fast_completion",
                        step_id=step_id,
                        execution_time=execution_time,
                        context={"workflow_id": workflow_id, "execution_id": execution_id},
                        confidence=0.8
                    )
                else:
                    pattern = self.success_patterns[pattern_id]
                    pattern.frequency += 1
                    pattern.last_seen = datetime.now()
                    pattern.execution_time = (pattern.execution_time + execution_time) / 2

            # Check for high-quality result pattern
            if result.get("quality_score", 0) > 0.8:
                pattern_id = f"high_quality_{step_id}_{workflow_id}"
                if pattern_id not in self.success_patterns:
                    self.success_patterns[pattern_id] = SuccessPattern(
                        pattern_id=pattern_id,
                        success_type="high_quality",
                        step_id=step_id,
                        execution_time=execution_time,
                        context={"workflow_id": workflow_id, "execution_id": execution_id},
                        confidence=result.get("quality_score", 0.8)
                    )
                else:
                    pattern = self.success_patterns[pattern_id]
                    pattern.frequency += 1
                    pattern.last_seen = datetime.now()

        except Exception as e:
            logger.error(f"Failed to analyze success pattern: {e}")

    async def _analyze_execution_patterns(self,
                                        execution_id: str,
                                        final_results: Dict[str, Any]):
        """Analyze overall execution patterns"""
        try:
            execution = self.active_executions[execution_id]
            workflow_id = execution["workflow_id"]

            # Analyze execution success rate
            total_steps = len(execution["step_metrics"])
            successful_steps = len(execution["completed_steps"])
            success_rate = successful_steps / total_steps if total_steps > 0 else 0

            if success_rate >= 0.9:
                pattern_id = f"high_success_rate_{workflow_id}"
                if pattern_id not in self.success_patterns:
                    self.success_patterns[pattern_id] = SuccessPattern(
                        pattern_id=pattern_id,
                        success_type="high_success_rate",
                        step_id="workflow",
                        execution_time=execution["total_execution_time"],
                        context={"workflow_id": workflow_id, "success_rate": success_rate},
                        confidence=success_rate
                    )
                else:
                    pattern = self.success_patterns[pattern_id]
                    pattern.frequency += 1
                    pattern.last_seen = datetime.now()
                    pattern.confidence = (pattern.confidence + success_rate) / 2

        except Exception as e:
            logger.error(f"Failed to analyze execution patterns: {e}")

    def _generate_error_hash(self, error_type: str, error_message: str, step_id: str) -> str:
        """Generate hash for error pattern identification"""
        import hashlib
        error_text = f"{error_type}:{error_message}:{step_id}"
        return hashlib.md5(error_text.encode()).hexdigest()[:8]

    def _generate_recovery_suggestions(self, error_type: str, error_message: str) -> List[str]:
        """Generate recovery suggestions for errors"""
        suggestions = []

        # Common error patterns and suggestions
        if "timeout" in error_message.lower():
            suggestions.append("Increase timeout settings")
            suggestions.append("Optimize step performance")
            suggestions.append("Check for external service availability")

        elif "connection" in error_message.lower():
            suggestions.append("Check network connectivity")
            suggestions.append("Verify service endpoints")
            suggestions.append("Implement retry logic")

        elif "permission" in error_message.lower() or "access" in error_message.lower():
            suggestions.append("Check authentication and authorization")
            suggestions.append("Verify access permissions")
            suggestions.append("Review user roles and permissions")

        elif "validation" in error_message.lower():
            suggestions.append("Validate input data")
            suggestions.append("Check data format and schema")
            suggestions.append("Add input validation steps")

        elif "resource" in error_message.lower():
            suggestions.append("Check resource availability")
            suggestions.append("Monitor resource usage")
            suggestions.append("Implement resource scaling")

        else:
            suggestions.append("Review error logs for details")
            suggestions.append("Check system health")
            suggestions.append("Consult documentation")

        return suggestions

# Global instance
_workflow_knowledge_capture = None

def get_workflow_knowledge_capture(knowledge_bridge: KnowledgeAgentBridge = None) -> WorkflowKnowledgeCapture:
    """Get or create the global workflow knowledge capture instance"""
    global _workflow_knowledge_capture
    if _workflow_knowledge_capture is None:
        _workflow_knowledge_capture = WorkflowKnowledgeCapture(knowledge_bridge)
    return _workflow_knowledge_capture