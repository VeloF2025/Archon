"""
Agent Handoff Tools - MCP tools for handoff management.

This module provides MCP tools for managing intelligent agent handoffs,
including handoff requests, recommendations, and analytics.

Key Features:
- Handoff execution and monitoring
- Capability-based recommendations
- Real-time handoff analytics
- Learning and optimization insights
- Context preservation management
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field

from ...agents.orchestration.agent_handoff_engine import (
    HandoffStrategy, HandoffTrigger, HandoffStatus
)
from ...agents.orchestration.archon_agency import ArchonAgency

logger = logging.getLogger(__name__)


# Input models for MCP tools
class HandoffRequestInput(BaseModel):
    """Input model for handoff requests."""
    source_agent: str = Field(..., description="Name of the source agent")
    target_agent: str = Field(..., description="Name of the target agent")
    message: str = Field(..., description="Original message that triggered handoff")
    task_description: str = Field(..., description="Description of the task")
    strategy: str = Field("sequential", description="Handoff strategy: sequential, collaborative, conditional, parallel, delegation")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the handoff")


class HandoffRecommendationInput(BaseModel):
    """Input model for handoff recommendations."""
    message: str = Field(..., description="Current message/task")
    task_description: str = Field(..., description="Description of the task")
    current_agent: str = Field(..., description="Name of the current agent")


class HandoffAnalyticsInput(BaseModel):
    """Input model for handoff analytics."""
    time_range_hours: int = Field(24, description="Time range in hours for analytics")
    agent_filter: Optional[str] = Field(None, description="Filter by specific agent")


class ContextPreservationInput(BaseModel):
    """Input model for context preservation."""
    context_package_id: str = Field(..., description="ID of the context package")
    target_agent: str = Field(..., description="Name of the target agent")


# Output models for MCP tools
class HandoffResultOutput(BaseModel):
    """Output model for handoff results."""
    success: bool
    handoff_id: str
    status: str
    response_content: Optional[str] = None
    execution_time: float
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class HandoffRecommendationOutput(BaseModel):
    """Output model for handoff recommendations."""
    recommended: bool
    agents: List[Dict[str, Any]]
    strategy_recommendation: Optional[Dict[str, Any]] = None
    required_capabilities: List[str]
    reasoning: str


class HandoffAnalyticsOutput(BaseModel):
    """Output model for handoff analytics."""
    total_handoffs: int
    success_rate: float
    strategy_performance: Dict[str, Any]
    agent_performance: Dict[str, Any]
    learning_insights: Dict[str, Any]
    time_range_hours: int


class ContextPreservationOutput(BaseModel):
    """Output model for context preservation."""
    success: bool
    context_package_id: str
    size_bytes: int
    elements_count: int
    validation_status: str


class AgentHandoffTools:
    """
    MCP tools for agent handoff management.

    This class provides tools for:
    - Executing handoffs between agents
    - Getting handoff recommendations
    - Analyzing handoff performance
    - Managing context preservation
    """

    def __init__(self, agency: ArchonAgency):
        """Initialize handoff tools with agency reference."""
        self.agency = agency
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute_handoff(self, input_data: HandoffRequestInput) -> HandoffResultOutput:
        """
        Execute a handoff between agents.

        Args:
            input_data: Handoff request parameters

        Returns:
            HandoffResultOutput with execution results
        """
        try:
            self.logger.info(f"Executing handoff: {input_data.source_agent} -> {input_data.target_agent}")

            # Execute handoff through agency
            handoff_result = await self.agency.request_handoff(
                source_agent=input_data.source_agent,
                target_agent=input_data.target_agent,
                message=input_data.message,
                task_description=input_data.task_description,
                strategy=input_data.strategy,
                context=input_data.context
            )

            return HandoffResultOutput(
                success=handoff_result.status == HandoffStatus.COMPLETED,
                handoff_id=handoff_result.handoff_id,
                status=handoff_result.status.value,
                response_content=handoff_result.response_content,
                execution_time=handoff_result.execution_time,
                error_message=handoff_result.error_message,
                metrics=handoff_result.metrics
            )

        except Exception as e:
            self.logger.error(f"Error executing handoff: {e}")
            return HandoffResultOutput(
                success=False,
                handoff_id="",
                status="failed",
                execution_time=0.0,
                error_message=str(e)
            )

    async def get_handoff_recommendations(self, input_data: HandoffRecommendationInput) -> HandoffRecommendationOutput:
        """
        Get handoff recommendations for a task.

        Args:
            input_data: Recommendation request parameters

        Returns:
            HandoffRecommendationOutput with recommendations
        """
        try:
            self.logger.info(f"Getting handoff recommendations for {input_data.current_agent}")

            recommendations = await self.agency.get_handoff_recommendations(
                message=input_data.message,
                task_description=input_data.task_description,
                current_agent=input_data.current_agent
            )

            return HandoffRecommendationOutput(
                recommended=recommendations.get("recommended", False),
                agents=recommendations.get("agents", []),
                strategy_recommendation=recommendations.get("strategy_recommendation"),
                required_capabilities=recommendations.get("required_capabilities", []),
                reasoning=recommendations.get("reasoning", "")
            )

        except Exception as e:
            self.logger.error(f"Error getting handoff recommendations: {e}")
            return HandoffRecommendationOutput(
                recommended=False,
                agents=[],
                required_capabilities=[],
                reasoning=f"Error: {str(e)}"
            )

    async def get_handoff_analytics(self, input_data: HandoffAnalyticsInput) -> HandoffAnalyticsOutput:
        """
        Get handoff performance analytics.

        Args:
            input_data: Analytics request parameters

        Returns:
            HandoffAnalyticsOutput with performance metrics
        """
        try:
            self.logger.info(f"Getting handoff analytics for {input_data.time_range_hours} hours")

            # Get comprehensive statistics
            stats = await self.agency.get_handoff_statistics()

            # Extract relevant analytics
            handoff_engine = stats.get("handoff_engine", {})
            learning = stats.get("learning", {})

            # Filter by agent if specified
            if input_data.agent_filter:
                agent_history = await self.agency.get_agent_handoff_history(input_data.agent_filter)
                total_handoffs = len(agent_history)
                successful_handoffs = len([h for h in agent_history if h.get("status") == "completed"])
                success_rate = successful_handoffs / total_handoffs if total_handoffs > 0 else 0.0
            else:
                total_handoffs = handoff_engine.get("total_handoffs", 0)
                success_rate = handoff_engine.get("success_rate", 0.0)

            return HandoffAnalyticsOutput(
                total_handoffs=total_handoffs,
                success_rate=success_rate,
                strategy_performance=handoff_engine.get("strategy_statistics", {}),
                agent_performance=handoff_engine,
                learning_insights=learning,
                time_range_hours=input_data.time_range_hours
            )

        except Exception as e:
            self.logger.error(f"Error getting handoff analytics: {e}")
            return HandoffAnalyticsOutput(
                total_handoffs=0,
                success_rate=0.0,
                strategy_performance={},
                agent_performance={},
                learning_insights={},
                time_range_hours=input_data.time_range_hours
            )

    async def validate_context_package(self, input_data: ContextPreservationInput) -> ContextPreservationOutput:
        """
        Validate a context package for handoff.

        Args:
            input_data: Context validation parameters

        Returns:
            ContextPreservationOutput with validation results
        """
        try:
            self.logger.info(f"Validating context package: {input_data.context_package_id}")

            # Validate context integrity
            is_valid = await self.agency.context_engine.validate_context_integrity(
                input_data.context_package_id
            )

            # Get package details
            package = self.agency.context_engine.context_packages.get(input_data.context_package_id)
            if not package:
                return ContextPreservationOutput(
                    success=False,
                    context_package_id=input_data.context_package_id,
                    size_bytes=0,
                    elements_count=0,
                    validation_status="not_found"
                )

            return ContextPreservationOutput(
                success=is_valid,
                context_package_id=input_data.context_package_id,
                size_bytes=package.size_bytes,
                elements_count=len(package.elements),
                validation_status="valid" if is_valid else "invalid"
            )

        except Exception as e:
            self.logger.error(f"Error validating context package: {e}")
            return ContextPreservationOutput(
                success=False,
                context_package_id=input_data.context_package_id,
                size_bytes=0,
                elements_count=0,
                validation_status="error"
            )

    async def get_agent_capabilities(self, agent_name: str) -> Dict[str, Any]:
        """
        Get capabilities for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Dictionary with agent capabilities
        """
        try:
            if agent_name not in self.agency.agents:
                return {"error": f"Agent '{agent_name}' not found"}

            if agent_name not in self.agency.handoff_engine.agent_capabilities:
                return {"error": f"Capabilities not available for '{agent_name}'"}

            capability = self.agency.handoff_engine.agent_capabilities[agent_name]

            return {
                "agent_name": agent_name,
                "capabilities": list(capability.capabilities),
                "expertise_levels": capability.expertise_level,
                "performance_metrics": capability.performance_metrics,
                "availability": capability.availability,
                "last_active": capability.last_active.isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting agent capabilities: {e}")
            return {"error": str(e)}

    async def get_available_strategies(self) -> Dict[str, Any]:
        """
        Get available handoff strategies and their descriptions.

        Returns:
            Dictionary with strategy information
        """
        try:
            strategies = {}

            for strategy in HandoffStrategy:
                strategies[strategy.value] = {
                    "description": self._get_strategy_description(strategy),
                    "best_for": self._get_strategy_best_use_case(strategy)
                }

            return {"strategies": strategies}

        except Exception as e:
            self.logger.error(f"Error getting available strategies: {e}")
            return {"error": str(e)}

    async def run_learning_cycle(self) -> Dict[str, Any]:
        """
        Manually trigger a learning cycle.

        Returns:
            Dictionary with learning cycle results
        """
        try:
            self.logger.info("Manually triggering learning cycle")

            await self.agency.run_learning_cycle()

            # Get updated insights
            stats = await self.agency.get_handoff_statistics()
            learning_insights = stats.get("learning", {})

            return {
                "success": True,
                "message": "Learning cycle completed",
                "updated_insights": learning_insights,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error running learning cycle: {e}")
            return {
                "success": False,
                "message": f"Learning cycle failed: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }

    async def cleanup_expired_contexts(self) -> Dict[str, Any]:
        """
        Clean up expired context packages.

        Returns:
            Dictionary with cleanup results
        """
        try:
            self.logger.info("Cleaning up expired contexts")

            # Get statistics before cleanup
            before_stats = self.agency.context_engine.get_context_statistics()
            before_count = before_stats.get("total_packages", 0)

            # Perform cleanup
            await self.agency.context_engine.cleanup_expired_contexts()

            # Get statistics after cleanup
            after_stats = self.agency.context_engine.get_context_statistics()
            after_count = after_stats.get("total_packages", 0)

            cleaned_count = before_count - after_count

            return {
                "success": True,
                "packages_before": before_count,
                "packages_after": after_count,
                "cleaned_count": cleaned_count,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error cleaning up expired contexts: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def _get_strategy_description(self, strategy: HandoffStrategy) -> str:
        """Get description for a strategy."""
        descriptions = {
            HandoffStrategy.SEQUENTIAL: "Agents work in sequence, one after another",
            HandoffStrategy.COLLABORATIVE: "Agents work together on the same task",
            HandoffStrategy.CONDITIONAL: "Handoff based on specific conditions",
            HandoffStrategy.PARALLEL: "Multiple agents work concurrently",
            HandoffStrategy.DELEGATION: "Complete task transfer to another agent"
        }
        return descriptions.get(strategy, "Unknown strategy")

    def _get_strategy_best_use_case(self, strategy: HandoffStrategy) -> str:
        """Get best use case for a strategy."""
        use_cases = {
            HandoffStrategy.SEQUENTIAL: "Simple linear workflows",
            HandoffStrategy.COLLABORATIVE: "Complex tasks requiring multiple expertise areas",
            HandoffStrategy.CONDITIONAL: "Situations with specific triggers or thresholds",
            HandoffStrategy.PARALLEL: "Time-sensitive tasks that can be parallelized",
            HandoffStrategy.DELEGATION: "Tasks that require specialized expertise"
        }
        return use_cases.get(strategy, "General use")


# MCP Tool functions for registration
async def execute_handoff(
    source_agent: str,
    target_agent: str,
    message: str,
    task_description: str,
    strategy: str = "sequential",
    context: Optional[Dict[str, Any]] = None,
    # Agency instance injected by MCP server
    agency: Optional[ArchonAgency] = None
) -> Dict[str, Any]:
    """
    Execute a handoff between agents.

    Args:
        source_agent: Name of the source agent
        target_agent: Name of the target agent
        message: Original message that triggered handoff
        task_description: Description of the task
        strategy: Handoff strategy to use
        context: Additional context for the handoff
        agency: Injected ArchonAgency instance

    Returns:
        Dictionary with handoff results
    """
    if not agency:
        raise ValueError("Agency instance not provided")

    tools = AgentHandoffTools(agency)
    input_data = HandoffRequestInput(
        source_agent=source_agent,
        target_agent=target_agent,
        message=message,
        task_description=task_description,
        strategy=strategy,
        context=context
    )

    result = await tools.execute_handoff(input_data)
    return result.dict()


async def get_handoff_recommendations(
    message: str,
    task_description: str,
    current_agent: str,
    # Agency instance injected by MCP server
    agency: Optional[ArchonAgency] = None
) -> Dict[str, Any]:
    """
    Get handoff recommendations for a task.

    Args:
        message: Current message/task
        task_description: Description of the task
        current_agent: Name of the current agent
        agency: Injected ArchonAgency instance

    Returns:
        Dictionary with handoff recommendations
    """
    if not agency:
        raise ValueError("Agency instance not provided")

    tools = AgentHandoffTools(agency)
    input_data = HandoffRecommendationInput(
        message=message,
        task_description=task_description,
        current_agent=current_agent
    )

    result = await tools.get_handoff_recommendations(input_data)
    return result.dict()


async def get_handoff_analytics(
    time_range_hours: int = 24,
    agent_filter: Optional[str] = None,
    # Agency instance injected by MCP server
    agency: Optional[ArchonAgency] = None
) -> Dict[str, Any]:
    """
    Get handoff performance analytics.

    Args:
        time_range_hours: Time range in hours for analytics
        agent_filter: Filter by specific agent
        agency: Injected ArchonAgency instance

    Returns:
        Dictionary with performance analytics
    """
    if not agency:
        raise ValueError("Agency instance not provided")

    tools = AgentHandoffTools(agency)
    input_data = HandoffAnalyticsInput(
        time_range_hours=time_range_hours,
        agent_filter=agent_filter
    )

    result = await tools.get_handoff_analytics(input_data)
    return result.dict()