"""
Handoff API Routes

REST API endpoints for the Agent Handoff System
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

from ...agents.orchestration.agent_handoff_engine import (
    AgentHandoffEngine, HandoffStrategy, HandoffTrigger, HandoffStatus
)
from ...agents.orchestration.archon_agency import ArchonAgency
from ...agents.enhanced_agent_capabilities import EnhancedAgentCapabilitySystem
from ...agents.capability_matching import CapabilityMatcher
from ...mcp_server.modules.agent_handoff_tools import AgentHandoffTools

router = APIRouter(prefix="/api/handoff", tags=["handoff"])

# Initialize services
agency = ArchonAgency()
handoff_tools = AgentHandoffTools(agency)
capability_system = EnhancedAgentCapabilitySystem()
capability_matcher = CapabilityMatcher(capability_system)

# Pydantic models for API
class HandoffRequestModel(BaseModel):
    source_agent_id: str = Field(..., description="ID of the source agent")
    target_agent_id: str = Field(..., description="ID of the target agent")
    message: str = Field(..., description="Message for the handoff")
    task_description: str = Field(..., description="Description of the task")
    strategy: HandoffStrategy = Field(default=HandoffStrategy.SEQUENTIAL)
    trigger: HandoffTrigger = Field(default=HandoffTrigger.EXPLICIT_REQUEST)
    confidence_score: float = Field(default=0.8, ge=0.0, le=1.0)
    priority: int = Field(default=3, ge=1, le=5)
    context: Dict[str, Any] = Field(default_factory=dict)

class HandoffRecommendationRequest(BaseModel):
    task_description: str = Field(..., description="Description of the task")
    current_agent_id: str = Field(..., description="ID of the current agent")

class OptimalStrategyRequest(BaseModel):
    task_complexity: float = Field(..., ge=0.0, le=1.0)
    agent_count: int = Field(..., ge=1)
    time_constraint_minutes: Optional[float] = Field(None, ge=0.0)

# API Endpoints
@router.get("/health")
async def health_check():
    """Health check endpoint for handoff system."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@router.get("/active")
async def get_active_handoffs(
    project_id: Optional[str] = Query(None, description="Project ID filter")
):
    """Get currently active handoffs."""
    try:
        # Get active handoffs from handoff engine
        active_handoffs = []

        # Get handoff history and filter for active ones
        history = await agency.get_handoff_history()
        for handoff in history:
            if handoff.get("status") in ["pending", "initiated", "in_progress"]:
                if not project_id or handoff.get("project_id") == project_id:
                    active_handoffs.append({
                        "handoff_id": handoff["handoff_id"],
                        "source_agent": handoff["source_agent_id"],
                        "target_agent": handoff["target_agent_id"],
                        "status": handoff["status"],
                        "progress": handoff.get("progress", 0),
                        "strategy": handoff.get("strategy", "sequential"),
                        "start_time": handoff.get("created_at", datetime.utcnow().isoformat()),
                        "estimated_completion": handoff.get("estimated_completion"),
                        "confidence_score": handoff.get("confidence_score", 0.8),
                        "task_description": handoff.get("task_description", "")
                    })

        return {"active_handoffs": active_handoffs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_handoff_history(
    project_id: Optional[str] = Query(None, description="Project ID filter"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of results")
):
    """Get handoff history."""
    try:
        history = await agency.get_handoff_history()

        # Filter by project if specified
        if project_id:
            history = [h for h in history if h.get("project_id") == project_id]

        # Sort by timestamp and limit
        history.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        limited_history = history[:limit]

        formatted_history = []
        for handoff in limited_history:
            created_at = handoff.get("created_at", datetime.utcnow().isoformat())
            completed_at = handoff.get("completed_at", created_at)

            # Calculate duration
            try:
                start_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
                duration = (end_time - start_time).total_seconds() * 1000  # Convert to milliseconds
            except:
                duration = 0

            formatted_history.append({
                "handoff_id": handoff["handoff_id"],
                "source_agent": handoff.get("source_agent_id", "unknown"),
                "target_agent": handoff.get("target_agent_id", "unknown"),
                "status": handoff.get("status", "unknown"),
                "strategy": handoff.get("strategy", "sequential"),
                "duration": duration,
                "success": handoff.get("status") == "completed",
                "timestamp": created_at,
                "task_summary": handoff.get("task_description", "")[:100] + "..." if len(handoff.get("task_description", "")) > 100 else handoff.get("task_description", "")
            })

        return {"history": formatted_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/execute")
async def execute_handoff(request: HandoffRequestModel):
    """Execute a handoff between agents."""
    try:
        # Generate handoff ID
        handoff_id = str(uuid4())

        # Create handoff request for the engine
        handoff_request = {
            "handoff_id": handoff_id,
            "source_agent_id": request.source_agent_id,
            "target_agent_id": request.target_agent_id,
            "message": request.message,
            "task_description": request.task_description,
            "strategy": request.strategy.value,
            "trigger": request.trigger.value,
            "confidence_score": request.confidence_score,
            "priority": request.priority,
            "context": request.context,
            "created_at": datetime.utcnow().isoformat(),
            "project_id": request.context.get("project_id")
        }

        # Execute handoff through agency
        result = await agency.request_handoff(
            source_agent=request.source_agent_id,
            target_agent=request.target_agent_id,
            message=request.message,
            task_description=request.task_description,
            strategy=request.strategy.value,
            context=request.context
        )

        return {
            "handoff_id": handoff_id,
            "status": result.status.value if hasattr(result, 'status') else "completed",
            "source_agent_id": request.source_agent_id,
            "target_agent_id": request.target_agent_id,
            "response_content": getattr(result, 'response_content', None),
            "execution_time": getattr(result, 'execution_time', 0),
            "error_message": getattr(result, 'error_message', None),
            "metrics": getattr(result, 'metrics', {}),
            "context_package_id": getattr(result, 'context_package_id', None),
            "completed_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{handoff_id}/cancel")
async def cancel_handoff(handoff_id: str):
    """Cancel an active handoff."""
    try:
        # In a real implementation, this would update the handoff status
        # For now, we'll return a success response
        return {"cancelled": True, "handoff_id": handoff_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommendations")
async def get_handoff_recommendations(request: HandoffRecommendationRequest):
    """Get handoff recommendations for a task."""
    try:
        recommendations = await agency.get_handoff_recommendations(
            message=request.task_description,
            task_description=request.task_description,
            current_agent=request.current_agent_id
        )

        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics")
async def get_handoff_analytics(
    project_id: Optional[str] = Query(None, description="Project ID filter"),
    time_range_hours: int = Query(24, ge=1, le=168, description="Time range in hours")
):
    """Get handoff analytics and insights."""
    try:
        analytics = await handoff_tools.get_handoff_analytics(
            time_range_hours=time_range_hours,
            agent_filter=project_id
        )

        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/visualization")
async def get_handoff_visualization(
    project_id: Optional[str] = Query(None, description="Project ID filter")
):
    """Get visualization data for handoffs."""
    try:
        # Get all the data needed for visualization
        [active_handoffs, handoff_history, analytics] = await promise_all([
            get_active_handoffs(project_id),
            get_handoff_history(project_id, 100),
            get_handoff_analytics(project_id, 24)
        ])

        # Get agent states
        agent_states = []
        for agent_id, agent in agency.agents.items():
            # Get agent handoff performance
            agent_perf = analytics.get("agent_performance", {}).get(agent_id, {})

            agent_states.append({
                "agent_id": agent_id,
                "agent_name": getattr(agent, 'name', agent_id),
                "agent_type": getattr(agent, 'agent_type', 'unknown'),
                "current_status": getattr(agent, 'state', 'available'),
                "current_handoff_id": None,  # Would need to track active handoffs per agent
                "handoff_stats": {
                    "initiated_today": agent_perf.get("handoffs_initiated", 0),
                    "received_today": agent_perf.get("handoffs_received", 0),
                    "success_rate": (agent_perf.get("success_rate_initiated", 0) + agent_perf.get("success_rate_received", 0)) / 2,
                    "avg_response_time": agent_perf.get("avg_response_time", 0)
                },
                "capabilities": list(getattr(agent, 'capabilities', {}).keys()),
                "load_factor": getattr(agent, 'load_factor', 0.0)
            })

        # Calculate performance metrics
        today_handoffs = [h for h in handoff_history.get("history", [])
                         if datetime.fromisoformat(h["timestamp"].replace('Z', '+00:00')).date() == datetime.utcnow().date()]

        success_rate = len([h for h in today_handoffs if h["success"]]) / len(today_handoffs) if today_handoffs else 0
        avg_handoff_time = sum(h["duration"] for h in today_handoffs) / len(today_handoffs) if today_handoffs else 0

        strategy_counts = {}
        for h in today_handoffs:
            strategy = h["strategy"]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        most_used_strategy = max(strategy_counts.items(), key=lambda x: x[1])[0] if strategy_counts else "sequential"

        return {
            "active_handoffs": active_handoffs.get("active_handoffs", []),
            "handoff_history": handoff_history.get("history", []),
            "agent_states": agent_states,
            "performance_metrics": {
                "total_handoffs_today": len(today_handoffs),
                "success_rate_today": success_rate,
                "avg_handoff_time": avg_handoff_time,
                "most_used_strategy": most_used_strategy,
                "most_active_agent": "unknown",  # Would need to calculate from agent performance
                "confidence_trend": 0.0,  # Would need historical confidence data
                "performance_score": success_rate
            },
            "last_updated": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/{agent_id}/capabilities")
async def get_agent_capabilities(agent_id: str):
    """Get capabilities for a specific agent."""
    try:
        capabilities = await handoff_tools.get_agent_capabilities(agent_id)
        return {"capabilities": capabilities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/capabilities")
async def get_all_agent_capabilities(
    project_id: Optional[str] = Query(None, description="Project ID filter")
):
    """Get capabilities for all agents."""
    try:
        all_capabilities = {}

        for agent_id, agent in agency.agents.items():
            if project_id and getattr(agent, 'project_id', None) != project_id:
                continue

            capabilities = await handoff_tools.get_agent_capabilities(agent_id)
            all_capabilities[agent_id] = capabilities

        return {"agent_capabilities": all_capabilities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategies")
async def get_available_strategies():
    """Get available handoff strategies."""
    try:
        strategies = await handoff_tools.get_available_strategies()
        return strategies
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimal-strategy")
async def get_optimal_strategy(request: OptimalStrategyRequest):
    """Get optimal strategy for given parameters."""
    try:
        # Simple strategy selection logic based on complexity and agent count
        if request.task_complexity > 0.7 and request.agent_count > 2:
            strategy = HandoffStrategy.COLLABORATIVE
            confidence = 0.8
            reasoning = "High complexity with multiple agents benefits from collaborative approach"
        elif request.task_complexity > 0.5 or request.time_constraint_minutes and request.time_constraint_minutes < 5:
            strategy = HandoffStrategy.PARALLEL
            confidence = 0.75
            reasoning = "Medium to high complexity or time constraints benefit from parallel execution"
        else:
            strategy = HandoffStrategy.SEQUENTIAL
            confidence = 0.9
            reasoning = "Lower complexity tasks are well-suited for sequential processing"

        return {
            "strategy": strategy.value,
            "confidence_score": confidence,
            "reasoning": reasoning
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/learning/run-cycle")
async def run_learning_cycle():
    """Run a manual learning cycle."""
    try:
        result = await handoff_tools.run_learning_cycle()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup-contexts")
async def cleanup_expired_contexts():
    """Clean up expired context packages."""
    try:
        result = await handoff_tools.cleanup_expired_contexts()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper function to simulate Promise.all
async def promise_all(awaitables):
    """Simulate JavaScript Promise.all for Python coroutines."""
    return [await awaitable for awaitable in awaitables]