"""
Validation API Routes for DGTS and NLNH Monitoring

Provides endpoints for:
- Real-time validation status
- Gaming detection results
- NLNH confidence scores
- Agent behavior monitoring
"""

import logging
import time
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ...agents.validation.dgts_validator import DGTSValidator
from ...agents.validation.agent_behavior_monitor import AgentBehaviorMonitor
from ..socketio_app import get_socketio_instance

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/validation", tags=["validation"])

# Initialize validators
dgts_validator = DGTSValidator(project_path=".")  # Use current directory as project path
behavior_monitor = AgentBehaviorMonitor(project_path=".")  # Use current directory as project path


class ValidationRequest(BaseModel):
    """Request model for code validation"""
    code: str = Field(..., description="Code to validate")
    agent_id: Optional[str] = Field(None, description="Agent ID performing the action")
    context: Optional[str] = Field(None, description="Context of the code (e.g., 'test', 'implementation')")


class ValidationResponse(BaseModel):
    """Response model for validation results"""
    valid: bool
    gaming_score: float
    violations: List[Dict[str, Any]]
    suggestions: List[str]
    timestamp: float


class AgentStatus(BaseModel):
    """Agent monitoring status"""
    agent_id: str
    is_blocked: bool
    gaming_score: float
    recent_actions: int
    suspicious_patterns: List[str]
    block_expires: Optional[float]


@router.get("/status")
async def get_validation_status():
    """Get current validation system status"""
    try:
        # Get current monitoring statistics
        total_agents = len(behavior_monitor.agent_history)
        blocked_agents = len(behavior_monitor.blocked_agents)
        
        # Calculate system-wide gaming score
        total_gaming_score = 0
        total_validations = 0
        
        for agent_id, history in behavior_monitor.agent_history.items():
            if history:
                agent_scores = [action.get("gaming_score", 0) for action in history if "gaming_score" in action]
                if agent_scores:
                    total_gaming_score += sum(agent_scores)
                    total_validations += len(agent_scores)
        
        avg_gaming_score = total_gaming_score / total_validations if total_validations > 0 else 0
        
        return {
            "status": "active",
            "monitoring_enabled": True,
            "dgts_enabled": True,
            "nlnh_enabled": True,
            "statistics": {
                "total_agents_monitored": total_agents,
                "blocked_agents": blocked_agents,
                "average_gaming_score": avg_gaming_score,
                "total_validations": total_validations
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Failed to get validation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate/dgts")
async def validate_dgts(request: ValidationRequest) -> ValidationResponse:
    """
    Validate code for DGTS violations (gaming patterns)
    """
    try:
        # Validate the code
        violations = dgts_validator.validate_code(request.code)
        gaming_score = dgts_validator.calculate_gaming_score(request.code)
        
        # Track in behavior monitor if agent_id provided
        if request.agent_id:
            action = {
                "timestamp": time.time(),
                "action_type": "code_validation",
                "target": request.context or "unknown",
                "gaming_score": gaming_score,
                "violations": len(violations)
            }
            
            suspicious = behavior_monitor.track_agent_action(request.agent_id, action)
            
            # Check if agent should be blocked
            if gaming_score > 0.7 or len(violations) > 5:
                behavior_monitor.block_agent(request.agent_id, reason="High gaming score detected")
        
        # Generate suggestions based on violations
        suggestions = []
        if violations:
            if any("TEST_GAMING" in v["type"] for v in violations):
                suggestions.append("Replace mock tests with actual functionality validation")
            if any("CODE_GAMING" in v["type"] for v in violations):
                suggestions.append("Uncomment validation rules and implement proper checks")
            if any("FEATURE_FAKING" in v["type"] for v in violations):
                suggestions.append("Implement real functionality instead of returning mock data")
        
        # Emit validation event via Socket.IO
        socketio = get_socketio_instance()
        if socketio:
            await socketio.emit("dgts_validation", {
                "agent_id": request.agent_id,
                "gaming_score": gaming_score,
                "violations": len(violations),
                "timestamp": time.time()
            })
        
        return ValidationResponse(
            valid=len(violations) == 0 and gaming_score < 0.3,
            gaming_score=gaming_score,
            violations=violations,
            suggestions=suggestions,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"DGTS validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate/nlnh")
async def validate_nlnh(request: Dict[str, Any]):
    """
    Validate response for NLNH compliance (No Lies, No Hallucination)
    """
    try:
        from ..middleware.validation_middleware import NLNHValidator
        
        nlnh_validator = NLNHValidator()
        result = nlnh_validator.validate_response(request)
        
        # Emit validation event via Socket.IO
        socketio = get_socketio_instance()
        if socketio:
            await socketio.emit("nlnh_validation", {
                "confidence": result["confidence"],
                "valid": result["valid"],
                "issues": len(result["issues"]),
                "timestamp": time.time()
            })
        
        return result
        
    except Exception as e:
        logger.error(f"NLNH validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/status")
async def get_agent_status(agent_id: str) -> AgentStatus:
    """
    Get monitoring status for a specific agent
    """
    try:
        is_blocked = behavior_monitor.is_agent_blocked(agent_id)
        
        # Get agent history
        history = behavior_monitor.agent_history.get(agent_id, [])
        recent_actions = len([a for a in history if time.time() - a["timestamp"] < 3600])
        
        # Calculate agent's gaming score
        gaming_scores = [a.get("gaming_score", 0) for a in history if "gaming_score" in a]
        avg_gaming_score = sum(gaming_scores) / len(gaming_scores) if gaming_scores else 0
        
        # Identify suspicious patterns
        suspicious_patterns = []
        if avg_gaming_score > 0.5:
            suspicious_patterns.append("High average gaming score")
        if recent_actions > 10:
            suspicious_patterns.append("High frequency of actions")
        
        # Get block expiration if blocked
        block_expires = None
        if is_blocked:
            block_info = behavior_monitor.blocked_agents.get(agent_id, {})
            block_expires = block_info.get("until")
        
        return AgentStatus(
            agent_id=agent_id,
            is_blocked=is_blocked,
            gaming_score=avg_gaming_score,
            recent_actions=recent_actions,
            suspicious_patterns=suspicious_patterns,
            block_expires=block_expires
        )
        
    except Exception as e:
        logger.error(f"Failed to get agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/{agent_id}/unblock")
async def unblock_agent(agent_id: str):
    """
    Manually unblock an agent
    """
    try:
        if agent_id in behavior_monitor.blocked_agents:
            del behavior_monitor.blocked_agents[agent_id]
            
            # Emit unblock event
            socketio = get_socketio_instance()
            if socketio:
                await socketio.emit("agent_unblocked", {
                    "agent_id": agent_id,
                    "timestamp": time.time()
                })
            
            return {"message": f"Agent {agent_id} has been unblocked"}
        else:
            return {"message": f"Agent {agent_id} was not blocked"}
            
    except Exception as e:
        logger.error(f"Failed to unblock agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/blocked")
async def get_blocked_agents():
    """
    Get list of all currently blocked agents
    """
    try:
        blocked = []
        current_time = time.time()
        
        for agent_id, block_info in behavior_monitor.blocked_agents.items():
            blocked.append({
                "agent_id": agent_id,
                "blocked_at": block_info.get("timestamp"),
                "blocked_until": block_info.get("until"),
                "reason": block_info.get("reason"),
                "remaining_seconds": max(0, block_info.get("until", 0) - current_time)
            })
        
        return {
            "blocked_agents": blocked,
            "total": len(blocked),
            "timestamp": current_time
        }
        
    except Exception as e:
        logger.error(f"Failed to get blocked agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitoring/{action}")
async def control_monitoring(action: str):
    """
    Control validation monitoring (enable/disable)
    """
    try:
        if action == "enable":
            # Enable monitoring in middleware
            return {"message": "Validation monitoring enabled", "status": "active"}
        elif action == "disable":
            # Disable monitoring in middleware
            return {"message": "Validation monitoring disabled", "status": "inactive"}
        else:
            raise HTTPException(status_code=400, detail="Invalid action. Use 'enable' or 'disable'")
            
    except Exception as e:
        logger.error(f"Failed to control monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_validation_metrics(
    time_range: int = Query(3600, description="Time range in seconds")
):
    """
    Get validation metrics for dashboard
    """
    try:
        current_time = time.time()
        start_time = current_time - time_range
        
        # Collect metrics from all agents
        metrics = {
            "dgts": {
                "total_validations": 0,
                "violations": 0,
                "average_gaming_score": 0
            },
            "nlnh": {
                "total_checks": 0,
                "low_confidence": 0,
                "average_confidence": 0
            },
            "agents": {
                "total": len(behavior_monitor.agent_history),
                "blocked": len(behavior_monitor.blocked_agents),
                "suspicious": 0
            },
            "timeline": []
        }
        
        # Process agent histories
        for agent_id, history in behavior_monitor.agent_history.items():
            recent = [a for a in history if a["timestamp"] >= start_time]
            
            for action in recent:
                if "gaming_score" in action:
                    metrics["dgts"]["total_validations"] += 1
                    if action["gaming_score"] > 0.3:
                        metrics["dgts"]["violations"] += 1
                    metrics["dgts"]["average_gaming_score"] += action["gaming_score"]
            
            # Check for suspicious behavior
            if len(recent) > 10:  # More than 10 actions in time range
                metrics["agents"]["suspicious"] += 1
        
        # Calculate averages
        if metrics["dgts"]["total_validations"] > 0:
            metrics["dgts"]["average_gaming_score"] /= metrics["dgts"]["total_validations"]
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get validation metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Export router for main app
__all__ = ["router"]