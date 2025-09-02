"""
Claude Code Task Bridge API
Comprehensive endpoint for Claude Code Task tool integration
Following ARCHON OPERATIONAL MANIFEST Phase 4 requirements
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
import httpx
from datetime import datetime

from ..services.claude_code_bridge import (
    get_claude_code_bridge, 
    TaskToolRequest, 
    TaskToolResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/claude-code", tags=["claude-code-bridge"])

@router.post("/task", response_model=TaskToolResponse)
async def execute_claude_task(
    request: TaskToolRequest
) -> TaskToolResponse:
    """
    Execute a task using Claude Code Task tool integration.
    Routes to appropriate Archon agent and returns results.
    
    This is the main endpoint for Claude Code integration.
    """
    bridge = await get_claude_code_bridge()
    return await bridge.handle_task_request(request)

@router.get("/task/{task_id}", response_model=TaskToolResponse)
async def get_task_status(task_id: str) -> TaskToolResponse:
    """Get status of a Claude Code task"""
    bridge = await get_claude_code_bridge()
    if task_id not in bridge.active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return bridge.active_tasks[task_id]

@router.get("/tasks", response_model=List[TaskToolResponse])
async def list_tasks() -> List[TaskToolResponse]:
    """List all Claude Code tasks"""
    bridge = await get_claude_code_bridge()
    return list(bridge.active_tasks.values())

@router.get("/agents")
async def get_available_agents():
    """Get list of available agents for Claude Code integration"""
    bridge = await get_claude_code_bridge()
    return {
        "agents": bridge.get_available_agents(),
        "total_mapped": len(bridge.agent_mapping),
        "integration_stats": bridge.get_integration_stats()
    }

@router.get("/status")
async def get_bridge_status():
    """Get Claude Code bridge status and integration statistics"""
    bridge = await get_claude_code_bridge()
    return bridge.get_integration_stats()

@router.post("/file-trigger")
async def trigger_file_change(
    file_path: str = Query(..., description="File path that changed"),
    event_type: str = Query(default="modified", description="Type of file change")
) -> Dict[str, Any]:
    """Trigger autonomous agent spawning based on file changes"""
    bridge = await get_claude_code_bridge()
    
    if not bridge.autonomous_enabled:
        return {"message": "Autonomous workflows disabled", "triggered": False}
    
    # Handle file change through bridge
    await bridge._handle_file_change(file_path, event_type)
    
    return {
        "message": f"File change processed: {file_path}",
        "event_type": event_type,
        "triggered": True,
        "timestamp": datetime.now().isoformat()
    }

@router.post("/start-monitoring")
async def start_file_monitoring(
    project_path: str = Query(..., description="Project path to monitor")
) -> Dict[str, str]:
    """Start file monitoring for autonomous agent spawning"""
    bridge = await get_claude_code_bridge()
    await bridge.start_file_monitoring(project_path)
    
    return {
        "message": f"File monitoring started for: {project_path}",
        "observers_active": len(bridge.file_observers)
    }

@router.delete("/task/{task_id}")
async def cancel_task(task_id: str) -> Dict[str, str]:
    """Cancel a Claude Code task"""
    bridge = await get_claude_code_bridge()
    if task_id not in bridge.active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = bridge.active_tasks[task_id]
    if task.success or task.error:
        return {"message": f"Task {task_id} already completed"}
    
    # Mark as cancelled (implementation depends on async task management)
    return {"message": f"Task {task_id} cancellation requested"}

@router.post("/delegate")
async def delegate_task_to_agent(
    from_agent: str = Query(..., description="Agent making the delegation"),
    to_agent: str = Query(..., description="Target agent for delegation"),
    task: TaskToolRequest = ...
) -> TaskToolResponse:
    """Enable agent-to-agent task delegation for autonomous workflows"""
    bridge = await get_claude_code_bridge()
    
    # Add delegation context
    task.context = task.context or {}
    task.context["delegated_from"] = from_agent
    task.context["delegation_chain"] = task.context.get("delegation_chain", []) + [from_agent]
    
    logger.info(f"🔄 Agent delegation: {from_agent} → {to_agent}")
    return await bridge.handle_task_request(task)

@router.get("/health")
async def claude_code_health_check():
    """Health check specifically for Claude Code integration"""
    try:
        bridge = await get_claude_code_bridge()
        stats = bridge.get_integration_stats()
        
        return {
            "status": "healthy",
            "service": "claude-code-bridge",
            "timestamp": datetime.now().isoformat(),
            "integration_working": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Claude Code bridge health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Bridge unhealthy: {str(e)}")

# Legacy endpoint support for backward compatibility
@router.post("/task-legacy")
async def execute_claude_task_legacy(
    agent_type: str,
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    timeout: Optional[int] = 60
) -> TaskToolResponse:
    """Legacy endpoint for backward compatibility"""
    request = TaskToolRequest(
        subagent_type=agent_type,
        description=f"Legacy task for {agent_type}",
        prompt=prompt,
        context=context or {},
        timeout=timeout
    )
    
    bridge = await get_claude_code_bridge()
    return await bridge.handle_task_request(request)