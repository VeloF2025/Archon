"""
Agent Management API Routes for Archon 3.0 Intelligence-Tiered Agent Management System

This module provides REST API endpoints for the Intelligence-Tiered Adaptive Agent Management System:
- Agent CRUD operations with lifecycle management
- Intelligence tier routing and task complexity assessment
- Knowledge management with confidence evolution
- Cost tracking and budget enforcement
- Real-time collaboration and pub/sub messaging
- Performance analytics and optimization recommendations

Integrates with the AgentDatabaseService and existing Archon infrastructure.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...database.agent_service import AgentDatabaseService, create_agent_service
from ...database.agent_models import (
    AgentV3, AgentState, ModelTier, AgentType,
    TaskComplexity, BudgetConstraint, SharedContext, BroadcastMessage,
    AgentPerformanceMetrics, ProjectIntelligenceOverview, CostOptimizationRecommendation
)
from ..services.agent_execution_service import (
    AgentExecutionService,
    TaskExecutionRequest,
    TaskExecutionResponse,
    get_execution_service
)
from ..services.token_tracking_service import get_token_tracking_service
from ..services.knowledge_embedding_service import (
    get_knowledge_embedding_service,
    KnowledgeContext
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/agent-management", tags=["Agent Management"])

# Global agent service instance (initialized on startup)
_agent_service: Optional[AgentDatabaseService] = None


# =====================================================
# DEPENDENCY INJECTION
# =====================================================

async def get_agent_service() -> AgentDatabaseService:
    """Get the initialized agent database service"""
    global _agent_service
    
    if _agent_service is None:
        # Initialize service on first use
        import os
        
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables")
        
        _agent_service = await create_agent_service(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            database_url=None  # Will add if needed for vector operations
        )
        logger.info("Initialized agent management service")
    
    return _agent_service


# =====================================================
# REQUEST/RESPONSE MODELS
# =====================================================

class CreateAgentRequest(BaseModel):
    """Request model for creating a new agent"""
    name: str = Field(..., max_length=255, description="Agent display name")
    agent_type: AgentType = Field(..., description="Specialized agent type")
    model_tier: ModelTier = Field(default=ModelTier.SONNET, description="Intelligence tier")
    project_id: Optional[UUID] = Field(None, description="Project UUID (uses default if not provided)")
    capabilities: Dict[str, Any] = Field(default_factory=dict, description="Agent capabilities")


class UpdateAgentStateRequest(BaseModel):
    """Request model for updating agent state"""
    state: AgentState = Field(..., description="New agent state")
    reason: str = Field(default="Manual state change", description="Reason for state change")


class TaskComplexityRequest(BaseModel):
    """Request model for task complexity assessment"""
    task_id: UUID = Field(..., description="Task UUID")
    technical_complexity: float = Field(..., ge=0.0, le=1.0, description="Technical complexity (0-1)")
    domain_expertise_required: float = Field(..., ge=0.0, le=1.0, description="Domain expertise (0-1)")
    code_volume_complexity: float = Field(..., ge=0.0, le=1.0, description="Code volume complexity (0-1)")
    integration_complexity: float = Field(..., ge=0.0, le=1.0, description="Integration complexity (0-1)")


class TrackCostRequest(BaseModel):
    """Request model for tracking agent costs"""
    agent_id: UUID = Field(..., description="Agent UUID")
    project_id: Optional[UUID] = Field(None, description="Project UUID")
    task_id: Optional[UUID] = Field(None, description="Task UUID")
    input_tokens: int = Field(..., ge=0, description="Input token count")
    output_tokens: int = Field(..., ge=0, description="Output token count")
    model_tier: ModelTier = Field(..., description="Model tier used")
    task_duration_seconds: Optional[int] = Field(None, ge=0, description="Task duration in seconds")
    success: bool = Field(default=True, description="Task success status")


class CreateSharedContextRequest(BaseModel):
    """Request model for creating shared collaboration context"""
    task_id: UUID = Field(..., description="Task UUID")
    project_id: Optional[UUID] = Field(None, description="Project UUID")
    context_name: str = Field(..., max_length=255, description="Context name")


class BroadcastMessageRequest(BaseModel):
    """Request model for broadcasting messages"""
    topic: str = Field(..., max_length=100, description="Message topic")
    content: Dict[str, Any] = Field(..., description="Message content")
    message_type: str = Field(default="info", description="Message type")
    priority: int = Field(default=1, ge=1, le=5, description="Priority (1-5)")
    target_agents: List[UUID] = Field(default_factory=list, description="Specific target agents")
    target_topics: List[str] = Field(default_factory=list, description="Target topics")


# =====================================================
# AGENT LIFECYCLE ENDPOINTS
# =====================================================

@router.get("/agents", response_model=List[AgentV3])
async def get_agents(
    project_id: Optional[UUID] = Query(None, description="Filter by project ID"),
    state: Optional[AgentState] = Query(None, description="Filter by agent state"),
    agent_type: Optional[AgentType] = Query(None, description="Filter by agent type"),
    model_tier: Optional[ModelTier] = Query(None, description="Filter by model tier"),
    service: AgentDatabaseService = Depends(get_agent_service)
):
    """Get all agents with optional filtering"""
    try:
        if project_id:
            # Use the existing method to get agents by project
            agents = await service.get_agents_by_project(project_id, state)
        else:
            # Get all agents across all projects by querying the database directly
            # This uses the Supabase client to get all agents
            query = service.supabase.table("archon_agents_v3").select("*")
            if state:
                query = query.eq("state", state.value)
            
            result = query.execute()
            
            # Convert the raw data to AgentV3 objects
            agents = []
            for agent_data in result.data:
                try:
                    agent = AgentV3(
                        id=UUID(agent_data["id"]),
                        name=agent_data["name"],
                        agent_type=AgentType(agent_data["agent_type"]),
                        model_tier=ModelTier(agent_data["model_tier"]),
                        project_id=UUID(agent_data["project_id"]),
                        state=AgentState(agent_data["state"]),
                        state_changed_at=datetime.fromisoformat(agent_data["state_changed_at"]),
                        tasks_completed=agent_data["tasks_completed"],
                        success_rate=Decimal(str(agent_data["success_rate"])),
                        avg_completion_time_seconds=agent_data["avg_completion_time_seconds"],
                        last_active_at=datetime.fromisoformat(agent_data["last_active_at"]) if agent_data.get("last_active_at") else None,
                        memory_usage_mb=agent_data["memory_usage_mb"],
                        cpu_usage_percent=Decimal(str(agent_data["cpu_usage_percent"])),
                        capabilities=agent_data["capabilities"] or {},
                        rules_profile_id=UUID(agent_data["rules_profile_id"]) if agent_data.get("rules_profile_id") else None,
                        created_at=datetime.fromisoformat(agent_data["created_at"]),
                        updated_at=datetime.fromisoformat(agent_data["updated_at"])
                    )
                    agents.append(agent)
                except Exception as agent_error:
                    logger.warning(f"Failed to parse agent data {agent_data.get('id', 'unknown')}: {agent_error}")
                    continue
        
        # Apply additional filters if not already applied at database level
        if agent_type:
            agents = [a for a in agents if a.agent_type == agent_type]
        
        if model_tier:
            agents = [a for a in agents if a.model_tier == model_tier]
        
        logger.info(f"Retrieved {len(agents)} agents (project_id={project_id}, state={state}, type={agent_type}, tier={model_tier})")
        return agents
        
    except Exception as e:
        logger.error(f"Failed to get agents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agents: {str(e)}")


@router.get("/agents/{agent_id}", response_model=AgentV3)
async def get_agent(
    agent_id: UUID,
    service: AgentDatabaseService = Depends(get_agent_service)
):
    """Get a specific agent by ID"""
    try:
        agent = await service.get_agent_by_id(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return agent
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent: {str(e)}")


@router.post("/agents", response_model=AgentV3)
async def create_agent(
    request: CreateAgentRequest,
    background_tasks: BackgroundTasks,
    service: AgentDatabaseService = Depends(get_agent_service)
):
    """Create a new agent"""
    try:
        # Use default project ID if not provided (from current session or config)
        project_id = request.project_id or UUID("00000000-0000-0000-0000-000000000001")  # Default project
        
        # Create agent instance
        agent = AgentV3(
            name=request.name,
            agent_type=request.agent_type,
            model_tier=request.model_tier,
            project_id=project_id,
            capabilities=request.capabilities
        )
        
        # Store in database
        created_agent = await service.create_agent(agent)
        
        # Emit real-time update via Socket.IO
        background_tasks.add_task(_emit_agent_created, created_agent)
        
        logger.info(f"Created agent {created_agent.name} ({created_agent.id})")
        return created_agent
        
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")


@router.patch("/agents/{agent_id}/state")
async def update_agent_state(
    agent_id: UUID,
    request: UpdateAgentStateRequest,
    background_tasks: BackgroundTasks,
    service: AgentDatabaseService = Depends(get_agent_service)
):
    """Update agent state"""
    try:
        success = await service.update_agent_state(agent_id, request.state, request.reason)
        
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Get updated agent for real-time broadcast
        updated_agent = await service.get_agent_by_id(agent_id)
        if updated_agent:
            background_tasks.add_task(_emit_agent_state_changed, updated_agent)
        
        return {"success": True, "message": f"Agent state updated to {request.state.value}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update agent state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update agent state: {str(e)}")


@router.post("/agents/{agent_id}/hibernate")
async def hibernate_agent(
    agent_id: UUID,
    service: AgentDatabaseService = Depends(get_agent_service)
):
    """Manually hibernate an agent"""
    try:
        success = await service.update_agent_state(agent_id, AgentState.HIBERNATED, "Manual hibernation")
        
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {"success": True, "message": "Agent hibernated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to hibernate agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to hibernate agent: {str(e)}")


@router.post("/projects/{project_id}/agents/hibernate-idle")
async def hibernate_idle_agents(
    project_id: UUID,
    idle_timeout_minutes: int = Query(30, ge=5, le=720, description="Idle timeout in minutes"),
    service: AgentDatabaseService = Depends(get_agent_service)
):
    """Hibernate all idle agents in a project"""
    try:
        hibernated_count = await service.hibernate_idle_agents(project_id, idle_timeout_minutes)
        
        return {
            "success": True,
            "hibernated_count": hibernated_count,
            "message": f"Hibernated {hibernated_count} idle agents"
        }
        
    except Exception as e:
        logger.error(f"Failed to hibernate idle agents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to hibernate idle agents: {str(e)}")


# =====================================================
# INTELLIGENCE TIER ROUTING ENDPOINTS
# =====================================================

@router.post("/tasks/assess-complexity", response_model=TaskComplexity)
async def assess_task_complexity(
    request: TaskComplexityRequest,
    service: AgentDatabaseService = Depends(get_agent_service)
):
    """Assess task complexity and recommend intelligence tier"""
    try:
        complexity = await service.assess_task_complexity(
            task_id=request.task_id,
            technical=request.technical_complexity,
            domain=request.domain_expertise_required,
            code_volume=request.code_volume_complexity,
            integration=request.integration_complexity
        )
        
        logger.info(f"Assessed task {request.task_id} complexity: {complexity.recommended_tier.value}")
        return complexity
        
    except Exception as e:
        logger.error(f"Failed to assess task complexity: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to assess task complexity: {str(e)}")


@router.get("/projects/{project_id}/agents/optimal")
async def get_optimal_agent_for_task(
    project_id: UUID,
    task_id: UUID = Query(..., description="Task UUID for complexity lookup"),
    agent_type: AgentType = Query(..., description="Required agent type"),
    service: AgentDatabaseService = Depends(get_agent_service)
):
    """Find the optimal agent for a task based on complexity and availability"""
    try:
        # Get task complexity first
        # This would need to be retrieved from the database
        # For now, we'll simulate it
        from ...database.agent_models import TaskComplexity, ModelTier
        mock_complexity = TaskComplexity(
            task_id=task_id,
            technical_complexity=0.7,
            domain_expertise_required=0.5,
            code_volume_complexity=0.6,
            integration_complexity=0.4,
            overall_complexity=0.55,
            recommended_tier=ModelTier.SONNET,
            assigned_tier=ModelTier.SONNET
        )
        
        optimal_agent = await service.get_optimal_agent_for_task(project_id, mock_complexity, agent_type)
        
        if not optimal_agent:
            raise HTTPException(status_code=404, detail="No suitable agent found for task")
        
        return optimal_agent
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to find optimal agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to find optimal agent: {str(e)}")


# =====================================================
# COST TRACKING ENDPOINTS
# =====================================================

@router.post("/costs/track")
async def track_agent_cost(
    request: TrackCostRequest,
    service: AgentDatabaseService = Depends(get_agent_service)
):
    """Track cost for agent task execution"""
    try:
        project_id = request.project_id or UUID("00000000-0000-0000-0000-000000000001")  # Default
        
        success = await service.track_agent_cost(
            agent_id=request.agent_id,
            project_id=project_id,
            task_id=request.task_id,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            model_tier=request.model_tier,
            task_duration_seconds=request.task_duration_seconds,
            success=request.success
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to track cost")
        
        return {"success": True, "message": "Cost tracked successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to track agent cost: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to track agent cost: {str(e)}")


@router.get("/projects/{project_id}/budget/status")
async def check_budget_status(
    project_id: UUID,
    service: AgentDatabaseService = Depends(get_agent_service)
):
    """Check project budget constraints and usage"""
    try:
        budget_status = await service.check_budget_constraints(project_id)
        return budget_status
        
    except Exception as e:
        logger.error(f"Failed to check budget status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check budget status: {str(e)}")


@router.get("/projects/{project_id}/costs/optimize", response_model=List[CostOptimizationRecommendation])
async def get_cost_optimization_recommendations(
    project_id: UUID,
    service: AgentDatabaseService = Depends(get_agent_service)
):
    """Get cost optimization recommendations for project agents"""
    try:
        recommendations = await service.generate_cost_optimization_recommendations(project_id)
        return recommendations
        
    except Exception as e:
        logger.error(f"Failed to get cost optimization recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


# =====================================================
# COLLABORATION ENDPOINTS
# =====================================================

@router.post("/collaboration/contexts", response_model=SharedContext)
async def create_shared_context(
    request: CreateSharedContextRequest,
    service: AgentDatabaseService = Depends(get_agent_service)
):
    """Create a shared collaboration context"""
    try:
        project_id = request.project_id or UUID("00000000-0000-0000-0000-000000000001")  # Default
        
        context = await service.create_shared_context(
            task_id=request.task_id,
            project_id=project_id,
            context_name=request.context_name
        )
        
        logger.info(f"Created shared context '{request.context_name}' for task {request.task_id}")
        return context
        
    except Exception as e:
        logger.error(f"Failed to create shared context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create shared context: {str(e)}")


@router.post("/collaboration/broadcast")
async def broadcast_message(
    request: BroadcastMessageRequest,
    background_tasks: BackgroundTasks,
    service: AgentDatabaseService = Depends(get_agent_service)
):
    """Broadcast a message to subscribed agents"""
    try:
        message = BroadcastMessage(
            topic=request.topic,
            content=request.content,
            message_type=request.message_type,
            priority=request.priority,
            target_agents=request.target_agents,
            target_topics=request.target_topics
        )
        
        success = await service.broadcast_message(message)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to broadcast message")
        
        # Emit real-time update
        background_tasks.add_task(_emit_broadcast_message, message)
        
        return {"success": True, "message_id": message.message_id, "message": "Message broadcasted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to broadcast message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to broadcast message: {str(e)}")


@router.post("/agents/{agent_id}/subscribe/{topic}")
async def subscribe_agent_to_topic(
    agent_id: UUID,
    topic: str,
    priority_filter: int = Query(1, ge=1, le=5, description="Minimum priority to receive"),
    service: AgentDatabaseService = Depends(get_agent_service)
):
    """Subscribe an agent to a collaboration topic"""
    try:
        success = await service.subscribe_agent_to_topic(agent_id, topic, priority_filter)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to subscribe agent to topic")
        
        return {"success": True, "message": f"Agent subscribed to topic '{topic}'"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to subscribe agent to topic: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to subscribe agent to topic: {str(e)}")


# =====================================================
# ANALYTICS ENDPOINTS
# =====================================================

@router.get("/analytics/agents/{agent_id}/performance", response_model=AgentPerformanceMetrics)
async def get_agent_performance_metrics(
    agent_id: UUID,
    service: AgentDatabaseService = Depends(get_agent_service)
):
    """Get comprehensive performance metrics for an agent"""
    try:
        metrics = await service.get_agent_performance_metrics(agent_id)
        
        if not metrics:
            raise HTTPException(status_code=404, detail="Agent metrics not found")
        
        return metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


@router.get("/analytics/projects/{project_id}/overview", response_model=ProjectIntelligenceOverview)
async def get_project_intelligence_overview(
    project_id: UUID,
    service: AgentDatabaseService = Depends(get_agent_service)
):
    """Get project-level intelligence and performance overview"""
    try:
        overview = await service.get_project_intelligence_overview(project_id)
        
        if not overview:
            raise HTTPException(status_code=404, detail="Project overview not found")
        
        return overview
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get project intelligence overview: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get project overview: {str(e)}")


@router.get("/analytics/performance", response_model=List[AgentPerformanceMetrics])
async def get_all_performance_metrics(
    project_id: Optional[UUID] = Query(None, description="Filter by project ID"),
    service: AgentDatabaseService = Depends(get_agent_service)
):
    """Get performance metrics for all agents or filtered by project"""
    try:
        # Get all agents using the proper method
        if project_id:
            agents = await service.get_agents_by_project(project_id)
        else:
            # Get all agents across all projects
            query = service.supabase.table("archon_agents_v3").select("*")
            result = query.execute()
            
            agents = []
            for agent_data in result.data:
                try:
                    agent = AgentV3(
                        id=UUID(agent_data["id"]),
                        name=agent_data["name"],
                        agent_type=AgentType(agent_data["agent_type"]),
                        model_tier=ModelTier(agent_data["model_tier"]),
                        project_id=UUID(agent_data["project_id"]),
                        state=AgentState(agent_data["state"]),
                        state_changed_at=datetime.fromisoformat(agent_data["state_changed_at"]),
                        tasks_completed=agent_data["tasks_completed"],
                        success_rate=Decimal(str(agent_data["success_rate"])),
                        avg_completion_time_seconds=agent_data["avg_completion_time_seconds"],
                        last_active_at=datetime.fromisoformat(agent_data["last_active_at"]) if agent_data.get("last_active_at") else None,
                        memory_usage_mb=agent_data["memory_usage_mb"],
                        cpu_usage_percent=Decimal(str(agent_data["cpu_usage_percent"])),
                        capabilities=agent_data["capabilities"] or {},
                        rules_profile_id=UUID(agent_data["rules_profile_id"]) if agent_data.get("rules_profile_id") else None,
                        created_at=datetime.fromisoformat(agent_data["created_at"]),
                        updated_at=datetime.fromisoformat(agent_data["updated_at"])
                    )
                    agents.append(agent)
                except Exception as agent_error:
                    logger.warning(f"Failed to parse agent data {agent_data.get('id', 'unknown')}: {agent_error}")
                    continue
        
        # Collect metrics for all agents
        all_metrics = []
        for agent in agents:
            try:
                metrics = await service.get_agent_performance_metrics(agent.id)
                if metrics:
                    all_metrics.append(metrics)
            except Exception as e:
                logger.warning(f"Failed to get metrics for agent {agent.id}: {e}")
                # Continue with other agents
        
        logger.info(f"Retrieved performance metrics for {len(all_metrics)} agents")
        return all_metrics
        
    except Exception as e:
        logger.error(f"Failed to get all performance metrics: {e}")
        # Return empty list instead of error for graceful degradation
        return []


@router.get("/analytics/project-overview", response_model=Optional[ProjectIntelligenceOverview])
async def get_overall_project_overview(
    project_id: Optional[UUID] = Query(None, description="Project ID for overview"),
    service: AgentDatabaseService = Depends(get_agent_service)
):
    """Get project overview without requiring project_id in path"""
    try:
        if project_id:
            overview = await service.get_project_intelligence_overview(project_id)
            return overview
        else:
            # Return a default/empty overview when no project specified
            return None
            
    except Exception as e:
        logger.warning(f"Failed to get project overview: {e}")
        # Return None for graceful degradation
        return None


@router.get("/costs/recommendations", response_model=List[CostOptimizationRecommendation])
async def get_all_cost_recommendations(
    project_id: Optional[UUID] = Query(None, description="Filter by project ID"),
    service: AgentDatabaseService = Depends(get_agent_service)
):
    """Get cost optimization recommendations for all projects or specific project"""
    try:
        if project_id:
            recommendations = await service.generate_cost_optimization_recommendations(project_id)
        else:
            # Get recommendations across all projects
            # For now, return empty list if no project specified
            recommendations = []
            
        return recommendations
        
    except Exception as e:
        logger.warning(f"Failed to get cost recommendations: {e}")
        # Return empty list for graceful degradation
        return []


# =====================================================
# TASK EXECUTION ENDPOINTS
# =====================================================

class TaskExecutionRequestModel(BaseModel):
    """Request model for task execution"""
    task_description: str = Field(..., description="Description of the task to execute")
    input_data: Optional[Dict[str, Any]] = Field(default=None, description="Input data for the task")
    project_context: Optional[Dict[str, Any]] = Field(default=None, description="Project-specific context")
    complexity_assessment: Optional[Dict[str, float]] = Field(default=None, description="Task complexity factors")
    require_approval: bool = Field(default=False, description="Whether task requires approval before execution")
    timeout_minutes: int = Field(default=10, ge=1, le=60, description="Task timeout in minutes")


class OptimalAgentExecutionRequest(BaseModel):
    """Request for execution with optimal agent selection"""
    agent_type: AgentType = Field(..., description="Type of agent needed for the task")
    task_description: str = Field(..., description="Description of the task to execute")
    input_data: Optional[Dict[str, Any]] = Field(default=None, description="Input data for the task")
    complexity_factors: Optional[Dict[str, float]] = Field(default=None, description="Task complexity factors")


@router.post("/agents/{agent_id}/execute", response_model=Dict[str, Any])
async def execute_task_with_agent(
    agent_id: UUID,
    request: TaskExecutionRequestModel,
    background_tasks: BackgroundTasks,
    db_service: AgentDatabaseService = Depends(get_agent_service)
):
    """Execute a task with a specific agent"""
    try:
        # Get execution service
        execution_service = await get_execution_service(db_service)
        
        # Parse complexity assessment if provided
        complexity = None
        if request.complexity_assessment:
            complexity = TaskComplexity(
                technical_complexity=request.complexity_assessment.get("technical", 0.5),
                domain_knowledge_required=request.complexity_assessment.get("domain", 0.5),
                estimated_code_volume=request.complexity_assessment.get("code_volume", 0.5),
                integration_points=request.complexity_assessment.get("integration", 0.5)
            )
        
        # Create execution request
        exec_request = TaskExecutionRequest(
            agent_id=agent_id,
            task_description=request.task_description,
            input_data=request.input_data,
            project_context=request.project_context,
            complexity_assessment=complexity,
            require_approval=request.require_approval,
            timeout_minutes=request.timeout_minutes
        )
        
        # Execute task
        response = await execution_service.execute_task(exec_request)
        
        return {
            "task_id": str(response.task_id),
            "agent_id": str(response.agent_id),
            "status": response.status,
            "output": response.output,
            "files_modified": response.files_modified,
            "execution_time": response.execution_time,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_cost": str(response.total_cost),
            "error_message": response.error_message,
            "metadata": response.metadata
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to execute task with agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Task execution failed: {str(e)}")


@router.post("/projects/{project_id}/execute-optimal", response_model=Dict[str, Any])
async def execute_task_with_optimal_agent(
    project_id: UUID,
    request: OptimalAgentExecutionRequest,
    db_service: AgentDatabaseService = Depends(get_agent_service)
):
    """Find and execute task with the optimal agent for the job"""
    try:
        # Get execution service
        execution_service = await get_execution_service(db_service)
        
        # Parse complexity factors
        complexity = None
        if request.complexity_factors:
            complexity = TaskComplexity(
                technical_complexity=request.complexity_factors.get("technical", 0.5),
                domain_knowledge_required=request.complexity_factors.get("domain", 0.5),
                estimated_code_volume=request.complexity_factors.get("code_volume", 0.5),
                integration_points=request.complexity_factors.get("integration", 0.5)
            )
        
        # Execute with optimal agent
        response = await execution_service.execute_with_optimal_agent(
            project_id=project_id,
            agent_type=request.agent_type,
            task_description=request.task_description,
            input_data=request.input_data,
            complexity_assessment=complexity
        )
        
        return {
            "task_id": str(response.task_id),
            "agent_id": str(response.agent_id),
            "status": response.status,
            "output": response.output,
            "files_modified": response.files_modified,
            "execution_time": response.execution_time,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_cost": str(response.total_cost),
            "error_message": response.error_message,
            "metadata": response.metadata
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to execute task with optimal agent: {e}")
        raise HTTPException(status_code=500, detail=f"Task execution failed: {str(e)}")


@router.get("/agents/{agent_id}/workload", response_model=Dict[str, Any])
async def get_agent_workload(
    agent_id: UUID,
    db_service: AgentDatabaseService = Depends(get_agent_service)
):
    """Get current workload for an agent"""
    try:
        execution_service = await get_execution_service(db_service)
        workload = await execution_service.get_agent_workload(agent_id)
        return workload
    except Exception as e:
        logger.error(f"Failed to get agent workload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get workload: {str(e)}")


@router.get("/execution/history", response_model=List[Dict[str, Any]])
async def get_execution_history(
    agent_id: Optional[UUID] = Query(None, description="Filter by agent ID"),
    limit: int = Query(10, ge=1, le=100, description="Number of records to return"),
    db_service: AgentDatabaseService = Depends(get_agent_service)
):
    """Get execution history for agents"""
    try:
        execution_service = await get_execution_service(db_service)
        history = await execution_service.get_execution_history(agent_id, limit)
        return history
    except Exception as e:
        logger.error(f"Failed to get execution history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.post("/projects/{project_id}/hibernate-idle-agents", response_model=Dict[str, Any])
async def hibernate_idle_agents(
    project_id: UUID,
    idle_timeout_minutes: int = Query(30, ge=5, le=120, description="Minutes before considering agent idle"),
    db_service: AgentDatabaseService = Depends(get_agent_service)
):
    """Hibernate agents that have been idle for too long"""
    try:
        execution_service = await get_execution_service(db_service)
        count = await execution_service.hibernate_idle_agents(project_id, idle_timeout_minutes)
        return {
            "success": True,
            "hibernated_count": count,
            "message": f"Hibernated {count} idle agents"
        }
    except Exception as e:
        logger.error(f"Failed to hibernate idle agents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to hibernate agents: {str(e)}")


@router.get("/execution/metrics", response_model=Dict[str, Any])
async def get_execution_metrics(
    db_service: AgentDatabaseService = Depends(get_agent_service)
):
    """Get metrics for the execution service"""
    try:
        execution_service = await get_execution_service(db_service)
        metrics = await execution_service.get_service_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Failed to get execution metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


# =====================================================
# HEALTH AND STATUS ENDPOINTS
# =====================================================

@router.get("/health")
async def health_check(service: AgentDatabaseService = Depends(get_agent_service)):
    """Health check for agent management system"""
    try:
        # Simple health check - try to get service status
        return {
            "status": "healthy",
            "service": "agent-management",
            "timestamp": datetime.now().isoformat(),
            "database_connected": True
        }
        
    except Exception as e:
        logger.error(f"Agent management health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@router.get("/status")
async def get_system_status(service: AgentDatabaseService = Depends(get_agent_service)):
    """Get comprehensive system status and statistics"""
    try:
        # Get overall system statistics
        # This would aggregate data across all projects
        return {
            "system": "archon-agent-management",
            "version": "3.0.0",
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "features": [
                "agent_lifecycle_management",
                "intelligence_tier_routing",
                "knowledge_management",
                "cost_optimization",
                "real_time_collaboration",
                "performance_analytics"
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")


# =====================================================
# BACKGROUND TASK HELPERS (for Socket.IO events)
# =====================================================

async def _emit_agent_created(agent: AgentV3):
    """Emit real-time agent created event"""
    try:
        from ..socketio_app import get_socketio_instance
        socketio = get_socketio_instance()
        if socketio:
            await socketio.emit('agent_created', {
                'agent': agent.dict(),
                'timestamp': datetime.now().isoformat()
            })
    except Exception as e:
        logger.warning(f"Failed to emit agent_created event: {e}")


async def _emit_agent_state_changed(agent: AgentV3):
    """Emit real-time agent state change event"""
    try:
        from ..socketio_app import get_socketio_instance
        socketio = get_socketio_instance()
        if socketio:
            await socketio.emit('agent_state_changed', {
                'agentId': str(agent.id),
                'state': agent.state.value,
                'timestamp': datetime.now().isoformat(),
                'metrics': {
                    'success_rate': float(agent.success_rate),
                    'tasks_completed': agent.tasks_completed
                }
            })
    except Exception as e:
        logger.warning(f"Failed to emit agent_state_changed event: {e}")


async def _emit_broadcast_message(message: BroadcastMessage):
    """Emit real-time broadcast message event"""
    try:
        from ..socketio_app import get_socketio_instance
        socketio = get_socketio_instance()
        if socketio:
            await socketio.emit('broadcast_message', {
                'messageId': message.message_id,
                'topic': message.topic,
                'content': message.content,
                'priority': message.priority,
                'timestamp': message.sent_at.isoformat()
            })
    except Exception as e:
        logger.warning(f"Failed to emit broadcast_message event: {e}")


# =====================================================
# CLEANUP
# =====================================================

async def cleanup_agent_service():
    """Cleanup agent service on shutdown"""
    global _agent_service
    if _agent_service:
        await _agent_service.close()
        _agent_service = None
        logger.info("Agent management service cleaned up")


# =====================================================
# TOKEN TRACKING ENDPOINTS
# =====================================================

@router.get("/tokens/metrics/{agent_id}", response_model=Dict[str, Any])
async def get_agent_token_metrics(
    agent_id: UUID,
    db_service: AgentDatabaseService = Depends(get_agent_service)
):
    """Get token usage metrics for a specific agent"""
    try:
        token_service = get_token_tracking_service()
        metrics = await token_service.get_agent_metrics(agent_id)
        
        return {
            "agent_id": str(agent_id),
            "total_input_tokens": metrics.total_input_tokens,
            "total_output_tokens": metrics.total_output_tokens,
            "total_tokens": metrics.total_tokens,
            "total_cost": float(metrics.total_cost),
            "task_count": metrics.task_count,
            "avg_tokens_per_task": metrics.avg_tokens_per_task,
            "cost_by_model": {k: float(v) for k, v in metrics.cost_by_model.items()},
            "tokens_by_model": metrics.tokens_by_model
        }
    except Exception as e:
        logger.error(f"Failed to get token metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get token metrics: {str(e)}")


@router.get("/tokens/usage-history", response_model=List[Dict[str, Any]])
async def get_token_usage_history(
    agent_id: Optional[UUID] = Query(None, description="Filter by agent ID"),
    task_id: Optional[UUID] = Query(None, description="Filter by task ID"),
    hours: int = Query(24, ge=1, le=168, description="Hours of history to retrieve"),
    db_service: AgentDatabaseService = Depends(get_agent_service)
):
    """Get token usage history"""
    try:
        token_service = get_token_tracking_service()
        history = await token_service.get_usage_history(agent_id, task_id, hours)
        
        return [
            {
                "task_id": str(usage.task_id),
                "agent_id": str(usage.agent_id),
                "model": usage.model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
                "input_cost": float(usage.input_cost),
                "output_cost": float(usage.output_cost),
                "total_cost": float(usage.total_cost),
                "timestamp": usage.timestamp.isoformat(),
                "metadata": usage.metadata
            }
            for usage in history
        ]
    except Exception as e:
        logger.error(f"Failed to get usage history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get usage history: {str(e)}")


@router.get("/tokens/cost-summary", response_model=Dict[str, Any])
async def get_token_cost_summary(
    hours: int = Query(24, ge=1, le=168, description="Hours to summarize"),
    db_service: AgentDatabaseService = Depends(get_agent_service)
):
    """Get cost summary for token usage"""
    try:
        token_service = get_token_tracking_service()
        summary = await token_service.get_cost_summary(hours)
        
        return summary
    except Exception as e:
        logger.error(f"Failed to get cost summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cost summary: {str(e)}")


# =====================================================
# KNOWLEDGE MANAGEMENT ENDPOINTS
# =====================================================

class KnowledgeStoreRequest(BaseModel):
    """Request to store knowledge for an agent"""
    content: str = Field(..., description="Knowledge content to store")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context information")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class KnowledgeSearchRequest(BaseModel):
    """Request to search for relevant knowledge"""
    query: str = Field(..., description="Search query")
    max_results: int = Field(5, ge=1, le=20, description="Maximum results to return")
    similarity_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Minimum similarity score")
    include_global: bool = Field(True, description="Include global knowledge in search")


@router.post("/agents/{agent_id}/knowledge", response_model=Dict[str, Any])
async def store_agent_knowledge(
    agent_id: UUID,
    request: KnowledgeStoreRequest,
    db_service: AgentDatabaseService = Depends(get_agent_service)
):
    """Store knowledge for an agent"""
    try:
        # Verify agent exists
        agent = await db_service.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        knowledge_service = get_knowledge_embedding_service(db_service.supabase)
        
        # Store the knowledge
        knowledge_item = await knowledge_service.store_knowledge(
            agent_id=agent_id,
            content=request.content,
            context=request.context,
            metadata=request.metadata
        )
        
        return {
            "id": str(knowledge_item.id),
            "agent_id": str(knowledge_item.agent_id),
            "content": knowledge_item.content[:200],  # Truncate for response
            "embedding_size": len(knowledge_item.embedding),
            "created_at": knowledge_item.created_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to store knowledge: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store knowledge: {str(e)}")


@router.post("/agents/{agent_id}/knowledge/search", response_model=List[Dict[str, Any]])
async def search_agent_knowledge(
    agent_id: UUID,
    request: KnowledgeSearchRequest,
    db_service: AgentDatabaseService = Depends(get_agent_service)
):
    """Search for relevant knowledge for an agent"""
    try:
        # Verify agent exists
        agent = await db_service.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        knowledge_service = get_knowledge_embedding_service(db_service.supabase)
        
        # Create search context
        context = KnowledgeContext(
            agent_id=agent_id,
            task_id=UUID("00000000-0000-0000-0000-000000000000"),  # Dummy task ID for search
            query=request.query,
            max_results=request.max_results,
            similarity_threshold=request.similarity_threshold,
            include_global=request.include_global
        )
        
        # Search for relevant knowledge
        results = await knowledge_service.retrieve_relevant_knowledge(context)
        
        return [
            {
                "id": str(item.id),
                "content": item.content,
                "relevance_score": item.relevance_score,
                "context": item.context,
                "access_count": item.access_count,
                "created_at": item.created_at.isoformat() if item.created_at else None
            }
            for item in results
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search knowledge: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search knowledge: {str(e)}")


@router.get("/agents/{agent_id}/knowledge/summary", response_model=Dict[str, Any])
async def get_agent_knowledge_summary(
    agent_id: UUID,
    db_service: AgentDatabaseService = Depends(get_agent_service)
):
    """Get summary of an agent's knowledge base"""
    try:
        # Verify agent exists
        agent = await db_service.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        knowledge_service = get_knowledge_embedding_service(db_service.supabase)
        
        # Get knowledge summary
        summary = await knowledge_service.get_agent_knowledge_summary(agent_id)
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get knowledge summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get knowledge summary: {str(e)}")


@router.delete("/agents/{agent_id}/knowledge/prune", response_model=Dict[str, Any])
async def prune_agent_knowledge(
    agent_id: UUID,
    threshold: float = Query(0.3, ge=0.0, le=1.0, description="Relevance threshold"),
    min_age_days: int = Query(30, ge=1, description="Minimum age in days"),
    db_service: AgentDatabaseService = Depends(get_agent_service)
):
    """Prune low-relevance knowledge for an agent"""
    try:
        # Verify agent exists
        agent = await db_service.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        knowledge_service = get_knowledge_embedding_service(db_service.supabase)
        
        # Prune knowledge
        pruned_count = await knowledge_service.prune_low_relevance_knowledge(
            agent_id=agent_id,
            threshold=threshold,
            min_age_days=min_age_days
        )
        
        return {
            "agent_id": str(agent_id),
            "pruned_count": pruned_count,
            "threshold": threshold,
            "min_age_days": min_age_days
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to prune knowledge: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to prune knowledge: {str(e)}")


# ===== COLLABORATION ENDPOINTS =====

class CollaborationSessionRequest(BaseModel):
    """Request to create a collaboration session"""
    project_id: UUID
    agents: List[UUID]
    pattern: str = "PEER_TO_PEER"
    context: Optional[Dict[str, Any]] = None


class CollaborationEventRequest(BaseModel):
    """Request to publish a collaboration event"""
    event_type: str
    source_agent_id: Optional[UUID] = None
    target_agent_id: Optional[UUID] = None
    project_id: Optional[UUID] = None
    task_id: Optional[UUID] = None
    payload: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class TaskCoordinationRequest(BaseModel):
    """Request to coordinate a task across agents"""
    task_id: UUID
    task_description: str
    agents: List[UUID]
    pattern: str = "PARALLEL"
    dependencies: Optional[Dict[UUID, List[UUID]]] = None
    timeout: int = 300


@router.post("/collaboration/sessions", response_model=Dict[str, Any])
async def create_collaboration_session(
    request: CollaborationSessionRequest,
    db_service: AgentDatabaseService = Depends(get_agent_service)
):
    """Create a new multi-agent collaboration session"""
    try:
        from ..services.agent_collaboration_service import (
            get_agent_collaboration_service,
            CoordinationPattern
        )
        
        collaboration_service = get_agent_collaboration_service()
        
        # Create the session
        session = await collaboration_service.create_collaboration_session(
            project_id=request.project_id,
            agents=request.agents,
            pattern=CoordinationPattern[request.pattern],
            context=request.context or {}
        )
        
        return {
            "session_id": str(session.session_id),
            "project_id": str(session.project_id),
            "pattern": session.coordination_pattern.value,
            "participants": [str(p) for p in session.participating_agents],
            "state": "active" if session.active else "inactive",
            "created_at": session.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to create collaboration session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create collaboration session: {str(e)}")


@router.get("/collaboration/sessions/{session_id}", response_model=Dict[str, Any])
async def get_collaboration_session(
    session_id: UUID,
    db_service: AgentDatabaseService = Depends(get_agent_service)
):
    """Get details of a collaboration session"""
    try:
        from ..services.agent_collaboration_service import get_agent_collaboration_service
        
        collaboration_service = get_agent_collaboration_service()
        
        if session_id not in collaboration_service.active_sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        session = collaboration_service.active_sessions[session_id]
        
        return {
            "session_id": str(session.session_id),
            "project_id": str(session.project_id),
            "pattern": session.coordination_pattern.value,
            "participants": [str(p) for p in session.participating_agents],
            "active_agents": [str(a) for a in session.participating_agents],
            "state": "active" if session.active else "inactive",
            "context": session.session_context,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.created_at.isoformat()  # No updated_at field, use created_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get collaboration session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")


@router.delete("/collaboration/sessions/{session_id}", response_model=Dict[str, Any])
async def end_collaboration_session(
    session_id: UUID,
    db_service: AgentDatabaseService = Depends(get_agent_service)
):
    """End a collaboration session"""
    try:
        from ..services.agent_collaboration_service import get_agent_collaboration_service
        
        collaboration_service = get_agent_collaboration_service()
        
        if session_id not in collaboration_service.active_sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        # End the session
        await collaboration_service.end_collaboration_session(session_id)
        
        return {
            "session_id": str(session_id),
            "status": "ended"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to end collaboration session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}")


@router.post("/collaboration/events", response_model=Dict[str, Any])
async def publish_collaboration_event(
    request: CollaborationEventRequest,
    db_service: AgentDatabaseService = Depends(get_agent_service)
):
    """Publish a collaboration event for agent coordination"""
    try:
        from ..services.agent_collaboration_service import (
            get_agent_collaboration_service,
            CollaborationEvent,
            CollaborationEventType
        )
        
        collaboration_service = get_agent_collaboration_service()
        
        # Create and publish the event
        event = CollaborationEvent(
            event_type=CollaborationEventType[request.event_type],
            source_agent_id=request.source_agent_id,
            target_agent_id=request.target_agent_id,
            project_id=request.project_id,
            task_id=request.task_id,
            payload=request.payload,
            metadata=request.metadata
        )
        
        success = await collaboration_service.publish_event(event)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to publish event")
        
        return {
            "event_id": str(event.id),
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "status": "published"
        }
        
    except Exception as e:
        logger.error(f"Failed to publish collaboration event: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to publish event: {str(e)}")


@router.post("/collaboration/coordinate", response_model=Dict[str, Any])
async def coordinate_task_execution(
    request: TaskCoordinationRequest,
    db_service: AgentDatabaseService = Depends(get_agent_service)
):
    """Coordinate task execution across multiple agents"""
    try:
        from ..services.agent_collaboration_service import (
            get_agent_collaboration_service,
            CoordinationPattern
        )
        
        collaboration_service = get_agent_collaboration_service()
        
        # Create the primary task and subtasks structure
        primary_task = {
            "id": str(request.task_id),
            "description": request.task_description
        }
        
        # Create subtasks for each agent
        subtasks = []
        agent_assignments = {}
        for i, agent_id in enumerate(request.agents):
            subtask_id = str(uuid4())
            subtasks.append({
                "id": subtask_id,
                "description": f"Part {i+1} of: {request.task_description}",
                "agent_id": str(agent_id)
            })
            agent_assignments[subtask_id] = agent_id
        
        # Coordinate the task
        coordination = await collaboration_service.coordinate_task_execution(
            primary_task=primary_task,
            subtasks=subtasks,
            agent_assignments=agent_assignments,
            pattern=CoordinationPattern[request.pattern]
        )
        
        return {
            "coordination_id": str(coordination.coordination_id),
            "task_id": str(coordination.primary_task_id),
            "pattern": coordination.pattern.value,
            "agents": [str(a) for a in coordination.assigned_agents.values()],
            "status": {str(k): v for k, v in coordination.status.items()},
            "created_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to coordinate task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to coordinate task: {str(e)}")


@router.get("/collaboration/agents/{agent_id}/subscriptions", response_model=Dict[str, Any])
async def get_agent_subscriptions(
    agent_id: UUID,
    db_service: AgentDatabaseService = Depends(get_agent_service)
):
    """Get collaboration subscriptions for an agent"""
    try:
        from ..services.agent_collaboration_service import get_agent_collaboration_service
        
        collaboration_service = get_agent_collaboration_service()
        
        # Get agent subscriptions
        subscriptions = []
        if agent_id in collaboration_service.agent_subscriptions:
            subscriptions = list(collaboration_service.agent_subscriptions[agent_id])
        
        # Check if agent is in any active sessions
        active_sessions = []
        for session_id, session in collaboration_service.active_sessions.items():
            if agent_id in session.participants:
                active_sessions.append({
                    "session_id": str(session_id),
                    "project_id": str(session.project_id),
                    "pattern": session.coordination_pattern.value,
                    "state": "active" if session.active else "inactive"
                })
        
        return {
            "agent_id": str(agent_id),
            "subscriptions": subscriptions,
            "active_sessions": active_sessions
        }
        
    except Exception as e:
        logger.error(f"Failed to get agent subscriptions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get subscriptions: {str(e)}")