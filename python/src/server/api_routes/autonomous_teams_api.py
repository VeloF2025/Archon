"""
API Routes for Phase 9 Autonomous Development Teams

This module provides RESTful API endpoints for managing autonomous development teams,
including team assembly, workflow orchestration, performance tracking, and
global knowledge network integration.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...utils import get_supabase_client
from ...agents.autonomous_teams.team_assembly import TeamAssemblyEngine, ProjectRequirements
from ...agents.autonomous_teams.workflow_orchestrator import WorkflowOrchestrator, WorkflowExecution
from ...agents.autonomous_teams.team_performance_tracker import TeamPerformanceTracker, PerformanceDataPoint, PerformanceMetric
from ...agents.autonomous_teams.cross_project_knowledge import CrossProjectKnowledgeEngine, ProjectMetrics
from ...agents.autonomous_teams.global_knowledge_network import GlobalKnowledgeNetwork, KnowledgeItem, NetworkQuery, NetworkRole

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/autonomous-teams", tags=["autonomous-teams"])

# Global instances (initialized on startup)
team_assembly_engine: Optional[TeamAssemblyEngine] = None
workflow_orchestrator: Optional[WorkflowOrchestrator] = None
performance_tracker: Optional[TeamPerformanceTracker] = None
knowledge_engine: Optional[CrossProjectKnowledgeEngine] = None
knowledge_network: Optional[GlobalKnowledgeNetwork] = None


# Pydantic Models for API

class TeamCreationRequest(BaseModel):
    """Request to create a new autonomous development team."""
    project_name: str = Field(..., description="Name of the project")
    project_description: str = Field(..., description="Detailed project description")
    requirements: Dict[str, Any] = Field(..., description="Project requirements and specifications")
    timeline_weeks: int = Field(default=12, ge=1, le=52, description="Expected project timeline in weeks")
    team_preferences: Optional[Dict[str, Any]] = Field(default=None, description="Team composition preferences")


class WorkflowStartRequest(BaseModel):
    """Request to start a workflow for a team."""
    project_id: str = Field(..., description="Project identifier")
    workflow_name: str = Field(..., description="Name for the workflow")
    template_name: str = Field(default="web_application", description="Workflow template to use")
    customizations: Optional[Dict[str, Any]] = Field(default=None, description="Workflow customizations")


class PerformanceDataRequest(BaseModel):
    """Request to record performance data."""
    team_id: str = Field(..., description="Team identifier")
    metric: str = Field(..., description="Performance metric name")
    value: float = Field(..., description="Metric value")
    project_id: Optional[str] = Field(default=None, description="Associated project")
    agent_id: Optional[str] = Field(default=None, description="Associated agent")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class KnowledgeContributionRequest(BaseModel):
    """Request to contribute knowledge to the network."""
    type: str = Field(..., description="Type of knowledge")
    title: str = Field(..., description="Knowledge title")
    description: str = Field(..., description="Knowledge description")
    content: Dict[str, Any] = Field(..., description="Knowledge content")
    domain: str = Field(..., description="Domain/category")
    technologies: List[str] = Field(..., description="Related technologies")
    complexity_level: int = Field(default=5, ge=1, le=10, description="Complexity level (1-10)")
    privacy_level: str = Field(default="anonymized", description="Privacy level for sharing")


class KnowledgeQueryRequest(BaseModel):
    """Request to query the knowledge network."""
    domain: Optional[str] = Field(default=None, description="Domain filter")
    technologies: List[str] = Field(default_factory=list, description="Technology filters")
    knowledge_types: List[str] = Field(default_factory=list, description="Knowledge type filters")
    complexity_range: tuple[int, int] = Field(default=(1, 10), description="Complexity range filter")
    min_success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum success rate")
    min_confidence: float = Field(default=0.0, ge=0.0, le=10.0, description="Minimum confidence score")
    max_results: int = Field(default=50, ge=1, le=200, description="Maximum number of results")


# Initialization

async def initialize_autonomous_teams():
    """Initialize all autonomous teams components."""
    global team_assembly_engine, workflow_orchestrator, performance_tracker, knowledge_engine, knowledge_network
    
    try:
        # Initialize team assembly engine
        team_assembly_engine = TeamAssemblyEngine()
        logger.info("Team assembly engine initialized")
        
        # Initialize workflow orchestrator
        workflow_orchestrator = WorkflowOrchestrator()
        await workflow_orchestrator.start_orchestration()
        logger.info("Workflow orchestrator initialized and started")
        
        # Initialize performance tracker
        performance_tracker = TeamPerformanceTracker()
        await performance_tracker.start_monitoring()
        logger.info("Performance tracker initialized and started")
        
        # Initialize knowledge engine
        knowledge_engine = CrossProjectKnowledgeEngine()
        logger.info("Cross-project knowledge engine initialized")
        
        # Initialize global knowledge network
        node_config = {
            "name": "Archon Development Node",
            "organization_type": "ai_development_platform",
            "role": NetworkRole.CONTRIBUTOR,
            "domains": ["web_development", "ai_development", "automation"],
            "technologies": ["Python", "TypeScript", "React", "FastAPI", "PostgreSQL"],
            "expertise_level": {
                "web_development": 9.0,
                "ai_development": 9.5,
                "automation": 8.5
            }
        }
        knowledge_network = GlobalKnowledgeNetwork(node_config)
        logger.info("Global knowledge network initialized")
        
    except Exception as e:
        logger.error(f"Error initializing autonomous teams: {e}", exc_info=True)
        raise


async def cleanup_autonomous_teams():
    """Clean up autonomous teams components."""
    global workflow_orchestrator, performance_tracker, knowledge_network
    
    try:
        if workflow_orchestrator:
            await workflow_orchestrator.stop_orchestration()
            logger.info("Workflow orchestrator stopped")
        
        if performance_tracker:
            await performance_tracker.stop_monitoring()
            logger.info("Performance tracker stopped")
        
        if knowledge_network:
            await knowledge_network.stop_network_sync()
            logger.info("Knowledge network sync stopped")
    
    except Exception as e:
        logger.error(f"Error during cleanup: {e}", exc_info=True)


# Team Assembly Endpoints

@router.post("/teams/create")
async def create_autonomous_team(request: TeamCreationRequest):
    """Create a new autonomous development team for a project."""
    
    if not team_assembly_engine:
        raise HTTPException(status_code=503, detail="Team assembly engine not initialized")
    
    try:
        # Convert request to project requirements
        requirements = ProjectRequirements(
            project_name=request.project_name,
            project_description=request.project_description,
            technical_requirements=request.requirements.get("technical", {}),
            functional_requirements=request.requirements.get("functional", {}),
            non_functional_requirements=request.requirements.get("non_functional", {}),
            constraints=request.requirements.get("constraints", {}),
            timeline_weeks=request.timeline_weeks
        )
        
        # Assemble the team
        team_composition = await team_assembly_engine.assemble_team(
            requirements=requirements,
            preferences=request.team_preferences
        )
        
        # Store team in database
        supabase = get_supabase_client()
        team_data = {
            "id": team_composition.team_id,
            "name": f"{request.project_name} Development Team",
            "project_name": request.project_name,
            "project_description": request.project_description,
            "composition": team_composition.dict(),
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        result = supabase.table("autonomous_teams").insert(team_data).execute()
        
        # Create performance profile
        if performance_tracker:
            await performance_tracker.create_team_profile(
                team_id=team_composition.team_id,
                team_name=f"{request.project_name} Development Team",
                team_members=[agent.id for agent in team_composition.agents]
            )
        
        return {
            "success": True,
            "team_composition": team_composition.dict(),
            "message": f"Successfully assembled team with {len(team_composition.agents)} agents"
        }
    
    except Exception as e:
        logger.error(f"Error creating autonomous team: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/teams/{team_id}")
async def get_team_details(team_id: str):
    """Get detailed information about an autonomous team."""
    
    try:
        supabase = get_supabase_client()
        result = supabase.table("autonomous_teams").select("*").eq("id", team_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Team not found")
        
        team_data = result.data[0]
        
        # Get performance data if available
        performance_data = None
        if performance_tracker:
            performance_data = await performance_tracker.get_team_dashboard_data(team_id)
        
        return {
            "team": team_data,
            "performance": performance_data
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting team details: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/teams")
async def list_autonomous_teams(
    status: Optional[str] = Query(None, description="Filter by team status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of teams to return")
):
    """List all autonomous development teams."""
    
    try:
        supabase = get_supabase_client()
        query = supabase.table("autonomous_teams").select("*")
        
        if status:
            query = query.eq("status", status)
        
        result = query.limit(limit).order("created_at", desc=True).execute()
        
        return {
            "teams": result.data,
            "count": len(result.data)
        }
    
    except Exception as e:
        logger.error(f"Error listing teams: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Workflow Orchestration Endpoints

@router.post("/workflows/start")
async def start_workflow(request: WorkflowStartRequest):
    """Start a new workflow for an autonomous team."""
    
    if not workflow_orchestrator:
        raise HTTPException(status_code=503, detail="Workflow orchestrator not initialized")
    
    try:
        # Create workflow
        workflow = await workflow_orchestrator.create_workflow(
            project_id=request.project_id,
            workflow_name=request.workflow_name,
            template_name=request.template_name,
            customizations=request.customizations
        )
        
        # Start workflow execution
        success = await workflow_orchestrator.start_workflow(workflow.id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to start workflow")
        
        return {
            "success": True,
            "workflow_id": workflow.id,
            "message": f"Workflow '{request.workflow_name}' started successfully",
            "estimated_duration_hours": workflow.estimated_duration_hours,
            "total_tasks": len(workflow.tasks)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting workflow: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflows/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """Get detailed status of a workflow."""
    
    if not workflow_orchestrator:
        raise HTTPException(status_code=503, detail="Workflow orchestrator not initialized")
    
    try:
        status = await workflow_orchestrator.get_workflow_status(workflow_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return status
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflows")
async def list_workflows():
    """List all active workflows."""
    
    if not workflow_orchestrator:
        raise HTTPException(status_code=503, detail="Workflow orchestrator not initialized")
    
    try:
        workflows = await workflow_orchestrator.get_all_workflows()
        
        return {
            "workflows": workflows,
            "count": len(workflows)
        }
    
    except Exception as e:
        logger.error(f"Error listing workflows: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Performance Tracking Endpoints

@router.post("/performance/record")
async def record_performance_data(request: PerformanceDataRequest):
    """Record performance data for a team."""
    
    if not performance_tracker:
        raise HTTPException(status_code=503, detail="Performance tracker not initialized")
    
    try:
        # Convert metric name to enum
        try:
            metric = PerformanceMetric(request.metric)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid metric: {request.metric}")
        
        # Create performance data point
        data_point = PerformanceDataPoint(
            team_id=request.team_id,
            metric=metric,
            value=request.value,
            project_id=request.project_id or "",
            agent_id=request.agent_id,
            context=request.context or {}
        )
        
        # Record the data
        await performance_tracker.record_performance_data(data_point)
        
        return {
            "success": True,
            "message": f"Performance data recorded for team {request.team_id}"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording performance data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/{team_id}")
async def get_team_performance(team_id: str):
    """Get comprehensive performance data for a team."""
    
    if not performance_tracker:
        raise HTTPException(status_code=503, detail="Performance tracker not initialized")
    
    try:
        dashboard_data = await performance_tracker.get_team_dashboard_data(team_id)
        
        if not dashboard_data:
            raise HTTPException(status_code=404, detail="Team performance data not found")
        
        return dashboard_data
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting team performance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/{team_id}/analytics")
async def get_team_analytics(team_id: str, days_back: int = Query(30, ge=7, le=365)):
    """Get advanced analytics for a team's performance."""
    
    if not performance_tracker:
        raise HTTPException(status_code=503, detail="Performance tracker not initialized")
    
    try:
        analytics = await performance_tracker.analyze_team_performance(team_id, days_back)
        
        if not analytics:
            raise HTTPException(status_code=404, detail="Team not found")
        
        return {
            "team_id": analytics.team_id,
            "analysis_period": {
                "start": analytics.analysis_period[0].isoformat(),
                "end": analytics.analysis_period[1].isoformat()
            },
            "patterns": analytics.performance_patterns,
            "correlations": {
                f"{k[0].value}_vs_{k[1].value}": round(v, 3)
                for k, v in analytics.metric_correlations.items()
            },
            "forecast": {
                metric.value: forecast[:7]  # Next 7 data points
                for metric, forecast in analytics.performance_forecast.items()
            },
            "ranking": analytics.team_ranking,
            "percentiles": {
                metric.value: round(percentile, 1)
                for metric, percentile in analytics.percentile_scores.items()
            },
            "recommendations": analytics.best_practice_recommendations
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting team analytics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/{team_id}/optimizations")
async def get_optimization_recommendations(team_id: str):
    """Get optimization recommendations for a team."""
    
    if not performance_tracker:
        raise HTTPException(status_code=503, detail="Performance tracker not initialized")
    
    try:
        optimizations = await performance_tracker.generate_optimization_recommendations(team_id)
        
        return {
            "team_id": team_id,
            "optimizations": [
                {
                    "type": opt.optimization_type,
                    "description": opt.description,
                    "expected_impact": round(opt.expected_impact, 2),
                    "implementation_effort": opt.implementation_effort,
                    "priority": opt.priority,
                    "success_probability": round(opt.success_probability, 2),
                    "time_to_impact_days": opt.time_to_impact,
                    "implementation_steps": opt.implementation_steps,
                    "success_metrics": opt.success_metrics
                }
                for opt in optimizations
            ],
            "count": len(optimizations)
        }
    
    except Exception as e:
        logger.error(f"Error getting optimization recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Knowledge Network Endpoints

@router.post("/knowledge/contribute")
async def contribute_knowledge(request: KnowledgeContributionRequest):
    """Contribute knowledge to the global network."""
    
    if not knowledge_network:
        raise HTTPException(status_code=503, detail="Knowledge network not initialized")
    
    try:
        # Create knowledge item
        from ...agents.autonomous_teams.global_knowledge_network import KnowledgeType, PrivacyLevel
        
        knowledge_item = KnowledgeItem(
            type=KnowledgeType(request.type),
            title=request.title,
            description=request.description,
            content=request.content,
            domain=request.domain,
            technologies=request.technologies,
            complexity_level=request.complexity_level,
            privacy_level=PrivacyLevel(request.privacy_level)
        )
        
        # Contribute to network
        success = await knowledge_network.contribute_knowledge(knowledge_item, share_globally=True)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to contribute knowledge")
        
        return {
            "success": True,
            "knowledge_id": knowledge_item.id,
            "message": "Knowledge contributed successfully"
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")
    except Exception as e:
        logger.error(f"Error contributing knowledge: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge/query")
async def query_knowledge_network(request: KnowledgeQueryRequest):
    """Query the global knowledge network."""
    
    if not knowledge_network:
        raise HTTPException(status_code=503, detail="Knowledge network not initialized")
    
    try:
        # Create network query
        from ...agents.autonomous_teams.global_knowledge_network import KnowledgeType, PrivacyLevel
        
        query = NetworkQuery(
            domain=request.domain,
            technologies=request.technologies,
            knowledge_types=[KnowledgeType(kt) for kt in request.knowledge_types] if request.knowledge_types else [],
            complexity_range=request.complexity_range,
            min_success_rate=request.min_success_rate,
            min_confidence=request.min_confidence,
            max_results=request.max_results,
            preferred_privacy_level=PrivacyLevel.ANONYMIZED
        )
        
        # Execute query
        results = await knowledge_network.query_network(query)
        
        # Convert results to API format
        knowledge_items = []
        for item in results:
            knowledge_items.append({
                "id": item.id,
                "type": item.type.value,
                "title": item.title,
                "description": item.description,
                "content": item.content,
                "domain": item.domain,
                "technologies": item.technologies,
                "complexity_level": item.complexity_level,
                "success_rate": round(item.success_rate, 3),
                "confidence_score": round(item.confidence_score, 2),
                "validation_score": round(item.validation_score, 2),
                "usage_count": item.usage_count,
                "created_at": item.created_at.isoformat()
            })
        
        return {
            "results": knowledge_items,
            "count": len(knowledge_items),
            "query_id": query.query_id
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid query: {e}")
    except Exception as e:
        logger.error(f"Error querying knowledge network: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge/{knowledge_id}/feedback")
async def provide_knowledge_feedback(
    knowledge_id: str,
    success: bool = Query(..., description="Whether the knowledge was successful"),
    details: Optional[str] = Query(None, description="Additional feedback details")
):
    """Provide feedback on knowledge usage."""
    
    if not knowledge_network:
        raise HTTPException(status_code=503, detail="Knowledge network not initialized")
    
    try:
        feedback_recorded = await knowledge_network.provide_feedback(
            knowledge_id=knowledge_id,
            success=success,
            details=details or ""
        )
        
        if not feedback_recorded:
            raise HTTPException(status_code=404, detail="Knowledge item not found")
        
        return {
            "success": True,
            "message": f"{'Positive' if success else 'Negative'} feedback recorded"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error providing feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge/network-status")
async def get_network_status():
    """Get status of the global knowledge network."""
    
    if not knowledge_network:
        raise HTTPException(status_code=503, detail="Knowledge network not initialized")
    
    try:
        status = await knowledge_network.get_network_status()
        return status
    
    except Exception as e:
        logger.error(f"Error getting network status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Analytics and Reporting Endpoints

@router.get("/analytics/overview")
async def get_analytics_overview():
    """Get overview analytics for autonomous teams."""
    
    try:
        analytics = {}
        
        # Team statistics
        supabase = get_supabase_client()
        teams_result = supabase.table("autonomous_teams").select("*").execute()
        
        analytics["teams"] = {
            "total_teams": len(teams_result.data),
            "active_teams": len([t for t in teams_result.data if t.get("status") == "active"]),
            "total_projects": len(set(t.get("project_name") for t in teams_result.data))
        }
        
        # Workflow statistics
        if workflow_orchestrator:
            workflows = await workflow_orchestrator.get_all_workflows()
            analytics["workflows"] = {
                "total_workflows": len(workflows),
                "in_progress": len([w for w in workflows if w.get("status") == "in_progress"]),
                "completed": len([w for w in workflows if w.get("status") == "completed"])
            }
        
        # Knowledge statistics
        if knowledge_network:
            network_status = await knowledge_network.get_network_status()
            analytics["knowledge"] = {
                "total_knowledge_items": network_status["network"]["total_knowledge_items"],
                "connected_nodes": network_status["network"]["connected_nodes"]
            }
        
        return analytics
    
    except Exception as e:
        logger.error(f"Error getting analytics overview: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/cross-project")
async def get_cross_project_analytics():
    """Get cross-project knowledge synthesis analytics."""
    
    if not knowledge_engine:
        raise HTTPException(status_code=503, detail="Knowledge engine not initialized")
    
    try:
        # Generate knowledge synthesis
        synthesis = await knowledge_engine.generate_knowledge_synthesis()
        
        # Get pattern statistics
        stats = await knowledge_engine.get_pattern_statistics()
        
        return {
            "synthesis": {
                "title": synthesis.title,
                "summary": synthesis.summary,
                "patterns_analyzed": synthesis.patterns_analyzed,
                "projects_analyzed": synthesis.projects_analyzed,
                "key_insights": synthesis.key_insights,
                "recommendations": synthesis.recommendations,
                "risk_factors": synthesis.risk_factors,
                "success_predictors": synthesis.success_predictors,
                "generated_at": synthesis.created_at.isoformat()
            },
            "statistics": stats
        }
    
    except Exception as e:
        logger.error(f"Error getting cross-project analytics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Health Check Endpoint

@router.get("/health")
async def health_check():
    """Health check for autonomous teams services."""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "team_assembly": team_assembly_engine is not None,
            "workflow_orchestrator": workflow_orchestrator is not None and workflow_orchestrator._running,
            "performance_tracker": performance_tracker is not None and performance_tracker._running,
            "knowledge_engine": knowledge_engine is not None,
            "knowledge_network": knowledge_network is not None
        }
    }
    
    # Check if any critical services are down
    critical_services = ["team_assembly", "workflow_orchestrator", "performance_tracker"]
    if not all(health_status["services"][service] for service in critical_services):
        health_status["status"] = "degraded"
    
    return health_status