"""
Creative Collaboration API Routes for Phase 10: Creative AI Collaboration

Provides REST API endpoints for creative AI collaboration features including
design-thinking AI partners, human-AI pair programming, collaborative design,
and innovation acceleration.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import json
import logging
from io import StringIO

from ...agents.creative_collaboration.creative_agent_system import (
    CreativeAgentSystem, CreativeProblem, CreativeSession, CreativeAgent,
    CreativePhase, SessionParticipant
)
from ...agents.creative_collaboration.ai_collaboration_partner import (
    AICollaborationPartner, DeveloperProfile, CollaborationContext,
    DeveloperPersonality, CodingStyle, CollaborationHistory
)
from ...agents.creative_collaboration.collaborative_design_platform import (
    CollaborativeDesignPlatform, DesignCanvas, DesignElement, DesignSession,
    DesignPhase, AIDesignAgent
)
from ...agents.creative_collaboration.innovation_acceleration_engine import (
    InnovationAccelerationEngine, CreativeProblem as InnovativeProblem,
    SolutionConcept, CrossDomainInsight, BreakthroughIndicator,
    InnovationMetrics, InnovationType, ProblemComplexity, SolutionStatus
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/creative", tags=["creative-collaboration"])

# Global instances (in production, these would be dependency injected)
creative_system = CreativeAgentSystem()
ai_partner = AICollaborationPartner()
design_platform = CollaborativeDesignPlatform() 
innovation_engine = InnovationAccelerationEngine()

# Pydantic models for API

class ProblemRequest(BaseModel):
    title: str = Field(..., description="Problem title")
    description: str = Field(..., description="Detailed problem description")
    domain: str = Field(..., description="Problem domain (e.g., 'software', 'design', 'business')")
    success_criteria: List[str] = Field(default_factory=list, description="Success criteria")
    constraints: List[str] = Field(default_factory=list, description="Problem constraints")
    stakeholders: List[str] = Field(default_factory=list, description="Stakeholder list")
    complexity: Optional[str] = Field("moderate", description="Problem complexity level")

class SessionRequest(BaseModel):
    problem_id: str = Field(..., description="Problem ID to work on")
    session_type: str = Field("brainstorming", description="Session type")
    participants: List[str] = Field(default_factory=list, description="Human participants")
    duration_minutes: Optional[int] = Field(60, description="Session duration")

class DeveloperProfileRequest(BaseModel):
    developer_id: str = Field(..., description="Developer identifier")
    personality_type: str = Field(..., description="Developer personality type")
    coding_style: str = Field(..., description="Preferred coding style")
    experience_level: str = Field(..., description="Experience level")
    preferred_languages: List[str] = Field(..., description="Programming languages")
    interests: List[str] = Field(default_factory=list, description="Technical interests")

class CollaborationRequest(BaseModel):
    developer_id: str = Field(..., description="Developer identifier")
    task_description: str = Field(..., description="Current task")
    code_context: Optional[str] = Field(None, description="Current code context")
    session_goals: List[str] = Field(default_factory=list, description="Session objectives")

class DesignSessionRequest(BaseModel):
    project_name: str = Field(..., description="Design project name")
    design_brief: str = Field(..., description="Design brief description")
    target_audience: str = Field(..., description="Target audience")
    design_goals: List[str] = Field(..., description="Design objectives")
    constraints: List[str] = Field(default_factory=list, description="Design constraints")
    participants: List[str] = Field(default_factory=list, description="Session participants")

class DesignElementRequest(BaseModel):
    element_type: str = Field(..., description="Type of design element")
    content: Dict[str, Any] = Field(..., description="Element content/properties")
    position: Dict[str, float] = Field(..., description="Position coordinates")

class InnovationProblemRequest(BaseModel):
    title: str = Field(..., description="Innovation problem title")
    description: str = Field(..., description="Problem description")
    domain: str = Field(..., description="Problem domain")
    complexity: str = Field("moderate", description="Problem complexity")
    success_criteria: List[str] = Field(..., description="Success criteria")
    constraints: List[str] = Field(..., description="Constraints")
    stakeholders: List[str] = Field(..., description="Stakeholders")

# Creative Agent System Endpoints

@router.post("/problems", summary="Create creative problem")
async def create_creative_problem(problem: ProblemRequest):
    """Create a new creative problem for AI collaboration"""
    try:
        creative_problem = CreativeProblem(
            title=problem.title,
            description=problem.description,
            domain=problem.domain,
            success_criteria=problem.success_criteria,
            constraints=problem.constraints,
            stakeholders=problem.stakeholders
        )
        
        problem_id = await creative_system.register_problem(creative_problem)
        
        return {
            "problem_id": problem_id,
            "status": "created",
            "message": "Creative problem registered successfully",
            "next_steps": [
                "Start a creative session",
                "Invite human participants",
                "Begin brainstorming"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error creating creative problem: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/problems", summary="List creative problems")
async def list_creative_problems(
    domain: Optional[str] = Query(None, description="Filter by domain"),
    limit: int = Query(10, description="Limit results")
):
    """List registered creative problems"""
    try:
        problems = await creative_system.get_problems(domain=domain, limit=limit)
        
        return {
            "problems": [
                {
                    "id": p.id,
                    "title": p.title,
                    "domain": p.domain,
                    "created_at": p.created_at.isoformat(),
                    "status": p.status.value
                } for p in problems
            ],
            "total": len(problems)
        }
        
    except Exception as e:
        logger.error(f"Error listing creative problems: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions", summary="Start creative session")
async def start_creative_session(session: SessionRequest):
    """Start a new creative collaboration session"""
    try:
        session_id = await creative_system.start_creative_session(
            problem_id=session.problem_id,
            human_participants=session.participants,
            session_type=session.session_type,
            duration_minutes=session.duration_minutes
        )
        
        return {
            "session_id": session_id,
            "status": "started",
            "phase": "ideation",
            "active_agents": await creative_system.get_session_agents(session_id),
            "message": "Creative session started successfully"
        }
        
    except Exception as e:
        logger.error(f"Error starting creative session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}", summary="Get session status")
async def get_session_status(session_id: str):
    """Get current status of a creative session"""
    try:
        session = await creative_system.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        return {
            "session_id": session_id,
            "status": session.status.value,
            "current_phase": session.current_phase.value,
            "participants": [p.name for p in session.participants],
            "progress": await creative_system.get_session_progress(session_id),
            "insights": await creative_system.get_session_insights(session_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/contribute", summary="Contribute to session")
async def contribute_to_session(
    session_id: str,
    contribution: Dict[str, Any]
):
    """Add a contribution to an active creative session"""
    try:
        result = await creative_system.add_contribution(
            session_id=session_id,
            contributor=contribution.get("contributor", "human"),
            content=contribution.get("content", ""),
            contribution_type=contribution.get("type", "idea")
        )
        
        return {
            "contribution_id": result["contribution_id"],
            "status": "accepted",
            "ai_response": result.get("ai_response"),
            "session_progress": result.get("progress")
        }
        
    except Exception as e:
        logger.error(f"Error adding contribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/advance", summary="Advance session phase")
async def advance_session_phase(session_id: str):
    """Advance the session to the next creative phase"""
    try:
        result = await creative_system.advance_phase(session_id)
        
        return {
            "session_id": session_id,
            "new_phase": result["phase"],
            "phase_summary": result["summary"],
            "recommended_actions": result["recommendations"]
        }
        
    except Exception as e:
        logger.error(f"Error advancing session phase: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/export", summary="Export session results")
async def export_session_results(session_id: str, format: str = Query("json")):
    """Export creative session results in various formats"""
    try:
        results = await creative_system.export_session(session_id, format)
        
        if format == "json":
            return results
        elif format == "pdf":
            # Return PDF stream
            return StreamingResponse(
                StringIO(results),
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=session_{session_id}.pdf"}
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting session results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# AI Collaboration Partner Endpoints

@router.post("/developers/profile", summary="Create developer profile")
async def create_developer_profile(profile: DeveloperProfileRequest):
    """Create or update a developer's collaboration profile"""
    try:
        developer_profile = DeveloperProfile(
            developer_id=profile.developer_id,
            personality_type=DeveloperPersonality(profile.personality_type),
            coding_style=CodingStyle(profile.coding_style),
            experience_level=profile.experience_level,
            preferred_languages=profile.preferred_languages,
            interests=profile.interests
        )
        
        await ai_partner.create_or_update_profile(developer_profile)
        
        return {
            "developer_id": profile.developer_id,
            "status": "profile_created",
            "message": "Developer profile created successfully",
            "personalization_level": await ai_partner.calculate_personalization_readiness(profile.developer_id)
        }
        
    except Exception as e:
        logger.error(f"Error creating developer profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/developers/{developer_id}/profile", summary="Get developer profile")
async def get_developer_profile(developer_id: str):
    """Retrieve developer's collaboration profile"""
    try:
        profile = await ai_partner.get_developer_profile(developer_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Developer profile not found")
            
        return {
            "developer_id": developer_id,
            "personality_type": profile.personality_type.value,
            "coding_style": profile.coding_style.value,
            "experience_level": profile.experience_level,
            "preferred_languages": profile.preferred_languages,
            "interests": profile.interests,
            "collaboration_stats": await ai_partner.get_collaboration_stats(developer_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting developer profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/developers/{developer_id}/collaborate", summary="Start AI collaboration")
async def start_ai_collaboration(
    developer_id: str,
    collaboration: CollaborationRequest
):
    """Start an AI-assisted coding collaboration session"""
    try:
        context = CollaborationContext(
            developer_id=developer_id,
            task_description=collaboration.task_description,
            code_context=collaboration.code_context,
            session_goals=collaboration.session_goals,
            timestamp=datetime.utcnow()
        )
        
        session_id = await ai_partner.start_collaboration_session(context)
        
        return {
            "session_id": session_id,
            "developer_id": developer_id,
            "status": "collaboration_started",
            "ai_personality": await ai_partner.get_adapted_personality(developer_id),
            "initial_suggestions": await ai_partner.get_initial_suggestions(session_id)
        }
        
    except Exception as e:
        logger.error(f"Error starting AI collaboration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/collaboration/{session_id}/interact", summary="Interact with AI partner")
async def interact_with_ai_partner(
    session_id: str,
    interaction: Dict[str, Any]
):
    """Send interaction to AI collaboration partner"""
    try:
        response = await ai_partner.process_interaction(
            session_id=session_id,
            interaction_type=interaction.get("type", "question"),
            content=interaction.get("content", ""),
            context=interaction.get("context", {})
        )
        
        return {
            "session_id": session_id,
            "response_type": response["type"],
            "content": response["content"],
            "suggestions": response.get("suggestions", []),
            "follow_up_questions": response.get("follow_up_questions", [])
        }
        
    except Exception as e:
        logger.error(f"Error processing AI interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/collaboration/{session_id}/insights", summary="Get collaboration insights")
async def get_collaboration_insights(session_id: str):
    """Get insights and analytics from collaboration session"""
    try:
        insights = await ai_partner.get_session_insights(session_id)
        
        return {
            "session_id": session_id,
            "productivity_metrics": insights["productivity"],
            "learning_insights": insights["learning"],
            "collaboration_quality": insights["quality"],
            "recommendations": insights["recommendations"]
        }
        
    except Exception as e:
        logger.error(f"Error getting collaboration insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Collaborative Design Platform Endpoints

@router.post("/design/sessions", summary="Create design session")
async def create_design_session(session: DesignSessionRequest):
    """Create a new collaborative design session"""
    try:
        session_id = await design_platform.create_design_session(
            project_name=session.project_name,
            design_brief=session.design_brief,
            target_audience=session.target_audience,
            design_goals=session.design_goals,
            constraints=session.constraints,
            participants=session.participants
        )
        
        return {
            "session_id": session_id,
            "project_name": session.project_name,
            "status": "created",
            "canvas_id": await design_platform.get_session_canvas_id(session_id),
            "ai_agents": await design_platform.get_available_ai_agents()
        }
        
    except Exception as e:
        logger.error(f"Error creating design session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/design/sessions/{session_id}", summary="Get design session")
async def get_design_session(session_id: str):
    """Get design session details and current state"""
    try:
        session = await design_platform.get_design_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Design session not found")
            
        return {
            "session_id": session_id,
            "project_name": session.project_name,
            "current_phase": session.current_phase.value,
            "participants": session.participants,
            "canvas_elements": len(session.canvas.elements),
            "progress": await design_platform.get_session_progress(session_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting design session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/design/sessions/{session_id}/elements", summary="Add design element")
async def add_design_element(session_id: str, element: DesignElementRequest):
    """Add a new design element to the canvas"""
    try:
        element_id = await design_platform.add_canvas_element(
            session_id=session_id,
            element_type=element.element_type,
            content=element.content,
            position=element.position
        )
        
        # Get AI feedback on the new element
        ai_feedback = await design_platform.get_ai_element_feedback(session_id, element_id)
        
        return {
            "element_id": element_id,
            "status": "added",
            "ai_feedback": ai_feedback,
            "canvas_state": await design_platform.get_canvas_state(session_id)
        }
        
    except Exception as e:
        logger.error(f"Error adding design element: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/design/sessions/{session_id}/elements/{element_id}", summary="Update design element")
async def update_design_element(
    session_id: str,
    element_id: str,
    update: Dict[str, Any]
):
    """Update an existing design element"""
    try:
        await design_platform.update_canvas_element(
            session_id=session_id,
            element_id=element_id,
            updates=update
        )
        
        # Get AI feedback on the update
        ai_feedback = await design_platform.get_ai_element_feedback(session_id, element_id)
        
        return {
            "element_id": element_id,
            "status": "updated",
            "ai_feedback": ai_feedback
        }
        
    except Exception as e:
        logger.error(f"Error updating design element: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/design/sessions/{session_id}/elements/{element_id}", summary="Delete design element")
async def delete_design_element(session_id: str, element_id: str):
    """Delete a design element from the canvas"""
    try:
        await design_platform.delete_canvas_element(session_id, element_id)
        
        return {
            "element_id": element_id,
            "status": "deleted"
        }
        
    except Exception as e:
        logger.error(f"Error deleting design element: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/design/sessions/{session_id}/ai-suggestions", summary="Get AI design suggestions")
async def get_ai_design_suggestions(session_id: str, context: Dict[str, Any]):
    """Get AI-powered design suggestions for the current canvas state"""
    try:
        suggestions = await design_platform.get_ai_design_suggestions(
            session_id=session_id,
            suggestion_type=context.get("type", "general"),
            focus_area=context.get("focus_area"),
            constraints=context.get("constraints", [])
        )
        
        return {
            "session_id": session_id,
            "suggestions": suggestions,
            "suggestion_count": len(suggestions)
        }
        
    except Exception as e:
        logger.error(f"Error getting AI design suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/design/sessions/{session_id}/advance-phase", summary="Advance design phase")
async def advance_design_phase(session_id: str):
    """Advance the design session to the next phase"""
    try:
        result = await design_platform.advance_design_phase(session_id)
        
        return {
            "session_id": session_id,
            "new_phase": result["phase"],
            "phase_requirements": result["requirements"],
            "ai_guidance": result["guidance"]
        }
        
    except Exception as e:
        logger.error(f"Error advancing design phase: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/design/sessions/{session_id}/export", summary="Export design assets")
async def export_design_assets(
    session_id: str,
    format: str = Query("svg", description="Export format")
):
    """Export design assets in various formats"""
    try:
        assets = await design_platform.export_design_assets(session_id, format)
        
        return {
            "session_id": session_id,
            "format": format,
            "assets": assets,
            "download_urls": await design_platform.get_asset_download_urls(session_id)
        }
        
    except Exception as e:
        logger.error(f"Error exporting design assets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Innovation Acceleration Engine Endpoints

@router.post("/innovation/problems", summary="Create innovation problem")
async def create_innovation_problem(problem: InnovationProblemRequest):
    """Create a complex problem for innovation acceleration"""
    try:
        innovation_problem = InnovativeProblem(
            id=f"prob_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            title=problem.title,
            description=problem.description,
            domain=problem.domain,
            complexity=ProblemComplexity(problem.complexity),
            dimensions=[],  # Will be populated during decomposition
            success_criteria=problem.success_criteria,
            constraints=problem.constraints,
            stakeholders=problem.stakeholders
        )
        
        # Decompose the problem
        decomposition = await innovation_engine.decompose_problem(innovation_problem)
        
        return {
            "problem_id": innovation_problem.id,
            "status": "created",
            "complexity_analysis": decomposition["complexity_analysis"],
            "dimensions_identified": len(decomposition["core_dimensions"]),
            "recommended_approaches": decomposition["recommended_approaches"]
        }
        
    except Exception as e:
        logger.error(f"Error creating innovation problem: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/innovation/problems/{problem_id}/explore", summary="Explore solution space")
async def explore_solution_space(
    problem_id: str,
    depth: int = Query(3, description="Exploration depth")
):
    """Systematically explore solutions for an innovation problem"""
    try:
        solutions = await innovation_engine.explore_solution_space(problem_id, depth)
        
        return {
            "problem_id": problem_id,
            "solutions_generated": len(solutions),
            "solutions": [
                {
                    "id": sol.id,
                    "title": sol.title,
                    "approach": sol.approach,
                    "innovation_type": sol.innovation_type.value,
                    "innovation_score": sol.innovation_score,
                    "feasibility_score": sol.feasibility_score,
                    "potential_impact": sol.potential_impact
                } for sol in solutions[:10]  # Return top 10
            ],
            "exploration_depth": depth
        }
        
    except Exception as e:
        logger.error(f"Error exploring solution space: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/innovation/problems/{problem_id}/cross-domain", summary="Get cross-domain insights")
async def get_cross_domain_insights(problem_id: str):
    """Generate cross-domain insights for innovation problem"""
    try:
        if problem_id not in innovation_engine.problems:
            raise HTTPException(status_code=404, detail="Innovation problem not found")
            
        problem = innovation_engine.problems[problem_id]
        insights = await innovation_engine.generate_cross_domain_inspiration(problem)
        
        return {
            "problem_id": problem_id,
            "insights": [
                {
                    "source_domain": insight.source_domain,
                    "target_domain": insight.target_domain,
                    "principle": insight.principle,
                    "example": insight.example,
                    "applicability_score": insight.applicability_score,
                    "adaptation_notes": insight.adaptation_notes
                } for insight in insights
            ],
            "insight_count": len(insights)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating cross-domain insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/innovation/solutions/{solution_id}/analyze-breakthrough", summary="Analyze breakthrough potential")
async def analyze_breakthrough_potential(solution_id: str):
    """Analyze a solution's potential for breakthrough innovation"""
    try:
        if solution_id not in innovation_engine.solutions:
            raise HTTPException(status_code=404, detail="Solution not found")
            
        solution = innovation_engine.solutions[solution_id]
        indicators = await innovation_engine.detect_breakthrough_potential(solution)
        
        return {
            "solution_id": solution_id,
            "breakthrough_indicators": [
                {
                    "indicator_type": ind.indicator_type,
                    "confidence": ind.confidence,
                    "evidence": ind.evidence,
                    "implications": ind.implications
                } for ind in indicators
            ],
            "overall_breakthrough_score": max([ind.confidence for ind in indicators]) if indicators else 0.0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing breakthrough potential: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/innovation/solutions/{solution_id}/metrics", summary="Get innovation metrics")
async def get_innovation_metrics(solution_id: str):
    """Get comprehensive innovation metrics for a solution"""
    try:
        if solution_id not in innovation_engine.solutions:
            raise HTTPException(status_code=404, detail="Solution not found")
            
        solution = innovation_engine.solutions[solution_id]
        metrics = await innovation_engine.calculate_innovation_metrics(solution)
        
        return {
            "solution_id": solution_id,
            "metrics": {
                "novelty_score": metrics.novelty_score,
                "usefulness_score": metrics.usefulness_score,
                "elegance_score": metrics.elegance_score,
                "scalability_score": metrics.scalability_score,
                "sustainability_score": metrics.sustainability_score,
                "market_potential": metrics.market_potential,
                "technical_feasibility": metrics.technical_feasibility,
                "overall_innovation_score": metrics.overall_innovation_score
            },
            "calculated_at": metrics.calculated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting innovation metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/innovation/pipeline/optimize", summary="Optimize innovation pipeline")
async def optimize_innovation_pipeline(problem_ids: List[str]):
    """Optimize innovation pipeline across multiple problems"""
    try:
        optimization = await innovation_engine.optimize_innovation_pipeline(problem_ids)
        
        return {
            "optimization_results": optimization,
            "recommended_actions": [
                "Focus on high-potential solutions first",
                "Allocate resources based on recommendations", 
                "Monitor risk factors closely",
                "Track progress against projected outcomes"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error optimizing innovation pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics and Insights Endpoints

@router.get("/analytics/creative-sessions", summary="Get creative session analytics")
async def get_creative_session_analytics(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
):
    """Get analytics across all creative sessions"""
    try:
        analytics = await creative_system.get_session_analytics(start_date, end_date)
        
        return {
            "total_sessions": analytics["total_sessions"],
            "active_sessions": analytics["active_sessions"],
            "completion_rate": analytics["completion_rate"],
            "average_duration": analytics["average_duration"],
            "top_domains": analytics["top_domains"],
            "agent_utilization": analytics["agent_utilization"],
            "success_metrics": analytics["success_metrics"]
        }
        
    except Exception as e:
        logger.error(f"Error getting creative session analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/collaboration", summary="Get collaboration analytics")
async def get_collaboration_analytics():
    """Get analytics on human-AI collaboration patterns"""
    try:
        analytics = await ai_partner.get_collaboration_analytics()
        
        return {
            "total_developers": analytics["total_developers"],
            "active_sessions": analytics["active_sessions"],
            "personality_distribution": analytics["personality_distribution"],
            "coding_style_trends": analytics["coding_style_trends"],
            "productivity_metrics": analytics["productivity_metrics"],
            "satisfaction_scores": analytics["satisfaction_scores"]
        }
        
    except Exception as e:
        logger.error(f"Error getting collaboration analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/innovation", summary="Get innovation analytics")
async def get_innovation_analytics():
    """Get analytics on innovation acceleration performance"""
    try:
        analytics = {
            "total_problems": len(innovation_engine.problems),
            "total_solutions": len(innovation_engine.solutions),
            "breakthrough_solutions": len([s for s in innovation_engine.solutions.values() if s.innovation_score > 0.8]),
            "cross_domain_insights": len(innovation_engine.cross_domain_insights),
            "average_innovation_score": sum(s.innovation_score for s in innovation_engine.solutions.values()) / max(len(innovation_engine.solutions), 1),
            "domain_distribution": {},
            "innovation_type_distribution": {}
        }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting innovation analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", summary="Health check")
async def health_check():
    """Health check for creative collaboration services"""
    return {
        "status": "healthy",
        "services": {
            "creative_system": "operational",
            "ai_partner": "operational", 
            "design_platform": "operational",
            "innovation_engine": "operational"
        },
        "timestamp": datetime.utcnow().isoformat()
    }