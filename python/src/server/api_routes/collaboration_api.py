"""
Real-time Collaboration API
Provides REST endpoints for collaborative coding features
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
import logging
from datetime import datetime
import json

from ...agents.collaboration.session_manager import (
    CollaborationSessionManager, 
    CollaborationSession,
    SessionSettings
)
from ...agents.collaboration.conflict_resolver import (
    ConflictResolver,
    CodeChange,
    ConflictResolutionRequest,
    ConflictResolutionResult
)
from ...agents.collaboration.awareness_engine import (
    AwarenessEngine,
    AwarenessUpdate,
    UserStatus
)
from ...agents.collaboration.sync_coordinator import (
    SyncCoordinator,
    Operation,
    OperationType
)
from ...agents.collaboration.collaborative_predictor import (
    CollaborativePredictor,
    CollaborativePredictionRequest,
    PredictionScope
)
from ..services.collaboration_service import CollaborationService, get_collaboration_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/collaboration", tags=["Collaboration"])


# Request/Response Models

class CreateSessionRequest(BaseModel):
    """Request to create a collaboration session"""
    project_id: str
    name: str
    description: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None


class JoinSessionRequest(BaseModel):
    """Request to join a collaboration session"""
    display_name: str
    avatar_url: Optional[str] = None


class ApplyChangeRequest(BaseModel):
    """Request to apply a code change"""
    file_path: str
    start_line: int
    end_line: int
    start_column: int
    end_column: int
    old_content: str
    new_content: str
    operation: str = "replace"


class InitializeDocumentRequest(BaseModel):
    """Request to initialize document for collaboration"""
    file_path: str
    initial_content: str


class ApplyOperationRequest(BaseModel):
    """Request to apply operational transformation operation"""
    file_path: str
    operation_type: str
    position: int
    length: int = 0
    content: str = ""


class CollaborativeSuggestionRequest(BaseModel):
    """Request for collaborative AI suggestions"""
    file_path: str
    code_context: str
    cursor_position: Tuple[int, int]
    current_selection: Optional[str] = None
    prediction_scope: str = "session"
    include_team_patterns: bool = True
    max_suggestions: int = 10


# Session Management Endpoints

@router.post("/sessions", response_model=Dict[str, Any])
async def create_session(
    request: CreateSessionRequest,
    owner_id: str = Query(..., description="Session owner user ID"),
    collaboration_service: CollaborationService = Depends(get_collaboration_service)
) -> Dict[str, Any]:
    """Create a new collaboration session"""
    try:
        session = await collaboration_service.session_manager.create_session(
            project_id=request.project_id,
            owner_id=owner_id,
            name=request.name,
            description=request.description,
            settings=SessionSettings(**request.settings) if request.settings else None
        )
        
        return {
            "success": True,
            "session": session.to_dict(),
            "message": f"Collaboration session '{request.name}' created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating collaboration session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/join", response_model=Dict[str, Any])
async def join_session(
    session_id: str,
    user_id: str,
    request: JoinSessionRequest,
    collaboration_service: CollaborationService = Depends(get_collaboration_service)
) -> Dict[str, Any]:
    """Join a collaboration session"""
    try:
        success = await collaboration_service.session_manager.join_session(
            session_id=session_id,
            user_id=user_id,
            display_name=request.display_name,
            avatar_url=request.avatar_url
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found or unable to join")
        
        # Also add to awareness engine
        await collaboration_service.awareness_engine.join_session(
            user_id=user_id,
            session_id=session_id,
            display_name=request.display_name,
            avatar_url=request.avatar_url
        )
        
        return {
            "success": True,
            "message": f"Successfully joined session {session_id}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error joining session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/leave", response_model=Dict[str, Any])
async def leave_session(
    session_id: str,
    user_id: str,
    collaboration_service: CollaborationService = Depends(get_collaboration_service)
) -> Dict[str, Any]:
    """Leave a collaboration session"""
    try:
        success = await collaboration_service.session_manager.leave_session(
            session_id=session_id,
            user_id=user_id
        )
        
        # Also remove from awareness engine
        await collaboration_service.awareness_engine.leave_session(user_id, session_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found or user not in session")
        
        return {
            "success": True,
            "message": f"Successfully left session {session_id}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error leaving session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", response_model=Dict[str, Any])
async def get_session(
    session_id: str,
    collaboration_service: CollaborationService = Depends(get_collaboration_service)
) -> Dict[str, Any]:
    """Get collaboration session details"""
    try:
        session = await collaboration_service.session_manager.get_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get additional session data
        awareness = await collaboration_service.awareness_engine.get_session_awareness(session_id)
        active_conflicts = await collaboration_service.conflict_resolver.get_active_conflicts(session_id)
        sync_stats = await collaboration_service.sync_coordinator.get_sync_statistics(session_id)
        
        return {
            "success": True,
            "session": session.to_dict(),
            "awareness": awareness,
            "active_conflicts": len(active_conflicts),
            "sync_statistics": sync_stats
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=Dict[str, Any])
async def list_sessions(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    project_id: Optional[str] = Query(None, description="Filter by project ID"),
    active_only: bool = Query(True, description="Show only active sessions"),
    collaboration_service: CollaborationService = Depends(get_collaboration_service)
) -> Dict[str, Any]:
    """List collaboration sessions"""
    try:
        sessions = await collaboration_service.session_manager.list_sessions(
            user_id=user_id,
            project_id=project_id,
            active_only=active_only
        )
        
        return {
            "success": True,
            "sessions": [session.to_dict() for session in sessions],
            "count": len(sessions)
        }
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Code Change Management

@router.post("/sessions/{session_id}/changes", response_model=Dict[str, Any])
async def apply_code_change(
    session_id: str,
    user_id: str,
    request: ApplyChangeRequest,
    collaboration_service: CollaborationService = Depends(get_collaboration_service)
) -> Dict[str, Any]:
    """Apply a code change to the session"""
    try:
        result = await collaboration_service.session_manager.apply_change(
            session_id=session_id,
            user_id=user_id,
            file_path=request.file_path,
            start_line=request.start_line,
            end_line=request.end_line,
            start_column=request.start_column,
            end_column=request.end_column,
            old_content=request.old_content,
            new_content=request.new_content,
            operation=request.operation
        )
        
        return {
            "success": result.success,
            "change_id": result.change_id,
            "conflicts": [conflict.conflict_id for conflict in result.conflicts],
            "message": "Change applied successfully" if result.success else "Change has conflicts"
        }
    except Exception as e:
        logger.error(f"Error applying code change: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/changes", response_model=Dict[str, Any])
async def get_session_changes(
    session_id: str,
    file_path: Optional[str] = Query(None, description="Filter by file path"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    since: Optional[str] = Query(None, description="Get changes since timestamp"),
    collaboration_service: CollaborationService = Depends(get_collaboration_service)
) -> Dict[str, Any]:
    """Get code changes for a session"""
    try:
        changes = await collaboration_service.session_manager.get_changes(
            session_id=session_id,
            file_path=file_path,
            user_id=user_id,
            since=since
        )
        
        return {
            "success": True,
            "changes": [change.to_dict() for change in changes],
            "count": len(changes)
        }
    except Exception as e:
        logger.error(f"Error getting session changes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Conflict Resolution

@router.get("/sessions/{session_id}/conflicts", response_model=Dict[str, Any])
async def get_conflicts(
    session_id: str,
    collaboration_service: CollaborationService = Depends(get_collaboration_service)
) -> Dict[str, Any]:
    """Get active conflicts for a session"""
    try:
        conflicts = await collaboration_service.conflict_resolver.get_active_conflicts(session_id)
        
        return {
            "success": True,
            "conflicts": [conflict.to_dict() if hasattr(conflict, 'to_dict') else vars(conflict) for conflict in conflicts],
            "count": len(conflicts)
        }
    except Exception as e:
        logger.error(f"Error getting conflicts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/conflicts/{conflict_id}/resolve", response_model=Dict[str, Any])
async def resolve_conflict(
    session_id: str,
    conflict_id: str,
    request: ConflictResolutionRequest,
    collaboration_service: CollaborationService = Depends(get_collaboration_service)
) -> Dict[str, Any]:
    """Resolve a specific conflict"""
    try:
        result = await collaboration_service.conflict_resolver.resolve_conflict(
            conflict_id=conflict_id,
            strategy=request.strategy,
            manual_resolution=request.manual_resolution
        )
        
        return {
            "success": result.success,
            "resolution": result.resolution,
            "strategy": result.applied_strategy.value,
            "merge_result": result.merge_result.value
        }
    except Exception as e:
        logger.error(f"Error resolving conflict: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Developer Awareness

@router.post("/sessions/{session_id}/awareness", response_model=Dict[str, Any])
async def update_awareness(
    session_id: str,
    user_id: str,
    update: AwarenessUpdate,
    collaboration_service: CollaborationService = Depends(get_collaboration_service)
) -> Dict[str, Any]:
    """Update developer awareness information"""
    try:
        success = await collaboration_service.awareness_engine.update_awareness(update)
        
        return {
            "success": success,
            "message": "Awareness updated successfully" if success else "Failed to update awareness"
        }
    except Exception as e:
        logger.error(f"Error updating awareness: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/awareness", response_model=Dict[str, Any])
async def get_session_awareness(
    session_id: str,
    collaboration_service: CollaborationService = Depends(get_collaboration_service)
) -> Dict[str, Any]:
    """Get awareness information for a session"""
    try:
        awareness = await collaboration_service.awareness_engine.get_session_awareness(session_id)
        
        return {
            "success": True,
            **awareness
        }
    except Exception as e:
        logger.error(f"Error getting session awareness: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/awareness/file", response_model=Dict[str, Any])
async def get_file_awareness(
    session_id: str,
    file_path: str = Query(..., description="File path to get awareness for"),
    collaboration_service: CollaborationService = Depends(get_collaboration_service)
) -> Dict[str, Any]:
    """Get awareness information for a specific file"""
    try:
        file_awareness = await collaboration_service.awareness_engine.get_file_awareness(
            session_id, file_path
        )
        
        return {
            "success": True,
            **file_awareness
        }
    except Exception as e:
        logger.error(f"Error getting file awareness: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Document Synchronization

@router.post("/sessions/{session_id}/documents/initialize", response_model=Dict[str, Any])
async def initialize_document(
    session_id: str,
    user_id: str,
    request: InitializeDocumentRequest,
    collaboration_service: CollaborationService = Depends(get_collaboration_service)
) -> Dict[str, Any]:
    """Initialize a document for collaborative editing"""
    try:
        document = await collaboration_service.sync_coordinator.initialize_document(
            session_id=session_id,
            file_path=request.file_path,
            initial_content=request.initial_content,
            user_id=user_id
        )
        
        return {
            "success": True,
            "document": {
                "file_path": document.file_path,
                "version": document.version,
                "checksum": document.checksum,
                "sync_state": document.sync_state.value,
                "active_editors": list(document.active_editors)
            },
            "message": f"Document {request.file_path} initialized for collaboration"
        }
    except Exception as e:
        logger.error(f"Error initializing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/operations", response_model=Dict[str, Any])
async def apply_operation(
    session_id: str,
    user_id: str,
    request: ApplyOperationRequest,
    collaboration_service: CollaborationService = Depends(get_collaboration_service)
) -> Dict[str, Any]:
    """Apply operational transformation operation"""
    try:
        operation = Operation(
            op_id=f"op_{user_id}_{datetime.now().isoformat()}",
            op_type=OperationType(request.operation_type),
            position=request.position,
            length=request.length,
            content=request.content,
            user_id=user_id
        )
        
        success = await collaboration_service.sync_coordinator.apply_operation(
            session_id=session_id,
            file_path=request.file_path,
            operation=operation,
            user_id=user_id
        )
        
        return {
            "success": success,
            "operation_id": operation.op_id,
            "message": "Operation applied successfully" if success else "Failed to apply operation"
        }
    except Exception as e:
        logger.error(f"Error applying operation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/documents", response_model=Dict[str, Any])
async def get_session_documents(
    session_id: str,
    collaboration_service: CollaborationService = Depends(get_collaboration_service)
) -> Dict[str, Any]:
    """Get all documents in a collaboration session"""
    try:
        documents = await collaboration_service.sync_coordinator.get_session_documents(session_id)
        
        document_list = []
        for doc in documents:
            document_list.append({
                "file_path": doc.file_path,
                "version": doc.version,
                "last_modified": doc.last_modified.isoformat(),
                "checksum": doc.checksum,
                "sync_state": doc.sync_state.value,
                "active_editors": list(doc.active_editors),
                "pending_operations": len(doc.pending_operations)
            })
        
        return {
            "success": True,
            "documents": document_list,
            "count": len(document_list)
        }
    except Exception as e:
        logger.error(f"Error getting session documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Collaborative AI Suggestions

@router.post("/sessions/{session_id}/suggestions", response_model=Dict[str, Any])
async def get_collaborative_suggestions(
    session_id: str,
    user_id: str,
    request: CollaborativeSuggestionRequest,
    collaboration_service: CollaborationService = Depends(get_collaboration_service)
) -> Dict[str, Any]:
    """Get AI suggestions enhanced with collaborative context"""
    try:
        prediction_request = CollaborativePredictionRequest(
            session_id=session_id,
            user_id=user_id,
            file_path=request.file_path,
            code_context=request.code_context,
            cursor_position=request.cursor_position,
            current_selection=request.current_selection,
            prediction_scope=PredictionScope(request.prediction_scope),
            include_team_patterns=request.include_team_patterns,
            max_suggestions=request.max_suggestions
        )
        
        suggestions = await collaboration_service.collaborative_predictor.get_collaborative_suggestions(
            prediction_request
        )
        
        return {
            "success": True,
            "suggestions": [suggestion.to_dict() for suggestion in suggestions],
            "count": len(suggestions),
            "session_id": session_id
        }
    except Exception as e:
        logger.error(f"Error getting collaborative suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/activity", response_model=Dict[str, Any])
async def record_team_activity(
    session_id: str,
    user_id: str,
    activity_data: Dict[str, Any] = Body(...),
    collaboration_service: CollaborationService = Depends(get_collaboration_service)
) -> Dict[str, Any]:
    """Record team activity for collaborative learning"""
    try:
        await collaboration_service.collaborative_predictor.record_team_activity(
            session_id=session_id,
            user_id=user_id,
            activity_data=activity_data
        )
        
        return {
            "success": True,
            "message": "Team activity recorded successfully"
        }
    except Exception as e:
        logger.error(f"Error recording team activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Statistics and Analytics

@router.get("/sessions/{session_id}/statistics", response_model=Dict[str, Any])
async def get_session_statistics(
    session_id: str,
    collaboration_service: CollaborationService = Depends(get_collaboration_service)
) -> Dict[str, Any]:
    """Get comprehensive statistics for a collaboration session"""
    try:
        stats = {
            "session_id": session_id,
            "awareness": await collaboration_service.awareness_engine.get_awareness_statistics(session_id),
            "conflicts": await collaboration_service.conflict_resolver.get_conflict_statistics(),
            "synchronization": await collaboration_service.sync_coordinator.get_sync_statistics(session_id),
            "collaborative_ai": await collaboration_service.collaborative_predictor.get_collaborative_statistics(session_id)
        }
        
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error getting session statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics", response_model=Dict[str, Any])
async def get_global_statistics(
    collaboration_service: CollaborationService = Depends(get_collaboration_service)
) -> Dict[str, Any]:
    """Get global collaboration statistics"""
    try:
        stats = {
            "awareness": await collaboration_service.awareness_engine.get_awareness_statistics(),
            "conflicts": await collaboration_service.conflict_resolver.get_conflict_statistics(),
            "synchronization": await collaboration_service.sync_coordinator.get_sync_statistics(),
            "collaborative_ai": await collaboration_service.collaborative_predictor.get_collaborative_statistics(),
            "sessions": {
                "total_active": len(collaboration_service.session_manager.active_sessions)
            }
        }
        
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error getting global statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health Check

@router.get("/health", response_model=Dict[str, Any])
async def health_check(
    collaboration_service: CollaborationService = Depends(get_collaboration_service)
) -> Dict[str, Any]:
    """Check collaboration service health"""
    try:
        health_status = {
            "session_manager": "healthy",
            "conflict_resolver": "healthy", 
            "awareness_engine": "healthy",
            "sync_coordinator": "healthy",
            "collaborative_predictor": "healthy",
            "redis_connection": "unknown"
        }
        
        # Check Redis connection if available
        if collaboration_service.session_manager.redis:
            try:
                await collaboration_service.session_manager.redis.ping()
                health_status["redis_connection"] = "healthy"
            except Exception:
                health_status["redis_connection"] = "unhealthy"
        
        overall_healthy = all(
            status in ["healthy", "unknown"] 
            for status in health_status.values()
        )
        
        return {
            "success": True,
            "status": "healthy" if overall_healthy else "degraded",
            "services": health_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error checking collaboration health: {e}")
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }