"""
Advanced ML Models API
Provides REST endpoints for all advanced ML capabilities
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import logging
from datetime import datetime
import asyncio

from ...agents.ml_models.ml_model_coordinator import (
    MLModelCoordinator, 
    UnifiedAnalysisRequest, 
    UnifiedAnalysisResult
)
from ...agents.ml_models.neural_code_analyzer import CodeAnalysisRequest
from ...agents.ml_models.pattern_prediction_model import PatternPredictionRequest
from ...agents.ml_models.semantic_similarity_engine import SimilaritySearchRequest
from ...agents.ml_models.adaptive_learning_system import UserFeedback
from ..services.ml_models_service import MLModelsService, get_ml_models_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml-models", tags=["ML Models"])


# Request/Response Models

class CodeAnalysisApiRequest(BaseModel):
    """API request for neural code analysis"""
    code: str
    language: str = "python"
    file_path: Optional[str] = None
    analysis_types: List[str] = Field(default=["semantic_understanding", "code_quality"])
    context: Dict[str, Any] = Field(default_factory=dict)


class PatternPredictionApiRequest(BaseModel):
    """API request for pattern prediction"""
    current_code: str
    file_path: Optional[str] = None
    project_context: Dict[str, Any] = Field(default_factory=dict)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    prediction_types: List[str] = Field(default=["design_pattern", "refactoring_pattern"])
    max_predictions: int = 10


class SimilaritySearchApiRequest(BaseModel):
    """API request for similarity search"""
    query_code: str
    search_scope: str = "function_level"
    similarity_types: List[str] = Field(default=["semantic", "structural"])
    threshold: float = 0.7
    max_results: int = 10
    file_filters: List[str] = Field(default_factory=list)
    language: str = "python"


class AddCodeSegmentRequest(BaseModel):
    """Request to add code segment for similarity analysis"""
    segment_id: str
    content: str
    file_path: str
    start_line: int
    end_line: int
    segment_type: str
    language: str = "python"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UserFeedbackApiRequest(BaseModel):
    """API request for user feedback"""
    interaction_id: str
    user_id: str
    feedback_type: str  # "positive", "negative", "neutral", "corrective", "explicit_rating"
    feedback_score: Optional[float] = None
    feedback_text: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class UnifiedAnalysisApiRequest(BaseModel):
    """API request for unified ML analysis"""
    code: str
    file_path: Optional[str] = None
    project_context: Dict[str, Any] = Field(default_factory=dict)
    analysis_types: List[str] = Field(default=["all"])
    intelligence_level: str = "enhanced"  # "basic", "enhanced", "advanced", "expert"
    include_patterns: bool = True
    include_similarity: bool = True
    include_learning: bool = True
    user_id: Optional[str] = None
    max_processing_time: int = 30


# Neural Code Analysis Endpoints

@router.post("/neural-analysis", response_model=Dict[str, Any])
async def analyze_code_neural(
    request: CodeAnalysisApiRequest,
    ml_service: MLModelsService = Depends(get_ml_models_service)
) -> Dict[str, Any]:
    """Perform neural code analysis"""
    try:
        analysis_request = CodeAnalysisRequest(
            code=request.code,
            language=request.language,
            file_path=request.file_path,
            analysis_types=request.analysis_types,
            context=request.context
        )
        
        results = await ml_service.coordinator.neural_analyzer.analyze_code(analysis_request)
        
        return {
            "success": True,
            "analyses": [result.to_dict() for result in results],
            "count": len(results),
            "message": f"Completed {len(results)} neural analyses"
        }
    except Exception as e:
        logger.error(f"Error in neural code analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/neural-analysis/model-info", response_model=Dict[str, Any])
async def get_neural_model_info(
    ml_service: MLModelsService = Depends(get_ml_models_service)
) -> Dict[str, Any]:
    """Get information about the neural analyzer model"""
    try:
        model_info = await ml_service.coordinator.neural_analyzer.get_model_info()
        return {
            "success": True,
            "model_info": model_info
        }
    except Exception as e:
        logger.error(f"Error getting neural model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Pattern Prediction Endpoints

@router.post("/pattern-prediction", response_model=Dict[str, Any])
async def predict_patterns(
    request: PatternPredictionApiRequest,
    ml_service: MLModelsService = Depends(get_ml_models_service)
) -> Dict[str, Any]:
    """Predict code patterns and recommendations"""
    try:
        prediction_request = PatternPredictionRequest(
            current_code=request.current_code,
            file_path=request.file_path,
            project_context=request.project_context,
            user_preferences=request.user_preferences,
            prediction_types=request.prediction_types,
            max_predictions=request.max_predictions
        )
        
        predictions = await ml_service.coordinator.pattern_predictor.predict_patterns(prediction_request)
        
        return {
            "success": True,
            "predictions": [pred.to_dict() for pred in predictions],
            "count": len(predictions),
            "message": f"Generated {len(predictions)} pattern predictions"
        }
    except Exception as e:
        logger.error(f"Error in pattern prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pattern-prediction/statistics", response_model=Dict[str, Any])
async def get_pattern_statistics(
    ml_service: MLModelsService = Depends(get_ml_models_service)
) -> Dict[str, Any]:
    """Get pattern prediction statistics"""
    try:
        stats = await ml_service.coordinator.pattern_predictor.get_pattern_statistics()
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error getting pattern statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pattern-prediction/record-usage", response_model=Dict[str, Any])
async def record_pattern_usage(
    pattern_id: str,
    usage_context: Dict[str, Any] = Body(...),
    outcome: Dict[str, Any] = Body(...),
    ml_service: MLModelsService = Depends(get_ml_models_service)
) -> Dict[str, Any]:
    """Record pattern usage for learning"""
    try:
        await ml_service.coordinator.pattern_predictor.record_pattern_usage(
            pattern_id=pattern_id,
            usage_context=usage_context,
            outcome=outcome
        )
        
        return {
            "success": True,
            "message": f"Recorded usage for pattern: {pattern_id}"
        }
    except Exception as e:
        logger.error(f"Error recording pattern usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Semantic Similarity Endpoints

@router.post("/similarity/search", response_model=Dict[str, Any])
async def search_similar_code(
    request: SimilaritySearchApiRequest,
    ml_service: MLModelsService = Depends(get_ml_models_service)
) -> Dict[str, Any]:
    """Search for similar code segments"""
    try:
        search_request = SimilaritySearchRequest(
            query_code=request.query_code,
            search_scope=request.search_scope,
            similarity_types=request.similarity_types,
            threshold=request.threshold,
            max_results=request.max_results,
            file_filters=request.file_filters,
            language=request.language
        )
        
        matches = await ml_service.coordinator.similarity_engine.find_similar_code(search_request)
        
        return {
            "success": True,
            "matches": [match.to_dict() for match in matches],
            "count": len(matches),
            "message": f"Found {len(matches)} similar code segments"
        }
    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/similarity/add-segment", response_model=Dict[str, Any])
async def add_code_segment(
    request: AddCodeSegmentRequest,
    ml_service: MLModelsService = Depends(get_ml_models_service)
) -> Dict[str, Any]:
    """Add a code segment to the similarity engine"""
    try:
        from ...agents.ml_models.semantic_similarity_engine import CodeSegment
        
        segment = CodeSegment(
            segment_id=request.segment_id,
            content=request.content,
            file_path=request.file_path,
            start_line=request.start_line,
            end_line=request.end_line,
            segment_type=request.segment_type,
            language=request.language,
            metadata=request.metadata
        )
        
        success = await ml_service.coordinator.similarity_engine.add_code_segment(segment)
        
        return {
            "success": success,
            "message": f"Added code segment: {request.segment_id}" if success else "Failed to add code segment"
        }
    except Exception as e:
        logger.error(f"Error adding code segment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/similarity/detect-duplicates", response_model=Dict[str, Any])
async def detect_duplicates(
    scope: str = Query("function_level", description="Scope of duplicate detection"),
    threshold: float = Query(0.85, description="Similarity threshold for duplicates"),
    ml_service: MLModelsService = Depends(get_ml_models_service)
) -> Dict[str, Any]:
    """Detect duplicate code segments"""
    try:
        from ...agents.ml_models.semantic_similarity_engine import SimilarityScope
        
        scope_enum = SimilarityScope(scope)
        duplicates = await ml_service.coordinator.similarity_engine.detect_duplicates(
            scope=scope_enum,
            threshold=threshold
        )
        
        return {
            "success": True,
            "duplicates": [dup.to_dict() for dup in duplicates],
            "count": len(duplicates),
            "message": f"Found {len(duplicates)} potential duplicates"
        }
    except Exception as e:
        logger.error(f"Error detecting duplicates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/similarity/analyze-relationships", response_model=Dict[str, Any])
async def analyze_code_relationships(
    file_paths: List[str] = Body(..., description="Files to analyze relationships for"),
    ml_service: MLModelsService = Depends(get_ml_models_service)
) -> Dict[str, Any]:
    """Analyze relationships between code segments"""
    try:
        relationships = await ml_service.coordinator.similarity_engine.analyze_code_relationships(
            file_paths
        )
        
        return {
            "success": True,
            "relationships": relationships,
            "message": f"Analyzed relationships for {len(file_paths)} files"
        }
    except Exception as e:
        logger.error(f"Error analyzing code relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/similarity/statistics", response_model=Dict[str, Any])
async def get_similarity_statistics(
    ml_service: MLModelsService = Depends(get_ml_models_service)
) -> Dict[str, Any]:
    """Get similarity engine statistics"""
    try:
        stats = await ml_service.coordinator.similarity_engine.get_similarity_statistics()
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error getting similarity statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Adaptive Learning Endpoints

@router.post("/learning/feedback", response_model=Dict[str, Any])
async def submit_user_feedback(
    request: UserFeedbackApiRequest,
    ml_service: MLModelsService = Depends(get_ml_models_service)
) -> Dict[str, Any]:
    """Submit user feedback for adaptive learning"""
    try:
        feedback = UserFeedback(
            interaction_id=request.interaction_id,
            user_id=request.user_id,
            feedback_type=request.feedback_type,
            feedback_score=request.feedback_score,
            feedback_text=request.feedback_text,
            context=request.context
        )
        
        success = await ml_service.coordinator.learning_system.process_user_feedback(feedback)
        
        return {
            "success": success,
            "message": "Feedback processed successfully" if success else "Failed to process feedback"
        }
    except Exception as e:
        logger.error(f"Error processing user feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/learning/suggestions/{user_id}", response_model=Dict[str, Any])
async def get_adaptive_suggestions(
    user_id: str,
    context: Dict[str, Any] = Body(default_factory=dict),
    ml_service: MLModelsService = Depends(get_ml_models_service)
) -> Dict[str, Any]:
    """Get adaptive suggestions for a user"""
    try:
        suggestions = await ml_service.coordinator.learning_system.get_adaptive_suggestions(
            user_id, context
        )
        
        return {
            "success": True,
            "suggestions": suggestions,
            "count": len(suggestions),
            "user_id": user_id
        }
    except Exception as e:
        logger.error(f"Error getting adaptive suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/learning/adapt-models", response_model=Dict[str, Any])
async def adapt_models(
    force_adaptation: bool = Query(False, description="Force adaptation regardless of frequency limits"),
    ml_service: MLModelsService = Depends(get_ml_models_service)
) -> Dict[str, Any]:
    """Trigger model adaptation based on learned patterns"""
    try:
        adaptation_results = await ml_service.coordinator.learning_system.adapt_models(force_adaptation)
        
        return {
            "success": True,
            "adaptation_results": adaptation_results,
            "adapted_models": [model for model, success in adaptation_results.items() if success],
            "message": f"Adapted {sum(adaptation_results.values())} out of {len(adaptation_results)} models"
        }
    except Exception as e:
        logger.error(f"Error adapting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/learning/insights", response_model=Dict[str, Any])
async def get_learning_insights(
    ml_service: MLModelsService = Depends(get_ml_models_service)
) -> Dict[str, Any]:
    """Get insights about the learning process"""
    try:
        insights = await ml_service.coordinator.learning_system.get_learning_insights()
        
        return {
            "success": True,
            "insights": insights
        }
    except Exception as e:
        logger.error(f"Error getting learning insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Unified Analysis Endpoints

@router.post("/unified-analysis", response_model=Dict[str, Any])
async def perform_unified_analysis(
    request: UnifiedAnalysisApiRequest,
    ml_service: MLModelsService = Depends(get_ml_models_service)
) -> Dict[str, Any]:
    """Perform unified analysis using all ML models"""
    try:
        unified_request = UnifiedAnalysisRequest(
            code=request.code,
            file_path=request.file_path,
            project_context=request.project_context,
            analysis_types=request.analysis_types,
            intelligence_level=request.intelligence_level,
            include_patterns=request.include_patterns,
            include_similarity=request.include_similarity,
            include_learning=request.include_learning,
            user_id=request.user_id,
            max_processing_time=request.max_processing_time
        )
        
        result = await ml_service.coordinator.analyze_code_unified(unified_request)
        
        return {
            "success": True,
            "result": result.to_dict(),
            "message": f"Unified analysis completed in {result.processing_time:.2f}s using {len(result.models_used)} models"
        }
    except Exception as e:
        logger.error(f"Error in unified analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# System Status and Management Endpoints

@router.get("/system/status", response_model=Dict[str, Any])
async def get_system_status(
    ml_service: MLModelsService = Depends(get_ml_models_service)
) -> Dict[str, Any]:
    """Get comprehensive status of ML model system"""
    try:
        status = await ml_service.coordinator.get_system_status()
        
        return {
            "success": True,
            "system_status": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/models", response_model=Dict[str, Any])
async def get_registered_models(
    ml_service: MLModelsService = Depends(get_ml_models_service)
) -> Dict[str, Any]:
    """Get information about registered ML models"""
    try:
        models_info = {}
        for model_name, capability in ml_service.coordinator.registered_models.items():
            models_info[model_name] = capability.to_dict()
        
        return {
            "success": True,
            "registered_models": models_info,
            "total_models": len(models_info)
        }
    except Exception as e:
        logger.error(f"Error getting registered models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/performance", response_model=Dict[str, Any])
async def get_performance_metrics(
    model_name: Optional[str] = Query(None, description="Specific model to get metrics for"),
    days: int = Query(7, description="Number of days of history to include"),
    ml_service: MLModelsService = Depends(get_ml_models_service)
) -> Dict[str, Any]:
    """Get performance metrics for ML models"""
    try:
        performance_data = {}
        
        if model_name:
            # Get specific model performance
            if model_name in ml_service.coordinator.model_performance_history:
                history = ml_service.coordinator.model_performance_history[model_name]
                performance_data[model_name] = history[-days*24:] if days else history  # Rough daily sampling
        else:
            # Get all model performance
            for name, history in ml_service.coordinator.model_performance_history.items():
                performance_data[name] = history[-days*24:] if days else history
        
        return {
            "success": True,
            "performance_data": performance_data,
            "models_included": list(performance_data.keys()),
            "days_requested": days
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/save-state", response_model=Dict[str, Any])
async def save_learning_state(
    filepath: Optional[str] = Query(None, description="Custom filepath to save state"),
    ml_service: MLModelsService = Depends(get_ml_models_service)
) -> Dict[str, Any]:
    """Save the current learning state"""
    try:
        success = await ml_service.coordinator.learning_system.save_learning_state(filepath)
        
        return {
            "success": success,
            "message": "Learning state saved successfully" if success else "Failed to save learning state",
            "filepath": filepath or "default location"
        }
    except Exception as e:
        logger.error(f"Error saving learning state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/load-state", response_model=Dict[str, Any])
async def load_learning_state(
    filepath: Optional[str] = Query(None, description="Custom filepath to load state from"),
    ml_service: MLModelsService = Depends(get_ml_models_service)
) -> Dict[str, Any]:
    """Load learning state from file"""
    try:
        success = await ml_service.coordinator.learning_system.load_learning_state(filepath)
        
        return {
            "success": success,
            "message": "Learning state loaded successfully" if success else "Failed to load learning state",
            "filepath": filepath or "default location"
        }
    except Exception as e:
        logger.error(f"Error loading learning state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Batch Processing Endpoints

@router.post("/batch/analyze-files", response_model=Dict[str, Any])
async def batch_analyze_files(
    file_paths: List[str] = Body(..., description="List of file paths to analyze"),
    analysis_types: List[str] = Body(default=["semantic_understanding"], description="Types of analysis to perform"),
    intelligence_level: str = Body("enhanced", description="Intelligence level for analysis"),
    user_id: Optional[str] = Body(None, description="User ID for personalization"),
    ml_service: MLModelsService = Depends(get_ml_models_service)
) -> Dict[str, Any]:
    """Perform batch analysis on multiple files"""
    try:
        results = []
        failed_files = []
        
        # Limit concurrent analyses to prevent resource exhaustion
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent analyses
        
        async def analyze_single_file(file_path: str):
            async with semaphore:
                try:
                    # Read file content (simplified - in production would handle file reading)
                    # For now, we'll create a dummy request
                    request = UnifiedAnalysisRequest(
                        code=f"# Content of {file_path}\n# This would be actual file content",
                        file_path=file_path,
                        analysis_types=analysis_types,
                        intelligence_level=intelligence_level,
                        user_id=user_id
                    )
                    
                    result = await ml_service.coordinator.analyze_code_unified(request)
                    return {"file_path": file_path, "analysis": result.to_dict()}
                
                except Exception as e:
                    logger.error(f"Error analyzing file {file_path}: {e}")
                    return {"file_path": file_path, "error": str(e)}
        
        # Execute batch analysis
        tasks = [analyze_single_file(file_path) for file_path in file_paths]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in batch_results:
            if isinstance(result, Exception):
                failed_files.append({"error": str(result)})
            elif "error" in result:
                failed_files.append(result)
            else:
                results.append(result)
        
        return {
            "success": True,
            "results": results,
            "successful_analyses": len(results),
            "failed_analyses": len(failed_files),
            "failed_files": failed_files,
            "total_requested": len(file_paths)
        }
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health Check

@router.get("/health", response_model=Dict[str, Any])
async def health_check(
    ml_service: MLModelsService = Depends(get_ml_models_service)
) -> Dict[str, Any]:
    """Health check for ML models service"""
    try:
        coordinator_status = await ml_service.coordinator.get_system_status()
        
        health_status = {
            "ml_models_api": "healthy",
            "ml_coordinator": "healthy" if coordinator_status["system_health"] != "poor" else "degraded",
            "registered_models": len(coordinator_status["registered_models"]),
            "active_analyses": coordinator_status["active_analyses"],
            "system_health": coordinator_status["system_health"]
        }
        
        overall_healthy = all(
            status in ["healthy", "good"] 
            for status in [health_status["ml_models_api"], health_status["ml_coordinator"]]
        )
        
        return {
            "success": True,
            "status": "healthy" if overall_healthy else "degraded",
            "services": health_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in ML models health check: {e}")
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }