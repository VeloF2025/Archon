"""
Confidence API Routes - DeepConf Integration

Provides API endpoints for confidence scoring, real-time updates, and historical analysis.
Integrates with the DeepConf engine for multi-dimensional confidence assessment.

Author: Archon AI System
Version: 1.0.0
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator
import socketio

from ..services.credential_service import CredentialService
from ...agents.deepconf.engine import DeepConfEngine
from ...agents.deepconf.types import ConfidenceScore, ConfidenceExplanation

# Set up logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/confidence", tags=["confidence"])

# Global DeepConf engine instance (lazy-loaded)
_deepconf_engine: Optional[DeepConfEngine] = None

# Socket.IO instance for real-time updates
_sio: Optional[socketio.AsyncServer] = None

def get_socketio():
    """Get Socket.IO instance"""
    global _sio
    if _sio is None:
        from ..socketio_app import sio
        _sio = sio
    return _sio

class TaskInput(BaseModel):
    """Task input for confidence calculation"""
    task_id: str = Field(..., description="Unique task identifier")
    content: str = Field(..., description="Task content/description")
    complexity: Optional[str] = Field(None, description="Task complexity level")
    domain: Optional[str] = Field(None, description="Task domain")
    priority: Optional[int] = Field(1, description="Task priority (1=high, 3=low)")
    model_source: Optional[str] = Field("unknown", description="Model source")
    context_size: Optional[int] = Field(None, description="Context size in tokens")
    
    @validator('complexity')
    def validate_complexity(cls, v):
        if v is not None:
            valid_values = ['simple', 'moderate', 'complex', 'very_complex']
            if v not in valid_values:
                raise ValueError(f"Complexity must be one of {valid_values}")
        return v

class TaskContext(BaseModel):
    """Task execution context"""
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    environment: Optional[str] = Field("unknown", description="Execution environment")
    model_history: Optional[List[Dict[str, Any]]] = Field(None, description="Model execution history")
    performance_data: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")
    timestamp: Optional[float] = Field(None, description="Context timestamp")

class ConfidenceRequest(BaseModel):
    """Request for confidence calculation"""
    task: TaskInput
    context: Optional[TaskContext] = Field(default_factory=TaskContext)

class ConfidenceResponse(BaseModel):
    """Response with confidence score"""
    success: bool
    confidence_score: Dict[str, Any]
    calculation_time: float
    cached: bool = False

class ExecutionUpdate(BaseModel):
    """Real-time execution update"""
    task_id: str
    progress: float = Field(..., ge=0.0, le=1.0, description="Execution progress (0-1)")
    intermediate_result: str = Field("", description="Intermediate result description")
    timestamp: Optional[float] = Field(default_factory=time.time)

class ValidationRequest(BaseModel):
    """Request for confidence validation"""
    confidence_score: Dict[str, Any]
    actual_result: Dict[str, Any]

class CalibrationRequest(BaseModel):
    """Request for model calibration"""
    historical_data: List[Dict[str, Any]] = Field(
        ..., 
        description="Historical data with predicted_confidence and actual_success fields"
    )

def get_deepconf_engine() -> DeepConfEngine:
    """Get or initialize DeepConf engine (lazy loading)"""
    global _deepconf_engine
    
    if _deepconf_engine is None:
        try:
            # Initialize with default configuration
            _deepconf_engine = DeepConfEngine()
            logger.info("DeepConf engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DeepConf engine: {e}")
            raise HTTPException(status_code=500, detail=f"DeepConf initialization failed: {str(e)}")
    
    return _deepconf_engine

@router.get("/system")
async def get_system_confidence():
    """
    Get overall system confidence metrics
    
    Returns system-wide confidence assessment based on recent calculations
    and overall system health.
    """
    try:
        engine = get_deepconf_engine()
        
        # Create a system-wide task for analysis
        from types import SimpleNamespace
        system_task = SimpleNamespace(
            task_id="system_overall",
            content="overall system confidence assessment",
            complexity="moderate",
            domain="system",
            priority=1,
            model_source="system_metrics"
        )
        system_context = SimpleNamespace(
            user_id="system",
            environment="production",
            timestamp=time.time()
        )
        
        # Calculate system confidence
        confidence_score = await engine.calculate_confidence(system_task, system_context)
        
        logger.info(
            "System confidence calculated",
            extra={
                "confidence_score": confidence_score.overall_confidence,
                "calculation_type": "system_wide"
            }
        )
        
        return {
            "success": True,
            "confidence_score": confidence_score.to_dict(),
            "system_health": {
                "cache_size": len(engine._confidence_cache),
                "historical_data_points": len(engine._historical_data),
                "active_tasks": len(engine._active_tasks)
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"System confidence calculation failed: {e}")
        raise HTTPException(status_code=500, detail=f"System confidence failed: {str(e)}")

@router.get("/history")
async def get_confidence_history(
    hours: int = 24,
    granularity: str = "hour"
):
    """
    Get confidence history over time
    
    Returns historical confidence data points aggregated by time period.
    """
    try:
        engine = get_deepconf_engine()
        
        # Validate granularity
        valid_granularities = ["minute", "hour", "day"]
        if granularity not in valid_granularities:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid granularity. Must be one of: {valid_granularities}"
            )
        
        # Get historical data from BOTH persistent storage AND in-memory cache
        historical_data = list(engine._historical_data)  # In-memory cache
        
        # ALSO get data from persistent storage to ensure we have ALL historical data
        try:
            from ...agents.deepconf.storage import get_storage
            storage = get_storage()
            
            # Get recent confidence history from persistent storage
            storage_data = storage.get_recent_confidence_history(limit=10000)  # Large limit to get all data
            
            # Convert storage data to engine format and merge
            for record in storage_data:
                try:
                    # Parse confidence_score JSON if needed
                    confidence_score = record['confidence_score']
                    if isinstance(confidence_score, str):
                        import json
                        confidence_score = json.loads(confidence_score)
                    
                    # Create historical entry in engine format
                    historical_entry = {
                        'timestamp': record['timestamp'],
                        'task_id': record['task_id'], 
                        'agent_id': record['agent_id'],
                        'phase': record['phase'],
                        'confidence_score': confidence_score,
                        'execution_duration': record.get('execution_duration', 1.0),
                        'success': record.get('success', True),
                        'result_quality': record.get('result_quality', 0.8),
                        'domain': record.get('domain', 'general'),
                        'complexity': record.get('complexity', 'moderate'),
                        'source': 'persistent_storage'
                    }
                    
                    # Add if not already present (avoid duplicates)
                    task_timestamp_key = f"{record['task_id']}_{record['timestamp']}"
                    existing_keys = [f"{d.get('task_id', '')}_{d.get('timestamp', 0)}" for d in historical_data]
                    
                    if task_timestamp_key not in existing_keys:
                        historical_data.append(historical_entry)
                        
                except Exception as record_error:
                    logger.warning(f"Failed to parse storage record: {record_error}")
                    continue
                    
            logger.info(f"📊 Merged data from storage: {len(storage_data)} records, total: {len(historical_data)} data points")
            
        except Exception as storage_error:
            logger.warning(f"Could not access persistent storage: {storage_error}")
            # Continue with just in-memory data
        
        # Filter by time range
        current_time = time.time()
        time_cutoff = current_time - (hours * 3600)  # Convert hours to seconds
        
        # Filter historical data by time range
        filtered_data = []
        for data_point in historical_data:
            if data_point.get('timestamp', 0) >= time_cutoff:
                filtered_data.append(data_point)
        
        # NO SYNTHETIC DATA - only return real historical data
        if not filtered_data:
            logger.info(f"No real confidence history available for {hours} hour period")
            return {
                "success": True,
                "history": [],
                "metadata": {
                    "hours": hours,
                    "granularity": granularity,
                    "data_points": 0,
                    "message": "No real confidence data available for this time period. Data will appear as system performs actual confidence calculations.",
                    "time_range": {
                        "start": time_cutoff,
                        "end": current_time
                    }
                },
                "timestamp": time.time()
            }
        
        # Sort by timestamp (oldest first)
        filtered_data.sort(key=lambda x: x.get('timestamp', 0))
        
        logger.info(
            f"Retrieved confidence history: {len(filtered_data)} points over {hours} hours",
            extra={
                "data_points": len(filtered_data),
                "time_range_hours": hours,
                "granularity": granularity
            }
        )
        
        return {
            "success": True,
            "history": filtered_data,
            "metadata": {
                "hours": hours,
                "granularity": granularity,
                "data_points": len(filtered_data),
                "time_range": {
                    "start": time_cutoff,
                    "end": current_time
                }
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Confidence history retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"History retrieval failed: {str(e)}")

@router.get("/scwt")
async def get_scwt_metrics(
    phase: Optional[str] = None
):
    """
    Get SCWT (Structural-Contextual-Weighted-Temporal) benchmark metrics
    
    Returns REAL SCWT performance metrics from actual system operations.
    NO SYNTHETIC DATA - only returns metrics from real confidence calculations.
    """
    try:
        engine = get_deepconf_engine()
        
        # Get REAL historical confidence data from BOTH in-memory AND persistent storage
        scwt_metrics = []
        historical_data = list(engine._historical_data)  # In-memory cache
        
        # ALSO get data from persistent storage to ensure we have ALL historical data
        try:
            from ...agents.deepconf.storage import get_storage
            storage = get_storage()
            
            # Get ALL confidence history from persistent storage for SCWT metrics
            storage_data = storage.get_recent_confidence_history(limit=50000)  # Very large limit for SCWT
            
            # Convert storage data to engine format and merge
            for record in storage_data:
                try:
                    # Parse confidence_score JSON if needed
                    confidence_score = record['confidence_score']
                    if isinstance(confidence_score, str):
                        import json
                        confidence_score = json.loads(confidence_score)
                    
                    # Create historical entry in engine format
                    historical_entry = {
                        'timestamp': record['timestamp'],
                        'task_id': record['task_id'], 
                        'agent_id': record['agent_id'],
                        'phase': record['phase'],
                        'confidence_score': confidence_score,
                        'execution_duration': record.get('execution_duration', 1.0),
                        'success': record.get('success', True),
                        'result_quality': record.get('result_quality', 0.8),
                        'domain': record.get('domain', 'general'),
                        'complexity': record.get('complexity', 'moderate'),
                        'source': 'persistent_storage'
                    }
                    
                    # Add if not already present (avoid duplicates for SCWT calculation)
                    task_timestamp_key = f"{record['task_id']}_{record['timestamp']}"
                    existing_keys = [f"{d.get('task_id', '')}_{d.get('timestamp', 0)}" for d in historical_data]
                    
                    if task_timestamp_key not in existing_keys:
                        historical_data.append(historical_entry)
                        
                except Exception as record_error:
                    logger.warning(f"Failed to parse storage record for SCWT: {record_error}")
                    continue
                    
            logger.info(f"🏆 SCWT METRICS: Merged {len(storage_data)} storage records, total: {len(historical_data)} data points")
            
        except Exception as storage_error:
            logger.warning(f"Could not access persistent storage for SCWT: {storage_error}")
            # Continue with just in-memory data
        
        # Only return real data - no synthetic generation
        if not historical_data:
            logger.info("No real SCWT data available yet - system needs actual confidence calculations")
            return {
                "success": True,
                "scwt_metrics": [],
                "summary": {
                    "phase": phase or "general",
                    "message": "No real SCWT data available yet. System will populate as real confidence calculations occur.",
                    "data_points": 0
                },
                "system_state": {
                    "cache_size": len(engine._confidence_cache),
                    "historical_points": len(engine._historical_data),
                    "active_tasks": len(engine._active_tasks),
                    "total_calculations": len(engine._performance_metrics.get('confidence_calculation', []))
                },
                "timestamp": time.time()
            }
        
        # Process REAL historical data only
        for data_point in historical_data:
            if isinstance(data_point, dict) and 'confidence_score' in data_point:
                confidence_data = data_point['confidence_score']
                
                # Extract real SCWT metrics from confidence calculations
                metric = {
                    "timestamp": data_point.get('timestamp', time.time()),
                    "structural_weight": confidence_data.get('confidence_factors', {}).get('technical_complexity', 0.0),
                    "context_weight": confidence_data.get('contextual_confidence', 0.0),
                    "temporal_weight": confidence_data.get('confidence_factors', {}).get('historical_performance', 0.0),
                    "combined_score": confidence_data.get('overall_confidence', 0.0),
                    "confidence": confidence_data.get('overall_confidence', 0.0),
                    "phase": phase or data_point.get('phase', 'general'),
                    "task_id": data_point.get('task_id', 'unknown'),
                    "agent_id": data_point.get('agent_id', 'deepconf_engine')
                }
                
                scwt_metrics.append(metric)
        
        # Sort by timestamp (oldest first)
        scwt_metrics.sort(key=lambda x: x['timestamp'])
        
        # Calculate real averages from actual data
        if scwt_metrics:
            avg_structural = sum(m['structural_weight'] for m in scwt_metrics) / len(scwt_metrics)
            avg_context = sum(m['context_weight'] for m in scwt_metrics) / len(scwt_metrics)
            avg_temporal = sum(m['temporal_weight'] for m in scwt_metrics) / len(scwt_metrics)
            avg_combined = sum(m['combined_score'] for m in scwt_metrics) / len(scwt_metrics)
        else:
            avg_structural = avg_context = avg_temporal = avg_combined = 0.0
        
        logger.info(
            f"Retrieved REAL SCWT metrics: {len(scwt_metrics)} data points",
            extra={
                "phase": phase or "general",
                "data_source": "real_historical_data",
                "data_points": len(scwt_metrics),
                "avg_structural": avg_structural,
                "avg_context": avg_context,
                "avg_temporal": avg_temporal,
                "avg_combined": avg_combined
            }
        )
        
        return {
            "success": True,
            "scwt_metrics": scwt_metrics,
            "summary": {
                "phase": phase or "general",
                "average_structural": avg_structural,
                "average_context": avg_context,
                "average_temporal": avg_temporal,
                "average_combined": avg_combined,
                "data_points": len(scwt_metrics),
                "data_source": "real_calculations"
            },
            "system_state": {
                "cache_size": len(engine._confidence_cache),
                "historical_points": len(engine._historical_data),
                "active_tasks": len(engine._active_tasks),
                "total_calculations": len(engine._performance_metrics.get('confidence_calculation', []))
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"SCWT metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"SCWT metrics failed: {str(e)}")

@router.get("/health")
async def confidence_health():
    """Check DeepConf engine health status"""
    try:
        engine = get_deepconf_engine()
        
        # Perform a quick test calculation
        from types import SimpleNamespace
        test_task = SimpleNamespace(
            task_id="health_check",
            content="test calculation",
            complexity="simple",
            domain="test",
            model_source="health_check"
        )
        test_context = SimpleNamespace(
            user_id="system",
            environment="test"
        )
        
        start_time = time.time()
        confidence_score = await engine.calculate_confidence(test_task, test_context)
        calculation_time = time.time() - start_time
        
        return {
            "status": "healthy",
            "engine_initialized": True,
            "test_calculation_time": round(calculation_time, 3),
            "test_confidence": confidence_score.overall_confidence,
            "cache_size": len(engine._confidence_cache),
            "historical_data_points": len(engine._historical_data)
        }
        
    except Exception as e:
        logger.error(f"DeepConf health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "engine_initialized": _deepconf_engine is not None
        }

@router.post("/analyze", response_model=ConfidenceResponse)
async def analyze_confidence(request: ConfidenceRequest):
    """
    Calculate multi-dimensional confidence score for a task
    
    This endpoint uses the DeepConf engine to analyze task complexity,
    domain expertise, data availability, and other factors to generate
    a comprehensive confidence assessment.
    """
    try:
        engine = get_deepconf_engine()
        
        # Convert Pydantic models to SimpleNamespace objects
        from types import SimpleNamespace
        
        # Create task object
        task = SimpleNamespace(
            task_id=request.task.task_id,
            content=request.task.content,
            complexity=request.task.complexity,
            domain=request.task.domain,
            priority=request.task.priority,
            model_source=request.task.model_source,
            context_size=request.task.context_size
        )
        
        # Create context object
        context = SimpleNamespace(
            user_id=request.context.user_id if request.context else None,
            session_id=request.context.session_id if request.context else None,
            environment=request.context.environment if request.context else "unknown",
            model_history=request.context.model_history if request.context else None,
            performance_data=request.context.performance_data if request.context else None,
            timestamp=request.context.timestamp if request.context else time.time()
        )
        
        # Calculate confidence
        start_time = time.time()
        confidence_score = await engine.calculate_confidence(task, context)
        calculation_time = time.time() - start_time
        
        logger.info(
            f"Calculated confidence for task {request.task.task_id}",
            extra={
                "task_id": request.task.task_id,
                "confidence_score": confidence_score.overall_confidence,
                "factual_confidence": confidence_score.factual_confidence,
                "reasoning_confidence": confidence_score.reasoning_confidence,
                "contextual_confidence": confidence_score.contextual_confidence,
                "epistemic_uncertainty": confidence_score.epistemic_uncertainty,
                "aleatoric_uncertainty": confidence_score.aleatoric_uncertainty,
                "uncertainty_bounds": confidence_score.uncertainty_bounds,
                "calculation_time": calculation_time,
                "primary_factors": confidence_score.primary_factors,
                "gaming_detection_score": confidence_score.gaming_detection_score,
                "calibration_applied": confidence_score.calibration_applied,
                "agent_role": request.task.domain or "unknown",
                "task_complexity": request.task.complexity or "unknown"
            }
        )
        
        return ConfidenceResponse(
            success=True,
            confidence_score=confidence_score.to_dict(),
            calculation_time=calculation_time,
            cached=False  # TODO: Implement cache detection
        )
        
    except Exception as e:
        logger.error(f"Confidence analysis failed for task {request.task.task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Confidence analysis failed: {str(e)}")

@router.post("/start-tracking/{task_id}")
async def start_confidence_tracking(task_id: str):
    """
    Start real-time confidence tracking for a task
    
    This creates a confidence tracking stream that can be updated
    in real-time as the task executes.
    """
    try:
        engine = get_deepconf_engine()
        
        # Start tracking
        stream_id = engine.start_confidence_tracking(task_id)
        
        logger.info(f"Started confidence tracking for task {task_id} with stream {stream_id}")
        
        return {
            "success": True,
            "task_id": task_id,
            "stream_id": stream_id,
            "message": f"Confidence tracking started for task {task_id}"
        }
        
    except Exception as e:
        logger.error(f"Failed to start confidence tracking for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start tracking: {str(e)}")

@router.post("/update-realtime")
async def update_confidence_realtime(update: ExecutionUpdate, background_tasks: BackgroundTasks):
    """
    Update confidence in real-time based on execution progress
    
    This endpoint receives execution updates and recalculates confidence
    based on current progress and intermediate results.
    """
    try:
        engine = get_deepconf_engine()
        
        # Prepare execution update
        execution_update = {
            'progress': update.progress,
            'intermediate_result': update.intermediate_result,
            'timestamp': update.timestamp
        }
        
        # Update confidence
        confidence_score = await engine.update_confidence_realtime(
            update.task_id, 
            execution_update
        )
        
        # Emit Socket.IO update asynchronously
        background_tasks.add_task(
            _emit_confidence_update,
            update.task_id,
            confidence_score.to_dict(),
            update.progress
        )
        
        logger.info(
            f"Updated real-time confidence for task {update.task_id}",
            extra={
                "task_id": update.task_id,
                "confidence_score": confidence_score.overall_confidence,
                "progress": update.progress,
                "intermediate_result": update.intermediate_result,
                "update_timestamp": update.timestamp,
                "uncertainty_bounds": confidence_score.uncertainty_bounds,
                "confidence_trend": "realtime_update"
            }
        )
        
        return {
            "success": True,
            "task_id": update.task_id,
            "confidence_score": confidence_score.to_dict(),
            "progress": update.progress,
            "updated_at": confidence_score.timestamp
        }
        
    except ValueError as ve:
        logger.warning(f"Invalid confidence update for task {update.task_id}: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Real-time confidence update failed for task {update.task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Confidence update failed: {str(e)}")

@router.get("/task/{task_id}/history")
async def get_confidence_history(task_id: str):
    """
    Get confidence history for a task
    
    Returns the confidence tracking history including all real-time updates
    and confidence score changes during task execution.
    """
    try:
        engine = get_deepconf_engine()
        
        # Check if task is being tracked
        if task_id not in engine._active_tasks:
            raise HTTPException(status_code=404, detail=f"No confidence tracking found for task {task_id}")
        
        tracking_info = engine._active_tasks[task_id]
        
        # Convert confidence scores to dictionaries
        confidence_history = [
            score.to_dict() for score in tracking_info['confidence_history']
        ]
        
        return {
            "success": True,
            "task_id": task_id,
            "stream_id": tracking_info['stream_id'],
            "start_time": tracking_info['start_time'],
            "last_update": tracking_info['last_update'],
            "confidence_history": confidence_history,
            "total_updates": len(confidence_history)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get confidence history for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@router.post("/explain")
async def explain_confidence(request: Dict[str, Any]):
    """
    Generate human-readable explanation for a confidence score
    
    Takes a confidence score and returns detailed explanations of the
    factors that influenced the score and suggestions for improvement.
    """
    try:
        engine = get_deepconf_engine()
        
        # Convert dictionary to ConfidenceScore object
        confidence_score = ConfidenceScore.from_dict(request)
        
        # Generate explanation
        explanation = engine.explain_confidence(confidence_score)
        
        return {
            "success": True,
            "explanation": {
                "primary_factors": explanation.primary_factors,
                "confidence_reasoning": explanation.confidence_reasoning,
                "uncertainty_sources": explanation.uncertainty_sources,
                "improvement_suggestions": explanation.improvement_suggestions,
                "factor_importance_ranking": explanation.factor_importance_ranking
            }
        }
        
    except Exception as e:
        logger.error(f"Confidence explanation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@router.post("/validate")
async def validate_confidence(request: ValidationRequest):
    """
    Validate confidence score against actual task results
    
    This endpoint compares predicted confidence with actual task outcomes
    to measure the accuracy of confidence predictions and improve calibration.
    """
    try:
        engine = get_deepconf_engine()
        
        # Convert confidence score dictionary to ConfidenceScore object
        confidence_score = ConfidenceScore.from_dict(request.confidence_score)
        
        # Convert actual result to SimpleNamespace
        from types import SimpleNamespace
        actual_result = SimpleNamespace(**request.actual_result)
        
        # Validate confidence
        validation_results = await engine.validate_confidence(confidence_score, actual_result)
        
        logger.info(
            f"Confidence validation completed for task {confidence_score.task_id}",
            extra={
                "task_id": confidence_score.task_id,
                "validation_accuracy": validation_results['accuracy'],
                "calibration_error": validation_results['calibration_error'],
                "predicted_confidence": validation_results['predicted_confidence'],
                "actual_success": validation_results['actual_success'],
                "actual_quality": validation_results['actual_quality'],
                "meets_prd_requirements": validation_results['meets_prd_requirements'],
                "validation_type": "confidence_validation"
            }
        )
        
        return {
            "success": True,
            **validation_results
        }
        
    except Exception as e:
        logger.error(f"Confidence validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.post("/calibrate")
async def calibrate_model(request: CalibrationRequest):
    """
    Calibrate confidence model using historical performance data
    
    This endpoint improves confidence prediction accuracy by learning
    from historical task outcomes and adjusting confidence calculations.
    """
    try:
        engine = get_deepconf_engine()
        
        # Validate historical data format
        required_fields = ['predicted_confidence', 'actual_success']
        for i, item in enumerate(request.historical_data):
            for field in required_fields:
                if field not in item:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Historical data item {i} missing required field: {field}"
                    )
        
        # Perform calibration
        calibration_results = await engine.calibrate_model(request.historical_data)
        
        logger.info(
            f"Model calibration completed",
            extra={
                "calibration_improved": calibration_results['calibration_improved'],
                "accuracy_delta": calibration_results.get('accuracy_delta', 0),
                "confidence_shift": calibration_results.get('confidence_shift', 0),
                "pre_calibration_accuracy": calibration_results.get('pre_calibration_accuracy', 0),
                "post_calibration_accuracy": calibration_results.get('post_calibration_accuracy', 0),
                "calibration_samples": calibration_results.get('calibration_samples', 0),
                "calibration_type": "model_calibration"
            }
        )
        
        return {
            "success": True,
            **calibration_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model calibration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")

@router.get("/factors/{task_id}")
async def get_confidence_factors(task_id: str, task_content: str, domain: Optional[str] = None):
    """
    Get detailed confidence factors for a task
    
    Returns analysis of individual factors that influence confidence
    including their importance, impact, and supporting evidence.
    """
    try:
        engine = get_deepconf_engine()
        
        # Create task object
        from types import SimpleNamespace
        task = SimpleNamespace(
            task_id=task_id,
            content=task_content,
            domain=domain
        )
        
        # Get confidence factors
        factors = engine.get_confidence_factors(task)
        
        # Convert to dictionary format
        factors_data = []
        for factor in factors:
            factors_data.append({
                "name": factor.name,
                "importance": factor.importance,
                "impact": factor.impact,
                "description": factor.description,
                "evidence": factor.evidence
            })
        
        return {
            "success": True,
            "task_id": task_id,
            "confidence_factors": factors_data,
            "total_factors": len(factors_data)
        }
        
    except Exception as e:
        logger.error(f"Failed to get confidence factors for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get factors: {str(e)}")

@router.get("/metrics")
async def get_confidence_metrics():
    """
    Get confidence system performance metrics
    
    Returns performance statistics, cache status, and calibration information
    for monitoring the confidence system health and performance.
    """
    try:
        engine = get_deepconf_engine()
        
        # Calculate performance metrics
        confidence_calculations = engine._performance_metrics.get('confidence_calculation', [])
        
        metrics = {
            "performance": {
                "total_calculations": len(confidence_calculations),
                "average_calculation_time": sum(confidence_calculations) / len(confidence_calculations) if confidence_calculations else 0,
                "min_calculation_time": min(confidence_calculations) if confidence_calculations else 0,
                "max_calculation_time": max(confidence_calculations) if confidence_calculations else 0
            },
            "cache": {
                "cache_size": len(engine._confidence_cache),
                "cache_ttl": engine._cache_ttl
            },
            "calibration": {
                "historical_data_points": len(engine._historical_data),
                "calibration_model_active": engine._calibration_model is not None,
                "min_calibration_samples": engine.config['min_calibration_samples']
            },
            "tracking": {
                "active_tasks": len(engine._active_tasks),
                "confidence_streams": len(engine._confidence_streams)
            }
        }
        
        return {
            "success": True,
            "timestamp": time.time(),
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Failed to get confidence metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

async def _emit_confidence_update(task_id: str, confidence_score: Dict[str, Any], progress: float):
    """Emit Socket.IO confidence update (background task)"""
    try:
        sio = get_socketio()
        
        await sio.emit('confidence_update', {
            'task_id': task_id,
            'confidence_score': confidence_score,
            'progress': progress,
            'timestamp': time.time()
        })
        
        logger.debug(f"Emitted confidence update for task {task_id}")
        
    except Exception as e:
        logger.error(f"Failed to emit confidence update for task {task_id}: {e}")

# Export router for main app
__all__ = ['router']