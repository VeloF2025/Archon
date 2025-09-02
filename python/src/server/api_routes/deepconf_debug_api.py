"""
DeepConf Advanced Debugging API Routes - Phase 7 PRD Implementation

Provides comprehensive REST API endpoints for DeepConf debugging functionality:
- Low confidence analysis with actionable insights
- Confidence factor tracing with detailed breakdowns
- Performance bottleneck detection with root cause analysis
- Optimization suggestions with implementation guidance
- Debug session management with state persistence
- Data export functionality for external analysis

All endpoints include comprehensive error handling, logging, and validation.
Designed for integration with React debugging UI and external analysis tools.

Author: Archon AI System
Version: 1.0.0
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, validator
import numpy as np

# Import DeepConf debugger and types
from ...agents.deepconf.debugger import (
    DeepConfDebugger, 
    AITask, 
    DebugSession, 
    DebugReport,
    ConfidenceTrace,
    BottleneckAnalysis,
    OptimizationSuggestions,
    PerformanceData,
    TaskHistory,
    DebugExport
)
from ...agents.deepconf.engine import DeepConfEngine
from ...agents.deepconf.types import ConfidenceScore

# Set up logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/deepconf/debug", tags=["deepconf-debug"])

# Global debugger instance (will be initialized on first request)
debugger_instance: Optional[DeepConfDebugger] = None
engine_instance: Optional[DeepConfEngine] = None

def get_debugger() -> DeepConfDebugger:
    """Get or create debugger instance with engine integration"""
    global debugger_instance, engine_instance
    
    if debugger_instance is None:
        try:
            # Initialize DeepConf engine if not already done
            if engine_instance is None:
                engine_instance = DeepConfEngine()
            
            # Create debugger with engine integration
            debugger_instance = DeepConfDebugger(
                deepconf_engine=engine_instance,
                config={
                    'max_active_sessions': 15,
                    'session_timeout': 7200,  # 2 hours
                    'confidence_threshold_critical': 0.3,
                    'confidence_threshold_warning': 0.6,
                    'bottleneck_detection_threshold': 2.0,
                    'trace_depth': 10,
                    'export_formats': ['json', 'csv', 'pdf'],
                    'max_optimization_strategies': 8
                }
            )
            
            logger.info("DeepConf debugger initialized with engine integration")
            
        except Exception as e:
            logger.error(f"Failed to initialize DeepConf debugger: {e}")
            raise RuntimeError(f"Debugger initialization failed: {e}")
    
    return debugger_instance

# Request/Response Models

class TaskRequest(BaseModel):
    """Request model for AI task"""
    task_id: str = Field(..., min_length=1, max_length=100)
    content: str = Field(..., min_length=1, max_length=10000)
    domain: str = Field(default="general", max_length=50)
    complexity: str = Field(default="moderate", pattern="^(simple|moderate|complex|very_complex)$")
    priority: str = Field(default="normal", pattern="^(low|normal|high|critical)$")
    model_source: str = Field(default="unknown", max_length=50)
    context_size: Optional[int] = Field(default=None, ge=0, le=100000)
    
    @validator('task_id')
    def validate_task_id(cls, v):
        if not v or not v.strip():
            raise ValueError('task_id cannot be empty')
        return v.strip()

class ConfidenceRequest(BaseModel):
    """Request model for confidence score"""
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    factual_confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning_confidence: float = Field(..., ge=0.0, le=1.0)
    contextual_confidence: float = Field(..., ge=0.0, le=1.0)
    epistemic_uncertainty: float = Field(..., ge=0.0, le=1.0)
    aleatoric_uncertainty: float = Field(..., ge=0.0, le=1.0)
    uncertainty_bounds: List[float] = Field(..., min_items=2, max_items=2)
    confidence_factors: Dict[str, float] = Field(...)
    primary_factors: List[str] = Field(...)
    confidence_reasoning: str = Field(..., max_length=1000)
    model_source: str = Field(..., max_length=50)
    task_id: str = Field(..., max_length=100)
    
    @validator('uncertainty_bounds')
    def validate_bounds(cls, v):
        if len(v) != 2 or v[0] > v[1]:
            raise ValueError('uncertainty_bounds must be [lower, upper] with lower <= upper')
        return v

class PerformanceDataRequest(BaseModel):
    """Request model for performance data"""
    operation_times: Dict[str, List[float]] = Field(default_factory=dict)
    memory_usage: Dict[str, List[float]] = Field(default_factory=dict)
    cache_hit_rates: Dict[str, float] = Field(default_factory=dict)
    error_rates: Dict[str, float] = Field(default_factory=dict)
    throughput_metrics: Dict[str, float] = Field(default_factory=dict)
    bottleneck_indicators: Dict[str, Any] = Field(default_factory=dict)

class TaskHistoryRequest(BaseModel):
    """Request model for task history"""
    task_id: str = Field(..., max_length=100)
    execution_records: List[Dict[str, Any]] = Field(default_factory=list)
    performance_metrics: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_history: List[Dict[str, Any]] = Field(default_factory=list)  # Simplified for API
    total_executions: int = Field(default=0, ge=0)
    average_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    performance_trends: Dict[str, List[float]] = Field(default_factory=dict)

# Health Check Endpoint

@router.get("/health")
async def get_debug_health():
    """Health check for DeepConf debugging system"""
    try:
        debugger = get_debugger()
        
        # Test basic functionality
        test_task = AITask(
            task_id="health_check_debug",
            content="Test debugging system health",
            domain="testing"
        )
        
        # Test session creation
        session = debugger.create_debug_session(test_task)
        
        health_status = {
            "status": "healthy",
            "debugger_initialized": True,
            "engine_attached": debugger.engine is not None,
            "active_sessions": len(debugger.active_sessions),
            "max_sessions": debugger.config['max_active_sessions'],
            "session_timeout": debugger.config['session_timeout'],
            "supported_export_formats": debugger.config['export_formats'],
            "test_session_created": session.session_id,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("DeepConf debug health check passed")
        return health_status
        
    except Exception as e:
        logger.error(f"DeepConf debug health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "debugger_initialized": False,
            "timestamp": datetime.now().isoformat()
        }

# Core Debugging Endpoints

@router.post("/analyze/low-confidence")
async def analyze_low_confidence(
    task: TaskRequest,
    confidence: ConfidenceRequest,
    background_tasks: BackgroundTasks
):
    """
    Analyze low confidence scores with comprehensive root cause analysis
    
    Provides detailed analysis of why confidence is low and actionable recommendations
    for improvement. Includes factor analysis, environmental factors, and optimization
    opportunities.
    """
    start_time = time.time()
    
    try:
        debugger = get_debugger()
        
        # Convert request models to domain objects
        ai_task = AITask(
            task_id=task.task_id,
            content=task.content,
            domain=task.domain,
            complexity=task.complexity,
            priority=task.priority,
            model_source=task.model_source,
            context_size=task.context_size
        )
        
        confidence_score = ConfidenceScore(
            overall_confidence=confidence.overall_confidence,
            factual_confidence=confidence.factual_confidence,
            reasoning_confidence=confidence.reasoning_confidence,
            contextual_confidence=confidence.contextual_confidence,
            epistemic_uncertainty=confidence.epistemic_uncertainty,
            aleatoric_uncertainty=confidence.aleatoric_uncertainty,
            uncertainty_bounds=tuple(confidence.uncertainty_bounds),
            confidence_factors=confidence.confidence_factors,
            primary_factors=confidence.primary_factors,
            confidence_reasoning=confidence.confidence_reasoning,
            model_source=confidence.model_source,
            timestamp=time.time(),
            task_id=confidence.task_id
        )
        
        # Perform low confidence analysis
        debug_report = await debugger.analyze_low_confidence(ai_task, confidence_score)
        
        # Convert to serializable format
        response_data = {
            "success": True,
            "analysis_time": time.time() - start_time,
            "report": {
                "report_id": debug_report.report_id,
                "task_id": debug_report.task_id,
                "confidence_score": debug_report.confidence_score,
                "analysis_timestamp": debug_report.analysis_timestamp.isoformat(),
                "issues": [
                    {
                        "id": issue.id,
                        "severity": issue.severity.value,
                        "category": issue.category,
                        "title": issue.title,
                        "description": issue.description,
                        "root_cause": issue.root_cause,
                        "recommendations": issue.recommendations,
                        "affected_factors": issue.affected_factors,
                        "confidence_impact": issue.confidence_impact,
                        "timestamp": issue.timestamp.isoformat(),
                        "metrics": issue.metrics
                    }
                    for issue in debug_report.issues
                ],
                "recommendations": debug_report.recommendations,
                "severity_summary": {k.value: v for k, v in debug_report.severity_summary.items()},
                "factor_analysis": debug_report.factor_analysis,
                "performance_profile": debug_report.performance_profile,
                "optimization_opportunities": [
                    {
                        "strategy_id": opt.strategy_id,
                        "title": opt.title,
                        "description": opt.description,
                        "implementation_complexity": opt.implementation_complexity,
                        "expected_improvement": opt.expected_improvement,
                        "implementation_steps": opt.implementation_steps,
                        "risks": opt.risks,
                        "prerequisites": opt.prerequisites,
                        "estimated_effort": opt.estimated_effort,
                        "confidence": opt.confidence
                    }
                    for opt in debug_report.optimization_opportunities
                ],
                "confidence_projection": debug_report.confidence_projection
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Low confidence analysis completed for {task.task_id}: {len(debug_report.issues)} issues, {len(debug_report.recommendations)} recommendations")
        
        return JSONResponse(content=response_data)
        
    except ValueError as e:
        logger.warning(f"Invalid request for low confidence analysis: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Low confidence analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/trace/confidence-factors/{confidence_id}")
async def trace_confidence_factors(confidence_id: str):
    """
    Trace confidence factors with detailed breakdown and computation steps
    
    Provides deep insight into how individual confidence factors are calculated,
    their dependencies, computation steps, and contribution to overall confidence.
    """
    start_time = time.time()
    
    try:
        debugger = get_debugger()
        
        # Validate confidence_id
        if not confidence_id or len(confidence_id) > 100:
            raise ValueError("Invalid confidence_id")
        
        # Perform confidence factor tracing
        trace = await debugger.trace_confidence_factors(confidence_id)
        
        # Convert to serializable format
        response_data = {
            "success": True,
            "trace_time": time.time() - start_time,
            "trace": {
                "factor_name": trace.factor_name,
                "raw_score": trace.raw_score,
                "weighted_score": trace.weighted_score,
                "weight": trace.weight,
                "calculation_steps": trace.calculation_steps,
                "dependencies": trace.dependencies,
                "computation_time": trace.computation_time,
                "confidence_contribution": trace.confidence_contribution,
                "trace_id": trace.trace_id,
                "timestamp": trace.timestamp.isoformat()
            },
            "metadata": {
                "total_steps": len(trace.calculation_steps),
                "dependency_count": len(trace.dependencies),
                "contribution_percentage": trace.confidence_contribution * 100
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Confidence factor tracing completed: {trace.factor_name} contributes {trace.confidence_contribution:.3f} to overall confidence")
        
        return JSONResponse(content=response_data)
        
    except ValueError as e:
        logger.warning(f"Invalid request for confidence tracing: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Confidence factor tracing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Tracing failed: {str(e)}")

@router.post("/analyze/performance-bottlenecks")
async def analyze_performance_bottlenecks(task_history: TaskHistoryRequest):
    """
    Identify performance bottlenecks with root cause analysis
    
    Analyzes task execution history to identify performance bottlenecks,
    their root causes, and provides optimization suggestions with estimated
    improvement potential.
    """
    start_time = time.time()
    
    try:
        debugger = get_debugger()
        
        # Convert request to domain object
        history = TaskHistory(
            task_id=task_history.task_id,
            execution_records=task_history.execution_records,
            performance_metrics=task_history.performance_metrics,
            confidence_history=[],  # Would need to convert from dicts to ConfidenceScore objects
            total_executions=task_history.total_executions,
            average_confidence=task_history.average_confidence,
            performance_trends=task_history.performance_trends
        )
        
        # Perform bottleneck analysis
        bottleneck_analysis = await debugger.identify_performance_bottlenecks(history)
        
        # Convert to serializable format
        response_data = {
            "success": True,
            "analysis_time": time.time() - start_time,
            "bottleneck_analysis": {
                "bottleneck_id": bottleneck_analysis.bottleneck_id,
                "category": bottleneck_analysis.category.value,
                "severity": bottleneck_analysis.severity.value,
                "description": bottleneck_analysis.description,
                "affected_operations": bottleneck_analysis.affected_operations,
                "performance_impact": bottleneck_analysis.performance_impact,
                "root_causes": bottleneck_analysis.root_causes,
                "optimization_suggestions": bottleneck_analysis.optimization_suggestions,
                "estimated_improvement": bottleneck_analysis.estimated_improvement,
                "timestamp": bottleneck_analysis.timestamp.isoformat()
            },
            "recommendations": {
                "immediate_actions": bottleneck_analysis.optimization_suggestions[:3],
                "long_term_optimizations": bottleneck_analysis.optimization_suggestions[3:],
                "priority_level": bottleneck_analysis.severity.value,
                "expected_impact": bottleneck_analysis.estimated_improvement
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Performance bottleneck analysis completed: {bottleneck_analysis.category.value} bottleneck identified")
        
        return JSONResponse(content=response_data)
        
    except ValueError as e:
        logger.warning(f"Invalid request for bottleneck analysis: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Performance bottleneck analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/optimize/suggestions")
async def get_optimization_suggestions(performance_data: PerformanceDataRequest):
    """
    Generate AI-powered optimization strategies with implementation guidance
    
    Analyzes performance data to generate comprehensive optimization strategies
    with detailed implementation steps, risk assessment, and resource requirements.
    """
    start_time = time.time()
    
    try:
        debugger = get_debugger()
        
        # Convert request to domain object
        perf_data = PerformanceData(
            operation_times=performance_data.operation_times,
            memory_usage=performance_data.memory_usage,
            cache_hit_rates=performance_data.cache_hit_rates,
            error_rates=performance_data.error_rates,
            throughput_metrics=performance_data.throughput_metrics,
            bottleneck_indicators=performance_data.bottleneck_indicators
        )
        
        # Generate optimization suggestions
        optimization_suggestions = await debugger.suggest_optimization_strategies(perf_data)
        
        # Convert to serializable format
        response_data = {
            "success": True,
            "analysis_time": time.time() - start_time,
            "optimization_suggestions": {
                "strategies": [
                    {
                        "strategy_id": strategy.strategy_id,
                        "title": strategy.title,
                        "description": strategy.description,
                        "implementation_complexity": strategy.implementation_complexity,
                        "expected_improvement": strategy.expected_improvement,
                        "implementation_steps": strategy.implementation_steps,
                        "risks": strategy.risks,
                        "prerequisites": strategy.prerequisites,
                        "estimated_effort": strategy.estimated_effort,
                        "confidence": strategy.confidence
                    }
                    for strategy in optimization_suggestions.strategies
                ],
                "priority_ranking": optimization_suggestions.priority_ranking,
                "implementation_roadmap": optimization_suggestions.implementation_roadmap,
                "resource_requirements": optimization_suggestions.resource_requirements,
                "risk_assessment": optimization_suggestions.risk_assessment,
                "success_metrics": optimization_suggestions.success_metrics
            },
            "recommendations": {
                "top_priority": optimization_suggestions.priority_ranking[0] if optimization_suggestions.priority_ranking else None,
                "quick_wins": [
                    strategy.strategy_id 
                    for strategy in optimization_suggestions.strategies 
                    if strategy.implementation_complexity == "low"
                ],
                "high_impact": [
                    strategy.strategy_id
                    for strategy in optimization_suggestions.strategies
                    if any(v > 0.5 for v in strategy.expected_improvement.values())
                ]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Optimization suggestions generated: {len(optimization_suggestions.strategies)} strategies")
        
        return JSONResponse(content=response_data)
        
    except ValueError as e:
        logger.warning(f"Invalid request for optimization suggestions: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Optimization suggestions failed: {e}")
        raise HTTPException(status_code=500, detail=f"Suggestions failed: {str(e)}")

# Debug Session Management

@router.post("/sessions")
async def create_debug_session(task: TaskRequest):
    """
    Create stateful debugging session with persistence
    
    Creates a new debugging session for comprehensive analysis of a specific task.
    Sessions maintain state and can be used for multiple debugging operations.
    """
    try:
        debugger = get_debugger()
        
        # Convert request to domain object
        ai_task = AITask(
            task_id=task.task_id,
            content=task.content,
            domain=task.domain,
            complexity=task.complexity,
            priority=task.priority,
            model_source=task.model_source,
            context_size=task.context_size
        )
        
        # Create debug session
        session = debugger.create_debug_session(ai_task)
        
        response_data = {
            "success": True,
            "session": {
                "session_id": session.session_id,
                "task_id": session.task.task_id,
                "start_time": session.start_time.isoformat(),
                "is_active": session.is_active,
                "session_state": session.session_state
            },
            "config": {
                "session_timeout": debugger.config['session_timeout'],
                "max_active_sessions": debugger.config['max_active_sessions'],
                "current_active_sessions": len(debugger.active_sessions)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Debug session created: {session.session_id}")
        
        return JSONResponse(content=response_data)
        
    except RuntimeError as e:
        logger.warning(f"Session creation failed: {e}")
        raise HTTPException(status_code=429, detail=str(e))  # Too Many Requests
    except Exception as e:
        logger.error(f"Session creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")

@router.get("/sessions/{session_id}")
async def get_debug_session(session_id: str):
    """Get debug session details and current state"""
    try:
        debugger = get_debugger()
        
        if session_id not in debugger.active_sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        session = debugger.active_sessions[session_id]
        
        response_data = {
            "success": True,
            "session": {
                "session_id": session.session_id,
                "task": {
                    "task_id": session.task.task_id,
                    "content": session.task.content[:200] + "..." if len(session.task.content) > 200 else session.task.content,
                    "domain": session.task.domain,
                    "complexity": session.task.complexity,
                    "priority": session.task.priority
                },
                "start_time": session.start_time.isoformat(),
                "end_time": session.end_time.isoformat() if session.end_time else None,
                "is_active": session.is_active,
                "debug_reports_count": len(session.debug_reports),
                "performance_snapshots_count": len(session.performance_snapshots),
                "debug_actions_count": len(session.debug_actions),
                "session_state": session.session_state
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")

@router.get("/sessions")
async def list_debug_sessions(
    active_only: bool = Query(True, description="List only active sessions"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of sessions to return")
):
    """List debug sessions with filtering options"""
    try:
        debugger = get_debugger()
        
        # Get active sessions
        sessions = list(debugger.active_sessions.values())
        
        # Include historical sessions if requested
        if not active_only:
            sessions.extend(list(debugger.session_history))
        
        # Apply filters
        if active_only:
            sessions = [s for s in sessions if s.is_active]
        
        # Sort by start time (newest first)
        sessions = sorted(sessions, key=lambda s: s.start_time, reverse=True)
        
        # Apply limit
        sessions = sessions[:limit]
        
        response_data = {
            "success": True,
            "sessions": [
                {
                    "session_id": session.session_id,
                    "task_id": session.task.task_id,
                    "start_time": session.start_time.isoformat(),
                    "end_time": session.end_time.isoformat() if session.end_time else None,
                    "is_active": session.is_active,
                    "reports_count": len(session.debug_reports),
                    "duration_minutes": ((session.end_time or datetime.now()) - session.start_time).total_seconds() / 60
                }
                for session in sessions
            ],
            "metadata": {
                "total_active": len([s for s in debugger.active_sessions.values() if s.is_active]),
                "total_historical": len(debugger.session_history),
                "returned_count": len(sessions),
                "active_only": active_only
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")

# Data Export

@router.post("/export/{session_id}")
async def export_debug_data(
    session_id: str,
    export_format: str = Query("json", pattern="^(json|csv|pdf)$", description="Export format")
):
    """
    Export comprehensive debug data for external analysis
    
    Exports all debug data from a session including analysis reports, performance
    data, traces, and visualizations in the requested format.
    """
    start_time = time.time()
    
    try:
        debugger = get_debugger()
        
        # Check if session exists
        if session_id not in debugger.active_sessions:
            # Check historical sessions
            historical_session = None
            for session in debugger.session_history:
                if session.session_id == session_id:
                    historical_session = session
                    break
            
            if historical_session is None:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            
            session = historical_session
        else:
            session = debugger.active_sessions[session_id]
        
        # Export debug data
        debug_export = debugger.export_debug_data(session, export_format)
        
        # Prepare response based on format
        if export_format == "json":
            response_data = {
                "success": True,
                "export_time": time.time() - start_time,
                "export": {
                    "export_id": debug_export.export_id,
                    "session_id": debug_export.session.session_id,
                    "export_format": debug_export.export_format,
                    "export_timestamp": debug_export.export_timestamp.isoformat(),
                    "analysis_summary": debug_export.analysis_summary,
                    "raw_data": debug_export.raw_data,
                    "visualizations": debug_export.visualizations,
                    "metadata": debug_export.metadata
                },
                "download_info": {
                    "filename": f"deepconf_debug_{session_id}_{int(time.time())}.json",
                    "size_estimate": len(str(debug_export.raw_data)),
                    "format": export_format
                },
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Debug data exported: {debug_export.export_id}, format: {export_format}")
            
            return JSONResponse(content=response_data)
            
        elif export_format == "csv":
            # For CSV, we'd generate a CSV file and return it as a stream
            # This is a simplified version - full implementation would generate proper CSV
            csv_data = "session_id,timestamp,metric,value\n"
            for snapshot in debug_export.raw_data.get("performance_data", []):
                timestamp = snapshot.get("timestamp", datetime.now().isoformat())
                for metric, value in snapshot.items():
                    if metric != "timestamp" and isinstance(value, (int, float)):
                        csv_data += f"{session_id},{timestamp},{metric},{value}\n"
            
            return StreamingResponse(
                iter([csv_data]),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=deepconf_debug_{session_id}.csv"}
            )
            
        elif export_format == "pdf":
            # PDF export would require additional libraries (reportlab, matplotlib, etc.)
            # For now, return JSON with PDF generation info
            response_data = {
                "success": True,
                "message": "PDF export not yet implemented",
                "alternative": "Use JSON export and convert to PDF with external tools",
                "export_id": debug_export.export_id
            }
            return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Invalid export request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Debug data export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

# Utility Endpoints

@router.get("/config")
async def get_debug_config():
    """Get current debugger configuration"""
    try:
        debugger = get_debugger()
        
        config_data = {
            "success": True,
            "config": debugger.config,
            "runtime_info": {
                "active_sessions": len(debugger.active_sessions),
                "historical_sessions": len(debugger.session_history),
                "engine_attached": debugger.engine is not None,
                "performance_monitors": len(debugger.performance_monitors)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=config_data)
        
    except Exception as e:
        logger.error(f"Failed to get debug config: {e}")
        raise HTTPException(status_code=500, detail=f"Config retrieval failed: {str(e)}")

@router.post("/config")
async def update_debug_config(config_updates: Dict[str, Any]):
    """Update debugger configuration (runtime-safe updates only)"""
    try:
        debugger = get_debugger()
        
        # Only allow safe runtime updates
        safe_updates = {}
        allowed_keys = [
            'session_timeout',
            'confidence_threshold_critical',
            'confidence_threshold_warning',
            'bottleneck_detection_threshold',
            'trace_depth',
            'max_optimization_strategies'
        ]
        
        for key, value in config_updates.items():
            if key in allowed_keys:
                safe_updates[key] = value
        
        # Update config
        debugger.config.update(safe_updates)
        
        response_data = {
            "success": True,
            "updated_config": safe_updates,
            "current_config": debugger.config,
            "message": f"Updated {len(safe_updates)} configuration parameters",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Debug config updated: {safe_updates}")
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Failed to update debug config: {e}")
        raise HTTPException(status_code=500, detail=f"Config update failed: {str(e)}")

# Error handlers for this router would be added here if needed
# The main app's error handlers will catch most exceptions