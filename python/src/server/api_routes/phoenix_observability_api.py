"""
Phoenix Observability API Routes

Provides REST API endpoints for Phoenix observability features:
- Performance metrics and analytics
- Agent behavior tracking
- Cost analysis and optimization
- Knowledge base usage analytics
- Vector database performance monitoring
- RAG pipeline analytics
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from ..services.phoenix_observability_service import get_phoenix_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/phoenix", tags=["phoenix-observability"])

# Pydantic models for API responses
class AIMetricsResponse(BaseModel):
    """AI/ML metrics response model"""
    llm_calls_total: int
    llm_calls_successful: int
    llm_calls_failed: int
    total_tokens_used: int
    total_cost_usd: float
    average_response_time: float
    error_rate: float
    cache_hit_rate: float
    rag_queries_total: int
    rag_queries_successful: int
    embedding_generations: int
    knowledge_base_hits: int
    model_usage_counts: Dict[str, int]
    provider_costs: Dict[str, float]

class AgentMetricsResponse(BaseModel):
    """Agent metrics response model"""
    agents_active: int
    tasks_completed: int
    tasks_failed: int
    average_task_duration: float
    collaboration_events: int
    handoff_success_rate: float
    agent_types: Dict[str, int]
    agent_success_rates: Dict[str, float]
    collaboration_patterns: Dict[str, int]

class VectorDBMetricsResponse(BaseModel):
    """Vector database metrics response model"""
    query_count: int
    query_success_count: int
    average_query_time_ms: float
    index_size_mb: float
    total_vectors: int
    embedding_dimension: int
    cache_hit_rate: float
    indexing_performance_ms: float
    similarity_threshold_stats: Dict[str, float]

class RAGPipelineMetricsResponse(BaseModel):
    """RAG pipeline metrics response model"""
    retrieval_count: int
    retrieval_success_count: int
    average_retrieval_time_ms: float
    average_documents_retrieved: float
    reranking_count: int
    reranking_success_count: int
    average_reranking_time_ms: float
    context_relevance_scores: List[float]
    answer_quality_scores: List[float]

class PerformanceSummaryResponse(BaseModel):
    """Comprehensive performance summary response"""
    uptime_seconds: float
    ai_metrics: AIMetricsResponse
    agent_metrics: AgentMetricsResponse
    vector_db_metrics: VectorDBMetricsResponse
    rag_pipeline_metrics: RAGPipelineMetricsResponse
    performance_events_count: int
    error_events_count: int
    collaboration_events_count: int
    knowledge_base_usage_count: int
    error_rate: float
    average_response_time: float
    observability_level: str
    phoenix_url: Optional[str]
    cost_tracking_enabled: bool
    performance_alerts_count: int

class CostAnalysisResponse(BaseModel):
    """Cost analysis response model"""
    period_hours: int
    total_estimated_cost: float
    costs_by_provider: Dict[str, float]
    costs_by_model: Dict[str, float]
    total_tokens: int
    average_cost_per_token: float
    optimization_suggestions: List[Dict[str, Any]]

class CollaborationGraphResponse(BaseModel):
    """Agent collaboration graph response"""
    graph_data: Dict[str, List[str]]
    collaboration_patterns: Dict[str, int]
    total_collaborations: int
    most_active_collaborations: List[tuple]

class KnowledgeBaseAnalyticsResponse(BaseModel):
    """Knowledge base analytics response"""
    period_hours: int
    total_queries: int
    source_type_distribution: Dict[str, int]
    total_documents_accessed: int
    average_documents_per_query: float
    average_relevance_score: float
    most_used_sources: List[tuple]

@router.get("/health", summary="Check Phoenix service health")
async def phoenix_health():
    """Check if Phoenix observability service is running and accessible"""
    try:
        service = await get_phoenix_service()
        return {
            "status": "healthy",
            "phoenix_initialized": service.initialized,
            "phoenix_url": "http://localhost:6006" if service.config.enabled else None,
            "observability_enabled": service.config.enabled,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Phoenix health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Phoenix service unavailable: {str(e)}")

@router.get("/metrics/ai", response_model=AIMetricsResponse, summary="Get AI/ML metrics")
async def get_ai_metrics():
    """Get current AI/ML performance metrics"""
    try:
        service = await get_phoenix_service()
        metrics = await service.get_ai_metrics()
        return AIMetricsResponse(**metrics.__dict__)
    except Exception as e:
        logger.error(f"Failed to get AI metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve AI metrics: {str(e)}")

@router.get("/metrics/agents", response_model=AgentMetricsResponse, summary="Get agent metrics")
async def get_agent_metrics():
    """Get current agent behavior metrics"""
    try:
        service = await get_phoenix_service()
        metrics = await service.get_agent_metrics()
        return AgentMetricsResponse(**metrics.__dict__)
    except Exception as e:
        logger.error(f"Failed to get agent metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agent metrics: {str(e)}")

@router.get("/metrics/vector-db", response_model=VectorDBMetricsResponse, summary="Get vector DB metrics")
async def get_vector_db_metrics():
    """Get vector database performance metrics"""
    try:
        service = await get_phoenix_service()
        metrics = await service.get_vector_db_metrics()
        return VectorDBMetricsResponse(**metrics.__dict__)
    except Exception as e:
        logger.error(f"Failed to get vector DB metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve vector DB metrics: {str(e)}")

@router.get("/metrics/rag-pipeline", response_model=RAGPipelineMetricsResponse, summary="Get RAG pipeline metrics")
async def get_rag_pipeline_metrics():
    """Get RAG pipeline performance metrics"""
    try:
        service = await get_phoenix_service()
        metrics = await service.get_rag_pipeline_metrics()
        return RAGPipelineMetricsResponse(**metrics.__dict__)
    except Exception as e:
        logger.error(f"Failed to get RAG pipeline metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve RAG pipeline metrics: {str(e)}")

@router.get("/summary", response_model=PerformanceSummaryResponse, summary="Get comprehensive performance summary")
async def get_performance_summary():
    """Get comprehensive performance summary across all components"""
    try:
        service = await get_phoenix_service()
        summary = await service.get_performance_summary()
        return PerformanceSummaryResponse(**summary)
    except Exception as e:
        logger.error(f"Failed to get performance summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance summary: {str(e)}")

@router.get("/analytics/cost", response_model=CostAnalysisResponse, summary="Get cost analysis")
async def get_cost_analysis(
    hours: int = Query(default=24, ge=1, le=168, description="Analysis period in hours (max 7 days)")
):
    """Get cost analysis and optimization suggestions"""
    try:
        service = await get_phoenix_service()
        analysis = await service.get_cost_analysis(hours)
        return CostAnalysisResponse(**analysis)
    except Exception as e:
        logger.error(f"Failed to get cost analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve cost analysis: {str(e)}")

@router.get("/analytics/collaboration-graph", response_model=CollaborationGraphResponse, summary="Get agent collaboration graph")
async def get_agent_collaboration_graph():
    """Get agent collaboration graph data and patterns"""
    try:
        service = await get_phoenix_service()
        graph_data = await service.get_agent_collaboration_graph()
        return CollaborationGraphResponse(**graph_data)
    except Exception as e:
        logger.error(f"Failed to get collaboration graph: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve collaboration graph: {str(e)}")

@router.get("/analytics/knowledge-base", response_model=KnowledgeBaseAnalyticsResponse, summary="Get knowledge base analytics")
async def get_knowledge_base_analytics(
    hours: int = Query(default=24, ge=1, le=168, description="Analysis period in hours (max 7 days)")
):
    """Get knowledge base usage analytics and patterns"""
    try:
        service = await get_phoenix_service()
        analytics = await service.get_knowledge_base_analytics(hours)
        return KnowledgeBaseAnalyticsResponse(**analytics)
    except Exception as e:
        logger.error(f"Failed to get knowledge base analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve knowledge base analytics: {str(e)}")

@router.get("/errors", summary="Get recent errors")
async def get_recent_errors(
    limit: int = Query(default=50, ge=1, le=200, description="Maximum number of errors to return")
):
    """Get recent errors for debugging and troubleshooting"""
    try:
        service = await get_phoenix_service()
        errors = await service.get_recent_errors(limit)
        return {
            "errors": errors,
            "total_count": len(errors),
            "limit": limit,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get recent errors: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve recent errors: {str(e)}")

@router.get("/insights", summary="Get performance insights")
async def get_performance_insights():
    """Get automated performance insights and recommendations"""
    try:
        service = await get_phoenix_service()
        insights = await service.get_insights()
        return {
            "insights": insights,
            "timestamp": datetime.utcnow().isoformat(),
            "phoenix_dashboard": "http://localhost:6006" if service.config.enabled else None
        }
    except Exception as e:
        logger.error(f"Failed to get performance insights: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance insights: {str(e)}")

@router.get("/dashboard", summary="Get Phoenix dashboard info")
async def get_phoenix_dashboard_info():
    """Get Phoenix dashboard URL and configuration"""
    try:
        service = await get_phoenix_service()
        return {
            "phoenix_url": "http://localhost:6006" if service.config.enabled else None,
            "project_name": service.config.project_name,
            "observability_level": service.config.observability_level.value,
            "auto_instrumentation": service.config.auto_instrumentation,
            "initialized": service.initialized,
            "features": {
                "llm_tracing": True,
                "agent_tracking": True,
                "vector_db_monitoring": True,
                "rag_pipeline_analytics": True,
                "cost_tracking": service.cost_tracking_enabled,
                "collaboration_analysis": True,
                "knowledge_base_analytics": True,
                "error_tracking": True
            }
        }
    except Exception as e:
        logger.error(f"Failed to get dashboard info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dashboard info: {str(e)}")

# Record events endpoints
@router.post("/record/knowledge-usage", summary="Record knowledge base usage")
async def record_knowledge_usage(
    source_id: str,
    source_type: str,
    documents_accessed: int,
    relevance_score: float = Field(ge=0.0, le=1.0),
    query: str
):
    """Record knowledge base usage for analytics"""
    try:
        service = await get_phoenix_service()
        service.record_knowledge_base_usage(
            source_id=source_id,
            source_type=source_type,
            documents_accessed=documents_accessed,
            relevance_score=relevance_score,
            query=query
        )
        return {
            "status": "recorded",
            "timestamp": datetime.utcnow().isoformat(),
            "source_id": source_id,
            "source_type": source_type
        }
    except Exception as e:
        logger.error(f"Failed to record knowledge usage: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record knowledge usage: {str(e)}")

@router.post("/record/embedding-generation", summary="Record embedding generation")
async def record_embedding_generation(
    model: str,
    text_count: int,
    total_tokens: int,
    duration_ms: float,
    success: bool = True
):
    """Record embedding generation metrics"""
    try:
        service = await get_phoenix_service()
        service.record_embedding_generation(
            model=model,
            text_count=text_count,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
            success=success
        )
        return {
            "status": "recorded",
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "text_count": text_count,
            "total_tokens": total_tokens,
            "duration_ms": duration_ms,
            "success": success
        }
    except Exception as e:
        logger.error(f"Failed to record embedding generation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record embedding generation: {str(e)}")

@router.post("/record/agent-collaboration", summary="Record agent collaboration")
async def record_agent_collaboration(
    from_agent: str,
    to_agent: str,
    collaboration_type: str,
    context: Dict[str, Any]
):
    """Record agent collaboration events"""
    try:
        service = await get_phoenix_service()
        service.record_agent_collaboration(
            from_agent=from_agent,
            to_agent=to_agent,
            collaboration_type=collaboration_type,
            context=context
        )
        return {
            "status": "recorded",
            "timestamp": datetime.utcnow().isoformat(),
            "from_agent": from_agent,
            "to_agent": to_agent,
            "collaboration_type": collaboration_type
        }
    except Exception as e:
        logger.error(f"Failed to record agent collaboration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record agent collaboration: {str(e)}")