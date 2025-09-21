"""
Multi-Model Intelligence Fusion API
API endpoints for the multi-model AI system with intelligent routing and optimization.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import asyncio
import json
import logging
from datetime import datetime

from ...agents.multi_model.model_ensemble import ModelEnsemble, TaskType, TaskRequest, RoutingStrategy
from ...agents.multi_model.predictive_scaler import PredictiveAgentScaler
from ...agents.multi_model.benchmark_system import BenchmarkSuite, BenchmarkType
from ...agents.multi_model.intelligent_router import IntelligentModelRouter
from ..services.redis_service import get_redis_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/multi-model", tags=["Multi-Model Intelligence"])

# Global instances (initialized on startup)
model_ensemble: Optional[ModelEnsemble] = None
predictive_scaler: Optional[PredictiveAgentScaler] = None
benchmark_suite: Optional[BenchmarkSuite] = None
intelligent_router: Optional[IntelligentModelRouter] = None


# Request/Response Models
class MultiModelRequest(BaseModel):
    """Request for multi-model execution."""
    prompt: str = Field(..., description="The prompt to execute")
    task_type: str = Field(default="analysis", description="Type of task (coding, creative_writing, analysis, etc.)")
    strategy: Optional[str] = Field(None, description="Routing strategy (performance_first, cost_first, balanced, etc.)")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens for response")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    urgency: str = Field(default="normal", description="Task urgency (low, normal, high, critical)")
    cost_budget: Optional[float] = Field(None, description="Maximum cost budget for this request")
    quality_requirement: float = Field(default=0.8, description="Minimum quality requirement (0-1)")
    system_prompt: Optional[str] = Field(None, description="System prompt for context")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for routing")


class MultiModelResponse(BaseModel):
    """Response from multi-model execution."""
    content: str
    model_used: str
    provider: str
    routing_decision: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    cost: float
    response_time: float
    quality_score: float
    success: bool
    error_message: Optional[str] = None


class BenchmarkRequest(BaseModel):
    """Request for running benchmarks."""
    models_to_test: Optional[List[str]] = Field(None, description="Specific models to test")
    benchmark_types: Optional[List[str]] = Field(None, description="Types of benchmarks to run")
    include_detailed_results: bool = Field(default=False, description="Include detailed results")


class RoutingStatusResponse(BaseModel):
    """Response for routing system status."""
    routing_metrics: Dict[str, Any]
    ml_model_status: Dict[str, Any]
    cost_optimization: Dict[str, Any]
    circuit_breakers: Dict[str, Any]
    ab_testing: Dict[str, Any]
    budget_status: Dict[str, Any]


async def get_multi_model_system():
    """Initialize multi-model system components."""
    global model_ensemble, predictive_scaler, benchmark_suite, intelligent_router
    
    if not model_ensemble:
        redis_client = await get_redis_client()
        
        # Initialize components
        model_ensemble = ModelEnsemble(redis_client=redis_client)
        predictive_scaler = PredictiveAgentScaler(model_ensemble, redis_client=redis_client)
        benchmark_suite = BenchmarkSuite(model_ensemble, redis_client=redis_client)
        intelligent_router = IntelligentModelRouter(
            model_ensemble=model_ensemble,
            predictive_scaler=predictive_scaler,
            benchmark_suite=benchmark_suite,
            redis_client=redis_client
        )
        
        # Start predictive scaler
        await predictive_scaler.start()
        
        logger.info("Multi-model intelligence system initialized")
    
    return {
        "ensemble": model_ensemble,
        "scaler": predictive_scaler,
        "benchmark": benchmark_suite,
        "router": intelligent_router
    }


@router.post("/execute", response_model=MultiModelResponse)
async def execute_multi_model_request(
    request: MultiModelRequest,
    background_tasks: BackgroundTasks
) -> MultiModelResponse:
    """Execute a request using the multi-model intelligence system."""
    try:
        system = await get_multi_model_system()
        router_instance = system["router"]
        
        # Convert request to internal format
        task_request = TaskRequest(
            prompt=request.prompt,
            task_type=TaskType(request.task_type),
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            cost_budget=request.cost_budget,
            quality_requirement=request.quality_requirement,
            urgency=request.urgency,
            system_prompt=request.system_prompt,
            context=request.context
        )
        
        # Get routing decision
        routing_strategy = None
        if request.strategy:
            routing_strategy = RoutingStrategy(request.strategy)
        
        routing_decision = await router_instance.route_request(
            task_request, 
            strategy=routing_strategy,
            context=request.context
        )
        
        # Execute with selected model
        start_time = asyncio.get_event_loop().time()
        response = await system["ensemble"].execute_task(task_request)
        execution_time = asyncio.get_event_loop().time() - start_time
        
        # Update budget tracking in background
        if response.success and response.cost > 0:
            background_tasks.add_task(
                router_instance.budget_manager.record_spending,
                response.cost,
                routing_decision.selected_model,
                request.task_type
            )
        
        return MultiModelResponse(
            content=response.content,
            model_used=response.model_name,
            provider=response.provider.value,
            routing_decision={
                "selected_model": routing_decision.selected_model,
                "strategy_used": routing_decision.strategy_used.value,
                "confidence_score": routing_decision.confidence_score,
                "reasoning": routing_decision.reasoning,
                "fallback_models": routing_decision.fallback_models,
                "factors_considered": routing_decision.factors_considered
            },
            performance_metrics={
                "tokens_used": response.tokens_used,
                "response_time": response.response_time,
                "execution_time": execution_time,
                "metadata": response.metadata
            },
            cost=response.cost,
            response_time=response.response_time,
            quality_score=response.quality_score,
            success=response.success,
            error_message=response.error_message
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except Exception as e:
        logger.error(f"Multi-model execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")


@router.post("/execute/stream")
async def execute_multi_model_stream(
    request: MultiModelRequest
) -> StreamingResponse:
    """Execute a request with streaming response."""
    try:
        system = await get_multi_model_system()
        
        async def generate_stream():
            try:
                # Get routing decision
                task_request = TaskRequest(
                    prompt=request.prompt,
                    task_type=TaskType(request.task_type),
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    cost_budget=request.cost_budget,
                    quality_requirement=request.quality_requirement,
                    urgency=request.urgency,
                    system_prompt=request.system_prompt,
                    context=request.context
                )
                
                routing_strategy = None
                if request.strategy:
                    routing_strategy = RoutingStrategy(request.strategy)
                
                routing_decision = await system["router"].route_request(
                    task_request, 
                    strategy=routing_strategy,
                    context=request.context
                )
                
                # Send routing decision first
                yield f"data: {json.dumps({'type': 'routing_decision', 'data': routing_decision.__dict__, 'timestamp': datetime.now().isoformat()})}\n\n"
                
                # Execute and stream response
                response = await system["ensemble"].execute_task(task_request)
                
                # For now, send complete response (in future, could implement true streaming)
                result_data = {
                    "type": "complete",
                    "content": response.content,
                    "model_used": response.model_name,
                    "provider": response.provider.value,
                    "cost": response.cost,
                    "response_time": response.response_time,
                    "quality_score": response.quality_score,
                    "success": response.success,
                    "error_message": response.error_message,
                    "timestamp": datetime.now().isoformat()
                }
                
                yield f"data: {json.dumps(result_data)}\n\n"
                
            except Exception as e:
                error_data = {
                    "type": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        logger.error(f"Stream execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/status")
async def get_models_status():
    """Get status of all available models."""
    try:
        system = await get_multi_model_system()
        status = await system["ensemble"].get_model_status()
        return {"models": status, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Failed to get models status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/routing/status", response_model=RoutingStatusResponse)
async def get_routing_status() -> RoutingStatusResponse:
    """Get routing system status and metrics."""
    try:
        system = await get_multi_model_system()
        status = await system["router"].get_routing_status()
        return RoutingStatusResponse(**status)
    except Exception as e:
        logger.error(f"Failed to get routing status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scaling/status")
async def get_scaling_status():
    """Get predictive scaling status."""
    try:
        system = await get_multi_model_system()
        status = await system["scaler"].get_scaling_status()
        return {"scaling_status": status, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Failed to get scaling status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/benchmark/run")
async def run_benchmark(
    request: BenchmarkRequest,
    background_tasks: BackgroundTasks
):
    """Run performance benchmarks across providers."""
    try:
        system = await get_multi_model_system()
        
        # Convert string benchmark types to enums
        benchmark_types = None
        if request.benchmark_types:
            benchmark_types = [BenchmarkType(bt) for bt in request.benchmark_types]
        
        # Run benchmark in background
        background_tasks.add_task(
            system["benchmark"].run_comprehensive_benchmark,
            models_to_test=request.models_to_test,
            benchmark_types=benchmark_types
        )
        
        return {
            "message": "Benchmark started",
            "models_to_test": request.models_to_test or "all",
            "benchmark_types": request.benchmark_types or "all",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/benchmark/results")
async def get_benchmark_results():
    """Get latest benchmark results."""
    try:
        system = await get_multi_model_system()
        results = await system["benchmark"].get_latest_benchmark_results()
        
        if not results:
            return {"message": "No benchmark results available", "timestamp": datetime.now().isoformat()}
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to get benchmark results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimization/rules")
async def update_cost_optimization_rules(
    rules: Dict[str, Dict[str, Any]]
):
    """Update cost optimization rules."""
    try:
        system = await get_multi_model_system()
        
        # Update rules in router
        for rule_name, rule_config in rules.items():
            if rule_name in system["router"].cost_optimization_rules:
                rule = system["router"].cost_optimization_rules[rule_name]
                
                if "active" in rule_config:
                    rule.active = rule_config["active"]
                if "savings_target" in rule_config:
                    rule.savings_target = rule_config["savings_target"]
                if "priority" in rule_config:
                    rule.priority = rule_config["priority"]
        
        return {
            "message": "Cost optimization rules updated",
            "updated_rules": list(rules.keys()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to update optimization rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimization/rules")
async def get_cost_optimization_rules():
    """Get current cost optimization rules."""
    try:
        system = await get_multi_model_system()
        
        rules_data = {}
        for rule_name, rule in system["router"].cost_optimization_rules.items():
            rules_data[rule_name] = {
                "name": rule.name,
                "condition": rule.condition,
                "action": rule.action,
                "priority": rule.priority,
                "active": rule.active,
                "savings_target": rule.savings_target
            }
        
        return {
            "rules": rules_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get optimization rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/budget/update")
async def update_budget(
    daily_budget: float = Field(..., description="New daily budget amount")
):
    """Update daily budget limit."""
    try:
        system = await get_multi_model_system()
        
        system["router"].budget_manager.daily_budget = daily_budget
        
        return {
            "message": "Budget updated",
            "new_daily_budget": daily_budget,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to update budget: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/budget/status")
async def get_budget_status():
    """Get current budget status."""
    try:
        system = await get_multi_model_system()
        status = await system["router"].budget_manager.get_budget_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get budget status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategy/test")
async def start_ab_test(
    test_strategy: str = Field(..., description="Strategy to test"),
    test_percentage: float = Field(default=0.1, description="Percentage of traffic for test"),
    duration_hours: int = Field(default=24, description="Test duration in hours")
):
    """Start A/B test for routing strategy."""
    try:
        system = await get_multi_model_system()
        
        # Validate strategy
        try:
            strategy_enum = RoutingStrategy(test_strategy)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid strategy: {test_strategy}")
        
        # Update A/B test configuration
        system["router"].ab_test_config.update({
            "enabled": True,
            "current_test": test_strategy,
            "test_percentage": test_percentage,
            "start_time": datetime.now(),
            "duration_hours": duration_hours
        })
        
        return {
            "message": "A/B test started",
            "test_strategy": test_strategy,
            "test_percentage": test_percentage,
            "duration_hours": duration_hours,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategy/stop_test")
async def stop_ab_test():
    """Stop current A/B test."""
    try:
        system = await get_multi_model_system()
        
        system["router"].ab_test_config.update({
            "enabled": False,
            "current_test": None
        })
        
        return {
            "message": "A/B test stopped",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to stop A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/performance")
async def get_performance_analytics():
    """Get performance analytics across all models."""
    try:
        system = await get_multi_model_system()
        
        # Collect analytics from various components
        routing_status = await system["router"].get_routing_status()
        model_status = await system["ensemble"].get_model_status()
        scaling_status = await system["scaler"].get_scaling_status()
        
        analytics = {
            "overview": {
                "total_requests": routing_status["routing_metrics"]["total_requests"],
                "avg_cost_per_request": routing_status["routing_metrics"]["avg_cost_per_request"],
                "models_available": len([m for m, s in model_status.items() if s["available"]]),
                "total_models": len(model_status),
            },
            "model_performance": model_status,
            "routing_metrics": routing_status["routing_metrics"],
            "cost_optimization": {
                "budget_utilization": routing_status["budget_status"]["utilization_percentage"],
                "active_rules": routing_status["cost_optimization"]["active_rules"],
                "estimated_savings": "Not yet calculated"  # Would need historical comparison
            },
            "scaling_metrics": scaling_status,
            "timestamp": datetime.now().isoformat()
        }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to get performance analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for multi-model system."""
    try:
        system = await get_multi_model_system()
        
        # Basic health checks
        health_status = {
            "status": "healthy",
            "components": {
                "model_ensemble": "healthy" if system["ensemble"] else "unavailable",
                "predictive_scaler": "healthy" if system["scaler"] else "unavailable", 
                "benchmark_suite": "healthy" if system["benchmark"] else "unavailable",
                "intelligent_router": "healthy" if system["router"] else "unavailable"
            },
            "models_available": len([
                m for m, s in (await system["ensemble"].get_model_status()).items() 
                if s["available"]
            ]) if system["ensemble"] else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        # Check if any critical components are down
        if any(status == "unavailable" for status in health_status["components"].values()):
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Startup event to initialize system
@router.on_event("startup")
async def startup_event():
    """Initialize multi-model system on startup."""
    try:
        logger.info("Initializing multi-model intelligence system...")
        await get_multi_model_system()
        logger.info("Multi-model intelligence system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize multi-model system: {e}")


# Shutdown event to cleanup
@router.on_event("shutdown") 
async def shutdown_event():
    """Cleanup multi-model system on shutdown."""
    try:
        logger.info("Shutting down multi-model intelligence system...")
        
        global predictive_scaler
        if predictive_scaler:
            await predictive_scaler.stop()
        
        logger.info("Multi-model intelligence system shut down successfully")
    except Exception as e:
        logger.error(f"Error during multi-model system shutdown: {e}")