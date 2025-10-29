"""
Phoenix Observability Integration for Main Server

Integration point for Phoenix observability service in the main FastAPI server.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI

from .api_routes.phoenix_observability_api import router as phoenix_router
from .services.phoenix_observability_service import get_phoenix_service

logger = logging.getLogger(__name__)


async def initialize_phoenix_observability():
    """Initialize Phoenix observability service"""
    try:
        phoenix_service = await get_phoenix_service()
        success = await phoenix_service.initialize()

        if success:
            logger.info("Phoenix observability initialized successfully")
            logger.info(f"Phoenix dashboard available at: http://localhost:6006")
        else:
            logger.warning("Phoenix observability initialization failed - service will run without observability")

        return success
    except Exception as e:
        logger.error(f"Failed to initialize Phoenix observability: {e}")
        return False


async def shutdown_phoenix_observability():
    """Shutdown Phoenix observability service"""
    try:
        phoenix_service = await get_phoenix_service()
        await phoenix_service.shutdown()
        logger.info("Phoenix observability shutdown complete")
    except Exception as e:
        logger.error(f"Error during Phoenix shutdown: {e}")


def add_phoenix_routes_to_app(app: FastAPI):
    """Add Phoenix observability routes to FastAPI app"""
    try:
        app.include_router(phoenix_router)
        logger.info("Phoenix observability routes added to application")

        # Add health check endpoint that includes Phoenix status
        @app.get("/api/health/phoenix", tags=["health"])
        async def phoenix_health():
            """Phoenix observability health check"""
            try:
                phoenix_service = await get_phoenix_service()
                return {
                    "status": "healthy",
                    "phoenix_initialized": phoenix_service.initialized,
                    "phoenix_url": "http://localhost:6006" if phoenix_service.config.enabled else None,
                    "observability_enabled": phoenix_service.config.enabled
                }
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "phoenix_url": None
                }

        logger.info("Phoenix health check endpoint added")

    except Exception as e:
        logger.error(f"Failed to add Phoenix routes: {e}")


@asynccontextmanager
async def phoenix_lifespan_context(app: FastAPI):
    """Lifespan context manager for Phoenix observability"""
    # Startup
    logger.info("Starting Phoenix observability...")
    await initialize_phoenix_observability()

    # Add routes after initialization
    add_phoenix_routes_to_app(app)

    yield

    # Shutdown
    logger.info("Shutting down Phoenix observability...")
    await shutdown_phoenix_observability()


# Utility function to create FastAPI app with Phoenix integration
def create_app_with_phoenix(
    title: str = "Archon API Server with Phoenix Observability",
    description: str = "Archon AI platform with comprehensive observability",
    version: str = "2.0.0"
) -> FastAPI:
    """Create FastAPI app with Phoenix observability integration"""

    app = FastAPI(
        title=title,
        description=description,
        version=version,
        lifespan=phoenix_lifespan_context
    )

    # Phoenix routes will be added in the lifespan context
    return app


# Integration examples for existing services

async def trace_llm_call_with_phoenix(
    provider: str,
    model: str,
    operation: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """Utility function to trace LLM calls with Phoenix"""
    try:
        phoenix_service = await get_phoenix_service()
        return phoenix_service.trace_llm_call(provider, model, operation, metadata)
    except Exception as e:
        logger.warning(f"Failed to create LLM trace with Phoenix: {e}")
        # Return a no-op context manager if Phoenix is not available
        from contextlib import nullcontext
        return nullcontext()


async def trace_agent_operation_with_phoenix(
    agent_name: str,
    operation: str,
    task_type: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """Utility function to trace agent operations with Phoenix"""
    try:
        phoenix_service = await get_phoenix_service()
        return phoenix_service.trace_agent_operation(agent_name, operation, task_type, metadata)
    except Exception as e:
        logger.warning(f"Failed to create agent trace with Phoenix: {e}")
        from contextlib import nullcontext
        return nullcontext()


async def record_phoenix_metrics(
    event_type: str,
    component: str,
    metrics: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
):
    """Utility function to record metrics with Phoenix"""
    try:
        phoenix_service = await get_phoenix_service()
        phoenix_service.record_performance_event(event_type, component, metrics, metadata)
    except Exception as e:
        logger.warning(f"Failed to record Phoenix metrics: {e}")


# Middleware integration example
async def phoenix_middleware_request_handler(request, call_next):
    """Example middleware for Phoenix request tracing"""
    try:
        phoenix_service = await get_phoenix_service()

        # Record request start
        import time
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Record request completion
        duration = time.time() - start_time

        phoenix_service.record_performance_event(
            event_type="http_request",
            component="api_server",
            metrics={
                "duration_ms": duration * 1000,
                "status_code": response.status_code,
                "method": request.method,
                "path": request.url.path
            },
            metadata={
                "user_agent": request.headers.get("user-agent"),
                "remote_addr": request.client.host if request.client else None
            }
        )

        return response

    except Exception as e:
        logger.warning(f"Phoenix middleware error: {e}")
        # Continue without observability if Phoenix fails
        return await call_next(request)