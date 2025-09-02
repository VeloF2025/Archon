"""
External Validator Agent - FastAPI Application
Phase 5 of Archon Project
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .config import ValidatorConfig
from .models import (
    ValidationRequest,
    ValidationResponse,
    ConfigureValidatorRequest,
    ValidatorHealthResponse,
    ValidationStatus
)
from .validation_engine import ValidationEngine
from .mcp_integration import MCPIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
config = ValidatorConfig()
validation_engine = ValidationEngine(config)
mcp_integration = MCPIntegration(validation_engine)

# Metrics tracking
start_time = time.time()
total_validations = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("Starting External Validator Agent v1.0.0")
    
    # Load API key from database if available
    await config.load_from_database()
    
    # Initialize validation engine
    await validation_engine.initialize()
    
    # Register MCP tools
    await mcp_integration.register_tools()
    
    logger.info("External Validator ready on port 8053")
    
    yield
    
    # Cleanup
    await validation_engine.cleanup()
    logger.info("External Validator shutting down")


# Create FastAPI app
app = FastAPI(
    title="External Validator Agent",
    description="Phase 5: Independent validation service for Archon system",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3737", "http://localhost:8181"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=ValidatorHealthResponse)
async def health_check():
    """Health check endpoint"""
    
    llm_connected = await validation_engine.check_llm_connection()
    deterministic_available = validation_engine.check_deterministic_tools()
    
    status = "healthy"
    if not llm_connected:
        status = "degraded"
    if not deterministic_available:
        status = "degraded" if status == "healthy" else "unhealthy"
    
    return ValidatorHealthResponse(
        status=status,
        version="1.0.0",
        llm_provider=config.llm_config.provider,
        llm_connected=llm_connected,
        deterministic_available=deterministic_available,
        uptime_seconds=int(time.time() - start_time),
        total_validations=total_validations
    )


@app.post("/validate", response_model=ValidationResponse)
async def validate(
    request: ValidationRequest,
    background_tasks: BackgroundTasks
):
    """Main validation endpoint"""
    
    global total_validations
    total_validations += 1
    
    try:
        # Perform validation
        response = await validation_engine.validate(request)
        
        # Log validation in background
        background_tasks.add_task(
            log_validation,
            request.request_id,
            response.status,
            len(response.issues)
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {str(e)}"
        )


@app.post("/configure")
async def configure_validator(request: ConfigureValidatorRequest):
    """Configure validator settings"""
    
    try:
        # Update LLM configuration
        if any([request.provider, request.api_key, request.model, request.temperature]):
            config.update_llm_config(
                provider=request.provider,
                api_key=request.api_key,
                model=request.model,
                temperature=request.temperature
            )
        
        # Update validation configuration
        if request.confidence_threshold is not None:
            config.validation_config.confidence_threshold = request.confidence_threshold
        
        if request.enable_proactive_triggers is not None:
            config.validation_config.enable_proactive_triggers = request.enable_proactive_triggers
        
        # Save configuration
        config.save_config()
        
        # Reinitialize validation engine with new config
        await validation_engine.initialize()
        
        return {
            "status": "success",
            "message": "Validator configuration updated",
            "config": {
                "llm_provider": config.llm_config.provider,
                "model": config.llm_config.model,
                "temperature": config.llm_config.temperature,
                "confidence_threshold": config.validation_config.confidence_threshold
            }
        }
        
    except Exception as e:
        logger.error(f"Configuration error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Configuration failed: {str(e)}"
        )


@app.post("/refresh-api-key")
async def refresh_api_key():
    """Refresh API key from database"""
    
    try:
        # Force reload from database
        config._db_config_loaded = False
        await config.load_from_database()
        
        # Reinitialize validation engine with new config
        await validation_engine.initialize()
        
        has_key = config.llm_config.api_key is not None
        
        return {
            "status": "success",
            "message": "API key refreshed from database" if has_key else "No validator API key in database",
            "provider": config.llm_config.provider if has_key else None,
            "model": config.llm_config.model if has_key else None
        }
    except Exception as e:
        logger.error(f"Failed to refresh API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_configuration():
    """Get current validator configuration"""
    
    return {
        "llm": {
            "provider": config.llm_config.provider,
            "model": config.llm_config.model,
            "temperature": config.llm_config.temperature,
            "has_api_key": config.llm_config.api_key is not None
        },
        "validation": {
            "enable_deterministic": config.validation_config.enable_deterministic,
            "enable_cross_check": config.validation_config.enable_cross_check,
            "confidence_threshold": config.validation_config.confidence_threshold,
            "max_context_tokens": config.validation_config.max_context_tokens,
            "enable_proactive_triggers": config.validation_config.enable_proactive_triggers
        }
    }


@app.post("/trigger/{event_type}")
async def trigger_validation(
    event_type: str,
    payload: dict,
    background_tasks: BackgroundTasks
):
    """Webhook endpoint for proactive validation triggers"""
    
    if not config.validation_config.enable_proactive_triggers:
        return {"status": "disabled", "message": "Proactive triggers are disabled"}
    
    try:
        # Map event type to validation request
        validation_request = map_event_to_validation(event_type, payload)
        
        if validation_request:
            # Perform validation in background
            background_tasks.add_task(
                perform_background_validation,
                validation_request
            )
            
            return {
                "status": "triggered",
                "event_type": event_type,
                "message": "Validation triggered in background"
            }
        else:
            return {
                "status": "skipped",
                "event_type": event_type,
                "message": "Event type not configured for validation"
            }
            
    except Exception as e:
        logger.error(f"Trigger error: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/metrics")
async def get_metrics():
    """Get validation metrics"""
    
    metrics = await validation_engine.get_metrics()
    
    return {
        "total_validations": total_validations,
        "uptime_seconds": int(time.time() - start_time),
        "engine_metrics": metrics
    }


# Helper functions

async def log_validation(
    request_id: Optional[str],
    status: ValidationStatus,
    issue_count: int
):
    """Log validation for metrics and auditing"""
    
    logger.info(
        f"Validation completed - ID: {request_id}, "
        f"Status: {status}, Issues: {issue_count}"
    )


async def perform_background_validation(request: ValidationRequest):
    """Perform validation in background"""
    
    try:
        response = await validation_engine.validate(request)
        
        # Send notification if validation failed
        if response.status == ValidationStatus.FAIL:
            await send_validation_alert(response)
            
    except Exception as e:
        logger.error(f"Background validation error: {e}", exc_info=True)


async def send_validation_alert(response: ValidationResponse):
    """Send alert for failed validation"""
    
    # This would integrate with Archon's notification system
    logger.warning(
        f"Validation alert - Request: {response.request_id}, "
        f"Issues: {len(response.issues)}, "
        f"Summary: {response.summary}"
    )


def map_event_to_validation(event_type: str, payload: dict) -> Optional[ValidationRequest]:
    """Map webhook event to validation request"""
    
    event_mappings = {
        "agent_output": lambda p: ValidationRequest(
            output=p.get("output", ""),
            context=p.get("context"),
            validation_type="output"
        ),
        "code_change": lambda p: ValidationRequest(
            output=p.get("code", ""),
            file_paths=p.get("files"),
            validation_type="code"
        ),
        "prompt_submission": lambda p: ValidationRequest(
            prompt=p.get("prompt"),
            output=p.get("response", ""),
            validation_type="prompt"
        )
    }
    
    if event_type in event_mappings:
        return event_mappings[event_type](payload)
    
    return None


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8053,
        reload=True,
        log_level="info"
    )