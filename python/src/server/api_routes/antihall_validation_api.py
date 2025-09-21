"""
Anti-Hallucination Validation API Routes

Provides REST API endpoints for code validation, confidence checking,
and hallucination prevention in AI-generated code.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from ..services.validation_service import (
    get_validation_service,
    initialize_validation_service,
    ValidationService
)
from ...agents.validation.enhanced_antihall_validator import ValidationResult
from ...agents.validation.confidence_based_responses import ConfidenceLevel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/validation", tags=["validation"])

# Pydantic models for API

class CodeReferenceRequest(BaseModel):
    """Request to validate a code reference"""
    reference: str = Field(..., description="Code reference to validate (function, class, method, etc.)")
    reference_type: str = Field("generic", description="Type of reference: function, class, method, import, file, generic")
    context: Optional[str] = Field(None, description="Additional context (e.g., class name for methods)")

class CodeSnippetRequest(BaseModel):
    """Request to validate a code snippet"""
    code: str = Field(..., description="Code snippet to validate")
    language: str = Field("python", description="Programming language: python, typescript, javascript")
    min_confidence: Optional[float] = Field(None, description="Minimum confidence threshold (default: 0.75)")

class ConfidenceCheckRequest(BaseModel):
    """Request to check confidence for a response"""
    content: str = Field(..., description="Content to evaluate")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context for confidence assessment")
    question_type: str = Field("general", description="Type of question: general, code_implementation, bug_fix, architecture")

class AgentResponseRequest(BaseModel):
    """Request to validate an AI agent response"""
    response: str = Field(..., description="AI agent response to validate")
    contains_code: bool = Field(False, description="Whether response contains code")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

class RealTimeValidationRequest(BaseModel):
    """Request for real-time line validation"""
    line: str = Field(..., description="Single line of code to validate")
    context: Dict[str, Any] = Field(default_factory=dict, description="Current coding context")
    language: str = Field("python", description="Programming language")

# Dependency to get validation service
async def get_service() -> ValidationService:
    """Get or initialize validation service"""
    service = get_validation_service()
    if not service:
        service = await initialize_validation_service()
    return service

# API Endpoints

@router.post("/reference", summary="Validate code reference")
async def validate_code_reference(
    request: CodeReferenceRequest,
    service: ValidationService = Depends(get_service)
):
    """
    Validate a specific code reference (function, class, method, etc.)
    to ensure it exists in the codebase
    """
    try:
        report = await service.validate_code_reference(
            request.reference,
            request.reference_type
        )
        
        return {
            "reference": request.reference,
            "reference_type": request.reference_type,
            "exists": report.result == ValidationResult.EXISTS,
            "result": report.result.value,
            "confidence": report.confidence,
            "actual_location": report.actual_location,
            "similar_matches": report.similar_matches,
            "suggestion": report.suggestion,
            "evidence": report.evidence
        }
        
    except Exception as e:
        logger.error(f"Error validating reference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/snippet", summary="Validate code snippet")
async def validate_code_snippet(
    request: CodeSnippetRequest,
    service: ValidationService = Depends(get_service)
):
    """
    Validate an entire code snippet for hallucinations and confidence.
    Returns detailed validation results and confidence scores.
    """
    try:
        result = await service.validate_code_snippet(
            request.code,
            request.language,
            request.min_confidence
        )
        
        return {
            "valid": result["valid"],
            "validation_summary": result["validation_summary"],
            "timestamp": result["timestamp"],
            "recommendations": [
                "Fix invalid references before using this code",
                "Verify uncertain references manually",
                "Consider using suggested alternatives"
            ] if not result["valid"] else []
        }
        
    except Exception as e:
        logger.error(f"Error validating snippet: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/confidence", summary="Check confidence level")
async def check_confidence(
    request: ConfidenceCheckRequest,
    service: ValidationService = Depends(get_service)
):
    """
    Check confidence level for a response and get appropriate formatting.
    Enforces the 75% confidence rule.
    """
    try:
        result = await service.validate_with_confidence(
            request.content,
            request.context
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error checking confidence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agent-response", summary="Validate AI agent response")
async def validate_agent_response(
    request: AgentResponseRequest,
    service: ValidationService = Depends(get_service)
):
    """
    Validate an AI agent's response for hallucinations, uncertainty patterns,
    and confidence levels. Rewrites response if needed.
    """
    try:
        result = await service.validate_agent_response(
            request.response,
            {
                "contains_code": request.contains_code,
                **request.context
            }
        )
        
        return {
            "original_response": result["original_response"],
            "confidence_score": result["confidence_result"]["confidence_score"],
            "confidence_level": result["confidence_result"]["confidence_level"],
            "validated_response": result["confidence_result"]["response"],
            "validation_passed": all(v["valid"] for v in result["validation_results"]),
            "validation_details": result["validation_results"],
            "uncertainty_detected": result["uncertainty_detected"],
            "uncertainty_patterns": result["uncertainty_patterns"],
            "suggestions": result["confidence_result"].get("suggestions", [])
        }
        
    except Exception as e:
        logger.error(f"Error validating agent response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/real-time", summary="Real-time line validation")
async def validate_line_real_time(
    request: RealTimeValidationRequest,
    service: ValidationService = Depends(get_service)
):
    """
    Validate a single line of code in real-time as it's being typed.
    Returns immediate feedback about potential hallucinations.
    """
    try:
        error = await service.perform_real_time_validation(
            request.line,
            request.context
        )
        
        return {
            "line": request.line,
            "valid": error is None,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in real-time validation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics", summary="Get validation statistics")
async def get_validation_statistics(
    service: ValidationService = Depends(get_service)
):
    """
    Get comprehensive statistics about validation service performance,
    including cache hit rates, validation success rates, and confidence metrics.
    """
    try:
        stats = service.get_statistics()
        
        return {
            "statistics": stats,
            "performance_metrics": {
                "cache_hit_rate": f"{stats['cache_hit_rate']:.2%}",
                "validation_success_rate": f"{stats['validation_success_rate']:.2%}",
                "confidence_block_rate": f"{stats['confidence_block_rate']:.2%}",
                "auto_fix_success_rate": f"{stats['auto_fix_success_rate']:.2%}"
            },
            "summary": {
                "total_validations": stats["total_validations"],
                "hallucinations_prevented": stats["hallucinations_prevented"],
                "low_confidence_blocks": stats["low_confidence_blocks"],
                "average_confidence": f"{stats['average_confidence']:.2%}"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear-cache", summary="Clear validation cache")
async def clear_validation_cache(
    service: ValidationService = Depends(get_service)
):
    """
    Clear the validation cache. Useful after significant code changes.
    """
    try:
        service.clear_cache()
        
        return {
            "status": "success",
            "message": "Validation cache cleared successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", summary="Validation service health check")
async def validation_health_check(
    service: ValidationService = Depends(get_service)
):
    """
    Check if validation service is healthy and operational.
    """
    try:
        stats = service.get_statistics()
        
        return {
            "status": "healthy",
            "service": "validation",
            "min_confidence_threshold": service.config.min_confidence_threshold,
            "real_time_validation": service.config.enable_real_time_validation,
            "auto_fix_enabled": service.config.enable_auto_fix,
            "cache_enabled": service.config.cache_validation_results,
            "project_root": service.config.project_root,
            "code_index_size": len(service.validator.code_index),
            "total_validations": stats["total_validations"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "unhealthy",
            "service": "validation",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Batch validation endpoints

@router.post("/batch/references", summary="Validate multiple references")
async def validate_batch_references(
    references: List[CodeReferenceRequest],
    service: ValidationService = Depends(get_service)
):
    """
    Validate multiple code references in a single request.
    Useful for pre-validating all references in a code file.
    """
    try:
        results = []
        
        for ref_request in references:
            report = await service.validate_code_reference(
                ref_request.reference,
                ref_request.reference_type
            )
            
            results.append({
                "reference": ref_request.reference,
                "exists": report.result == ValidationResult.EXISTS,
                "confidence": report.confidence,
                "location": report.actual_location
            })
            
        # Calculate summary
        total = len(results)
        valid = sum(1 for r in results if r["exists"])
        
        return {
            "results": results,
            "summary": {
                "total_references": total,
                "valid_references": valid,
                "invalid_references": total - valid,
                "validation_rate": valid / total if total > 0 else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error in batch validation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time validation stream (if needed)
# This would require additional WebSocket setup

@router.get("/config", summary="Get validation configuration")
async def get_validation_config(
    service: ValidationService = Depends(get_service)
):
    """
    Get current validation service configuration
    """
    return {
        "project_root": service.config.project_root,
        "min_confidence_threshold": service.config.min_confidence_threshold,
        "enable_real_time_validation": service.config.enable_real_time_validation,
        "enable_auto_fix": service.config.enable_auto_fix,
        "cache_validation_results": service.config.cache_validation_results,
        "max_cache_size": service.config.max_cache_size
    }