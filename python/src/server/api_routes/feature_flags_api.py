"""
Feature Flags API Routes

Provides REST endpoints for managing and evaluating feature flags.
Enables runtime control of features without code deployments.
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from datetime import datetime

from ..services.feature_flag_service import (
    get_feature_flag_service,
    FeatureFlag,
    FlagType,
    FlagStatus
)

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/feature-flags", tags=["feature-flags"])


class FlagEvaluationRequest(BaseModel):
    """Request for evaluating a feature flag"""
    flag_key: str = Field(..., description="Feature flag key")
    user_id: Optional[str] = Field(None, description="User ID for targeting")
    attributes: Optional[Dict[str, Any]] = Field(None, description="User attributes")


class FlagCreateRequest(BaseModel):
    """Request for creating a new feature flag"""
    key: str = Field(..., description="Unique flag identifier")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="What this flag controls")
    flag_type: str = Field("boolean", description="Type of flag")
    default_value: Any = Field(False, description="Default value")
    enabled_for_all: bool = Field(False, description="Enable for all users")
    percentage: Optional[float] = Field(None, ge=0, le=100, description="Percentage rollout")
    user_ids: Optional[List[str]] = Field(None, description="Specific user IDs")
    variants: Optional[Dict[str, float]] = Field(None, description="A/B test variants")
    start_date: Optional[datetime] = Field(None, description="Schedule start")
    end_date: Optional[datetime] = Field(None, description="Schedule end")
    tags: Optional[List[str]] = Field(None, description="Flag tags")


class FlagUpdateRequest(BaseModel):
    """Request for updating a feature flag"""
    name: Optional[str] = Field(None, description="Human-readable name")
    description: Optional[str] = Field(None, description="What this flag controls")
    status: Optional[str] = Field(None, description="Flag status")
    enabled_for_all: Optional[bool] = Field(None, description="Enable for all users")
    percentage: Optional[float] = Field(None, ge=0, le=100, description="Percentage rollout")
    user_ids: Optional[List[str]] = Field(None, description="Specific user IDs")
    variants: Optional[Dict[str, float]] = Field(None, description="A/B test variants")
    default_value: Optional[Any] = Field(None, description="Default value")


@router.get("/health")
async def get_health():
    """Health check for feature flags service"""
    try:
        service = get_feature_flag_service()
        analytics = service.get_analytics()
        return {
            "status": "healthy",
            "service": "feature-flags",
            "total_flags": analytics["total_flags"],
            "active_flags": analytics["active_flags"]
        }
    except Exception as e:
        logger.error(f"Feature flags health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.post("/evaluate")
async def evaluate_flag(request: FlagEvaluationRequest):
    """
    Evaluate a feature flag for a specific user
    
    Returns the flag value based on targeting rules
    """
    try:
        service = get_feature_flag_service()
        
        # Check if it's a variant flag
        config = service.get_flag_config(request.flag_key)
        if config and config.get("flag_type") == FlagType.VARIANT.value:
            variant = service.get_variant(
                request.flag_key, 
                request.user_id,
                request.attributes
            )
            return {
                "flag_key": request.flag_key,
                "enabled": variant is not None,
                "variant": variant,
                "user_id": request.user_id
            }
        
        # Regular flag evaluation
        enabled = service.is_enabled(
            request.flag_key,
            request.user_id,
            request.attributes
        )
        
        return {
            "flag_key": request.flag_key,
            "enabled": enabled,
            "user_id": request.user_id
        }
        
    except Exception as e:
        logger.error(f"Error evaluating flag: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluate/{flag_key}")
async def evaluate_flag_simple(
    flag_key: str,
    user_id: Optional[str] = Query(None, description="User ID for targeting")
):
    """
    Simple GET endpoint for evaluating a flag
    
    Useful for client-side evaluation
    """
    try:
        service = get_feature_flag_service()
        
        # Check if it's a variant flag
        config = service.get_flag_config(flag_key)
        if config and config.get("flag_type") == FlagType.VARIANT.value:
            variant = service.get_variant(flag_key, user_id)
            return {
                "flag_key": flag_key,
                "enabled": variant is not None,
                "variant": variant,
                "user_id": user_id
            }
        
        enabled = service.is_enabled(flag_key, user_id)
        
        return {
            "flag_key": flag_key,
            "enabled": enabled,
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Error evaluating flag {flag_key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/all")
async def get_all_flags(
    user_id: Optional[str] = Query(None, description="Evaluate for specific user"),
    include_archived: bool = Query(False, description="Include archived flags")
):
    """
    Get all feature flags and their evaluated values
    
    Returns a dictionary of all flags with their current values
    """
    try:
        service = get_feature_flag_service()
        flags = await service.get_all_flags(user_id, include_archived)
        
        return {
            "flags": flags,
            "user_id": user_id,
            "count": len(flags)
        }
        
    except Exception as e:
        logger.error(f"Error getting all flags: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_flags(
    status: Optional[str] = Query(None, description="Filter by status"),
    tag: Optional[str] = Query(None, description="Filter by tag")
):
    """
    List all feature flag configurations
    
    Returns detailed configuration for all flags
    """
    try:
        service = get_feature_flag_service()
        all_flags = []
        
        for key in service._flags:
            config = service.get_flag_config(key)
            if config:
                # Apply filters
                if status and config.get("status") != status:
                    continue
                if tag and tag not in config.get("tags", []):
                    continue
                    
                all_flags.append(config)
        
        return {
            "flags": all_flags,
            "count": len(all_flags)
        }
        
    except Exception as e:
        logger.error(f"Error listing flags: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{flag_key}")
async def get_flag_config(flag_key: str):
    """
    Get detailed configuration for a specific flag
    """
    try:
        service = get_feature_flag_service()
        config = service.get_flag_config(flag_key)
        
        if not config:
            raise HTTPException(status_code=404, detail=f"Flag '{flag_key}' not found")
        
        return config
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting flag config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create")
async def create_flag(request: FlagCreateRequest):
    """
    Create a new feature flag
    """
    try:
        service = get_feature_flag_service()
        
        # Create flag object
        flag = FeatureFlag(
            key=request.key,
            name=request.name,
            description=request.description,
            flag_type=FlagType(request.flag_type),
            status=FlagStatus.DRAFT,
            default_value=request.default_value,
            enabled_for_all=request.enabled_for_all,
            percentage=request.percentage or 0.0,
            user_ids=request.user_ids or [],
            variants=request.variants or {},
            start_date=request.start_date,
            end_date=request.end_date,
            tags=request.tags or []
        )
        
        success = await service.create_flag(flag)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"Flag '{request.key}' already exists")
        
        return {
            "success": True,
            "flag_key": request.key,
            "message": f"Feature flag '{request.key}' created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating flag: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{flag_key}")
async def update_flag(flag_key: str, request: FlagUpdateRequest):
    """
    Update an existing feature flag
    """
    try:
        service = get_feature_flag_service()
        
        # Build updates dictionary
        updates = {}
        if request.name is not None:
            updates["name"] = request.name
        if request.description is not None:
            updates["description"] = request.description
        if request.status is not None:
            updates["status"] = FlagStatus(request.status)
        if request.enabled_for_all is not None:
            updates["enabled_for_all"] = request.enabled_for_all
        if request.percentage is not None:
            updates["percentage"] = request.percentage
        if request.user_ids is not None:
            updates["user_ids"] = request.user_ids
        if request.variants is not None:
            updates["variants"] = request.variants
        if request.default_value is not None:
            updates["default_value"] = request.default_value
        
        success = await service.update_flag(flag_key, updates)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Flag '{flag_key}' not found")
        
        return {
            "success": True,
            "flag_key": flag_key,
            "message": f"Feature flag '{flag_key}' updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating flag: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{flag_key}/activate")
async def activate_flag(flag_key: str):
    """
    Activate a feature flag
    """
    try:
        service = get_feature_flag_service()
        success = await service.update_flag(flag_key, {"status": FlagStatus.ACTIVE})
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Flag '{flag_key}' not found")
        
        return {
            "success": True,
            "flag_key": flag_key,
            "message": f"Feature flag '{flag_key}' activated"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating flag: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{flag_key}/deactivate")
async def deactivate_flag(flag_key: str):
    """
    Deactivate a feature flag
    """
    try:
        service = get_feature_flag_service()
        success = await service.update_flag(flag_key, {"status": FlagStatus.DRAFT})
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Flag '{flag_key}' not found")
        
        return {
            "success": True,
            "flag_key": flag_key,
            "message": f"Feature flag '{flag_key}' deactivated"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deactivating flag: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{flag_key}/emergency-disable")
async def emergency_disable_flag(flag_key: str):
    """
    Emergency disable a feature flag
    
    Use this when a feature is causing issues in production
    """
    try:
        service = get_feature_flag_service()
        success = await service.emergency_disable(flag_key)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Flag '{flag_key}' not found")
        
        logger.warning(f"Emergency disabled feature flag '{flag_key}'")
        
        return {
            "success": True,
            "flag_key": flag_key,
            "message": f"Feature flag '{flag_key}' emergency disabled",
            "warning": "This flag has been disabled due to emergency. Review and fix issues before re-enabling."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error emergency disabling flag: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{flag_key}")
async def delete_flag(flag_key: str):
    """
    Delete (archive) a feature flag
    
    Flags are archived rather than deleted to maintain history
    """
    try:
        service = get_feature_flag_service()
        success = await service.delete_flag(flag_key)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Flag '{flag_key}' not found")
        
        return {
            "success": True,
            "flag_key": flag_key,
            "message": f"Feature flag '{flag_key}' archived"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting flag: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/summary")
async def get_analytics():
    """
    Get analytics and usage statistics for feature flags
    """
    try:
        service = get_feature_flag_service()
        analytics = service.get_analytics()
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Export router for main app
__all__ = ["router"]