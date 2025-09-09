"""
Template Management API

Provides comprehensive template management endpoints including:
- Template CRUD operations
- Template marketplace functionality
- Project generation from templates
- Template validation and search
"""

import asyncio
import uuid
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..services.client_manager import get_supabase_client
from ...agents.templates import (
    TemplateRegistry, TemplateEngine, TemplateValidator,
    Template, TemplateSearchRequest, TemplateGenerationRequest,
    TemplateType, TemplateCategory
)
import logging

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/templates", tags=["templates"])

# Initialize template services
template_registry = TemplateRegistry()
template_engine = TemplateEngine()
template_validator = TemplateValidator()

# Track active generation tasks
active_generations: dict[str, asyncio.Task] = {}


# API Models
class TemplateCreateRequest(BaseModel):
    template: Template


class TemplateUpdateRequest(BaseModel):
    metadata: Optional[dict] = None
    rating: Optional[float] = None


class GenerationProgressUpdate(BaseModel):
    generation_id: str
    status: str
    percentage: int
    message: str
    files_created: List[str] = []
    errors: List[str] = []


# Template CRUD Operations

@router.post("/", response_model=dict)
async def create_template(request: TemplateCreateRequest):
    """Create a new template in the registry."""
    try:
        logger.info(f"Creating template | id={request.template.id} | name={request.template.metadata.name}")
        
        # Check if template already exists
        existing = template_registry.get_template(request.template.id)
        if existing:
            raise HTTPException(
                status_code=409, 
                detail=f"Template with ID '{request.template.id}' already exists"
            )
        
        # Register template
        success = template_registry.register_template(request.template)
        if not success:
            raise HTTPException(
                status_code=400, 
                detail="Failed to register template (validation errors)"
            )
        
        logger.info(f"Template created successfully | id={request.template.id}")
        
        return {
            "success": True,
            "message": f"Template '{request.template.id}' created successfully",
            "template_id": request.template.id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create template | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Template creation failed: {str(e)}")


@router.get("/{template_id}", response_model=Template)
async def get_template(template_id: str):
    """Get a specific template by ID."""
    try:
        template = template_registry.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")
        
        return template
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get template | id={template_id} | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve template: {str(e)}")


@router.put("/{template_id}", response_model=dict)
async def update_template(template_id: str, request: TemplateUpdateRequest):
    """Update template metadata."""
    try:
        template = template_registry.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")
        
        # Update rating if provided
        if request.rating is not None:
            template_registry.update_template_rating(template_id, request.rating)
        
        # TODO: Implement metadata updates when needed
        if request.metadata:
            logger.info(f"Metadata update requested for {template_id} (not yet implemented)")
        
        return {
            "success": True,
            "message": f"Template '{template_id}' updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update template | id={template_id} | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Template update failed: {str(e)}")


@router.delete("/{template_id}", response_model=dict)
async def delete_template(template_id: str):
    """Delete a template from the registry."""
    try:
        # Check if template exists
        template = template_registry.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")
        
        # Delete template
        success = template_registry.delete_template(template_id)
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to delete template '{template_id}'")
        
        logger.info(f"Template deleted | id={template_id}")
        
        return {
            "success": True,
            "message": f"Template '{template_id}' deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete template | id={template_id} | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Template deletion failed: {str(e)}")


# Template Discovery and Search

@router.post("/search", response_model=dict)
async def search_templates(request: TemplateSearchRequest):
    """Search templates based on criteria."""
    try:
        results = template_registry.search_templates(request)
        
        return {
            "success": True,
            "results": results["templates"],
            "pagination": {
                "total": results["total"],
                "offset": results["offset"], 
                "limit": results["limit"],
                "has_more": results["offset"] + results["limit"] < results["total"]
            }
        }
        
    except Exception as e:
        logger.error(f"Template search failed | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/", response_model=dict)
async def list_templates(
    category: Optional[TemplateCategory] = Query(None),
    type_: Optional[TemplateType] = Query(None, alias="type"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """List templates with optional filtering."""
    try:
        templates = template_registry.list_templates(category=category, type_=type_)
        
        # Apply pagination
        total = len(templates)
        paginated = templates[offset:offset + limit]
        
        return {
            "success": True,
            "templates": paginated,
            "pagination": {
                "total": total,
                "offset": offset,
                "limit": limit,
                "has_more": offset + limit < total
            }
        }
        
    except Exception as e:
        logger.error(f"Template listing failed | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Listing failed: {str(e)}")


@router.get("/categories/list", response_model=List[str])
async def list_categories():
    """List all available template categories."""
    return [category.value for category in TemplateCategory]


@router.get("/types/list", response_model=List[str])
async def list_types():
    """List all available template types."""
    return [type_.value for type_ in TemplateType]


@router.get("/stats", response_model=dict)
async def get_registry_stats():
    """Get template registry statistics."""
    try:
        stats = template_registry.get_registry_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Failed to get registry stats | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")


# Template Validation

@router.post("/{template_id}/validate", response_model=dict)
async def validate_template(template_id: str):
    """Validate a template."""
    try:
        template = template_registry.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")
        
        validation_result = template_validator.validate_template(template)
        
        return {
            "success": True,
            "validation": validation_result.dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template validation failed | id={template_id} | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


# Project Generation

@router.post("/generate", response_model=dict)
async def generate_project(request: TemplateGenerationRequest, background_tasks: BackgroundTasks):
    """Generate a project from a template."""
    try:
        # Validate template exists
        template = template_registry.get_template(request.template_id)
        if not template:
            raise HTTPException(status_code=404, detail=f"Template '{request.template_id}' not found")
        
        # Generate unique generation ID
        generation_id = str(uuid.uuid4())
        
        logger.info(f"Starting project generation | template={request.template_id} | generation_id={generation_id}")
        
        # Start background generation task
        task = asyncio.create_task(
            _perform_project_generation(generation_id, template, request)
        )
        active_generations[generation_id] = task
        
        # Increment download count
        template_registry.increment_download_count(request.template_id)
        
        return {
            "success": True,
            "generation_id": generation_id,
            "message": f"Project generation started from template '{request.template_id}'",
            "output_directory": request.output_directory,
            "estimated_time": "30-120 seconds"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start project generation | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed to start: {str(e)}")


@router.get("/generation/{generation_id}/status", response_model=dict)
async def get_generation_status(generation_id: str):
    """Get the status of a project generation."""
    try:
        if generation_id not in active_generations:
            raise HTTPException(status_code=404, detail=f"Generation '{generation_id}' not found")
        
        task = active_generations[generation_id]
        
        if task.done():
            # Task completed, get result
            try:
                result = task.result()
                # Clean up completed task
                del active_generations[generation_id]
                
                return {
                    "success": True,
                    "status": "completed" if result.success else "failed",
                    "result": result.dict()
                }
            except Exception as e:
                del active_generations[generation_id]
                return {
                    "success": False,
                    "status": "failed",
                    "error": str(e)
                }
        else:
            return {
                "success": True,
                "status": "running",
                "message": "Generation in progress..."
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get generation status | id={generation_id} | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.delete("/generation/{generation_id}/cancel", response_model=dict)
async def cancel_generation(generation_id: str):
    """Cancel an active project generation."""
    try:
        if generation_id not in active_generations:
            raise HTTPException(status_code=404, detail=f"Generation '{generation_id}' not found")
        
        task = active_generations[generation_id]
        if not task.done():
            task.cancel()
        
        del active_generations[generation_id]
        
        logger.info(f"Generation cancelled | id={generation_id}")
        
        return {
            "success": True,
            "message": f"Generation '{generation_id}' cancelled"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel generation | id={generation_id} | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Cancellation failed: {str(e)}")


# Template Import/Export

@router.post("/import", response_model=dict)
async def import_template(directory_path: str):
    """Import a template from a directory."""
    try:
        template_id = template_registry.import_template_from_directory(directory_path)
        if not template_id:
            raise HTTPException(status_code=400, detail=f"Failed to import template from '{directory_path}'")
        
        logger.info(f"Template imported | id={template_id} | from={directory_path}")
        
        return {
            "success": True,
            "template_id": template_id,
            "message": f"Template imported successfully from '{directory_path}'"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template import failed | path={directory_path} | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


@router.post("/{template_id}/export", response_model=dict)
async def export_template(template_id: str, export_path: str):
    """Export a template to a directory or archive."""
    try:
        # Check template exists
        template = template_registry.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")
        
        success = template_registry.export_template(template_id, export_path)
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to export template '{template_id}'")
        
        return {
            "success": True,
            "message": f"Template '{template_id}' exported to '{export_path}'"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template export failed | id={template_id} | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# Helper Functions

async def _perform_project_generation(generation_id: str, template: Template, request: TemplateGenerationRequest):
    """Perform project generation in background."""
    try:
        # Progress callback for generation updates
        async def progress_callback(message: str, percentage: int):
            # In a real implementation, this would emit to WebSocket or store in database
            logger.info(f"Generation {generation_id}: {percentage}% - {message}")
        
        # Generate project
        result = await template_engine.generate_project(template, request, progress_callback)
        
        logger.info(f"Project generation completed | id={generation_id} | success={result.success}")
        
        return result
        
    except Exception as e:
        logger.error(f"Project generation failed | id={generation_id} | error={str(e)}")
        raise


# Health Check

@router.get("/health", response_model=dict)
async def template_health():
    """Template API health check."""
    try:
        stats = template_registry.get_registry_stats()
        return {
            "status": "healthy",
            "service": "template-api",
            "templates_count": stats["total_templates"],
            "registry_path": str(template_registry.registry_path)
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "service": "template-api",
            "error": str(e)
        }