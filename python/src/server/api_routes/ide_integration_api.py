"""
IDE Integration API Routes
REST API endpoints for IDE plugin management and integration
"""

import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging

from ..services.ide_integration_service import IDEIntegrationService
from ...agents.ide_integration.plugin_base import PluginCommand, PluginCapability, PluginStatus
from ...agents.ide_integration.ide_plugin_manager import PluginPriority

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/ide", tags=["ide-integration"])

# Dependency to get IDE integration service
async def get_ide_service() -> IDEIntegrationService:
    """Get IDE integration service instance"""
    return IDEIntegrationService()

# Request/Response Models

class PluginInstallRequest(BaseModel):
    """Request model for plugin installation"""
    plugin_id: str = Field(..., description="Plugin identifier")
    version: str = Field(default="latest", description="Plugin version")
    configuration: Optional[Dict[str, Any]] = Field(default=None, description="Plugin configuration")
    auto_start: bool = Field(default=True, description="Auto-start after installation")

class PluginCommandRequest(BaseModel):
    """Request model for plugin command execution"""
    command_name: str = Field(..., description="Command name")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Command parameters")
    timeout_seconds: int = Field(default=30, description="Command timeout")
    requires_user_confirmation: bool = Field(default=False, description="Requires user confirmation")

class PluginConfigurationRequest(BaseModel):
    """Request model for plugin configuration update"""
    settings: Dict[str, Any] = Field(..., description="Configuration settings")
    enabled_capabilities: Optional[List[str]] = Field(default=None, description="Enabled capabilities")
    communication_endpoints: Optional[Dict[str, str]] = Field(default=None, description="Communication endpoints")

class CompletionRequest(BaseModel):
    """Request model for code completion"""
    file_path: str = Field(..., description="File path")
    language: str = Field(..., description="Programming language")
    line_number: int = Field(..., description="Line number")
    column_number: int = Field(..., description="Column number")
    current_line: str = Field(..., description="Current line content")
    preceding_lines: List[str] = Field(default_factory=list, description="Preceding lines")
    following_lines: List[str] = Field(default_factory=list, description="Following lines")
    prefix: str = Field(default="", description="Completion prefix")
    max_completions: int = Field(default=50, description="Maximum completions")
    include_snippets: bool = Field(default=True, description="Include code snippets")

class PluginEventRequest(BaseModel):
    """Request model for plugin events"""
    event_type: str = Field(..., description="Event type")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    source: str = Field(default="api", description="Event source")
    target_capabilities: Optional[List[str]] = Field(default=None, description="Target capabilities")

# Plugin Management Endpoints

@router.get("/plugins")
async def list_plugins(
    ide_service: IDEIntegrationService = Depends(get_ide_service),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    capability_filter: Optional[str] = Query(None, description="Filter by capability")
) -> JSONResponse:
    """List all installed plugins"""
    try:
        plugins = await ide_service.list_plugins(status_filter, capability_filter)
        return JSONResponse({
            "success": True,
            "plugins": plugins,
            "total_count": len(plugins)
        })
    except Exception as e:
        logger.error(f"Error listing plugins: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/plugins/{plugin_id}")
async def get_plugin_details(
    plugin_id: str = Path(..., description="Plugin ID"),
    ide_service: IDEIntegrationService = Depends(get_ide_service)
) -> JSONResponse:
    """Get detailed information about a specific plugin"""
    try:
        plugin_info = await ide_service.get_plugin_details(plugin_id)
        if not plugin_info:
            raise HTTPException(status_code=404, detail=f"Plugin {plugin_id} not found")
        
        return JSONResponse({
            "success": True,
            "plugin": plugin_info
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting plugin details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/plugins/install")
async def install_plugin(
    request: PluginInstallRequest,
    background_tasks: BackgroundTasks,
    ide_service: IDEIntegrationService = Depends(get_ide_service)
) -> JSONResponse:
    """Install a new plugin"""
    try:
        # Start installation in background
        background_tasks.add_task(
            ide_service.install_plugin,
            request.plugin_id,
            request.version,
            request.configuration,
            request.auto_start
        )
        
        return JSONResponse({
            "success": True,
            "message": f"Installation started for plugin {request.plugin_id}",
            "plugin_id": request.plugin_id
        })
    except Exception as e:
        logger.error(f"Error starting plugin installation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/plugins/{plugin_id}")
async def uninstall_plugin(
    plugin_id: str = Path(..., description="Plugin ID"),
    background_tasks: BackgroundTasks,
    ide_service: IDEIntegrationService = Depends(get_ide_service)
) -> JSONResponse:
    """Uninstall a plugin"""
    try:
        background_tasks.add_task(ide_service.uninstall_plugin, plugin_id)
        
        return JSONResponse({
            "success": True,
            "message": f"Uninstallation started for plugin {plugin_id}",
            "plugin_id": plugin_id
        })
    except Exception as e:
        logger.error(f"Error starting plugin uninstallation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/plugins/{plugin_id}/start")
async def start_plugin(
    plugin_id: str = Path(..., description="Plugin ID"),
    ide_service: IDEIntegrationService = Depends(get_ide_service)
) -> JSONResponse:
    """Start a plugin"""
    try:
        success = await ide_service.start_plugin(plugin_id)
        return JSONResponse({
            "success": success,
            "message": f"Plugin {plugin_id} {'started' if success else 'failed to start'}",
            "plugin_id": plugin_id
        })
    except Exception as e:
        logger.error(f"Error starting plugin: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/plugins/{plugin_id}/stop")
async def stop_plugin(
    plugin_id: str = Path(..., description="Plugin ID"),
    ide_service: IDEIntegrationService = Depends(get_ide_service)
) -> JSONResponse:
    """Stop a plugin"""
    try:
        success = await ide_service.stop_plugin(plugin_id)
        return JSONResponse({
            "success": success,
            "message": f"Plugin {plugin_id} {'stopped' if success else 'failed to stop'}",
            "plugin_id": plugin_id
        })
    except Exception as e:
        logger.error(f"Error stopping plugin: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/plugins/{plugin_id}/configuration")
async def update_plugin_configuration(
    plugin_id: str = Path(..., description="Plugin ID"),
    request: PluginConfigurationRequest,
    ide_service: IDEIntegrationService = Depends(get_ide_service)
) -> JSONResponse:
    """Update plugin configuration"""
    try:
        success = await ide_service.update_plugin_configuration(
            plugin_id,
            request.settings,
            request.enabled_capabilities,
            request.communication_endpoints
        )
        
        return JSONResponse({
            "success": success,
            "message": f"Configuration {'updated' if success else 'failed to update'} for plugin {plugin_id}",
            "plugin_id": plugin_id
        })
    except Exception as e:
        logger.error(f"Error updating plugin configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Command Execution Endpoints

@router.post("/plugins/{plugin_id}/commands")
async def execute_plugin_command(
    plugin_id: str = Path(..., description="Plugin ID"),
    request: PluginCommandRequest,
    ide_service: IDEIntegrationService = Depends(get_ide_service)
) -> JSONResponse:
    """Execute a command on a specific plugin"""
    try:
        response = await ide_service.execute_plugin_command(
            plugin_id,
            request.command_name,
            request.parameters,
            request.timeout_seconds
        )
        
        return JSONResponse({
            "success": response.success,
            "data": response.data,
            "error_message": response.error_message,
            "error_code": response.error_code,
            "execution_time_ms": response.execution_time_ms,
            "response_id": response.response_id
        })
    except Exception as e:
        logger.error(f"Error executing plugin command: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/commands/broadcast")
async def broadcast_command(
    request: PluginCommandRequest,
    target_capabilities: Optional[List[str]] = Query(None, description="Target capabilities"),
    ide_service: IDEIntegrationService = Depends(get_ide_service)
) -> JSONResponse:
    """Broadcast a command to all plugins with specified capabilities"""
    try:
        capabilities = []
        if target_capabilities:
            for cap in target_capabilities:
                try:
                    capabilities.append(PluginCapability(cap))
                except ValueError:
                    logger.warning(f"Invalid capability: {cap}")
        
        results = await ide_service.broadcast_command(
            request.command_name,
            request.parameters,
            capabilities if capabilities else None
        )
        
        return JSONResponse({
            "success": True,
            "results": results,
            "total_plugins": len(results)
        })
    except Exception as e:
        logger.error(f"Error broadcasting command: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Code Completion Endpoints

@router.post("/completion")
async def get_code_completion(
    request: CompletionRequest,
    ide_service: IDEIntegrationService = Depends(get_ide_service)
) -> JSONResponse:
    """Get code completion suggestions"""
    try:
        completions = await ide_service.get_code_completion(
            file_path=request.file_path,
            language=request.language,
            line_number=request.line_number,
            column_number=request.column_number,
            current_line=request.current_line,
            preceding_lines=request.preceding_lines,
            following_lines=request.following_lines,
            prefix=request.prefix,
            max_completions=request.max_completions,
            include_snippets=request.include_snippets
        )
        
        return JSONResponse({
            "success": True,
            "completions": [
                {
                    "label": item.label,
                    "kind": item.kind.value if hasattr(item.kind, 'value') else str(item.kind),
                    "detail": item.detail,
                    "documentation": item.documentation,
                    "insert_text": item.insert_text,
                    "priority": item.priority,
                    "confidence": item.confidence,
                    "source": item.source
                }
                for item in completions.items
            ],
            "processing_time_ms": completions.processing_time_ms,
            "total_candidates": completions.total_candidates
        })
    except Exception as e:
        logger.error(f"Error getting code completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Event Management Endpoints

@router.post("/events/broadcast")
async def broadcast_event(
    request: PluginEventRequest,
    ide_service: IDEIntegrationService = Depends(get_ide_service)
) -> JSONResponse:
    """Broadcast an event to plugins"""
    try:
        capabilities = []
        if request.target_capabilities:
            for cap in request.target_capabilities:
                try:
                    capabilities.append(PluginCapability(cap))
                except ValueError:
                    logger.warning(f"Invalid capability: {cap}")
        
        results = await ide_service.broadcast_event(
            request.event_type,
            request.data,
            request.source,
            capabilities if capabilities else None
        )
        
        return JSONResponse({
            "success": True,
            "results": results,
            "total_plugins": len(results)
        })
    except Exception as e:
        logger.error(f"Error broadcasting event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Status and Monitoring Endpoints

@router.get("/status")
async def get_system_status(
    ide_service: IDEIntegrationService = Depends(get_ide_service)
) -> JSONResponse:
    """Get overall IDE integration system status"""
    try:
        status = await ide_service.get_system_status()
        return JSONResponse({
            "success": True,
            "status": status
        })
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/plugins/{plugin_id}/status")
async def get_plugin_status(
    plugin_id: str = Path(..., description="Plugin ID"),
    ide_service: IDEIntegrationService = Depends(get_ide_service)
) -> JSONResponse:
    """Get status of a specific plugin"""
    try:
        status = await ide_service.get_plugin_status(plugin_id)
        if not status:
            raise HTTPException(status_code=404, detail=f"Plugin {plugin_id} not found")
        
        return JSONResponse({
            "success": True,
            "plugin_id": plugin_id,
            "status": status
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting plugin status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_health_check(
    ide_service: IDEIntegrationService = Depends(get_ide_service)
) -> JSONResponse:
    """Get health status of all plugins"""
    try:
        health_info = await ide_service.get_health_check()
        return JSONResponse({
            "success": True,
            "health": health_info,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics and Metrics Endpoints

@router.get("/analytics")
async def get_analytics(
    ide_service: IDEIntegrationService = Depends(get_ide_service),
    plugin_id: Optional[str] = Query(None, description="Filter by plugin ID"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)")
) -> JSONResponse:
    """Get analytics and usage metrics"""
    try:
        analytics = await ide_service.get_analytics(plugin_id, start_date, end_date)
        return JSONResponse({
            "success": True,
            "analytics": analytics
        })
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_metrics(
    ide_service: IDEIntegrationService = Depends(get_ide_service)
) -> JSONResponse:
    """Get system metrics"""
    try:
        metrics = await ide_service.get_metrics()
        return JSONResponse({
            "success": True,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Communication Management Endpoints

@router.get("/communication/status")
async def get_communication_status(
    ide_service: IDEIntegrationService = Depends(get_ide_service)
) -> JSONResponse:
    """Get communication channels status"""
    try:
        comm_status = await ide_service.get_communication_status()
        return JSONResponse({
            "success": True,
            "communication": comm_status
        })
    except Exception as e:
        logger.error(f"Error getting communication status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/communication/{plugin_id}/connect")
async def connect_to_plugin(
    plugin_id: str = Path(..., description="Plugin ID"),
    channel_name: str = Body(..., description="Communication channel name"),
    config: Dict[str, Any] = Body(default_factory=dict, description="Connection config"),
    ide_service: IDEIntegrationService = Depends(get_ide_service)
) -> JSONResponse:
    """Connect to a plugin via specified communication channel"""
    try:
        success = await ide_service.connect_to_plugin(plugin_id, channel_name, config)
        return JSONResponse({
            "success": success,
            "message": f"Connection {'established' if success else 'failed'} with plugin {plugin_id}",
            "plugin_id": plugin_id,
            "channel": channel_name
        })
    except Exception as e:
        logger.error(f"Error connecting to plugin: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/communication/{plugin_id}/disconnect")
async def disconnect_from_plugin(
    plugin_id: str = Path(..., description="Plugin ID"),
    ide_service: IDEIntegrationService = Depends(get_ide_service)
) -> JSONResponse:
    """Disconnect from a plugin"""
    try:
        success = await ide_service.disconnect_from_plugin(plugin_id)
        return JSONResponse({
            "success": success,
            "message": f"Disconnection {'successful' if success else 'failed'} for plugin {plugin_id}",
            "plugin_id": plugin_id
        })
    except Exception as e:
        logger.error(f"Error disconnecting from plugin: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Development and Debug Endpoints

@router.get("/debug/registry")
async def get_plugin_registry(
    ide_service: IDEIntegrationService = Depends(get_ide_service)
) -> JSONResponse:
    """Get plugin registry information for debugging"""
    try:
        registry_info = await ide_service.get_registry_debug_info()
        return JSONResponse({
            "success": True,
            "registry": registry_info
        })
    except Exception as e:
        logger.error(f"Error getting registry info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/debug/test-command")
async def test_command(
    plugin_id: str = Body(..., description="Plugin ID"),
    command_name: str = Body(..., description="Command name"),
    parameters: Dict[str, Any] = Body(default_factory=dict, description="Command parameters"),
    ide_service: IDEIntegrationService = Depends(get_ide_service)
) -> JSONResponse:
    """Test command execution for debugging"""
    try:
        response = await ide_service.test_command(plugin_id, command_name, parameters)
        return JSONResponse({
            "success": True,
            "test_result": response,
            "plugin_id": plugin_id,
            "command": command_name
        })
    except Exception as e:
        logger.error(f"Error testing command: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch Operations Endpoints

@router.post("/batch/start-plugins")
async def batch_start_plugins(
    plugin_ids: List[str] = Body(..., description="List of plugin IDs to start"),
    background_tasks: BackgroundTasks,
    ide_service: IDEIntegrationService = Depends(get_ide_service)
) -> JSONResponse:
    """Start multiple plugins in batch"""
    try:
        background_tasks.add_task(ide_service.batch_start_plugins, plugin_ids)
        return JSONResponse({
            "success": True,
            "message": f"Batch start initiated for {len(plugin_ids)} plugins",
            "plugin_ids": plugin_ids
        })
    except Exception as e:
        logger.error(f"Error starting batch plugins: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/stop-plugins")
async def batch_stop_plugins(
    plugin_ids: List[str] = Body(..., description="List of plugin IDs to stop"),
    background_tasks: BackgroundTasks,
    ide_service: IDEIntegrationService = Depends(get_ide_service)
) -> JSONResponse:
    """Stop multiple plugins in batch"""
    try:
        background_tasks.add_task(ide_service.batch_stop_plugins, plugin_ids)
        return JSONResponse({
            "success": True,
            "message": f"Batch stop initiated for {len(plugin_ids)} plugins",
            "plugin_ids": plugin_ids
        })
    except Exception as e:
        logger.error(f"Error stopping batch plugins: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Configuration Management Endpoints

@router.get("/configuration/defaults")
async def get_default_configuration(
    plugin_type: Optional[str] = Query(None, description="Plugin type"),
    ide_service: IDEIntegrationService = Depends(get_ide_service)
) -> JSONResponse:
    """Get default configuration for plugin type"""
    try:
        config = await ide_service.get_default_configuration(plugin_type)
        return JSONResponse({
            "success": True,
            "configuration": config,
            "plugin_type": plugin_type
        })
    except Exception as e:
        logger.error(f"Error getting default configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/configuration/validate")
async def validate_configuration(
    plugin_id: str = Body(..., description="Plugin ID"),
    configuration: Dict[str, Any] = Body(..., description="Configuration to validate"),
    ide_service: IDEIntegrationService = Depends(get_ide_service)
) -> JSONResponse:
    """Validate plugin configuration"""
    try:
        validation_result = await ide_service.validate_configuration(plugin_id, configuration)
        return JSONResponse({
            "success": True,
            "validation": validation_result,
            "plugin_id": plugin_id
        })
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))