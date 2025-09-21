"""
IDE Integration Service
Business logic service for IDE plugin management and integration
"""

import asyncio
from typing import Dict, Any, List, Optional, Union, Type
from datetime import datetime, timedelta
from dataclasses import asdict
import logging
import json

from ...agents.ide_integration.ide_plugin_manager import IDEPluginManager, PluginPriority
from ...agents.ide_integration.plugin_base import (
    PluginBase, PluginConfiguration, PluginCapability, PluginCommand, PluginResponse,
    PluginEvent, PluginStatus
)
from ...agents.ide_integration.code_completion_engine import (
    CodeCompletionEngine, CompletionRequest, CodeContext, CompletionTrigger,
    CompletionResponse
)
from ...agents.ide_integration.ide_communication import IDECommunicationProtocol, ConnectionConfig

logger = logging.getLogger(__name__)


class IDEIntegrationService:
    """Service for managing IDE plugin integration"""
    
    def __init__(self):
        self.plugin_manager = IDEPluginManager()
        self.completion_engine = CodeCompletionEngine()
        self.communication = IDECommunicationProtocol()
        
        # Service state
        self._initialized = False
        self._background_tasks = set()
        
        # Initialize service
        asyncio.create_task(self._initialize())
    
    async def _initialize(self) -> None:
        """Initialize the IDE integration service"""
        try:
            logger.info("Initializing IDE Integration Service...")
            
            # Start plugin manager background tasks
            await self.plugin_manager.start_health_monitoring()
            await self.plugin_manager.start_analytics_collection()
            
            self._initialized = True
            logger.info("IDE Integration Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize IDE Integration Service: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the service"""
        try:
            logger.info("Shutting down IDE Integration Service...")
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown plugin manager
            await self.plugin_manager.shutdown()
            
            logger.info("IDE Integration Service shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down IDE Integration Service: {e}")
    
    # Plugin Management Methods
    
    async def list_plugins(self, status_filter: Optional[str] = None, 
                          capability_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all plugins with optional filters"""
        try:
            all_statuses = await self.plugin_manager.get_all_plugin_statuses()
            plugins = []
            
            for plugin_id, status in all_statuses.get("plugins", {}).items():
                plugin_info = {
                    "plugin_id": plugin_id,
                    "name": status.get("metadata", {}).get("name", plugin_id),
                    "version": status.get("metadata", {}).get("version", "unknown"),
                    "description": status.get("metadata", {}).get("description", ""),
                    "status": status.get("status", "unknown"),
                    "capabilities": status.get("metadata", {}).get("capabilities", []),
                    "supported_languages": status.get("metadata", {}).get("supported_languages", []),
                    "last_activity": status.get("health", {}).get("last_health_check"),
                    "error_count": status.get("health", {}).get("error_count", 0),
                    "success_rate": status.get("analytics", {}).get("success_rate", 0)
                }
                
                # Apply filters
                if status_filter and status.get("status") != status_filter:
                    continue
                
                if capability_filter:
                    capabilities = status.get("metadata", {}).get("capabilities", [])
                    if capability_filter not in capabilities:
                        continue
                
                plugins.append(plugin_info)
            
            return plugins
            
        except Exception as e:
            logger.error(f"Error listing plugins: {e}")
            raise
    
    async def get_plugin_details(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific plugin"""
        try:
            status = await self.plugin_manager.get_plugin_status(plugin_id)
            if not status:
                return None
            
            # Get plugin instance for additional details
            plugin = self.plugin_manager.plugins.get(plugin_id)
            
            details = {
                "plugin_id": plugin_id,
                "metadata": status.get("metadata", {}),
                "status": status,
                "installation": None,
                "configuration": None,
                "communication_channels": [],
                "recent_activity": []
            }
            
            # Add installation details
            if plugin_id in self.plugin_manager.installations:
                installation = self.plugin_manager.installations[plugin_id]
                details["installation"] = {
                    "version": installation.version,
                    "install_path": installation.install_path,
                    "installed_at": installation.installed_at.isoformat(),
                    "last_updated": installation.last_updated.isoformat(),
                    "auto_update": installation.auto_update,
                    "dependencies": installation.dependencies
                }
            
            # Add configuration details
            if plugin and plugin.configuration:
                details["configuration"] = {
                    "settings": plugin.configuration.settings,
                    "enabled_capabilities": [cap.value for cap in plugin.configuration.enabled_capabilities],
                    "communication_endpoints": plugin.configuration.communication_endpoints,
                    "auto_update": plugin.configuration.auto_update,
                    "logging_level": plugin.configuration.logging_level
                }
            
            return details
            
        except Exception as e:
            logger.error(f"Error getting plugin details: {e}")
            raise
    
    async def install_plugin(self, plugin_id: str, version: str = "latest",
                           configuration: Optional[Dict[str, Any]] = None,
                           auto_start: bool = True) -> bool:
        """Install a new plugin"""
        try:
            logger.info(f"Installing plugin: {plugin_id} (version: {version})")
            
            # Create plugin configuration
            config = None
            if configuration:
                config = PluginConfiguration(
                    plugin_id=plugin_id,
                    settings=configuration.get("settings", {}),
                    enabled_capabilities=[
                        PluginCapability(cap) for cap in configuration.get("enabled_capabilities", [])
                    ],
                    communication_endpoints=configuration.get("communication_endpoints", {}),
                    auto_update=configuration.get("auto_update", True),
                    logging_level=configuration.get("logging_level", "INFO")
                )
            
            # Install plugin
            success = await self.plugin_manager.install_plugin(plugin_id, version, config)
            
            if success and auto_start:
                # Start plugin after installation
                start_success = await self.plugin_manager.start_plugin(plugin_id)
                if not start_success:
                    logger.warning(f"Plugin {plugin_id} installed but failed to start")
            
            return success
            
        except Exception as e:
            logger.error(f"Error installing plugin {plugin_id}: {e}")
            raise
    
    async def uninstall_plugin(self, plugin_id: str) -> bool:
        """Uninstall a plugin"""
        try:
            logger.info(f"Uninstalling plugin: {plugin_id}")
            return await self.plugin_manager.uninstall_plugin(plugin_id)
        except Exception as e:
            logger.error(f"Error uninstalling plugin {plugin_id}: {e}")
            raise
    
    async def start_plugin(self, plugin_id: str) -> bool:
        """Start a specific plugin"""
        try:
            return await self.plugin_manager.start_plugin(plugin_id)
        except Exception as e:
            logger.error(f"Error starting plugin {plugin_id}: {e}")
            raise
    
    async def stop_plugin(self, plugin_id: str) -> bool:
        """Stop a specific plugin"""
        try:
            return await self.plugin_manager.stop_plugin(plugin_id)
        except Exception as e:
            logger.error(f"Error stopping plugin {plugin_id}: {e}")
            raise
    
    async def update_plugin_configuration(self, plugin_id: str, settings: Dict[str, Any],
                                        enabled_capabilities: Optional[List[str]] = None,
                                        communication_endpoints: Optional[Dict[str, str]] = None) -> bool:
        """Update plugin configuration"""
        try:
            config_updates = {"settings": settings}
            
            if enabled_capabilities is not None:
                config_updates["enabled_capabilities"] = enabled_capabilities
            
            if communication_endpoints is not None:
                config_updates["communication_endpoints"] = communication_endpoints
            
            return await self.plugin_manager.update_plugin_configuration(plugin_id, config_updates)
            
        except Exception as e:
            logger.error(f"Error updating plugin configuration: {e}")
            raise
    
    # Command Execution Methods
    
    async def execute_plugin_command(self, plugin_id: str, command_name: str,
                                   parameters: Dict[str, Any], 
                                   timeout_seconds: int = 30) -> PluginResponse:
        """Execute a command on a specific plugin"""
        try:
            command = PluginCommand(
                command_id=f"cmd_{datetime.now().timestamp()}",
                command_name=command_name,
                parameters=parameters,
                timeout_seconds=timeout_seconds
            )
            
            return await self.plugin_manager.execute_plugin_command(plugin_id, command)
            
        except Exception as e:
            logger.error(f"Error executing command {command_name} on plugin {plugin_id}: {e}")
            raise
    
    async def broadcast_command(self, command_name: str, parameters: Dict[str, Any],
                              target_capabilities: Optional[List[PluginCapability]] = None) -> Dict[str, Any]:
        """Broadcast a command to plugins with specified capabilities"""
        try:
            command = PluginCommand(
                command_id=f"broadcast_{datetime.now().timestamp()}",
                command_name=command_name,
                parameters=parameters
            )
            
            results = {}
            plugins = self.plugin_manager.plugins.values()
            
            # Filter by capabilities if specified
            if target_capabilities:
                plugins = [p for p in plugins 
                          if any(p.supports_capability(cap) for cap in target_capabilities)]
            
            # Execute command on each plugin
            for plugin in plugins:
                try:
                    response = await plugin.execute_command(command)
                    results[plugin.plugin_id] = {
                        "success": response.success,
                        "data": response.data,
                        "error_message": response.error_message,
                        "execution_time_ms": response.execution_time_ms
                    }
                except Exception as e:
                    results[plugin.plugin_id] = {
                        "success": False,
                        "error_message": str(e),
                        "execution_time_ms": 0
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Error broadcasting command {command_name}: {e}")
            raise
    
    # Code Completion Methods
    
    async def get_code_completion(self, file_path: str, language: str, line_number: int,
                                column_number: int, current_line: str, 
                                preceding_lines: List[str], following_lines: List[str],
                                prefix: str = "", max_completions: int = 50,
                                include_snippets: bool = True) -> CompletionResponse:
        """Get code completion suggestions"""
        try:
            # Create completion context
            code_context = CodeContext(
                file_path=file_path,
                language=language,
                line_number=line_number,
                column_number=column_number,
                current_line=current_line,
                preceding_lines=preceding_lines,
                following_lines=following_lines,
                prefix=prefix,
                suffix=current_line[column_number:] if column_number < len(current_line) else ""
            )
            
            # Create completion request
            completion_request = CompletionRequest(
                context=code_context,
                trigger=CompletionTrigger.EXPLICIT_INVOCATION,
                max_completions=max_completions,
                include_snippets=include_snippets
            )
            
            # Get completions
            return await self.completion_engine.get_completions(completion_request)
            
        except Exception as e:
            logger.error(f"Error getting code completion: {e}")
            raise
    
    # Event Management Methods
    
    async def broadcast_event(self, event_type: str, data: Dict[str, Any], source: str = "service",
                            target_capabilities: Optional[List[PluginCapability]] = None) -> Dict[str, bool]:
        """Broadcast an event to plugins"""
        try:
            event = PluginEvent(
                plugin_id="service",
                event_type=event_type,
                data=data,
                source=source
            )
            
            return await self.plugin_manager.broadcast_event(event, target_capabilities)
            
        except Exception as e:
            logger.error(f"Error broadcasting event {event_type}: {e}")
            raise
    
    # Status and Monitoring Methods
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        try:
            plugin_statuses = await self.plugin_manager.get_all_plugin_statuses()
            
            system_status = {
                "service_initialized": self._initialized,
                "plugin_manager": plugin_statuses,
                "completion_engine": self.completion_engine.get_engine_stats(),
                "communication": self.communication.get_connection_status(),
                "background_tasks": len(self._background_tasks),
                "timestamp": datetime.now().isoformat()
            }
            
            return system_status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            raise
    
    async def get_plugin_status(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific plugin"""
        try:
            return await self.plugin_manager.get_plugin_status(plugin_id)
        except Exception as e:
            logger.error(f"Error getting plugin status: {e}")
            raise
    
    async def get_health_check(self) -> Dict[str, Any]:
        """Get health status of all plugins"""
        try:
            health_info = {
                "overall_health": "healthy",
                "plugins": {},
                "issues": [],
                "recommendations": []
            }
            
            # Get all plugin statuses
            all_statuses = await self.plugin_manager.get_all_plugin_statuses()
            
            total_plugins = 0
            healthy_plugins = 0
            
            for plugin_id, status in all_statuses.get("plugins", {}).items():
                plugin_health = {
                    "status": status.get("status", "unknown"),
                    "error_count": status.get("health", {}).get("error_count", 0),
                    "success_rate": status.get("analytics", {}).get("success_rate", 0),
                    "last_check": status.get("health", {}).get("last_health_check"),
                    "response_time": status.get("health", {}).get("response_time_ms", 0)
                }
                
                health_info["plugins"][plugin_id] = plugin_health
                total_plugins += 1
                
                # Determine if plugin is healthy
                if (status.get("status") == "active" and 
                    status.get("health", {}).get("error_count", 0) < 5 and
                    status.get("analytics", {}).get("success_rate", 0) > 0.8):
                    healthy_plugins += 1
                else:
                    # Add to issues
                    health_info["issues"].append({
                        "plugin_id": plugin_id,
                        "issue": "Plugin experiencing issues",
                        "details": {
                            "status": status.get("status"),
                            "error_count": status.get("health", {}).get("error_count"),
                            "success_rate": status.get("analytics", {}).get("success_rate")
                        }
                    })
            
            # Calculate overall health
            if total_plugins == 0:
                health_info["overall_health"] = "no_plugins"
            elif healthy_plugins / total_plugins >= 0.8:
                health_info["overall_health"] = "healthy"
            elif healthy_plugins / total_plugins >= 0.5:
                health_info["overall_health"] = "degraded"
            else:
                health_info["overall_health"] = "unhealthy"
            
            # Add recommendations
            if len(health_info["issues"]) > 0:
                health_info["recommendations"].append("Review and restart problematic plugins")
            
            if total_plugins == 0:
                health_info["recommendations"].append("Install and configure IDE plugins")
            
            return health_info
            
        except Exception as e:
            logger.error(f"Error getting health check: {e}")
            raise
    
    # Analytics and Metrics Methods
    
    async def get_analytics(self, plugin_id: Optional[str] = None, 
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> Dict[str, Any]:
        """Get analytics and usage metrics"""
        try:
            analytics_data = {
                "period": {
                    "start": start_date or (datetime.now() - timedelta(days=7)).isoformat(),
                    "end": end_date or datetime.now().isoformat()
                },
                "plugins": {},
                "summary": {}
            }
            
            # Get analytics for specific plugin or all plugins
            if plugin_id:
                if plugin_id in self.plugin_manager.analytics:
                    analytics = self.plugin_manager.analytics[plugin_id]
                    analytics_data["plugins"][plugin_id] = {
                        "total_commands": analytics.total_commands,
                        "successful_commands": analytics.successful_commands,
                        "failed_commands": analytics.failed_commands,
                        "success_rate": (analytics.successful_commands / max(analytics.total_commands, 1)) * 100,
                        "average_response_time": analytics.average_response_time,
                        "most_used_commands": analytics.most_used_commands,
                        "peak_usage_hours": analytics.peak_usage_hours,
                        "user_satisfaction_score": analytics.user_satisfaction_score,
                        "last_updated": analytics.last_analytics_update.isoformat()
                    }
            else:
                # Get analytics for all plugins
                for pid, analytics in self.plugin_manager.analytics.items():
                    analytics_data["plugins"][pid] = {
                        "total_commands": analytics.total_commands,
                        "successful_commands": analytics.successful_commands,
                        "failed_commands": analytics.failed_commands,
                        "success_rate": (analytics.successful_commands / max(analytics.total_commands, 1)) * 100,
                        "average_response_time": analytics.average_response_time,
                        "most_used_commands": analytics.most_used_commands,
                        "peak_usage_hours": analytics.peak_usage_hours,
                        "user_satisfaction_score": analytics.user_satisfaction_score,
                        "last_updated": analytics.last_analytics_update.isoformat()
                    }
            
            # Calculate summary
            total_commands = sum(data["total_commands"] for data in analytics_data["plugins"].values())
            total_successful = sum(data["successful_commands"] for data in analytics_data["plugins"].values())
            
            analytics_data["summary"] = {
                "total_plugins": len(analytics_data["plugins"]),
                "total_commands": total_commands,
                "overall_success_rate": (total_successful / max(total_commands, 1)) * 100,
                "average_response_time": sum(data["average_response_time"] for data in analytics_data["plugins"].values()) / max(len(analytics_data["plugins"]), 1),
                "average_satisfaction": sum(data["user_satisfaction_score"] for data in analytics_data["plugins"].values()) / max(len(analytics_data["plugins"]), 1)
            }
            
            return analytics_data
            
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            raise
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        try:
            registry_status = self.plugin_manager.get_registry_status() if hasattr(self.plugin_manager, 'get_registry_status') else {}
            
            metrics = {
                "plugin_registry": registry_status,
                "completion_engine": self.completion_engine.get_engine_stats(),
                "communication": self.communication.get_connection_status(),
                "system": {
                    "initialized": self._initialized,
                    "background_tasks": len(self._background_tasks),
                    "uptime_seconds": (datetime.now() - datetime.now()).total_seconds()  # Would track actual uptime
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            raise
    
    # Communication Management Methods
    
    async def get_communication_status(self) -> Dict[str, Any]:
        """Get communication channels status"""
        try:
            return self.communication.get_connection_status()
        except Exception as e:
            logger.error(f"Error getting communication status: {e}")
            raise
    
    async def connect_to_plugin(self, plugin_id: str, channel_name: str, 
                              config: Dict[str, Any]) -> bool:
        """Connect to a plugin via specified communication channel"""
        try:
            # Create connection configuration
            conn_config = ConnectionConfig(
                host=config.get("host", "localhost"),
                port=config.get("port", 8080),
                use_ssl=config.get("use_ssl", False),
                timeout_seconds=config.get("timeout_seconds", 30),
                reconnect_attempts=config.get("reconnect_attempts", 5),
                custom_headers=config.get("custom_headers", {})
            )
            
            return await self.communication.connect_to_ide(plugin_id, channel_name, conn_config)
            
        except Exception as e:
            logger.error(f"Error connecting to plugin {plugin_id}: {e}")
            raise
    
    async def disconnect_from_plugin(self, plugin_id: str) -> bool:
        """Disconnect from a plugin"""
        try:
            return await self.communication.disconnect_from_ide(plugin_id)
        except Exception as e:
            logger.error(f"Error disconnecting from plugin {plugin_id}: {e}")
            raise
    
    # Debug and Development Methods
    
    async def get_registry_debug_info(self) -> Dict[str, Any]:
        """Get plugin registry information for debugging"""
        try:
            debug_info = {
                "plugins": {},
                "plugin_classes": list(self.plugin_manager.plugin_classes.keys()),
                "installations": {},
                "health_metrics": {},
                "analytics": {}
            }
            
            # Plugin information
            for plugin_id, plugin in self.plugin_manager.plugins.items():
                debug_info["plugins"][plugin_id] = {
                    "class_name": plugin.__class__.__name__,
                    "status": plugin.status.value,
                    "metadata": plugin.get_metadata(),
                    "capabilities": [cap.value for cap in plugin.metadata.capabilities],
                    "session_id": plugin._session_id,
                    "error_count": plugin._error_count,
                    "last_error": plugin._last_error
                }
            
            # Installation information
            for plugin_id, installation in self.plugin_manager.installations.items():
                debug_info["installations"][plugin_id] = {
                    "version": installation.version,
                    "install_path": installation.install_path,
                    "installed_at": installation.installed_at.isoformat(),
                    "auto_update": installation.auto_update,
                    "dependencies": installation.dependencies
                }
            
            # Health metrics
            for plugin_id, health in self.plugin_manager.health_metrics.items():
                debug_info["health_metrics"][plugin_id] = {
                    "status": health.status.value,
                    "last_check": health.last_health_check.isoformat(),
                    "response_time_ms": health.response_time_ms,
                    "error_count": health.error_count,
                    "success_rate": health.success_rate,
                    "uptime_seconds": health.uptime_seconds
                }
            
            # Analytics
            for plugin_id, analytics in self.plugin_manager.analytics.items():
                debug_info["analytics"][plugin_id] = {
                    "total_commands": analytics.total_commands,
                    "success_rate": (analytics.successful_commands / max(analytics.total_commands, 1)) * 100,
                    "average_response_time": analytics.average_response_time,
                    "most_used_commands": analytics.most_used_commands
                }
            
            return debug_info
            
        except Exception as e:
            logger.error(f"Error getting registry debug info: {e}")
            raise
    
    async def test_command(self, plugin_id: str, command_name: str, 
                         parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Test command execution for debugging"""
        try:
            start_time = datetime.now()
            
            # Execute command
            response = await self.execute_plugin_command(plugin_id, command_name, parameters, 60)
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            test_result = {
                "plugin_id": plugin_id,
                "command_name": command_name,
                "parameters": parameters,
                "response": {
                    "success": response.success,
                    "data": response.data,
                    "error_message": response.error_message,
                    "error_code": response.error_code,
                    "execution_time_ms": response.execution_time_ms
                },
                "test_execution_time_ms": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            
            return test_result
            
        except Exception as e:
            logger.error(f"Error testing command: {e}")
            raise
    
    # Batch Operations Methods
    
    async def batch_start_plugins(self, plugin_ids: List[str]) -> Dict[str, bool]:
        """Start multiple plugins in batch"""
        try:
            results = {}
            
            # Create tasks for parallel execution
            tasks = []
            for plugin_id in plugin_ids:
                task = asyncio.create_task(self.plugin_manager.start_plugin(plugin_id))
                tasks.append((plugin_id, task))
            
            # Wait for all tasks to complete
            for plugin_id, task in tasks:
                try:
                    result = await task
                    results[plugin_id] = result
                except Exception as e:
                    logger.error(f"Error starting plugin {plugin_id}: {e}")
                    results[plugin_id] = False
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch start plugins: {e}")
            raise
    
    async def batch_stop_plugins(self, plugin_ids: List[str]) -> Dict[str, bool]:
        """Stop multiple plugins in batch"""
        try:
            results = {}
            
            # Create tasks for parallel execution
            tasks = []
            for plugin_id in plugin_ids:
                task = asyncio.create_task(self.plugin_manager.stop_plugin(plugin_id))
                tasks.append((plugin_id, task))
            
            # Wait for all tasks to complete
            for plugin_id, task in tasks:
                try:
                    result = await task
                    results[plugin_id] = result
                except Exception as e:
                    logger.error(f"Error stopping plugin {plugin_id}: {e}")
                    results[plugin_id] = False
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch stop plugins: {e}")
            raise
    
    # Configuration Management Methods
    
    async def get_default_configuration(self, plugin_type: Optional[str] = None) -> Dict[str, Any]:
        """Get default configuration for plugin type"""
        try:
            # Return default configuration template
            default_config = {
                "settings": {},
                "enabled_capabilities": [],
                "communication_endpoints": {},
                "auto_update": True,
                "logging_level": "INFO",
                "custom_keybindings": {},
                "ui_preferences": {}
            }
            
            # Plugin-type specific defaults
            if plugin_type:
                type_specific = {
                    "vscode": {
                        "communication_endpoints": {"websocket": "ws://localhost:8054"},
                        "enabled_capabilities": ["code_completion", "syntax_highlighting", "error_detection"]
                    },
                    "intellij": {
                        "communication_endpoints": {"http": "http://localhost:8055"},
                        "enabled_capabilities": ["code_completion", "refactoring", "debugging"]
                    },
                    "vim": {
                        "communication_endpoints": {"websocket": "ws://localhost:8056"},
                        "enabled_capabilities": ["code_completion", "custom_commands"]
                    },
                    "emacs": {
                        "communication_endpoints": {"http": "http://localhost:8057"},
                        "enabled_capabilities": ["code_completion", "intelligent_suggestions"]
                    }
                }
                
                if plugin_type in type_specific:
                    default_config.update(type_specific[plugin_type])
            
            return default_config
            
        except Exception as e:
            logger.error(f"Error getting default configuration: {e}")
            raise
    
    async def validate_configuration(self, plugin_id: str, 
                                   configuration: Dict[str, Any]) -> Dict[str, Any]:
        """Validate plugin configuration"""
        try:
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "suggestions": []
            }
            
            # Basic validation
            required_fields = ["settings"]
            for field in required_fields:
                if field not in configuration:
                    validation_result["errors"].append(f"Required field '{field}' is missing")
                    validation_result["valid"] = False
            
            # Validate capabilities
            if "enabled_capabilities" in configuration:
                capabilities = configuration["enabled_capabilities"]
                if not isinstance(capabilities, list):
                    validation_result["errors"].append("enabled_capabilities must be a list")
                    validation_result["valid"] = False
                else:
                    valid_capabilities = [cap.value for cap in PluginCapability]
                    for cap in capabilities:
                        if cap not in valid_capabilities:
                            validation_result["warnings"].append(f"Unknown capability: {cap}")
            
            # Validate communication endpoints
            if "communication_endpoints" in configuration:
                endpoints = configuration["communication_endpoints"]
                if not isinstance(endpoints, dict):
                    validation_result["errors"].append("communication_endpoints must be a dictionary")
                    validation_result["valid"] = False
            
            # Add suggestions
            if validation_result["valid"]:
                validation_result["suggestions"].append("Configuration looks good!")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating configuration: {e}")
            raise