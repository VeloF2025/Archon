"""
Base Plugin Framework
Provides abstract base classes and core interfaces for IDE plugins
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel, Field
import json
import uuid


class PluginCapability(Enum):
    """Available plugin capabilities"""
    CODE_COMPLETION = "code_completion"
    SYNTAX_HIGHLIGHTING = "syntax_highlighting"
    ERROR_DETECTION = "error_detection"
    REFACTORING = "refactoring"
    DEBUGGING = "debugging"
    REAL_TIME_COLLABORATION = "real_time_collaboration"
    INTELLIGENT_SUGGESTIONS = "intelligent_suggestions"
    CODE_ANALYSIS = "code_analysis"
    DOCUMENTATION_GENERATION = "documentation_generation"
    TEST_GENERATION = "test_generation"
    PERFORMANCE_PROFILING = "performance_profiling"
    SECURITY_ANALYSIS = "security_analysis"
    GIT_INTEGRATION = "git_integration"
    PROJECT_MANAGEMENT = "project_management"
    CUSTOM_COMMANDS = "custom_commands"


class PluginStatus(Enum):
    """Plugin status states"""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    ERROR = "error"
    UPDATING = "updating"
    DISABLED = "disabled"


class CommunicationProtocol(Enum):
    """Communication protocols supported"""
    HTTP = "http"
    WEBSOCKET = "websocket"
    JSON_RPC = "json_rpc"
    LSP = "lsp"  # Language Server Protocol
    DAP = "dap"  # Debug Adapter Protocol
    CUSTOM = "custom"


@dataclass
class PluginMetadata:
    """Metadata for IDE plugins"""
    name: str
    version: str
    description: str
    author: str
    supported_languages: List[str]
    capabilities: List[PluginCapability]
    protocols: List[CommunicationProtocol]
    min_ide_version: str
    max_ide_version: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    configuration_schema: Dict[str, Any] = field(default_factory=dict)
    installation_url: Optional[str] = None
    documentation_url: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class PluginConfiguration:
    """Plugin configuration settings"""
    plugin_id: str
    settings: Dict[str, Any] = field(default_factory=dict)
    enabled_capabilities: List[PluginCapability] = field(default_factory=list)
    communication_endpoints: Dict[str, str] = field(default_factory=dict)
    auto_update: bool = True
    logging_level: str = "INFO"
    custom_keybindings: Dict[str, str] = field(default_factory=dict)
    ui_preferences: Dict[str, Any] = field(default_factory=dict)


class PluginEvent(BaseModel):
    """Plugin event structure"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    plugin_id: str
    event_type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(default_factory=dict)
    source: str
    target: Optional[str] = None


class PluginCommand(BaseModel):
    """Plugin command structure"""
    command_id: str
    command_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    expected_response_type: str = "json"
    timeout_seconds: int = 30
    requires_user_confirmation: bool = False


class PluginResponse(BaseModel):
    """Plugin response structure"""
    response_id: str
    command_id: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    execution_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.now)


class PluginBase(ABC):
    """Abstract base class for all IDE plugins"""
    
    def __init__(self, metadata: PluginMetadata, configuration: Optional[PluginConfiguration] = None):
        self.metadata = metadata
        self.configuration = configuration or PluginConfiguration(plugin_id=metadata.name)
        self.status = PluginStatus.INACTIVE
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.command_handlers: Dict[str, Callable] = {}
        self._session_id = str(uuid.uuid4())
        self._communication_channels: Dict[str, Any] = {}
        self._active_connections = 0
        self._error_count = 0
        self._last_error: Optional[str] = None
        
    @property
    def plugin_id(self) -> str:
        """Get unique plugin identifier"""
        return self.metadata.name
        
    @property
    def is_active(self) -> bool:
        """Check if plugin is active"""
        return self.status == PluginStatus.ACTIVE
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the plugin"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown the plugin"""
        pass
    
    @abstractmethod
    async def execute_command(self, command: PluginCommand) -> PluginResponse:
        """Execute a plugin command"""
        pass
    
    @abstractmethod 
    def supports_capability(self, capability: PluginCapability) -> bool:
        """Check if plugin supports a specific capability"""
        pass
    
    @abstractmethod
    async def handle_event(self, event: PluginEvent) -> None:
        """Handle plugin events"""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get plugin status information"""
        pass
    
    async def start(self) -> bool:
        """Start the plugin"""
        try:
            self.status = PluginStatus.INITIALIZING
            success = await self.initialize()
            if success:
                self.status = PluginStatus.ACTIVE
                await self.emit_event("plugin_started", {"plugin_id": self.plugin_id})
            else:
                self.status = PluginStatus.ERROR
                self._error_count += 1
            return success
        except Exception as e:
            self.status = PluginStatus.ERROR
            self._last_error = str(e)
            self._error_count += 1
            return False
    
    async def stop(self) -> bool:
        """Stop the plugin"""
        try:
            success = await self.shutdown()
            self.status = PluginStatus.INACTIVE
            await self.emit_event("plugin_stopped", {"plugin_id": self.plugin_id})
            return success
        except Exception as e:
            self._last_error = str(e)
            self._error_count += 1
            return False
    
    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """Register an event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def register_command_handler(self, command_name: str, handler: Callable) -> None:
        """Register a command handler"""
        self.command_handlers[command_name] = handler
    
    async def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event"""
        event = PluginEvent(
            plugin_id=self.plugin_id,
            event_type=event_type,
            data=data,
            source=self.plugin_id
        )
        
        # Handle internally registered handlers
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(event)
                except Exception as e:
                    self._error_count += 1
                    self._last_error = f"Event handler error: {str(e)}"
    
    async def update_configuration(self, new_config: Dict[str, Any]) -> bool:
        """Update plugin configuration"""
        try:
            for key, value in new_config.items():
                if hasattr(self.configuration, key):
                    setattr(self.configuration, key, value)
                else:
                    self.configuration.settings[key] = value
            
            await self.emit_event("configuration_updated", {"changes": new_config})
            return True
        except Exception as e:
            self._last_error = str(e)
            self._error_count += 1
            return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata as dictionary"""
        return {
            "name": self.metadata.name,
            "version": self.metadata.version,
            "description": self.metadata.description,
            "author": self.metadata.author,
            "supported_languages": self.metadata.supported_languages,
            "capabilities": [cap.value for cap in self.metadata.capabilities],
            "protocols": [proto.value for proto in self.metadata.protocols],
            "min_ide_version": self.metadata.min_ide_version,
            "max_ide_version": self.metadata.max_ide_version,
            "dependencies": self.metadata.dependencies,
            "installation_url": self.metadata.installation_url,
            "documentation_url": self.metadata.documentation_url
        }
    
    def get_health_info(self) -> Dict[str, Any]:
        """Get plugin health information"""
        return {
            "plugin_id": self.plugin_id,
            "status": self.status.value,
            "active_connections": self._active_connections,
            "error_count": self._error_count,
            "last_error": self._last_error,
            "session_id": self._session_id,
            "uptime": (datetime.now() - self.metadata.created_at).total_seconds()
        }
    
    async def validate_command(self, command: PluginCommand) -> bool:
        """Validate command before execution"""
        # Check if command is supported
        if command.command_name not in self.command_handlers:
            return False
        
        # Check if plugin is active
        if not self.is_active:
            return False
        
        # Additional validation can be implemented by subclasses
        return True
    
    def create_response(self, command: PluginCommand, success: bool, 
                       data: Optional[Dict[str, Any]] = None,
                       error_message: Optional[str] = None,
                       error_code: Optional[str] = None,
                       execution_time_ms: int = 0) -> PluginResponse:
        """Create a standardized plugin response"""
        return PluginResponse(
            response_id=str(uuid.uuid4()),
            command_id=command.command_id,
            success=success,
            data=data,
            error_message=error_message,
            error_code=error_code,
            execution_time_ms=execution_time_ms
        )


class PluginRegistry:
    """Registry for managing plugin instances"""
    
    def __init__(self):
        self._plugins: Dict[str, PluginBase] = {}
        self._metadata_cache: Dict[str, PluginMetadata] = {}
        
    def register_plugin(self, plugin: PluginBase) -> bool:
        """Register a plugin instance"""
        try:
            self._plugins[plugin.plugin_id] = plugin
            self._metadata_cache[plugin.plugin_id] = plugin.metadata
            return True
        except Exception:
            return False
    
    def unregister_plugin(self, plugin_id: str) -> bool:
        """Unregister a plugin"""
        try:
            if plugin_id in self._plugins:
                del self._plugins[plugin_id]
            if plugin_id in self._metadata_cache:
                del self._metadata_cache[plugin_id]
            return True
        except Exception:
            return False
    
    def get_plugin(self, plugin_id: str) -> Optional[PluginBase]:
        """Get plugin by ID"""
        return self._plugins.get(plugin_id)
    
    def get_all_plugins(self) -> List[PluginBase]:
        """Get all registered plugins"""
        return list(self._plugins.values())
    
    def get_plugins_by_capability(self, capability: PluginCapability) -> List[PluginBase]:
        """Get plugins that support a specific capability"""
        return [plugin for plugin in self._plugins.values() 
                if plugin.supports_capability(capability)]
    
    def get_active_plugins(self) -> List[PluginBase]:
        """Get all active plugins"""
        return [plugin for plugin in self._plugins.values() if plugin.is_active]
    
    async def start_all_plugins(self) -> Dict[str, bool]:
        """Start all registered plugins"""
        results = {}
        for plugin_id, plugin in self._plugins.items():
            results[plugin_id] = await plugin.start()
        return results
    
    async def stop_all_plugins(self) -> Dict[str, bool]:
        """Stop all registered plugins"""
        results = {}
        for plugin_id, plugin in self._plugins.items():
            results[plugin_id] = await plugin.stop()
        return results
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get registry status"""
        total_plugins = len(self._plugins)
        active_plugins = len(self.get_active_plugins())
        
        capabilities_count = {}
        for plugin in self._plugins.values():
            for capability in plugin.metadata.capabilities:
                capabilities_count[capability.value] = capabilities_count.get(capability.value, 0) + 1
        
        return {
            "total_plugins": total_plugins,
            "active_plugins": active_plugins,
            "inactive_plugins": total_plugins - active_plugins,
            "capabilities_distribution": capabilities_count,
            "plugin_list": list(self._plugins.keys())
        }


# Global plugin registry instance
plugin_registry = PluginRegistry()