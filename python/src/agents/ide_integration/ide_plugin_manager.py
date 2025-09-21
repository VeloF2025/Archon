"""
IDE Plugin Manager
Centralized management system for all IDE plugins and integrations
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Union, Type, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
from pathlib import Path
import importlib
import inspect
from concurrent.futures import ThreadPoolExecutor

from .plugin_base import (
    PluginBase, PluginMetadata, PluginConfiguration, PluginCapability,
    PluginCommand, PluginResponse, PluginEvent, PluginStatus, plugin_registry
)
from .ide_communication import IDECommunicationProtocol
from .vscode_plugin import VSCodePlugin
from .intellij_plugin import IntelliJPlugin  # Will be implemented
from .vim_plugin import VimPlugin  # Will be implemented
from .emacs_plugin import EmacsPlugin  # Will be implemented

logger = logging.getLogger(__name__)


class PluginPriority(Enum):
    """Plugin priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class PluginInstallation:
    """Plugin installation information"""
    plugin_id: str
    version: str
    install_path: str
    installed_at: datetime
    last_updated: datetime
    auto_update: bool = True
    dependencies: List[str] = field(default_factory=list)
    configuration: Optional[PluginConfiguration] = None


@dataclass
class PluginHealth:
    """Plugin health metrics"""
    plugin_id: str
    status: PluginStatus
    last_health_check: datetime
    response_time_ms: float
    memory_usage_mb: float
    error_count: int
    success_rate: float
    uptime_seconds: float
    last_error: Optional[str] = None


@dataclass
class PluginAnalytics:
    """Plugin usage analytics"""
    plugin_id: str
    total_commands: int
    successful_commands: int
    failed_commands: int
    average_response_time: float
    most_used_commands: Dict[str, int]
    peak_usage_hours: List[int]
    user_satisfaction_score: float
    last_analytics_update: datetime


class IDEPluginManager:
    """Comprehensive IDE plugin management system"""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path.home() / ".archon" / "plugins"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Plugin management
        self.plugins: Dict[str, PluginBase] = {}
        self.plugin_classes: Dict[str, Type[PluginBase]] = {}
        self.installations: Dict[str, PluginInstallation] = {}
        self.health_metrics: Dict[str, PluginHealth] = {}
        self.analytics: Dict[str, PluginAnalytics] = {}
        
        # Communication
        self.communication = IDECommunicationProtocol()
        
        # Management
        self.auto_discovery_enabled = True
        self.health_check_interval = 60  # seconds
        self.max_concurrent_plugins = 10
        self.plugin_timeout = 30  # seconds
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._analytics_task: Optional[asyncio.Task] = None
        self._executor = ThreadPoolExecutor(max_workers=5)
        
        # Event handling
        self.event_handlers: Dict[str, List[Callable]] = {}
        self._global_event_queue = asyncio.Queue()
        
        # Initialize built-in plugins
        self._register_builtin_plugins()
        
        # Load configuration
        asyncio.create_task(self._load_configuration())
    
    def _register_builtin_plugins(self) -> None:
        """Register built-in plugin classes"""
        builtin_plugins = {
            "vscode": VSCodePlugin,
            # "intellij": IntelliJPlugin,  # Will be implemented
            # "vim": VimPlugin,  # Will be implemented  
            # "emacs": EmacsPlugin,  # Will be implemented
            # "sublime": SublimeTextPlugin,  # Will be implemented
            # "atom": AtomPlugin  # Will be implemented
        }
        
        self.plugin_classes.update(builtin_plugins)
        logger.info(f"Registered {len(builtin_plugins)} built-in plugin classes")
    
    async def _load_configuration(self) -> None:
        """Load plugin configurations from disk"""
        try:
            config_file = self.config_dir / "plugin_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Load installations
                for install_data in config_data.get("installations", []):
                    installation = PluginInstallation(
                        plugin_id=install_data["plugin_id"],
                        version=install_data["version"],
                        install_path=install_data["install_path"],
                        installed_at=datetime.fromisoformat(install_data["installed_at"]),
                        last_updated=datetime.fromisoformat(install_data["last_updated"]),
                        auto_update=install_data.get("auto_update", True),
                        dependencies=install_data.get("dependencies", [])
                    )
                    self.installations[installation.plugin_id] = installation
                
                logger.info(f"Loaded configuration for {len(self.installations)} plugins")
                
        except Exception as e:
            logger.error(f"Error loading plugin configuration: {e}")
    
    async def _save_configuration(self) -> None:
        """Save plugin configurations to disk"""
        try:
            config_data = {
                "installations": [],
                "last_updated": datetime.now().isoformat()
            }
            
            for installation in self.installations.values():
                config_data["installations"].append({
                    "plugin_id": installation.plugin_id,
                    "version": installation.version,
                    "install_path": installation.install_path,
                    "installed_at": installation.installed_at.isoformat(),
                    "last_updated": installation.last_updated.isoformat(),
                    "auto_update": installation.auto_update,
                    "dependencies": installation.dependencies
                })
            
            config_file = self.config_dir / "plugin_config.json"
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving plugin configuration: {e}")
    
    async def install_plugin(self, plugin_id: str, version: str = "latest", 
                           config: Optional[PluginConfiguration] = None) -> bool:
        """Install a plugin"""
        try:
            logger.info(f"Installing plugin: {plugin_id} (version: {version})")
            
            # Check if plugin class is available
            if plugin_id not in self.plugin_classes:
                if self.auto_discovery_enabled:
                    await self._discover_plugin_class(plugin_id)
                
                if plugin_id not in self.plugin_classes:
                    logger.error(f"Plugin class not found: {plugin_id}")
                    return False
            
            # Create plugin instance
            plugin_class = self.plugin_classes[plugin_id]
            plugin_instance = plugin_class(config)
            
            # Initialize plugin
            success = await plugin_instance.start()
            if not success:
                logger.error(f"Failed to start plugin: {plugin_id}")
                return False
            
            # Register plugin
            self.plugins[plugin_id] = plugin_instance
            plugin_registry.register_plugin(plugin_instance)
            
            # Create installation record
            installation = PluginInstallation(
                plugin_id=plugin_id,
                version=version,
                install_path=str(self.config_dir / plugin_id),
                installed_at=datetime.now(),
                last_updated=datetime.now(),
                configuration=config
            )
            self.installations[plugin_id] = installation
            
            # Initialize health tracking
            self.health_metrics[plugin_id] = PluginHealth(
                plugin_id=plugin_id,
                status=plugin_instance.status,
                last_health_check=datetime.now(),
                response_time_ms=0.0,
                memory_usage_mb=0.0,
                error_count=0,
                success_rate=1.0,
                uptime_seconds=0.0
            )
            
            # Initialize analytics
            self.analytics[plugin_id] = PluginAnalytics(
                plugin_id=plugin_id,
                total_commands=0,
                successful_commands=0,
                failed_commands=0,
                average_response_time=0.0,
                most_used_commands={},
                peak_usage_hours=[],
                user_satisfaction_score=5.0,
                last_analytics_update=datetime.now()
            )
            
            await self._save_configuration()
            await self._emit_event("plugin_installed", {"plugin_id": plugin_id})
            
            logger.info(f"Successfully installed plugin: {plugin_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error installing plugin {plugin_id}: {e}")
            return False
    
    async def uninstall_plugin(self, plugin_id: str) -> bool:
        """Uninstall a plugin"""
        try:
            logger.info(f"Uninstalling plugin: {plugin_id}")
            
            # Stop plugin if running
            if plugin_id in self.plugins:
                plugin = self.plugins[plugin_id]
                await plugin.stop()
                plugin_registry.unregister_plugin(plugin_id)
                del self.plugins[plugin_id]
            
            # Remove installation record
            if plugin_id in self.installations:
                del self.installations[plugin_id]
            
            # Remove metrics
            self.health_metrics.pop(plugin_id, None)
            self.analytics.pop(plugin_id, None)
            
            await self._save_configuration()
            await self._emit_event("plugin_uninstalled", {"plugin_id": plugin_id})
            
            logger.info(f"Successfully uninstalled plugin: {plugin_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error uninstalling plugin {plugin_id}: {e}")
            return False
    
    async def start_plugin(self, plugin_id: str) -> bool:
        """Start a specific plugin"""
        try:
            plugin = self.plugins.get(plugin_id)
            if not plugin:
                logger.error(f"Plugin not found: {plugin_id}")
                return False
            
            success = await plugin.start()
            if success:
                await self._emit_event("plugin_started", {"plugin_id": plugin_id})
            
            return success
            
        except Exception as e:
            logger.error(f"Error starting plugin {plugin_id}: {e}")
            return False
    
    async def stop_plugin(self, plugin_id: str) -> bool:
        """Stop a specific plugin"""
        try:
            plugin = self.plugins.get(plugin_id)
            if not plugin:
                logger.error(f"Plugin not found: {plugin_id}")
                return False
            
            success = await plugin.stop()
            if success:
                await self._emit_event("plugin_stopped", {"plugin_id": plugin_id})
            
            return success
            
        except Exception as e:
            logger.error(f"Error stopping plugin {plugin_id}: {e}")
            return False
    
    async def execute_plugin_command(self, plugin_id: str, command: PluginCommand) -> PluginResponse:
        """Execute command on specific plugin"""
        try:
            plugin = self.plugins.get(plugin_id)
            if not plugin:
                return PluginResponse(
                    response_id=str(uuid.uuid4()),
                    command_id=command.command_id,
                    success=False,
                    error_message=f"Plugin not found: {plugin_id}",
                    error_code="PLUGIN_NOT_FOUND",
                    execution_time_ms=0
                )
            
            # Record analytics
            if plugin_id in self.analytics:
                analytics = self.analytics[plugin_id]
                analytics.total_commands += 1
                if command.command_name in analytics.most_used_commands:
                    analytics.most_used_commands[command.command_name] += 1
                else:
                    analytics.most_used_commands[command.command_name] = 1
            
            # Execute command with timeout
            response = await asyncio.wait_for(
                plugin.execute_command(command),
                timeout=self.plugin_timeout
            )
            
            # Update analytics
            if plugin_id in self.analytics:
                analytics = self.analytics[plugin_id]
                if response.success:
                    analytics.successful_commands += 1
                else:
                    analytics.failed_commands += 1
                
                # Update average response time
                total_commands = analytics.successful_commands + analytics.failed_commands
                current_avg = analytics.average_response_time
                new_avg = ((current_avg * (total_commands - 1)) + response.execution_time_ms) / total_commands
                analytics.average_response_time = new_avg
                analytics.last_analytics_update = datetime.now()
            
            return response
            
        except asyncio.TimeoutError:
            logger.error(f"Plugin command timeout: {plugin_id}.{command.command_name}")
            return PluginResponse(
                response_id=str(uuid.uuid4()),
                command_id=command.command_id,
                success=False,
                error_message="Command execution timeout",
                error_code="TIMEOUT",
                execution_time_ms=self.plugin_timeout * 1000
            )
        except Exception as e:
            logger.error(f"Error executing plugin command: {e}")
            return PluginResponse(
                response_id=str(uuid.uuid4()),
                command_id=command.command_id,
                success=False,
                error_message=str(e),
                error_code="EXECUTION_ERROR",
                execution_time_ms=0
            )
    
    async def broadcast_event(self, event: PluginEvent, target_capabilities: Optional[List[PluginCapability]] = None) -> Dict[str, bool]:
        """Broadcast event to all or specific plugins"""
        results = {}
        
        for plugin_id, plugin in self.plugins.items():
            try:
                # Filter by capabilities if specified
                if target_capabilities:
                    if not any(plugin.supports_capability(cap) for cap in target_capabilities):
                        continue
                
                await plugin.handle_event(event)
                results[plugin_id] = True
                
            except Exception as e:
                logger.error(f"Error broadcasting event to {plugin_id}: {e}")
                results[plugin_id] = False
        
        return results
    
    async def get_plugins_by_capability(self, capability: PluginCapability) -> List[PluginBase]:
        """Get all plugins supporting a specific capability"""
        return [plugin for plugin in self.plugins.values() 
                if plugin.supports_capability(capability)]
    
    async def get_plugin_status(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a specific plugin"""
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            return None
        
        status = await plugin.get_status()
        
        # Add management metrics
        if plugin_id in self.health_metrics:
            status["health"] = {
                "last_health_check": self.health_metrics[plugin_id].last_health_check.isoformat(),
                "response_time_ms": self.health_metrics[plugin_id].response_time_ms,
                "memory_usage_mb": self.health_metrics[plugin_id].memory_usage_mb,
                "error_count": self.health_metrics[plugin_id].error_count,
                "success_rate": self.health_metrics[plugin_id].success_rate,
                "uptime_seconds": self.health_metrics[plugin_id].uptime_seconds
            }
        
        if plugin_id in self.analytics:
            analytics = self.analytics[plugin_id]
            status["analytics"] = {
                "total_commands": analytics.total_commands,
                "success_rate": (analytics.successful_commands / max(analytics.total_commands, 1)) * 100,
                "average_response_time": analytics.average_response_time,
                "most_used_commands": analytics.most_used_commands,
                "user_satisfaction_score": analytics.user_satisfaction_score
            }
        
        return status
    
    async def get_all_plugin_statuses(self) -> Dict[str, Any]:
        """Get status of all plugins"""
        statuses = {}
        
        for plugin_id in self.plugins.keys():
            status = await self.get_plugin_status(plugin_id)
            if status:
                statuses[plugin_id] = status
        
        return {
            "plugins": statuses,
            "summary": {
                "total_plugins": len(self.plugins),
                "active_plugins": len([p for p in self.plugins.values() if p.is_active]),
                "capabilities_coverage": self._calculate_capabilities_coverage(),
                "average_response_time": self._calculate_average_response_time(),
                "overall_success_rate": self._calculate_overall_success_rate()
            }
        }
    
    async def update_plugin_configuration(self, plugin_id: str, config_updates: Dict[str, Any]) -> bool:
        """Update plugin configuration"""
        try:
            plugin = self.plugins.get(plugin_id)
            if not plugin:
                return False
            
            success = await plugin.update_configuration(config_updates)
            if success:
                # Update installation record
                if plugin_id in self.installations:
                    self.installations[plugin_id].last_updated = datetime.now()
                await self._save_configuration()
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating plugin configuration {plugin_id}: {e}")
            return False
    
    async def start_health_monitoring(self) -> None:
        """Start background health monitoring"""
        if self._health_check_task and not self._health_check_task.done():
            return
        
        self._health_check_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Started plugin health monitoring")
    
    async def stop_health_monitoring(self) -> None:
        """Stop background health monitoring"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped plugin health monitoring")
    
    async def start_analytics_collection(self) -> None:
        """Start background analytics collection"""
        if self._analytics_task and not self._analytics_task.done():
            return
        
        self._analytics_task = asyncio.create_task(self._analytics_loop())
        logger.info("Started plugin analytics collection")
    
    async def stop_analytics_collection(self) -> None:
        """Stop background analytics collection"""
        if self._analytics_task:
            self._analytics_task.cancel()
            try:
                await self._analytics_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped plugin analytics collection")
    
    async def shutdown(self) -> None:
        """Shutdown plugin manager"""
        logger.info("Shutting down plugin manager...")
        
        # Stop background tasks
        await self.stop_health_monitoring()
        await self.stop_analytics_collection()
        
        # Stop all plugins
        for plugin_id in list(self.plugins.keys()):
            await self.stop_plugin(plugin_id)
        
        # Save final configuration
        await self._save_configuration()
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        logger.info("Plugin manager shutdown complete")
    
    # Private methods
    
    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop"""
        while True:
            try:
                for plugin_id, plugin in self.plugins.items():
                    await self._check_plugin_health(plugin_id, plugin)
                
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(10)
    
    async def _check_plugin_health(self, plugin_id: str, plugin: PluginBase) -> None:
        """Check health of individual plugin"""
        try:
            start_time = datetime.now()
            
            # Get plugin status
            status = await plugin.get_status()
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update health metrics
            if plugin_id in self.health_metrics:
                health = self.health_metrics[plugin_id]
                health.last_health_check = datetime.now()
                health.response_time_ms = response_time
                health.status = plugin.status
                # Memory usage would be calculated differently in production
                health.memory_usage_mb = 0.0  # Placeholder
                health.uptime_seconds = (datetime.now() - plugin.metadata.created_at).total_seconds()
                
                # Calculate success rate
                if plugin_id in self.analytics:
                    analytics = self.analytics[plugin_id]
                    total = analytics.total_commands
                    if total > 0:
                        health.success_rate = analytics.successful_commands / total
                    else:
                        health.success_rate = 1.0
                
        except Exception as e:
            logger.error(f"Health check failed for {plugin_id}: {e}")
            if plugin_id in self.health_metrics:
                health = self.health_metrics[plugin_id]
                health.error_count += 1
                health.last_error = str(e)
    
    async def _analytics_loop(self) -> None:
        """Background analytics collection loop"""
        while True:
            try:
                # Collect and analyze plugin usage patterns
                for plugin_id, analytics in self.analytics.items():
                    await self._analyze_plugin_usage(plugin_id, analytics)
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in analytics loop: {e}")
                await asyncio.sleep(60)
    
    async def _analyze_plugin_usage(self, plugin_id: str, analytics: PluginAnalytics) -> None:
        """Analyze plugin usage patterns"""
        try:
            # Update peak usage hours based on current hour
            current_hour = datetime.now().hour
            if current_hour not in analytics.peak_usage_hours and analytics.total_commands > 0:
                analytics.peak_usage_hours.append(current_hour)
                
                # Keep only top 5 peak hours
                if len(analytics.peak_usage_hours) > 5:
                    analytics.peak_usage_hours = analytics.peak_usage_hours[-5:]
            
            # Calculate user satisfaction score based on success rate and response time
            success_rate = (analytics.successful_commands / max(analytics.total_commands, 1))
            response_score = max(0, 5 - (analytics.average_response_time / 1000))  # Penalty for slow responses
            analytics.user_satisfaction_score = (success_rate * 5 + response_score) / 2
            
            analytics.last_analytics_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error analyzing plugin usage for {plugin_id}: {e}")
    
    async def _discover_plugin_class(self, plugin_id: str) -> None:
        """Discover plugin class dynamically"""
        try:
            # Try to import plugin module
            module_name = f"archon.plugins.{plugin_id}"
            module = importlib.import_module(module_name)
            
            # Find plugin class in module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, PluginBase) and 
                    obj != PluginBase):
                    self.plugin_classes[plugin_id] = obj
                    logger.info(f"Discovered plugin class: {plugin_id}")
                    return
                    
        except ImportError:
            logger.warning(f"Could not import plugin module: {plugin_id}")
        except Exception as e:
            logger.error(f"Error discovering plugin class {plugin_id}: {e}")
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit manager event"""
        event = PluginEvent(
            plugin_id="plugin_manager",
            event_type=event_type,
            data=data,
            source="plugin_manager"
        )
        
        await self._global_event_queue.put(event)
    
    def _calculate_capabilities_coverage(self) -> Dict[str, int]:
        """Calculate capability coverage across plugins"""
        coverage = {}
        for capability in PluginCapability:
            count = sum(1 for plugin in self.plugins.values() 
                       if plugin.supports_capability(capability))
            coverage[capability.value] = count
        return coverage
    
    def _calculate_average_response_time(self) -> float:
        """Calculate average response time across all plugins"""
        total_time = 0.0
        total_commands = 0
        
        for analytics in self.analytics.values():
            if analytics.total_commands > 0:
                total_time += analytics.average_response_time * analytics.total_commands
                total_commands += analytics.total_commands
        
        return total_time / max(total_commands, 1)
    
    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate across all plugins"""
        total_successful = sum(analytics.successful_commands for analytics in self.analytics.values())
        total_commands = sum(analytics.total_commands for analytics in self.analytics.values())
        
        return (total_successful / max(total_commands, 1)) * 100