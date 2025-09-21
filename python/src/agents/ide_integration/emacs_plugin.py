"""
Emacs Plugin
Comprehensive GNU Emacs integration for Archon AI development platform
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import logging
from pathlib import Path
import re

from .plugin_base import (
    PluginBase, PluginMetadata, PluginConfiguration, PluginCapability, 
    PluginCommand, PluginResponse, PluginEvent, PluginStatus,
    CommunicationProtocol
)
from .ide_communication import IDECommunicationProtocol, HTTPChannel, ConnectionConfig
from .code_completion_engine import CodeCompletionEngine, CompletionRequest, CodeContext, CompletionTrigger

logger = logging.getLogger(__name__)


@dataclass
class EmacsBuffer:
    """Emacs buffer representation"""
    name: str
    file_name: Optional[str]
    major_mode: str
    content: str
    modified: bool = False
    read_only: bool = False
    point: int = 0  # Current cursor position
    mark: Optional[int] = None  # Mark position
    local_variables: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmacsWindow:
    """Emacs window representation"""
    buffer: EmacsBuffer
    start: int  # Start position of visible area
    point: int  # Current point in window
    width: int
    height: int
    selected: bool = False


@dataclass
class EmacsFrame:
    """Emacs frame representation"""
    name: str
    windows: List[EmacsWindow]
    selected_window: int = 0
    width: int = 80
    height: int = 40


class EmacsPlugin(PluginBase):
    """GNU Emacs plugin implementation"""
    
    def __init__(self, config: Optional[PluginConfiguration] = None):
        metadata = PluginMetadata(
            name="archon-emacs", 
            version="2.0.0",
            description="Archon AI Development Platform - GNU Emacs Integration",
            author="Archon Development Team",
            supported_languages=["emacs-lisp", "python", "javascript", "typescript", "c", "cpp", "java", "go", "rust", "haskell"],
            capabilities=[
                PluginCapability.CODE_COMPLETION,
                PluginCapability.SYNTAX_HIGHLIGHTING,
                PluginCapability.ERROR_DETECTION,
                PluginCapability.REFACTORING,
                PluginCapability.INTELLIGENT_SUGGESTIONS,
                PluginCapability.CODE_ANALYSIS,
                PluginCapability.DOCUMENTATION_GENERATION,
                PluginCapability.TEST_GENERATION,
                PluginCapability.SECURITY_ANALYSIS,
                PluginCapability.GIT_INTEGRATION,
                PluginCapability.PROJECT_MANAGEMENT,
                PluginCapability.CUSTOM_COMMANDS
            ],
            protocols=[CommunicationProtocol.HTTP, CommunicationProtocol.JSON_RPC, CommunicationProtocol.CUSTOM],
            min_ide_version="27.1",
            installation_url="https://github.com/archon-ai/archon-emacs",
            documentation_url="https://docs.archon.dev/emacs-plugin"
        )
        
        super().__init__(metadata, config)
        
        # Emacs-specific components
        self.communication = IDECommunicationProtocol()
        self.completion_engine = CodeCompletionEngine()
        self.buffers: Dict[str, EmacsBuffer] = {}
        self.windows: Dict[int, EmacsWindow] = {}
        self.frames: Dict[str, EmacsFrame] = {}
        self.current_buffer: Optional[str] = None
        self.current_window: Optional[int] = None
        self.current_frame: str = "default"
        self.http_port = 8057
        
        # Emacs-specific features
        self.elisp_functions: Dict[str, Callable] = {}
        self.hooks: Dict[str, List[str]] = {}
        self.keybindings: Dict[str, str] = {}
        self.minor_modes: Dict[str, bool] = {}
        
        # Register command handlers
        self._register_command_handlers()
        
        # Register event handlers
        self._register_event_handlers()
    
    def _register_command_handlers(self) -> None:
        """Register Emacs command handlers"""
        commands = {
            # Buffer operations
            "buffer/list": self._handle_list_buffers,
            "buffer/switch": self._handle_switch_buffer,
            "buffer/create": self._handle_create_buffer,
            "buffer/save": self._handle_save_buffer,
            "buffer/kill": self._handle_kill_buffer,
            "buffer/rename": self._handle_rename_buffer,
            "buffer/get_content": self._handle_get_buffer_content,
            "buffer/set_content": self._handle_set_buffer_content,
            
            # Point and mark operations
            "point/get": self._handle_get_point,
            "point/set": self._handle_set_point,
            "point/move": self._handle_move_point,
            "mark/set": self._handle_set_mark,
            "mark/get": self._handle_get_mark,
            "region/get": self._handle_get_region,
            "region/replace": self._handle_replace_region,
            
            # Window operations
            "window/list": self._handle_list_windows,
            "window/select": self._handle_select_window,
            "window/split": self._handle_split_window,
            "window/delete": self._handle_delete_window,
            "window/configuration": self._handle_window_configuration,
            
            # Text operations
            "text/insert": self._handle_insert_text,
            "text/delete": self._handle_delete_text,
            "text/search": self._handle_search_text,
            "text/replace": self._handle_replace_text,
            "text/indent": self._handle_indent_text,
            
            # Completion operations
            "completion/at_point": self._handle_completion_at_point,
            "completion/capf": self._handle_completion_capf,
            "completion/company": self._handle_company_completion,
            
            # Navigation
            "navigate/definition": self._handle_goto_definition,
            "navigate/references": self._handle_find_references,
            "navigate/symbol": self._handle_goto_symbol,
            "navigate/imenu": self._handle_imenu,
            
            # Evaluation and REPL
            "eval/expression": self._handle_eval_expression,
            "eval/buffer": self._handle_eval_buffer,
            "eval/region": self._handle_eval_region,
            "repl/start": self._handle_start_repl,
            
            # Mode operations
            "mode/major": self._handle_set_major_mode,
            "mode/minor/toggle": self._handle_toggle_minor_mode,
            "mode/minor/list": self._handle_list_minor_modes,
            
            # Key bindings
            "keymap/define": self._handle_define_key,
            "keymap/lookup": self._handle_lookup_key,
            "keymap/describe": self._handle_describe_key,
            
            # Hooks
            "hook/add": self._handle_add_hook,
            "hook/remove": self._handle_remove_hook,
            "hook/run": self._handle_run_hooks,
            
            # Archon-specific commands
            "archon/analyze": self._handle_archon_analyze,
            "archon/complete": self._handle_archon_complete,
            "archon/refactor": self._handle_archon_refactor,
            "archon/generate_tests": self._handle_archon_generate_tests,
            "archon/format": self._handle_archon_format,
            "archon/lint": self._handle_archon_lint,
            "archon/git_status": self._handle_archon_git_status,
            "archon/project_browser": self._handle_archon_project_browser,
            "archon/flycheck": self._handle_archon_flycheck
        }
        
        for command, handler in commands.items():
            self.register_command_handler(command, handler)
    
    def _register_event_handlers(self) -> None:
        """Register Emacs event handlers"""
        events = {
            "buffer_list_update": self._on_buffer_list_update,
            "buffer_modified": self._on_buffer_modified,
            "buffer_saved": self._on_buffer_saved,
            "window_configuration_change": self._on_window_configuration_change,
            "pre_command": self._on_pre_command,
            "post_command": self._on_post_command,
            "find_file": self._on_find_file,
            "kill_emacs": self._on_kill_emacs,
            "focus_in": self._on_focus_in,
            "focus_out": self._on_focus_out
        }
        
        for event, handler in events.items():
            self.register_event_handler(event, handler)
    
    async def initialize(self) -> bool:
        """Initialize Emacs plugin"""
        try:
            logger.info("Initializing Emacs plugin...")
            
            # Setup HTTP communication
            http_config = ConnectionConfig(
                host="localhost",
                port=self.http_port,
                timeout_seconds=30
            )
            
            http_channel = HTTPChannel(http_config)
            self.communication.register_channel("http", http_channel)
            
            # Initialize completion engine
            await self._initialize_completion_engine()
            
            # Setup Emacs integration
            await self._setup_emacs_integration()
            
            # Create default frame
            self._create_default_frame()
            
            logger.info("Emacs plugin initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Emacs plugin: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown Emacs plugin"""
        try:
            logger.info("Shutting down Emacs plugin...")
            
            # Disconnect communication channels
            for ide_id in list(self.communication._active_connections.keys()):
                await self.communication.disconnect_from_ide(ide_id)
            
            # Clear state
            self.buffers.clear()
            self.windows.clear()
            self.frames.clear()
            self.elisp_functions.clear()
            self.hooks.clear()
            
            logger.info("Emacs plugin shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Error shutting down Emacs plugin: {e}")
            return False
    
    async def execute_command(self, command: PluginCommand) -> PluginResponse:
        """Execute Emacs plugin command"""
        start_time = datetime.now()
        
        try:
            if not await self.validate_command(command):
                return self.create_response(
                    command, False,
                    error_message="Invalid command",
                    error_code="INVALID_COMMAND"
                )
            
            if command.command_name in self.command_handlers:
                handler = self.command_handlers[command.command_name]
                result = await handler(command.parameters)
                
                execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
                return self.create_response(
                    command, True,
                    data=result,
                    execution_time_ms=execution_time
                )
            else:
                return self.create_response(
                    command, False,
                    error_message=f"Unknown command: {command.command_name}",
                    error_code="UNKNOWN_COMMAND"
                )
                
        except Exception as e:
            logger.error(f"Error executing command {command.command_name}: {e}")
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            return self.create_response(
                command, False,
                error_message=str(e),
                error_code="EXECUTION_ERROR",
                execution_time_ms=execution_time
            )
    
    def supports_capability(self, capability: PluginCapability) -> bool:
        """Check if plugin supports capability"""
        return capability in self.metadata.capabilities
    
    async def handle_event(self, event: PluginEvent) -> None:
        """Handle Emacs events"""
        try:
            event_type = event.event_type
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    await handler(event)
        except Exception as e:
            logger.error(f"Error handling event {event.event_type}: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get Emacs plugin status"""
        return {
            **self.get_health_info(),
            "buffers_open": len(self.buffers),
            "windows_open": len(self.windows),
            "frames_open": len(self.frames),
            "current_buffer": self.current_buffer,
            "current_window": self.current_window,
            "current_frame": self.current_frame,
            "minor_modes_active": sum(1 for active in self.minor_modes.values() if active),
            "elisp_functions": len(self.elisp_functions),
            "hooks_registered": sum(len(hook_list) for hook_list in self.hooks.values()),
            "keybindings": len(self.keybindings),
            "communication_status": self.communication.get_connection_status(),
            "completion_stats": self.completion_engine.get_engine_stats(),
            "supported_languages": self.metadata.supported_languages,
            "http_port": self.http_port
        }
    
    # Command Handlers - Buffer Operations
    
    async def _handle_list_buffers(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list buffers command"""
        try:
            buffer_list = []
            for name, buffer in self.buffers.items():
                buffer_info = {
                    "name": name,
                    "file_name": buffer.file_name,
                    "major_mode": buffer.major_mode,
                    "modified": buffer.modified,
                    "read_only": buffer.read_only,
                    "point": buffer.point,
                    "size": len(buffer.content)
                }
                buffer_list.append(buffer_info)
            
            return {"buffers": buffer_list}
            
        except Exception as e:
            logger.error(f"Error listing buffers: {e}")
            return {"buffers": [], "error": str(e)}
    
    async def _handle_switch_buffer(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle switch buffer command"""
        try:
            buffer_name = params.get("buffer_name")
            
            if buffer_name in self.buffers:
                self.current_buffer = buffer_name
                await self.emit_event("buffer_switched", {"buffer_name": buffer_name})
                return {"success": True, "current_buffer": buffer_name}
            else:
                return {"success": False, "error": "Buffer not found"}
                
        except Exception as e:
            logger.error(f"Error switching buffer: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_create_buffer(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle create buffer command"""
        try:
            buffer_name = params.get("buffer_name")
            file_name = params.get("file_name")
            major_mode = params.get("major_mode", "text-mode")
            
            if buffer_name:
                buffer = EmacsBuffer(
                    name=buffer_name,
                    file_name=file_name,
                    major_mode=major_mode,
                    content=""
                )
                self.buffers[buffer_name] = buffer
                self.current_buffer = buffer_name
                
                return {"success": True, "buffer_name": buffer_name}
            else:
                return {"success": False, "error": "Buffer name required"}
                
        except Exception as e:
            logger.error(f"Error creating buffer: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_save_buffer(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle save buffer command"""
        try:
            buffer_name = params.get("buffer_name", self.current_buffer)
            
            if buffer_name and buffer_name in self.buffers:
                buffer = self.buffers[buffer_name]
                # In real implementation, would save to file
                buffer.modified = False
                await self.emit_event("buffer_saved", {"buffer_name": buffer_name})
                return {"success": True, "file_name": buffer.file_name}
            else:
                return {"success": False, "error": "Buffer not found"}
                
        except Exception as e:
            logger.error(f"Error saving buffer: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_kill_buffer(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle kill buffer command"""
        try:
            buffer_name = params.get("buffer_name", self.current_buffer)
            
            if buffer_name and buffer_name in self.buffers:
                del self.buffers[buffer_name]
                if self.current_buffer == buffer_name:
                    self.current_buffer = None
                await self.emit_event("buffer_killed", {"buffer_name": buffer_name})
                return {"success": True}
            else:
                return {"success": False, "error": "Buffer not found"}
                
        except Exception as e:
            logger.error(f"Error killing buffer: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_rename_buffer(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle rename buffer command"""
        try:
            old_name = params.get("old_name", self.current_buffer)
            new_name = params.get("new_name")
            
            if old_name in self.buffers and new_name:
                buffer = self.buffers[old_name]
                buffer.name = new_name
                self.buffers[new_name] = buffer
                del self.buffers[old_name]
                
                if self.current_buffer == old_name:
                    self.current_buffer = new_name
                
                return {"success": True, "new_name": new_name}
            else:
                return {"success": False, "error": "Buffer not found or new name required"}
                
        except Exception as e:
            logger.error(f"Error renaming buffer: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_get_buffer_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get buffer content command"""
        try:
            buffer_name = params.get("buffer_name", self.current_buffer)
            
            if buffer_name and buffer_name in self.buffers:
                buffer = self.buffers[buffer_name]
                return {
                    "content": buffer.content,
                    "point": buffer.point,
                    "mark": buffer.mark,
                    "modified": buffer.modified
                }
            else:
                return {"error": "Buffer not found"}
                
        except Exception as e:
            logger.error(f"Error getting buffer content: {e}")
            return {"error": str(e)}
    
    async def _handle_set_buffer_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle set buffer content command"""
        try:
            buffer_name = params.get("buffer_name", self.current_buffer)
            content = params.get("content", "")
            
            if buffer_name and buffer_name in self.buffers:
                buffer = self.buffers[buffer_name]
                buffer.content = content
                buffer.modified = True
                buffer.point = min(buffer.point, len(content))
                
                return {"success": True, "length": len(content)}
            else:
                return {"success": False, "error": "Buffer not found"}
                
        except Exception as e:
            logger.error(f"Error setting buffer content: {e}")
            return {"success": False, "error": str(e)}
    
    # Point and Mark Operations
    
    async def _handle_get_point(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get point command"""
        try:
            buffer_name = params.get("buffer_name", self.current_buffer)
            
            if buffer_name and buffer_name in self.buffers:
                buffer = self.buffers[buffer_name]
                line, column = self._point_to_line_column(buffer.point, buffer.content)
                return {"point": buffer.point, "line": line, "column": column}
            else:
                return {"error": "Buffer not found"}
                
        except Exception as e:
            logger.error(f"Error getting point: {e}")
            return {"error": str(e)}
    
    async def _handle_set_point(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle set point command"""
        try:
            buffer_name = params.get("buffer_name", self.current_buffer)
            point = params.get("point")
            line = params.get("line")
            column = params.get("column")
            
            if buffer_name and buffer_name in self.buffers:
                buffer = self.buffers[buffer_name]
                
                if point is not None:
                    buffer.point = min(max(0, point), len(buffer.content))
                elif line is not None and column is not None:
                    buffer.point = self._line_column_to_point(line, column, buffer.content)
                
                return {"success": True, "point": buffer.point}
            else:
                return {"success": False, "error": "Buffer not found"}
                
        except Exception as e:
            logger.error(f"Error setting point: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_move_point(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle move point command"""
        try:
            buffer_name = params.get("buffer_name", self.current_buffer)
            offset = params.get("offset", 0)
            
            if buffer_name and buffer_name in self.buffers:
                buffer = self.buffers[buffer_name]
                new_point = min(max(0, buffer.point + offset), len(buffer.content))
                buffer.point = new_point
                
                return {"success": True, "point": new_point}
            else:
                return {"success": False, "error": "Buffer not found"}
                
        except Exception as e:
            logger.error(f"Error moving point: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_set_mark(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle set mark command"""
        try:
            buffer_name = params.get("buffer_name", self.current_buffer)
            mark_point = params.get("point")
            
            if buffer_name and buffer_name in self.buffers:
                buffer = self.buffers[buffer_name]
                buffer.mark = mark_point if mark_point is not None else buffer.point
                
                return {"success": True, "mark": buffer.mark}
            else:
                return {"success": False, "error": "Buffer not found"}
                
        except Exception as e:
            logger.error(f"Error setting mark: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_get_mark(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get mark command"""
        try:
            buffer_name = params.get("buffer_name", self.current_buffer)
            
            if buffer_name and buffer_name in self.buffers:
                buffer = self.buffers[buffer_name]
                return {"mark": buffer.mark}
            else:
                return {"error": "Buffer not found"}
                
        except Exception as e:
            logger.error(f"Error getting mark: {e}")
            return {"error": str(e)}
    
    async def _handle_get_region(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get region command"""
        try:
            buffer_name = params.get("buffer_name", self.current_buffer)
            
            if buffer_name and buffer_name in self.buffers:
                buffer = self.buffers[buffer_name]
                
                if buffer.mark is not None:
                    start = min(buffer.point, buffer.mark)
                    end = max(buffer.point, buffer.mark)
                    region_text = buffer.content[start:end]
                    
                    return {
                        "text": region_text,
                        "start": start,
                        "end": end,
                        "length": end - start
                    }
                else:
                    return {"error": "No mark set"}
            else:
                return {"error": "Buffer not found"}
                
        except Exception as e:
            logger.error(f"Error getting region: {e}")
            return {"error": str(e)}
    
    async def _handle_replace_region(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle replace region command"""
        try:
            buffer_name = params.get("buffer_name", self.current_buffer)
            text = params.get("text", "")
            
            if buffer_name and buffer_name in self.buffers:
                buffer = self.buffers[buffer_name]
                
                if buffer.mark is not None:
                    start = min(buffer.point, buffer.mark)
                    end = max(buffer.point, buffer.mark)
                    
                    new_content = buffer.content[:start] + text + buffer.content[end:]
                    buffer.content = new_content
                    buffer.point = start + len(text)
                    buffer.mark = None
                    buffer.modified = True
                    
                    return {"success": True, "new_point": buffer.point}
                else:
                    return {"success": False, "error": "No mark set"}
            else:
                return {"success": False, "error": "Buffer not found"}
                
        except Exception as e:
            logger.error(f"Error replacing region: {e}")
            return {"success": False, "error": str(e)}
    
    # Text Operations
    
    async def _handle_insert_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle insert text command"""
        try:
            buffer_name = params.get("buffer_name", self.current_buffer)
            text = params.get("text", "")
            
            if buffer_name and buffer_name in self.buffers:
                buffer = self.buffers[buffer_name]
                
                new_content = buffer.content[:buffer.point] + text + buffer.content[buffer.point:]
                buffer.content = new_content
                buffer.point += len(text)
                buffer.modified = True
                
                return {"success": True, "new_point": buffer.point}
            else:
                return {"success": False, "error": "Buffer not found"}
                
        except Exception as e:
            logger.error(f"Error inserting text: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_delete_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle delete text command"""
        try:
            buffer_name = params.get("buffer_name", self.current_buffer)
            start = params.get("start")
            end = params.get("end")
            count = params.get("count", 1)
            
            if buffer_name and buffer_name in self.buffers:
                buffer = self.buffers[buffer_name]
                
                if start is not None and end is not None:
                    # Delete range
                    buffer.content = buffer.content[:start] + buffer.content[end:]
                    buffer.point = start
                else:
                    # Delete count characters from point
                    delete_end = min(buffer.point + count, len(buffer.content))
                    buffer.content = buffer.content[:buffer.point] + buffer.content[delete_end:]
                
                buffer.modified = True
                return {"success": True, "new_point": buffer.point}
            else:
                return {"success": False, "error": "Buffer not found"}
                
        except Exception as e:
            logger.error(f"Error deleting text: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_search_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle search text command"""
        try:
            buffer_name = params.get("buffer_name", self.current_buffer)
            pattern = params.get("pattern", "")
            forward = params.get("forward", True)
            regexp = params.get("regexp", False)
            
            if buffer_name and buffer_name in self.buffers:
                buffer = self.buffers[buffer_name]
                
                if regexp:
                    matches = []
                    import re
                    for match in re.finditer(pattern, buffer.content):
                        matches.append({
                            "start": match.start(),
                            "end": match.end(),
                            "text": match.group()
                        })
                else:
                    matches = []
                    start = 0
                    while True:
                        pos = buffer.content.find(pattern, start)
                        if pos == -1:
                            break
                        matches.append({
                            "start": pos,
                            "end": pos + len(pattern),
                            "text": pattern
                        })
                        start = pos + 1
                
                return {"matches": matches}
            else:
                return {"matches": [], "error": "Buffer not found"}
                
        except Exception as e:
            logger.error(f"Error searching text: {e}")
            return {"matches": [], "error": str(e)}
    
    async def _handle_replace_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle replace text command"""
        try:
            buffer_name = params.get("buffer_name", self.current_buffer)
            old_text = params.get("old_text", "")
            new_text = params.get("new_text", "")
            regexp = params.get("regexp", False)
            
            if buffer_name and buffer_name in self.buffers:
                buffer = self.buffers[buffer_name]
                
                if regexp:
                    import re
                    new_content, count = re.subn(old_text, new_text, buffer.content)
                else:
                    new_content = buffer.content.replace(old_text, new_text)
                    count = buffer.content.count(old_text)
                
                buffer.content = new_content
                buffer.modified = True
                
                return {"success": True, "replacements": count}
            else:
                return {"success": False, "error": "Buffer not found"}
                
        except Exception as e:
            logger.error(f"Error replacing text: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_indent_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle indent text command"""
        try:
            buffer_name = params.get("buffer_name", self.current_buffer)
            start = params.get("start")
            end = params.get("end")
            
            if buffer_name and buffer_name in self.buffers:
                # Mock indentation
                return {"success": True, "indented": True}
            else:
                return {"success": False, "error": "Buffer not found"}
                
        except Exception as e:
            logger.error(f"Error indenting text: {e}")
            return {"success": False, "error": str(e)}
    
    # Completion Operations
    
    async def _handle_completion_at_point(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle completion at point"""
        try:
            buffer_name = params.get("buffer_name", self.current_buffer)
            
            if buffer_name and buffer_name in self.buffers:
                buffer = self.buffers[buffer_name]
                
                # Extract context around point
                content = buffer.content
                point = buffer.point
                
                # Find word boundaries
                start = point
                while start > 0 and content[start-1].isalnum():
                    start -= 1
                
                prefix = content[start:point]
                
                # Create completion context
                lines = content.split('\n')
                line_start = content.rfind('\n', 0, point) + 1
                line_num = content[:point].count('\n')
                col_num = point - line_start
                
                code_context = CodeContext(
                    file_path=buffer.file_name or buffer.name,
                    language=self._major_mode_to_language(buffer.major_mode),
                    line_number=line_num,
                    column_number=col_num,
                    current_line=lines[line_num] if line_num < len(lines) else "",
                    preceding_lines=lines[:line_num],
                    following_lines=lines[line_num+1:],
                    prefix=prefix
                )
                
                completion_request = CompletionRequest(
                    context=code_context,
                    trigger=CompletionTrigger.EXPLICIT_INVOCATION,
                    max_completions=30
                )
                
                response = await self.completion_engine.get_completions(completion_request)
                
                # Convert to Emacs format
                completions = []
                for item in response.items:
                    completion = {
                        "completion": item.insert_text or item.label,
                        "annotation": item.detail,
                        "kind": self._completion_kind_to_emacs(item.kind),
                        "company-doc-buffer": item.documentation
                    }
                    completions.append(completion)
                
                return {
                    "completions": completions,
                    "prefix": prefix,
                    "start": start,
                    "end": point
                }
            
            return {"completions": []}
            
        except Exception as e:
            logger.error(f"Error in completion at point: {e}")
            return {"completions": []}
    
    async def _handle_completion_capf(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle completion-at-point-functions"""
        return await self._handle_completion_at_point(params)
    
    async def _handle_company_completion(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle company-mode completion"""
        action = params.get("action", "candidates")
        
        if action == "candidates":
            return await self._handle_completion_at_point(params)
        elif action == "doc-buffer":
            candidate = params.get("candidate", "")
            return {"doc": f"Documentation for {candidate}"}
        elif action == "location":
            candidate = params.get("candidate", "")
            return {"location": None}  # Would return actual location
        else:
            return {"result": None}
    
    # Navigation
    
    async def _handle_goto_definition(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle goto definition"""
        try:
            # Mock definition location
            return {
                "found": True,
                "file": "/path/to/definition.py",
                "line": 42,
                "column": 10
            }
            
        except Exception as e:
            logger.error(f"Error going to definition: {e}")
            return {"found": False, "error": str(e)}
    
    async def _handle_find_references(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle find references"""
        try:
            # Mock references
            return {
                "references": [
                    {"file": "/path/to/ref1.py", "line": 15, "column": 5},
                    {"file": "/path/to/ref2.py", "line": 23, "column": 10}
                ]
            }
            
        except Exception as e:
            logger.error(f"Error finding references: {e}")
            return {"references": []}
    
    async def _handle_goto_symbol(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle goto symbol"""
        try:
            symbol = params.get("symbol", "")
            
            return {
                "found": True,
                "line": 100,
                "column": 1
            }
            
        except Exception as e:
            logger.error(f"Error going to symbol: {e}")
            return {"found": False, "error": str(e)}
    
    async def _handle_imenu(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle imenu"""
        try:
            buffer_name = params.get("buffer_name", self.current_buffer)
            
            # Mock imenu items
            return {
                "items": [
                    {"name": "function1", "line": 10, "type": "function"},
                    {"name": "class1", "line": 25, "type": "class"},
                    {"name": "variable1", "line": 5, "type": "variable"}
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting imenu: {e}")
            return {"items": []}
    
    # Event Handlers
    
    async def _on_buffer_list_update(self, event: PluginEvent) -> None:
        """Handle buffer list update event"""
        pass
    
    async def _on_buffer_modified(self, event: PluginEvent) -> None:
        """Handle buffer modified event"""
        buffer_name = event.data.get("buffer_name")
        if buffer_name in self.buffers:
            self.buffers[buffer_name].modified = True
    
    async def _on_buffer_saved(self, event: PluginEvent) -> None:
        """Handle buffer saved event"""
        buffer_name = event.data.get("buffer_name")
        if buffer_name in self.buffers:
            self.buffers[buffer_name].modified = False
    
    async def _on_window_configuration_change(self, event: PluginEvent) -> None:
        """Handle window configuration change"""
        pass
    
    async def _on_pre_command(self, event: PluginEvent) -> None:
        """Handle pre-command hook"""
        pass
    
    async def _on_post_command(self, event: PluginEvent) -> None:
        """Handle post-command hook"""
        pass
    
    async def _on_find_file(self, event: PluginEvent) -> None:
        """Handle find file event"""
        file_name = event.data.get("file_name")
        logger.info(f"File opened: {file_name}")
    
    async def _on_kill_emacs(self, event: PluginEvent) -> None:
        """Handle kill emacs event"""
        logger.info("Emacs is closing")
    
    async def _on_focus_in(self, event: PluginEvent) -> None:
        """Handle focus in event"""
        pass
    
    async def _on_focus_out(self, event: PluginEvent) -> None:
        """Handle focus out event"""
        pass
    
    # Helper Methods
    
    async def _initialize_completion_engine(self) -> None:
        """Initialize completion engine"""
        pass
    
    async def _setup_emacs_integration(self) -> None:
        """Setup Emacs integration"""
        pass
    
    def _create_default_frame(self) -> None:
        """Create default frame"""
        frame = EmacsFrame(
            name="default",
            windows=[],
            width=80,
            height=40
        )
        self.frames["default"] = frame
        self.current_frame = "default"
    
    def _point_to_line_column(self, point: int, content: str) -> Tuple[int, int]:
        """Convert point to line and column"""
        lines = content[:point].split('\n')
        line = len(lines) - 1
        column = len(lines[-1])
        return line + 1, column + 1  # Emacs uses 1-based indexing
    
    def _line_column_to_point(self, line: int, column: int, content: str) -> int:
        """Convert line and column to point"""
        lines = content.split('\n')
        if line <= 0 or line > len(lines):
            return 0
        
        point = sum(len(lines[i]) + 1 for i in range(line - 1))  # +1 for newlines
        point += min(column - 1, len(lines[line - 1]))
        return min(point, len(content))
    
    def _major_mode_to_language(self, major_mode: str) -> str:
        """Convert Emacs major mode to language"""
        mode_map = {
            "python-mode": "python",
            "js-mode": "javascript",
            "js2-mode": "javascript", 
            "typescript-mode": "typescript",
            "c-mode": "c",
            "c++-mode": "cpp",
            "java-mode": "java",
            "go-mode": "go",
            "rust-mode": "rust",
            "haskell-mode": "haskell",
            "emacs-lisp-mode": "emacs-lisp",
            "lisp-mode": "lisp"
        }
        return mode_map.get(major_mode, "text")
    
    def _completion_kind_to_emacs(self, kind) -> str:
        """Convert completion kind to Emacs format"""
        kind_map = {
            "variable": "variable",
            "function": "function",
            "method": "method",
            "class": "class",
            "module": "module", 
            "keyword": "keyword",
            "snippet": "snippet"
        }
        kind_str = kind.value if hasattr(kind, 'value') else str(kind)
        return kind_map.get(kind_str, "unknown")
    
    # Additional stub handlers for remaining commands
    async def _handle_list_windows(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"windows": list(self.windows.keys())}
    
    async def _handle_select_window(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True}
    
    async def _handle_split_window(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True, "new_window": 2}
    
    async def _handle_delete_window(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True}
    
    async def _handle_window_configuration(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"configuration": "mock_config"}
    
    async def _handle_eval_expression(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": "evaluated"}
    
    async def _handle_eval_buffer(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True}
    
    async def _handle_eval_region(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True}
    
    async def _handle_start_repl(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"repl_started": True}
    
    async def _handle_set_major_mode(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True}
    
    async def _handle_toggle_minor_mode(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True}
    
    async def _handle_list_minor_modes(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"modes": list(self.minor_modes.keys())}
    
    async def _handle_define_key(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True}
    
    async def _handle_lookup_key(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"command": "mock-command"}
    
    async def _handle_describe_key(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"description": "Mock key description"}
    
    async def _handle_add_hook(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True}
    
    async def _handle_remove_hook(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True}
    
    async def _handle_run_hooks(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True}
    
    # Archon-specific handlers
    async def _handle_archon_analyze(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"analysis": {"complexity": 5, "issues": []}}
    
    async def _handle_archon_complete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return await self._handle_completion_at_point(params)
    
    async def _handle_archon_refactor(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True, "changes": []}
    
    async def _handle_archon_generate_tests(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"tests": "mock test code"}
    
    async def _handle_archon_format(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True}
    
    async def _handle_archon_lint(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"issues": []}
    
    async def _handle_archon_git_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "clean"}
    
    async def _handle_archon_project_browser(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"files": ["file1.py", "file2.py"]}
    
    async def _handle_archon_flycheck(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"errors": [], "warnings": []}