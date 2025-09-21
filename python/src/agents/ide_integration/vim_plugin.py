"""
Vim Plugin
Comprehensive Vim/Neovim integration for Archon AI development platform
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import logging
import socket
import threading
from pathlib import Path

from .plugin_base import (
    PluginBase, PluginMetadata, PluginConfiguration, PluginCapability, 
    PluginCommand, PluginResponse, PluginEvent, PluginStatus,
    CommunicationProtocol
)
from .ide_communication import IDECommunicationProtocol, WebSocketChannel, ConnectionConfig
from .code_completion_engine import CodeCompletionEngine, CompletionRequest, CodeContext, CompletionTrigger

logger = logging.getLogger(__name__)


@dataclass
class VimBuffer:
    """Vim buffer representation"""
    number: int
    name: str
    filetype: str
    lines: List[str]
    modified: bool = False
    readonly: bool = False
    hidden: bool = False


@dataclass
class VimWindow:
    """Vim window representation"""
    number: int
    buffer: VimBuffer
    cursor: tuple  # (line, column)
    width: int
    height: int


@dataclass
class VimMode:
    """Vim mode information"""
    mode: str  # 'n', 'i', 'v', 'c', etc.
    blocking: bool = False
    mode_idx: int = 0


class VimPlugin(PluginBase):
    """Vim/Neovim plugin implementation"""
    
    def __init__(self, config: Optional[PluginConfiguration] = None):
        metadata = PluginMetadata(
            name="archon-vim",
            version="2.0.0",
            description="Archon AI Development Platform - Vim/Neovim Integration",
            author="Archon Development Team",
            supported_languages=["python", "javascript", "typescript", "go", "rust", "c", "cpp", "java", "lua", "vim"],
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
                PluginCapability.CUSTOM_COMMANDS
            ],
            protocols=[CommunicationProtocol.WEBSOCKET, CommunicationProtocol.JSON_RPC, CommunicationProtocol.CUSTOM],
            min_ide_version="8.0",  # Vim 8.0 or Neovim 0.5+
            installation_url="https://github.com/archon-ai/archon-vim",
            documentation_url="https://docs.archon.dev/vim-plugin"
        )
        
        super().__init__(metadata, config)
        
        # Vim-specific components
        self.communication = IDECommunicationProtocol()
        self.completion_engine = CodeCompletionEngine()
        self.buffers: Dict[int, VimBuffer] = {}
        self.windows: Dict[int, VimWindow] = {}
        self.current_buffer: Optional[int] = None
        self.current_window: Optional[int] = None
        self.vim_mode = VimMode(mode="n")
        self.websocket_port = 8056
        self.nvim_socket = None
        self.is_neovim = False
        
        # Vim command mappings
        self.vim_commands: Dict[str, str] = {}
        self.autocommands: Dict[str, List[str]] = {}
        
        # Register command handlers
        self._register_command_handlers()
        
        # Register event handlers
        self._register_event_handlers()
    
    def _register_command_handlers(self) -> None:
        """Register Vim command handlers"""
        commands = {
            # Vim basic operations
            "buffer/list": self._handle_list_buffers,
            "buffer/switch": self._handle_switch_buffer,
            "buffer/save": self._handle_save_buffer,
            "buffer/close": self._handle_close_buffer,
            "window/list": self._handle_list_windows,
            "window/split": self._handle_split_window,
            "cursor/move": self._handle_move_cursor,
            "cursor/position": self._handle_get_cursor_position,
            "mode/get": self._handle_get_mode,
            "mode/set": self._handle_set_mode,
            
            # Completion and suggestions
            "completion/omnifunc": self._handle_omnifunc_completion,
            "completion/insert": self._handle_insert_completion,
            "completion/user": self._handle_user_completion,
            
            # Text manipulation
            "text/insert": self._handle_insert_text,
            "text/replace": self._handle_replace_text,
            "text/delete": self._handle_delete_text,
            "text/search": self._handle_search_text,
            "text/substitute": self._handle_substitute_text,
            
            # Navigation
            "navigate/definition": self._handle_goto_definition,
            "navigate/references": self._handle_find_references,
            "navigate/symbol": self._handle_goto_symbol,
            
            # Vim-specific commands
            "command/execute": self._handle_execute_command,
            "mapping/create": self._handle_create_mapping,
            "autocommand/create": self._handle_create_autocommand,
            "quickfix/populate": self._handle_populate_quickfix,
            "location/populate": self._handle_populate_location_list,
            
            # Archon-specific commands
            "archon/analyze": self._handle_archon_analyze,
            "archon/complete": self._handle_archon_complete,
            "archon/refactor": self._handle_archon_refactor,
            "archon/generate_tests": self._handle_archon_generate_tests,
            "archon/format": self._handle_archon_format,
            "archon/lint": self._handle_archon_lint,
            "archon/git_status": self._handle_archon_git_status,
            "archon/project_tree": self._handle_archon_project_tree
        }
        
        for command, handler in commands.items():
            self.register_command_handler(command, handler)
    
    def _register_event_handlers(self) -> None:
        """Register Vim event handlers"""
        events = {
            "buffer_enter": self._on_buffer_enter,
            "buffer_leave": self._on_buffer_leave,
            "buffer_write": self._on_buffer_write,
            "cursor_moved": self._on_cursor_moved,
            "insert_enter": self._on_insert_enter,
            "insert_leave": self._on_insert_leave,
            "mode_changed": self._on_mode_changed,
            "vim_leave": self._on_vim_leave,
            "text_changed": self._on_text_changed
        }
        
        for event, handler in events.items():
            self.register_event_handler(event, handler)
    
    async def initialize(self) -> bool:
        """Initialize Vim plugin"""
        try:
            logger.info("Initializing Vim plugin...")
            
            # Detect Vim or Neovim
            self.is_neovim = await self._detect_neovim()
            
            # Setup communication
            await self._setup_communication()
            
            # Initialize completion engine
            await self._initialize_completion_engine()
            
            # Setup Vim integration
            await self._setup_vim_integration()
            
            logger.info(f"Vim plugin initialized for {'Neovim' if self.is_neovim else 'Vim'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Vim plugin: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown Vim plugin"""
        try:
            logger.info("Shutting down Vim plugin...")
            
            # Disconnect communication channels
            for ide_id in list(self.communication._active_connections.keys()):
                await self.communication.disconnect_from_ide(ide_id)
            
            # Close Neovim socket if open
            if self.nvim_socket:
                self.nvim_socket.close()
                self.nvim_socket = None
            
            # Clear state
            self.buffers.clear()
            self.windows.clear()
            
            logger.info("Vim plugin shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Error shutting down Vim plugin: {e}")
            return False
    
    async def execute_command(self, command: PluginCommand) -> PluginResponse:
        """Execute Vim plugin command"""
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
        """Handle Vim events"""
        try:
            event_type = event.event_type
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    await handler(event)
        except Exception as e:
            logger.error(f"Error handling event {event.event_type}: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get Vim plugin status"""
        return {
            **self.get_health_info(),
            "vim_type": "neovim" if self.is_neovim else "vim",
            "buffers_open": len(self.buffers),
            "windows_open": len(self.windows),
            "current_buffer": self.current_buffer,
            "current_window": self.current_window,
            "mode": self.vim_mode.mode,
            "communication_status": self.communication.get_connection_status(),
            "completion_stats": self.completion_engine.get_engine_stats(),
            "supported_languages": self.metadata.supported_languages,
            "websocket_port": self.websocket_port,
            "custom_commands": len(self.vim_commands),
            "autocommands": len(self.autocommands)
        }
    
    # Command Handlers
    
    async def _handle_list_buffers(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list buffers command"""
        try:
            buffer_list = []
            for buf_num, buffer in self.buffers.items():
                buffer_info = {
                    "number": buf_num,
                    "name": buffer.name,
                    "filetype": buffer.filetype,
                    "modified": buffer.modified,
                    "readonly": buffer.readonly,
                    "hidden": buffer.hidden,
                    "line_count": len(buffer.lines)
                }
                buffer_list.append(buffer_info)
            
            return {"buffers": buffer_list}
            
        except Exception as e:
            logger.error(f"Error listing buffers: {e}")
            return {"buffers": [], "error": str(e)}
    
    async def _handle_switch_buffer(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle switch buffer command"""
        try:
            buffer_num = params.get("buffer_number")
            
            if buffer_num in self.buffers:
                self.current_buffer = buffer_num
                await self.emit_event("buffer_switched", {"buffer_number": buffer_num})
                return {"success": True, "current_buffer": buffer_num}
            else:
                return {"success": False, "error": "Buffer not found"}
                
        except Exception as e:
            logger.error(f"Error switching buffer: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_save_buffer(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle save buffer command"""
        try:
            buffer_num = params.get("buffer_number", self.current_buffer)
            
            if buffer_num in self.buffers:
                buffer = self.buffers[buffer_num]
                # In real implementation, would write to file
                buffer.modified = False
                await self.emit_event("buffer_saved", {"buffer_number": buffer_num})
                return {"success": True, "buffer_name": buffer.name}
            else:
                return {"success": False, "error": "Buffer not found"}
                
        except Exception as e:
            logger.error(f"Error saving buffer: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_close_buffer(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle close buffer command"""
        try:
            buffer_num = params.get("buffer_number", self.current_buffer)
            
            if buffer_num in self.buffers:
                del self.buffers[buffer_num]
                if self.current_buffer == buffer_num:
                    self.current_buffer = None
                await self.emit_event("buffer_closed", {"buffer_number": buffer_num})
                return {"success": True}
            else:
                return {"success": False, "error": "Buffer not found"}
                
        except Exception as e:
            logger.error(f"Error closing buffer: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_list_windows(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list windows command"""
        try:
            window_list = []
            for win_num, window in self.windows.items():
                window_info = {
                    "number": win_num,
                    "buffer_number": window.buffer.number,
                    "cursor": window.cursor,
                    "width": window.width,
                    "height": window.height
                }
                window_list.append(window_info)
            
            return {"windows": window_list}
            
        except Exception as e:
            logger.error(f"Error listing windows: {e}")
            return {"windows": [], "error": str(e)}
    
    async def _handle_split_window(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle split window command"""
        try:
            direction = params.get("direction", "horizontal")  # horizontal, vertical
            
            # Mock window splitting
            new_win_num = max(self.windows.keys()) + 1 if self.windows else 1
            
            # Create mock window
            if self.current_buffer and self.current_buffer in self.buffers:
                buffer = self.buffers[self.current_buffer]
                new_window = VimWindow(
                    number=new_win_num,
                    buffer=buffer,
                    cursor=(1, 1),
                    width=80 if direction == "vertical" else 160,
                    height=40 if direction == "horizontal" else 20
                )
                self.windows[new_win_num] = new_window
                self.current_window = new_win_num
            
            return {"success": True, "window_number": new_win_num}
            
        except Exception as e:
            logger.error(f"Error splitting window: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_move_cursor(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle move cursor command"""
        try:
            line = params.get("line")
            column = params.get("column")
            
            if self.current_window and self.current_window in self.windows:
                window = self.windows[self.current_window]
                window.cursor = (line, column)
                await self.emit_event("cursor_moved", {"line": line, "column": column})
                return {"success": True, "position": window.cursor}
            else:
                return {"success": False, "error": "No current window"}
                
        except Exception as e:
            logger.error(f"Error moving cursor: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_get_cursor_position(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get cursor position command"""
        try:
            if self.current_window and self.current_window in self.windows:
                window = self.windows[self.current_window]
                return {"line": window.cursor[0], "column": window.cursor[1]}
            else:
                return {"error": "No current window"}
                
        except Exception as e:
            logger.error(f"Error getting cursor position: {e}")
            return {"error": str(e)}
    
    async def _handle_get_mode(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get mode command"""
        return {
            "mode": self.vim_mode.mode,
            "blocking": self.vim_mode.blocking,
            "mode_idx": self.vim_mode.mode_idx
        }
    
    async def _handle_set_mode(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle set mode command"""
        try:
            new_mode = params.get("mode")
            if new_mode:
                self.vim_mode.mode = new_mode
                await self.emit_event("mode_changed", {"mode": new_mode})
                return {"success": True, "mode": new_mode}
            else:
                return {"success": False, "error": "Mode required"}
                
        except Exception as e:
            logger.error(f"Error setting mode: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_omnifunc_completion(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle omnifunc completion"""
        try:
            findstart = params.get("findstart", 0)
            base = params.get("base", "")
            
            if findstart:
                # Return start column for completion
                return {"start_col": 0}
            else:
                # Return completions
                completions = await self._get_vim_completions(base)
                return {"completions": completions}
                
        except Exception as e:
            logger.error(f"Error in omnifunc completion: {e}")
            return {"completions": []}
    
    async def _handle_insert_completion(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle insert mode completion"""
        try:
            prefix = params.get("prefix", "")
            line = params.get("line", 1)
            column = params.get("column", 1)
            
            # Create completion context
            if self.current_buffer and self.current_buffer in self.buffers:
                buffer = self.buffers[self.current_buffer]
                
                code_context = CodeContext(
                    file_path=buffer.name,
                    language=self._filetype_to_language(buffer.filetype),
                    line_number=line - 1,  # Vim uses 1-based indexing
                    column_number=column - 1,
                    current_line=buffer.lines[line - 1] if line <= len(buffer.lines) else "",
                    preceding_lines=buffer.lines[:line-1],
                    following_lines=buffer.lines[line:],
                    prefix=prefix
                )
                
                completion_request = CompletionRequest(
                    context=code_context,
                    trigger=CompletionTrigger.CHARACTER_TYPED,
                    max_completions=20
                )
                
                response = await self.completion_engine.get_completions(completion_request)
                
                # Convert to Vim format
                vim_completions = []
                for item in response.items:
                    vim_completion = {
                        "word": item.insert_text or item.label,
                        "abbr": item.label,
                        "menu": item.detail,
                        "info": item.documentation,
                        "kind": self._completion_kind_to_vim(item.kind),
                        "icase": 1,
                        "dup": 0
                    }
                    vim_completions.append(vim_completion)
                
                return {"completions": vim_completions}
            
            return {"completions": []}
            
        except Exception as e:
            logger.error(f"Error in insert completion: {e}")
            return {"completions": []}
    
    async def _handle_user_completion(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user-defined completion"""
        return await self._handle_insert_completion(params)
    
    async def _handle_insert_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle insert text command"""
        try:
            text = params.get("text", "")
            line = params.get("line")
            column = params.get("column")
            
            if self.current_buffer and self.current_buffer in self.buffers:
                buffer = self.buffers[self.current_buffer]
                
                if line is None or column is None:
                    # Insert at cursor
                    if self.current_window and self.current_window in self.windows:
                        window = self.windows[self.current_window]
                        line, column = window.cursor
                
                # Insert text (simplified implementation)
                if line <= len(buffer.lines):
                    current_line = buffer.lines[line - 1]
                    new_line = current_line[:column-1] + text + current_line[column-1:]
                    buffer.lines[line - 1] = new_line
                    buffer.modified = True
                
                return {"success": True}
            
            return {"success": False, "error": "No current buffer"}
            
        except Exception as e:
            logger.error(f"Error inserting text: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_replace_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle replace text command"""
        try:
            old_text = params.get("old_text", "")
            new_text = params.get("new_text", "")
            line = params.get("line")
            
            if self.current_buffer and self.current_buffer in self.buffers:
                buffer = self.buffers[self.current_buffer]
                
                if line and line <= len(buffer.lines):
                    buffer.lines[line - 1] = buffer.lines[line - 1].replace(old_text, new_text)
                    buffer.modified = True
                    return {"success": True}
                else:
                    # Replace in all lines
                    for i, line_text in enumerate(buffer.lines):
                        buffer.lines[i] = line_text.replace(old_text, new_text)
                    buffer.modified = True
                    return {"success": True}
            
            return {"success": False, "error": "No current buffer"}
            
        except Exception as e:
            logger.error(f"Error replacing text: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_delete_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle delete text command"""
        try:
            start_line = params.get("start_line")
            end_line = params.get("end_line")
            
            if self.current_buffer and self.current_buffer in self.buffers:
                buffer = self.buffers[self.current_buffer]
                
                if start_line and end_line:
                    # Delete range of lines
                    del buffer.lines[start_line-1:end_line]
                    buffer.modified = True
                    return {"success": True}
            
            return {"success": False, "error": "No current buffer or invalid range"}
            
        except Exception as e:
            logger.error(f"Error deleting text: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_search_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle search text command"""
        try:
            pattern = params.get("pattern", "")
            
            if self.current_buffer and self.current_buffer in self.buffers:
                buffer = self.buffers[self.current_buffer]
                matches = []
                
                for i, line in enumerate(buffer.lines):
                    if pattern in line:
                        matches.append({
                            "line": i + 1,
                            "column": line.find(pattern) + 1,
                            "text": line.strip()
                        })
                
                return {"matches": matches}
            
            return {"matches": []}
            
        except Exception as e:
            logger.error(f"Error searching text: {e}")
            return {"matches": []}
    
    async def _handle_substitute_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle substitute text command"""
        try:
            pattern = params.get("pattern", "")
            replacement = params.get("replacement", "")
            flags = params.get("flags", "")
            
            # Mock substitute implementation
            return {"success": True, "substitutions": 3}
            
        except Exception as e:
            logger.error(f"Error substituting text: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_goto_definition(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle goto definition command"""
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
        """Handle find references command"""
        try:
            # Mock references
            return {
                "references": [
                    {"file": "/path/to/file1.py", "line": 15, "column": 5},
                    {"file": "/path/to/file2.py", "line": 23, "column": 10}
                ]
            }
            
        except Exception as e:
            logger.error(f"Error finding references: {e}")
            return {"references": []}
    
    async def _handle_goto_symbol(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle goto symbol command"""
        try:
            symbol = params.get("symbol", "")
            
            # Mock symbol location
            return {
                "found": True,
                "file": "/path/to/symbol.py",
                "line": 100,
                "column": 1,
                "symbol_type": "function"
            }
            
        except Exception as e:
            logger.error(f"Error going to symbol: {e}")
            return {"found": False, "error": str(e)}
    
    async def _handle_execute_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle execute Vim command"""
        try:
            command = params.get("command", "")
            
            # Mock command execution
            return {"success": True, "output": f"Executed: {command}"}
            
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_create_mapping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle create key mapping"""
        try:
            mode = params.get("mode", "n")
            lhs = params.get("lhs", "")
            rhs = params.get("rhs", "")
            
            mapping_key = f"{mode}:{lhs}"
            self.vim_commands[mapping_key] = rhs
            
            return {"success": True, "mapping": f"{lhs} -> {rhs}"}
            
        except Exception as e:
            logger.error(f"Error creating mapping: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_create_autocommand(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle create autocommand"""
        try:
            event = params.get("event", "")
            pattern = params.get("pattern", "*")
            command = params.get("command", "")
            
            if event not in self.autocommands:
                self.autocommands[event] = []
            
            self.autocommands[event].append(f"{pattern}:{command}")
            
            return {"success": True, "autocommand": f"{event} {pattern} {command}"}
            
        except Exception as e:
            logger.error(f"Error creating autocommand: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_populate_quickfix(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle populate quickfix list"""
        try:
            items = params.get("items", [])
            
            # Mock quickfix population
            return {"success": True, "items_added": len(items)}
            
        except Exception as e:
            logger.error(f"Error populating quickfix: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_populate_location_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle populate location list"""
        try:
            items = params.get("items", [])
            
            # Mock location list population
            return {"success": True, "items_added": len(items)}
            
        except Exception as e:
            logger.error(f"Error populating location list: {e}")
            return {"success": False, "error": str(e)}
    
    # Archon-specific handlers
    
    async def _handle_archon_analyze(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Archon code analysis"""
        try:
            buffer_num = params.get("buffer", self.current_buffer)
            
            if buffer_num and buffer_num in self.buffers:
                buffer = self.buffers[buffer_num]
                
                analysis = {
                    "file": buffer.name,
                    "language": buffer.filetype,
                    "lines": len(buffer.lines),
                    "complexity": 5.5,
                    "issues": [
                        {"line": 10, "type": "warning", "message": "Long line"},
                        {"line": 25, "type": "info", "message": "Consider refactoring"}
                    ]
                }
                
                return {"analysis": analysis}
            
            return {"error": "No buffer to analyze"}
            
        except Exception as e:
            logger.error(f"Error analyzing code: {e}")
            return {"error": str(e)}
    
    async def _handle_archon_complete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Archon intelligent completion"""
        return await self._handle_insert_completion(params)
    
    async def _handle_archon_refactor(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Archon refactoring"""
        try:
            refactor_type = params.get("type", "extract_method")
            
            return {
                "success": True,
                "type": refactor_type,
                "preview": f"Refactoring: {refactor_type}",
                "changes": []
            }
            
        except Exception as e:
            logger.error(f"Error refactoring: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_archon_generate_tests(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle test generation"""
        try:
            function_name = params.get("function", "test_function")
            
            test_code = f"""
def test_{function_name}():
    # Arrange
    
    # Act
    result = {function_name}()
    
    # Assert
    assert result is not None
"""
            
            return {
                "generated_tests": test_code,
                "test_count": 1
            }
            
        except Exception as e:
            logger.error(f"Error generating tests: {e}")
            return {"error": str(e)}
    
    async def _handle_archon_format(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code formatting"""
        try:
            buffer_num = params.get("buffer", self.current_buffer)
            
            if buffer_num and buffer_num in self.buffers:
                # Mock formatting
                return {"success": True, "formatted": True}
            
            return {"success": False, "error": "No buffer to format"}
            
        except Exception as e:
            logger.error(f"Error formatting: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_archon_lint(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code linting"""
        try:
            buffer_num = params.get("buffer", self.current_buffer)
            
            if buffer_num and buffer_num in self.buffers:
                # Mock linting results
                return {
                    "issues": [
                        {"line": 5, "severity": "error", "message": "Syntax error"},
                        {"line": 12, "severity": "warning", "message": "Unused variable"}
                    ]
                }
            
            return {"issues": []}
            
        except Exception as e:
            logger.error(f"Error linting: {e}")
            return {"issues": []}
    
    async def _handle_archon_git_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Git status"""
        try:
            return {
                "branch": "main",
                "status": "clean",
                "modified_files": [],
                "untracked_files": []
            }
            
        except Exception as e:
            logger.error(f"Error getting Git status: {e}")
            return {"error": str(e)}
    
    async def _handle_archon_project_tree(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle project tree"""
        try:
            return {
                "tree": [
                    {"name": "src/", "type": "directory"},
                    {"name": "src/main.py", "type": "file"},
                    {"name": "tests/", "type": "directory"},
                    {"name": "README.md", "type": "file"}
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting project tree: {e}")
            return {"tree": []}
    
    # Event Handlers
    
    async def _on_buffer_enter(self, event: PluginEvent) -> None:
        """Handle buffer enter event"""
        buffer_num = event.data.get("buffer_number")
        self.current_buffer = buffer_num
        logger.debug(f"Entered buffer: {buffer_num}")
    
    async def _on_buffer_leave(self, event: PluginEvent) -> None:
        """Handle buffer leave event"""
        buffer_num = event.data.get("buffer_number")
        logger.debug(f"Left buffer: {buffer_num}")
    
    async def _on_buffer_write(self, event: PluginEvent) -> None:
        """Handle buffer write event"""
        buffer_num = event.data.get("buffer_number")
        if buffer_num in self.buffers:
            self.buffers[buffer_num].modified = False
        logger.debug(f"Buffer written: {buffer_num}")
    
    async def _on_cursor_moved(self, event: PluginEvent) -> None:
        """Handle cursor moved event"""
        line = event.data.get("line")
        column = event.data.get("column")
        logger.debug(f"Cursor moved to: {line},{column}")
    
    async def _on_insert_enter(self, event: PluginEvent) -> None:
        """Handle insert mode enter"""
        self.vim_mode.mode = "i"
        logger.debug("Entered insert mode")
    
    async def _on_insert_leave(self, event: PluginEvent) -> None:
        """Handle insert mode leave"""
        self.vim_mode.mode = "n"
        logger.debug("Left insert mode")
    
    async def _on_mode_changed(self, event: PluginEvent) -> None:
        """Handle mode changed event"""
        mode = event.data.get("mode")
        self.vim_mode.mode = mode
        logger.debug(f"Mode changed to: {mode}")
    
    async def _on_vim_leave(self, event: PluginEvent) -> None:
        """Handle Vim leave event"""
        logger.info("Vim is closing")
    
    async def _on_text_changed(self, event: PluginEvent) -> None:
        """Handle text changed event"""
        buffer_num = event.data.get("buffer_number")
        if buffer_num in self.buffers:
            self.buffers[buffer_num].modified = True
    
    # Helper Methods
    
    async def _detect_neovim(self) -> bool:
        """Detect if running Neovim or Vim"""
        # This would detect Neovim vs Vim in real implementation
        return True  # Assume Neovim for now
    
    async def _setup_communication(self) -> None:
        """Setup communication channels"""
        websocket_config = ConnectionConfig(
            host="localhost",
            port=self.websocket_port,
            timeout_seconds=30
        )
        
        websocket_channel = WebSocketChannel(websocket_config)
        self.communication.register_channel("websocket", websocket_channel)
    
    async def _initialize_completion_engine(self) -> None:
        """Initialize completion engine"""
        pass
    
    async def _setup_vim_integration(self) -> None:
        """Setup Vim/Neovim integration"""
        pass
    
    async def _get_vim_completions(self, base: str) -> List[Dict[str, Any]]:
        """Get completions in Vim format"""
        # Mock completions
        return [
            {"word": "function", "abbr": "function", "menu": "keyword", "kind": "k"},
            {"word": "variable", "abbr": "variable", "menu": "identifier", "kind": "v"}
        ]
    
    def _filetype_to_language(self, filetype: str) -> str:
        """Convert Vim filetype to language"""
        mapping = {
            "python": "python",
            "javascript": "javascript", 
            "typescript": "typescript",
            "java": "java",
            "c": "c",
            "cpp": "cpp",
            "go": "go",
            "rust": "rust",
            "lua": "lua",
            "vim": "vim"
        }
        return mapping.get(filetype, "text")
    
    def _completion_kind_to_vim(self, kind) -> str:
        """Convert completion kind to Vim format"""
        kind_map = {
            "variable": "v",
            "function": "f",
            "method": "m", 
            "class": "c",
            "module": "m",
            "keyword": "k",
            "snippet": "s"
        }
        kind_str = kind.value if hasattr(kind, 'value') else str(kind)
        return kind_map.get(kind_str, "t")