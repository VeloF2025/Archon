"""
Visual Studio Code Plugin
Comprehensive VSCode integration for Archon AI development platform
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import logging
import websockets
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
class VSCodeDocument:
    """VSCode document representation"""
    uri: str
    language_id: str
    version: int
    text: str
    line_count: int
    is_dirty: bool = False
    file_name: Optional[str] = None


@dataclass
class VSCodePosition:
    """Position in VSCode document"""
    line: int
    character: int


@dataclass
class VSCodeRange:
    """Range in VSCode document"""
    start: VSCodePosition
    end: VSCodePosition


@dataclass
class VSCodeTextEdit:
    """Text edit for VSCode"""
    range: VSCodeRange
    new_text: str


@dataclass
class VSCodeDiagnostic:
    """Diagnostic for VSCode"""
    range: VSCodeRange
    message: str
    severity: int  # 1=Error, 2=Warning, 3=Information, 4=Hint
    source: str = "archon"
    code: Optional[str] = None


class VSCodePlugin(PluginBase):
    """VSCode plugin implementation"""
    
    def __init__(self, config: Optional[PluginConfiguration] = None):
        metadata = PluginMetadata(
            name="archon-vscode",
            version="2.0.0",
            description="Archon AI Development Platform - VSCode Integration",
            author="Archon Development Team",
            supported_languages=["python", "javascript", "typescript", "java", "go", "rust", "html", "css"],
            capabilities=[
                PluginCapability.CODE_COMPLETION,
                PluginCapability.SYNTAX_HIGHLIGHTING,
                PluginCapability.ERROR_DETECTION,
                PluginCapability.REFACTORING,
                PluginCapability.DEBUGGING,
                PluginCapability.REAL_TIME_COLLABORATION,
                PluginCapability.INTELLIGENT_SUGGESTIONS,
                PluginCapability.CODE_ANALYSIS,
                PluginCapability.DOCUMENTATION_GENERATION,
                PluginCapability.TEST_GENERATION,
                PluginCapability.SECURITY_ANALYSIS,
                PluginCapability.GIT_INTEGRATION,
                PluginCapability.PROJECT_MANAGEMENT
            ],
            protocols=[CommunicationProtocol.WEBSOCKET, CommunicationProtocol.HTTP, CommunicationProtocol.LSP],
            min_ide_version="1.60.0",
            installation_url="https://marketplace.visualstudio.com/items?itemName=archon.archon-vscode",
            documentation_url="https://docs.archon.dev/vscode-plugin"
        )
        
        super().__init__(metadata, config)
        
        # VSCode-specific components
        self.communication = IDECommunicationProtocol()
        self.completion_engine = CodeCompletionEngine()
        self.documents: Dict[str, VSCodeDocument] = {}
        self.diagnostics: Dict[str, List[VSCodeDiagnostic]] = {}
        self.active_editor: Optional[str] = None
        self.workspace_folders: List[str] = []
        self.language_server_port = 8053
        self.websocket_port = 8054
        
        # Register command handlers
        self._register_command_handlers()
        
        # Register event handlers
        self._register_event_handlers()
    
    def _register_command_handlers(self) -> None:
        """Register VSCode command handlers"""
        commands = {
            "textDocument/completion": self._handle_completion,
            "textDocument/hover": self._handle_hover,
            "textDocument/signatureHelp": self._handle_signature_help,
            "textDocument/definition": self._handle_go_to_definition,
            "textDocument/references": self._handle_find_references,
            "textDocument/rename": self._handle_rename,
            "textDocument/formatting": self._handle_format_document,
            "textDocument/rangeFormatting": self._handle_format_range,
            "textDocument/codeAction": self._handle_code_actions,
            "textDocument/codeLens": self._handle_code_lens,
            "textDocument/documentSymbol": self._handle_document_symbols,
            "workspace/symbol": self._handle_workspace_symbols,
            "textDocument/didOpen": self._handle_document_open,
            "textDocument/didChange": self._handle_document_change,
            "textDocument/didSave": self._handle_document_save,
            "textDocument/didClose": self._handle_document_close,
            "archon/analyzeCode": self._handle_analyze_code,
            "archon/generateTests": self._handle_generate_tests,
            "archon/generateDocs": self._handle_generate_documentation,
            "archon/refactorCode": self._handle_refactor_code,
            "archon/securityScan": self._handle_security_scan,
            "archon/projectStatus": self._handle_project_status,
            "archon/syncCollaboration": self._handle_sync_collaboration
        }
        
        for command, handler in commands.items():
            self.register_command_handler(command, handler)
    
    def _register_event_handlers(self) -> None:
        """Register VSCode event handlers"""
        events = {
            "document_opened": self._on_document_opened,
            "document_changed": self._on_document_changed,
            "document_saved": self._on_document_saved,
            "selection_changed": self._on_selection_changed,
            "workspace_changed": self._on_workspace_changed,
            "configuration_changed": self._on_configuration_changed
        }
        
        for event, handler in events.items():
            self.register_event_handler(event, handler)
    
    async def initialize(self) -> bool:
        """Initialize VSCode plugin"""
        try:
            logger.info("Initializing VSCode plugin...")
            
            # Setup communication channels
            websocket_config = ConnectionConfig(
                host="localhost",
                port=self.websocket_port,
                timeout_seconds=30,
                heartbeat_interval=30.0
            )
            
            websocket_channel = WebSocketChannel(websocket_config)
            self.communication.register_channel("websocket", websocket_channel)
            
            # Initialize completion engine
            await self._initialize_completion_engine()
            
            # Setup language server capabilities
            await self._setup_language_server()
            
            logger.info("VSCode plugin initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize VSCode plugin: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown VSCode plugin"""
        try:
            logger.info("Shutting down VSCode plugin...")
            
            # Disconnect all channels
            for ide_id in list(self.communication._active_connections.keys()):
                await self.communication.disconnect_from_ide(ide_id)
            
            # Clear document cache
            self.documents.clear()
            self.diagnostics.clear()
            
            logger.info("VSCode plugin shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Error shutting down VSCode plugin: {e}")
            return False
    
    async def execute_command(self, command: PluginCommand) -> PluginResponse:
        """Execute VSCode plugin command"""
        start_time = datetime.now()
        
        try:
            # Validate command
            if not await self.validate_command(command):
                return self.create_response(
                    command, False, 
                    error_message="Invalid command",
                    error_code="INVALID_COMMAND"
                )
            
            # Execute command
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
        """Handle VSCode events"""
        try:
            event_type = event.event_type
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    await handler(event)
        except Exception as e:
            logger.error(f"Error handling event {event.event_type}: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get VSCode plugin status"""
        return {
            **self.get_health_info(),
            "documents_open": len(self.documents),
            "active_editor": self.active_editor,
            "workspace_folders": self.workspace_folders,
            "diagnostics_count": sum(len(diag_list) for diag_list in self.diagnostics.values()),
            "communication_status": self.communication.get_connection_status(),
            "completion_stats": self.completion_engine.get_engine_stats(),
            "supported_languages": self.metadata.supported_languages,
            "language_server_port": self.language_server_port,
            "websocket_port": self.websocket_port
        }
    
    # Command Handlers
    
    async def _handle_completion(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code completion request"""
        try:
            text_document = params["textDocument"]
            position = params["position"]
            context = params.get("context", {})
            
            # Get document
            document = self.documents.get(text_document["uri"])
            if not document:
                return {"items": []}
            
            # Create completion context
            lines = document.text.split('\n')
            current_line = lines[position["line"]] if position["line"] < len(lines) else ""
            
            # Extract prefix (text before cursor)
            prefix = current_line[:position["character"]]
            
            # Create completion request
            code_context = CodeContext(
                file_path=text_document["uri"],
                language=document.language_id,
                line_number=position["line"],
                column_number=position["character"],
                current_line=current_line,
                preceding_lines=lines[:position["line"]],
                following_lines=lines[position["line"]+1:],
                prefix=prefix.split()[-1] if prefix.split() else "",
                suffix=current_line[position["character"]:],
            )
            
            trigger_kind = context.get("triggerKind", 1)
            trigger = CompletionTrigger.EXPLICIT_INVOCATION if trigger_kind == 2 else CompletionTrigger.CHARACTER_TYPED
            
            completion_request = CompletionRequest(
                context=code_context,
                trigger=trigger,
                trigger_character=context.get("triggerCharacter"),
                max_completions=50,
                include_snippets=True
            )
            
            # Get completions
            response = await self.completion_engine.get_completions(completion_request)
            
            # Convert to VSCode format
            vscode_items = []
            for item in response.items:
                vscode_item = {
                    "label": item.label,
                    "kind": self._get_vscode_completion_kind(item.kind),
                    "detail": item.detail,
                    "documentation": item.documentation,
                    "insertText": item.insert_text or item.label,
                    "filterText": item.filter_text or item.label,
                    "sortText": f"{item.priority:03d}_{item.label}",
                    "preselect": item.priority > 90,
                    "deprecated": item.deprecated
                }
                
                if item.additional_edits:
                    vscode_item["additionalTextEdits"] = item.additional_edits
                
                if item.command:
                    vscode_item["command"] = item.command
                
                vscode_items.append(vscode_item)
            
            return {
                "items": vscode_items,
                "isIncomplete": response.is_incomplete
            }
            
        except Exception as e:
            logger.error(f"Error handling completion: {e}")
            return {"items": []}
    
    async def _handle_hover(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle hover request"""
        try:
            text_document = params["textDocument"]
            position = params["position"]
            
            document = self.documents.get(text_document["uri"])
            if not document:
                return {}
            
            # Extract word at position
            lines = document.text.split('\n')
            current_line = lines[position["line"]] if position["line"] < len(lines) else ""
            
            # Simple hover implementation - would integrate with language analysis
            word_start = position["character"]
            word_end = position["character"]
            
            while word_start > 0 and current_line[word_start-1].isalnum():
                word_start -= 1
            while word_end < len(current_line) and current_line[word_end].isalnum():
                word_end += 1
            
            word = current_line[word_start:word_end]
            
            if word:
                return {
                    "contents": {
                        "kind": "markdown",
                        "value": f"**{word}**\n\nArchon AI analysis for `{word}` would appear here."
                    },
                    "range": {
                        "start": {"line": position["line"], "character": word_start},
                        "end": {"line": position["line"], "character": word_end}
                    }
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error handling hover: {e}")
            return {}
    
    async def _handle_signature_help(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle signature help request"""
        # Implementation for signature help
        return {
            "signatures": [],
            "activeSignature": 0,
            "activeParameter": 0
        }
    
    async def _handle_go_to_definition(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle go to definition request"""
        # Implementation for go to definition
        return []
    
    async def _handle_find_references(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle find references request"""
        # Implementation for find references
        return []
    
    async def _handle_rename(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle rename request"""
        # Implementation for rename
        return {"changes": {}}
    
    async def _handle_format_document(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle document formatting"""
        # Implementation for document formatting
        return []
    
    async def _handle_format_range(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle range formatting"""
        # Implementation for range formatting
        return []
    
    async def _handle_code_actions(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle code actions request"""
        # Implementation for code actions
        return []
    
    async def _handle_code_lens(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle code lens request"""
        # Implementation for code lens
        return []
    
    async def _handle_document_symbols(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle document symbols request"""
        # Implementation for document symbols
        return []
    
    async def _handle_workspace_symbols(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle workspace symbols request"""
        # Implementation for workspace symbols
        return []
    
    async def _handle_document_open(self, params: Dict[str, Any]) -> None:
        """Handle document open event"""
        try:
            text_document = params["textDocument"]
            
            document = VSCodeDocument(
                uri=text_document["uri"],
                language_id=text_document["languageId"],
                version=text_document["version"],
                text=text_document["text"],
                line_count=text_document["text"].count('\n') + 1,
                file_name=Path(text_document["uri"]).name if "file://" in text_document["uri"] else None
            )
            
            self.documents[document.uri] = document
            
            await self.emit_event("document_opened", {
                "uri": document.uri,
                "language": document.language_id
            })
            
        except Exception as e:
            logger.error(f"Error handling document open: {e}")
    
    async def _handle_document_change(self, params: Dict[str, Any]) -> None:
        """Handle document change event"""
        try:
            text_document = params["textDocument"]
            content_changes = params["contentChanges"]
            
            document = self.documents.get(text_document["uri"])
            if not document:
                return
            
            # Apply changes (simplified - full implementation would handle incremental changes)
            for change in content_changes:
                if "range" not in change:
                    # Full document change
                    document.text = change["text"]
                else:
                    # Incremental change - would need proper implementation
                    pass
            
            document.version = text_document["version"]
            document.is_dirty = True
            
            await self.emit_event("document_changed", {
                "uri": document.uri,
                "version": document.version
            })
            
        except Exception as e:
            logger.error(f"Error handling document change: {e}")
    
    async def _handle_document_save(self, params: Dict[str, Any]) -> None:
        """Handle document save event"""
        try:
            text_document = params["textDocument"]
            
            document = self.documents.get(text_document["uri"])
            if document:
                document.is_dirty = False
                
                await self.emit_event("document_saved", {
                    "uri": document.uri
                })
            
        except Exception as e:
            logger.error(f"Error handling document save: {e}")
    
    async def _handle_document_close(self, params: Dict[str, Any]) -> None:
        """Handle document close event"""
        try:
            text_document = params["textDocument"]
            uri = text_document["uri"]
            
            if uri in self.documents:
                del self.documents[uri]
            
            if uri in self.diagnostics:
                del self.diagnostics[uri]
            
            await self.emit_event("document_closed", {
                "uri": uri
            })
            
        except Exception as e:
            logger.error(f"Error handling document close: {e}")
    
    # Archon-specific command handlers
    
    async def _handle_analyze_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code analysis request"""
        try:
            uri = params.get("uri")
            if not uri:
                return {"error": "URI required"}
            
            document = self.documents.get(uri)
            if not document:
                return {"error": "Document not found"}
            
            # Perform code analysis
            analysis = {
                "complexity": 5,  # Mock complexity score
                "maintainability": 8,  # Mock maintainability score
                "test_coverage": 75,  # Mock test coverage
                "security_issues": 2,  # Mock security issues count
                "performance_issues": [],  # Mock performance issues
                "suggestions": [
                    "Consider breaking down large functions",
                    "Add more comprehensive error handling",
                    "Include type hints for better code clarity"
                ],
                "metrics": {
                    "lines_of_code": document.line_count,
                    "cyclomatic_complexity": 5,
                    "cognitive_complexity": 3,
                    "technical_debt_ratio": 0.12
                }
            }
            
            return {"analysis": analysis}
            
        except Exception as e:
            logger.error(f"Error analyzing code: {e}")
            return {"error": str(e)}
    
    async def _handle_generate_tests(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle test generation request"""
        try:
            uri = params.get("uri")
            function_name = params.get("function")
            
            # Mock test generation
            test_code = f"""
def test_{function_name or 'example'}():
    # Test case 1: Normal operation
    result = {function_name or 'example_function'}()
    assert result is not None
    
    # Test case 2: Edge case
    # Add edge case tests here
    
    # Test case 3: Error handling
    # Add error handling tests here
"""
            
            return {
                "generated_tests": test_code,
                "test_count": 3,
                "coverage_estimate": 85
            }
            
        except Exception as e:
            logger.error(f"Error generating tests: {e}")
            return {"error": str(e)}
    
    async def _handle_generate_documentation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle documentation generation request"""
        try:
            uri = params.get("uri")
            
            # Mock documentation generation
            documentation = """
# Module Documentation

## Overview
This module provides functionality for...

## Functions

### function_name()
Description of the function.

**Parameters:**
- param1 (type): Description
- param2 (type): Description

**Returns:**
- return_type: Description

**Example:**
```python
result = function_name(param1, param2)
```
"""
            
            return {
                "documentation": documentation,
                "format": "markdown"
            }
            
        except Exception as e:
            logger.error(f"Error generating documentation: {e}")
            return {"error": str(e)}
    
    async def _handle_refactor_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code refactoring request"""
        try:
            uri = params.get("uri")
            refactor_type = params.get("type", "extract_method")
            
            # Mock refactoring suggestions
            suggestions = [
                {
                    "type": "extract_method",
                    "description": "Extract method from lines 15-25",
                    "confidence": 0.9,
                    "changes": []
                },
                {
                    "type": "rename_variable",
                    "description": "Rename variable 'data' to 'user_data' for clarity",
                    "confidence": 0.8,
                    "changes": []
                }
            ]
            
            return {
                "suggestions": suggestions,
                "refactoring_opportunities": len(suggestions)
            }
            
        except Exception as e:
            logger.error(f"Error refactoring code: {e}")
            return {"error": str(e)}
    
    async def _handle_security_scan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle security scan request"""
        try:
            uri = params.get("uri")
            
            # Mock security scan results
            security_issues = [
                {
                    "severity": "medium",
                    "type": "hardcoded_secret",
                    "line": 42,
                    "description": "Potential hardcoded API key detected",
                    "suggestion": "Use environment variables for API keys"
                },
                {
                    "severity": "low",
                    "type": "sql_injection",
                    "line": 67,
                    "description": "Potential SQL injection vulnerability",
                    "suggestion": "Use parameterized queries"
                }
            ]
            
            return {
                "security_issues": security_issues,
                "security_score": 7.5,
                "total_issues": len(security_issues)
            }
            
        except Exception as e:
            logger.error(f"Error performing security scan: {e}")
            return {"error": str(e)}
    
    async def _handle_project_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle project status request"""
        try:
            project_status = {
                "name": "Current Project",
                "files_count": len(self.documents),
                "languages": list(set(doc.language_id for doc in self.documents.values())),
                "total_lines": sum(doc.line_count for doc in self.documents.values()),
                "dirty_files": [uri for uri, doc in self.documents.items() if doc.is_dirty],
                "diagnostics_summary": {
                    "errors": sum(1 for diag_list in self.diagnostics.values() 
                                for diag in diag_list if diag.severity == 1),
                    "warnings": sum(1 for diag_list in self.diagnostics.values() 
                                  for diag in diag_list if diag.severity == 2),
                    "info": sum(1 for diag_list in self.diagnostics.values() 
                              for diag in diag_list if diag.severity == 3)
                }
            }
            
            return {"project_status": project_status}
            
        except Exception as e:
            logger.error(f"Error getting project status: {e}")
            return {"error": str(e)}
    
    async def _handle_sync_collaboration(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle collaboration sync request"""
        try:
            # Mock collaboration sync
            sync_status = {
                "connected_users": 3,
                "active_sessions": ["user1", "user2", "user3"],
                "sync_conflicts": 0,
                "last_sync": datetime.now().isoformat(),
                "collaborative_edits": 15
            }
            
            return {"sync_status": sync_status}
            
        except Exception as e:
            logger.error(f"Error syncing collaboration: {e}")
            return {"error": str(e)}
    
    # Event Handlers
    
    async def _on_document_opened(self, event: PluginEvent) -> None:
        """Handle document opened event"""
        logger.info(f"Document opened: {event.data.get('uri')}")
    
    async def _on_document_changed(self, event: PluginEvent) -> None:
        """Handle document changed event"""
        # Could trigger real-time analysis here
        pass
    
    async def _on_document_saved(self, event: PluginEvent) -> None:
        """Handle document saved event"""
        # Could trigger post-save analysis here
        pass
    
    async def _on_selection_changed(self, event: PluginEvent) -> None:
        """Handle selection changed event"""
        pass
    
    async def _on_workspace_changed(self, event: PluginEvent) -> None:
        """Handle workspace changed event"""
        pass
    
    async def _on_configuration_changed(self, event: PluginEvent) -> None:
        """Handle configuration changed event"""
        pass
    
    # Helper Methods
    
    async def _initialize_completion_engine(self) -> None:
        """Initialize code completion engine"""
        # Completion engine is already initialized in constructor
        pass
    
    async def _setup_language_server(self) -> None:
        """Setup language server capabilities"""
        # Setup language server protocol capabilities
        pass
    
    def _get_vscode_completion_kind(self, archon_kind) -> int:
        """Convert Archon completion kind to VSCode completion item kind"""
        kind_map = {
            "variable": 6,      # Variable
            "function": 3,      # Function  
            "method": 2,        # Method
            "class": 7,         # Class
            "module": 9,        # Module
            "keyword": 14,      # Keyword
            "snippet": 15,      # Snippet
            "property": 10,     # Property
            "parameter": 6,     # Variable (for parameters)
            "import": 9,        # Module (for imports)
            "type_hint": 25,    # TypeParameter
            "documentation": 16 # Text
        }
        return kind_map.get(archon_kind.value if hasattr(archon_kind, 'value') else str(archon_kind), 1)
    
    def publish_diagnostics(self, uri: str, diagnostics: List[VSCodeDiagnostic]) -> None:
        """Publish diagnostics to VSCode"""
        self.diagnostics[uri] = diagnostics
        
        # Would send diagnostics to VSCode via LSP
        diagnostic_data = []
        for diag in diagnostics:
            diagnostic_data.append({
                "range": {
                    "start": {"line": diag.range.start.line, "character": diag.range.start.character},
                    "end": {"line": diag.range.end.line, "character": diag.range.end.character}
                },
                "message": diag.message,
                "severity": diag.severity,
                "source": diag.source,
                "code": diag.code
            })
        
        # Send to VSCode (would use LSP protocol)
        asyncio.create_task(self.emit_event("diagnostics_published", {
            "uri": uri,
            "diagnostics": diagnostic_data
        }))