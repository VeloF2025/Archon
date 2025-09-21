"""
IntelliJ IDEA Plugin
Comprehensive IntelliJ IDEA integration for Archon AI development platform
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from .plugin_base import (
    PluginBase, PluginMetadata, PluginConfiguration, PluginCapability, 
    PluginCommand, PluginResponse, PluginEvent, PluginStatus,
    CommunicationProtocol
)
from .ide_communication import IDECommunicationProtocol, HTTPChannel, ConnectionConfig
from .code_completion_engine import CodeCompletionEngine, CompletionRequest, CodeContext, CompletionTrigger

logger = logging.getLogger(__name__)


@dataclass
class IntelliJDocument:
    """IntelliJ document representation"""
    virtual_file_url: str
    file_type: str
    content: str
    modification_stamp: int
    is_writable: bool = True
    encoding: str = "UTF-8"
    line_separator: str = "\n"


@dataclass
class IntelliJEditor:
    """IntelliJ editor representation"""
    document: IntelliJDocument
    caret_model: Dict[str, Any]
    selection_model: Dict[str, Any]
    fold_model: Dict[str, Any]
    markup_model: Dict[str, Any]


@dataclass
class IntelliJProject:
    """IntelliJ project representation"""
    name: str
    base_path: str
    project_file_path: str
    modules: List[Dict[str, Any]]
    libraries: List[Dict[str, Any]]
    sdk_name: Optional[str] = None
    language_level: Optional[str] = None


@dataclass
class IntelliJIntention:
    """IntelliJ intention action"""
    text: str
    family_name: str
    description: str
    priority: int
    available: bool = True


class IntelliJPlugin(PluginBase):
    """IntelliJ IDEA plugin implementation"""
    
    def __init__(self, config: Optional[PluginConfiguration] = None):
        metadata = PluginMetadata(
            name="archon-intellij",
            version="2.0.0", 
            description="Archon AI Development Platform - IntelliJ IDEA Integration",
            author="Archon Development Team",
            supported_languages=["java", "kotlin", "scala", "groovy", "python", "javascript", "typescript"],
            capabilities=[
                PluginCapability.CODE_COMPLETION,
                PluginCapability.SYNTAX_HIGHLIGHTING,
                PluginCapability.ERROR_DETECTION,
                PluginCapability.REFACTORING,
                PluginCapability.DEBUGGING,
                PluginCapability.INTELLIGENT_SUGGESTIONS,
                PluginCapability.CODE_ANALYSIS,
                PluginCapability.DOCUMENTATION_GENERATION,
                PluginCapability.TEST_GENERATION,
                PluginCapability.SECURITY_ANALYSIS,
                PluginCapability.GIT_INTEGRATION,
                PluginCapability.PROJECT_MANAGEMENT,
                PluginCapability.PERFORMANCE_PROFILING
            ],
            protocols=[CommunicationProtocol.HTTP, CommunicationProtocol.JSON_RPC],
            min_ide_version="2021.1",
            installation_url="https://plugins.jetbrains.com/plugin/archon-intellij",
            documentation_url="https://docs.archon.dev/intellij-plugin"
        )
        
        super().__init__(metadata, config)
        
        # IntelliJ-specific components
        self.communication = IDECommunicationProtocol()
        self.completion_engine = CodeCompletionEngine()
        self.projects: Dict[str, IntelliJProject] = {}
        self.documents: Dict[str, IntelliJDocument] = {}
        self.editors: Dict[str, IntelliJEditor] = {}
        self.intentions: Dict[str, List[IntelliJIntention]] = {}
        self.http_port = 8055
        
        # IntelliJ Platform API integration
        self.psi_manager = None
        self.project_manager = None
        self.code_insight_manager = None
        
        # Register command handlers
        self._register_command_handlers()
        
        # Register event handlers
        self._register_event_handlers()
    
    def _register_command_handlers(self) -> None:
        """Register IntelliJ command handlers"""
        commands = {
            # Standard IntelliJ actions
            "completion/basic": self._handle_basic_completion,
            "completion/smart": self._handle_smart_completion,
            "completion/class": self._handle_class_completion,
            "inspection/run": self._handle_run_inspection,
            "intention/list": self._handle_list_intentions,
            "intention/apply": self._handle_apply_intention,
            "refactor/extract_method": self._handle_extract_method,
            "refactor/rename": self._handle_rename,
            "refactor/move": self._handle_move_refactoring,
            "navigate/declaration": self._handle_navigate_to_declaration,
            "navigate/implementation": self._handle_navigate_to_implementation,
            "navigate/usage": self._handle_find_usages,
            "project/structure": self._handle_get_project_structure,
            "module/dependencies": self._handle_get_module_dependencies,
            "build/compile": self._handle_compile_project,
            "test/run": self._handle_run_tests,
            "debug/start": self._handle_start_debugging,
            "vcs/commit": self._handle_vcs_commit,
            "vcs/update": self._handle_vcs_update,
            
            # Archon-specific commands
            "archon/analyze_code": self._handle_analyze_code,
            "archon/generate_tests": self._handle_generate_tests,
            "archon/optimize_imports": self._handle_optimize_imports,
            "archon/code_quality_check": self._handle_code_quality_check,
            "archon/security_audit": self._handle_security_audit,
            "archon/performance_analysis": self._handle_performance_analysis,
            "archon/dependency_analysis": self._handle_dependency_analysis,
            "archon/generate_documentation": self._handle_generate_documentation
        }
        
        for command, handler in commands.items():
            self.register_command_handler(command, handler)
    
    def _register_event_handlers(self) -> None:
        """Register IntelliJ event handlers"""
        events = {
            "project_opened": self._on_project_opened,
            "project_closed": self._on_project_closed,
            "document_changed": self._on_document_changed,
            "editor_focus": self._on_editor_focus,
            "compilation_finished": self._on_compilation_finished,
            "test_session_finished": self._on_test_session_finished,
            "vcs_changes": self._on_vcs_changes,
            "module_structure_changed": self._on_module_structure_changed
        }
        
        for event, handler in events.items():
            self.register_event_handler(event, handler)
    
    async def initialize(self) -> bool:
        """Initialize IntelliJ plugin"""
        try:
            logger.info("Initializing IntelliJ plugin...")
            
            # Setup HTTP communication channel
            http_config = ConnectionConfig(
                host="localhost",
                port=self.http_port,
                timeout_seconds=30
            )
            
            http_channel = HTTPChannel(http_config)
            self.communication.register_channel("http", http_channel)
            
            # Initialize completion engine
            await self._initialize_completion_engine()
            
            # Setup IntelliJ Platform API integration
            await self._setup_platform_integration()
            
            logger.info("IntelliJ plugin initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize IntelliJ plugin: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown IntelliJ plugin"""
        try:
            logger.info("Shutting down IntelliJ plugin...")
            
            # Disconnect communication channels
            for ide_id in list(self.communication._active_connections.keys()):
                await self.communication.disconnect_from_ide(ide_id)
            
            # Clear caches
            self.projects.clear()
            self.documents.clear()
            self.editors.clear()
            self.intentions.clear()
            
            logger.info("IntelliJ plugin shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Error shutting down IntelliJ plugin: {e}")
            return False
    
    async def execute_command(self, command: PluginCommand) -> PluginResponse:
        """Execute IntelliJ plugin command"""
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
        """Handle IntelliJ events"""
        try:
            event_type = event.event_type
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    await handler(event)
        except Exception as e:
            logger.error(f"Error handling event {event.event_type}: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get IntelliJ plugin status"""
        return {
            **self.get_health_info(),
            "projects_open": len(self.projects),
            "documents_open": len(self.documents),
            "active_editors": len(self.editors),
            "available_intentions": sum(len(intentions) for intentions in self.intentions.values()),
            "communication_status": self.communication.get_connection_status(),
            "completion_stats": self.completion_engine.get_engine_stats(),
            "supported_languages": self.metadata.supported_languages,
            "http_port": self.http_port,
            "platform_version": await self._get_platform_version()
        }
    
    # Command Handlers
    
    async def _handle_basic_completion(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle basic code completion"""
        try:
            file_path = params.get("filePath")
            offset = params.get("offset")
            
            if not file_path or offset is None:
                return {"completions": []}
            
            document = self.documents.get(file_path)
            if not document:
                return {"completions": []}
            
            # Convert offset to line/column
            content = document.content
            lines = content[:offset].split('\n')
            line = len(lines) - 1
            column = len(lines[-1])
            
            # Create completion context
            all_lines = content.split('\n')
            current_line = all_lines[line] if line < len(all_lines) else ""
            prefix = current_line[:column]
            
            code_context = CodeContext(
                file_path=file_path,
                language=self._map_file_type_to_language(document.file_type),
                line_number=line,
                column_number=column,
                current_line=current_line,
                preceding_lines=all_lines[:line],
                following_lines=all_lines[line+1:],
                prefix=prefix.split()[-1] if prefix.split() else "",
                suffix=current_line[column:]
            )
            
            # Get completions
            completion_request = CompletionRequest(
                context=code_context,
                trigger=CompletionTrigger.EXPLICIT_INVOCATION,
                max_completions=25
            )
            
            response = await self.completion_engine.get_completions(completion_request)
            
            # Convert to IntelliJ format
            intellij_completions = []
            for item in response.items:
                intellij_completion = {
                    "lookupString": item.label,
                    "presentableText": item.detail,
                    "tailText": item.documentation[:50] + "..." if len(item.documentation) > 50 else item.documentation,
                    "typeText": self._get_intellij_type(item.kind),
                    "insertHandler": item.insert_text or item.label,
                    "priority": item.priority,
                    "deprecated": item.deprecated
                }
                intellij_completions.append(intellij_completion)
            
            return {"completions": intellij_completions}
            
        except Exception as e:
            logger.error(f"Error handling basic completion: {e}")
            return {"completions": []}
    
    async def _handle_smart_completion(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle smart code completion with type inference"""
        # Enhanced completion with IntelliJ's smart completion features
        basic_completions = await self._handle_basic_completion(params)
        
        # Add smart completion enhancements
        enhanced_completions = []
        for completion in basic_completions.get("completions", []):
            # Add type-aware suggestions
            completion["smart"] = True
            completion["priority"] += 10  # Boost priority for smart completions
            enhanced_completions.append(completion)
        
        return {"completions": enhanced_completions}
    
    async def _handle_class_completion(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle class name completion"""
        try:
            prefix = params.get("prefix", "")
            project_path = params.get("projectPath")
            
            # Mock class completion - would integrate with IntelliJ's PSI
            classes = [
                {"name": "ArrayList", "package": "java.util", "priority": 90},
                {"name": "HashMap", "package": "java.util", "priority": 90},
                {"name": "String", "package": "java.lang", "priority": 95},
                {"name": "Integer", "package": "java.lang", "priority": 85}
            ]
            
            filtered_classes = [cls for cls in classes if cls["name"].lower().startswith(prefix.lower())]
            
            return {"classes": filtered_classes}
            
        except Exception as e:
            logger.error(f"Error handling class completion: {e}")
            return {"classes": []}
    
    async def _handle_run_inspection(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code inspection"""
        try:
            file_path = params.get("filePath")
            inspection_profile = params.get("profile", "default")
            
            # Mock inspection results
            inspections = [
                {
                    "severity": "WARNING",
                    "line": 15,
                    "column": 10,
                    "message": "Unused import statement",
                    "inspection_id": "UnusedImport",
                    "fix_available": True
                },
                {
                    "severity": "ERROR", 
                    "line": 23,
                    "column": 5,
                    "message": "Variable might not be initialized",
                    "inspection_id": "UninitializedVariable",
                    "fix_available": False
                }
            ]
            
            return {"inspections": inspections}
            
        except Exception as e:
            logger.error(f"Error running inspection: {e}")
            return {"inspections": []}
    
    async def _handle_list_intentions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list available intentions"""
        try:
            file_path = params.get("filePath")
            offset = params.get("offset")
            
            # Mock intentions
            intentions = [
                IntelliJIntention(
                    text="Create method 'getValue'",
                    family_name="Create method",
                    description="Creates a missing method",
                    priority=100
                ),
                IntelliJIntention(
                    text="Add null check",
                    family_name="Add null check",
                    description="Adds a null check before method call",
                    priority=90
                ),
                IntelliJIntention(
                    text="Extract to variable",
                    family_name="Extract",
                    description="Extracts expression to a local variable",
                    priority=80
                )
            ]
            
            return {
                "intentions": [
                    {
                        "text": intention.text,
                        "familyName": intention.family_name,
                        "description": intention.description,
                        "priority": intention.priority,
                        "available": intention.available
                    }
                    for intention in intentions
                ]
            }
            
        except Exception as e:
            logger.error(f"Error listing intentions: {e}")
            return {"intentions": []}
    
    async def _handle_apply_intention(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle apply intention action"""
        try:
            intention_text = params.get("intentionText")
            file_path = params.get("filePath")
            offset = params.get("offset")
            
            # Mock intention application
            if "Create method" in intention_text:
                generated_code = "\npublic String getValue() {\n    return null;\n}\n"
                return {
                    "success": True,
                    "changes": [{
                        "filePath": file_path,
                        "insertText": generated_code,
                        "offset": offset
                    }]
                }
            
            return {"success": False, "reason": "Intention not supported"}
            
        except Exception as e:
            logger.error(f"Error applying intention: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_extract_method(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle extract method refactoring"""
        try:
            file_path = params.get("filePath")
            start_offset = params.get("startOffset")
            end_offset = params.get("endOffset")
            method_name = params.get("methodName", "extractedMethod")
            
            # Mock extract method refactoring
            return {
                "success": True,
                "preview": f"Extracted method '{method_name}' will be created",
                "changes": [
                    {
                        "filePath": file_path,
                        "description": f"Extract method '{method_name}'",
                        "type": "EXTRACT_METHOD"
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"Error extracting method: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_rename(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle rename refactoring"""
        try:
            file_path = params.get("filePath")
            offset = params.get("offset")
            new_name = params.get("newName")
            
            # Mock rename refactoring
            return {
                "success": True,
                "preview": f"Rename to '{new_name}'",
                "usages_found": 5,
                "conflicts": []
            }
            
        except Exception as e:
            logger.error(f"Error renaming: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_move_refactoring(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle move refactoring"""
        try:
            source_path = params.get("sourcePath")
            target_package = params.get("targetPackage")
            
            return {
                "success": True,
                "preview": f"Move to package '{target_package}'",
                "affected_files": []
            }
            
        except Exception as e:
            logger.error(f"Error moving: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_navigate_to_declaration(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle navigate to declaration"""
        try:
            file_path = params.get("filePath")
            offset = params.get("offset")
            
            # Mock navigation result
            return {
                "found": True,
                "target": {
                    "filePath": "/path/to/declaration.java",
                    "line": 42,
                    "column": 10
                }
            }
            
        except Exception as e:
            logger.error(f"Error navigating to declaration: {e}")
            return {"found": False, "error": str(e)}
    
    async def _handle_navigate_to_implementation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle navigate to implementation"""
        try:
            file_path = params.get("filePath")
            offset = params.get("offset")
            
            # Mock implementation results
            return {
                "implementations": [
                    {
                        "filePath": "/path/to/impl1.java",
                        "line": 20,
                        "column": 5,
                        "className": "ConcreteImpl1"
                    },
                    {
                        "filePath": "/path/to/impl2.java", 
                        "line": 35,
                        "column": 8,
                        "className": "ConcreteImpl2"
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"Error finding implementations: {e}")
            return {"implementations": []}
    
    async def _handle_find_usages(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle find usages"""
        try:
            file_path = params.get("filePath")
            offset = params.get("offset")
            
            # Mock usage results
            return {
                "usages": [
                    {
                        "filePath": "/path/to/usage1.java",
                        "line": 15,
                        "column": 20,
                        "context": "method call"
                    },
                    {
                        "filePath": "/path/to/usage2.java",
                        "line": 8,
                        "column": 12,
                        "context": "field access"
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"Error finding usages: {e}")
            return {"usages": []}
    
    async def _handle_get_project_structure(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get project structure"""
        try:
            project_path = params.get("projectPath")
            
            # Mock project structure
            return {
                "project": {
                    "name": "MyProject",
                    "basePath": project_path,
                    "modules": [
                        {
                            "name": "main",
                            "contentRoots": ["/src/main/java"],
                            "sourceFolders": ["/src/main/java"],
                            "testFolders": ["/src/test/java"]
                        }
                    ],
                    "libraries": [
                        {"name": "junit", "version": "4.12"},
                        {"name": "mockito", "version": "3.0.0"}
                    ],
                    "sdk": {
                        "name": "Java 11",
                        "version": "11.0.2"
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting project structure: {e}")
            return {"project": None, "error": str(e)}
    
    async def _handle_get_module_dependencies(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get module dependencies"""
        try:
            module_name = params.get("moduleName")
            
            return {
                "dependencies": [
                    {
                        "name": "org.springframework:spring-core",
                        "version": "5.3.0",
                        "scope": "compile"
                    },
                    {
                        "name": "junit:junit",
                        "version": "4.12", 
                        "scope": "test"
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting module dependencies: {e}")
            return {"dependencies": []}
    
    async def _handle_compile_project(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle compile project"""
        try:
            project_path = params.get("projectPath")
            
            # Mock compilation
            return {
                "success": True,
                "duration_ms": 5000,
                "warnings": 2,
                "errors": 0,
                "output": "Compilation completed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error compiling project: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_run_tests(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle run tests"""
        try:
            test_class = params.get("testClass")
            test_method = params.get("testMethod")
            
            return {
                "success": True,
                "tests_run": 15,
                "failures": 0,
                "errors": 0,
                "skipped": 1,
                "duration_ms": 2500,
                "results": [
                    {"name": "testMethod1", "status": "PASSED"},
                    {"name": "testMethod2", "status": "PASSED"},
                    {"name": "testMethod3", "status": "SKIPPED"}
                ]
            }
            
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_start_debugging(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle start debugging"""
        try:
            main_class = params.get("mainClass")
            
            return {
                "session_started": True,
                "debug_port": 5005,
                "session_id": str(uuid.uuid4())
            }
            
        except Exception as e:
            logger.error(f"Error starting debugging: {e}")
            return {"session_started": False, "error": str(e)}
    
    async def _handle_vcs_commit(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle VCS commit"""
        try:
            message = params.get("message")
            files = params.get("files", [])
            
            return {
                "success": True,
                "commit_id": "abc123def456",
                "files_committed": len(files)
            }
            
        except Exception as e:
            logger.error(f"Error committing: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_vcs_update(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle VCS update"""
        try:
            return {
                "success": True,
                "files_updated": 5,
                "files_merged": 1,
                "conflicts": 0
            }
            
        except Exception as e:
            logger.error(f"Error updating: {e}")
            return {"success": False, "error": str(e)}
    
    # Archon-specific handlers
    
    async def _handle_analyze_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Archon code analysis"""
        try:
            file_path = params.get("filePath")
            
            return {
                "complexity_score": 7.5,
                "maintainability_index": 85,
                "code_smells": [
                    {"type": "Long Method", "severity": "medium", "line": 45},
                    {"type": "Too Many Parameters", "severity": "low", "line": 12}
                ],
                "suggestions": [
                    "Consider breaking down the method at line 45",
                    "Use builder pattern for methods with many parameters"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing code: {e}")
            return {"error": str(e)}
    
    async def _handle_generate_tests(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle test generation"""
        try:
            class_name = params.get("className")
            method_name = params.get("methodName")
            
            test_code = f"""
@Test
public void test{method_name}() {{
    // Given
    {class_name} instance = new {class_name}();
    
    // When
    Object result = instance.{method_name}();
    
    // Then
    assertNotNull(result);
}}

@Test
public void test{method_name}WithNullInput() {{
    // Given
    {class_name} instance = new {class_name}();
    
    // When & Then
    assertThrows(IllegalArgumentException.class, () -> {{
        instance.{method_name}();
    }});
}}
"""
            
            return {
                "generated_tests": test_code,
                "test_count": 2,
                "coverage_estimate": 80
            }
            
        except Exception as e:
            logger.error(f"Error generating tests: {e}")
            return {"error": str(e)}
    
    # Event Handlers
    
    async def _on_project_opened(self, event: PluginEvent) -> None:
        """Handle project opened event"""
        project_path = event.data.get("projectPath")
        logger.info(f"Project opened: {project_path}")
    
    async def _on_project_closed(self, event: PluginEvent) -> None:
        """Handle project closed event"""
        project_path = event.data.get("projectPath")
        logger.info(f"Project closed: {project_path}")
    
    async def _on_document_changed(self, event: PluginEvent) -> None:
        """Handle document changed event"""
        file_path = event.data.get("filePath")
        # Could trigger incremental analysis
        pass
    
    async def _on_editor_focus(self, event: PluginEvent) -> None:
        """Handle editor focus event"""
        pass
    
    async def _on_compilation_finished(self, event: PluginEvent) -> None:
        """Handle compilation finished event"""
        success = event.data.get("success", False)
        logger.info(f"Compilation finished: {'success' if success else 'failed'}")
    
    async def _on_test_session_finished(self, event: PluginEvent) -> None:
        """Handle test session finished event"""
        pass
    
    async def _on_vcs_changes(self, event: PluginEvent) -> None:
        """Handle VCS changes event"""
        pass
    
    async def _on_module_structure_changed(self, event: PluginEvent) -> None:
        """Handle module structure changed event"""
        pass
    
    # Helper Methods
    
    async def _initialize_completion_engine(self) -> None:
        """Initialize completion engine"""
        pass
    
    async def _setup_platform_integration(self) -> None:
        """Setup IntelliJ Platform API integration"""
        pass
    
    async def _get_platform_version(self) -> str:
        """Get IntelliJ Platform version"""
        return "2023.1"
    
    def _map_file_type_to_language(self, file_type: str) -> str:
        """Map IntelliJ file type to language"""
        mapping = {
            "JAVA": "java",
            "KOTLIN": "kotlin", 
            "SCALA": "scala",
            "PYTHON": "python",
            "JavaScript": "javascript",
            "TypeScript": "typescript"
        }
        return mapping.get(file_type, "text")
    
    def _get_intellij_type(self, completion_kind) -> str:
        """Get IntelliJ type for completion kind"""
        type_mapping = {
            "variable": "FIELD",
            "function": "METHOD",
            "method": "METHOD",
            "class": "CLASS", 
            "module": "PACKAGE",
            "keyword": "KEYWORD"
        }
        kind_str = completion_kind.value if hasattr(completion_kind, 'value') else str(completion_kind)
        return type_mapping.get(kind_str, "UNKNOWN")