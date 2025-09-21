"""
Code Completion Engine
Advanced AI-powered code completion and intelligent suggestions for IDEs
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
from pydantic import BaseModel, Field
import re
import ast
import tokenize
import io
from collections import defaultdict, deque
import numpy as np
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


class CompletionType(Enum):
    """Types of code completions"""
    VARIABLE = "variable"
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    MODULE = "module"
    KEYWORD = "keyword"
    SNIPPET = "snippet"
    PROPERTY = "property"
    PARAMETER = "parameter"
    IMPORT = "import"
    TYPE_HINT = "type_hint"
    DOCUMENTATION = "documentation"


class CompletionTrigger(Enum):
    """Completion trigger types"""
    CHARACTER_TYPED = "character_typed"
    EXPLICIT_INVOCATION = "explicit_invocation"
    AUTOMATIC = "automatic"
    SIGNATURE_HELP = "signature_help"
    HOVER = "hover"


class ContextType(Enum):
    """Types of code context"""
    FUNCTION_BODY = "function_body"
    CLASS_BODY = "class_body"
    MODULE_LEVEL = "module_level"
    IMPORT_STATEMENT = "import_statement"
    FUNCTION_PARAMETERS = "function_parameters"
    FUNCTION_CALL = "function_call"
    STRING_LITERAL = "string_literal"
    COMMENT = "comment"
    DOCSTRING = "docstring"


@dataclass
class CodeContext:
    """Context information for code completion"""
    file_path: str
    language: str
    line_number: int
    column_number: int
    current_line: str
    preceding_lines: List[str] = field(default_factory=list)
    following_lines: List[str] = field(default_factory=list)
    cursor_position: int = 0
    prefix: str = ""
    suffix: str = ""
    context_type: ContextType = ContextType.MODULE_LEVEL
    scope_variables: Dict[str, str] = field(default_factory=dict)
    imported_modules: List[str] = field(default_factory=list)
    class_hierarchy: List[str] = field(default_factory=list)
    function_context: Optional[str] = None


@dataclass 
class CompletionItem:
    """Individual completion suggestion"""
    label: str
    kind: CompletionType
    detail: str = ""
    documentation: str = ""
    insert_text: str = ""
    filter_text: str = ""
    sort_text: str = ""
    priority: int = 0
    confidence: float = 0.0
    source: str = "archon"
    additional_edits: List[Dict[str, Any]] = field(default_factory=list)
    command: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False


@dataclass
class CompletionRequest:
    """Request for code completion"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context: CodeContext
    trigger: CompletionTrigger = CompletionTrigger.CHARACTER_TYPED
    trigger_character: Optional[str] = None
    max_completions: int = 50
    include_snippets: bool = True
    include_documentation: bool = True
    filter_types: List[CompletionType] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CompletionResponse:
    """Response containing completion suggestions"""
    request_id: str
    items: List[CompletionItem] = field(default_factory=list)
    is_incomplete: bool = False
    processing_time_ms: int = 0
    total_candidates: int = 0
    context_analysis: Dict[str, Any] = field(default_factory=dict)
    suggestions_metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class LanguageParser:
    """Language-specific parsing and analysis"""
    
    def __init__(self, language: str):
        self.language = language.lower()
        self.keywords = self._get_language_keywords()
        self.built_ins = self._get_built_in_functions()
        
    def _get_language_keywords(self) -> List[str]:
        """Get language-specific keywords"""
        keywords_map = {
            "python": ["def", "class", "if", "else", "elif", "for", "while", "try", "except", 
                      "import", "from", "as", "return", "yield", "with", "async", "await",
                      "lambda", "global", "nonlocal", "True", "False", "None"],
            "javascript": ["function", "var", "let", "const", "if", "else", "for", "while",
                          "try", "catch", "finally", "return", "async", "await", "class",
                          "extends", "import", "export", "from", "default", "null", "undefined"],
            "typescript": ["function", "var", "let", "const", "if", "else", "for", "while",
                          "try", "catch", "finally", "return", "async", "await", "class",
                          "extends", "import", "export", "from", "default", "interface",
                          "type", "namespace", "enum", "public", "private", "protected"],
            "java": ["public", "private", "protected", "class", "interface", "extends",
                    "implements", "if", "else", "for", "while", "try", "catch", "finally",
                    "return", "static", "final", "abstract", "synchronized", "volatile"],
            "go": ["func", "var", "const", "type", "struct", "interface", "if", "else",
                  "for", "range", "switch", "case", "default", "return", "go", "defer",
                  "select", "chan", "map", "package", "import"]
        }
        return keywords_map.get(self.language, [])
    
    def _get_built_in_functions(self) -> List[str]:
        """Get language-specific built-in functions"""
        builtins_map = {
            "python": ["print", "len", "range", "enumerate", "zip", "map", "filter", "sum",
                      "min", "max", "abs", "round", "sorted", "reversed", "any", "all",
                      "isinstance", "hasattr", "getattr", "setattr", "type", "str", "int"],
            "javascript": ["console.log", "parseInt", "parseFloat", "isNaN", "isFinite",
                          "setTimeout", "setInterval", "clearTimeout", "clearInterval",
                          "JSON.parse", "JSON.stringify", "Object.keys", "Array.from"],
            "typescript": ["console.log", "parseInt", "parseFloat", "isNaN", "isFinite",
                          "setTimeout", "setInterval", "clearTimeout", "clearInterval",
                          "JSON.parse", "JSON.stringify", "Object.keys", "Array.from"],
            "java": ["System.out.println", "String.valueOf", "Integer.parseInt", 
                    "Double.parseDouble", "Math.abs", "Math.max", "Math.min", "Math.round"],
            "go": ["fmt.Println", "fmt.Printf", "len", "cap", "make", "new", "append",
                  "copy", "delete", "panic", "recover"]
        }
        return builtins_map.get(self.language, [])
    
    def parse_context(self, context: CodeContext) -> Dict[str, Any]:
        """Parse code context for completion"""
        analysis = {
            "scope_variables": {},
            "imported_modules": [],
            "function_signatures": {},
            "class_definitions": {},
            "context_type": ContextType.MODULE_LEVEL,
            "indentation_level": 0,
            "in_string": False,
            "in_comment": False
        }
        
        if self.language == "python":
            return self._parse_python_context(context)
        elif self.language in ["javascript", "typescript"]:
            return self._parse_js_context(context)
        elif self.language == "java":
            return self._parse_java_context(context)
        elif self.language == "go":
            return self._parse_go_context(context)
        
        return analysis
    
    def _parse_python_context(self, context: CodeContext) -> Dict[str, Any]:
        """Parse Python-specific context"""
        analysis = {
            "scope_variables": {},
            "imported_modules": [],
            "function_signatures": {},
            "class_definitions": {},
            "context_type": ContextType.MODULE_LEVEL,
            "indentation_level": 0,
            "in_string": False,
            "in_comment": False,
            "decorator_context": False,
            "inside_function": None,
            "inside_class": None
        }
        
        try:
            # Analyze all preceding lines
            full_code = "\n".join(context.preceding_lines + [context.current_line])
            
            # Parse with AST
            try:
                tree = ast.parse(full_code)
                analysis.update(self._analyze_python_ast(tree))
            except SyntaxError:
                # Partial parsing for incomplete code
                for i, line in enumerate(context.preceding_lines):
                    self._analyze_python_line(line, i, analysis)
            
            # Analyze current line context
            current_line = context.current_line.strip()
            analysis["indentation_level"] = len(context.current_line) - len(current_line)
            
            # Determine context type
            if current_line.startswith("import ") or current_line.startswith("from "):
                analysis["context_type"] = ContextType.IMPORT_STATEMENT
            elif "def " in current_line:
                analysis["context_type"] = ContextType.FUNCTION_BODY
            elif "class " in current_line:
                analysis["context_type"] = ContextType.CLASS_BODY
            elif current_line.startswith('"""') or current_line.startswith("'''"):
                analysis["context_type"] = ContextType.DOCSTRING
            elif current_line.startswith("#"):
                analysis["context_type"] = ContextType.COMMENT
            
        except Exception as e:
            logger.error(f"Error parsing Python context: {e}")
        
        return analysis
    
    def _analyze_python_ast(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze Python AST for context information"""
        analysis = {
            "scope_variables": {},
            "imported_modules": [],
            "function_signatures": {},
            "class_definitions": {}
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    analysis["imported_modules"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    analysis["imported_modules"].append(node.module)
            elif isinstance(node, ast.FunctionDef):
                args = [arg.arg for arg in node.args.args]
                analysis["function_signatures"][node.name] = args
            elif isinstance(node, ast.ClassDef):
                analysis["class_definitions"][node.name] = {
                    "methods": [],
                    "attributes": []
                }
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        analysis["scope_variables"][target.id] = "variable"
        
        return analysis
    
    def _analyze_python_line(self, line: str, line_num: int, analysis: Dict[str, Any]) -> None:
        """Analyze individual Python line"""
        stripped = line.strip()
        
        # Import statements
        if stripped.startswith("import ") or stripped.startswith("from "):
            import_match = re.search(r"import\s+([^\s,]+)", stripped)
            if import_match:
                analysis["imported_modules"].append(import_match.group(1))
        
        # Function definitions
        func_match = re.match(r"def\s+(\w+)\s*\((.*?)\)", stripped)
        if func_match:
            func_name = func_match.group(1)
            params = [p.strip().split(':')[0].strip() for p in func_match.group(2).split(',') if p.strip()]
            analysis["function_signatures"][func_name] = params
        
        # Class definitions
        class_match = re.match(r"class\s+(\w+)", stripped)
        if class_match:
            analysis["class_definitions"][class_match.group(1)] = {
                "methods": [],
                "attributes": []
            }
        
        # Variable assignments
        assign_match = re.match(r"(\w+)\s*=", stripped)
        if assign_match:
            analysis["scope_variables"][assign_match.group(1)] = "variable"
    
    def _parse_js_context(self, context: CodeContext) -> Dict[str, Any]:
        """Parse JavaScript/TypeScript context"""
        analysis = {
            "scope_variables": {},
            "imported_modules": [],
            "function_signatures": {},
            "class_definitions": {},
            "context_type": ContextType.MODULE_LEVEL,
            "inside_function": None,
            "inside_class": None,
            "inside_object": False
        }
        
        for line in context.preceding_lines:
            stripped = line.strip()
            
            # Import/require statements
            if "import" in stripped or "require" in stripped:
                import_match = re.search(r"from\s+['\"]([^'\"]+)['\"]", stripped)
                if import_match:
                    analysis["imported_modules"].append(import_match.group(1))
            
            # Function definitions
            func_matches = [
                re.match(r"function\s+(\w+)", stripped),
                re.match(r"(\w+)\s*:\s*function", stripped),
                re.match(r"(\w+)\s*=>\s*", stripped),
                re.match(r"const\s+(\w+)\s*=\s*\(", stripped)
            ]
            
            for match in func_matches:
                if match:
                    analysis["function_signatures"][match.group(1)] = []
                    break
            
            # Variable declarations
            var_match = re.match(r"(?:var|let|const)\s+(\w+)", stripped)
            if var_match:
                analysis["scope_variables"][var_match.group(1)] = "variable"
        
        return analysis
    
    def _parse_java_context(self, context: CodeContext) -> Dict[str, Any]:
        """Parse Java context"""
        analysis = {
            "scope_variables": {},
            "imported_modules": [],
            "function_signatures": {},
            "class_definitions": {},
            "context_type": ContextType.MODULE_LEVEL,
            "package": None,
            "inside_class": None
        }
        
        for line in context.preceding_lines:
            stripped = line.strip()
            
            # Package declaration
            package_match = re.match(r"package\s+([^;]+)", stripped)
            if package_match:
                analysis["package"] = package_match.group(1)
            
            # Import statements
            import_match = re.match(r"import\s+([^;]+)", stripped)
            if import_match:
                analysis["imported_modules"].append(import_match.group(1))
            
            # Class definitions
            class_match = re.match(r"(?:public\s+)?class\s+(\w+)", stripped)
            if class_match:
                analysis["class_definitions"][class_match.group(1)] = {
                    "methods": [],
                    "fields": []
                }
            
            # Method definitions
            method_match = re.match(r"(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(", stripped)
            if method_match:
                analysis["function_signatures"][method_match.group(1)] = []
        
        return analysis
    
    def _parse_go_context(self, context: CodeContext) -> Dict[str, Any]:
        """Parse Go context"""
        analysis = {
            "scope_variables": {},
            "imported_modules": [],
            "function_signatures": {},
            "struct_definitions": {},
            "context_type": ContextType.MODULE_LEVEL,
            "package": None,
            "inside_function": None
        }
        
        for line in context.preceding_lines:
            stripped = line.strip()
            
            # Package declaration
            package_match = re.match(r"package\s+(\w+)", stripped)
            if package_match:
                analysis["package"] = package_match.group(1)
            
            # Import statements
            if stripped.startswith("import"):
                import_match = re.search(r'"([^"]+)"', stripped)
                if import_match:
                    analysis["imported_modules"].append(import_match.group(1))
            
            # Function definitions
            func_match = re.match(r"func\s+(\w+)", stripped)
            if func_match:
                analysis["function_signatures"][func_match.group(1)] = []
            
            # Struct definitions
            struct_match = re.match(r"type\s+(\w+)\s+struct", stripped)
            if struct_match:
                analysis["struct_definitions"][struct_match.group(1)] = {
                    "fields": []
                }
            
            # Variable declarations
            var_match = re.match(r"var\s+(\w+)", stripped)
            if var_match:
                analysis["scope_variables"][var_match.group(1)] = "variable"
        
        return analysis


class CompletionEngine:
    """Main completion engine with AI-powered suggestions"""
    
    def __init__(self):
        self.parsers: Dict[str, LanguageParser] = {}
        self.completion_cache: Dict[str, CompletionResponse] = {}
        self.user_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.snippet_library: Dict[str, List[Dict[str, Any]]] = {}
        self.recent_completions: deque = deque(maxlen=1000)
        self._load_snippet_library()
    
    def _load_snippet_library(self) -> None:
        """Load code snippets for different languages"""
        self.snippet_library = {
            "python": [
                {
                    "label": "if __name__ == '__main__'",
                    "insert_text": "if __name__ == '__main__':\n    ${1:main()}",
                    "description": "Main guard",
                    "kind": CompletionType.SNIPPET
                },
                {
                    "label": "try/except",
                    "insert_text": "try:\n    ${1:pass}\nexcept ${2:Exception} as ${3:e}:\n    ${4:pass}",
                    "description": "Try-except block",
                    "kind": CompletionType.SNIPPET
                },
                {
                    "label": "class",
                    "insert_text": "class ${1:ClassName}:\n    def __init__(self${2:, args}):\n        ${3:pass}",
                    "description": "Class definition",
                    "kind": CompletionType.SNIPPET
                },
                {
                    "label": "def",
                    "insert_text": "def ${1:function_name}(${2:args}):\n    \"\"\"${3:Description}\"\"\"\n    ${4:pass}",
                    "description": "Function definition",
                    "kind": CompletionType.SNIPPET
                }
            ],
            "javascript": [
                {
                    "label": "function",
                    "insert_text": "function ${1:functionName}(${2:params}) {\n    ${3:// code}\n}",
                    "description": "Function declaration",
                    "kind": CompletionType.SNIPPET
                },
                {
                    "label": "arrow function",
                    "insert_text": "const ${1:functionName} = (${2:params}) => {\n    ${3:// code}\n};",
                    "description": "Arrow function",
                    "kind": CompletionType.SNIPPET
                },
                {
                    "label": "try/catch",
                    "insert_text": "try {\n    ${1:// code}\n} catch (${2:error}) {\n    ${3:// handle error}\n}",
                    "description": "Try-catch block",
                    "kind": CompletionType.SNIPPET
                }
            ],
            "typescript": [
                {
                    "label": "interface",
                    "insert_text": "interface ${1:InterfaceName} {\n    ${2:property}: ${3:type};\n}",
                    "description": "Interface definition",
                    "kind": CompletionType.SNIPPET
                },
                {
                    "label": "type alias",
                    "insert_text": "type ${1:TypeName} = ${2:type};",
                    "description": "Type alias",
                    "kind": CompletionType.SNIPPET
                }
            ]
        }
    
    def get_parser(self, language: str) -> LanguageParser:
        """Get or create language parser"""
        if language not in self.parsers:
            self.parsers[language] = LanguageParser(language)
        return self.parsers[language]
    
    async def get_completions(self, request: CompletionRequest) -> CompletionResponse:
        """Get code completions for given request"""
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = self._create_cache_key(request)
            if cache_key in self.completion_cache:
                cached = self.completion_cache[cache_key]
                # Return cached result if recent
                if (datetime.now() - cached.timestamp).total_seconds() < 60:
                    return cached
            
            # Parse context
            parser = self.get_parser(request.context.language)
            context_analysis = parser.parse_context(request.context)
            
            # Generate completions
            completions = []
            
            # Add keyword completions
            completions.extend(await self._get_keyword_completions(request, parser))
            
            # Add variable completions
            completions.extend(await self._get_variable_completions(request, context_analysis))
            
            # Add function completions
            completions.extend(await self._get_function_completions(request, context_analysis))
            
            # Add import completions
            completions.extend(await self._get_import_completions(request))
            
            # Add snippet completions
            if request.include_snippets:
                completions.extend(await self._get_snippet_completions(request))
            
            # Add AI-powered intelligent completions
            completions.extend(await self._get_ai_completions(request, context_analysis))
            
            # Filter and rank completions
            filtered_completions = await self._filter_and_rank_completions(
                completions, request, context_analysis
            )
            
            # Create response
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            response = CompletionResponse(
                request_id=request.request_id,
                items=filtered_completions[:request.max_completions],
                processing_time_ms=int(processing_time),
                total_candidates=len(completions),
                context_analysis=context_analysis,
                suggestions_metadata={
                    "keywords": len([c for c in completions if c.kind == CompletionType.KEYWORD]),
                    "variables": len([c for c in completions if c.kind == CompletionType.VARIABLE]),
                    "functions": len([c for c in completions if c.kind == CompletionType.FUNCTION]),
                    "snippets": len([c for c in completions if c.kind == CompletionType.SNIPPET])
                }
            )
            
            # Cache response
            self.completion_cache[cache_key] = response
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating completions: {e}")
            return CompletionResponse(
                request_id=request.request_id,
                items=[],
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                context_analysis={"error": str(e)}
            )
    
    async def _get_keyword_completions(self, request: CompletionRequest, parser: LanguageParser) -> List[CompletionItem]:
        """Get language keyword completions"""
        completions = []
        prefix = request.context.prefix.lower()
        
        for keyword in parser.keywords:
            if keyword.lower().startswith(prefix):
                completions.append(CompletionItem(
                    label=keyword,
                    kind=CompletionType.KEYWORD,
                    detail=f"{request.context.language} keyword",
                    insert_text=keyword,
                    priority=90,
                    confidence=0.9
                ))
        
        return completions
    
    async def _get_variable_completions(self, request: CompletionRequest, context_analysis: Dict[str, Any]) -> List[CompletionItem]:
        """Get variable completions from scope"""
        completions = []
        prefix = request.context.prefix
        
        for var_name, var_type in context_analysis.get("scope_variables", {}).items():
            if var_name.startswith(prefix):
                completions.append(CompletionItem(
                    label=var_name,
                    kind=CompletionType.VARIABLE,
                    detail=f"Variable ({var_type})",
                    insert_text=var_name,
                    priority=80,
                    confidence=0.8
                ))
        
        return completions
    
    async def _get_function_completions(self, request: CompletionRequest, context_analysis: Dict[str, Any]) -> List[CompletionItem]:
        """Get function completions"""
        completions = []
        prefix = request.context.prefix
        
        # Add functions from current scope
        for func_name, params in context_analysis.get("function_signatures", {}).items():
            if func_name.startswith(prefix):
                param_list = ", ".join(params) if params else ""
                completions.append(CompletionItem(
                    label=func_name,
                    kind=CompletionType.FUNCTION,
                    detail=f"Function({param_list})",
                    insert_text=f"{func_name}({param_list})",
                    priority=85,
                    confidence=0.85
                ))
        
        # Add built-in functions
        parser = self.get_parser(request.context.language)
        for builtin in parser.built_ins:
            if builtin.startswith(prefix):
                completions.append(CompletionItem(
                    label=builtin,
                    kind=CompletionType.FUNCTION,
                    detail="Built-in function",
                    insert_text=builtin,
                    priority=75,
                    confidence=0.75
                ))
        
        return completions
    
    async def _get_import_completions(self, request: CompletionRequest) -> List[CompletionItem]:
        """Get import completions"""
        completions = []
        
        if request.context.context_type == ContextType.IMPORT_STATEMENT:
            language = request.context.language
            common_imports = {
                "python": ["os", "sys", "json", "re", "datetime", "collections", "itertools", 
                          "functools", "pathlib", "typing", "dataclasses", "asyncio"],
                "javascript": ["react", "lodash", "axios", "moment", "uuid"],
                "typescript": ["react", "lodash", "axios", "moment", "uuid", "@types/node"],
                "java": ["java.util.List", "java.util.Map", "java.util.ArrayList", "java.io.IOException"],
                "go": ["fmt", "os", "io", "net/http", "encoding/json", "time"]
            }
            
            for module in common_imports.get(language, []):
                if module.startswith(request.context.prefix):
                    completions.append(CompletionItem(
                        label=module,
                        kind=CompletionType.MODULE,
                        detail=f"{language} module",
                        insert_text=module,
                        priority=70,
                        confidence=0.7
                    ))
        
        return completions
    
    async def _get_snippet_completions(self, request: CompletionRequest) -> List[CompletionItem]:
        """Get code snippet completions"""
        completions = []
        language = request.context.language
        prefix = request.context.prefix
        
        snippets = self.snippet_library.get(language, [])
        for snippet in snippets:
            if snippet["label"].startswith(prefix):
                completions.append(CompletionItem(
                    label=snippet["label"],
                    kind=CompletionType.SNIPPET,
                    detail=snippet["description"],
                    insert_text=snippet["insert_text"],
                    priority=60,
                    confidence=0.6
                ))
        
        return completions
    
    async def _get_ai_completions(self, request: CompletionRequest, context_analysis: Dict[str, Any]) -> List[CompletionItem]:
        """Get AI-powered intelligent completions"""
        completions = []
        
        # This would integrate with ML models for intelligent suggestions
        # For now, provide pattern-based suggestions
        
        context = request.context
        language = context.language
        
        # Pattern-based suggestions
        if language == "python":
            if "for " in context.current_line:
                completions.append(CompletionItem(
                    label="enumerate",
                    kind=CompletionType.FUNCTION,
                    detail="Enumerate items with index",
                    insert_text="enumerate(${1:iterable})",
                    priority=95,
                    confidence=0.95,
                    source="ai_pattern"
                ))
            
            if "if " in context.current_line and "is" in context.current_line:
                completions.append(CompletionItem(
                    label="is not None",
                    kind=CompletionType.KEYWORD,
                    detail="Check if not None",
                    insert_text="is not None",
                    priority=90,
                    confidence=0.9,
                    source="ai_pattern"
                ))
        
        elif language in ["javascript", "typescript"]:
            if "map" in context.current_line:
                completions.append(CompletionItem(
                    label="arrow function",
                    kind=CompletionType.SNIPPET,
                    detail="Arrow function for map",
                    insert_text="(${1:item}) => ${2:item.property}",
                    priority=95,
                    confidence=0.95,
                    source="ai_pattern"
                ))
        
        return completions
    
    async def _filter_and_rank_completions(self, completions: List[CompletionItem], 
                                         request: CompletionRequest,
                                         context_analysis: Dict[str, Any]) -> List[CompletionItem]:
        """Filter and rank completions by relevance"""
        # Filter by prefix
        prefix = request.context.prefix.lower()
        filtered = []
        
        for completion in completions:
            # Fuzzy matching
            similarity = self._calculate_similarity(prefix, completion.label.lower())
            if similarity > 0.3:  # Minimum similarity threshold
                completion.confidence *= similarity
                filtered.append(completion)
        
        # Apply user preferences and patterns
        for completion in filtered:
            user_boost = self._calculate_user_pattern_boost(completion, request.context.file_path)
            completion.confidence *= user_boost
            completion.priority = int(completion.confidence * 100)
        
        # Sort by priority and confidence
        filtered.sort(key=lambda x: (x.priority, x.confidence), reverse=True)
        
        return filtered
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two strings"""
        if not text1 or not text2:
            return 0.0
        
        # Exact prefix match gets highest score
        if text2.startswith(text1):
            return 1.0
        
        # Use sequence matcher for fuzzy matching
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _calculate_user_pattern_boost(self, completion: CompletionItem, file_path: str) -> float:
        """Calculate boost based on user usage patterns"""
        base_boost = 1.0
        
        # Check recent usage
        for recent_completion in list(self.recent_completions)[-50:]:
            if recent_completion.get("label") == completion.label:
                base_boost *= 1.1
                break
        
        # Check file-specific patterns
        file_patterns = self.user_patterns.get(file_path, {})
        if completion.label in file_patterns:
            usage_count = file_patterns[completion.label]
            base_boost *= (1.0 + usage_count * 0.05)  # 5% boost per usage
        
        return min(base_boost, 2.0)  # Cap boost at 2x
    
    def record_completion_acceptance(self, completion: CompletionItem, context: CodeContext) -> None:
        """Record that user accepted a completion"""
        self.recent_completions.append({
            "label": completion.label,
            "kind": completion.kind.value,
            "file_path": context.file_path,
            "timestamp": datetime.now()
        })
        
        # Update user patterns
        self.user_patterns[context.file_path][completion.label] += 1
    
    def _create_cache_key(self, request: CompletionRequest) -> str:
        """Create cache key for completion request"""
        key_parts = [
            request.context.file_path,
            str(request.context.line_number),
            str(request.context.column_number),
            request.context.prefix,
            request.trigger.value
        ]
        return "|".join(key_parts)
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get completion engine statistics"""
        return {
            "total_completions_served": len(self.recent_completions),
            "cache_size": len(self.completion_cache),
            "supported_languages": list(self.parsers.keys()),
            "snippet_languages": list(self.snippet_library.keys()),
            "user_patterns_count": sum(len(patterns) for patterns in self.user_patterns.values()),
            "average_completion_confidence": np.mean([c.get("confidence", 0) for c in self.recent_completions]) if self.recent_completions else 0
        }