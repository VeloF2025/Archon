"""
Context Analyzer for Predictive Assistant
Analyzes code context to provide relevant suggestions
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import ast
import re
from datetime import datetime
from collections import defaultdict
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CodeContext(BaseModel):
    """Represents the context of code being analyzed"""
    file_path: str
    cursor_position: Tuple[int, int]  # (line, column)
    current_line: str
    previous_lines: List[str] = Field(default_factory=list)
    next_lines: List[str] = Field(default_factory=list)
    language: str = "python"
    imports: List[str] = Field(default_factory=list)
    classes: List[str] = Field(default_factory=list)
    functions: List[str] = Field(default_factory=list)
    variables: Dict[str, str] = Field(default_factory=dict)  # name -> type
    current_scope: str = "module"
    indentation_level: int = 0


class ContextAnalyzer:
    """Analyzes code context for intelligent predictions"""
    
    def __init__(self):
        self.context_cache = {}
        self.pattern_matchers = self._initialize_pattern_matchers()
    
    def _initialize_pattern_matchers(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for code analysis"""
        return {
            "import": re.compile(r'^(?:from\s+(\S+)\s+)?import\s+(.+)'),
            "class": re.compile(r'^class\s+(\w+)(?:\((.*?)\))?:'),
            "function": re.compile(r'^(?:async\s+)?def\s+(\w+)\s*\((.*?)\)(?:\s*->\s*(.+?))?:'),
            "assignment": re.compile(r'^(\w+)\s*(?::\s*(.+?))?\s*=\s*(.+)'),
            "method_call": re.compile(r'(\w+)\.(\w*)$'),
            "dict_access": re.compile(r'(\w+)\[(["\']?)(\w*)'),
            "completion_trigger": re.compile(r'[\.\[\(\,\s](\w*)$'),
            "type_hint": re.compile(r':\s*(\w+)(?:\[(.+?)\])?'),
        }
    
    async def analyze_context(
        self,
        code: str,
        cursor_position: Tuple[int, int],
        file_path: Optional[str] = None,
        language: str = "python"
    ) -> CodeContext:
        """
        Analyze the code context around the cursor
        
        Args:
            code: Full code content
            cursor_position: (line, column) position of cursor
            file_path: Optional file path
            language: Programming language
            
        Returns:
            Analyzed code context
        """
        try:
            lines = code.split('\n')
            line_num, col_num = cursor_position
            
            # Ensure valid cursor position
            line_num = min(max(0, line_num), len(lines) - 1)
            
            current_line = lines[line_num] if lines else ""
            
            # Get surrounding lines
            context_window = 10
            start_idx = max(0, line_num - context_window)
            end_idx = min(len(lines), line_num + context_window + 1)
            
            previous_lines = lines[start_idx:line_num]
            next_lines = lines[line_num + 1:end_idx]
            
            # Analyze the code structure
            imports = self._extract_imports(lines)
            classes = self._extract_classes(lines)
            functions = self._extract_functions(lines)
            variables = self._extract_variables(lines, line_num)
            
            # Determine current scope
            current_scope = self._determine_scope(lines, line_num)
            
            # Calculate indentation
            indentation_level = self._get_indentation_level(current_line)
            
            context = CodeContext(
                file_path=file_path or "untitled",
                cursor_position=cursor_position,
                current_line=current_line,
                previous_lines=previous_lines,
                next_lines=next_lines,
                language=language,
                imports=imports,
                classes=classes,
                functions=functions,
                variables=variables,
                current_scope=current_scope,
                indentation_level=indentation_level
            )
            
            # Cache the context
            if file_path:
                self.context_cache[file_path] = context
            
            return context
            
        except Exception as e:
            logger.error(f"Error analyzing context: {e}")
            return CodeContext(
                file_path=file_path or "untitled",
                cursor_position=cursor_position,
                current_line="",
                language=language
            )
    
    def _extract_imports(self, lines: List[str]) -> List[str]:
        """Extract import statements from code"""
        imports = []
        
        for line in lines:
            match = self.pattern_matchers["import"].match(line.strip())
            if match:
                module = match.group(1) or ""
                items = match.group(2)
                
                if module:
                    imports.append(f"{module}.{items}")
                else:
                    imports.append(items)
        
        return imports
    
    def _extract_classes(self, lines: List[str]) -> List[str]:
        """Extract class definitions from code"""
        classes = []
        
        for line in lines:
            match = self.pattern_matchers["class"].match(line.strip())
            if match:
                class_name = match.group(1)
                classes.append(class_name)
        
        return classes
    
    def _extract_functions(self, lines: List[str]) -> List[str]:
        """Extract function definitions from code"""
        functions = []
        
        for line in lines:
            match = self.pattern_matchers["function"].match(line.strip())
            if match:
                func_name = match.group(1)
                functions.append(func_name)
        
        return functions
    
    def _extract_variables(self, lines: List[str], current_line: int) -> Dict[str, str]:
        """Extract variables and their types up to current line"""
        variables = {}
        
        for i, line in enumerate(lines[:current_line + 1]):
            match = self.pattern_matchers["assignment"].match(line.strip())
            if match:
                var_name = match.group(1)
                var_type = match.group(2) or "Any"
                variables[var_name] = var_type
        
        return variables
    
    def _determine_scope(self, lines: List[str], current_line: int) -> str:
        """Determine the current scope (module, class, function)"""
        scope_stack = ["module"]
        
        for i, line in enumerate(lines[:current_line + 1]):
            stripped = line.strip()
            
            # Check for class definition
            if self.pattern_matchers["class"].match(stripped):
                class_match = self.pattern_matchers["class"].match(stripped)
                if class_match:
                    scope_stack.append(f"class:{class_match.group(1)}")
            
            # Check for function definition
            elif self.pattern_matchers["function"].match(stripped):
                func_match = self.pattern_matchers["function"].match(stripped)
                if func_match:
                    scope_stack.append(f"function:{func_match.group(1)}")
            
            # Check for dedent (simplified)
            elif stripped and not line[0].isspace() and len(scope_stack) > 1:
                scope_stack = ["module"]
        
        return scope_stack[-1] if scope_stack else "module"
    
    def _get_indentation_level(self, line: str) -> int:
        """Get the indentation level of a line"""
        indent = 0
        for char in line:
            if char == ' ':
                indent += 1
            elif char == '\t':
                indent += 4  # Assuming 4 spaces per tab
            else:
                break
        
        return indent // 4  # Assuming 4 spaces per indent level
    
    async def get_completion_context(
        self,
        context: CodeContext
    ) -> Dict[str, Any]:
        """
        Get specific context for code completion
        
        Args:
            context: Code context
            
        Returns:
            Completion-specific context
        """
        current_line = context.current_line[:context.cursor_position[1]]
        
        # Check for method call
        method_match = self.pattern_matchers["method_call"].search(current_line)
        if method_match:
            object_name = method_match.group(1)
            partial_method = method_match.group(2)
            
            return {
                "type": "method_completion",
                "object": object_name,
                "object_type": context.variables.get(object_name, "unknown"),
                "partial": partial_method,
                "available_methods": self._get_object_methods(object_name, context)
            }
        
        # Check for dictionary access
        dict_match = self.pattern_matchers["dict_access"].search(current_line)
        if dict_match:
            dict_name = dict_match.group(1)
            partial_key = dict_match.group(3)
            
            return {
                "type": "dict_completion",
                "dictionary": dict_name,
                "partial": partial_key,
                "available_keys": self._get_dict_keys(dict_name, context)
            }
        
        # Check for import completion
        if current_line.strip().startswith(("import ", "from ")):
            return {
                "type": "import_completion",
                "partial": current_line.split()[-1] if current_line.split() else "",
                "available_modules": self._get_available_modules(context)
            }
        
        # Default to variable/keyword completion
        trigger_match = self.pattern_matchers["completion_trigger"].search(current_line)
        partial = trigger_match.group(1) if trigger_match else ""
        
        return {
            "type": "general_completion",
            "partial": partial,
            "scope": context.current_scope,
            "available_items": self._get_available_items(context)
        }
    
    def _get_object_methods(
        self,
        object_name: str,
        context: CodeContext
    ) -> List[str]:
        """Get available methods for an object"""
        # This would connect to type inference system
        # For now, return common methods based on type
        
        object_type = context.variables.get(object_name, "")
        
        common_methods = {
            "str": ["upper", "lower", "strip", "split", "replace", "format"],
            "list": ["append", "extend", "insert", "remove", "pop", "sort"],
            "dict": ["get", "keys", "values", "items", "update", "pop"],
            "set": ["add", "remove", "discard", "union", "intersection"],
        }
        
        return common_methods.get(object_type.lower(), [])
    
    def _get_dict_keys(
        self,
        dict_name: str,
        context: CodeContext
    ) -> List[str]:
        """Get available keys for a dictionary"""
        # This would analyze the code to find dictionary keys
        # For now, return empty list
        return []
    
    def _get_available_modules(
        self,
        context: CodeContext
    ) -> List[str]:
        """Get available modules for import completion"""
        # Common Python modules
        stdlib_modules = [
            "os", "sys", "json", "re", "datetime", "collections",
            "itertools", "functools", "pathlib", "typing", "asyncio"
        ]
        
        # Modules already imported
        imported = [imp.split('.')[0] for imp in context.imports]
        
        # Filter out already imported
        available = [m for m in stdlib_modules if m not in imported]
        
        return available
    
    def _get_available_items(
        self,
        context: CodeContext
    ) -> Dict[str, List[str]]:
        """Get all available items in current scope"""
        items = {
            "variables": list(context.variables.keys()),
            "functions": context.functions,
            "classes": context.classes,
            "keywords": [
                "if", "else", "elif", "for", "while", "try", "except",
                "finally", "with", "as", "def", "class", "return", "yield",
                "import", "from", "raise", "assert", "pass", "break", "continue"
            ]
        }
        
        # Add scope-specific items
        if context.current_scope.startswith("class:"):
            items["keywords"].extend(["self", "super", "__init__"])
        
        return items
    
    async def get_semantic_context(
        self,
        context: CodeContext,
        embedding_func=None
    ) -> Dict[str, Any]:
        """
        Get semantic context using embeddings
        
        Args:
            context: Code context
            embedding_func: Function to generate embeddings
            
        Returns:
            Semantic context information
        """
        semantic_info = {
            "intent": self._infer_intent(context),
            "patterns": self._detect_patterns(context),
            "complexity": self._calculate_complexity(context)
        }
        
        # Generate embedding if function provided
        if embedding_func:
            code_snippet = "\n".join(
                context.previous_lines[-3:] + 
                [context.current_line] + 
                context.next_lines[:3]
            )
            semantic_info["embedding"] = await embedding_func(code_snippet)
        
        return semantic_info
    
    def _infer_intent(self, context: CodeContext) -> str:
        """Infer the developer's intent from context"""
        current_line = context.current_line.strip()
        
        if current_line.startswith("def "):
            return "defining_function"
        elif current_line.startswith("class "):
            return "defining_class"
        elif "import" in current_line:
            return "importing_module"
        elif "=" in current_line:
            return "variable_assignment"
        elif current_line.endswith("."):
            return "accessing_attribute"
        elif current_line.endswith("("):
            return "calling_function"
        else:
            return "general_coding"
    
    def _detect_patterns(self, context: CodeContext) -> List[str]:
        """Detect coding patterns in context"""
        patterns = []
        
        # Check for common patterns
        code_block = "\n".join(context.previous_lines[-5:] + [context.current_line])
        
        if "try:" in code_block and "except" in code_block:
            patterns.append("error_handling")
        
        if "with open" in code_block:
            patterns.append("file_operation")
        
        if "async def" in code_block:
            patterns.append("async_programming")
        
        if "@" in code_block and "def" in code_block:
            patterns.append("decorator_usage")
        
        if "for " in code_block or "while " in code_block:
            patterns.append("iteration")
        
        return patterns
    
    def _calculate_complexity(self, context: CodeContext) -> int:
        """Calculate complexity score of current context"""
        complexity = 0
        
        # Factors that increase complexity
        complexity += len(context.imports) // 5
        complexity += len(context.classes) // 2
        complexity += len(context.functions) // 3
        complexity += len(context.variables) // 5
        complexity += context.indentation_level
        
        # Cap at 10
        return min(complexity, 10)