"""
Completion Provider for Predictive Assistant
Provides code completion functionality for IDEs and editors
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
import re
from pydantic import BaseModel, Field

from .context_analyzer import CodeContext, ContextAnalyzer
from .suggestion_engine import SuggestionEngine, CodeSuggestion

logger = logging.getLogger(__name__)


class CompletionRequest(BaseModel):
    """Request for code completion"""
    code: str
    cursor_position: Tuple[int, int]
    file_path: Optional[str] = None
    language: str = "python"
    max_completions: int = 10
    trigger_character: Optional[str] = None


class CompletionResponse(BaseModel):
    """Response with code completions"""
    completions: List[CodeSuggestion]
    context_type: str
    execution_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CompletionProvider:
    """Provides intelligent code completions"""
    
    def __init__(
        self,
        context_analyzer: Optional[ContextAnalyzer] = None,
        suggestion_engine: Optional[SuggestionEngine] = None
    ):
        self.context_analyzer = context_analyzer or ContextAnalyzer()
        self.suggestion_engine = suggestion_engine or SuggestionEngine()
        self.completion_cache = {}
        self.trigger_characters = [".", "[", "(", ",", " ", ":"]
    
    async def provide_completions(
        self,
        request: CompletionRequest
    ) -> CompletionResponse:
        """
        Provide code completions for a request
        
        Args:
            request: Completion request
            
        Returns:
            Completion response with suggestions
        """
        start_time = datetime.utcnow()
        
        try:
            # Analyze context
            context = await self.context_analyzer.analyze_context(
                code=request.code,
                cursor_position=request.cursor_position,
                file_path=request.file_path,
                language=request.language
            )
            
            # Get completion context
            completion_context = await self.context_analyzer.get_completion_context(context)
            
            # Get semantic context if needed
            semantic_context = None
            if self._should_use_semantic_context(completion_context):
                semantic_context = await self.context_analyzer.get_semantic_context(context)
            
            # Generate suggestions
            suggestions = await self.suggestion_engine.generate_suggestions(
                context=context,
                completion_context=completion_context,
                semantic_context=semantic_context
            )
            
            # Filter and limit completions
            completions = self._filter_completions(
                suggestions,
                request.max_completions,
                request.trigger_character
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return CompletionResponse(
                completions=completions,
                context_type=completion_context.get("type", "unknown"),
                execution_time=execution_time,
                metadata={
                    "total_suggestions": len(suggestions),
                    "filtered_count": len(completions),
                    "trigger": request.trigger_character,
                    "scope": context.current_scope
                }
            )
            
        except Exception as e:
            logger.error(f"Error providing completions: {e}")
            return CompletionResponse(
                completions=[],
                context_type="error",
                execution_time=0.0,
                metadata={"error": str(e)}
            )
    
    def _should_use_semantic_context(
        self,
        completion_context: Dict[str, Any]
    ) -> bool:
        """Determine if semantic context should be used"""
        # Use semantic context for general completions and patterns
        return completion_context.get("type") in [
            "general_completion",
            "pattern_completion"
        ]
    
    def _filter_completions(
        self,
        suggestions: List[CodeSuggestion],
        max_completions: int,
        trigger_character: Optional[str]
    ) -> List[CodeSuggestion]:
        """Filter completions based on trigger and limit"""
        filtered = suggestions
        
        # Filter based on trigger character
        if trigger_character == ".":
            # Method/attribute completions only
            filtered = [s for s in suggestions if s.category in ["completion", "method"]]
        elif trigger_character == "[":
            # Dictionary/list completions
            filtered = [s for s in suggestions if s.category in ["completion", "index"]]
        elif trigger_character == "(":
            # Parameter completions
            filtered = [s for s in suggestions if s.category in ["completion", "parameter"]]
        
        # Limit number of completions
        return filtered[:max_completions]
    
    async def get_signature_help(
        self,
        code: str,
        cursor_position: Tuple[int, int],
        file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get signature help for function calls
        
        Args:
            code: Code content
            cursor_position: Cursor position
            file_path: Optional file path
            
        Returns:
            Signature help information
        """
        try:
            context = await self.context_analyzer.analyze_context(
                code=code,
                cursor_position=cursor_position,
                file_path=file_path
            )
            
            # Find function call at cursor
            current_line = context.current_line[:cursor_position[1]]
            
            # Simple function call detection
            if "(" in current_line:
                func_match = re.search(r'(\w+)\s*\([^)]*$', current_line)
                if func_match:
                    func_name = func_match.group(1)
                    
                    # Get function signature
                    signature = self._get_function_signature(func_name, context)
                    
                    if signature:
                        return {
                            "function": func_name,
                            "signature": signature,
                            "parameters": self._parse_parameters(signature),
                            "active_parameter": self._get_active_parameter(current_line)
                        }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting signature help: {e}")
            return {}
    
    def _get_function_signature(
        self,
        func_name: str,
        context: CodeContext
    ) -> Optional[str]:
        """Get function signature from context"""
        # Check if function is defined in context
        if func_name in context.functions:
            # Would need to parse the actual function definition
            # For now, return a placeholder
            return f"{func_name}(args) -> result"
        
        # Check built-in functions
        builtins = {
            "print": "print(*values, sep=' ', end='\\n', file=sys.stdout, flush=False)",
            "len": "len(obj) -> int",
            "range": "range(stop) -> range | range(start, stop[, step]) -> range",
            "open": "open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)"
        }
        
        return builtins.get(func_name)
    
    def _parse_parameters(self, signature: str) -> List[Dict[str, str]]:
        """Parse parameters from function signature"""
        # Simple parameter parsing
        params = []
        
        if "(" in signature and ")" in signature:
            param_str = signature[signature.index("(") + 1:signature.index(")")]
            
            for param in param_str.split(","):
                param = param.strip()
                if param:
                    # Handle default values
                    if "=" in param:
                        name, default = param.split("=", 1)
                        params.append({
                            "name": name.strip(),
                            "default": default.strip(),
                            "required": False
                        })
                    else:
                        params.append({
                            "name": param,
                            "required": True
                        })
        
        return params
    
    def _get_active_parameter(self, current_line: str) -> int:
        """Get the active parameter index based on cursor position"""
        # Count commas to determine parameter index
        if "(" in current_line:
            after_paren = current_line[current_line.rindex("(") + 1:]
            comma_count = after_paren.count(",")
            return comma_count
        
        return 0
    
    async def get_hover_info(
        self,
        code: str,
        cursor_position: Tuple[int, int],
        file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get hover information for symbol at cursor
        
        Args:
            code: Code content
            cursor_position: Cursor position
            file_path: Optional file path
            
        Returns:
            Hover information
        """
        try:
            context = await self.context_analyzer.analyze_context(
                code=code,
                cursor_position=cursor_position,
                file_path=file_path
            )
            
            # Get word at cursor
            word = self._get_word_at_cursor(context.current_line, cursor_position[1])
            
            if word:
                # Check variables
                if word in context.variables:
                    return {
                        "symbol": word,
                        "type": "variable",
                        "info": f"Variable: {word}",
                        "type_hint": context.variables[word]
                    }
                
                # Check functions
                if word in context.functions:
                    return {
                        "symbol": word,
                        "type": "function",
                        "info": f"Function: {word}",
                        "signature": self._get_function_signature(word, context)
                    }
                
                # Check classes
                if word in context.classes:
                    return {
                        "symbol": word,
                        "type": "class",
                        "info": f"Class: {word}"
                    }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting hover info: {e}")
            return {}
    
    def _get_word_at_cursor(
        self,
        line: str,
        cursor_col: int
    ) -> Optional[str]:
        """Get the word at cursor position"""
        if cursor_col > len(line):
            return None
        
        # Find word boundaries
        start = cursor_col
        while start > 0 and (line[start - 1].isalnum() or line[start - 1] == "_"):
            start -= 1
        
        end = cursor_col
        while end < len(line) and (line[end].isalnum() or line[end] == "_"):
            end += 1
        
        word = line[start:end]
        return word if word else None
    
    def get_trigger_characters(self) -> List[str]:
        """Get list of trigger characters for completion"""
        return self.trigger_characters
    
    def clear_cache(self):
        """Clear the completion cache"""
        self.completion_cache.clear()
        logger.info("Completion cache cleared")