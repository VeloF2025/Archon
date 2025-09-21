"""
Suggestion Engine for Predictive Assistant
Generates intelligent code suggestions based on context and patterns
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
from collections import defaultdict
from pydantic import BaseModel, Field

from .context_analyzer import CodeContext
from ..pattern_recognition.pattern_storage import PatternStorage
from ..knowledge_graph.query_engine import GraphQueryEngine

logger = logging.getLogger(__name__)


class CodeSuggestion(BaseModel):
    """Represents a code suggestion"""
    text: str
    description: str
    confidence: float = 1.0
    category: str = "general"  # general, pattern, snippet, completion
    source: str = "analyzer"  # analyzer, patterns, knowledge_graph, history
    metadata: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 0  # Higher is better
    preview: Optional[str] = None  # Preview of code after applying suggestion


class SuggestionEngine:
    """Generates intelligent code suggestions"""
    
    def __init__(
        self,
        pattern_storage: Optional[PatternStorage] = None,
        graph_query_engine: Optional[GraphQueryEngine] = None
    ):
        self.pattern_storage = pattern_storage
        self.graph_query_engine = graph_query_engine
        self.suggestion_cache = {}
        self.history = []
        self.user_preferences = self._load_preferences()
    
    def _load_preferences(self) -> Dict[str, Any]:
        """Load user preferences for suggestions"""
        return {
            "max_suggestions": 10,
            "min_confidence": 0.5,
            "prefer_patterns": True,
            "include_snippets": True,
            "auto_import": True,
            "style_guide": "pep8"
        }
    
    async def generate_suggestions(
        self,
        context: CodeContext,
        completion_context: Dict[str, Any],
        semantic_context: Optional[Dict[str, Any]] = None
    ) -> List[CodeSuggestion]:
        """
        Generate code suggestions based on context
        
        Args:
            context: Code context
            completion_context: Specific completion context
            semantic_context: Optional semantic context
            
        Returns:
            List of code suggestions
        """
        try:
            suggestions = []
            
            # Generate different types of suggestions in parallel
            tasks = [
                self._generate_completion_suggestions(context, completion_context),
                self._generate_pattern_suggestions(context, semantic_context),
                self._generate_snippet_suggestions(context, completion_context),
                self._generate_knowledge_suggestions(context, semantic_context)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    suggestions.extend(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Suggestion generation error: {result}")
            
            # Rank and filter suggestions
            suggestions = self._rank_suggestions(suggestions, context)
            suggestions = self._filter_suggestions(suggestions)
            
            # Cache suggestions
            cache_key = f"{context.file_path}:{context.cursor_position}"
            self.suggestion_cache[cache_key] = suggestions
            
            # Track in history
            self.history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "context": context.current_line,
                "suggestions_count": len(suggestions)
            })
            
            return suggestions[:self.user_preferences["max_suggestions"]]
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return []
    
    async def _generate_completion_suggestions(
        self,
        context: CodeContext,
        completion_context: Dict[str, Any]
    ) -> List[CodeSuggestion]:
        """Generate basic completion suggestions"""
        suggestions = []
        completion_type = completion_context.get("type", "general_completion")
        
        if completion_type == "method_completion":
            # Method completions
            object_name = completion_context.get("object", "")
            partial = completion_context.get("partial", "")
            methods = completion_context.get("available_methods", [])
            
            for method in methods:
                if not partial or method.startswith(partial):
                    suggestions.append(CodeSuggestion(
                        text=method + "()",
                        description=f"Call {method} on {object_name}",
                        confidence=0.9 if not partial else 0.95,
                        category="completion",
                        source="analyzer",
                        metadata={"object": object_name, "method": method},
                        priority=8
                    ))
        
        elif completion_type == "import_completion":
            # Import completions
            partial = completion_context.get("partial", "")
            modules = completion_context.get("available_modules", [])
            
            for module in modules:
                if not partial or module.startswith(partial):
                    suggestions.append(CodeSuggestion(
                        text=module,
                        description=f"Import {module} module",
                        confidence=0.85,
                        category="completion",
                        source="analyzer",
                        metadata={"module": module},
                        priority=7
                    ))
        
        elif completion_type == "general_completion":
            # General completions
            partial = completion_context.get("partial", "")
            available_items = completion_context.get("available_items", {})
            
            # Variables
            for var in available_items.get("variables", []):
                if not partial or var.startswith(partial):
                    suggestions.append(CodeSuggestion(
                        text=var,
                        description=f"Variable {var}",
                        confidence=0.8,
                        category="completion",
                        source="analyzer",
                        priority=6
                    ))
            
            # Functions
            for func in available_items.get("functions", []):
                if not partial or func.startswith(partial):
                    suggestions.append(CodeSuggestion(
                        text=func + "()",
                        description=f"Call function {func}",
                        confidence=0.75,
                        category="completion",
                        source="analyzer",
                        priority=5
                    ))
        
        return suggestions
    
    async def _generate_pattern_suggestions(
        self,
        context: CodeContext,
        semantic_context: Optional[Dict[str, Any]] = None
    ) -> List[CodeSuggestion]:
        """Generate pattern-based suggestions"""
        suggestions = []
        
        if not self.pattern_storage:
            return suggestions
        
        try:
            # Get intent from semantic context
            intent = semantic_context.get("intent", "general_coding") if semantic_context else "general_coding"
            
            # Search for relevant patterns
            patterns = await self.pattern_storage.search_patterns(
                query=intent,
                language=context.language,
                limit=5
            )
            
            for pattern in patterns:
                if pattern.effectiveness_score >= 0.7:
                    # Generate suggestion from pattern
                    suggestion_text = self._generate_pattern_code(pattern, context)
                    
                    suggestions.append(CodeSuggestion(
                        text=suggestion_text,
                        description=f"Apply {pattern.name} pattern",
                        confidence=pattern.confidence,
                        category="pattern",
                        source="patterns",
                        metadata={
                            "pattern_id": pattern.id,
                            "pattern_name": pattern.name,
                            "effectiveness": pattern.effectiveness_score
                        },
                        priority=10,
                        preview=self._generate_preview(context, suggestion_text)
                    ))
        
        except Exception as e:
            logger.error(f"Error generating pattern suggestions: {e}")
        
        return suggestions
    
    async def _generate_snippet_suggestions(
        self,
        context: CodeContext,
        completion_context: Dict[str, Any]
    ) -> List[CodeSuggestion]:
        """Generate code snippet suggestions"""
        suggestions = []
        
        if not self.user_preferences.get("include_snippets"):
            return suggestions
        
        # Common snippets based on context
        snippets = self._get_contextual_snippets(context, completion_context)
        
        for snippet in snippets:
            suggestions.append(CodeSuggestion(
                text=snippet["code"],
                description=snippet["description"],
                confidence=snippet.get("confidence", 0.7),
                category="snippet",
                source="analyzer",
                metadata=snippet.get("metadata", {}),
                priority=snippet.get("priority", 6),
                preview=self._generate_preview(context, snippet["code"])
            ))
        
        return suggestions
    
    async def _generate_knowledge_suggestions(
        self,
        context: CodeContext,
        semantic_context: Optional[Dict[str, Any]] = None
    ) -> List[CodeSuggestion]:
        """Generate suggestions from knowledge graph"""
        suggestions = []
        
        if not self.graph_query_engine:
            return suggestions
        
        try:
            # Query knowledge graph for relevant concepts
            query = f"find code examples for {context.current_scope} in {context.language}"
            
            if semantic_context and semantic_context.get("patterns"):
                query += f" with patterns {', '.join(semantic_context['patterns'])}"
            
            result = await self.graph_query_engine.query(
                query=query,
                query_type="natural"
            )
            
            for item in result.results[:3]:  # Top 3 results
                if "code" in item:
                    suggestions.append(CodeSuggestion(
                        text=item["code"],
                        description=item.get("description", "Knowledge-based suggestion"),
                        confidence=item.get("confidence", 0.6),
                        category="snippet",
                        source="knowledge_graph",
                        metadata={"source_id": item.get("id")},
                        priority=7
                    ))
        
        except Exception as e:
            logger.error(f"Error generating knowledge suggestions: {e}")
        
        return suggestions
    
    def _generate_pattern_code(
        self,
        pattern: Any,
        context: CodeContext
    ) -> str:
        """Generate code from a pattern"""
        # This would be more sophisticated in production
        # For now, return a simple template
        
        if pattern.name == "Singleton":
            return """_instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance"""
        
        elif pattern.name == "Factory":
            return """@staticmethod
    def create(type_name: str):
        if type_name == "A":
            return TypeA()
        elif type_name == "B":
            return TypeB()
        else:
            raise ValueError(f"Unknown type: {type_name}")"""
        
        else:
            return f"# {pattern.name} pattern implementation"
    
    def _get_contextual_snippets(
        self,
        context: CodeContext,
        completion_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get context-appropriate code snippets"""
        snippets = []
        
        # Error handling snippet
        if context.current_line.strip().startswith("try"):
            snippets.append({
                "code": """try:
    # Code here
    pass
except Exception as e:
    logger.error(f"Error: {e}")
    raise""",
                "description": "Try-except block with logging",
                "confidence": 0.8,
                "priority": 7
            })
        
        # Function definition snippet
        if context.current_line.strip() == "def":
            snippets.append({
                "code": """def function_name(param1: str, param2: int) -> bool:
    \"\"\"
    Description of function
    
    Args:
        param1: Description
        param2: Description
        
    Returns:
        Description
    \"\"\"
    pass""",
                "description": "Function with type hints and docstring",
                "confidence": 0.85,
                "priority": 8
            })
        
        # Class definition snippet
        if context.current_line.strip() == "class":
            snippets.append({
                "code": """class ClassName:
    \"\"\"Class description\"\"\"
    
    def __init__(self):
        \"\"\"Initialize the class\"\"\"
        pass""",
                "description": "Class with constructor",
                "confidence": 0.85,
                "priority": 8
            })
        
        # Async function snippet
        if "async" in context.current_line:
            snippets.append({
                "code": """async def async_function():
    \"\"\"Async function\"\"\"
    result = await some_async_operation()
    return result""",
                "description": "Async function template",
                "confidence": 0.75,
                "priority": 6
            })
        
        return snippets
    
    def _generate_preview(
        self,
        context: CodeContext,
        suggestion_text: str
    ) -> str:
        """Generate a preview of code after applying suggestion"""
        # Simple preview - in production would be more sophisticated
        lines = []
        
        # Add previous lines
        if context.previous_lines:
            lines.extend(context.previous_lines[-2:])
        
        # Add current line with suggestion
        current_with_suggestion = context.current_line[:context.cursor_position[1]] + suggestion_text
        lines.append(current_with_suggestion)
        
        # Add next lines
        if context.next_lines:
            lines.extend(context.next_lines[:2])
        
        return "\n".join(lines)
    
    def _rank_suggestions(
        self,
        suggestions: List[CodeSuggestion],
        context: CodeContext
    ) -> List[CodeSuggestion]:
        """Rank suggestions by relevance and quality"""
        
        for suggestion in suggestions:
            # Calculate final score
            score = suggestion.priority * 10
            score += suggestion.confidence * 50
            
            # Boost pattern suggestions if preferred
            if self.user_preferences.get("prefer_patterns") and suggestion.category == "pattern":
                score *= 1.2
            
            # Boost based on source
            if suggestion.source == "knowledge_graph":
                score *= 1.1
            
            # Store score in metadata
            suggestion.metadata["score"] = score
        
        # Sort by score
        suggestions.sort(key=lambda s: s.metadata.get("score", 0), reverse=True)
        
        return suggestions
    
    def _filter_suggestions(
        self,
        suggestions: List[CodeSuggestion]
    ) -> List[CodeSuggestion]:
        """Filter suggestions based on preferences and quality"""
        
        min_confidence = self.user_preferences.get("min_confidence", 0.5)
        
        filtered = []
        seen_texts = set()
        
        for suggestion in suggestions:
            # Filter by confidence
            if suggestion.confidence < min_confidence:
                continue
            
            # Remove duplicates
            if suggestion.text in seen_texts:
                continue
            
            seen_texts.add(suggestion.text)
            filtered.append(suggestion)
        
        return filtered
    
    async def apply_suggestion(
        self,
        suggestion: CodeSuggestion,
        code: str,
        cursor_position: Tuple[int, int]
    ) -> str:
        """
        Apply a suggestion to code
        
        Args:
            suggestion: The suggestion to apply
            code: Current code
            cursor_position: Cursor position
            
        Returns:
            Modified code
        """
        lines = code.split('\n')
        line_num, col_num = cursor_position
        
        if line_num < len(lines):
            line = lines[line_num]
            
            # Insert suggestion at cursor position
            new_line = line[:col_num] + suggestion.text + line[col_num:]
            lines[line_num] = new_line
            
            # Add any imports if needed
            if suggestion.metadata.get("requires_import"):
                import_line = suggestion.metadata["requires_import"]
                # Add import at the top
                lines.insert(0, import_line)
        
        return '\n'.join(lines)
    
    def get_suggestion_stats(self) -> Dict[str, Any]:
        """Get statistics about suggestions"""
        stats = {
            "total_generated": len(self.history),
            "cache_size": len(self.suggestion_cache),
            "preferences": self.user_preferences
        }
        
        if self.history:
            stats["recent_activity"] = self.history[-10:]
        
        return stats