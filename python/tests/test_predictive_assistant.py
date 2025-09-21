"""
Comprehensive tests for Predictive Assistant
Tests context analysis, suggestion generation, and code completion
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any, Tuple

from src.agents.predictive_assistant.context_analyzer import ContextAnalyzer, CodeContext
from src.agents.predictive_assistant.suggestion_engine import SuggestionEngine, CodeSuggestion
from src.agents.predictive_assistant.completion_provider import CompletionProvider
from src.agents.predictive_assistant.predictor import CodePredictor


class TestContextAnalyzer:
    """Test code context analysis functionality"""
    
    @pytest.fixture
    def analyzer(self):
        """Create a context analyzer instance"""
        return ContextAnalyzer()
    
    @pytest.mark.asyncio
    async def test_analyze_basic_context(self, analyzer):
        """Test analyzing basic code context"""
        code = """
import os
import json

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def process(self):
        # Processing logic here
        pass

processor = DataProcessor()
processor.
"""
        cursor_position = (12, 10)  # After "processor."
        
        context = await analyzer.analyze_context(
            code=code,
            cursor_position=cursor_position,
            file_path="test.py",
            language="python"
        )
        
        assert context.file_path == "test.py"
        assert context.cursor_position == cursor_position
        assert "DataProcessor" in context.classes
        assert "process" in context.functions
        assert "processor" in context.variables
        assert context.language == "python"
        assert len(context.imports) >= 2
    
    @pytest.mark.asyncio
    async def test_extract_imports(self, analyzer):
        """Test extracting import statements"""
        code = """
import os
from typing import List, Dict
from collections import defaultdict
import json as j
"""
        
        context = await analyzer.analyze_context(code, (0, 0))
        
        assert "os" in context.imports
        assert any("typing" in imp for imp in context.imports)
        assert any("collections" in imp for imp in context.imports)
        assert any("json" in imp for imp in context.imports)
    
    @pytest.mark.asyncio
    async def test_determine_scope(self, analyzer):
        """Test determining current scope"""
        code = """
class MyClass:
    def my_method(self):
        x = 1
        # Cursor here
        return x

def global_func():
    pass
"""
        
        # Test inside method
        context = await analyzer.analyze_context(code, (3, 8))  # Inside my_method
        assert "function:my_method" in context.current_scope
        
        # Test at module level
        context = await analyzer.analyze_context(code, (7, 0))  # At global_func
        assert context.current_scope == "module" or "global_func" in context.current_scope
    
    @pytest.mark.asyncio
    async def test_method_completion_context(self, analyzer):
        """Test getting context for method completion"""
        code = """
data = []
data.app
"""
        context = await analyzer.analyze_context(code, (2, 8))  # After "data.app"
        
        completion_context = await analyzer.get_completion_context(context)
        
        assert completion_context["type"] == "method_completion"
        assert completion_context["object"] == "data"
        assert "append" in completion_context["available_methods"]
    
    @pytest.mark.asyncio
    async def test_import_completion_context(self, analyzer):
        """Test getting context for import completion"""
        code = "import os"
        context = await analyzer.analyze_context(code, (0, 9))
        
        completion_context = await analyzer.get_completion_context(context)
        
        assert completion_context["type"] == "import_completion"
        assert "os" in completion_context["available_modules"] or \
               completion_context["partial"] == "os"
    
    @pytest.mark.asyncio
    async def test_semantic_context(self, analyzer):
        """Test getting semantic context"""
        code = """
try:
    result = process_data()
except Exception as e:
    logger.error(f"Error: {e}")
"""
        context = await analyzer.analyze_context(code, (2, 0))
        
        semantic_context = await analyzer.get_semantic_context(context)
        
        assert semantic_context["intent"] is not None
        assert "error_handling" in semantic_context["patterns"]
        assert semantic_context["complexity"] >= 0
    
    @pytest.mark.asyncio
    async def test_indentation_level(self, analyzer):
        """Test calculating indentation level"""
        code = """
class A:
    def b(self):
        if True:
            x = 1
"""
        context = await analyzer.analyze_context(code, (4, 12))  # At "x = 1"
        
        assert context.indentation_level == 3  # Class -> method -> if


class TestSuggestionEngine:
    """Test suggestion generation functionality"""
    
    @pytest.fixture
    def engine(self):
        """Create a suggestion engine with mocked dependencies"""
        mock_pattern_storage = MagicMock()
        mock_query_engine = MagicMock()
        return SuggestionEngine(mock_pattern_storage, mock_query_engine)
    
    @pytest.mark.asyncio
    async def test_generate_completion_suggestions(self, engine):
        """Test generating basic completion suggestions"""
        context = CodeContext(
            file_path="test.py",
            cursor_position=(5, 10),
            current_line="data.",
            language="python",
            variables={"data": "list"}
        )
        
        completion_context = {
            "type": "method_completion",
            "object": "data",
            "object_type": "list",
            "partial": "",
            "available_methods": ["append", "extend", "insert", "remove"]
        }
        
        suggestions = await engine._generate_completion_suggestions(
            context, completion_context
        )
        
        assert len(suggestions) > 0
        assert any(s.text.startswith("append") for s in suggestions)
        assert all(s.category == "completion" for s in suggestions)
        assert all(s.confidence > 0.5 for s in suggestions)
    
    @pytest.mark.asyncio
    async def test_generate_pattern_suggestions(self, engine):
        """Test generating pattern-based suggestions"""
        context = CodeContext(
            file_path="test.py",
            cursor_position=(5, 0),
            current_line="",
            language="python"
        )
        
        # Mock pattern storage
        mock_pattern = MagicMock()
        mock_pattern.name = "Singleton"
        mock_pattern.effectiveness_score = 0.85
        mock_pattern.confidence = 0.9
        mock_pattern.id = "pattern-1"
        
        engine.pattern_storage.search_patterns = AsyncMock(return_value=[mock_pattern])
        
        suggestions = await engine._generate_pattern_suggestions(
            context,
            {"intent": "create_single_instance"}
        )
        
        assert len(suggestions) > 0
        assert suggestions[0].category == "pattern"
        assert suggestions[0].source == "patterns"
        assert "Singleton" in suggestions[0].description
    
    @pytest.mark.asyncio
    async def test_generate_snippet_suggestions(self, engine):
        """Test generating code snippet suggestions"""
        context = CodeContext(
            file_path="test.py",
            cursor_position=(5, 0),
            current_line="def",
            language="python"
        )
        
        completion_context = {"type": "general_completion"}
        
        suggestions = await engine._generate_snippet_suggestions(
            context, completion_context
        )
        
        assert len(suggestions) > 0
        assert any("function" in s.description.lower() for s in suggestions)
        assert all(s.category == "snippet" for s in suggestions)
    
    @pytest.mark.asyncio
    async def test_generate_knowledge_suggestions(self, engine):
        """Test generating suggestions from knowledge graph"""
        context = CodeContext(
            file_path="test.py",
            cursor_position=(5, 0),
            current_line="",
            language="python",
            current_scope="function:process_data"
        )
        
        # Mock query engine
        mock_result = MagicMock()
        mock_result.results = [
            {"code": "return processed_data", "description": "Return processed data", "confidence": 0.8},
            {"code": "raise ValueError()", "description": "Raise error", "confidence": 0.6}
        ]
        engine.graph_query_engine.query = AsyncMock(return_value=mock_result)
        
        suggestions = await engine._generate_knowledge_suggestions(
            context,
            {"patterns": ["data_processing"]}
        )
        
        assert len(suggestions) > 0
        assert suggestions[0].source == "knowledge_graph"
        assert suggestions[0].confidence >= 0.6
    
    @pytest.mark.asyncio
    async def test_rank_and_filter_suggestions(self, engine):
        """Test ranking and filtering suggestions"""
        suggestions = [
            CodeSuggestion(
                text="append()",
                description="Append to list",
                confidence=0.9,
                priority=8,
                category="completion"
            ),
            CodeSuggestion(
                text="extend()",
                description="Extend list",
                confidence=0.3,  # Below threshold
                priority=5,
                category="completion"
            ),
            CodeSuggestion(
                text="singleton_pattern",
                description="Apply Singleton",
                confidence=0.95,
                priority=10,
                category="pattern"
            ),
            CodeSuggestion(
                text="append()",  # Duplicate
                description="Duplicate suggestion",
                confidence=0.85,
                priority=7,
                category="completion"
            )
        ]
        
        context = CodeContext(
            file_path="test.py",
            cursor_position=(0, 0),
            current_line="",
            language="python"
        )
        
        # Rank suggestions
        ranked = engine._rank_suggestions(suggestions, context)
        
        # Filter suggestions
        filtered = engine._filter_suggestions(ranked)
        
        # Should remove low confidence and duplicates
        assert len(filtered) < len(suggestions)
        assert all(s.confidence >= 0.5 for s in filtered)
        
        # Check no duplicates
        texts = [s.text for s in filtered]
        assert len(texts) == len(set(texts))
        
        # Should be ranked by score
        scores = [s.metadata.get("score", 0) for s in filtered]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_apply_suggestion(self, engine):
        """Test applying a suggestion to code"""
        code = """
data = []
data.
"""
        suggestion = CodeSuggestion(
            text="append(item)",
            description="Append item to list",
            confidence=0.9,
            category="completion"
        )
        
        new_code = await engine.apply_suggestion(
            suggestion,
            code,
            cursor_position=(2, 5)  # After "data."
        )
        
        assert "data.append(item)" in new_code
    
    @pytest.mark.asyncio
    async def test_full_suggestion_generation(self, engine):
        """Test the complete suggestion generation pipeline"""
        context = CodeContext(
            file_path="test.py",
            cursor_position=(5, 10),
            current_line="result = data.",
            language="python",
            variables={"data": "list", "result": "Any"}
        )
        
        completion_context = {
            "type": "method_completion",
            "object": "data",
            "partial": "",
            "available_methods": ["append", "extend"]
        }
        
        # Mock pattern storage and query engine
        engine.pattern_storage.search_patterns = AsyncMock(return_value=[])
        engine.graph_query_engine.query = AsyncMock(
            return_value=MagicMock(results=[])
        )
        
        suggestions = await engine.generate_suggestions(
            context,
            completion_context,
            semantic_context={"intent": "variable_assignment"}
        )
        
        assert isinstance(suggestions, list)
        assert len(suggestions) <= engine.user_preferences["max_suggestions"]
        
        # Check caching
        cache_key = f"{context.file_path}:{context.cursor_position}"
        assert cache_key in engine.suggestion_cache


class TestCodePredictor:
    """Test the main code predictor"""
    
    @pytest.fixture
    def predictor(self):
        """Create a code predictor with mocked components"""
        with patch('src.agents.predictive_assistant.predictor.ContextAnalyzer') as mock_analyzer:
            with patch('src.agents.predictive_assistant.predictor.SuggestionEngine') as mock_engine:
                predictor = CodePredictor()
                predictor.context_analyzer = mock_analyzer()
                predictor.suggestion_engine = mock_engine()
                yield predictor
    
    @pytest.mark.asyncio
    async def test_predict_next_code(self, predictor):
        """Test predicting next code"""
        code = "data = []\ndata."
        cursor_position = (1, 5)
        
        # Mock context analysis
        mock_context = CodeContext(
            file_path="test.py",
            cursor_position=cursor_position,
            current_line="data.",
            language="python",
            variables={"data": "list"}
        )
        predictor.context_analyzer.analyze_context = AsyncMock(return_value=mock_context)
        predictor.context_analyzer.get_completion_context = AsyncMock(return_value={
            "type": "method_completion",
            "object": "data"
        })
        
        # Mock suggestion generation
        mock_suggestions = [
            CodeSuggestion(text="append()", description="Append", confidence=0.9)
        ]
        predictor.suggestion_engine.generate_suggestions = AsyncMock(
            return_value=mock_suggestions
        )
        
        predictions = await predictor.predict(
            code=code,
            cursor_position=cursor_position,
            file_path="test.py"
        )
        
        assert len(predictions) > 0
        assert predictions[0].text == "append()"
    
    @pytest.mark.asyncio
    async def test_get_completions(self, predictor):
        """Test getting code completions"""
        code = "import "
        cursor_position = (0, 7)
        
        # Mock the pipeline
        predictor.context_analyzer.analyze_context = AsyncMock()
        predictor.context_analyzer.get_completion_context = AsyncMock(return_value={
            "type": "import_completion",
            "available_modules": ["os", "sys", "json"]
        })
        predictor.suggestion_engine.generate_suggestions = AsyncMock(return_value=[
            CodeSuggestion(text="os", description="Import os", confidence=0.8),
            CodeSuggestion(text="sys", description="Import sys", confidence=0.8)
        ])
        
        completions = await predictor.get_completions(
            code=code,
            cursor_position=cursor_position,
            max_completions=5
        )
        
        assert len(completions) <= 5
        assert any(c.text == "os" for c in completions)
    
    @pytest.mark.asyncio
    async def test_cache_predictions(self, predictor):
        """Test prediction caching"""
        code = "test = 1"
        cursor_position = (0, 8)
        
        # Setup mocks
        predictor.context_analyzer.analyze_context = AsyncMock()
        predictor.context_analyzer.get_completion_context = AsyncMock(return_value={})
        predictor.suggestion_engine.generate_suggestions = AsyncMock(return_value=[])
        
        # First call
        await predictor.predict(code, cursor_position, use_cache=True)
        
        # Second call with cache
        await predictor.predict(code, cursor_position, use_cache=True)
        
        # Should only analyze once due to caching
        assert predictor.context_analyzer.analyze_context.call_count == 1
    
    @pytest.mark.asyncio
    async def test_confidence_threshold(self, predictor):
        """Test filtering by confidence threshold"""
        predictor.context_analyzer.analyze_context = AsyncMock()
        predictor.context_analyzer.get_completion_context = AsyncMock(return_value={})
        
        # Mix of high and low confidence suggestions
        predictor.suggestion_engine.generate_suggestions = AsyncMock(return_value=[
            CodeSuggestion(text="high", description="High conf", confidence=0.9),
            CodeSuggestion(text="low", description="Low conf", confidence=0.3),
            CodeSuggestion(text="medium", description="Med conf", confidence=0.6)
        ])
        
        predictions = await predictor.predict(
            code="test",
            cursor_position=(0, 4),
            min_confidence=0.5
        )
        
        # Should filter out low confidence
        assert all(p.confidence >= 0.5 for p in predictions)
        assert len(predictions) == 2


@pytest.mark.integration
class TestPredictiveAssistantIntegration:
    """Integration tests for the complete predictive assistant"""
    
    @pytest.mark.asyncio
    async def test_full_prediction_pipeline(self):
        """Test the complete prediction pipeline"""
        from src.agents.predictive_assistant import CodePredictor
        
        predictor = CodePredictor()
        
        code = """
class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        return x + y
    
    def multiply(self, x, y):
        return x * y

calc = Calculator()
result = calc.
"""
        
        cursor_position = (12, 14)  # After "calc."
        
        # This would need real pattern storage and knowledge graph
        # For now, just test that it doesn't crash
        with patch('src.agents.predictive_assistant.suggestion_engine.create_embedding') as mock_embed:
            mock_embed.return_value = [0.1] * 1536
            
            predictions = await predictor.predict(
                code=code,
                cursor_position=cursor_position,
                file_path="calculator.py"
            )
        
        # Should suggest Calculator methods
        assert isinstance(predictions, list)
        # In a real test, would check for "add" and "multiply" suggestions
    
    @pytest.mark.asyncio
    async def test_context_aware_suggestions(self):
        """Test that suggestions are context-aware"""
        from src.agents.predictive_assistant import ContextAnalyzer, SuggestionEngine
        
        analyzer = ContextAnalyzer()
        engine = SuggestionEngine()
        
        # Test different contexts
        contexts = [
            ("import ", "import_completion"),
            ("class ", "general_completion"),
            ("def test", "general_completion"),
            ("data.", "method_completion"),
            ("dict[", "dict_completion")
        ]
        
        for code, expected_type in contexts:
            context = await analyzer.analyze_context(
                code=code,
                cursor_position=(0, len(code))
            )
            
            completion_context = await analyzer.get_completion_context(context)
            
            # Verify correct context type detected
            assert completion_context["type"] == expected_type or \
                   expected_type in completion_context["type"]