"""
Code Predictor for Predictive Assistant
Main predictor that coordinates context analysis and suggestion generation
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
import hashlib
from pydantic import BaseModel, Field

from .context_analyzer import ContextAnalyzer, CodeContext
from .suggestion_engine import SuggestionEngine, CodeSuggestion
from .completion_provider import CompletionProvider, CompletionRequest

logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """Request for code prediction"""
    code: str
    cursor_position: Tuple[int, int]
    file_path: Optional[str] = None
    language: str = "python"
    context_lines: int = 10
    max_predictions: int = 10
    min_confidence: float = 0.5


class PredictionResult(BaseModel):
    """Result of code prediction"""
    predictions: List[CodeSuggestion]
    context: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time: float = 0.0


class CodePredictor:
    """Main code predictor that coordinates all components"""
    
    def __init__(
        self,
        pattern_storage=None,
        graph_query_engine=None
    ):
        self.context_analyzer = ContextAnalyzer()
        self.suggestion_engine = SuggestionEngine(
            pattern_storage=pattern_storage,
            graph_query_engine=graph_query_engine
        )
        self.completion_provider = CompletionProvider(
            context_analyzer=self.context_analyzer,
            suggestion_engine=self.suggestion_engine
        )
        
        self.prediction_cache = {}
        self.cache_ttl = 60  # Cache for 60 seconds
        self.prediction_history = []
    
    async def predict(
        self,
        code: str,
        cursor_position: Tuple[int, int],
        file_path: Optional[str] = None,
        language: str = "python",
        min_confidence: float = 0.5,
        max_predictions: int = 10,
        use_cache: bool = True
    ) -> List[CodeSuggestion]:
        """
        Predict next code suggestions
        
        Args:
            code: Current code
            cursor_position: Cursor position (line, column)
            file_path: Optional file path
            language: Programming language
            min_confidence: Minimum confidence threshold
            max_predictions: Maximum number of predictions
            use_cache: Whether to use cached predictions
            
        Returns:
            List of code suggestions
        """
        start_time = datetime.utcnow()
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(code, cursor_position)
            cached = self._get_cached_prediction(cache_key)
            if cached:
                logger.debug(f"Using cached predictions for {cache_key}")
                return cached
        
        try:
            # Analyze context
            context = await self.context_analyzer.analyze_context(
                code=code,
                cursor_position=cursor_position,
                file_path=file_path,
                language=language
            )
            
            # Get completion context
            completion_context = await self.context_analyzer.get_completion_context(context)
            
            # Get semantic context
            semantic_context = await self.context_analyzer.get_semantic_context(context)
            
            # Generate suggestions
            suggestions = await self.suggestion_engine.generate_suggestions(
                context=context,
                completion_context=completion_context,
                semantic_context=semantic_context
            )
            
            # Filter by confidence
            filtered = [s for s in suggestions if s.confidence >= min_confidence]
            
            # Limit predictions
            predictions = filtered[:max_predictions]
            
            # Cache results
            if use_cache:
                self._cache_prediction(cache_key, predictions)
            
            # Track history
            self._track_prediction(context, predictions)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Generated {len(predictions)} predictions in {execution_time:.2f}s")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return []
    
    async def predict_next_token(
        self,
        code: str,
        cursor_position: Tuple[int, int],
        beam_width: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Predict next token with probabilities
        
        Args:
            code: Current code
            cursor_position: Cursor position
            beam_width: Number of beams for prediction
            
        Returns:
            List of (token, probability) tuples
        """
        predictions = await self.predict(
            code=code,
            cursor_position=cursor_position,
            max_predictions=beam_width
        )
        
        # Convert to token predictions
        tokens = []
        for pred in predictions:
            # Extract first token from suggestion
            token = pred.text.split()[0] if pred.text else ""
            probability = pred.confidence
            tokens.append((token, probability))
        
        return tokens
    
    async def predict_line_completion(
        self,
        code: str,
        cursor_position: Tuple[int, int]
    ) -> Optional[str]:
        """
        Predict completion for current line
        
        Args:
            code: Current code
            cursor_position: Cursor position
            
        Returns:
            Line completion or None
        """
        predictions = await self.predict(
            code=code,
            cursor_position=cursor_position,
            max_predictions=1,
            min_confidence=0.7
        )
        
        if predictions:
            return predictions[0].text
        
        return None
    
    async def get_completions(
        self,
        code: str,
        cursor_position: Tuple[int, int],
        trigger_character: Optional[str] = None,
        max_completions: int = 10
    ) -> List[CodeSuggestion]:
        """
        Get code completions (IDE-style)
        
        Args:
            code: Current code
            cursor_position: Cursor position
            trigger_character: Character that triggered completion
            max_completions: Maximum completions
            
        Returns:
            List of completions
        """
        request = CompletionRequest(
            code=code,
            cursor_position=cursor_position,
            trigger_character=trigger_character,
            max_completions=max_completions
        )
        
        response = await self.completion_provider.provide_completions(request)
        return response.completions
    
    async def predict_with_context(
        self,
        request: PredictionRequest
    ) -> PredictionResult:
        """
        Predict with full context information
        
        Args:
            request: Prediction request
            
        Returns:
            Prediction result with metadata
        """
        start_time = datetime.utcnow()
        
        # Get predictions
        predictions = await self.predict(
            code=request.code,
            cursor_position=request.cursor_position,
            file_path=request.file_path,
            language=request.language,
            min_confidence=request.min_confidence,
            max_predictions=request.max_predictions
        )
        
        # Get context information
        context = await self.context_analyzer.analyze_context(
            code=request.code,
            cursor_position=request.cursor_position,
            file_path=request.file_path,
            language=request.language
        )
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return PredictionResult(
            predictions=predictions,
            context={
                "scope": context.current_scope,
                "imports": context.imports,
                "variables": context.variables,
                "indentation": context.indentation_level
            },
            metadata={
                "total_predictions": len(predictions),
                "language": request.language,
                "cache_hit": False
            },
            execution_time=execution_time
        )
    
    async def learn_from_feedback(
        self,
        suggestion: CodeSuggestion,
        accepted: bool,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Learn from user feedback on suggestions
        
        Args:
            suggestion: The suggestion that was shown
            accepted: Whether user accepted the suggestion
            context: Additional context
        """
        feedback_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "suggestion": suggestion.dict(),
            "accepted": accepted,
            "context": context or {}
        }
        
        # Store feedback for learning
        if hasattr(self.suggestion_engine, 'pattern_storage') and self.suggestion_engine.pattern_storage:
            # Update pattern effectiveness based on feedback
            if "pattern_id" in suggestion.metadata:
                pattern_id = suggestion.metadata["pattern_id"]
                
                # Increase effectiveness if accepted, decrease if rejected
                current_score = suggestion.metadata.get("effectiveness", 0.5)
                new_score = min(1.0, current_score + 0.05) if accepted else max(0.0, current_score - 0.05)
                
                await self.suggestion_engine.pattern_storage.update_pattern_effectiveness(
                    pattern_id,
                    new_score
                )
        
        logger.info(f"Feedback recorded: {suggestion.text[:30]}... -> {'accepted' if accepted else 'rejected'}")
    
    def _get_cache_key(
        self,
        code: str,
        cursor_position: Tuple[int, int]
    ) -> str:
        """Generate cache key for prediction"""
        # Use hash of code around cursor for cache key
        lines = code.split('\n')
        line_num, col_num = cursor_position
        
        # Get context window
        start = max(0, line_num - 5)
        end = min(len(lines), line_num + 5)
        context_lines = lines[start:end]
        
        context_str = f"{'\n'.join(context_lines)}:{cursor_position}"
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def _get_cached_prediction(
        self,
        cache_key: str
    ) -> Optional[List[CodeSuggestion]]:
        """Get cached prediction if available and not expired"""
        if cache_key in self.prediction_cache:
            entry = self.prediction_cache[cache_key]
            age = (datetime.utcnow() - entry["timestamp"]).total_seconds()
            
            if age < self.cache_ttl:
                return entry["predictions"]
            else:
                # Remove expired entry
                del self.prediction_cache[cache_key]
        
        return None
    
    def _cache_prediction(
        self,
        cache_key: str,
        predictions: List[CodeSuggestion]
    ):
        """Cache predictions"""
        self.prediction_cache[cache_key] = {
            "predictions": predictions,
            "timestamp": datetime.utcnow()
        }
        
        # Limit cache size
        if len(self.prediction_cache) > 100:
            # Remove oldest entries
            sorted_keys = sorted(
                self.prediction_cache.keys(),
                key=lambda k: self.prediction_cache[k]["timestamp"]
            )
            for key in sorted_keys[:20]:
                del self.prediction_cache[key]
    
    def _track_prediction(
        self,
        context: CodeContext,
        predictions: List[CodeSuggestion]
    ):
        """Track prediction in history"""
        self.prediction_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "file_path": context.file_path,
            "scope": context.current_scope,
            "predictions_count": len(predictions),
            "top_prediction": predictions[0].text if predictions else None
        })
        
        # Limit history size
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-500:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get predictor statistics"""
        stats = {
            "cache_size": len(self.prediction_cache),
            "history_size": len(self.prediction_history),
            "suggestion_stats": self.suggestion_engine.get_suggestion_stats()
        }
        
        if self.prediction_history:
            # Calculate average predictions per request
            total_predictions = sum(h["predictions_count"] for h in self.prediction_history)
            stats["avg_predictions"] = total_predictions / len(self.prediction_history)
            
            # Most common scopes
            from collections import Counter
            scopes = Counter(h["scope"] for h in self.prediction_history)
            stats["common_scopes"] = dict(scopes.most_common(5))
        
        return stats
    
    def clear_cache(self):
        """Clear all caches"""
        self.prediction_cache.clear()
        self.completion_provider.clear_cache()
        logger.info("All prediction caches cleared")