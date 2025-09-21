"""
Predictive Assistant Module
Provides intelligent code suggestions and completions
"""

from .predictor import CodePredictor
from .suggestion_engine import SuggestionEngine
from .context_analyzer import ContextAnalyzer
from .completion_provider import CompletionProvider

__all__ = [
    'CodePredictor',
    'SuggestionEngine',
    'ContextAnalyzer',
    'CompletionProvider'
]