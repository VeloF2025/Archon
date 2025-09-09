"""
Pattern Recognition and Extraction System

This module provides intelligent pattern recognition capabilities that analyze
crawled projects to extract reusable architectural patterns and generate
templates for the community marketplace.
"""

from .pattern_models import (
    Pattern, PatternMetadata, PatternType, PatternComplexity, PatternCategory,
    PatternSearchRequest, PatternSubmission, PatternValidationResult,
    PatternRecommendation, PatternProvider, MultiProviderConfig
)

from .pattern_analyzer import ProjectStructureAnalyzer, PatternAnalyzer
from .pattern_validator import CommunityPatternValidator
from .multi_provider_engine import MultiProviderEngine

__all__ = [
    'Pattern', 'PatternMetadata', 'PatternType', 'PatternComplexity', 'PatternCategory',
    'PatternSearchRequest', 'PatternSubmission', 'PatternValidationResult',
    'PatternRecommendation', 'PatternProvider', 'MultiProviderConfig',
    'ProjectStructureAnalyzer', 'PatternAnalyzer', 'CommunityPatternValidator', 'MultiProviderEngine'
]