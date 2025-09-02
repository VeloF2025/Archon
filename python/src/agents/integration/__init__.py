"""
Archon Agent Integration Module

Provides integration layers for combining agents with various systems
including confidence scoring, performance monitoring, and workflow orchestration.
"""

from .confidence_integration import (
    ConfidenceIntegration,
    ConfidenceMetrics,
    get_confidence_integration,
    execute_agent_with_confidence
)

__all__ = [
    'ConfidenceIntegration',
    'ConfidenceMetrics',
    'get_confidence_integration',
    'execute_agent_with_confidence'
]