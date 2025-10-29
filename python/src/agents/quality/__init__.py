"""
Quality Assurance Module

Implements Actor/Critic pattern and quality gates for automated
quality control without human review of every step.
"""

from .actor_critic import (
    ActorCriticSystem,
    ActorOutput,
    CriticReview,
    QualityGate,
    QualityMetric,
    QualityLevel,
    CriticDecision,
    ActorCriticCycle,
    start_quality_cycle,
    evaluate_actor_output,
    register_metric_evaluator,
    CODE_QUALITY_GATE,
    SECURITY_GATE,
    DOCUMENTATION_GATE,
)

__all__ = [
    # Main system
    "ActorCriticSystem",
    # Models
    "ActorOutput",
    "CriticReview",
    "QualityGate",
    "QualityMetric",
    "QualityLevel",
    "CriticDecision",
    "ActorCriticCycle",
    # Functions
    "start_quality_cycle",
    "evaluate_actor_output",
    "register_metric_evaluator",
    # Predefined gates
    "CODE_QUALITY_GATE",
    "SECURITY_GATE",
    "DOCUMENTATION_GATE",
]
