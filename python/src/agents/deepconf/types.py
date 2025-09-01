"""
DeepConf Type Definitions
Shared types to prevent circular imports between engine.py and storage.py
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum
import time

class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate" 
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

class ConfidenceFactorType(Enum):
    """Confidence factors that influence scoring"""
    TECHNICAL_COMPLEXITY = "technical_complexity"
    DOMAIN_EXPERTISE = "domain_expertise"
    DATA_AVAILABILITY = "data_availability"
    MODEL_CAPABILITY = "model_capability"
    UNCERTAINTY = "uncertainty"
    HISTORICAL_PERFORMANCE = "historical_performance"
    CONTEXT_RICHNESS = "context_richness"

@dataclass
class ConfidenceFactor:
    """Individual confidence factor"""
    name: str
    importance: float
    impact: str  # 'positive', 'negative', 'neutral'
    description: str
    evidence: List[str]

@dataclass
class ConfidenceScore:
    """
    Comprehensive confidence score structure matching PRD requirements
    """
    # Multi-dimensional confidence components (PRD 4.1)
    overall_confidence: float
    factual_confidence: float
    reasoning_confidence: float
    contextual_confidence: float
    
    # Uncertainty quantification (PRD 4.1)
    epistemic_uncertainty: float  # Knowledge uncertainty
    aleatoric_uncertainty: float  # Data uncertainty
    uncertainty_bounds: tuple[float, float]
    
    # Confidence analysis
    confidence_factors: Dict[str, float]
    primary_factors: List[str]
    confidence_reasoning: str
    
    # Metadata
    model_source: str
    timestamp: float
    task_id: str
    calibration_applied: bool = False
    gaming_detection_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        from dataclasses import asdict
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfidenceScore':
        """Create from dictionary"""
        return cls(**data)

@dataclass
class ConfidenceExplanation:
    """Confidence explanation structure for transparency"""
    primary_factors: List[Dict[str, Any]]
    confidence_reasoning: str
    uncertainty_sources: List[str]
    improvement_suggestions: List[str]
    factor_importance_ranking: Dict[str, float]