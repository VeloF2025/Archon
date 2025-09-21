"""
Pattern Prediction Model
Advanced ML model for predicting and recommending code patterns
Uses temporal analysis, collaborative filtering, and pattern evolution tracking
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import logging
import numpy as np
import json
import hashlib
from collections import defaultdict, Counter

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of patterns for prediction"""
    DESIGN_PATTERN = "design_pattern"
    ARCHITECTURAL_PATTERN = "architectural_pattern"
    CODE_IDIOM = "code_idiom"
    ANTI_PATTERN = "anti_pattern"
    REFACTORING_PATTERN = "refactoring_pattern"
    PERFORMANCE_PATTERN = "performance_pattern"
    SECURITY_PATTERN = "security_pattern"


class PredictionContext(Enum):
    """Context for pattern prediction"""
    CURRENT_CODE = "current_code"
    TEAM_HISTORY = "team_history"
    PROJECT_EVOLUTION = "project_evolution"
    SIMILAR_PROJECTS = "similar_projects"
    INDUSTRY_TRENDS = "industry_trends"


class PatternEvolution(Enum):
    """Pattern evolution trends"""
    EMERGING = "emerging"
    GROWING = "growing"
    STABLE = "stable"
    DECLINING = "declining"
    DEPRECATED = "deprecated"


@dataclass
class PatternUsage:
    """Pattern usage statistics and metadata"""
    pattern_id: str
    pattern_name: str
    pattern_type: PatternType
    usage_count: int
    success_rate: float
    last_used: datetime
    contexts: List[str]
    user_ratings: List[float]
    performance_impact: float
    maintainability_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_name": self.pattern_name,
            "pattern_type": self.pattern_type.value,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "last_used": self.last_used.isoformat(),
            "contexts": self.contexts,
            "user_ratings": self.user_ratings,
            "performance_impact": self.performance_impact,
            "maintainability_score": self.maintainability_score
        }


@dataclass
class PatternPrediction:
    """Prediction for pattern recommendation"""
    pattern_id: str
    pattern_name: str
    pattern_type: PatternType
    confidence_score: float
    relevance_score: float
    prediction_context: PredictionContext
    reasoning: List[str]
    expected_benefits: List[str]
    implementation_effort: str  # "low", "medium", "high"
    risk_factors: List[str]
    similar_cases: List[Dict[str, Any]]
    temporal_trend: PatternEvolution
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_name": self.pattern_name,
            "pattern_type": self.pattern_type.value,
            "confidence_score": self.confidence_score,
            "relevance_score": self.relevance_score,
            "prediction_context": self.prediction_context.value,
            "reasoning": self.reasoning,
            "expected_benefits": self.expected_benefits,
            "implementation_effort": self.implementation_effort,
            "risk_factors": self.risk_factors,
            "similar_cases": self.similar_cases,
            "temporal_trend": self.temporal_trend.value
        }


@dataclass
class PatternSequence:
    """Sequence of patterns for temporal analysis"""
    sequence_id: str
    patterns: List[str]
    timestamps: List[datetime]
    context: Dict[str, Any]
    success_outcome: bool
    performance_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "patterns": self.patterns,
            "timestamps": [ts.isoformat() for ts in self.timestamps],
            "context": self.context,
            "success_outcome": self.success_outcome,
            "performance_metrics": self.performance_metrics
        }


class PatternPredictionRequest(BaseModel):
    """Request for pattern prediction"""
    current_code: str
    file_path: Optional[str] = None
    project_context: Dict[str, Any] = Field(default_factory=dict)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    prediction_types: List[str] = Field(default=["design_pattern", "refactoring_pattern"])
    max_predictions: int = 10
    include_reasoning: bool = True
    context_scope: str = "current_code"


class PatternPredictionModel:
    """
    Advanced ML model for intelligent pattern prediction and recommendation
    Uses collaborative filtering, temporal analysis, and contextual understanding
    """
    
    def __init__(self, pattern_storage=None, knowledge_graph=None):
        self.pattern_storage = pattern_storage
        self.knowledge_graph = knowledge_graph
        
        # Pattern usage tracking
        self.pattern_usage_history: Dict[str, List[PatternUsage]] = defaultdict(list)
        self.pattern_sequences: List[PatternSequence] = []
        self.collaborative_patterns: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Prediction models (simplified implementations)
        self.temporal_model = TemporalPatternAnalyzer()
        self.collaborative_filter = CollaborativePatternFilter()
        self.context_analyzer = PatternContextAnalyzer()
        
        # Pattern database
        self.known_patterns: Dict[str, Dict[str, Any]] = {}
        self.pattern_relationships: Dict[str, List[str]] = defaultdict(list)
        self.pattern_evolution_trends: Dict[str, PatternEvolution] = {}
        
        # Configuration
        self.prediction_threshold = 0.6
        self.temporal_window_days = 30
        self.max_sequence_length = 10
        
        self._initialize_pattern_database()
        
        logger.info("PatternPredictionModel initialized")
    
    def _initialize_pattern_database(self) -> None:
        """Initialize database of known patterns"""
        # Design Patterns
        self.known_patterns.update({
            "singleton": {
                "name": "Singleton Pattern",
                "type": PatternType.DESIGN_PATTERN,
                "description": "Ensure a class has only one instance",
                "benefits": ["Global access", "Memory efficiency", "Lazy initialization"],
                "risks": ["Testing difficulties", "Hidden dependencies"],
                "contexts": ["configuration", "logging", "database_connection"],
                "complexity": "medium"
            },
            "factory": {
                "name": "Factory Pattern",
                "type": PatternType.DESIGN_PATTERN,
                "description": "Create objects without specifying exact classes",
                "benefits": ["Loose coupling", "Extensibility", "Code reuse"],
                "risks": ["Increased complexity", "Indirect object creation"],
                "contexts": ["object_creation", "plugin_systems", "api_responses"],
                "complexity": "medium"
            },
            "observer": {
                "name": "Observer Pattern",
                "type": PatternType.DESIGN_PATTERN,
                "description": "Define one-to-many dependency between objects",
                "benefits": ["Loose coupling", "Dynamic relationships", "Event handling"],
                "risks": ["Memory leaks", "Update cascades"],
                "contexts": ["event_systems", "model_view", "notifications"],
                "complexity": "high"
            },
            "strategy": {
                "name": "Strategy Pattern", 
                "type": PatternType.DESIGN_PATTERN,
                "description": "Define family of algorithms and make them interchangeable",
                "benefits": ["Algorithm flexibility", "Easy testing", "Clean code"],
                "risks": ["Class proliferation", "Client awareness"],
                "contexts": ["algorithms", "sorting", "validation"],
                "complexity": "medium"
            }
        })
        
        # Architectural Patterns
        self.known_patterns.update({
            "mvc": {
                "name": "Model-View-Controller",
                "type": PatternType.ARCHITECTURAL_PATTERN,
                "description": "Separate concerns in application architecture",
                "benefits": ["Separation of concerns", "Testability", "Maintainability"],
                "risks": ["Complexity", "Performance overhead"],
                "contexts": ["web_applications", "user_interfaces"],
                "complexity": "high"
            },
            "microservices": {
                "name": "Microservices Pattern",
                "type": PatternType.ARCHITECTURAL_PATTERN,
                "description": "Decompose application into small services",
                "benefits": ["Scalability", "Technology diversity", "Fault isolation"],
                "risks": ["Network complexity", "Data consistency", "Operational overhead"],
                "contexts": ["distributed_systems", "cloud_native"],
                "complexity": "very_high"
            }
        })
        
        # Refactoring Patterns
        self.known_patterns.update({
            "extract_method": {
                "name": "Extract Method",
                "type": PatternType.REFACTORING_PATTERN,
                "description": "Extract code into a separate method",
                "benefits": ["Code reuse", "Improved readability", "Better testing"],
                "risks": ["Over-fragmentation"],
                "contexts": ["long_methods", "duplicate_code"],
                "complexity": "low"
            },
            "introduce_parameter_object": {
                "name": "Introduce Parameter Object",
                "type": PatternType.REFACTORING_PATTERN,
                "description": "Group parameters into an object",
                "benefits": ["Reduced parameter lists", "Better encapsulation"],
                "risks": ["Additional complexity"],
                "contexts": ["long_parameter_lists", "related_data"],
                "complexity": "medium"
            }
        })
        
        # Anti-patterns
        self.known_patterns.update({
            "god_class": {
                "name": "God Class Anti-pattern",
                "type": PatternType.ANTI_PATTERN,
                "description": "Class that knows too much or does too much",
                "benefits": [],
                "risks": ["Hard to maintain", "Testing difficulties", "Tight coupling"],
                "contexts": ["large_classes", "mixed_responsibilities"],
                "complexity": "high"
            }
        })
        
        # Initialize pattern relationships
        self.pattern_relationships.update({
            "singleton": ["factory", "observer"],
            "factory": ["strategy", "singleton"],
            "observer": ["mvc", "strategy"],
            "mvc": ["observer", "strategy"],
            "extract_method": ["introduce_parameter_object"]
        })
        
        # Initialize evolution trends
        self.pattern_evolution_trends.update({
            "singleton": PatternEvolution.STABLE,
            "factory": PatternEvolution.STABLE,
            "observer": PatternEvolution.GROWING,
            "strategy": PatternEvolution.STABLE,
            "mvc": PatternEvolution.STABLE,
            "microservices": PatternEvolution.GROWING,
            "extract_method": PatternEvolution.STABLE,
            "god_class": PatternEvolution.DECLINING
        })
    
    async def predict_patterns(
        self, 
        request: PatternPredictionRequest
    ) -> List[PatternPrediction]:
        """
        Predict recommended patterns based on current code and context
        """
        try:
            # Analyze current code context
            code_context = await self._analyze_code_context(
                request.current_code, 
                request.project_context
            )
            
            # Generate candidate patterns
            candidates = await self._generate_pattern_candidates(
                code_context, 
                request.prediction_types
            )
            
            # Score and rank patterns
            scored_patterns = []
            for pattern_id in candidates:
                prediction = await self._score_pattern_prediction(
                    pattern_id,
                    code_context,
                    request
                )
                if prediction.confidence_score >= self.prediction_threshold:
                    scored_patterns.append(prediction)
            
            # Sort by combined confidence and relevance
            scored_patterns.sort(
                key=lambda p: (p.confidence_score + p.relevance_score) / 2,
                reverse=True
            )
            
            # Apply temporal and collaborative filtering
            filtered_patterns = await self._apply_advanced_filtering(
                scored_patterns, 
                code_context, 
                request
            )
            
            logger.info(
                f"Generated {len(filtered_patterns)} pattern predictions "
                f"from {len(candidates)} candidates"
            )
            
            return filtered_patterns[:request.max_predictions]
            
        except Exception as e:
            logger.error(f"Error in pattern prediction: {e}")
            return []
    
    async def _analyze_code_context(
        self, 
        code: str, 
        project_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze code to extract contextual information"""
        context = {
            "code_characteristics": {},
            "structural_patterns": [],
            "domain_indicators": [],
            "complexity_metrics": {},
            "existing_patterns": [],
            "anti_pattern_indicators": [],
            "refactoring_opportunities": []
        }
        
        # Basic code analysis
        lines = code.split('\n')
        context["code_characteristics"] = {
            "lines_of_code": len(lines),
            "function_count": code.count('def '),
            "class_count": code.count('class '),
            "complexity_indicators": self._assess_complexity_indicators(code)
        }
        
        # Detect existing patterns
        context["existing_patterns"] = await self._detect_existing_patterns(code)
        
        # Identify anti-patterns
        context["anti_pattern_indicators"] = self._detect_anti_patterns(code)
        
        # Find refactoring opportunities
        context["refactoring_opportunities"] = self._identify_refactoring_opportunities(code)
        
        # Domain analysis
        context["domain_indicators"] = self._identify_domain_context(code, project_context)
        
        return context
    
    def _assess_complexity_indicators(self, code: str) -> Dict[str, Any]:
        """Assess complexity indicators in code"""
        indicators = {
            "long_methods": 0,
            "deep_nesting": 0,
            "long_parameter_lists": 0,
            "duplicate_code_smell": 0,
            "large_classes": 0
        }
        
        lines = code.split('\n')
        current_method_length = 0
        current_class_length = 0
        max_nesting = 0
        current_nesting = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Track method length
            if stripped.startswith('def '):
                if current_method_length > 50:
                    indicators["long_methods"] += 1
                current_method_length = 0
            elif current_method_length >= 0:
                current_method_length += 1
            
            # Track class length
            if stripped.startswith('class '):
                if current_class_length > 300:
                    indicators["large_classes"] += 1
                current_class_length = 0
            elif current_class_length >= 0:
                current_class_length += 1
            
            # Track nesting depth
            indent_level = len(line) - len(line.lstrip())
            current_nesting = indent_level // 4
            max_nesting = max(max_nesting, current_nesting)
            
            # Check parameter lists
            if 'def ' in line and line.count(',') > 5:
                indicators["long_parameter_lists"] += 1
        
        # Final method/class checks
        if current_method_length > 50:
            indicators["long_methods"] += 1
        if current_class_length > 300:
            indicators["large_classes"] += 1
        
        if max_nesting > 4:
            indicators["deep_nesting"] = max_nesting
        
        return indicators
    
    async def _detect_existing_patterns(self, code: str) -> List[str]:
        """Detect existing design patterns in code"""
        detected_patterns = []
        
        # Singleton detection
        if ('__new__' in code and 
            'instance' in code.lower() and 
            'cls' in code):
            detected_patterns.append("singleton")
        
        # Factory detection
        if ('create' in code.lower() and 
            'class' in code and 
            'return' in code):
            detected_patterns.append("factory")
        
        # Observer detection
        if ('notify' in code.lower() and 
            'subscribe' in code.lower() and 
            'observer' in code.lower()):
            detected_patterns.append("observer")
        
        # Strategy detection
        if ('strategy' in code.lower() or 
            ('algorithm' in code.lower() and 'execute' in code.lower())):
            detected_patterns.append("strategy")
        
        return detected_patterns
    
    def _detect_anti_patterns(self, code: str) -> List[str]:
        """Detect anti-patterns in code"""
        anti_patterns = []
        
        lines = code.split('\n')
        
        # God class detection
        class_lines = 0
        method_count = 0
        
        for line in lines:
            if line.strip().startswith('class '):
                class_lines = 0
                method_count = 0
            elif line.strip().startswith('def '):
                method_count += 1
            elif class_lines >= 0:
                class_lines += 1
        
        if class_lines > 500 or method_count > 20:
            anti_patterns.append("god_class")
        
        # Long method detection
        method_length = 0
        for line in lines:
            if line.strip().startswith('def '):
                if method_length > 100:
                    anti_patterns.append("long_method")
                method_length = 0
            else:
                method_length += 1
        
        return anti_patterns
    
    def _identify_refactoring_opportunities(self, code: str) -> List[str]:
        """Identify refactoring opportunities"""
        opportunities = []
        
        complexity_indicators = self._assess_complexity_indicators(code)
        
        if complexity_indicators["long_methods"] > 0:
            opportunities.append("extract_method")
        
        if complexity_indicators["long_parameter_lists"] > 0:
            opportunities.append("introduce_parameter_object")
        
        if complexity_indicators["large_classes"] > 0:
            opportunities.append("extract_class")
        
        if complexity_indicators["deep_nesting"] > 4:
            opportunities.append("reduce_nesting")
        
        # Check for duplicate code
        lines = code.split('\n')
        line_counts = Counter(line.strip() for line in lines if line.strip())
        if any(count > 3 for count in line_counts.values()):
            opportunities.append("extract_common_code")
        
        return opportunities
    
    def _identify_domain_context(
        self, 
        code: str, 
        project_context: Dict[str, Any]
    ) -> List[str]:
        """Identify domain-specific context"""
        domains = []
        code_lower = code.lower()
        
        # Web development
        web_keywords = ['flask', 'django', 'fastapi', 'request', 'response', 'http', 'api']
        if any(keyword in code_lower for keyword in web_keywords):
            domains.append("web_development")
        
        # Data processing
        data_keywords = ['pandas', 'numpy', 'dataframe', 'csv', 'json', 'database']
        if any(keyword in code_lower for keyword in data_keywords):
            domains.append("data_processing")
        
        # Machine learning
        ml_keywords = ['model', 'train', 'predict', 'tensorflow', 'pytorch', 'sklearn']
        if any(keyword in code_lower for keyword in ml_keywords):
            domains.append("machine_learning")
        
        # UI/Frontend
        ui_keywords = ['render', 'template', 'view', 'component', 'state', 'props']
        if any(keyword in code_lower for keyword in ui_keywords):
            domains.append("user_interface")
        
        return domains
    
    async def _generate_pattern_candidates(
        self, 
        code_context: Dict[str, Any], 
        prediction_types: List[str]
    ) -> List[str]:
        """Generate candidate patterns based on context"""
        candidates = set()
        
        # Filter patterns by requested types
        relevant_patterns = {
            pattern_id: pattern_data 
            for pattern_id, pattern_data in self.known_patterns.items()
            if pattern_data["type"].value in prediction_types
        }
        
        # Add patterns based on refactoring opportunities
        for opportunity in code_context.get("refactoring_opportunities", []):
            if opportunity in relevant_patterns:
                candidates.add(opportunity)
        
        # Add patterns based on anti-patterns (suggest fixes)
        anti_patterns = code_context.get("anti_pattern_indicators", [])
        if "god_class" in anti_patterns:
            candidates.update(["extract_class", "facade", "mvc"])
        if "long_method" in anti_patterns:
            candidates.add("extract_method")
        
        # Add patterns based on domain context
        domains = code_context.get("domain_indicators", [])
        if "web_development" in domains:
            candidates.update(["mvc", "facade", "strategy"])
        if "data_processing" in domains:
            candidates.update(["strategy", "template_method", "chain_of_responsibility"])
        
        # Add patterns based on complexity
        complexity = code_context.get("code_characteristics", {}).get("complexity_indicators", {})
        if complexity.get("long_parameter_lists", 0) > 0:
            candidates.add("introduce_parameter_object")
        
        # Add complementary patterns for existing ones
        existing_patterns = code_context.get("existing_patterns", [])
        for existing_pattern in existing_patterns:
            related_patterns = self.pattern_relationships.get(existing_pattern, [])
            candidates.update(related_patterns)
        
        return list(candidates & set(relevant_patterns.keys()))
    
    async def _score_pattern_prediction(
        self,
        pattern_id: str,
        code_context: Dict[str, Any],
        request: PatternPredictionRequest
    ) -> PatternPrediction:
        """Score a pattern prediction"""
        pattern_data = self.known_patterns[pattern_id]
        
        # Calculate confidence score
        confidence_score = await self._calculate_confidence_score(
            pattern_id, code_context, request
        )
        
        # Calculate relevance score
        relevance_score = await self._calculate_relevance_score(
            pattern_id, code_context, request
        )
        
        # Determine prediction context
        prediction_context = self._determine_prediction_context(code_context, request)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(pattern_id, code_context, pattern_data)
        
        # Get expected benefits
        expected_benefits = pattern_data.get("benefits", [])
        
        # Assess implementation effort
        implementation_effort = self._assess_implementation_effort(pattern_id, code_context)
        
        # Identify risk factors
        risk_factors = pattern_data.get("risks", [])
        
        # Find similar cases
        similar_cases = await self._find_similar_cases(pattern_id, code_context)
        
        # Get temporal trend
        temporal_trend = self.pattern_evolution_trends.get(pattern_id, PatternEvolution.STABLE)
        
        return PatternPrediction(
            pattern_id=pattern_id,
            pattern_name=pattern_data["name"],
            pattern_type=pattern_data["type"],
            confidence_score=confidence_score,
            relevance_score=relevance_score,
            prediction_context=prediction_context,
            reasoning=reasoning,
            expected_benefits=expected_benefits,
            implementation_effort=implementation_effort,
            risk_factors=risk_factors,
            similar_cases=similar_cases,
            temporal_trend=temporal_trend
        )
    
    async def _calculate_confidence_score(
        self,
        pattern_id: str,
        code_context: Dict[str, Any],
        request: PatternPredictionRequest
    ) -> float:
        """Calculate confidence score for pattern prediction"""
        confidence = 0.5  # Base confidence
        
        pattern_data = self.known_patterns[pattern_id]
        
        # Context matching
        pattern_contexts = pattern_data.get("contexts", [])
        domain_indicators = code_context.get("domain_indicators", [])
        
        context_match = len(set(pattern_contexts) & set(domain_indicators))
        if context_match > 0:
            confidence += 0.2
        
        # Anti-pattern indicators
        if pattern_data["type"] == PatternType.REFACTORING_PATTERN:
            refactoring_ops = code_context.get("refactoring_opportunities", [])
            if pattern_id in refactoring_ops:
                confidence += 0.3
        
        # Historical success rate
        usage_history = self.pattern_usage_history.get(pattern_id, [])
        if usage_history:
            avg_success_rate = sum(usage.success_rate for usage in usage_history) / len(usage_history)
            confidence += (avg_success_rate - 0.5) * 0.2
        
        # Temporal trends
        trend = self.pattern_evolution_trends.get(pattern_id, PatternEvolution.STABLE)
        if trend == PatternEvolution.GROWING:
            confidence += 0.1
        elif trend == PatternEvolution.DECLINING:
            confidence -= 0.1
        
        return min(1.0, max(0.0, confidence))
    
    async def _calculate_relevance_score(
        self,
        pattern_id: str,
        code_context: Dict[str, Any],
        request: PatternPredictionRequest
    ) -> float:
        """Calculate relevance score for pattern prediction"""
        relevance = 0.5  # Base relevance
        
        pattern_data = self.known_patterns[pattern_id]
        
        # Direct need indicators
        complexity_indicators = code_context.get("code_characteristics", {}).get("complexity_indicators", {})
        
        if pattern_id == "extract_method" and complexity_indicators.get("long_methods", 0) > 0:
            relevance += 0.4
        
        if pattern_id == "introduce_parameter_object" and complexity_indicators.get("long_parameter_lists", 0) > 0:
            relevance += 0.4
        
        # Domain fit
        domains = code_context.get("domain_indicators", [])
        pattern_contexts = pattern_data.get("contexts", [])
        domain_overlap = len(set(domains) & set(pattern_contexts))
        if domain_overlap > 0:
            relevance += 0.2
        
        # Code size appropriateness
        lines_of_code = code_context.get("code_characteristics", {}).get("lines_of_code", 0)
        complexity_level = pattern_data.get("complexity", "medium")
        
        if complexity_level == "high" and lines_of_code < 50:
            relevance -= 0.2  # Too complex for small code
        elif complexity_level == "low" and lines_of_code > 500:
            relevance += 0.1  # Good for large code
        
        return min(1.0, max(0.0, relevance))
    
    def _determine_prediction_context(
        self, 
        code_context: Dict[str, Any], 
        request: PatternPredictionRequest
    ) -> PredictionContext:
        """Determine the context for prediction"""
        if request.context_scope == "team_history":
            return PredictionContext.TEAM_HISTORY
        elif request.context_scope == "project_evolution":
            return PredictionContext.PROJECT_EVOLUTION
        else:
            return PredictionContext.CURRENT_CODE
    
    def _generate_reasoning(
        self,
        pattern_id: str,
        code_context: Dict[str, Any],
        pattern_data: Dict[str, Any]
    ) -> List[str]:
        """Generate reasoning for pattern recommendation"""
        reasoning = []
        
        # Context-based reasoning
        if pattern_id in code_context.get("refactoring_opportunities", []):
            reasoning.append(f"Code shows clear indicators for {pattern_data['name']} refactoring")
        
        # Anti-pattern based reasoning
        anti_patterns = code_context.get("anti_pattern_indicators", [])
        if "god_class" in anti_patterns and pattern_id in ["extract_class", "facade"]:
            reasoning.append("Large class detected - pattern will improve maintainability")
        
        # Complexity-based reasoning
        complexity = code_context.get("code_characteristics", {}).get("complexity_indicators", {})
        if complexity.get("long_methods", 0) > 0 and pattern_id == "extract_method":
            reasoning.append("Long methods detected - extraction will improve readability")
        
        # Domain-based reasoning
        domains = code_context.get("domain_indicators", [])
        pattern_contexts = pattern_data.get("contexts", [])
        if set(domains) & set(pattern_contexts):
            reasoning.append(f"Pattern commonly used in {', '.join(domains)} domain")
        
        return reasoning
    
    def _assess_implementation_effort(
        self, 
        pattern_id: str, 
        code_context: Dict[str, Any]
    ) -> str:
        """Assess implementation effort for pattern"""
        pattern_data = self.known_patterns[pattern_id]
        base_complexity = pattern_data.get("complexity", "medium")
        
        code_size = code_context.get("code_characteristics", {}).get("lines_of_code", 0)
        
        # Adjust based on code size
        if base_complexity == "low":
            return "low"
        elif base_complexity == "medium":
            return "medium" if code_size < 200 else "high"
        elif base_complexity == "high":
            return "high" if code_size < 500 else "very_high"
        else:
            return "medium"
    
    async def _find_similar_cases(
        self, 
        pattern_id: str, 
        code_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find similar cases where pattern was applied"""
        similar_cases = []
        
        # Search through usage history
        usage_history = self.pattern_usage_history.get(pattern_id, [])
        
        current_domains = code_context.get("domain_indicators", [])
        current_complexity = code_context.get("code_characteristics", {}).get("complexity_indicators", {})
        
        for usage in usage_history[-5:]:  # Last 5 cases
            # Simple similarity check
            context_overlap = len(set(usage.contexts) & set(current_domains))
            if context_overlap > 0:
                similar_cases.append({
                    "usage_id": usage.pattern_id,
                    "context": usage.contexts,
                    "success_rate": usage.success_rate,
                    "outcome": "success" if usage.success_rate > 0.7 else "mixed"
                })
        
        return similar_cases
    
    async def _apply_advanced_filtering(
        self,
        patterns: List[PatternPrediction],
        code_context: Dict[str, Any],
        request: PatternPredictionRequest
    ) -> List[PatternPrediction]:
        """Apply advanced filtering using temporal and collaborative data"""
        
        # Temporal filtering
        filtered_patterns = await self.temporal_model.filter_patterns(patterns, code_context)
        
        # Collaborative filtering
        filtered_patterns = await self.collaborative_filter.filter_patterns(
            filtered_patterns, 
            request.user_preferences
        )
        
        return filtered_patterns
    
    async def record_pattern_usage(
        self,
        pattern_id: str,
        usage_context: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> None:
        """Record pattern usage for learning"""
        usage = PatternUsage(
            pattern_id=pattern_id,
            pattern_name=self.known_patterns.get(pattern_id, {}).get("name", "Unknown"),
            pattern_type=self.known_patterns.get(pattern_id, {}).get("type", PatternType.DESIGN_PATTERN),
            usage_count=1,
            success_rate=outcome.get("success_rate", 0.5),
            last_used=datetime.now(timezone.utc),
            contexts=usage_context.get("domains", []),
            user_ratings=outcome.get("user_ratings", []),
            performance_impact=outcome.get("performance_impact", 0.0),
            maintainability_score=outcome.get("maintainability_score", 0.5)
        )
        
        self.pattern_usage_history[pattern_id].append(usage)
        
        # Limit history size
        if len(self.pattern_usage_history[pattern_id]) > 100:
            self.pattern_usage_history[pattern_id] = self.pattern_usage_history[pattern_id][-50:]
    
    async def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about pattern predictions and usage"""
        stats = {
            "total_patterns": len(self.known_patterns),
            "usage_history_size": sum(len(history) for history in self.pattern_usage_history.values()),
            "most_used_patterns": [],
            "trending_patterns": [],
            "pattern_types": defaultdict(int)
        }
        
        # Pattern type distribution
        for pattern_data in self.known_patterns.values():
            stats["pattern_types"][pattern_data["type"].value] += 1
        
        # Most used patterns
        usage_counts = {
            pattern_id: len(history) 
            for pattern_id, history in self.pattern_usage_history.items()
        }
        
        stats["most_used_patterns"] = sorted(
            usage_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        # Trending patterns
        stats["trending_patterns"] = [
            pattern_id for pattern_id, trend in self.pattern_evolution_trends.items()
            if trend == PatternEvolution.GROWING
        ]
        
        stats["pattern_types"] = dict(stats["pattern_types"])
        
        return stats


# Helper classes for advanced filtering

class TemporalPatternAnalyzer:
    """Analyzes temporal patterns in pattern usage"""
    
    async def filter_patterns(
        self, 
        patterns: List[PatternPrediction], 
        code_context: Dict[str, Any]
    ) -> List[PatternPrediction]:
        """Filter patterns based on temporal analysis"""
        # Simple implementation - boost growing patterns, reduce declining ones
        for pattern in patterns:
            if pattern.temporal_trend == PatternEvolution.GROWING:
                pattern.confidence_score *= 1.1
            elif pattern.temporal_trend == PatternEvolution.DECLINING:
                pattern.confidence_score *= 0.9
        
        return patterns


class CollaborativePatternFilter:
    """Applies collaborative filtering to pattern recommendations"""
    
    async def filter_patterns(
        self, 
        patterns: List[PatternPrediction], 
        user_preferences: Dict[str, Any]
    ) -> List[PatternPrediction]:
        """Filter patterns based on collaborative preferences"""
        # Simple implementation - boost patterns based on user preferences
        preferred_types = user_preferences.get("preferred_pattern_types", [])
        
        for pattern in patterns:
            if pattern.pattern_type.value in preferred_types:
                pattern.relevance_score *= 1.2
        
        return patterns


class PatternContextAnalyzer:
    """Analyzes context for pattern applicability"""
    
    def analyze_context(self, code: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context for pattern recommendations"""
        return {
            "suitable_patterns": [],
            "context_strength": 0.5,
            "domain_match": []
        }