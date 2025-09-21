"""
Pattern Recommendation Engine
Recommends patterns based on context and historical effectiveness
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PatternRecommendation(BaseModel):
    """Pattern recommendation with context"""
    pattern_id: str
    pattern_name: str
    category: str
    language: str
    relevance_score: float
    confidence: float
    reasoning: str
    example_usage: Optional[str] = None
    alternatives: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    recommended_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ContextAnalysis(BaseModel):
    """Analysis of code context for recommendations"""
    language: str
    framework: Optional[str] = None
    domain: Optional[str] = None  # e.g., "web", "data", "ml"
    complexity_level: str  # "simple", "moderate", "complex"
    existing_patterns: List[str]
    requirements: List[str]
    constraints: List[str]


class PatternRecommender:
    """Intelligent pattern recommendation system"""
    
    def __init__(self, storage_service, analyzer_service):
        self.storage = storage_service
        self.analyzer = analyzer_service
        self.recommendation_cache = {}
    
    async def recommend_patterns(
        self,
        context: str,
        language: str,
        requirements: Optional[List[str]] = None,
        avoid_antipatterns: bool = True,
        limit: int = 5
    ) -> List[PatternRecommendation]:
        """
        Recommend patterns based on context
        
        Args:
            context: Code context or description
            language: Programming language
            requirements: Specific requirements
            avoid_antipatterns: Whether to exclude antipatterns
            limit: Maximum recommendations
            
        Returns:
            List of pattern recommendations
        """
        try:
            # Analyze context
            context_analysis = self._analyze_context(context, language, requirements)
            
            # Search for relevant patterns
            candidate_patterns = await self.storage.search_similar_patterns(
                query=context,
                language=language,
                limit=limit * 3  # Get more candidates for filtering
            )
            
            # Filter antipatterns if requested
            if avoid_antipatterns:
                candidate_patterns = [
                    p for p in candidate_patterns
                    if not p.metadata.get("is_antipattern", False)
                ]
            
            # Score and rank patterns
            recommendations = []
            for pattern in candidate_patterns:
                recommendation = await self._create_recommendation(
                    pattern,
                    context_analysis
                )
                if recommendation:
                    recommendations.append(recommendation)
            
            # Sort by relevance score
            recommendations.sort(key=lambda r: r.relevance_score, reverse=True)
            
            # Limit results
            recommendations = recommendations[:limit]
            
            # Add alternatives and warnings
            for rec in recommendations:
                rec.alternatives = await self._find_alternatives(rec.pattern_id)
                rec.warnings = await self._generate_warnings(rec.pattern_id)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    async def recommend_for_refactoring(
        self,
        code: str,
        language: str,
        focus: Optional[str] = None  # "performance", "maintainability", "security"
    ) -> List[PatternRecommendation]:
        """
        Recommend patterns for code refactoring
        
        Args:
            code: Code to refactor
            language: Programming language
            focus: Refactoring focus area
            
        Returns:
            Refactoring pattern recommendations
        """
        try:
            # Detect existing patterns
            from .pattern_detector import PatternDetector
            detector = PatternDetector()
            existing_patterns = await detector.detect_patterns(code, language)
            
            # Find antipatterns to replace
            antipatterns = [p for p in existing_patterns if p.is_antipattern]
            
            recommendations = []
            
            # Recommend replacements for antipatterns
            for antipattern in antipatterns:
                if antipattern.suggested_alternative:
                    # Find patterns matching the alternative
                    alternative_patterns = await self.storage.search_similar_patterns(
                        query=antipattern.suggested_alternative,
                        language=language,
                        limit=2
                    )
                    
                    for alt_pattern in alternative_patterns:
                        rec = PatternRecommendation(
                            pattern_id=alt_pattern.id,
                            pattern_name=alt_pattern.name,
                            category=alt_pattern.category,
                            language=alt_pattern.language,
                            relevance_score=0.9,  # High relevance for antipattern replacement
                            confidence=alt_pattern.confidence,
                            reasoning=f"Replacement for antipattern: {antipattern.name}",
                            warnings=[f"Current code contains: {antipattern.name}"]
                        )
                        recommendations.append(rec)
            
            # Add focus-specific patterns
            if focus:
                focus_patterns = await self._get_focus_patterns(language, focus)
                for pattern in focus_patterns[:3]:
                    rec = PatternRecommendation(
                        pattern_id=pattern.id,
                        pattern_name=pattern.name,
                        category=pattern.category,
                        language=pattern.language,
                        relevance_score=pattern.effectiveness_score,
                        confidence=pattern.confidence,
                        reasoning=f"Recommended for {focus} improvement"
                    )
                    recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating refactoring recommendations: {e}")
            return []
    
    async def recommend_complementary_patterns(
        self,
        pattern_id: str,
        limit: int = 3
    ) -> List[PatternRecommendation]:
        """
        Recommend patterns that work well with the given pattern
        
        Args:
            pattern_id: Base pattern ID
            limit: Maximum recommendations
            
        Returns:
            Complementary pattern recommendations
        """
        try:
            # Get the base pattern
            base_pattern = await self.storage.get_pattern_by_id(pattern_id)
            if not base_pattern:
                return []
            
            # Get relationships
            relationships = self.analyzer.relationship_graph.get(pattern_id, [])
            
            # Find complementary patterns
            complementary = []
            for rel in relationships:
                if rel["type"] == "complementary" and rel["strength"] > 0.5:
                    pattern = await self.storage.get_pattern_by_id(rel["target"])
                    if pattern:
                        rec = PatternRecommendation(
                            pattern_id=pattern.id,
                            pattern_name=pattern.name,
                            category=pattern.category,
                            language=pattern.language,
                            relevance_score=rel["strength"],
                            confidence=pattern.confidence,
                            reasoning=f"Works well with {base_pattern.name}"
                        )
                        complementary.append(rec)
            
            # If not enough from relationships, find similar patterns
            if len(complementary) < limit:
                similar = await self.storage.search_similar_patterns(
                    query=base_pattern.name,
                    language=base_pattern.language,
                    category=base_pattern.category,
                    limit=limit - len(complementary)
                )
                
                for pattern in similar:
                    if pattern.id != pattern_id:
                        rec = PatternRecommendation(
                            pattern_id=pattern.id,
                            pattern_name=pattern.name,
                            category=pattern.category,
                            language=pattern.language,
                            relevance_score=0.6,
                            confidence=pattern.confidence,
                            reasoning=f"Similar to {base_pattern.name}"
                        )
                        complementary.append(rec)
            
            return complementary[:limit]
            
        except Exception as e:
            logger.error(f"Error finding complementary patterns: {e}")
            return []
    
    async def get_pattern_example(self, pattern_id: str, context: Optional[str] = None) -> Optional[str]:
        """
        Get example usage of a pattern
        
        Args:
            pattern_id: Pattern to get example for
            context: Optional context for example
            
        Returns:
            Example code snippet
        """
        try:
            pattern = await self.storage.get_pattern_by_id(pattern_id)
            if not pattern:
                return None
            
            # Check for stored examples
            if pattern.examples:
                # Return most relevant example
                if context and len(pattern.examples) > 1:
                    # TODO: Select best matching example based on context
                    return str(pattern.examples[0])
                return str(pattern.examples[0])
            
            # Generate example based on pattern type
            return self._generate_example(pattern, context)
            
        except Exception as e:
            logger.error(f"Error getting pattern example: {e}")
            return None
    
    def _analyze_context(
        self,
        context: str,
        language: str,
        requirements: Optional[List[str]] = None
    ) -> ContextAnalysis:
        """Analyze code context for pattern matching"""
        
        # Detect framework
        framework = None
        if "react" in context.lower():
            framework = "react"
        elif "django" in context.lower():
            framework = "django"
        elif "flask" in context.lower():
            framework = "flask"
        elif "express" in context.lower():
            framework = "express"
        
        # Detect domain
        domain = None
        if any(word in context.lower() for word in ["api", "rest", "graphql"]):
            domain = "web"
        elif any(word in context.lower() for word in ["model", "train", "predict"]):
            domain = "ml"
        elif any(word in context.lower() for word in ["database", "query", "orm"]):
            domain = "data"
        
        # Assess complexity
        complexity = "simple"
        if len(context) > 500:
            complexity = "complex"
        elif len(context) > 200:
            complexity = "moderate"
        
        # Extract patterns mentioned
        existing_patterns = []
        pattern_keywords = ["singleton", "factory", "observer", "decorator", "async", "promise"]
        for keyword in pattern_keywords:
            if keyword in context.lower():
                existing_patterns.append(keyword)
        
        # Extract requirements
        if not requirements:
            requirements = []
            if "performance" in context.lower():
                requirements.append("performance")
            if "secure" in context.lower() or "security" in context.lower():
                requirements.append("security")
            if "scale" in context.lower() or "scalable" in context.lower():
                requirements.append("scalability")
        
        # Extract constraints
        constraints = []
        if "legacy" in context.lower():
            constraints.append("legacy_compatibility")
        if "memory" in context.lower():
            constraints.append("memory_constraints")
        if "realtime" in context.lower() or "real-time" in context.lower():
            constraints.append("realtime_requirements")
        
        return ContextAnalysis(
            language=language,
            framework=framework,
            domain=domain,
            complexity_level=complexity,
            existing_patterns=existing_patterns,
            requirements=requirements or [],
            constraints=constraints
        )
    
    async def _create_recommendation(
        self,
        pattern,
        context_analysis: ContextAnalysis
    ) -> Optional[PatternRecommendation]:
        """Create recommendation for a pattern"""
        try:
            # Calculate relevance score
            relevance = self._calculate_relevance(pattern, context_analysis)
            
            # Skip if relevance too low
            if relevance < 0.3:
                return None
            
            # Generate reasoning
            reasoning = self._generate_reasoning(pattern, context_analysis, relevance)
            
            # Get example if highly relevant
            example = None
            if relevance > 0.7:
                example = await self.get_pattern_example(pattern.id)
            
            return PatternRecommendation(
                pattern_id=pattern.id,
                pattern_name=pattern.name,
                category=pattern.category,
                language=pattern.language,
                relevance_score=relevance,
                confidence=pattern.confidence,
                reasoning=reasoning,
                example_usage=example
            )
            
        except Exception as e:
            logger.error(f"Error creating recommendation: {e}")
            return None
    
    def _calculate_relevance(self, pattern, context_analysis: ContextAnalysis) -> float:
        """Calculate pattern relevance to context"""
        relevance = 0.5  # Base relevance
        
        # Language match
        if pattern.language == context_analysis.language:
            relevance += 0.2
        
        # Framework match
        if context_analysis.framework and context_analysis.framework in str(pattern.metadata):
            relevance += 0.15
        
        # Domain match
        if context_analysis.domain:
            if context_analysis.domain == "web" and pattern.category in ["structural", "behavioral"]:
                relevance += 0.1
            elif context_analysis.domain == "ml" and pattern.category in ["creational", "behavioral"]:
                relevance += 0.1
            elif context_analysis.domain == "data" and pattern.category in ["structural", "creational"]:
                relevance += 0.1
        
        # Complexity match
        if context_analysis.complexity_level == "complex" and pattern.category in ["structural", "creational"]:
            relevance += 0.1
        elif context_analysis.complexity_level == "simple" and "simple" in pattern.name.lower():
            relevance += 0.1
        
        # Requirements match
        for req in context_analysis.requirements:
            if req in str(pattern.metadata).lower() or req in pattern.name.lower():
                relevance += 0.1
        
        # Effectiveness bonus
        relevance += pattern.effectiveness_score * 0.15
        
        return min(relevance, 1.0)
    
    def _generate_reasoning(
        self,
        pattern,
        context_analysis: ContextAnalysis,
        relevance: float
    ) -> str:
        """Generate reasoning for recommendation"""
        reasons = []
        
        if relevance > 0.8:
            reasons.append("Highly relevant to your context")
        elif relevance > 0.6:
            reasons.append("Good match for your requirements")
        
        if pattern.effectiveness_score > 0.7:
            reasons.append(f"High effectiveness score ({pattern.effectiveness_score:.2f})")
        
        if pattern.usage_count > 10:
            reasons.append(f"Widely used ({pattern.usage_count} times)")
        
        if context_analysis.framework and context_analysis.framework in str(pattern.metadata):
            reasons.append(f"Compatible with {context_analysis.framework}")
        
        for req in context_analysis.requirements:
            if req in str(pattern.metadata).lower():
                reasons.append(f"Addresses {req} requirement")
        
        if not reasons:
            reasons.append("Potentially useful pattern")
        
        return ". ".join(reasons)
    
    async def _find_alternatives(self, pattern_id: str) -> List[str]:
        """Find alternative patterns"""
        try:
            # Get relationships
            relationships = self.analyzer.relationship_graph.get(pattern_id, [])
            
            alternatives = []
            for rel in relationships:
                if rel["type"] == "alternative":
                    pattern = await self.storage.get_pattern_by_id(rel["target"])
                    if pattern:
                        alternatives.append(pattern.name)
            
            return alternatives[:3]
            
        except Exception as e:
            logger.error(f"Error finding alternatives: {e}")
            return []
    
    async def _generate_warnings(self, pattern_id: str) -> List[str]:
        """Generate warnings for pattern usage"""
        try:
            pattern = await self.storage.get_pattern_by_id(pattern_id)
            if not pattern:
                return []
            
            warnings = []
            
            if pattern.metadata.get("is_antipattern"):
                warnings.append("This is an anti-pattern - use with caution")
            
            if pattern.effectiveness_score < 0.5:
                warnings.append("Low effectiveness score")
            
            if pattern.usage_count < 3:
                warnings.append("Limited usage data available")
            
            if pattern.metadata.get("performance_impact") == "high":
                warnings.append("May have significant performance impact")
            
            if pattern.metadata.get("deprecated"):
                warnings.append("Pattern is deprecated")
            
            return warnings
            
        except Exception as e:
            logger.error(f"Error generating warnings: {e}")
            return []
    
    async def _get_focus_patterns(self, language: str, focus: str) -> List[Any]:
        """Get patterns for specific focus area"""
        try:
            # Search for patterns related to focus
            query = f"{focus} optimization best practices"
            patterns = await self.storage.search_similar_patterns(
                query=query,
                language=language,
                limit=5
            )
            
            # Filter by effectiveness
            patterns = [p for p in patterns if p.effectiveness_score > 0.6]
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error getting focus patterns: {e}")
            return []
    
    def _generate_example(self, pattern, context: Optional[str] = None) -> str:
        """Generate example code for pattern"""
        # This would generate language-specific examples
        # For now, return a placeholder
        
        examples = {
            "singleton": """
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
""",
            "factory": """
class Factory:
    @staticmethod
    def create_object(type_name):
        if type_name == "TypeA":
            return TypeA()
        elif type_name == "TypeB":
            return TypeB()
        raise ValueError(f"Unknown type: {type_name}")
""",
            "decorator": """
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper
""",
            "context_manager": """
class ResourceManager:
    def __enter__(self):
        self.resource = acquire_resource()
        return self.resource
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        release_resource(self.resource)
"""
        }
        
        # Try to find matching example
        for key, example in examples.items():
            if key in pattern.name.lower():
                return example.strip()
        
        return f"# Example for {pattern.name}\n# Context: {context or 'General usage'}"