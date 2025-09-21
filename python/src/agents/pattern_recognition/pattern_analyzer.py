"""
Pattern Analysis Engine
Analyzes pattern effectiveness, relationships, and evolution
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PatternAnalysis(BaseModel):
    """Analysis results for patterns"""
    pattern_id: str
    pattern_name: str
    effectiveness_score: float
    frequency_score: float
    evolution_trend: str  # "improving", "declining", "stable"
    related_patterns: List[str]
    common_contexts: List[str]
    recommendations: List[str]
    risk_level: str  # "low", "medium", "high"
    impact_assessment: Dict[str, Any]
    analyzed_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class PatternRelationship(BaseModel):
    """Relationship between patterns"""
    pattern1_id: str
    pattern2_id: str
    relationship_type: str  # "alternative", "complementary", "conflicting", "prerequisite"
    strength: float  # 0.0 to 1.0
    evidence_count: int
    contexts: List[str]


class PatternAnalyzer:
    """Analyzes patterns for insights and recommendations"""
    
    def __init__(self, storage_service):
        self.storage = storage_service
        self.analysis_cache = {}
        self.relationship_graph = defaultdict(list)
    
    async def analyze_pattern(self, pattern_id: str) -> Optional[PatternAnalysis]:
        """
        Comprehensive analysis of a single pattern
        
        Args:
            pattern_id: Pattern to analyze
            
        Returns:
            Pattern analysis results
        """
        try:
            # Get pattern data
            pattern = await self.storage.get_pattern_by_id(pattern_id)
            if not pattern:
                logger.warning(f"Pattern {pattern_id} not found")
                return None
            
            # Calculate effectiveness score
            effectiveness = self._calculate_effectiveness(pattern)
            
            # Calculate frequency score
            frequency_score = self._calculate_frequency_score(pattern)
            
            # Determine evolution trend
            trend = await self._analyze_evolution_trend(pattern_id)
            
            # Find related patterns
            related = await self._find_related_patterns(pattern)
            
            # Identify common contexts
            contexts = self._extract_common_contexts(pattern)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                pattern, effectiveness, trend
            )
            
            # Assess risk level
            risk = self._assess_risk_level(pattern, effectiveness)
            
            # Impact assessment
            impact = self._assess_impact(pattern)
            
            analysis = PatternAnalysis(
                pattern_id=pattern_id,
                pattern_name=pattern.name,
                effectiveness_score=effectiveness,
                frequency_score=frequency_score,
                evolution_trend=trend,
                related_patterns=[p.id for p in related],
                common_contexts=contexts,
                recommendations=recommendations,
                risk_level=risk,
                impact_assessment=impact
            )
            
            # Cache the analysis
            self.analysis_cache[pattern_id] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing pattern {pattern_id}: {e}")
            return None
    
    async def analyze_pattern_relationships(self) -> List[PatternRelationship]:
        """
        Analyze relationships between all patterns
        
        Returns:
            List of pattern relationships
        """
        relationships = []
        
        try:
            # Get all patterns
            all_patterns = await self.storage.get_top_patterns(limit=100)
            
            # Analyze pairwise relationships
            for i, pattern1 in enumerate(all_patterns):
                for pattern2 in all_patterns[i+1:]:
                    relationship = await self._analyze_relationship(pattern1, pattern2)
                    if relationship and relationship.strength > 0.3:
                        relationships.append(relationship)
                        
                        # Update relationship graph
                        self.relationship_graph[pattern1.id].append({
                            "target": pattern2.id,
                            "type": relationship.relationship_type,
                            "strength": relationship.strength
                        })
                        self.relationship_graph[pattern2.id].append({
                            "target": pattern1.id,
                            "type": relationship.relationship_type,
                            "strength": relationship.strength
                        })
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error analyzing relationships: {e}")
            return []
    
    async def get_pattern_insights(
        self, 
        language: Optional[str] = None,
        timeframe_days: int = 30
    ) -> Dict[str, Any]:
        """
        Get aggregated insights about patterns
        
        Args:
            language: Filter by language
            timeframe_days: Analysis timeframe
            
        Returns:
            Insights dictionary
        """
        try:
            # Get patterns
            patterns = await self.storage.get_top_patterns(language=language)
            
            # Get antipatterns
            antipatterns = await self.storage.get_antipatterns(language=language)
            
            # Calculate statistics
            total_patterns = len(patterns)
            total_antipatterns = len(antipatterns)
            
            # Most effective patterns
            effective_patterns = sorted(
                patterns,
                key=lambda p: p.effectiveness_score,
                reverse=True
            )[:5]
            
            # Most frequently used
            frequent_patterns = sorted(
                patterns,
                key=lambda p: p.usage_count,
                reverse=True
            )[:5]
            
            # Category distribution
            category_dist = Counter(p.category for p in patterns)
            
            # Language distribution
            language_dist = Counter(p.language for p in patterns)
            
            # Recent trends
            recent_patterns = [
                p for p in patterns
                if p.created_at and self._is_recent(p.created_at, timeframe_days)
            ]
            
            # Risk assessment
            high_risk_patterns = [
                p for p in patterns
                if await self._is_high_risk(p)
            ]
            
            insights = {
                "summary": {
                    "total_patterns": total_patterns,
                    "total_antipatterns": total_antipatterns,
                    "languages_covered": list(language_dist.keys()),
                    "categories_covered": list(category_dist.keys())
                },
                "effectiveness": {
                    "top_patterns": [
                        {
                            "id": p.id,
                            "name": p.name,
                            "score": p.effectiveness_score,
                            "usage": p.usage_count
                        } for p in effective_patterns
                    ],
                    "average_effectiveness": np.mean([p.effectiveness_score for p in patterns]) if patterns else 0
                },
                "usage": {
                    "most_used": [
                        {
                            "id": p.id,
                            "name": p.name,
                            "count": p.usage_count,
                            "last_used": p.last_used
                        } for p in frequent_patterns
                    ],
                    "total_usage": sum(p.usage_count for p in patterns)
                },
                "distribution": {
                    "by_category": dict(category_dist),
                    "by_language": dict(language_dist)
                },
                "trends": {
                    "new_patterns_count": len(recent_patterns),
                    "new_patterns": [
                        {
                            "id": p.id,
                            "name": p.name,
                            "created": p.created_at
                        } for p in recent_patterns[:5]
                    ]
                },
                "risks": {
                    "high_risk_count": len(high_risk_patterns),
                    "high_risk_patterns": [
                        {
                            "id": p.id,
                            "name": p.name,
                            "risk_factors": await self._get_risk_factors(p)
                        } for p in high_risk_patterns[:5]
                    ]
                },
                "antipatterns": {
                    "count": total_antipatterns,
                    "most_common": [
                        {
                            "id": p.id,
                            "name": p.name,
                            "frequency": p.frequency,
                            "impact": p.metadata.get("performance_impact", "unknown")
                        } for p in antipatterns[:5]
                    ]
                },
                "recommendations": await self._generate_global_recommendations(patterns, antipatterns)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {}
    
    def _calculate_effectiveness(self, pattern) -> float:
        """Calculate pattern effectiveness score"""
        # Base effectiveness from stored score
        effectiveness = pattern.effectiveness_score
        
        # Adjust based on usage
        if pattern.usage_count > 10:
            effectiveness *= 1.1
        elif pattern.usage_count < 3:
            effectiveness *= 0.9
        
        # Penalize antipatterns
        if pattern.metadata.get("is_antipattern"):
            effectiveness *= 0.5
        
        # Consider confidence
        effectiveness *= pattern.confidence
        
        return min(effectiveness, 1.0)
    
    def _calculate_frequency_score(self, pattern) -> float:
        """Calculate normalized frequency score"""
        # Logarithmic scaling for frequency
        if pattern.frequency <= 0:
            return 0.0
        
        score = np.log1p(pattern.frequency) / 10.0  # Normalize to 0-1 range
        return min(score, 1.0)
    
    async def _analyze_evolution_trend(self, pattern_id: str) -> str:
        """Analyze pattern evolution trend"""
        # This would analyze historical data
        # For now, return a simple assessment
        pattern = await self.storage.get_pattern_by_id(pattern_id)
        
        if not pattern:
            return "unknown"
        
        if pattern.effectiveness_score > 0.7 and pattern.usage_count > 5:
            return "improving"
        elif pattern.effectiveness_score < 0.3 or pattern.usage_count < 2:
            return "declining"
        else:
            return "stable"
    
    async def _find_related_patterns(self, pattern) -> List[Any]:
        """Find patterns related to the given pattern"""
        # Use vector similarity to find related patterns
        related = await self.storage.search_similar_patterns(
            query=pattern.name,
            language=pattern.language,
            limit=5,
            threshold=0.6
        )
        
        # Filter out the pattern itself
        related = [p for p in related if p.id != pattern.id]
        
        return related
    
    def _extract_common_contexts(self, pattern) -> List[str]:
        """Extract common usage contexts"""
        contexts = []
        
        # Extract from examples
        if pattern.examples:
            for example in pattern.examples[:5]:
                if isinstance(example, dict):
                    context = example.get("context", example.get("type", "unknown"))
                    if context:
                        contexts.append(str(context))
        
        # Extract from metadata
        if pattern.metadata.get("contexts"):
            contexts.extend(pattern.metadata["contexts"])
        
        return list(set(contexts))[:5]
    
    def _generate_recommendations(self, pattern, effectiveness: float, trend: str) -> List[str]:
        """Generate recommendations for pattern usage"""
        recommendations = []
        
        if pattern.metadata.get("is_antipattern"):
            recommendations.append(f"Avoid this anti-pattern. {pattern.metadata.get('suggested_alternative', 'Consider alternatives')}")
        
        if effectiveness > 0.8:
            recommendations.append("High effectiveness - recommended for use")
        elif effectiveness < 0.3:
            recommendations.append("Low effectiveness - consider alternatives")
        
        if trend == "declining":
            recommendations.append("Pattern usage declining - may be outdated")
        elif trend == "improving":
            recommendations.append("Pattern gaining traction - good adoption candidate")
        
        if pattern.usage_count < 3:
            recommendations.append("Limited usage data - use with caution")
        
        return recommendations
    
    def _assess_risk_level(self, pattern, effectiveness: float) -> str:
        """Assess risk level of using the pattern"""
        if pattern.metadata.get("is_antipattern"):
            return "high"
        
        if effectiveness < 0.3:
            return "high"
        elif effectiveness < 0.6:
            return "medium"
        else:
            return "low"
    
    def _assess_impact(self, pattern) -> Dict[str, Any]:
        """Assess pattern impact on various aspects"""
        impact = {
            "performance": "neutral",
            "maintainability": "neutral",
            "security": "neutral",
            "scalability": "neutral"
        }
        
        # Performance impact
        if pattern.metadata.get("performance_impact"):
            impact["performance"] = pattern.metadata["performance_impact"]
        elif "performance" in pattern.name.lower():
            impact["performance"] = "positive"
        
        # Maintainability impact
        if pattern.category in ["structural", "creational"]:
            impact["maintainability"] = "positive"
        elif pattern.metadata.get("is_antipattern"):
            impact["maintainability"] = "negative"
        
        # Security impact
        if "security" in pattern.name.lower() or "auth" in pattern.name.lower():
            impact["security"] = "positive"
        elif pattern.name.lower() in ["catch_all_exception", "eval"]:
            impact["security"] = "negative"
        
        # Scalability impact
        if pattern.category == "behavioral" and pattern.effectiveness_score > 0.7:
            impact["scalability"] = "positive"
        
        return impact
    
    async def _analyze_relationship(self, pattern1, pattern2) -> Optional[PatternRelationship]:
        """Analyze relationship between two patterns"""
        try:
            # Calculate similarity
            if pattern1.embedding and pattern2.embedding:
                similarity = self._calculate_similarity(pattern1.embedding, pattern2.embedding)
            else:
                similarity = 0.0
            
            # Determine relationship type
            relationship_type = "unrelated"
            
            if pattern1.category == pattern2.category and similarity > 0.7:
                relationship_type = "alternative"
            elif pattern1.language == pattern2.language and similarity > 0.5:
                relationship_type = "complementary"
            elif pattern1.metadata.get("is_antipattern") != pattern2.metadata.get("is_antipattern"):
                relationship_type = "conflicting"
            
            if relationship_type == "unrelated":
                return None
            
            # Find common contexts
            contexts1 = set(self._extract_common_contexts(pattern1))
            contexts2 = set(self._extract_common_contexts(pattern2))
            common_contexts = list(contexts1.intersection(contexts2))
            
            return PatternRelationship(
                pattern1_id=pattern1.id,
                pattern2_id=pattern2.id,
                relationship_type=relationship_type,
                strength=similarity,
                evidence_count=len(common_contexts),
                contexts=common_contexts
            )
            
        except Exception as e:
            logger.error(f"Error analyzing relationship: {e}")
            return None
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def _is_recent(self, date_str: str, days: int) -> bool:
        """Check if date is within recent timeframe"""
        try:
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            cutoff = datetime.utcnow() - timedelta(days=days)
            return date > cutoff
        except:
            return False
    
    async def _is_high_risk(self, pattern) -> bool:
        """Check if pattern is high risk"""
        risk = self._assess_risk_level(pattern, pattern.effectiveness_score)
        return risk == "high"
    
    async def _get_risk_factors(self, pattern) -> List[str]:
        """Get risk factors for a pattern"""
        factors = []
        
        if pattern.metadata.get("is_antipattern"):
            factors.append("Identified as anti-pattern")
        
        if pattern.effectiveness_score < 0.3:
            factors.append("Low effectiveness score")
        
        if pattern.usage_count < 2:
            factors.append("Limited usage data")
        
        if pattern.metadata.get("performance_impact") == "high":
            factors.append("High performance impact")
        
        return factors
    
    async def _generate_global_recommendations(self, patterns: List, antipatterns: List) -> List[str]:
        """Generate global recommendations based on all patterns"""
        recommendations = []
        
        if len(antipatterns) > 10:
            recommendations.append("High number of anti-patterns detected - consider code review")
        
        avg_effectiveness = np.mean([p.effectiveness_score for p in patterns]) if patterns else 0
        if avg_effectiveness < 0.5:
            recommendations.append("Overall pattern effectiveness is low - review pattern usage")
        
        languages = set(p.language for p in patterns)
        if len(languages) > 5:
            recommendations.append("Multiple languages detected - ensure consistent patterns across stack")
        
        unused_patterns = [p for p in patterns if p.usage_count == 0]
        if len(unused_patterns) > 20:
            recommendations.append(f"{len(unused_patterns)} patterns never used - consider cleanup")
        
        return recommendations