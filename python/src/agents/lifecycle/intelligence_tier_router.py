"""
Intelligence Tier Router v3.0 - Smart Claude Model Tier Selection
Based on Archon_3.0_Intelligence_Tiered_Agent_Management_PRD.md

NLNH Protocol: Real complexity assessment with actual task analysis
DGTS Enforcement: No fake complexity scores, actual intelligent routing

User Preference: Sonnet as default tier, Haiku only for basic tasks
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Claude model tiers with capabilities"""
    OPUS = "opus"      # Complex architecture, security analysis, creative problem solving
    SONNET = "sonnet"  # Feature implementation, testing, refactoring, general development (DEFAULT)
    HAIKU = "haiku"    # Only the most basic tasks: formatting, simple fixes, basic docs


@dataclass
class TaskComplexityFactors:
    """Factors that influence task complexity scoring"""
    # Architectural complexity
    requires_architecture_decisions: float = 0.0
    involves_security_compliance: float = 0.0
    cross_system_integration: float = 0.0
    creative_problem_solving: float = 0.0
    
    # Implementation complexity
    multiple_file_changes: float = 0.0
    database_schema_changes: float = 0.0
    api_design_required: float = 0.0
    testing_strategy_needed: float = 0.0
    
    # Business logic complexity
    complex_business_rules: float = 0.0
    data_transformation: float = 0.0
    performance_optimization: float = 0.0
    error_handling_strategy: float = 0.0
    
    # Simple task indicators (negative complexity)
    simple_text_manipulation: float = 0.0
    formatting_only: float = 0.0
    basic_documentation: float = 0.0
    trivial_fix: float = 0.0
    
    # Contextual factors
    user_expertise_level: float = 0.0  # Higher user expertise = can handle more complex solutions
    time_constraint: float = 0.0       # Higher constraint = prefer proven solutions
    risk_tolerance: float = 0.0        # Lower tolerance = prefer safer, simpler approaches


@dataclass
class ComplexityAssessment:
    """Result of task complexity analysis"""
    task_description: str
    complexity_score: float  # 0.0-1.0
    recommended_tier: ModelTier
    confidence: float  # How confident we are in the assessment
    contributing_factors: List[str] = field(default_factory=list)
    reasoning: str = ""
    cost_estimate: Dict[str, float] = field(default_factory=dict)
    alternative_tiers: List[ModelTier] = field(default_factory=list)


class IntelligenceTierRouter:
    """
    Intelligence Tier Router for Archon v3.0
    Routes tasks to appropriate Claude model tiers with Sonnet-first preference
    
    User Preference: Sonnet default, Haiku only for basic tasks
    """
    
    def __init__(self):
        # Updated tier thresholds based on user preference
        self.tier_thresholds = {
            ModelTier.OPUS: 0.75,    # Only for truly complex tasks requiring highest capability
            ModelTier.SONNET: 0.15,  # DEFAULT: Most tasks go to Sonnet (lowered threshold)
            ModelTier.HAIKU: 0.0     # Only for the most basic tasks (raised from -0.4 to 0.0)
        }
        
        # Cost per 1K tokens (approximate Claude pricing)
        self.cost_per_1k_tokens = {
            ModelTier.OPUS: 0.015,   # $15 per 1M tokens
            ModelTier.SONNET: 0.003, # $3 per 1M tokens  
            ModelTier.HAIKU: 0.00025 # $0.25 per 1M tokens
        }
        
        # Task pattern recognition for intelligent routing
        self.complexity_patterns = self._initialize_complexity_patterns()
        
        logger.info("IntelligenceTierRouter initialized with Sonnet-first preference")
        logger.info(f"Tier thresholds: Opus≥{self.tier_thresholds[ModelTier.OPUS]}, "
                   f"Sonnet≥{self.tier_thresholds[ModelTier.SONNET]}, "
                   f"Haiku≥{self.tier_thresholds[ModelTier.HAIKU]}")

    def _initialize_complexity_patterns(self) -> Dict[str, Dict[str, float]]:
        """Initialize patterns for complexity assessment"""
        return {
            # HIGH COMPLEXITY (Opus territory)
            "opus_indicators": {
                "architecture": 0.4, "design": 0.4, "system design": 0.4,
                "security audit": 0.5, "security analysis": 0.5, "vulnerability": 0.4,
                "performance optimization": 0.3, "scalability": 0.3,
                "complex algorithm": 0.4, "data structure design": 0.3,
                "integration strategy": 0.3, "migration plan": 0.3,
                "creative solution": 0.4, "innovative approach": 0.4,
                "compliance": 0.3, "regulatory": 0.3,
                "multi-system": 0.3, "distributed": 0.3,
                "advanced": 0.3, "sophisticated": 0.3
            },
            
            # MEDIUM COMPLEXITY (Sonnet territory - most tasks)
            "sonnet_indicators": {
                "implement": 0.2, "build": 0.2, "create": 0.2, "develop": 0.2,
                "refactor": 0.2, "optimize": 0.2, "improve": 0.2,
                "add feature": 0.25, "new feature": 0.25,
                "api": 0.2, "endpoint": 0.2, "service": 0.2,
                "database": 0.2, "query": 0.15, "model": 0.2,
                "test": 0.2, "testing": 0.2, "validation": 0.2,
                "component": 0.2, "module": 0.2, "class": 0.15,
                "function": 0.15, "method": 0.15,
                "ui": 0.2, "interface": 0.2, "frontend": 0.2,
                "backend": 0.2, "server": 0.2,
                "integration": 0.25, "connect": 0.2,
                "enhancement": 0.2, "upgrade": 0.2,
                "configuration": 0.15, "setup": 0.15
            },
            
            # BASIC TASKS (Haiku territory - only very basic)
            "haiku_indicators": {
                "format": -0.3, "formatting": -0.3, "indent": -0.4,
                "fix typo": -0.4, "typos": -0.4, "spelling": -0.4,
                "add comment": -0.3, "comments": -0.3, "documentation": -0.2,
                "simple fix": -0.3, "basic fix": -0.3, "trivial": -0.4,
                "import": -0.2, "imports": -0.2, "organize imports": -0.3,
                "rename": -0.2, "rename variable": -0.3,
                "console.log": -0.4, "remove logs": -0.4,
                "whitespace": -0.4, "spacing": -0.3,
                "basic update": -0.2, "simple change": -0.3,
                "readme": -0.2, "basic docs": -0.3
            },
            
            # Context modifiers
            "complexity_modifiers": {
                "multiple files": 0.15, "many files": 0.2, "across files": 0.15,
                "breaking change": 0.2, "major change": 0.2,
                "backward compatibility": 0.15, "migration": 0.2,
                "production": 0.1, "critical": 0.2, "urgent": 0.1,
                "experimental": 0.15, "research": 0.2, "prototype": 0.15,
                "legacy": 0.15, "old code": 0.1, "technical debt": 0.15,
                "error handling": 0.1, "exception": 0.1, "edge case": 0.15,
                "async": 0.1, "concurrent": 0.15, "parallel": 0.15,
                "real-time": 0.2, "streaming": 0.15, "websocket": 0.15
            }
        }

    async def assess_task_complexity(
        self, 
        task_description: str, 
        context: Optional[Dict[str, Any]] = None,
        agent_type: Optional[str] = None,
        project_info: Optional[Dict[str, Any]] = None
    ) -> ComplexityAssessment:
        """
        Assess task complexity and recommend appropriate Claude tier
        
        Returns Sonnet by default unless task is clearly Opus-level or basic enough for Haiku
        """
        task_lower = task_description.lower()
        context = context or {}
        
        # Initialize complexity factors
        factors = TaskComplexityFactors()
        contributing_factors = []
        
        # Analyze task description for complexity indicators
        complexity_score = await self._calculate_base_complexity(task_lower, contributing_factors)
        
        # Apply contextual adjustments
        complexity_score = await self._apply_context_adjustments(
            complexity_score, context, task_lower, contributing_factors
        )
        
        # Apply agent-specific adjustments
        if agent_type:
            complexity_score = await self._apply_agent_type_adjustments(
                complexity_score, agent_type, contributing_factors
            )
        
        # Apply project-specific adjustments
        if project_info:
            complexity_score = await self._apply_project_adjustments(
                complexity_score, project_info, contributing_factors
            )
        
        # Determine recommended tier based on updated thresholds
        recommended_tier = await self._determine_tier_from_score(complexity_score)
        
        # Calculate confidence in assessment
        confidence = await self._calculate_confidence(task_description, complexity_score, contributing_factors)
        
        # Generate reasoning
        reasoning = await self._generate_reasoning(
            complexity_score, recommended_tier, contributing_factors
        )
        
        # Estimate costs for all tiers
        cost_estimate = await self._estimate_costs(task_description, recommended_tier)
        
        # Suggest alternative tiers
        alternative_tiers = await self._suggest_alternatives(complexity_score, recommended_tier)
        
        assessment = ComplexityAssessment(
            task_description=task_description,
            complexity_score=complexity_score,
            recommended_tier=recommended_tier,
            confidence=confidence,
            contributing_factors=contributing_factors,
            reasoning=reasoning,
            cost_estimate=cost_estimate,
            alternative_tiers=alternative_tiers
        )
        
        logger.info(f"Task complexity assessment: '{task_description[:50]}...' → "
                   f"{recommended_tier.value} (score: {complexity_score:.3f}, confidence: {confidence:.3f})")
        
        return assessment

    async def _calculate_base_complexity(self, task_lower: str, contributing_factors: List[str]) -> float:
        """Calculate base complexity score from task description"""
        complexity_score = 0.0
        
        # Check for Opus-level indicators
        for indicator, weight in self.complexity_patterns["opus_indicators"].items():
            if indicator in task_lower:
                complexity_score += weight
                contributing_factors.append(f"Opus indicator: {indicator} (+{weight})")
        
        # Check for Sonnet-level indicators
        for indicator, weight in self.complexity_patterns["sonnet_indicators"].items():
            if indicator in task_lower:
                complexity_score += weight
                contributing_factors.append(f"Sonnet indicator: {indicator} (+{weight})")
        
        # Check for basic task indicators (negative complexity)
        for indicator, weight in self.complexity_patterns["haiku_indicators"].items():
            if indicator in task_lower:
                complexity_score += weight  # These are negative values
                contributing_factors.append(f"Basic task indicator: {indicator} ({weight})")
        
        # Apply complexity modifiers
        for modifier, weight in self.complexity_patterns["complexity_modifiers"].items():
            if modifier in task_lower:
                complexity_score += weight
                contributing_factors.append(f"Complexity modifier: {modifier} (+{weight})")
        
        return complexity_score

    async def _apply_context_adjustments(
        self, 
        base_score: float, 
        context: Dict[str, Any], 
        task_lower: str, 
        contributing_factors: List[str]
    ) -> float:
        """Apply contextual adjustments to complexity score"""
        adjusted_score = base_score
        
        # File count adjustment
        file_count = context.get("file_count", 1)
        if file_count > 5:
            adjustment = min(0.2, file_count * 0.02)
            adjusted_score += adjustment
            contributing_factors.append(f"Multiple files ({file_count}): +{adjustment:.3f}")
        
        # Project size adjustment
        project_size = context.get("project_size", "medium")
        size_adjustments = {"small": -0.05, "medium": 0.0, "large": 0.1, "enterprise": 0.15}
        if project_size in size_adjustments:
            adjustment = size_adjustments[project_size]
            if adjustment != 0:
                adjusted_score += adjustment
                contributing_factors.append(f"Project size ({project_size}): {adjustment:+.3f}")
        
        # Urgency adjustment
        urgency = context.get("urgency", "normal")
        if urgency == "high" or "urgent" in task_lower:
            # High urgency slightly favors proven solutions (Sonnet over experimentation)
            adjustment = -0.05
            adjusted_score += adjustment
            contributing_factors.append(f"High urgency: {adjustment:.3f} (favor proven solutions)")
        
        # Risk level adjustment
        risk_level = context.get("risk_level", "medium")
        if risk_level == "high" or "critical" in task_lower or "production" in task_lower:
            # High risk tasks might need more careful consideration
            adjustment = 0.1
            adjusted_score += adjustment
            contributing_factors.append(f"High risk: +{adjustment:.3f}")
        
        return adjusted_score

    async def _apply_agent_type_adjustments(
        self, 
        base_score: float, 
        agent_type: str, 
        contributing_factors: List[str]
    ) -> float:
        """Apply agent type-specific adjustments"""
        adjusted_score = base_score
        
        # Agent type complexity bonuses
        agent_complexity_bonus = {
            "system-architect": 0.2,
            "security-auditor": 0.25,
            "performance-optimizer": 0.15,
            "database-architect": 0.15,
            "devops-engineer": 0.1,
            "api-design-architect": 0.1,
            "code-quality-reviewer": 0.05,
            "test-coverage-validator": 0.05,
            "code-implementer": 0.0,  # Neutral
            "frontend-developer": 0.0,
            "documentation-generator": -0.05,
            "code-refactoring-optimizer": 0.0,
            "ui-ux-optimizer": 0.0,
            "code-formatter": -0.15,  # Usually basic
            "import-organizer": -0.2   # Very basic
        }
        
        if agent_type in agent_complexity_bonus:
            adjustment = agent_complexity_bonus[agent_type]
            if adjustment != 0:
                adjusted_score += adjustment
                contributing_factors.append(f"Agent type ({agent_type}): {adjustment:+.3f}")
        
        return adjusted_score

    async def _apply_project_adjustments(
        self, 
        base_score: float, 
        project_info: Dict[str, Any], 
        contributing_factors: List[str]
    ) -> float:
        """Apply project-specific adjustments"""
        adjusted_score = base_score
        
        # Domain complexity adjustments
        domain = project_info.get("domain", "").lower()
        domain_adjustments = {
            "healthcare": 0.2,    # High compliance requirements
            "finance": 0.2,       # High security requirements
            "ai/ml": 0.15,        # Complex algorithms
            "blockchain": 0.15,   # Complex systems
            "embedded": 0.1,      # Hardware constraints
            "gaming": 0.05,       # Performance requirements
            "ecommerce": 0.05,    # Business logic complexity
            "web": 0.0,           # Standard complexity
            "mobile": 0.0,        # Standard complexity
            "desktop": 0.0        # Standard complexity
        }
        
        for domain_key, adjustment in domain_adjustments.items():
            if domain_key in domain:
                if adjustment != 0:
                    adjusted_score += adjustment
                    contributing_factors.append(f"Domain ({domain_key}): +{adjustment:.3f}")
                break
        
        # Technology stack complexity
        tech_stack = project_info.get("tech_stack", [])
        complex_technologies = [
            "kubernetes", "microservices", "distributed", "blockchain", 
            "machine learning", "ai", "real-time", "streaming", "big data"
        ]
        
        complex_tech_count = sum(1 for tech in tech_stack 
                                if any(complex_term in tech.lower() for complex_term in complex_technologies))
        
        if complex_tech_count > 0:
            adjustment = min(0.15, complex_tech_count * 0.05)
            adjusted_score += adjustment
            contributing_factors.append(f"Complex technologies ({complex_tech_count}): +{adjustment:.3f}")
        
        return adjusted_score

    async def _determine_tier_from_score(self, complexity_score: float) -> ModelTier:
        """Determine recommended tier from complexity score with Sonnet preference"""
        
        # Apply Sonnet-first preference logic
        if complexity_score >= self.tier_thresholds[ModelTier.OPUS]:
            return ModelTier.OPUS
        elif complexity_score >= self.tier_thresholds[ModelTier.SONNET]:
            return ModelTier.SONNET  # Most tasks fall here
        else:
            # Even for low scores, bias toward Sonnet unless clearly basic
            if complexity_score > -0.2:  # Only very negative scores go to Haiku
                return ModelTier.SONNET
            else:
                return ModelTier.HAIKU

    async def _calculate_confidence(
        self, 
        task_description: str, 
        complexity_score: float, 
        contributing_factors: List[str]
    ) -> float:
        """Calculate confidence in the complexity assessment"""
        base_confidence = 0.7
        
        # Higher confidence with more contributing factors
        factor_confidence = min(0.2, len(contributing_factors) * 0.02)
        
        # Higher confidence for clear indicators
        clear_indicators = ["implement", "create", "build", "format", "fix", "security", "architecture"]
        clear_matches = sum(1 for indicator in clear_indicators if indicator in task_description.lower())
        clear_confidence = min(0.15, clear_matches * 0.05)
        
        # Lower confidence for ambiguous tasks
        ambiguous_terms = ["improve", "optimize", "enhance", "better", "handle"]
        ambiguous_matches = sum(1 for term in ambiguous_terms if term in task_description.lower())
        ambiguous_penalty = min(0.1, ambiguous_matches * 0.03)
        
        confidence = base_confidence + factor_confidence + clear_confidence - ambiguous_penalty
        return max(0.3, min(0.95, confidence))

    async def _generate_reasoning(
        self, 
        complexity_score: float, 
        recommended_tier: ModelTier, 
        contributing_factors: List[str]
    ) -> str:
        """Generate human-readable reasoning for the tier recommendation"""
        
        tier_reasoning = {
            ModelTier.OPUS: f"Complex task requiring highest capability (score: {complexity_score:.3f} ≥ {self.tier_thresholds[ModelTier.OPUS]})",
            ModelTier.SONNET: f"Standard development task suitable for Sonnet (score: {complexity_score:.3f}, using Sonnet-first preference)",
            ModelTier.HAIKU: f"Basic task suitable for efficient processing (score: {complexity_score:.3f}, clearly basic operations)"
        }
        
        base_reason = tier_reasoning[recommended_tier]
        
        if contributing_factors:
            top_factors = contributing_factors[:3]  # Show top 3 factors
            factors_text = "; ".join(top_factors)
            return f"{base_reason}. Key factors: {factors_text}"
        
        return base_reason

    async def _estimate_costs(self, task_description: str, recommended_tier: ModelTier) -> Dict[str, float]:
        """Estimate costs for different tiers"""
        # Rough token estimate based on task description length and complexity
        base_tokens = max(1000, len(task_description.split()) * 200)  # Rough estimate
        
        costs = {}
        for tier in ModelTier:
            estimated_tokens = base_tokens
            
            # Adjust token estimate based on tier capability
            if tier == ModelTier.OPUS:
                estimated_tokens *= 1.3  # Opus might provide more detailed responses
            elif tier == ModelTier.HAIKU:
                estimated_tokens *= 0.7  # Haiku provides more concise responses
            
            cost = (estimated_tokens / 1000) * self.cost_per_1k_tokens[tier]
            costs[tier.value] = round(cost, 6)
        
        return costs

    async def _suggest_alternatives(self, complexity_score: float, recommended_tier: ModelTier) -> List[ModelTier]:
        """Suggest alternative tiers based on complexity score"""
        alternatives = []
        
        # If recommended is Sonnet, consider alternatives
        if recommended_tier == ModelTier.SONNET:
            # Could upgrade to Opus if score is close to threshold
            if complexity_score >= self.tier_thresholds[ModelTier.OPUS] - 0.1:
                alternatives.append(ModelTier.OPUS)
            
            # Could downgrade to Haiku if score is very low (but we bias toward Sonnet)
            if complexity_score < 0.1:
                alternatives.append(ModelTier.HAIKU)
        
        # If recommended is Opus, Sonnet is always an alternative
        elif recommended_tier == ModelTier.OPUS:
            alternatives.append(ModelTier.SONNET)
        
        # If recommended is Haiku, Sonnet is always an alternative
        elif recommended_tier == ModelTier.HAIKU:
            alternatives.append(ModelTier.SONNET)
        
        return alternatives

    async def get_tier_statistics(self) -> Dict[str, Any]:
        """Get statistics about tier usage and thresholds"""
        return {
            "tier_thresholds": {tier.value: threshold for tier, threshold in self.tier_thresholds.items()},
            "cost_per_1k_tokens": {tier.value: cost for tier, cost in self.cost_per_1k_tokens.items()},
            "default_preference": "sonnet",
            "routing_philosophy": "Sonnet-first with Haiku only for basic tasks",
            "complexity_patterns_count": {
                "opus_indicators": len(self.complexity_patterns["opus_indicators"]),
                "sonnet_indicators": len(self.complexity_patterns["sonnet_indicators"]),
                "haiku_indicators": len(self.complexity_patterns["haiku_indicators"]),
                "modifiers": len(self.complexity_patterns["complexity_modifiers"])
            }
        }