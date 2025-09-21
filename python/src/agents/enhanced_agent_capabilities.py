"""
Enhanced Agent Capabilities - Detailed agent capability modeling.

This module provides comprehensive capability modeling for agents,
including expertise tracking, performance monitoring, and capability evolution.

Key Features:
- Detailed capability taxonomy and classification
- Expertise level tracking and calibration
- Performance-based capability refinement
- Dynamic capability updates
- Capability gap analysis
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
from collections import defaultdict

from ...agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class CapabilityCategory(Enum):
    """Categories of agent capabilities."""
    TECHNICAL = "technical"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    COMMUNICATION = "communication"
    MANAGEMENT = "management"
    DOMAIN = "domain"
    TOOLS = "tools"
    SECURITY = "security"


class CapabilityLevel(Enum):
    """Levels of capability expertise."""
    NOVICE = "novice"           # 0.0 - 0.3
    INTERMEDIATE = "intermediate"  # 0.3 - 0.6
    ADVANCED = "advanced"      # 0.6 - 0.8
    EXPERT = "expert"          # 0.8 - 1.0


@dataclass
class CapabilityDefinition:
    """Definition of a specific capability."""
    name: str
    category: CapabilityCategory
    description: str
    keywords: List[str]
    complexity: float  # 0-1, where 1 is most complex
    rarity: float     # 0-1, where 1 is rarest
    related_capabilities: List[str] = field(default_factory=list)


@dataclass
class AgentCapabilityProfile:
    """Comprehensive capability profile for an agent."""
    agent_name: str
    capabilities: Dict[str, float]  # capability_name -> expertise_level
    capability_history: Dict[str, List[Tuple[datetime, float]]] = field(default_factory=dict)
    performance_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    learning_rate: float = 0.1
    last_updated: datetime = field(default_factory=datetime.utcnow)
    confidence_thresholds: Dict[str, float] = field(default_factory=dict)

    def get_expertise_level(self, capability_name: str) -> CapabilityLevel:
        """Get expertise level for a capability."""
        expertise = self.capabilities.get(capability_name, 0.0)

        if expertise < 0.3:
            return CapabilityLevel.NOVICE
        elif expertise < 0.6:
            return CapabilityLevel.INTERMEDIATE
        elif expertise < 0.8:
            return CapabilityLevel.ADVANCED
        else:
            return CapabilityLevel.EXPERT

    def update_capability(self, capability_name: str, new_expertise: float, context: Dict[str, Any] = None) -> None:
        """Update capability expertise level."""
        current_expertise = self.capabilities.get(capability_name, 0.0)

        # Apply learning rate for smooth updates
        updated_expertise = current_expertise + self.learning_rate * (new_expertise - current_expertise)
        updated_expertise = max(0.0, min(1.0, updated_expertise))  # Clamp to [0, 1]

        self.capabilities[capability_name] = updated_expertise

        # Record history
        if capability_name not in self.capability_history:
            self.capability_history[capability_name] = []

        self.capability_history[capability_name].append((datetime.utcnow(), updated_expertise))

        # Keep history manageable
        if len(self.capability_history[capability_name]) > 100:
            self.capability_history[capability_name] = self.capability_history[capability_name][-50:]

        self.last_updated = datetime.utcnow()

        # Update confidence threshold
        self._update_confidence_threshold(capability_name, updated_expertise)

    def _update_confidence_threshold(self, capability_name: str, expertise: float) -> None:
        """Update confidence threshold based on expertise level."""
        # Higher expertise = higher confidence threshold
        base_threshold = 0.5 + (expertise * 0.4)  # 0.5 to 0.9
        self.confidence_thresholds[capability_name] = base_threshold

    def get_capability_trend(self, capability_name: str, days: int = 30) -> float:
        """Get trend for a capability over specified days."""
        if capability_name not in self.capability_history:
            return 0.0

        cutoff_time = datetime.utcnow() - timedelta(days=days)
        recent_history = [
            expertise for timestamp, expertise in self.capability_history[capability_name]
            if timestamp > cutoff_time
        ]

        if len(recent_history) < 2:
            return 0.0

        # Calculate trend using linear regression
        x_values = list(range(len(recent_history)))
        y_values = recent_history

        try:
            slope = self._calculate_slope(x_values, y_values)
            return slope
        except:
            return 0.0

    def _calculate_slope(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate slope for trend analysis."""
        n = len(x_values)
        if n < 2:
            return 0.0

        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(y_values)

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def get_top_capabilities(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get top capabilities by expertise level."""
        sorted_capabilities = sorted(
            self.capabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_capabilities[:limit]

    def get_capability_gaps(self, required_capabilities: Set[str]) -> List[Tuple[str, float]]:
        """Get gaps between current and required capabilities."""
        gaps = []

        for capability in required_capabilities:
            current_level = self.capabilities.get(capability, 0.0)
            gap = 1.0 - current_level
            gaps.append((capability, gap))

        # Sort by gap size
        gaps.sort(key=lambda x: x[1], reverse=True)

        return gaps


class CapabilityTaxonomy:
    """Taxonomy of all possible capabilities."""

    def __init__(self):
        """Initialize capability taxonomy."""
        self.capabilities: Dict[str, CapabilityDefinition] = {}
        self._initialize_taxonomy()

    def _initialize_taxonomy(self) -> None:
        """Initialize the capability taxonomy."""
        # Technical capabilities
        self._add_capability(
            "programming",
            CapabilityCategory.TECHNICAL,
            "Writing and debugging code in various programming languages",
            ["code", "program", "develop", "implement", "debug"],
            complexity=0.8,
            rarity=0.3
        )

        self._add_capability(
            "web_development",
            CapabilityCategory.TECHNICAL,
            "Building web applications and websites",
            ["web", "frontend", "backend", "html", "css", "javascript"],
            complexity=0.7,
            rarity=0.4
        )

        self._add_capability(
            "database_management",
            CapabilityCategory.TECHNICAL,
            "Database design, implementation, and optimization",
            ["database", "sql", "query", "schema", "optimization"],
            complexity=0.7,
            rarity=0.5
        )

        self._add_capability(
            "system_architecture",
            CapabilityCategory.TECHNICAL,
            "Designing complex software systems and architectures",
            ["architecture", "design", "system", "scalable", "distributed"],
            complexity=0.9,
            rarity=0.8
        )

        # Analytical capabilities
        self._add_capability(
            "data_analysis",
            CapabilityCategory.ANALYTICAL,
            "Analyzing data and extracting insights",
            ["analyze", "data", "insights", "statistics", "metrics"],
            complexity=0.6,
            rarity=0.4
        )

        self._add_capability(
            "problem_solving",
            CapabilityCategory.ANALYTICAL,
            "Identifying and solving complex problems",
            ["problem", "solve", "solution", "debug", "troubleshoot"],
            complexity=0.8,
            rarity=0.2
        )

        self._add_capability(
            "research",
            CapabilityCategory.ANALYTICAL,
            "Conducting research and gathering information",
            ["research", "investigate", "gather", "information", "study"],
            complexity=0.5,
            rarity=0.3
        )

        # Creative capabilities
        self._add_capability(
            "design_thinking",
            CapabilityCategory.CREATIVE,
            "Creative problem-solving and design methodologies",
            ["design", "creative", "innovative", "ideation", "prototype"],
            complexity=0.7,
            rarity=0.6
        )

        self._add_capability(
            "writing",
            CapabilityCategory.CREATIVE,
            "Creating written content and documentation",
            ["write", "document", "content", "manual", "guide"],
            complexity=0.4,
            rarity=0.3
        )

        # Communication capabilities
        self._add_capability(
            "communication",
            CapabilityCategory.COMMUNICATION,
            "Clear and effective communication",
            ["communicate", "explain", "present", "discuss", "collaborate"],
            complexity=0.3,
            rarity=0.1
        )

        self._add_capability(
            "teaching",
            CapabilityCategory.COMMUNICATION,
            "Educating and explaining complex concepts",
            ["teach", "explain", "train", "mentor", "guide"],
            complexity=0.6,
            rarity=0.5
        )

        # Management capabilities
        self._add_capability(
            "project_management",
            CapabilityCategory.MANAGEMENT,
            "Planning and managing projects",
            ["manage", "plan", "coordinate", "schedule", "organize"],
            complexity=0.7,
            rarity=0.4
        )

        self._add_capability(
            "team_coordination",
            CapabilityCategory.MANAGEMENT,
            "Coordinating team efforts and collaboration",
            ["coordinate", "team", "collaborate", "delegate", "lead"],
            complexity=0.6,
            rarity=0.5
        )

        # Security capabilities
        self._add_capability(
            "security_analysis",
            CapabilityCategory.SECURITY,
            "Analyzing security vulnerabilities and implementing safeguards",
            ["security", "vulnerability", "encryption", "authentication", "compliance"],
            complexity=0.9,
            rarity=0.8
        )

        # Domain capabilities
        self._add_capability(
            "machine_learning",
            CapabilityCategory.DOMAIN,
            "Machine learning and AI development",
            ["ml", "ai", "model", "training", "algorithm"],
            complexity=0.9,
            rarity=0.7
        )

        self._add_capability(
            "devops",
            CapabilityCategory.DOMAIN,
            "DevOps practices and infrastructure management",
            ["devops", "ci/cd", "deployment", "infrastructure", "automation"],
            complexity=0.8,
            rarity=0.6
        )

        # Tools capabilities
        self._add_capability(
            "testing",
            CapabilityCategory.TOOLS,
            "Software testing and quality assurance",
            ["test", "qa", "unit", "integration", "automation"],
            complexity=0.6,
            rarity=0.4
        )

        self._add_capability(
            "version_control",
            CapabilityCategory.TOOLS,
            "Version control systems and collaboration tools",
            ["git", "svn", "version", "branch", "merge"],
            complexity=0.4,
            rarity=0.2
        )

    def _add_capability(
        self,
        name: str,
        category: CapabilityCategory,
        description: str,
        keywords: List[str],
        complexity: float,
        rarity: float
    ) -> None:
        """Add a capability to the taxonomy."""
        self.capabilities[name] = CapabilityDefinition(
            name=name,
            category=category,
            description=description,
            keywords=keywords,
            complexity=complexity,
            rarity=rarity
        )

    def get_capability(self, name: str) -> Optional[CapabilityDefinition]:
        """Get capability definition by name."""
        return self.capabilities.get(name)

    def find_capabilities_by_keywords(self, text: str) -> List[CapabilityDefinition]:
        """Find capabilities that match keywords in text."""
        text_lower = text.lower()
        matching_capabilities = []

        for capability in self.capabilities.values():
            if any(keyword in text_lower for keyword in capability.keywords):
                matching_capabilities.append(capability)

        return matching_capabilities

    def get_capabilities_by_category(self, category: CapabilityCategory) -> List[CapabilityDefinition]:
        """Get all capabilities in a category."""
        return [cap for cap in self.capabilities.values() if cap.category == category]

    def get_rare_capabilities(self, threshold: float = 0.7) -> List[CapabilityDefinition]:
        """Get capabilities above rarity threshold."""
        return [cap for cap in self.capabilities.values() if cap.rarity >= threshold]


class EnhancedAgentCapabilitySystem:
    """
    Enhanced system for managing agent capabilities.

    This system provides:
    - Comprehensive capability taxonomy
    - Dynamic capability tracking
    - Performance-based learning
    - Gap analysis and recommendations
    """

    def __init__(self):
        """Initialize the enhanced capability system."""
        self.taxonomy = CapabilityTaxonomy()
        self.agent_profiles: Dict[str, AgentCapabilityProfile] = {}
        self.capability_demand: Dict[str, int] = defaultdict(int)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def initialize_agent_profile(self, agent: BaseAgent) -> AgentCapabilityProfile:
        """Initialize capability profile for an agent."""
        profile = AgentCapabilityProfile(
            agent_name=agent.name,
            capabilities={},
            performance_metrics={}
        )

        # Extract initial capabilities from agent metadata
        initial_capabilities = self._extract_initial_capabilities(agent)
        for capability, level in initial_capabilities.items():
            profile.update_capability(capability, level)

        self.agent_profiles[agent.name] = profile
        self.logger.info(f"Initialized capability profile for {agent.name}")

        return profile

    def _extract_initial_capabilities(self, agent: BaseAgent) -> Dict[str, float]:
        """Extract initial capabilities from agent metadata."""
        capabilities = {}

        # Extract from agent class name
        class_name = agent.__class__.__name__.lower()

        # Basic capability mapping based on class name
        if any(term in class_name for term in ["developer", "coder", "programmer"]):
            capabilities["programming"] = 0.8
            capabilities["web_development"] = 0.6

        if any(term in class_name for term in ["analyst", "researcher"]):
            capabilities["data_analysis"] = 0.7
            capabilities["research"] = 0.8

        if any(term in class_name for term in ["designer", "architect"]):
            capabilities["design_thinking"] = 0.7
            capabilities["system_architecture"] = 0.6

        if any(term in class_name for term in ["tester", "qa"]):
            capabilities["testing"] = 0.9
            capabilities["problem_solving"] = 0.7

        if any(term in class_name for term in ["manager", "coordinator"]):
            capabilities["project_management"] = 0.7
            capabilities["team_coordination"] = 0.8

        if any(term in class_name for term in ["security"]):
            capabilities["security_analysis"] = 0.8

        # Default capabilities for all agents
        capabilities["communication"] = 0.6
        capabilities["problem_solving"] = 0.5

        return capabilities

    def update_agent_capabilities(
        self,
        agent_name: str,
        task_outcome: bool,
        task_capabilities: Set[str],
        execution_time: float,
        confidence: float
    ) -> None:
        """Update agent capabilities based on task performance."""
        if agent_name not in self.agent_profiles:
            self.logger.warning(f"Agent profile not found: {agent_name}")
            return

        profile = self.agent_profiles[agent_name]

        # Update demand tracking
        for capability in task_capabilities:
            self.capability_demand[capability] += 1

        # Calculate performance factor
        performance_factor = 1.0
        if task_outcome:
            performance_factor += 0.1  # Success bonus
        else:
            performance_factor -= 0.2  # Failure penalty

        # Time efficiency factor
        if execution_time < 30:  # Fast execution
            performance_factor += 0.05
        elif execution_time > 120:  # Slow execution
            performance_factor -= 0.05

        # Confidence factor
        if confidence > 0.8:
            performance_factor += 0.05
        elif confidence < 0.5:
            performance_factor -= 0.05

        # Update capabilities
        for capability in task_capabilities:
            if capability in profile.capabilities:
                current_expertise = profile.capabilities[capability]
                change = (performance_factor - 1.0) * 0.1  # Scale change
                new_expertise = max(0.0, min(1.0, current_expertise + change))
                profile.update_capability(capability, new_expertise)

        # Update performance metrics
        profile.performance_metrics["recent_tasks"] = profile.performance_metrics.get("recent_tasks", 0) + 1
        if task_outcome:
            profile.performance_metrics["successful_tasks"] = profile.performance_metrics.get("successful_tasks", 0) + 1

        self.logger.debug(f"Updated capabilities for {agent_name}: {task_capabilities}")

    def get_best_agent_for_task(
        self,
        required_capabilities: Set[str],
        available_agents: List[str],
        strategy: str = "balanced"
    ) -> Optional[str]:
        """
        Find the best agent for a set of required capabilities.

        Args:
            required_capabilities: Set of required capabilities
            available_agents: List of available agent names
            strategy: Selection strategy ("balanced", "expertise", "availability")

        Returns:
            Name of best agent or None
        """
        if not available_agents:
            return None

        agent_scores = {}

        for agent_name in available_agents:
            if agent_name not in self.agent_profiles:
                continue

            profile = self.agent_profiles[agent_name]
            score = self._calculate_agent_score(profile, required_capabilities, strategy)
            agent_scores[agent_name] = score

        if not agent_scores:
            return None

        # Return agent with highest score
        return max(agent_scores, key=agent_scores.get)

    def _calculate_agent_score(
        self,
        profile: AgentCapabilityProfile,
        required_capabilities: Set[str],
        strategy: str
    ) -> float:
        """Calculate agent score based on strategy."""
        if strategy == "expertise":
            # Focus on highest expertise
            expertise_scores = []
            for capability in required_capabilities:
                expertise = profile.capabilities.get(capability, 0.0)
                expertise_scores.append(expertise)

            return max(expertise_scores) if expertise_scores else 0.0

        elif strategy == "availability":
            # Focus on current availability (inverse of recent usage)
            recent_tasks = profile.performance_metrics.get("recent_tasks", 0)
            availability_score = max(0.0, 1.0 - (recent_tasks / 100.0))
            return availability_score

        else:  # balanced
            # Balanced approach considering expertise and coverage
            expertise_sum = sum(
                profile.capabilities.get(capability, 0.0)
                for capability in required_capabilities
            )

            coverage = len([
                cap for cap in required_capabilities
                if cap in profile.capabilities and profile.capabilities[cap] > 0.5
            ]) / len(required_capabilities) if required_capabilities else 0

            balanced_score = (expertise_sum / len(required_capabilities)) * 0.7 + coverage * 0.3
            return balanced_score

    def analyze_capability_gaps(self) -> Dict[str, Any]:
        """Analyze capability gaps across all agents."""
        all_capabilities = set(self.taxonomy.capabilities.keys())
        current_capabilities = set()

        for profile in self.agent_profiles.values():
            current_capabilities.update(profile.capabilities.keys())

        missing_capabilities = all_capabilities - current_capabilities
        underutilized_capabilities = []

        for capability in all_capabilities:
            if capability in self.capability_demand and capability in current_capabilities:
                demand = self.capability_demand[capability]
                avg_expertise = statistics.mean([
                    profile.capabilities.get(capability, 0.0)
                    for profile in self.agent_profiles.values()
                    if capability in profile.capabilities
                ])

                if demand > 10 and avg_expertise < 0.5:
                    underutilized_capabilities.append({
                        "capability": capability,
                        "demand": demand,
                        "average_expertise": avg_expertise
                    })

        return {
            "missing_capabilities": list(missing_capabilities),
            "underutilized_capabilities": underutilized_capabilities,
            "total_capabilities": len(all_capabilities),
            "covered_capabilities": len(current_capabilities)
        }

    def get_capability_recommendations(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get capability development recommendations for an agent."""
        if agent_name not in self.agent_profiles:
            return []

        profile = self.agent_profiles[agent_name]
        recommendations = []

        # Identify areas for improvement
        for capability, expertise in profile.capabilities.items():
            trend = profile.get_capability_trend(capability)
            level = profile.get_expertise_level(capability)

            if trend < -0.01:  # Declining capability
                recommendations.append({
                    "capability": capability,
                    "current_level": level.value,
                    "trend": "declining",
                    "priority": "high",
                    "suggestion": f"Practice {capability} skills to reverse declining trend"
                })

            elif expertise < 0.5 and level != CapabilityLevel.EXPERT:
                recommendations.append({
                    "capability": capability,
                    "current_level": level.value,
                    "trend": "stable",
                    "priority": "medium",
                    "suggestion": f"Develop {capability} skills to reach intermediate level"
                })

        # Suggest new capabilities based on demand
        high_demand_capabilities = [
            cap for cap, demand in self.capability_demand.items()
            if demand > 5 and cap not in profile.capabilities
        ]

        for capability in high_demand_capabilities[:3]:  # Top 3
            recommendations.append({
                "capability": capability,
                "current_level": "none",
                "trend": "new",
                "priority": "low",
                "suggestion": f"Consider developing {capability} skills (high demand)"
            })

        return recommendations

    def get_agent_capability_summary(self, agent_name: str) -> Dict[str, Any]:
        """Get comprehensive capability summary for an agent."""
        if agent_name not in self.agent_profiles:
            return {"error": f"Agent {agent_name} not found"}

        profile = self.agent_profiles[agent_name]

        # Group capabilities by category
        capabilities_by_category = defaultdict(list)
        for capability_name, expertise in profile.capabilities.items():
            capability_def = self.taxonomy.get_capability(capability_name)
            if capability_def:
                category = capability_def.category.value
                capabilities_by_category[category].append({
                    "name": capability_name,
                    "expertise": expertise,
                    "level": profile.get_expertise_level(capability_name).value
                })

        # Calculate overall metrics
        total_capabilities = len(profile.capabilities)
        expert_capabilities = len([
            cap for cap, exp in profile.capabilities.items()
            if exp >= 0.8
        ])

        recent_success_rate = (
            profile.performance_metrics.get("successful_tasks", 0) /
            max(1, profile.performance_metrics.get("recent_tasks", 0))
        )

        return {
            "agent_name": agent_name,
            "total_capabilities": total_capabilities,
            "expert_capabilities": expert_capabilities,
            "expertise_percentage": (expert_capabilities / max(1, total_capabilities)) * 100,
            "recent_success_rate": recent_success_rate,
            "capabilities_by_category": dict(capabilities_by_category),
            "top_capabilities": [
                {"name": cap, "expertise": exp}
                for cap, exp in profile.get_top_capabilities(5)
            ],
            "learning_rate": profile.learning_rate,
            "last_updated": profile.last_updated.isoformat()
        }