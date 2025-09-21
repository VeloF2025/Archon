"""
Cross-Project Knowledge Synthesis System for Phase 9 Autonomous Development Teams

This module implements the global knowledge network that learns from all projects,
identifies successful patterns, detects anti-patterns, and shares optimized solutions
across autonomous development teams.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from uuid import uuid4, UUID
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class PatternType(str, Enum):
    """Types of patterns identified across projects."""
    SUCCESS_PATTERN = "success_pattern"
    ANTI_PATTERN = "anti_pattern"
    OPTIMIZATION_PATTERN = "optimization_pattern"
    ARCHITECTURAL_PATTERN = "architectural_pattern"
    TESTING_PATTERN = "testing_pattern"
    DEPLOYMENT_PATTERN = "deployment_pattern"
    PERFORMANCE_PATTERN = "performance_pattern"
    SECURITY_PATTERN = "security_pattern"


class PatternConfidence(str, Enum):
    """Confidence levels for identified patterns."""
    LOW = "low"          # 1-2 occurrences
    MEDIUM = "medium"    # 3-5 occurrences
    HIGH = "high"        # 6-10 occurrences
    VERY_HIGH = "very_high"  # 11+ occurrences


class ProjectOutcome(str, Enum):
    """Project completion outcomes."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


@dataclass
class ProjectMetrics:
    """Metrics collected from completed projects."""
    project_id: str
    name: str = ""
    outcome: ProjectOutcome = ProjectOutcome.SUCCESS
    completion_time_hours: float = 0.0
    estimated_time_hours: float = 0.0
    team_size: int = 0
    technologies_used: List[str] = field(default_factory=list)
    architectural_patterns: List[str] = field(default_factory=list)
    test_coverage: float = 0.0
    performance_score: float = 0.0
    security_score: float = 0.0
    code_quality_score: float = 0.0
    bug_count: int = 0
    critical_bugs: int = 0
    deployment_success: bool = True
    user_satisfaction: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IdentifiedPattern:
    """A pattern identified from analyzing multiple projects."""
    id: str = field(default_factory=lambda: str(uuid4()))
    type: PatternType = PatternType.SUCCESS_PATTERN
    name: str = ""
    description: str = ""
    context: str = ""
    evidence_projects: List[str] = field(default_factory=list)
    confidence: PatternConfidence = PatternConfidence.LOW
    success_correlation: float = 0.0  # -1.0 to 1.0
    frequency: int = 0
    impact_score: float = 0.0  # Measured impact on project success
    conditions: List[str] = field(default_factory=list)  # When this pattern applies
    implementation_guide: str = ""
    related_patterns: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class KnowledgeSynthesis:
    """Synthesized knowledge from cross-project analysis."""
    id: str = field(default_factory=lambda: str(uuid4()))
    title: str = ""
    summary: str = ""
    patterns_analyzed: int = 0
    projects_analyzed: int = 0
    key_insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    success_predictors: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class CrossProjectKnowledgeEngine:
    """
    Main engine for cross-project knowledge synthesis and pattern recognition.
    
    Analyzes completed projects to identify patterns, anti-patterns, and best practices
    that can be shared across autonomous development teams.
    """
    
    def __init__(self, storage_path: str = "data/knowledge_synthesis"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.project_metrics: Dict[str, ProjectMetrics] = {}
        self.identified_patterns: Dict[str, IdentifiedPattern] = {}
        self.knowledge_syntheses: Dict[str, KnowledgeSynthesis] = {}
        
        # ML models for pattern analysis
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.clusterer = DBSCAN(eps=0.3, min_samples=2)
        
        # Pattern detection thresholds
        self.min_pattern_frequency = 2
        self.min_confidence_threshold = 0.6
        self.pattern_similarity_threshold = 0.8
        
        # Load existing data
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """Load existing knowledge base from storage."""
        try:
            # Load project metrics
            metrics_file = self.storage_path / "project_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    self.project_metrics = {
                        k: ProjectMetrics(**v) for k, v in data.items()
                    }
                logger.info(f"Loaded {len(self.project_metrics)} project metrics")
            
            # Load identified patterns
            patterns_file = self.storage_path / "patterns.json"
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    data = json.load(f)
                    self.identified_patterns = {
                        k: IdentifiedPattern(**v) for k, v in data.items()
                    }
                logger.info(f"Loaded {len(self.identified_patterns)} patterns")
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
    
    def _save_knowledge_base(self):
        """Save knowledge base to storage."""
        try:
            # Save project metrics
            metrics_file = self.storage_path / "project_metrics.json"
            with open(metrics_file, 'w') as f:
                data = {k: self._serialize_dataclass(v) for k, v in self.project_metrics.items()}
                json.dump(data, f, indent=2, default=str)
            
            # Save patterns
            patterns_file = self.storage_path / "patterns.json"
            with open(patterns_file, 'w') as f:
                data = {k: self._serialize_dataclass(v) for k, v in self.identified_patterns.items()}
                json.dump(data, f, indent=2, default=str)
            
            logger.info("Knowledge base saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
    
    def _serialize_dataclass(self, obj) -> Dict[str, Any]:
        """Serialize dataclass to dictionary."""
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if isinstance(value, datetime):
                    result[key] = value.isoformat()
                elif isinstance(value, Enum):
                    result[key] = value.value
                else:
                    result[key] = value
            return result
        return obj
    
    async def add_project_metrics(self, metrics: ProjectMetrics) -> bool:
        """Add metrics from a completed project."""
        try:
            self.project_metrics[metrics.project_id] = metrics
            logger.info(f"Added metrics for project {metrics.project_id}")
            
            # Trigger pattern analysis
            await self._analyze_patterns()
            
            # Save updated knowledge base
            self._save_knowledge_base()
            
            return True
        
        except Exception as e:
            logger.error(f"Error adding project metrics: {e}")
            return False
    
    async def _analyze_patterns(self):
        """Analyze all project data to identify patterns and anti-patterns."""
        if len(self.project_metrics) < 2:
            logger.info("Not enough projects for pattern analysis")
            return
        
        logger.info(f"Analyzing patterns from {len(self.project_metrics)} projects")
        
        # Analyze different types of patterns
        await self._identify_success_patterns()
        await self._identify_anti_patterns()
        await self._identify_optimization_patterns()
        await self._identify_architectural_patterns()
        await self._identify_testing_patterns()
        await self._identify_performance_patterns()
        
        logger.info(f"Pattern analysis complete. {len(self.identified_patterns)} patterns identified")
    
    async def _identify_success_patterns(self):
        """Identify patterns that correlate with project success."""
        successful_projects = [
            m for m in self.project_metrics.values()
            if m.outcome == ProjectOutcome.SUCCESS
        ]
        
        if len(successful_projects) < self.min_pattern_frequency:
            return
        
        # Analyze common technologies in successful projects
        tech_patterns = self._analyze_technology_patterns(successful_projects)
        for pattern in tech_patterns:
            pattern.type = PatternType.SUCCESS_PATTERN
            self.identified_patterns[pattern.id] = pattern
        
        # Analyze architectural patterns
        arch_patterns = self._analyze_architectural_patterns(successful_projects)
        for pattern in arch_patterns:
            pattern.type = PatternType.SUCCESS_PATTERN
            self.identified_patterns[pattern.id] = pattern
        
        # Analyze team size patterns
        team_patterns = self._analyze_team_size_patterns(successful_projects)
        for pattern in team_patterns:
            pattern.type = PatternType.SUCCESS_PATTERN
            self.identified_patterns[pattern.id] = pattern
    
    async def _identify_anti_patterns(self):
        """Identify patterns that correlate with project failure."""
        failed_projects = [
            m for m in self.project_metrics.values()
            if m.outcome in [ProjectOutcome.FAILURE, ProjectOutcome.CANCELLED]
        ]
        
        if len(failed_projects) < self.min_pattern_frequency:
            return
        
        # Analyze common failure modes
        failure_patterns = self._analyze_failure_patterns(failed_projects)
        for pattern in failure_patterns:
            pattern.type = PatternType.ANTI_PATTERN
            self.identified_patterns[pattern.id] = pattern
        
        # Analyze problematic technology combinations
        problematic_tech = self._analyze_problematic_technologies(failed_projects)
        for pattern in problematic_tech:
            pattern.type = PatternType.ANTI_PATTERN
            self.identified_patterns[pattern.id] = pattern
    
    def _analyze_technology_patterns(self, projects: List[ProjectMetrics]) -> List[IdentifiedPattern]:
        """Analyze technology usage patterns in given projects."""
        patterns = []
        
        # Count technology frequency
        tech_counter = Counter()
        for project in projects:
            tech_counter.update(project.technologies_used)
        
        # Find frequent technology combinations
        tech_combinations = defaultdict(int)
        for project in projects:
            techs = sorted(project.technologies_used)
            for i in range(len(techs)):
                for j in range(i + 1, len(techs)):
                    combo = f"{techs[i]} + {techs[j]}"
                    tech_combinations[combo] += 1
        
        # Create patterns for frequent combinations
        for combo, frequency in tech_combinations.items():
            if frequency >= self.min_pattern_frequency:
                pattern = IdentifiedPattern(
                    name=f"Technology Stack: {combo}",
                    description=f"Projects using {combo} show {frequency} successful outcomes",
                    context="Technology stack selection",
                    evidence_projects=[p.project_id for p in projects if 
                                     all(tech in p.technologies_used for tech in combo.split(" + "))],
                    confidence=self._calculate_confidence(frequency),
                    frequency=frequency,
                    success_correlation=self._calculate_success_correlation(projects, combo),
                    implementation_guide=f"Consider using {combo} for similar project types",
                    tags=combo.split(" + ")
                )
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_architectural_patterns(self, projects: List[ProjectMetrics]) -> List[IdentifiedPattern]:
        """Analyze architectural patterns in successful projects."""
        patterns = []
        
        # Count architectural pattern frequency
        arch_counter = Counter()
        for project in projects:
            arch_counter.update(project.architectural_patterns)
        
        # Create patterns for frequent architectural choices
        for arch_pattern, frequency in arch_counter.items():
            if frequency >= self.min_pattern_frequency:
                pattern = IdentifiedPattern(
                    name=f"Architectural Pattern: {arch_pattern}",
                    description=f"The {arch_pattern} pattern appears in {frequency} successful projects",
                    context="System architecture",
                    evidence_projects=[p.project_id for p in projects if arch_pattern in p.architectural_patterns],
                    confidence=self._calculate_confidence(frequency),
                    frequency=frequency,
                    success_correlation=self._calculate_success_correlation(projects, arch_pattern),
                    implementation_guide=f"Consider implementing {arch_pattern} for similar requirements",
                    tags=[arch_pattern, "architecture"]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_team_size_patterns(self, projects: List[ProjectMetrics]) -> List[IdentifiedPattern]:
        """Analyze team size patterns in successful projects."""
        patterns = []
        
        # Group projects by team size ranges
        team_size_groups = defaultdict(list)
        for project in projects:
            if project.team_size <= 3:
                team_size_groups["small"].append(project)
            elif project.team_size <= 7:
                team_size_groups["medium"].append(project)
            else:
                team_size_groups["large"].append(project)
        
        # Analyze success rates by team size
        for size_category, group_projects in team_size_groups.items():
            if len(group_projects) >= self.min_pattern_frequency:
                avg_completion_ratio = np.mean([
                    p.completion_time_hours / p.estimated_time_hours
                    if p.estimated_time_hours > 0 else 1.0
                    for p in group_projects
                ])
                
                pattern = IdentifiedPattern(
                    name=f"Team Size: {size_category.title()} Teams",
                    description=f"{size_category.title()} teams (size {min(p.team_size for p in group_projects)}-{max(p.team_size for p in group_projects)}) show avg completion ratio of {avg_completion_ratio:.2f}",
                    context="Team composition",
                    evidence_projects=[p.project_id for p in group_projects],
                    confidence=self._calculate_confidence(len(group_projects)),
                    frequency=len(group_projects),
                    impact_score=1.0 / avg_completion_ratio,  # Lower ratio = higher impact
                    implementation_guide=f"Consider {size_category} team size for optimal efficiency",
                    tags=["team_size", size_category]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_failure_patterns(self, failed_projects: List[ProjectMetrics]) -> List[IdentifiedPattern]:
        """Analyze common patterns in failed projects."""
        patterns = []
        
        # Analyze common failure indicators
        if len(failed_projects) >= self.min_pattern_frequency:
            avg_overrun = np.mean([
                p.completion_time_hours / p.estimated_time_hours
                if p.estimated_time_hours > 0 else 2.0
                for p in failed_projects
            ])
            
            if avg_overrun > 1.5:  # 50% overrun threshold
                pattern = IdentifiedPattern(
                    name="Time Estimation Anti-Pattern",
                    description=f"Failed projects show {avg_overrun:.1f}x time overrun on average",
                    context="Project planning",
                    evidence_projects=[p.project_id for p in failed_projects],
                    confidence=self._calculate_confidence(len(failed_projects)),
                    frequency=len(failed_projects),
                    success_correlation=-0.8,  # Strong negative correlation
                    implementation_guide="Implement more conservative time estimation with 50% buffer for complex projects",
                    tags=["estimation", "planning", "risk"]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_problematic_technologies(self, failed_projects: List[ProjectMetrics]) -> List[IdentifiedPattern]:
        """Analyze technology patterns in failed projects."""
        patterns = []
        
        # Find technologies that appear frequently in failures
        tech_counter = Counter()
        for project in failed_projects:
            tech_counter.update(project.technologies_used)
        
        # Compare with overall technology usage
        all_tech_counter = Counter()
        for project in self.project_metrics.values():
            all_tech_counter.update(project.technologies_used)
        
        # Identify problematic technologies
        for tech, failure_count in tech_counter.items():
            if failure_count >= self.min_pattern_frequency:
                total_usage = all_tech_counter[tech]
                failure_rate = failure_count / total_usage if total_usage > 0 else 0
                
                if failure_rate > 0.4:  # 40% failure rate threshold
                    pattern = IdentifiedPattern(
                        name=f"High-Risk Technology: {tech}",
                        description=f"{tech} appears in {failure_count} failed projects ({failure_rate:.1%} failure rate)",
                        context="Technology selection",
                        evidence_projects=[p.project_id for p in failed_projects if tech in p.technologies_used],
                        confidence=self._calculate_confidence(failure_count),
                        frequency=failure_count,
                        success_correlation=-failure_rate,
                        implementation_guide=f"Exercise caution when using {tech}. Ensure adequate expertise and support.",
                        tags=[tech, "high_risk", "technology"]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _identify_optimization_patterns(self):
        """Identify patterns that lead to optimization opportunities."""
        # Analyze projects with exceptional performance
        high_performance_projects = [
            m for m in self.project_metrics.values()
            if m.performance_score > 8.0 and m.outcome == ProjectOutcome.SUCCESS
        ]
        
        if len(high_performance_projects) >= self.min_pattern_frequency:
            # Analyze common optimization techniques
            optimization_patterns = self._extract_optimization_patterns(high_performance_projects)
            for pattern in optimization_patterns:
                pattern.type = PatternType.OPTIMIZATION_PATTERN
                self.identified_patterns[pattern.id] = pattern
    
    def _extract_optimization_patterns(self, projects: List[ProjectMetrics]) -> List[IdentifiedPattern]:
        """Extract optimization patterns from high-performance projects."""
        patterns = []
        
        # Analyze test coverage correlation with performance
        avg_coverage = np.mean([p.test_coverage for p in projects])
        if avg_coverage > 0.9:  # 90% coverage
            pattern = IdentifiedPattern(
                name="High Test Coverage Optimization",
                description=f"High-performance projects maintain {avg_coverage:.1%} test coverage on average",
                context="Testing strategy",
                evidence_projects=[p.project_id for p in projects],
                confidence=self._calculate_confidence(len(projects)),
                frequency=len(projects),
                success_correlation=0.7,
                implementation_guide="Maintain >90% test coverage for optimal performance and reliability",
                tags=["testing", "coverage", "performance"]
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _identify_testing_patterns(self):
        """Identify effective testing patterns."""
        projects_with_testing = [
            m for m in self.project_metrics.values()
            if m.test_coverage > 0.7 and m.bug_count < 5
        ]
        
        if len(projects_with_testing) >= self.min_pattern_frequency:
            pattern = IdentifiedPattern(
                name="Comprehensive Testing Pattern",
                description=f"Projects with >70% test coverage show {np.mean([p.bug_count for p in projects_with_testing]):.1f} bugs on average",
                context="Quality assurance",
                evidence_projects=[p.project_id for p in projects_with_testing],
                confidence=self._calculate_confidence(len(projects_with_testing)),
                frequency=len(projects_with_testing),
                success_correlation=0.8,
                implementation_guide="Implement comprehensive testing strategy with >70% coverage target",
                tags=["testing", "quality", "bug_prevention"],
                type=PatternType.TESTING_PATTERN
            )
            self.identified_patterns[pattern.id] = pattern
    
    async def _identify_performance_patterns(self):
        """Identify performance optimization patterns."""
        high_perf_projects = [
            m for m in self.project_metrics.values()
            if m.performance_score > 8.0 and m.user_satisfaction > 8.0
        ]
        
        if len(high_perf_projects) >= self.min_pattern_frequency:
            # Analyze common characteristics
            common_techs = self._find_common_technologies(high_perf_projects)
            
            for tech in common_techs:
                pattern = IdentifiedPattern(
                    name=f"Performance Technology: {tech}",
                    description=f"{tech} consistently appears in high-performance projects",
                    context="Performance optimization",
                    evidence_projects=[p.project_id for p in high_perf_projects if tech in p.technologies_used],
                    confidence=self._calculate_confidence(len([p for p in high_perf_projects if tech in p.technologies_used])),
                    success_correlation=0.7,
                    implementation_guide=f"Consider {tech} for performance-critical applications",
                    tags=[tech, "performance", "optimization"],
                    type=PatternType.PERFORMANCE_PATTERN
                )
                self.identified_patterns[pattern.id] = pattern
    
    def _find_common_technologies(self, projects: List[ProjectMetrics]) -> List[str]:
        """Find technologies common to most projects in the list."""
        tech_counter = Counter()
        for project in projects:
            tech_counter.update(project.technologies_used)
        
        threshold = len(projects) * 0.6  # Appears in 60% of projects
        return [tech for tech, count in tech_counter.items() if count >= threshold]
    
    def _calculate_confidence(self, frequency: int) -> PatternConfidence:
        """Calculate confidence level based on frequency."""
        if frequency >= 11:
            return PatternConfidence.VERY_HIGH
        elif frequency >= 6:
            return PatternConfidence.HIGH
        elif frequency >= 3:
            return PatternConfidence.MEDIUM
        else:
            return PatternConfidence.LOW
    
    def _calculate_success_correlation(self, projects: List[ProjectMetrics], pattern: str) -> float:
        """Calculate correlation between pattern and success rate."""
        # Simplified correlation calculation
        pattern_projects = [p for p in projects if pattern in str(p.__dict__)]
        if not pattern_projects:
            return 0.0
        
        success_rate = len([p for p in pattern_projects if p.outcome == ProjectOutcome.SUCCESS]) / len(pattern_projects)
        return success_rate * 2 - 1  # Scale to -1 to 1
    
    async def get_recommendations_for_project(
        self,
        technologies: List[str],
        team_size: int,
        estimated_duration: float,
        project_type: str = "web_application"
    ) -> List[Dict[str, Any]]:
        """Get pattern-based recommendations for a new project."""
        
        recommendations = []
        
        # Find relevant patterns
        relevant_patterns = []
        for pattern in self.identified_patterns.values():
            # Check technology overlap
            tech_overlap = any(tech in pattern.tags for tech in technologies)
            
            # Check team size relevance
            team_size_relevant = (
                "small" in pattern.tags and team_size <= 3 or
                "medium" in pattern.tags and 4 <= team_size <= 7 or
                "large" in pattern.tags and team_size > 7
            )
            
            if tech_overlap or team_size_relevant:
                relevant_patterns.append(pattern)
        
        # Sort by confidence and success correlation
        relevant_patterns.sort(
            key=lambda p: (p.confidence.value, p.success_correlation),
            reverse=True
        )
        
        # Generate recommendations
        for pattern in relevant_patterns[:10]:  # Top 10 recommendations
            if pattern.type == PatternType.SUCCESS_PATTERN:
                recommendation = {
                    "type": "adopt_pattern",
                    "priority": "high" if pattern.confidence == PatternConfidence.VERY_HIGH else "medium",
                    "title": f"Adopt {pattern.name}",
                    "description": pattern.description,
                    "implementation_guide": pattern.implementation_guide,
                    "confidence": pattern.confidence.value,
                    "evidence_count": len(pattern.evidence_projects)
                }
                recommendations.append(recommendation)
            
            elif pattern.type == PatternType.ANTI_PATTERN:
                recommendation = {
                    "type": "avoid_pattern",
                    "priority": "critical",
                    "title": f"Avoid {pattern.name}",
                    "description": pattern.description,
                    "risk_mitigation": pattern.implementation_guide,
                    "confidence": pattern.confidence.value,
                    "failure_rate": abs(pattern.success_correlation)
                }
                recommendations.append(recommendation)
        
        return recommendations
    
    async def generate_knowledge_synthesis(self) -> KnowledgeSynthesis:
        """Generate a comprehensive knowledge synthesis from all patterns."""
        
        synthesis = KnowledgeSynthesis(
            title="Cross-Project Knowledge Synthesis",
            summary=f"Analysis of {len(self.project_metrics)} projects identifying {len(self.identified_patterns)} patterns",
            patterns_analyzed=len(self.identified_patterns),
            projects_analyzed=len(self.project_metrics)
        )
        
        # Generate key insights
        success_patterns = [p for p in self.identified_patterns.values() if p.type == PatternType.SUCCESS_PATTERN]
        anti_patterns = [p for p in self.identified_patterns.values() if p.type == PatternType.ANTI_PATTERN]
        
        synthesis.key_insights = [
            f"Identified {len(success_patterns)} success patterns with {PatternConfidence.HIGH.value} or higher confidence",
            f"Detected {len(anti_patterns)} anti-patterns that correlate with project failure",
            f"Most successful technology combinations: {self._get_top_technology_patterns()}",
            f"Optimal team size appears to be {self._get_optimal_team_size()} members"
        ]
        
        # Generate recommendations
        synthesis.recommendations = [
            "Adopt proven technology stacks from successful projects",
            "Implement comprehensive testing strategies (>70% coverage)",
            "Use conservative time estimation with appropriate buffers",
            "Consider team size optimization based on project complexity"
        ]
        
        # Identify risk factors
        synthesis.risk_factors = [
            f"High-risk technologies: {self._get_high_risk_technologies()}",
            "Insufficient test coverage (<50%)",
            "Overly optimistic time estimates",
            "Team size mismatches for project complexity"
        ]
        
        # Success predictors
        synthesis.success_predictors = [
            "High test coverage (>80%)",
            "Proven technology stack combinations",
            "Appropriate team size for project scope",
            "Conservative time estimation"
        ]
        
        # Store synthesis
        self.knowledge_syntheses[synthesis.id] = synthesis
        
        return synthesis
    
    def _get_top_technology_patterns(self) -> str:
        """Get the most successful technology combinations."""
        success_patterns = [p for p in self.identified_patterns.values() 
                          if p.type == PatternType.SUCCESS_PATTERN and "+" in p.name]
        if not success_patterns:
            return "None identified"
        
        top_pattern = max(success_patterns, key=lambda p: p.success_correlation)
        return top_pattern.name.replace("Technology Stack: ", "")
    
    def _get_optimal_team_size(self) -> str:
        """Get the optimal team size based on patterns."""
        team_patterns = [p for p in self.identified_patterns.values() 
                        if "team_size" in p.tags]
        if not team_patterns:
            return "3-5"
        
        optimal_pattern = max(team_patterns, key=lambda p: p.impact_score)
        for tag in optimal_pattern.tags:
            if tag in ["small", "medium", "large"]:
                size_map = {"small": "2-3", "medium": "4-7", "large": "8+"}
                return size_map.get(tag, "3-5")
        
        return "3-5"
    
    def _get_high_risk_technologies(self) -> str:
        """Get technologies identified as high-risk."""
        risk_patterns = [p for p in self.identified_patterns.values() 
                        if p.type == PatternType.ANTI_PATTERN and "high_risk" in p.tags]
        if not risk_patterns:
            return "None identified"
        
        risk_techs = []
        for pattern in risk_patterns:
            tech_tags = [tag for tag in pattern.tags if tag not in ["high_risk", "technology"]]
            risk_techs.extend(tech_tags)
        
        return ", ".join(list(set(risk_techs))[:3])  # Top 3
    
    async def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about identified patterns."""
        
        pattern_stats = {
            "total_patterns": len(self.identified_patterns),
            "by_type": {},
            "by_confidence": {},
            "projects_analyzed": len(self.project_metrics),
            "success_rate": len([p for p in self.project_metrics.values() 
                               if p.outcome == ProjectOutcome.SUCCESS]) / len(self.project_metrics) if self.project_metrics else 0
        }
        
        # Count by type
        for pattern_type in PatternType:
            count = len([p for p in self.identified_patterns.values() if p.type == pattern_type])
            pattern_stats["by_type"][pattern_type.value] = count
        
        # Count by confidence
        for confidence in PatternConfidence:
            count = len([p for p in self.identified_patterns.values() if p.confidence == confidence])
            pattern_stats["by_confidence"][confidence.value] = count
        
        return pattern_stats


async def main():
    """Test the cross-project knowledge engine."""
    
    logging.basicConfig(level=logging.INFO)
    
    engine = CrossProjectKnowledgeEngine()
    
    # Add some test project metrics
    test_projects = [
        ProjectMetrics(
            project_id="proj-1",
            name="E-commerce Platform",
            outcome=ProjectOutcome.SUCCESS,
            completion_time_hours=320,
            estimated_time_hours=300,
            team_size=5,
            technologies_used=["React", "Node.js", "PostgreSQL", "Redis"],
            architectural_patterns=["Microservices", "Event-Driven"],
            test_coverage=0.85,
            performance_score=8.5,
            security_score=9.0,
            code_quality_score=8.8
        ),
        ProjectMetrics(
            project_id="proj-2",
            name="Mobile Banking App",
            outcome=ProjectOutcome.SUCCESS,
            completion_time_hours=450,
            estimated_time_hours=400,
            team_size=6,
            technologies_used=["React", "Node.js", "MongoDB", "AWS"],
            architectural_patterns=["Microservices", "CQRS"],
            test_coverage=0.92,
            performance_score=9.2,
            security_score=9.5,
            code_quality_score=9.1
        ),
        ProjectMetrics(
            project_id="proj-3",
            name="Analytics Dashboard",
            outcome=ProjectOutcome.FAILURE,
            completion_time_hours=800,
            estimated_time_hours=400,
            team_size=3,
            technologies_used=["Angular", "Python", "MySQL"],
            architectural_patterns=["Monolithic"],
            test_coverage=0.45,
            performance_score=6.0,
            security_score=7.0,
            code_quality_score=6.5,
            bug_count=25,
            critical_bugs=5
        )
    ]
    
    # Add projects to engine
    for project in test_projects:
        await engine.add_project_metrics(project)
    
    # Get recommendations for a new project
    recommendations = await engine.get_recommendations_for_project(
        technologies=["React", "Node.js"],
        team_size=5,
        estimated_duration=350
    )
    
    print("\nRecommendations for new project:")
    for rec in recommendations:
        print(f"- {rec['title']}: {rec['description']}")
    
    # Generate knowledge synthesis
    synthesis = await engine.generate_knowledge_synthesis()
    print(f"\nKnowledge Synthesis: {synthesis.title}")
    print(f"Key Insights: {synthesis.key_insights}")
    print(f"Recommendations: {synthesis.recommendations}")
    
    # Get statistics
    stats = await engine.get_pattern_statistics()
    print(f"\nPattern Statistics: {stats}")


if __name__ == "__main__":
    asyncio.run(main())