"""
Self-Assembling Agent Teams System
Automatically creates optimal development teams based on project requirements.
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Different types of development agents."""
    ARCHITECT = "architect"
    DEVELOPER = "developer"
    TESTER = "tester"
    REVIEWER = "reviewer"
    SECURITY = "security"
    DEVOPS = "devops"
    UI_UX = "ui_ux"
    TECH_WRITER = "tech_writer"
    PROJECT_MANAGER = "project_manager"
    DATA_ANALYST = "data_analyst"


class ProjectComplexity(Enum):
    """Project complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    ENTERPRISE = "enterprise"


class TechnologyStack(Enum):
    """Supported technology stacks."""
    WEB_FRONTEND = "web_frontend"
    WEB_BACKEND = "web_backend"
    MOBILE = "mobile"
    DESKTOP = "desktop"
    DATA_SCIENCE = "data_science"
    MACHINE_LEARNING = "machine_learning"
    DEVOPS_INFRA = "devops_infra"
    BLOCKCHAIN = "blockchain"
    IOT = "iot"
    GAME_DEVELOPMENT = "game_development"


@dataclass
class AgentCapability:
    """Represents an agent's capabilities and skills."""
    role: AgentRole
    technologies: Set[TechnologyStack]
    skill_level: float  # 0.0 to 1.0
    availability: bool
    current_load: float  # 0.0 to 1.0 (workload)
    specializations: List[str]
    performance_score: float
    collaboration_score: float
    last_active: datetime
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ProjectRequirements:
    """Project requirements for team assembly."""
    project_id: str
    title: str
    description: str
    technologies: Set[TechnologyStack]
    complexity: ProjectComplexity
    timeline_weeks: int
    budget_range: Tuple[float, float]  # (min, max)
    quality_requirements: Dict[str, float]
    special_requirements: List[str]
    deadline: Optional[datetime] = None
    priority: str = "medium"  # low, medium, high, critical
    client_type: str = "internal"  # internal, external, enterprise
    
    # Derived properties
    estimated_effort_hours: int = 0
    risk_factors: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)


@dataclass
class TeamComposition:
    """Optimal team composition for a project."""
    project_id: str
    agents: List[AgentCapability]
    team_lead: str  # Agent ID
    estimated_completion: datetime
    confidence_score: float
    cost_estimate: float
    risk_assessment: Dict[str, float]
    team_synergy_score: float
    alternative_compositions: List['TeamComposition'] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TeamPerformanceMetrics:
    """Track team performance over time."""
    team_id: str
    project_success_rate: float
    average_delivery_time: float
    quality_score: float
    budget_accuracy: float
    client_satisfaction: float
    collaboration_effectiveness: float
    learning_rate: float
    adaptability_score: float
    last_updated: datetime = field(default_factory=datetime.now)


class ProjectAnalyzer:
    """Analyzes project requirements to determine optimal team composition."""
    
    def __init__(self):
        # Technology complexity weights
        self.tech_complexity = {
            TechnologyStack.WEB_FRONTEND: 0.6,
            TechnologyStack.WEB_BACKEND: 0.7,
            TechnologyStack.MOBILE: 0.8,
            TechnologyStack.DESKTOP: 0.7,
            TechnologyStack.DATA_SCIENCE: 0.9,
            TechnologyStack.MACHINE_LEARNING: 1.0,
            TechnologyStack.DEVOPS_INFRA: 0.8,
            TechnologyStack.BLOCKCHAIN: 1.0,
            TechnologyStack.IOT: 0.9,
            TechnologyStack.GAME_DEVELOPMENT: 0.8,
        }
        
        # Role importance by technology stack
        self.role_importance = {
            TechnologyStack.WEB_FRONTEND: {
                AgentRole.DEVELOPER: 1.0,
                AgentRole.UI_UX: 0.9,
                AgentRole.TESTER: 0.8,
                AgentRole.REVIEWER: 0.7,
            },
            TechnologyStack.WEB_BACKEND: {
                AgentRole.ARCHITECT: 0.9,
                AgentRole.DEVELOPER: 1.0,
                AgentRole.SECURITY: 0.8,
                AgentRole.DEVOPS: 0.7,
                AgentRole.TESTER: 0.8,
            },
            TechnologyStack.MACHINE_LEARNING: {
                AgentRole.DATA_ANALYST: 1.0,
                AgentRole.DEVELOPER: 0.9,
                AgentRole.ARCHITECT: 0.8,
                AgentRole.TESTER: 0.6,
            },
            # Add more mappings as needed
        }
        
        # Base team sizes by complexity
        self.base_team_sizes = {
            ProjectComplexity.SIMPLE: {"min": 2, "max": 4},
            ProjectComplexity.MEDIUM: {"min": 3, "max": 7},
            ProjectComplexity.COMPLEX: {"min": 5, "max": 12},
            ProjectComplexity.ENTERPRISE: {"min": 8, "max": 20},
        }
    
    async def analyze_project(self, requirements: ProjectRequirements) -> Dict[str, Any]:
        """Perform comprehensive project analysis."""
        
        analysis = {
            "complexity_score": self._calculate_complexity_score(requirements),
            "technology_analysis": self._analyze_technologies(requirements),
            "effort_estimation": self._estimate_effort(requirements),
            "risk_assessment": self._assess_risks(requirements),
            "team_requirements": self._determine_team_requirements(requirements),
            "timeline_analysis": self._analyze_timeline(requirements),
            "quality_requirements": self._analyze_quality_requirements(requirements),
        }
        
        # Update requirements with derived properties
        requirements.estimated_effort_hours = analysis["effort_estimation"]["total_hours"]
        requirements.risk_factors = analysis["risk_assessment"]["high_risk_factors"]
        
        return analysis
    
    def _calculate_complexity_score(self, requirements: ProjectRequirements) -> float:
        """Calculate overall project complexity score."""
        
        # Base complexity from enum
        complexity_base = {
            ProjectComplexity.SIMPLE: 0.2,
            ProjectComplexity.MEDIUM: 0.5,
            ProjectComplexity.COMPLEX: 0.8,
            ProjectComplexity.ENTERPRISE: 1.0,
        }
        
        score = complexity_base[requirements.complexity]
        
        # Technology stack complexity
        tech_complexity = sum(
            self.tech_complexity.get(tech, 0.5) 
            for tech in requirements.technologies
        ) / max(len(requirements.technologies), 1)
        
        # Timeline pressure
        if requirements.timeline_weeks < 4:
            timeline_factor = 0.3  # Tight timeline increases complexity
        elif requirements.timeline_weeks < 12:
            timeline_factor = 0.1
        else:
            timeline_factor = -0.1  # Longer timeline reduces complexity
        
        # Special requirements
        special_factor = len(requirements.special_requirements) * 0.1
        
        final_score = min(score + tech_complexity * 0.3 + timeline_factor + special_factor, 1.0)
        return max(final_score, 0.0)
    
    def _analyze_technologies(self, requirements: ProjectRequirements) -> Dict[str, Any]:
        """Analyze technology stack requirements."""
        
        tech_analysis = {
            "primary_technologies": list(requirements.technologies),
            "complexity_by_tech": {
                tech.value: self.tech_complexity.get(tech, 0.5)
                for tech in requirements.technologies
            },
            "integration_complexity": self._calculate_integration_complexity(requirements.technologies),
            "required_expertise_areas": self._identify_expertise_areas(requirements.technologies),
            "technology_risks": self._identify_technology_risks(requirements.technologies),
        }
        
        return tech_analysis
    
    def _calculate_integration_complexity(self, technologies: Set[TechnologyStack]) -> float:
        """Calculate complexity of integrating multiple technologies."""
        
        if len(technologies) <= 1:
            return 0.1
        elif len(technologies) <= 3:
            return 0.3
        elif len(technologies) <= 5:
            return 0.6
        else:
            return 1.0  # High integration complexity
    
    def _identify_expertise_areas(self, technologies: Set[TechnologyStack]) -> List[str]:
        """Identify required areas of expertise."""
        
        expertise_map = {
            TechnologyStack.WEB_FRONTEND: ["React", "Vue", "Angular", "CSS", "JavaScript", "TypeScript"],
            TechnologyStack.WEB_BACKEND: ["Node.js", "Python", "Java", "Go", "API Design", "Databases"],
            TechnologyStack.MOBILE: ["React Native", "Flutter", "iOS", "Android", "Mobile UI/UX"],
            TechnologyStack.DATA_SCIENCE: ["Python", "R", "SQL", "Statistics", "Data Visualization"],
            TechnologyStack.MACHINE_LEARNING: ["TensorFlow", "PyTorch", "MLOps", "Model Training", "Data Engineering"],
            TechnologyStack.DEVOPS_INFRA: ["Kubernetes", "Docker", "AWS", "CI/CD", "Monitoring"],
        }
        
        all_expertise = []
        for tech in technologies:
            all_expertise.extend(expertise_map.get(tech, [tech.value]))
        
        return list(set(all_expertise))
    
    def _identify_technology_risks(self, technologies: Set[TechnologyStack]) -> List[str]:
        """Identify risks associated with technology choices."""
        
        risk_map = {
            TechnologyStack.BLOCKCHAIN: ["Market volatility", "Regulatory uncertainty", "Scalability issues"],
            TechnologyStack.MACHINE_LEARNING: ["Data quality", "Model bias", "Interpretability"],
            TechnologyStack.IOT: ["Security vulnerabilities", "Hardware dependencies", "Connectivity issues"],
            TechnologyStack.GAME_DEVELOPMENT: ["Performance optimization", "Platform compatibility", "User retention"],
        }
        
        all_risks = []
        for tech in technologies:
            all_risks.extend(risk_map.get(tech, []))
        
        return list(set(all_risks))
    
    def _estimate_effort(self, requirements: ProjectRequirements) -> Dict[str, Any]:
        """Estimate development effort in hours."""
        
        # Base effort by complexity
        base_hours = {
            ProjectComplexity.SIMPLE: 80,
            ProjectComplexity.MEDIUM: 320,
            ProjectComplexity.COMPLEX: 800,
            ProjectComplexity.ENTERPRISE: 2000,
        }
        
        base_effort = base_hours[requirements.complexity]
        
        # Technology multipliers
        tech_multiplier = sum(
            self.tech_complexity.get(tech, 0.5) 
            for tech in requirements.technologies
        ) / max(len(requirements.technologies), 1)
        
        # Special requirements multiplier
        special_multiplier = 1.0 + (len(requirements.special_requirements) * 0.2)
        
        # Quality requirements multiplier
        quality_multiplier = 1.0 + (sum(requirements.quality_requirements.values()) / len(requirements.quality_requirements or [1]) - 0.8)
        
        total_hours = int(base_effort * tech_multiplier * special_multiplier * quality_multiplier)
        
        return {
            "base_hours": base_effort,
            "technology_factor": tech_multiplier,
            "special_requirements_factor": special_multiplier,
            "quality_factor": quality_multiplier,
            "total_hours": total_hours,
            "estimated_weeks": max(1, total_hours // 40),  # 40 hours per week
        }
    
    def _assess_risks(self, requirements: ProjectRequirements) -> Dict[str, Any]:
        """Assess project risks."""
        
        risks = {
            "technical_risks": [],
            "timeline_risks": [],
            "resource_risks": [],
            "business_risks": [],
            "high_risk_factors": [],
            "overall_risk_score": 0.0,
        }
        
        # Technical risks
        if len(requirements.technologies) > 3:
            risks["technical_risks"].append("Complex technology integration")
            
        if any(tech in [TechnologyStack.BLOCKCHAIN, TechnologyStack.MACHINE_LEARNING] 
               for tech in requirements.technologies):
            risks["technical_risks"].append("Cutting-edge technology risks")
        
        # Timeline risks
        estimated_weeks = requirements.estimated_effort_hours // 40
        if requirements.timeline_weeks < estimated_weeks * 0.8:
            risks["timeline_risks"].append("Aggressive timeline")
            risks["high_risk_factors"].append("Timeline pressure")
        
        # Resource risks
        if requirements.budget_range[1] < requirements.estimated_effort_hours * 50:  # $50/hour baseline
            risks["resource_risks"].append("Budget constraints")
            risks["high_risk_factors"].append("Budget limitations")
        
        # Business risks
        if requirements.client_type == "external":
            risks["business_risks"].append("External client expectations")
        
        # Calculate overall risk score
        risk_factors = len(risks["technical_risks"]) + len(risks["timeline_risks"]) + \
                      len(risks["resource_risks"]) + len(risks["business_risks"])
        risks["overall_risk_score"] = min(risk_factors / 10.0, 1.0)
        
        return risks
    
    def _determine_team_requirements(self, requirements: ProjectRequirements) -> Dict[str, Any]:
        """Determine required team composition."""
        
        team_req = {
            "required_roles": [],
            "optional_roles": [],
            "min_team_size": 0,
            "max_team_size": 0,
            "skill_requirements": {},
        }
        
        # Base requirements from complexity
        size_range = self.base_team_sizes[requirements.complexity]
        team_req["min_team_size"] = size_range["min"]
        team_req["max_team_size"] = size_range["max"]
        
        # Role requirements based on technologies
        role_scores = {}
        for tech in requirements.technologies:
            tech_roles = self.role_importance.get(tech, {})
            for role, importance in tech_roles.items():
                if role not in role_scores:
                    role_scores[role] = 0
                role_scores[role] += importance
        
        # Always required roles
        team_req["required_roles"].append(AgentRole.DEVELOPER)
        if requirements.complexity in [ProjectComplexity.COMPLEX, ProjectComplexity.ENTERPRISE]:
            team_req["required_roles"].append(AgentRole.ARCHITECT)
        
        # Add roles based on scores
        for role, score in sorted(role_scores.items(), key=lambda x: x[1], reverse=True):
            if score > 0.7 and role not in team_req["required_roles"]:
                team_req["required_roles"].append(role)
            elif score > 0.4:
                team_req["optional_roles"].append(role)
        
        # Skill requirements
        team_req["skill_requirements"] = {
            role.value: max(0.7, min(1.0, 0.5 + requirements.complexity.value == "enterprise" and 0.3 or 0.2))
            for role in team_req["required_roles"]
        }
        
        return team_req
    
    def _analyze_timeline(self, requirements: ProjectRequirements) -> Dict[str, Any]:
        """Analyze timeline feasibility."""
        
        estimated_weeks = requirements.estimated_effort_hours // 40
        
        timeline_analysis = {
            "estimated_duration_weeks": estimated_weeks,
            "requested_duration_weeks": requirements.timeline_weeks,
            "timeline_feasibility": "feasible",
            "recommended_team_size": 0,
            "critical_path_items": [],
            "buffer_recommendations": {},
        }
        
        # Determine feasibility
        if requirements.timeline_weeks < estimated_weeks * 0.7:
            timeline_analysis["timeline_feasibility"] = "high_risk"
            timeline_analysis["recommended_team_size"] = min(
                int(estimated_weeks / requirements.timeline_weeks * 2), 8
            )
        elif requirements.timeline_weeks < estimated_weeks * 0.9:
            timeline_analysis["timeline_feasibility"] = "tight"
            timeline_analysis["recommended_team_size"] = int(estimated_weeks / requirements.timeline_weeks * 1.5)
        else:
            timeline_analysis["timeline_feasibility"] = "comfortable"
            timeline_analysis["recommended_team_size"] = max(2, estimated_weeks // requirements.timeline_weeks)
        
        # Critical path items
        if TechnologyStack.MACHINE_LEARNING in requirements.technologies:
            timeline_analysis["critical_path_items"].append("Model training and validation")
        if TechnologyStack.DEVOPS_INFRA in requirements.technologies:
            timeline_analysis["critical_path_items"].append("Infrastructure setup")
        
        return timeline_analysis
    
    def _analyze_quality_requirements(self, requirements: ProjectRequirements) -> Dict[str, Any]:
        """Analyze quality requirements and their implications."""
        
        quality_analysis = {
            "quality_gates": [],
            "testing_strategy": [],
            "documentation_requirements": [],
            "performance_targets": {},
            "security_requirements": [],
        }
        
        # Quality gates based on requirements
        for metric, threshold in requirements.quality_requirements.items():
            if threshold > 0.9:
                quality_analysis["quality_gates"].append(f"Strict {metric} requirements")
        
        # Testing strategy
        if requirements.complexity in [ProjectComplexity.COMPLEX, ProjectComplexity.ENTERPRISE]:
            quality_analysis["testing_strategy"].extend([
                "Unit testing (>90% coverage)",
                "Integration testing",
                "End-to-end testing",
                "Performance testing"
            ])
        
        # Security requirements
        if TechnologyStack.WEB_BACKEND in requirements.technologies:
            quality_analysis["security_requirements"].append("API security audit")
        if requirements.client_type == "enterprise":
            quality_analysis["security_requirements"].append("Compliance review")
        
        return quality_analysis


class TeamAssemblyEngine:
    """Assembles optimal teams based on project analysis."""
    
    def __init__(self):
        self.available_agents: List[AgentCapability] = []
        self.team_performance_history: Dict[str, TeamPerformanceMetrics] = {}
        self.project_analyzer = ProjectAnalyzer()
        
        # Machine learning models for team optimization
        self.team_success_predictor = None
        self.agent_synergy_calculator = None
        
        # Initialize with some default agents
        self._initialize_default_agents()
    
    def _initialize_default_agents(self):
        """Initialize with a default pool of agents."""
        
        # Architect agents
        self.available_agents.append(AgentCapability(
            role=AgentRole.ARCHITECT,
            technologies={TechnologyStack.WEB_BACKEND, TechnologyStack.WEB_FRONTEND, TechnologyStack.DEVOPS_INFRA},
            skill_level=0.9,
            availability=True,
            current_load=0.0,
            specializations=["System Design", "Microservices", "Cloud Architecture"],
            performance_score=0.85,
            collaboration_score=0.90,
            last_active=datetime.now()
        ))
        
        # Senior developer agents
        for i in range(3):
            self.available_agents.append(AgentCapability(
                role=AgentRole.DEVELOPER,
                technologies={TechnologyStack.WEB_BACKEND, TechnologyStack.WEB_FRONTEND},
                skill_level=0.8 + i * 0.05,
                availability=True,
                current_load=0.0,
                specializations=["Full Stack Development", "API Design", "Database Design"],
                performance_score=0.80 + i * 0.05,
                collaboration_score=0.85 + i * 0.03,
                last_active=datetime.now()
            ))
        
        # Tester agents
        self.available_agents.append(AgentCapability(
            role=AgentRole.TESTER,
            technologies={TechnologyStack.WEB_BACKEND, TechnologyStack.WEB_FRONTEND, TechnologyStack.MOBILE},
            skill_level=0.85,
            availability=True,
            current_load=0.0,
            specializations=["Automated Testing", "Performance Testing", "Security Testing"],
            performance_score=0.88,
            collaboration_score=0.82,
            last_active=datetime.now()
        ))
        
        # DevOps agent
        self.available_agents.append(AgentCapability(
            role=AgentRole.DEVOPS,
            technologies={TechnologyStack.DEVOPS_INFRA, TechnologyStack.WEB_BACKEND},
            skill_level=0.90,
            availability=True,
            current_load=0.0,
            specializations=["Kubernetes", "CI/CD", "AWS", "Monitoring"],
            performance_score=0.87,
            collaboration_score=0.78,
            last_active=datetime.now()
        ))
        
        # Security agent
        self.available_agents.append(AgentCapability(
            role=AgentRole.SECURITY,
            technologies={TechnologyStack.WEB_BACKEND, TechnologyStack.WEB_FRONTEND, TechnologyStack.MOBILE},
            skill_level=0.92,
            availability=True,
            current_load=0.0,
            specializations=["Security Audit", "Penetration Testing", "Compliance"],
            performance_score=0.91,
            collaboration_score=0.75,
            last_active=datetime.now()
        ))
        
        # UI/UX agent
        self.available_agents.append(AgentCapability(
            role=AgentRole.UI_UX,
            technologies={TechnologyStack.WEB_FRONTEND, TechnologyStack.MOBILE},
            skill_level=0.87,
            availability=True,
            current_load=0.0,
            specializations=["User Experience Design", "Accessibility", "Design Systems"],
            performance_score=0.84,
            collaboration_score=0.89,
            last_active=datetime.now()
        ))
        
        logger.info(f"Initialized team assembly engine with {len(self.available_agents)} default agents")
    
    async def assemble_team(
        self,
        requirements: ProjectRequirements,
        preferences: Optional[Dict[str, Any]] = None
    ) -> TeamComposition:
        """Assemble optimal team for the project."""
        
        # Analyze project requirements
        analysis = await self.project_analyzer.analyze_project(requirements)
        
        # Find candidate agents
        candidates = await self._find_candidate_agents(requirements, analysis)
        
        # Generate team compositions
        compositions = await self._generate_team_compositions(
            requirements, analysis, candidates, preferences
        )
        
        # Select best composition
        best_composition = await self._select_best_composition(compositions)
        
        # Calculate team synergy and performance predictions
        best_composition = await self._enhance_composition_analysis(best_composition, analysis)
        
        return best_composition
    
    async def _find_candidate_agents(
        self,
        requirements: ProjectRequirements,
        analysis: Dict[str, Any]
    ) -> Dict[AgentRole, List[AgentCapability]]:
        """Find candidate agents for each required role."""
        
        candidates = {}
        team_requirements = analysis["team_requirements"]
        
        # Find agents for required roles
        for role in team_requirements["required_roles"]:
            role_candidates = []
            
            for agent in self.available_agents:
                if (agent.role == role and 
                    agent.availability and 
                    agent.current_load < 0.8 and
                    self._agent_matches_requirements(agent, requirements)):
                    
                    role_candidates.append(agent)
            
            # Sort by suitability score
            role_candidates.sort(
                key=lambda a: self._calculate_agent_suitability(a, requirements, analysis),
                reverse=True
            )
            
            candidates[role] = role_candidates
        
        # Find agents for optional roles
        for role in team_requirements["optional_roles"]:
            if role not in candidates:
                role_candidates = []
                
                for agent in self.available_agents:
                    if (agent.role == role and 
                        agent.availability and 
                        agent.current_load < 0.6 and
                        self._agent_matches_requirements(agent, requirements)):
                        
                        role_candidates.append(agent)
                
                role_candidates.sort(
                    key=lambda a: self._calculate_agent_suitability(a, requirements, analysis),
                    reverse=True
                )
                
                candidates[role] = role_candidates
        
        return candidates
    
    def _agent_matches_requirements(
        self,
        agent: AgentCapability,
        requirements: ProjectRequirements
    ) -> bool:
        """Check if agent's technologies match project requirements."""
        
        # Agent must have at least one overlapping technology
        tech_overlap = agent.technologies.intersection(requirements.technologies)
        if not tech_overlap:
            return False
        
        # Check skill level requirements
        min_skill_required = 0.6
        if requirements.complexity == ProjectComplexity.ENTERPRISE:
            min_skill_required = 0.8
        elif requirements.complexity == ProjectComplexity.COMPLEX:
            min_skill_required = 0.7
        
        return agent.skill_level >= min_skill_required
    
    def _calculate_agent_suitability(
        self,
        agent: AgentCapability,
        requirements: ProjectRequirements,
        analysis: Dict[str, Any]
    ) -> float:
        """Calculate how suitable an agent is for the project."""
        
        score = 0.0
        
        # Technology match score (40%)
        tech_overlap = agent.technologies.intersection(requirements.technologies)
        tech_score = len(tech_overlap) / max(len(requirements.technologies), 1)
        score += tech_score * 0.4
        
        # Skill level score (30%)
        score += agent.skill_level * 0.3
        
        # Performance score (20%)
        score += agent.performance_score * 0.2
        
        # Availability score (10%)
        availability_score = 1.0 - agent.current_load
        score += availability_score * 0.1
        
        # Specialization bonus
        required_expertise = analysis["technology_analysis"]["required_expertise_areas"]
        specialization_matches = len([
            spec for spec in agent.specializations
            if any(req in spec for req in required_expertise)
        ])
        if specialization_matches > 0:
            score += 0.1 * min(specialization_matches / len(required_expertise), 0.5)
        
        return score
    
    async def _generate_team_compositions(
        self,
        requirements: ProjectRequirements,
        analysis: Dict[str, Any],
        candidates: Dict[AgentRole, List[AgentCapability]],
        preferences: Optional[Dict[str, Any]] = None
    ) -> List[TeamComposition]:
        """Generate multiple team composition options."""
        
        compositions = []
        team_req = analysis["team_requirements"]
        
        # Generate primary composition (best agents for each role)
        primary_team = []
        team_lead_id = None
        
        for role in team_req["required_roles"]:
            if role in candidates and candidates[role]:
                best_agent = candidates[role][0]
                primary_team.append(best_agent)
                
                # Architect or senior developer as team lead
                if role in [AgentRole.ARCHITECT, AgentRole.DEVELOPER] and not team_lead_id:
                    team_lead_id = best_agent.agent_id
        
        # Add optional roles based on budget and team size
        max_team_size = team_req["max_team_size"]
        current_size = len(primary_team)
        
        for role in team_req["optional_roles"]:
            if (current_size < max_team_size and 
                role in candidates and 
                candidates[role]):
                
                primary_team.append(candidates[role][0])
                current_size += 1
        
        # Create primary composition
        if primary_team:
            primary_composition = TeamComposition(
                project_id=requirements.project_id,
                agents=primary_team,
                team_lead=team_lead_id or primary_team[0].agent_id,
                estimated_completion=datetime.now() + timedelta(weeks=requirements.timeline_weeks),
                confidence_score=0.0,  # Will be calculated
                cost_estimate=0.0,    # Will be calculated
                risk_assessment={},   # Will be calculated
                team_synergy_score=0.0  # Will be calculated
            )
            compositions.append(primary_composition)
        
        # Generate alternative compositions with different agent combinations
        if len(compositions) > 0:
            alternatives = await self._generate_alternative_compositions(
                requirements, analysis, candidates, primary_composition
            )
            compositions.extend(alternatives[:3])  # Max 3 alternatives
        
        return compositions
    
    async def _generate_alternative_compositions(
        self,
        requirements: ProjectRequirements,
        analysis: Dict[str, Any],
        candidates: Dict[AgentRole, List[AgentCapability]],
        primary: TeamComposition
    ) -> List[TeamComposition]:
        """Generate alternative team compositions."""
        
        alternatives = []
        
        # Alternative 1: Cost-optimized team (younger agents)
        cost_optimized_team = []
        for role in analysis["team_requirements"]["required_roles"]:
            if role in candidates:
                # Pick agent with good skill but lower cost (based on performance score as proxy)
                suitable_agents = [a for a in candidates[role] if a.skill_level >= 0.6]
                if suitable_agents:
                    # Sort by cost-effectiveness (skill/performance ratio)
                    suitable_agents.sort(key=lambda a: a.skill_level / max(a.performance_score, 0.1))
                    cost_optimized_team.append(suitable_agents[0])
        
        if cost_optimized_team:
            alt1 = TeamComposition(
                project_id=requirements.project_id,
                agents=cost_optimized_team,
                team_lead=cost_optimized_team[0].agent_id,
                estimated_completion=datetime.now() + timedelta(weeks=requirements.timeline_weeks + 2),
                confidence_score=0.0,
                cost_estimate=0.0,
                risk_assessment={},
                team_synergy_score=0.0
            )
            alternatives.append(alt1)
        
        # Alternative 2: Speed-optimized team (more agents, parallel work)
        if requirements.timeline_weeks < 8:  # Only for tight timelines
            speed_team = list(primary.agents)  # Start with primary team
            
            # Add more developers if available
            if AgentRole.DEVELOPER in candidates:
                additional_devs = [
                    agent for agent in candidates[AgentRole.DEVELOPER]
                    if agent not in speed_team
                ][:2]  # Max 2 additional
                speed_team.extend(additional_devs)
            
            if len(speed_team) > len(primary.agents):
                alt2 = TeamComposition(
                    project_id=requirements.project_id,
                    agents=speed_team,
                    team_lead=primary.team_lead,
                    estimated_completion=datetime.now() + timedelta(weeks=max(1, requirements.timeline_weeks - 1)),
                    confidence_score=0.0,
                    cost_estimate=0.0,
                    risk_assessment={},
                    team_synergy_score=0.0
                )
                alternatives.append(alt2)
        
        return alternatives
    
    async def _select_best_composition(
        self,
        compositions: List[TeamComposition]
    ) -> TeamComposition:
        """Select the best team composition from alternatives."""
        
        if not compositions:
            raise ValueError("No valid team compositions generated")
        
        # Calculate scores for each composition
        for composition in compositions:
            composition.confidence_score = await self._calculate_composition_confidence(composition)
            composition.cost_estimate = self._estimate_team_cost(composition)
            composition.team_synergy_score = self._calculate_team_synergy(composition)
        
        # Score each composition
        def composition_score(comp: TeamComposition) -> float:
            return (comp.confidence_score * 0.4 + 
                    comp.team_synergy_score * 0.3 +
                    (1.0 / max(comp.cost_estimate / 10000, 1)) * 0.2 +  # Inverse cost factor
                    (len(comp.agents) / 10.0) * 0.1)  # Team size factor
        
        best_composition = max(compositions, key=composition_score)
        
        # Set alternatives
        alternatives = [c for c in compositions if c != best_composition]
        best_composition.alternative_compositions = alternatives
        
        return best_composition
    
    async def _calculate_composition_confidence(
        self,
        composition: TeamComposition
    ) -> float:
        """Calculate confidence score for team composition."""
        
        if not composition.agents:
            return 0.0
        
        # Average agent skill level (40%)
        avg_skill = sum(agent.skill_level for agent in composition.agents) / len(composition.agents)
        skill_score = avg_skill * 0.4
        
        # Average performance score (30%)
        avg_performance = sum(agent.performance_score for agent in composition.agents) / len(composition.agents)
        performance_score = avg_performance * 0.3
        
        # Role coverage (20%)
        unique_roles = len(set(agent.role for agent in composition.agents))
        role_coverage_score = min(unique_roles / 5.0, 1.0) * 0.2
        
        # Team availability (10%)
        avg_availability = sum(1.0 - agent.current_load for agent in composition.agents) / len(composition.agents)
        availability_score = avg_availability * 0.1
        
        return skill_score + performance_score + role_coverage_score + availability_score
    
    def _estimate_team_cost(self, composition: TeamComposition) -> float:
        """Estimate total cost for the team."""
        
        # Base hourly rates by role (in dollars)
        role_rates = {
            AgentRole.ARCHITECT: 120,
            AgentRole.DEVELOPER: 80,
            AgentRole.TESTER: 70,
            AgentRole.REVIEWER: 75,
            AgentRole.SECURITY: 100,
            AgentRole.DEVOPS: 90,
            AgentRole.UI_UX: 85,
            AgentRole.TECH_WRITER: 60,
            AgentRole.PROJECT_MANAGER: 95,
            AgentRole.DATA_ANALYST: 85,
        }
        
        total_cost = 0.0
        estimated_hours_per_agent = 160  # 4 weeks * 40 hours (rough estimate)
        
        for agent in composition.agents:
            base_rate = role_rates.get(agent.role, 75)
            # Adjust rate based on skill level
            adjusted_rate = base_rate * (0.7 + agent.skill_level * 0.6)  # 70% to 130% of base
            total_cost += adjusted_rate * estimated_hours_per_agent
        
        return total_cost
    
    def _calculate_team_synergy(self, composition: TeamComposition) -> float:
        """Calculate team synergy score."""
        
        if len(composition.agents) < 2:
            return 0.8  # Default for single agent
        
        # Average collaboration score (50%)
        avg_collaboration = sum(agent.collaboration_score for agent in composition.agents) / len(composition.agents)
        collab_score = avg_collaboration * 0.5
        
        # Role diversity bonus (30%)
        unique_roles = len(set(agent.role for agent in composition.agents))
        diversity_score = min(unique_roles / len(composition.agents), 1.0) * 0.3
        
        # Skill balance (20%)
        skill_levels = [agent.skill_level for agent in composition.agents]
        skill_variance = np.var(skill_levels) if len(skill_levels) > 1 else 0
        skill_balance_score = max(0, 1.0 - skill_variance) * 0.2
        
        return collab_score + diversity_score + skill_balance_score
    
    async def _enhance_composition_analysis(
        self,
        composition: TeamComposition,
        analysis: Dict[str, Any]
    ) -> TeamComposition:
        """Enhance composition with detailed analysis."""
        
        # Risk assessment
        composition.risk_assessment = {
            "technical_risk": analysis["risk_assessment"]["overall_risk_score"],
            "team_experience_risk": self._assess_team_experience_risk(composition),
            "collaboration_risk": self._assess_collaboration_risk(composition),
            "timeline_risk": self._assess_timeline_risk(composition, analysis),
        }
        
        # Adjust estimated completion based on team capabilities
        skill_factor = sum(agent.skill_level for agent in composition.agents) / len(composition.agents)
        if skill_factor > 0.85:
            # High skill team can work faster
            days_reduction = int((skill_factor - 0.85) * 14)  # Up to 2 weeks faster
            composition.estimated_completion -= timedelta(days=days_reduction)
        elif skill_factor < 0.65:
            # Lower skill team needs more time
            days_addition = int((0.65 - skill_factor) * 21)  # Up to 3 weeks longer
            composition.estimated_completion += timedelta(days=days_addition)
        
        return composition
    
    def _assess_team_experience_risk(self, composition: TeamComposition) -> float:
        """Assess risk based on team experience levels."""
        
        # Low risk if most agents are experienced
        experienced_agents = sum(1 for agent in composition.agents if agent.skill_level > 0.8)
        experience_ratio = experienced_agents / len(composition.agents)
        
        if experience_ratio > 0.7:
            return 0.1  # Low risk
        elif experience_ratio > 0.5:
            return 0.3  # Medium risk
        else:
            return 0.7  # High risk
    
    def _assess_collaboration_risk(self, composition: TeamComposition) -> float:
        """Assess collaboration risk based on team dynamics."""
        
        avg_collab_score = sum(agent.collaboration_score for agent in composition.agents) / len(composition.agents)
        
        # Convert collaboration score to risk (inverse relationship)
        return max(0.1, 1.0 - avg_collab_score)
    
    def _assess_timeline_risk(self, composition: TeamComposition, analysis: Dict[str, Any]) -> float:
        """Assess timeline risk based on team size and project complexity."""
        
        timeline_analysis = analysis["timeline_analysis"]
        
        if timeline_analysis["timeline_feasibility"] == "high_risk":
            return 0.8
        elif timeline_analysis["timeline_feasibility"] == "tight":
            return 0.5
        else:
            return 0.2
    
    async def get_available_agents(
        self,
        role_filter: Optional[AgentRole] = None,
        technology_filter: Optional[Set[TechnologyStack]] = None,
        min_skill_level: float = 0.0
    ) -> List[AgentCapability]:
        """Get available agents with optional filters."""
        
        filtered_agents = []
        
        for agent in self.available_agents:
            # Role filter
            if role_filter and agent.role != role_filter:
                continue
            
            # Technology filter
            if technology_filter and not agent.technologies.intersection(technology_filter):
                continue
            
            # Skill level filter
            if agent.skill_level < min_skill_level:
                continue
            
            # Availability filter
            if not agent.availability or agent.current_load >= 1.0:
                continue
            
            filtered_agents.append(agent)
        
        return filtered_agents
    
    async def add_agent(self, agent: AgentCapability) -> str:
        """Add new agent to the pool."""
        
        self.available_agents.append(agent)
        logger.info(f"Added new {agent.role.value} agent: {agent.agent_id}")
        return agent.agent_id
    
    async def update_agent_availability(
        self,
        agent_id: str,
        availability: bool,
        current_load: Optional[float] = None
    ):
        """Update agent availability and workload."""
        
        for agent in self.available_agents:
            if agent.agent_id == agent_id:
                agent.availability = availability
                if current_load is not None:
                    agent.current_load = current_load
                agent.last_active = datetime.now()
                logger.info(f"Updated agent {agent_id} availability: {availability}")
                return
        
        logger.warning(f"Agent {agent_id} not found for availability update")
    
    async def get_team_assembly_stats(self) -> Dict[str, Any]:
        """Get team assembly statistics."""
        
        total_agents = len(self.available_agents)
        available_agents = sum(1 for agent in self.available_agents if agent.availability and agent.current_load < 0.8)
        
        role_distribution = {}
        for agent in self.available_agents:
            role = agent.role.value
            if role not in role_distribution:
                role_distribution[role] = 0
            role_distribution[role] += 1
        
        avg_skill_level = sum(agent.skill_level for agent in self.available_agents) / max(total_agents, 1)
        avg_performance = sum(agent.performance_score for agent in self.available_agents) / max(total_agents, 1)
        
        return {
            "total_agents": total_agents,
            "available_agents": available_agents,
            "utilization_rate": (total_agents - available_agents) / max(total_agents, 1),
            "role_distribution": role_distribution,
            "average_skill_level": avg_skill_level,
            "average_performance_score": avg_performance,
            "active_teams": len(self.team_performance_history),
        }