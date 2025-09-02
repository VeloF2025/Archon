"""
Intelligent Task Router for Phase 2 Meta-Agent Orchestration
Routes tasks to optimal agents based on capabilities and load
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

from .parallel_executor import AgentTask

logger = logging.getLogger(__name__)


@dataclass
class AgentCapability:
    """Agent capability profile"""
    agent_id: str
    agent_role: str
    specializations: List[str] = field(default_factory=list)
    supported_languages: List[str] = field(default_factory=list)
    supported_frameworks: List[str] = field(default_factory=list)
    max_complexity: int = 10  # 1-10 scale
    average_response_time: float = 0.0
    success_rate: float = 1.0
    current_load: int = 0
    max_concurrent_tasks: int = 5
    last_task_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Routing decision with scoring details"""
    task_id: str
    agent_id: str
    agent_role: str
    score: float
    factors: Dict[str, float] = field(default_factory=dict)
    fallback_agents: List[str] = field(default_factory=list)
    routing_time: float = 0.0
    decision_reason: str = ""


@dataclass
class RoutingMetrics:
    """Metrics for routing performance"""
    total_routed: int = 0
    successful_routes: int = 0
    failed_routes: int = 0
    rerouted_tasks: int = 0
    average_routing_time: float = 0.0
    average_score: float = 0.0
    agent_utilization: Dict[str, float] = field(default_factory=dict)


class IntelligentTaskRouter:
    """
    Routes tasks to optimal agents based on capabilities,
    load, and historical performance.
    """
    
    def __init__(self):
        """Initialize the intelligent task router"""
        self.agent_capabilities: Dict[str, AgentCapability] = {}
        self.agent_metrics: Dict[str, Dict[str, Any]] = {}
        self.routing_history: List[RoutingDecision] = []
        self.routing_metrics = RoutingMetrics()
        
        # Scoring weights (configurable)
        self.scoring_weights = {
            'specialization_match': 0.40,
            'current_load': 0.30,
            'historical_success': 0.20,
            'response_time': 0.10
        }
        
        # Initialize default agents
        self._initialize_default_agents()
        
        logger.info("Initialized IntelligentTaskRouter")
    
    def _initialize_default_agents(self):
        """Initialize default agent capabilities"""
        # Python Backend Agent
        self.register_agent(AgentCapability(
            agent_id="python_backend_1",
            agent_role="python_backend",
            specializations=["backend", "api", "database", "data_processing"],
            supported_languages=["python"],
            supported_frameworks=["fastapi", "django", "flask", "sqlalchemy"],
            max_complexity=10,
            average_response_time=15.0,
            success_rate=0.95
        ))
        
        # TypeScript Frontend Agent
        self.register_agent(AgentCapability(
            agent_id="typescript_frontend_1",
            agent_role="typescript_frontend",
            specializations=["frontend", "ui", "react", "web"],
            supported_languages=["typescript", "javascript"],
            supported_frameworks=["react", "nextjs", "vue", "angular"],
            max_complexity=9,
            average_response_time=12.0,
            success_rate=0.93
        ))
        
        # Full Stack Agent
        self.register_agent(AgentCapability(
            agent_id="fullstack_1",
            agent_role="fullstack",
            specializations=["fullstack", "api", "ui", "database"],
            supported_languages=["python", "typescript", "javascript"],
            supported_frameworks=["fastapi", "react", "nextjs"],
            max_complexity=8,
            average_response_time=20.0,
            success_rate=0.90
        ))
        
        # Security Agent
        self.register_agent(AgentCapability(
            agent_id="security_1",
            agent_role="security_auditor",
            specializations=["security", "vulnerability", "authentication", "encryption"],
            supported_languages=["python", "typescript", "go"],
            supported_frameworks=["owasp", "oauth", "jwt"],
            max_complexity=10,
            average_response_time=25.0,
            success_rate=0.98
        ))
        
        # Database Agent
        self.register_agent(AgentCapability(
            agent_id="database_1",
            agent_role="database_architect",
            specializations=["database", "sql", "optimization", "migrations"],
            supported_languages=["sql", "python"],
            supported_frameworks=["postgresql", "mysql", "mongodb", "redis"],
            max_complexity=9,
            average_response_time=18.0,
            success_rate=0.94
        ))
    
    def register_agent(self, capability: AgentCapability):
        """Register an agent with its capabilities"""
        self.agent_capabilities[capability.agent_id] = capability
        self.agent_metrics[capability.agent_id] = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'last_updated': datetime.now()
        }
        logger.info(f"Registered agent {capability.agent_id} with role {capability.agent_role}")
    
    async def route_task(self, task: AgentTask) -> str:
        """
        Select optimal agent for task.
        
        Args:
            task: Task to route
            
        Returns:
            Agent ID for task execution
        """
        start_time = time.time()
        
        # Extract task requirements
        requirements = self._extract_task_requirements(task)
        
        # Score all available agents
        agent_scores = []
        for agent_id, capability in self.agent_capabilities.items():
            if self._is_agent_available(capability):
                score, factors = self._calculate_agent_score(capability, requirements)
                agent_scores.append((agent_id, score, factors))
        
        # Sort by score (highest first)
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        if not agent_scores:
            logger.warning(f"No available agents for task {task.task_id}")
            # Fallback to any agent
            return list(self.agent_capabilities.keys())[0] if self.agent_capabilities else ""
        
        # Select best agent
        best_agent_id, best_score, best_factors = agent_scores[0]
        
        # Create routing decision
        decision = RoutingDecision(
            task_id=task.task_id,
            agent_id=best_agent_id,
            agent_role=self.agent_capabilities[best_agent_id].agent_role,
            score=best_score,
            factors=best_factors,
            fallback_agents=[aid for aid, _, _ in agent_scores[1:4]],  # Top 3 fallbacks
            routing_time=time.time() - start_time,
            decision_reason=self._explain_routing_decision(best_factors)
        )
        
        # Update agent load
        self.agent_capabilities[best_agent_id].current_load += 1
        self.agent_capabilities[best_agent_id].last_task_time = datetime.now()
        
        # Record decision
        self.routing_history.append(decision)
        self.routing_metrics.total_routed += 1
        
        logger.info(f"Routed task {task.task_id} to {best_agent_id} (score: {best_score:.2f})")
        logger.debug(f"Routing factors: {best_factors}")
        
        return best_agent_id
    
    def _extract_task_requirements(self, task: AgentTask) -> Dict[str, Any]:
        """Extract requirements from task"""
        requirements = {
            'role': task.agent_role,
            'languages': [],
            'frameworks': [],
            'specializations': [],
            'complexity': 5  # Default medium complexity
        }
        
        # Parse task description for requirements
        if hasattr(task, 'input') and task.input:
            input_lower = task.input.lower() if isinstance(task.input, str) else str(task.input).lower()
            
            # Detect languages
            language_keywords = {
                'python': ['python', 'py', 'django', 'flask', 'fastapi'],
                'typescript': ['typescript', 'ts', 'tsx'],
                'javascript': ['javascript', 'js', 'jsx', 'node'],
                'sql': ['sql', 'database', 'query', 'postgresql', 'mysql']
            }
            
            for lang, keywords in language_keywords.items():
                if any(kw in input_lower for kw in keywords):
                    requirements['languages'].append(lang)
            
            # Detect frameworks
            framework_keywords = {
                'react': ['react', 'component', 'hooks', 'jsx'],
                'fastapi': ['fastapi', 'api', 'endpoint'],
                'nextjs': ['nextjs', 'next.js', 'ssr'],
                'django': ['django', 'orm'],
                'postgresql': ['postgresql', 'postgres', 'pg']
            }
            
            for framework, keywords in framework_keywords.items():
                if any(kw in input_lower for kw in keywords):
                    requirements['frameworks'].append(framework)
            
            # Detect specializations
            spec_keywords = {
                'frontend': ['ui', 'frontend', 'component', 'style', 'css'],
                'backend': ['backend', 'api', 'server', 'endpoint'],
                'database': ['database', 'sql', 'query', 'schema', 'migration'],
                'security': ['security', 'auth', 'encryption', 'vulnerability'],
                'optimization': ['optimize', 'performance', 'speed', 'cache']
            }
            
            for spec, keywords in spec_keywords.items():
                if any(kw in input_lower for kw in keywords):
                    requirements['specializations'].append(spec)
            
            # Estimate complexity (simple heuristic)
            if any(word in input_lower for word in ['complex', 'advanced', 'enterprise', 'large']):
                requirements['complexity'] = 8
            elif any(word in input_lower for word in ['simple', 'basic', 'small', 'quick']):
                requirements['complexity'] = 3
        
        return requirements
    
    def _calculate_agent_score(self, capability: AgentCapability, requirements: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate agent suitability score for task.
        
        Returns:
            Tuple of (total_score, factor_scores)
        """
        factors = {}
        
        # 1. Specialization match (40%)
        spec_score = 0.0
        if requirements['specializations']:
            matches = len(set(capability.specializations) & set(requirements['specializations']))
            spec_score = matches / len(requirements['specializations']) if requirements['specializations'] else 0.5
        else:
            # No specific specialization required, check role match
            spec_score = 1.0 if capability.agent_role == requirements['role'] else 0.5
        
        factors['specialization_match'] = spec_score
        
        # 2. Current load (30%)
        load_ratio = capability.current_load / capability.max_concurrent_tasks
        load_score = max(0, 1.0 - load_ratio)  # Higher score for lower load
        factors['current_load'] = load_score
        
        # 3. Historical success (20%)
        factors['historical_success'] = capability.success_rate
        
        # 4. Response time (10%)
        # Normalize response time (faster is better)
        if capability.average_response_time > 0:
            time_score = min(1.0, 10.0 / capability.average_response_time)  # 10s or less = perfect
        else:
            time_score = 0.5  # No data
        factors['response_time'] = time_score
        
        # Calculate weighted total
        total_score = sum(
            factors[key] * self.scoring_weights.get(key, 0)
            for key in factors
        )
        
        return total_score, factors
    
    def _is_agent_available(self, capability: AgentCapability) -> bool:
        """Check if agent is available for new tasks"""
        # Check load
        if capability.current_load >= capability.max_concurrent_tasks:
            return False
        
        # Check if agent was recently used (simple throttling)
        if capability.last_task_time:
            time_since_last = datetime.now() - capability.last_task_time
            if time_since_last < timedelta(milliseconds=100):  # 100ms cooldown
                return False
        
        return True
    
    def _explain_routing_decision(self, factors: Dict[str, float]) -> str:
        """Generate human-readable explanation of routing decision"""
        explanations = []
        
        for factor, score in factors.items():
            weight = self.scoring_weights.get(factor, 0)
            weighted_score = score * weight
            
            if factor == 'specialization_match':
                explanations.append(f"Specialization match: {score:.0%} (weight: {weight})")
            elif factor == 'current_load':
                explanations.append(f"Load availability: {score:.0%} (weight: {weight})")
            elif factor == 'historical_success':
                explanations.append(f"Success rate: {score:.0%} (weight: {weight})")
            elif factor == 'response_time':
                explanations.append(f"Response time score: {score:.0%} (weight: {weight})")
        
        return " | ".join(explanations)
    
    async def get_fallback_agent(self, task: AgentTask, failed_agent_id: str) -> str:
        """
        Find alternative agent if primary fails.
        
        Args:
            task: Task that failed
            failed_agent_id: Agent that failed
            
        Returns:
            Alternative agent ID
        """
        # Mark failure for the agent
        if failed_agent_id in self.agent_capabilities:
            self.agent_capabilities[failed_agent_id].current_load = max(0, self.agent_capabilities[failed_agent_id].current_load - 1)
            
            # Update success rate
            metrics = self.agent_metrics[failed_agent_id]
            metrics['tasks_failed'] += 1
            total = metrics['tasks_completed'] + metrics['tasks_failed']
            if total > 0:
                self.agent_capabilities[failed_agent_id].success_rate = metrics['tasks_completed'] / total
        
        # Find fallback from routing history
        for decision in reversed(self.routing_history):
            if decision.task_id == task.task_id and decision.fallback_agents:
                for fallback_id in decision.fallback_agents:
                    if fallback_id != failed_agent_id and self._is_agent_available(self.agent_capabilities.get(fallback_id)):
                        logger.info(f"Using fallback agent {fallback_id} for task {task.task_id}")
                        self.routing_metrics.rerouted_tasks += 1
                        return fallback_id
        
        # No fallback found, route normally but exclude failed agent
        temp_capability = self.agent_capabilities.pop(failed_agent_id, None)
        try:
            fallback_id = await self.route_task(task)
        finally:
            if temp_capability:
                self.agent_capabilities[failed_agent_id] = temp_capability
        
        return fallback_id
    
    def update_task_result(self, agent_id: str, task_id: str, success: bool, execution_time: float):
        """
        Update routing metrics based on task result.
        
        Args:
            agent_id: Agent that executed the task
            task_id: Task ID
            success: Whether task succeeded
            execution_time: Task execution time
        """
        if agent_id not in self.agent_capabilities:
            return
        
        # Update agent metrics
        capability = self.agent_capabilities[agent_id]
        metrics = self.agent_metrics[agent_id]
        
        if success:
            metrics['tasks_completed'] += 1
            self.routing_metrics.successful_routes += 1
        else:
            metrics['tasks_failed'] += 1
            self.routing_metrics.failed_routes += 1
        
        metrics['total_execution_time'] += execution_time
        
        # Update capability stats
        total_tasks = metrics['tasks_completed'] + metrics['tasks_failed']
        if total_tasks > 0:
            capability.success_rate = metrics['tasks_completed'] / total_tasks
            capability.average_response_time = metrics['total_execution_time'] / total_tasks
        
        # Update load
        capability.current_load = max(0, capability.current_load - 1)
        
        logger.debug(f"Updated metrics for agent {agent_id}: success_rate={capability.success_rate:.2%}, avg_time={capability.average_response_time:.1f}s")
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        stats = {
            'total_routed': self.routing_metrics.total_routed,
            'successful_routes': self.routing_metrics.successful_routes,
            'failed_routes': self.routing_metrics.failed_routes,
            'rerouted_tasks': self.routing_metrics.rerouted_tasks,
            'success_rate': self.routing_metrics.successful_routes / self.routing_metrics.total_routed if self.routing_metrics.total_routed > 0 else 0,
            'agents': {}
        }
        
        # Per-agent statistics
        for agent_id, capability in self.agent_capabilities.items():
            metrics = self.agent_metrics[agent_id]
            total_tasks = metrics['tasks_completed'] + metrics['tasks_failed']
            
            stats['agents'][agent_id] = {
                'role': capability.agent_role,
                'current_load': capability.current_load,
                'max_load': capability.max_concurrent_tasks,
                'utilization': capability.current_load / capability.max_concurrent_tasks,
                'tasks_completed': metrics['tasks_completed'],
                'tasks_failed': metrics['tasks_failed'],
                'success_rate': capability.success_rate,
                'average_response_time': capability.average_response_time,
                'total_tasks': total_tasks
            }
        
        return stats
    
    def reset_metrics(self):
        """Reset routing metrics (for testing)"""
        self.routing_metrics = RoutingMetrics()
        self.routing_history.clear()
        
        for agent_id in self.agent_capabilities:
            self.agent_capabilities[agent_id].current_load = 0
            self.agent_metrics[agent_id] = {
                'tasks_completed': 0,
                'tasks_failed': 0,
                'total_execution_time': 0.0,
                'last_updated': datetime.now()
            }
        
        logger.info("Reset all routing metrics")