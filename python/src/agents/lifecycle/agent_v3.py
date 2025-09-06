"""
Agent Lifecycle Management v3.0 - Core Implementation
Based on Agent_Lifecycle_Management_PRP.md specifications

NLNH Protocol: Real implementation that makes TDD tests pass
DGTS Enforcement: No fake behavior, actual state management
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
from pydantic import BaseModel

from ..base_agent import BaseAgent, ArchonDependencies, BaseAgentOutput
from .global_rules_integrator import GlobalRulesIntegrator, RulesProfile

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent state enumeration as specified in PRP Section 1.1.1"""
    CREATED = "created"
    ACTIVE = "active" 
    IDLE = "idle"
    HIBERNATED = "hibernated"
    ARCHIVED = "archived"


@dataclass
class StateTransitionLog:
    """State transition log entry"""
    id: str
    agent_id: str
    from_state: str
    to_state: str
    transition_timestamp: datetime
    trigger_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeItem:
    """Knowledge item for agent learning"""
    id: str
    agent_id: str
    project_id: str
    item_type: str  # pattern, decision, failure, optimization
    content: Dict[str, Any]
    confidence: float
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentV3(BaseAgent):
    """
    Agent v3 with full lifecycle management
    Implementation for test_agent_lifecycle_v3.py tests
    """
    
    def __init__(
        self,
        project_id: str,
        name: str,
        agent_type: str,
        model_tier: str,
        specialization: Optional[str] = None,
        **kwargs
    ):
        # Validate model tier
        if model_tier not in ["opus", "sonnet", "haiku"]:
            raise ValueError(f"Invalid model tier: {model_tier}. Must be opus, sonnet, or haiku")
            
        # Map model tier to actual Claude model
        model_mapping = {
            "opus": "claude-3-opus-20240229",
            "sonnet": "claude-3-5-sonnet-20241022", 
            "haiku": "claude-3-haiku-20240307"
        }
        
        super().__init__(
            model=f"anthropic:{model_mapping[model_tier]}",
            name=name,
            **kwargs
        )
        
        self.id = str(uuid.uuid4())
        self.project_id = project_id
        self.agent_type = agent_type
        self.model_tier = model_tier
        self.specialization = specialization
        
        # Lifecycle management
        self.state = AgentState.CREATED
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        self.hibernated_at: Optional[datetime] = None
        self.archived_at: Optional[datetime] = None
        
        # Performance tracking
        self.total_tasks = 0
        self.success_rate = 0.0
        self.avg_execution_time_ms = 0
        self.total_cost_usd = 0.0
        
        # Memory and knowledge
        self.memory: Dict[str, Any] = {}
        self.knowledge_items: List[KnowledgeItem] = []
        self.state_transition_log: List[StateTransitionLog] = []
        
        # Global rules integration
        self.global_rules_integrator = GlobalRulesIntegrator()
        self.rules_profile: Optional[RulesProfile] = None
        self._rules_loaded = False
        
        # Hibernation check task
        self._hibernation_check_task: Optional[asyncio.Task] = None
        self._start_hibernation_monitoring()
        
        # Load global rules asynchronously
        asyncio.create_task(self._initialize_global_rules())

    def _create_agent(self, **kwargs):
        """Create the underlying PydanticAI agent"""
        from pydantic_ai import Agent
        
        return Agent(
            model=self.model,
            system_prompt=self.get_manifest_enhanced_system_prompt(),
            **kwargs
        )

    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        base_prompt = f"""
You are {self.name}, a specialized {self.agent_type} agent in the Archon v3.0 system.

Model Tier: {self.model_tier}
Specialization: {self.specialization or "General purpose"}
Current State: {self.state.value}

Your capabilities are optimized for your tier:
- Opus: Complex architecture, security analysis, creative problem solving
- Sonnet: Feature implementation, testing, refactoring, general development  
- Haiku: Simple tasks, formatting, documentation, basic fixes

You maintain persistent memory and can learn from past experiences.
Always provide structured, actionable responses within your specialization.
"""
        
        # Add global rules if loaded
        if self._rules_loaded and self.rules_profile:
            enhanced_prompt = self.rules_profile.combined_system_prompt
            return f"{enhanced_prompt}\n\n{base_prompt}\n\n🔒 REMINDER: Follow all global rules and protocols above."
        
        return base_prompt

    async def _initialize_global_rules(self) -> None:
        """Initialize global rules integration for this agent"""
        try:
            logger.info(f"Loading global rules for agent {self.name}...")
            
            # Create rules profile for this agent
            self.rules_profile = await self.global_rules_integrator.create_agent_rules_profile(
                agent_id=self.id,
                agent_type=self.agent_type,
                model_tier=self.model_tier,
                project_id=self.project_id,
                specialization=self.specialization
            )
            
            self._rules_loaded = True
            
            logger.info(f"Global rules loaded for {self.name}: "
                       f"{len(self.rules_profile.global_rules)} global rules, "
                       f"{len(self.rules_profile.project_rules)} project rules")
            
            # Log critical rules for monitoring
            critical_rules = [r for r in self.rules_profile.global_rules + self.rules_profile.project_rules 
                            if r.enforcement_level in ['BLOCKING', 'CRITICAL', 'MANDATORY']]
            
            if critical_rules:
                logger.info(f"Agent {self.name} loaded {len(critical_rules)} critical enforcement rules")
                for rule in critical_rules[:3]:  # Log first 3
                    logger.debug(f"Critical rule: {rule.title}")
                    
        except Exception as e:
            logger.error(f"Failed to load global rules for agent {self.name}: {e}")
            self._rules_loaded = False

    async def validate_action_compliance(self, action: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate an action against global rules before execution"""
        if not self._rules_loaded or not self.rules_profile:
            return {"compliant": True, "message": "Rules not loaded, allowing action"}
        
        try:
            # Use global rules integrator to validate
            validation_result = await self.global_rules_integrator.validate_agent_compliance(
                agent_id=self.id,
                action=action,
                context=context or {}
            )
            
            # Log compliance violations
            if not validation_result["compliant"]:
                logger.warning(f"Agent {self.name} compliance violation for action '{action}': "
                             f"{validation_result['violations']}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Rules compliance validation failed for agent {self.name}: {e}")
            return {"compliant": True, "message": "Validation failed, allowing action with warning"}

    async def get_applicable_rules(self) -> List[str]:
        """Get list of rules applicable to this agent"""
        if not self._rules_loaded or not self.rules_profile:
            return []
        
        rules = []
        for rule in self.rules_profile.global_rules + self.rules_profile.project_rules:
            rules.append(f"{rule.title} ({rule.enforcement_level}): {rule.description[:100]}...")
        
        return rules

    async def refresh_global_rules(self) -> bool:
        """Refresh global rules from source files"""
        try:
            logger.info(f"Refreshing global rules for agent {self.name}...")
            
            # Force refresh of rules cache
            await self.global_rules_integrator.load_global_rules(force_refresh=True)
            
            # Reinitialize rules profile
            await self._initialize_global_rules()
            
            logger.info(f"Global rules refreshed for agent {self.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to refresh global rules for agent {self.name}: {e}")
            return False

    async def transition_to_active(self, reason: str) -> None:
        """Transition agent to ACTIVE state"""
        await self._validate_and_execute_transition(AgentState.ACTIVE, reason)

    async def transition_to_idle(self, reason: str) -> None:
        """Transition agent to IDLE state"""
        await self._validate_and_execute_transition(AgentState.IDLE, reason)
        
        # Start hibernation timer
        self._start_hibernation_monitoring()

    async def transition_to_hibernated(self, reason: str) -> None:
        """Transition agent to HIBERNATED state"""
        await self._validate_and_execute_transition(AgentState.HIBERNATED, reason)
        self.hibernated_at = datetime.now()
        
        # Stop hibernation monitoring
        if self._hibernation_check_task:
            self._hibernation_check_task.cancel()

    async def transition_to_archived(self, reason: str) -> None:
        """Transition agent to ARCHIVED state"""
        await self._validate_and_execute_transition(AgentState.ARCHIVED, reason)
        self.archived_at = datetime.now()
        
        # Stop all monitoring
        if self._hibernation_check_task:
            self._hibernation_check_task.cancel()

    async def _validate_and_execute_transition(self, target_state: AgentState, reason: str) -> None:
        """Validate and execute state transition with logging"""
        from_state = self.state
        
        # Validate transition rules (PRP Section 1.1.2)
        if not self._is_valid_transition(from_state, target_state):
            raise ValueError(
                f"Invalid state transition from {from_state.value} to {target_state.value}"
            )
        
        # For wake-up performance testing
        if from_state == AgentState.HIBERNATED and target_state == AgentState.IDLE:
            # Restore agent memory quickly (< 100ms requirement)
            await self._restore_from_hibernation()
        
        # Execute transition
        old_state = self.state
        self.state = target_state
        self.last_active = datetime.now()
        
        # Log transition
        transition_log = StateTransitionLog(
            id=str(uuid.uuid4()),
            agent_id=self.id,
            from_state=old_state.value,
            to_state=target_state.value,
            transition_timestamp=datetime.now(),
            trigger_reason=reason
        )
        self.state_transition_log.append(transition_log)
        
        logger.info(f"Agent {self.name} transitioned from {old_state.value} to {target_state.value}: {reason}")

    def _is_valid_transition(self, from_state: AgentState, to_state: AgentState) -> bool:
        """Validate state transition rules from PRP Section 1.1.2"""
        valid_transitions = {
            AgentState.CREATED: [AgentState.ACTIVE, AgentState.ARCHIVED],
            AgentState.ACTIVE: [AgentState.IDLE, AgentState.ARCHIVED],
            AgentState.IDLE: [AgentState.ACTIVE, AgentState.HIBERNATED, AgentState.ARCHIVED],
            AgentState.HIBERNATED: [AgentState.IDLE, AgentState.ARCHIVED],
            AgentState.ARCHIVED: []  # Terminal state
        }
        
        return to_state in valid_transitions.get(from_state, [])

    async def _restore_from_hibernation(self) -> None:
        """Restore agent from hibernation quickly (< 100ms requirement)"""
        start_time = datetime.now()
        
        # Quick memory validation and restoration
        if not self.memory:
            self.memory = {}
        
        # Validate knowledge items are intact
        for item in self.knowledge_items:
            if not item.id or not item.content:
                logger.warning(f"Knowledge item corruption detected for agent {self.name}")
        
        # Reset hibernation timestamp
        self.hibernated_at = None
        
        # Log restoration time for performance monitoring
        restore_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Agent {self.name} restored from hibernation in {restore_time:.2f}ms")

    def _start_hibernation_monitoring(self) -> None:
        """Start monitoring for auto-hibernation after 15 minutes idle"""
        if self._hibernation_check_task:
            self._hibernation_check_task.cancel()
            
        if self.state == AgentState.IDLE:
            self._hibernation_check_task = asyncio.create_task(self._hibernation_monitor())

    async def _hibernation_monitor(self) -> None:
        """Monitor for hibernation trigger after 15 minutes"""
        try:
            # Wait 15 minutes
            await asyncio.sleep(15 * 60)  # 15 minutes in seconds
            
            # Check if still idle
            if self.state == AgentState.IDLE:
                await self.transition_to_hibernated("15 minutes idle timeout")
        except asyncio.CancelledError:
            logger.debug(f"Hibernation monitoring cancelled for agent {self.name}")

    async def check_and_trigger_hibernation(self) -> bool:
        """Check if hibernation should be triggered based on last activity"""
        if self.state != AgentState.IDLE:
            return False
            
        # Check if 15 minutes have passed since last activity
        idle_duration = datetime.now() - self.last_active
        if idle_duration > timedelta(minutes=15):
            await self.transition_to_hibernated("15 minutes idle timeout")
            return True
            
        return False

    async def add_knowledge_item(self, knowledge_data: Dict[str, Any]) -> KnowledgeItem:
        """Add a knowledge item to the agent's learning base"""
        knowledge_item = KnowledgeItem(
            id=str(uuid.uuid4()),
            agent_id=self.id,
            project_id=self.project_id,
            item_type=knowledge_data.get("type", "pattern"),
            content=knowledge_data.get("content", {}),
            confidence=knowledge_data.get("confidence", 0.5),
            usage_count=knowledge_data.get("usage_count", 0),
            success_count=knowledge_data.get("success_count", 0),
            failure_count=knowledge_data.get("failure_count", 0),
            metadata=knowledge_data.get("metadata", {})
        )
        
        self.knowledge_items.append(knowledge_item)
        return knowledge_item

    async def get_knowledge_items(self) -> List[KnowledgeItem]:
        """Get all knowledge items for this agent"""
        return self.knowledge_items.copy()

    async def get_knowledge_item(self, item_id: str) -> Optional[KnowledgeItem]:
        """Get specific knowledge item by ID"""
        return next((item for item in self.knowledge_items if item.id == item_id), None)

    async def apply_knowledge_item(self, item_id: str, success: bool) -> None:
        """Apply knowledge item and update confidence based on success"""
        item = await self.get_knowledge_item(item_id)
        if not item:
            raise ValueError(f"Knowledge item {item_id} not found")
            
        # Update usage statistics
        item.usage_count += 1
        item.last_used = datetime.now()
        
        if success:
            item.success_count += 1
            # Increase confidence (PRP: success * 1.1, max 0.99)
            item.confidence = min(0.99, item.confidence * 1.1)
        else:
            item.failure_count += 1
            # Decrease confidence (PRP: failure * 0.9, min 0.1)
            item.confidence = max(0.1, item.confidence * 0.9)

    async def get_accessible_knowledge_items(self) -> List[KnowledgeItem]:
        """Get knowledge items accessible to this agent (project-scoped)"""
        # Only return knowledge items from the same project
        return [item for item in self.knowledge_items if item.project_id == self.project_id]

    async def get_state_transition_log(self) -> List[StateTransitionLog]:
        """Get state transition history for this agent"""
        return sorted(self.state_transition_log, key=lambda x: x.transition_timestamp, reverse=True)

    def __del__(self):
        """Cleanup hibernation monitoring on agent destruction"""
        if self._hibernation_check_task and not self._hibernation_check_task.done():
            self._hibernation_check_task.cancel()