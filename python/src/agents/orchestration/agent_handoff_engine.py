"""
Agent Handoff Engine - Core logic for intelligent agent handoffs.

This module provides the core functionality for intelligent agent handoffs,
including capability matching, context preservation, and learning from successful
handoff patterns.

Key Features:
- Intelligent agent selection based on capabilities and task requirements
- Context preservation during handoffs to maintain conversation continuity
- Multiple handoff strategies (collaborative, sequential, conditional)
- Performance monitoring and optimization of handoff decisions
- Learning from successful handoff patterns using DeepConf
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

from ...agents.base_agent import BaseAgent, ArchonDependencies
from ..orchestration.archon_send_message import MessagePriority, AgentMessage

# Type hint for ArchonAgency to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..orchestration.archon_agency import ArchonAgency

logger = logging.getLogger(__name__)


class HandoffStrategy(Enum):
    """Types of handoff strategies."""
    COLLABORATIVE = "collaborative"  # Agents work together on the task
    SEQUENTIAL = "sequential"        # Agents work in sequence
    CONDITIONAL = "conditional"      # Handoff based on conditions
    PARALLEL = "parallel"           # Agents work in parallel
    DELEGATION = "delegation"       # One agent delegates to another


class HandoffTrigger(Enum):
    """Events that trigger handoffs."""
    CAPABILITY_MATCH = "capability_match"
    TASK_COMPLEXITY = "task_complexity"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    EXPLICIT_REQUEST = "explicit_request"
    TIMEOUT = "timeout"
    ERROR_OCCURRED = "error_occurred"
    CONFIDENCE_LOW = "confidence_low"


class HandoffStatus(Enum):
    """Status of handoff operations."""
    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class AgentCapability:
    """Represents an agent's capabilities."""
    agent_name: str
    capabilities: Set[str]
    expertise_level: Dict[str, float]  # capability -> expertise level (0-1)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    last_active: datetime = field(default_factory=datetime.utcnow)
    availability: float = 1.0  # 0-1, where 1 is fully available

    def get_capability_score(self, required_capabilities: Set[str]) -> float:
        """Calculate capability match score."""
        if not required_capabilities:
            return 1.0

        matched = required_capabilities.intersection(self.capabilities)
        if not matched:
            return 0.0

        # Weight by expertise level
        total_score = sum(self.expertise_level.get(cap, 0.5) for cap in matched)
        max_score = len(required_capabilities)

        return min(total_score / max_score, 1.0)


@dataclass
class HandoffContext:
    """Context preserved during handoffs."""
    context_id: str
    original_message: str
    task_description: str
    sender_agent: str
    recipient_agent: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: Optional[ArchonDependencies] = None
    confidence_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "context_id": self.context_id,
            "original_message": self.original_message,
            "task_description": self.task_description,
            "sender_agent": self.sender_agent,
            "recipient_agent": self.recipient_agent,
            "conversation_history": self.conversation_history,
            "metadata": self.metadata,
            "confidence_score": self.confidence_score,
            "created_at": datetime.utcnow().isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HandoffContext":
        """Create context from dictionary."""
        return cls(
            context_id=data["context_id"],
            original_message=data["original_message"],
            task_description=data["task_description"],
            sender_agent=data["sender_agent"],
            recipient_agent=data["recipient_agent"],
            conversation_history=data.get("conversation_history", []),
            metadata=data.get("metadata", {}),
            confidence_score=data.get("confidence_score")
        )


@dataclass
class HandoffDecision:
    """Represents a handoff decision."""
    decision_id: str
    source_agent: str
    target_agent: str
    trigger: HandoffTrigger
    strategy: HandoffStrategy
    confidence_score: float
    reasoning: str
    context: HandoffContext
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary."""
        return {
            "decision_id": self.decision_id,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "trigger": self.trigger.value,
            "strategy": self.strategy.value,
            "confidence_score": self.confidence_score,
            "reasoning": self.reasoning,
            "context": self.context.to_dict(),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class HandoffResult:
    """Result of a handoff operation."""
    handoff_id: str
    decision: HandoffDecision
    status: HandoffStatus
    response_content: Optional[str] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "handoff_id": self.handoff_id,
            "decision": self.decision.to_dict(),
            "status": self.status.value,
            "response_content": self.response_content,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "metrics": self.metrics,
            "completed_at": datetime.utcnow().isoformat()
        }


class HandoffStrategyBase(ABC):
    """Base class for handoff strategies."""

    @abstractmethod
    async def execute(
        self,
        decision: HandoffDecision,
        agency: 'ArchonAgency'
    ) -> HandoffResult:
        """Execute the handoff strategy."""
        pass


class SequentialHandoffStrategy(HandoffStrategyBase):
    """Sequential handoff where agents work one after another."""

    async def execute(
        self,
        decision: HandoffDecision,
        agency: 'ArchonAgency'
    ) -> HandoffResult:
        """Execute sequential handoff."""
        start_time = datetime.utcnow()
        handoff_id = str(uuid.uuid4())

        try:
            logger.info(f"Executing sequential handoff from {decision.source_agent} to {decision.target_agent}")

            # Get target agent
            target_agent = agency.agents[decision.target_agent]

            # Create handoff message
            handoff_message = f"""
            [HANDOFF] Task transferred from {decision.source_agent}

            Original task: {decision.context.task_description}
            Original message: {decision.context.original_message}

            Context: {json.dumps(decision.context.metadata, indent=2)}

            Please continue the task with this context.
            """

            # Create dependencies with handoff context
            deps = ArchonDependencies(
                request_id=str(uuid.uuid4()),
                user_id=decision.context.dependencies.user_id if decision.context.dependencies else None,
                trace_id=str(uuid.uuid4()),
                context={
                    **decision.context.to_dict(),
                    "handoff_decision": decision.to_dict()
                }
            )

            # Execute target agent
            if hasattr(target_agent, 'run_with_confidence'):
                response, confidence = await target_agent.run_with_confidence(
                    handoff_message,
                    deps,
                    task_description=f"Handoff task from {decision.source_agent}: {decision.context.task_description}"
                )
            else:
                response = await target_agent.run(handoff_message, deps)
                confidence = None

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            return HandoffResult(
                handoff_id=handoff_id,
                decision=decision,
                status=HandoffStatus.COMPLETED,
                response_content=str(response) if response else "",
                execution_time=execution_time,
                metrics={
                    "confidence_score": confidence,
                    "strategy": "sequential"
                }
            )

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Sequential handoff failed: {e}")

            return HandoffResult(
                handoff_id=handoff_id,
                decision=decision,
                status=HandoffStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time,
                metrics={"strategy": "sequential"}
            )


class CollaborativeHandoffStrategy(HandoffStrategyBase):
    """Collaborative handoff where agents work together."""

    async def execute(
        self,
        decision: HandoffDecision,
        agency: 'ArchonAgency'
    ) -> HandoffResult:
        """Execute collaborative handoff."""
        start_time = datetime.utcnow()
        handoff_id = str(uuid.uuid4())

        try:
            logger.info(f"Executing collaborative handoff between {decision.source_agent} and {decision.target_agent}")

            # Get both agents
            source_agent = agency.agents[decision.source_agent]
            target_agent = agency.agents[decision.target_agent]

            # Create collaborative message for target agent
            collaborative_message = f"""
            [COLLABORATIVE HANDOFF] Working together with {decision.source_agent}

            Task: {decision.context.task_description}
            Original request: {decision.context.original_message}

            Context: {json.dumps(decision.context.metadata, indent=2)}

            Please provide your expertise on this task. The original agent will continue to coordinate.
            """

            # Execute target agent collaboratively
            target_deps = ArchonDependencies(
                request_id=str(uuid.uuid4()),
                user_id=decision.context.dependencies.user_id if decision.context.dependencies else None,
                trace_id=str(uuid.uuid4()),
                context={
                    **decision.context.to_dict(),
                    "collaborative_mode": True,
                    "coordinating_agent": decision.source_agent
                }
            )

            if hasattr(target_agent, 'run_with_confidence'):
                target_response, target_confidence = await target_agent.run_with_confidence(
                    collaborative_message,
                    target_deps,
                    task_description=f"Collaborative task with {decision.source_agent}: {decision.context.task_description}"
                )
            else:
                target_response = await target_agent.run(collaborative_message, target_deps)
                target_confidence = None

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            return HandoffResult(
                handoff_id=handoff_id,
                decision=decision,
                status=HandoffStatus.COMPLETED,
                response_content=str(target_response) if target_response else "",
                execution_time=execution_time,
                metrics={
                    "target_confidence": target_confidence,
                    "strategy": "collaborative"
                }
            )

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Collaborative handoff failed: {e}")

            return HandoffResult(
                handoff_id=handoff_id,
                decision=decision,
                status=HandoffStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time,
                metrics={"strategy": "collaborative"}
            )


class AgentHandoffEngine:
    """
    Core engine for intelligent agent handoffs.

    This engine manages the entire handoff lifecycle:
    - Capability analysis and agent selection
    - Context preservation and transfer
    - Strategy execution
    - Performance learning and optimization
    """

    def __init__(self, agency: 'ArchonAgency'):
        """Initialize the handoff engine."""
        self.agency = agency
        self.agent_capabilities: Dict[str, AgentCapability] = {}
        self.handoff_history: List[HandoffResult] = []
        self.strategies: Dict[HandoffStrategy, HandoffStrategyBase] = {
            HandoffStrategy.SEQUENTIAL: SequentialHandoffStrategy(),
            HandoffStrategy.COLLABORATIVE: CollaborativeHandoffStrategy()
        }

        # Initialize agent capabilities
        self._initialize_agent_capabilities()

        logger.info("Agent Handoff Engine initialized")

    def _initialize_agent_capabilities(self) -> None:
        """Initialize capabilities for all agents in the agency."""
        for agent_name, agent in self.agency.agents.items():
            capability = AgentCapability(
                agent_name=agent_name,
                capabilities=self._extract_agent_capabilities(agent),
                expertise_level=self._extract_expertise_levels(agent)
            )
            self.agent_capabilities[agent_name] = capability
            logger.debug(f"Initialized capabilities for {agent_name}: {capability.capabilities}")

    def _extract_agent_capabilities(self, agent: BaseAgent) -> Set[str]:
        """Extract capabilities from agent metadata."""
        capabilities = set()

        # Extract from agent class name
        class_name = agent.__class__.__name__.lower()

        # Basic capability mapping
        if "code" in class_name or "develop" in class_name:
            capabilities.update(["coding", "development", "programming"])
        if "test" in class_name:
            capabilities.update(["testing", "quality_assurance"])
        if "design" in class_name:
            capabilities.update(["design", "ui_ux", "architecture"])
        if "analyst" in class_name or "analysis" in class_name:
            capabilities.update(["analysis", "research", "data_processing"])
        if "manager" in class_name or "coordinator" in class_name:
            capabilities.update(["coordination", "management", "planning"])
        if "security" in class_name:
            capabilities.update(["security", "compliance", "risk_assessment"])
        if "performance" in class_name:
            capabilities.update(["optimization", "performance", "monitoring"])

        # Add generic capabilities
        capabilities.update(["communication", "problem_solving"])

        return capabilities

    def _extract_expertise_levels(self, agent: BaseAgent) -> Dict[str, float]:
        """Extract expertise levels for agent capabilities."""
        # Default expertise levels - could be enhanced with learning
        return {cap: 0.7 for cap in self._extract_agent_capabilities(agent)}

    async def analyze_handoff_needs(
        self,
        message: str,
        task_description: str,
        current_agent: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[HandoffDecision]:
        """
        Analyze if a handoff is needed and make a decision.

        Args:
            message: The original message
            task_description: Description of the task
            current_agent: Name of the current agent
            context: Additional context

        Returns:
            HandoffDecision if handoff is needed, None otherwise
        """
        try:
            # Extract required capabilities from task
            required_capabilities = self._extract_task_capabilities(task_description, message)

            # Find best agent for the task
            best_agent, match_score = self._find_best_agent(required_capabilities, current_agent)

            # Determine if handoff is beneficial
            if best_agent and best_agent != current_agent and match_score > 0.7:
                # Create handoff decision
                decision = HandoffDecision(
                    decision_id=str(uuid.uuid4()),
                    source_agent=current_agent,
                    target_agent=best_agent,
                    trigger=HandoffTrigger.CAPABILITY_MATCH,
                    strategy=self._select_strategy(required_capabilities),
                    confidence_score=match_score,
                    reasoning=f"Better capability match: {match_score:.2f}",
                    context=HandoffContext(
                        context_id=str(uuid.uuid4()),
                        original_message=message,
                        task_description=task_description,
                        sender_agent=current_agent,
                        recipient_agent=best_agent,
                        metadata=context or {}
                    )
                )

                logger.info(f"Handoff decision made: {current_agent} -> {best_agent} (score: {match_score:.2f})")
                return decision

            return None

        except Exception as e:
            logger.error(f"Error analyzing handoff needs: {e}")
            return None

    def _extract_task_capabilities(self, task_description: str, message: str) -> Set[str]:
        """Extract required capabilities from task description."""
        capabilities = set()
        full_text = f"{task_description} {message}".lower()

        # Capability keywords
        capability_keywords = {
            "coding": ["code", "program", "develop", "implement", "function", "class"],
            "testing": ["test", "unit", "integration", "assert", "mock"],
            "design": ["design", "architecture", "ui", "ux", "layout"],
            "analysis": ["analyze", "research", "data", "metrics", "statistics"],
            "security": ["security", "authentication", "authorization", "encrypt"],
            "performance": ["performance", "optimize", "speed", "efficiency"],
            "documentation": ["document", "readme", "guide", "manual"],
            "deployment": ["deploy", "release", "publish", "distribute"]
        }

        for capability, keywords in capability_keywords.items():
            if any(keyword in full_text for keyword in keywords):
                capabilities.add(capability)

        return capabilities

    def _find_best_agent(self, required_capabilities: Set[str], current_agent: str) -> Tuple[Optional[str], float]:
        """Find the best agent for given capabilities."""
        best_agent = None
        best_score = 0.0

        for agent_name, capability in self.agent_capabilities.items():
            if agent_name == current_agent:
                continue  # Skip current agent

            score = capability.get_capability_score(required_capabilities)
            if score > best_score:
                best_score = score
                best_agent = agent_name

        return best_agent, best_score

    def _select_strategy(self, required_capabilities: Set[str]) -> HandoffStrategy:
        """Select appropriate handoff strategy based on capabilities."""
        # Simple strategy selection - could be enhanced with learning
        if len(required_capabilities) > 3:
            return HandoffStrategy.COLLABORATIVE
        else:
            return HandoffStrategy.SEQUENTIAL

    async def execute_handoff(self, decision: HandoffDecision) -> HandoffResult:
        """Execute a handoff decision."""
        strategy = self.strategies.get(decision.strategy)
        if not strategy:
            raise ValueError(f"Unknown handoff strategy: {decision.strategy}")

        # Update status
        decision.status = HandoffStatus.IN_PROGRESS

        # Execute strategy
        result = await strategy.execute(decision, self.agency)

        # Record handoff
        self.handoff_history.append(result)

        # Update agent performance metrics
        self._update_performance_metrics(result)

        logger.info(f"Handoff completed: {result.status.value} in {result.execution_time:.2f}s")

        return result

    def _update_performance_metrics(self, result: HandoffResult) -> None:
        """Update performance metrics based on handoff result."""
        if result.decision.target_agent not in self.agent_capabilities:
            return

        capability = self.agent_capabilities[result.decision.target_agent]

        # Update metrics based on result
        if result.status == HandoffStatus.COMPLETED:
            capability.performance_metrics["successful_handoffs"] = capability.performance_metrics.get("successful_handoffs", 0) + 1
            capability.performance_metrics["total_handoff_time"] = capability.performance_metrics.get("total_handoff_time", 0.0) + result.execution_time
        else:
            capability.performance_metrics["failed_handoffs"] = capability.performance_metrics.get("failed_handoffs", 0) + 1

        capability.last_active = datetime.utcnow()

    async def request_handoff(
        self,
        source_agent: str,
        target_agent: str,
        message: str,
        task_description: str,
        strategy: HandoffStrategy = HandoffStrategy.SEQUENTIAL,
        context: Optional[Dict[str, Any]] = None
    ) -> HandoffResult:
        """
        Request a manual handoff between agents.

        Args:
            source_agent: Agent requesting the handoff
            target_agent: Agent to handoff to
            message: Original message
            task_description: Task description
            strategy: Handoff strategy to use
            context: Additional context

        Returns:
            HandoffResult
        """
        decision = HandoffDecision(
            decision_id=str(uuid.uuid4()),
            source_agent=source_agent,
            target_agent=target_agent,
            trigger=HandoffTrigger.EXPLICIT_REQUEST,
            strategy=strategy,
            confidence_score=1.0,  # Manual request = high confidence
            reasoning="Explicit handoff request",
            context=HandoffContext(
                context_id=str(uuid.uuid4()),
                original_message=message,
                task_description=task_description,
                sender_agent=source_agent,
                recipient_agent=target_agent,
                metadata=context or {}
            )
        )

        return await self.execute_handoff(decision)

    def get_handoff_statistics(self) -> Dict[str, Any]:
        """Get handoff performance statistics."""
        total_handoffs = len(self.handoff_history)
        successful_handoffs = len([h for h in self.handoff_history if h.status == HandoffStatus.COMPLETED])

        strategy_stats = {}
        for strategy in HandoffStrategy:
            strategy_handoffs = [h for h in self.handoff_history if h.decision.strategy == strategy]
            strategy_stats[strategy.value] = {
                "total": len(strategy_handoffs),
                "success_rate": len([h for h in strategy_handoffs if h.status == HandoffStatus.COMPLETED]) / len(strategy_handoffs) if strategy_handoffs else 0,
                "avg_execution_time": sum(h.execution_time for h in strategy_handoffs) / len(strategy_handoffs) if strategy_handoffs else 0
            }

        return {
            "total_handoffs": total_handoffs,
            "success_rate": successful_handoffs / total_handoffs if total_handoffs > 0 else 0,
            "strategy_statistics": strategy_stats,
            "agents_participating": len(set(h.decision.target_agent for h in self.handoff_history))
        }

    def get_agent_handoff_history(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get handoff history for a specific agent."""
        history = []

        for result in self.handoff_history:
            if result.decision.source_agent == agent_name or result.decision.target_agent == agent_name:
                history.append(result.to_dict())

        return history

    async def learn_from_handoffs(self) -> None:
        """Learn from handoff history to improve future decisions."""
        # This is a placeholder for learning functionality
        # In a full implementation, this would use machine learning
        # to improve capability matching and strategy selection

        for result in self.handoff_history[-10:]:  # Learn from recent handoffs
            if result.status == HandoffStatus.COMPLETED:
                # Update capability expertise based on successful handoffs
                target_agent = result.decision.target_agent
                if target_agent in self.agent_capabilities:
                    capability = self.agent_capabilities[target_agent]

                    # Slightly increase expertise for successful handoffs
                    for cap in capability.capabilities:
                        capability.expertise_level[cap] = min(
                            capability.expertise_level.get(cap, 0.5) + 0.01,
                            1.0
                        )

        logger.info("Learning from handoff history completed")