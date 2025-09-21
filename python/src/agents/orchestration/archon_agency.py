"""
Archon Agency - Enhanced multi-agent orchestration system.

This module integrates Agency Swarm's dynamic communication patterns with Archon's
enterprise-grade agent ecosystem, enabling dynamic agent collaboration while maintaining
backward compatibility with existing agents.

Based on Agency Swarm architecture but adapted for Archon's BaseAgent system.
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional, Union, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ...agents.base_agent import BaseAgent, ArchonDependencies
# Note: SendMessagesAgent not used in this implementation
from .archon_thread_manager import ArchonThreadManager, ThreadContext
from .archon_send_message import ArchonSendMessageTool
from .agent_handoff_engine import AgentHandoffEngine, HandoffDecision, HandoffResult
from .context_preservation import ContextPreservationEngine
from .handoff_learning import HandoffLearningEngine

logger = logging.getLogger(__name__)


class CommunicationFlowType(Enum):
    """Types of communication flows between agents."""
    DIRECT = "direct"  # agent1 -> agent2
    CHAIN = "chain"    # agent1 -> agent2 -> agent3
    BROADCAST = "broadcast"  # agent -> multiple agents


@dataclass
class CommunicationFlow:
    """Represents a communication flow between agents."""
    sender: BaseAgent
    receivers: List[BaseAgent]
    flow_type: CommunicationFlowType
    custom_tool_class: Optional[type] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if not self.receivers:
            raise ValueError("Communication flow must have at least one receiver")


@dataclass
class AgencyConfig:
    """Configuration for the Archon Agency."""
    name: Optional[str] = None
    shared_instructions: Optional[str] = None
    enable_persistence: bool = True
    enable_streaming: bool = True
    default_timeout: float = 120.0
    max_message_retries: int = 3
    enable_kafka_integration: bool = True
    enable_handoffs: bool = True
    enable_handoff_learning: bool = True
    handoff_confidence_threshold: float = 0.7
    max_context_size: int = 1024 * 1024  # 1MB
    auto_handoff_enabled: bool = True


class ArchonAgency:
    """
    Enhanced multi-agent orchestration system for Archon.

    Integrates Agency Swarm's dynamic communication patterns with Archon's
    enterprise-grade features while maintaining backward compatibility.

    Key Features:
    - Dynamic agent communication with `>` operator syntax
    - Thread isolation and conversation persistence
    - Integration with existing BaseAgent system
    - Support for both synchronous and streaming operations
    - Enterprise-grade error handling and monitoring
    """

    def __init__(
        self,
        *entry_point_agents: BaseAgent,
        communication_flows: Optional[List[Union[CommunicationFlow, Tuple[BaseAgent, BaseAgent], Tuple[BaseAgent, List[BaseAgent]]]]] = None,
        config: Optional[AgencyConfig] = None,
        **kwargs
    ):
        """
        Initialize the Archon Agency.

        Args:
            *entry_point_agents: Agents that can receive external messages
            communication_flows: Communication patterns between agents
            config: Agency configuration
            **kwargs: Additional configuration (deprecated, use config instead)
        """
        self.agents: Dict[str, BaseAgent] = {}
        self.entry_points: List[BaseAgent] = []
        self.communication_flows: List[CommunicationFlow] = []
        self.config = config or AgencyConfig()

        # Parse and register agents
        self._register_agents(entry_point_agents or [])

        # Parse communication flows
        if communication_flows:
            self._parse_communication_flows(communication_flows)

        # Initialize core components
        self.thread_manager = ArchonThreadManager(
            enable_persistence=self.config.enable_persistence
        )

        # Initialize send message tool
        self.send_message_tool = ArchonSendMessageTool(self)

        # Initialize handoff components
        self.handoff_engine = AgentHandoffEngine(self)
        self.context_engine = ContextPreservationEngine(max_context_size=self.config.max_context_size)
        self.learning_engine = HandoffLearningEngine()

        # Setup communication for agents
        self._setup_agent_communication()

        # Start periodic learning if enabled
        if self.config.enable_handoff_learning:
            asyncio.create_task(self._periodic_learning_cycle())

        logger.info(f"🚀 Archon Agency '{self.config.name or 'Unnamed'}' initialized")
        logger.info(f"   Entry points: {[agent.name for agent in self.entry_points]}")
        logger.info(f"   Total agents: {len(self.agents)}")
        logger.info(f"   Communication flows: {len(self.communication_flows)}")
        logger.info(f"   Handoffs enabled: {self.config.enable_handoffs}")
        logger.info(f"   Learning enabled: {self.config.enable_handoff_learning}")

    def _register_agents(self, agents: List[BaseAgent]) -> None:
        """Register agents with the agency."""
        for agent in agents:
            if agent.name in self.agents:
                logger.warning(f"Agent '{agent.name}' already registered, skipping duplicate")
                continue

            self.agents[agent.name] = agent
            self.entry_points.append(agent)

            logger.debug(f"Registered agent: {agent.name}")

    def _parse_communication_flows(
        self,
        flows: List[Union[CommunicationFlow, Tuple[BaseAgent, BaseAgent], Tuple[BaseAgent, List[BaseAgent]]]]
    ) -> None:
        """Parse various communication flow formats into CommunicationFlow objects."""
        for flow in flows:
            if isinstance(flow, CommunicationFlow):
                # Already parsed
                self.communication_flows.append(flow)
                self._ensure_agents_registered([flow.sender] + flow.receivers)

            elif isinstance(flow, tuple) and len(flow) == 2:
                sender, receiver_or_receivers = flow

                if isinstance(receiver_or_receivers, list):
                    # (Agent, [Agent1, Agent2, ...]) - broadcast
                    receivers = receiver_or_receivers
                    flow_type = CommunicationFlowType.BROADCAST
                else:
                    # (Agent, Agent) - direct
                    receivers = [receiver_or_receivers]
                    flow_type = CommunicationFlowType.DIRECT

                comm_flow = CommunicationFlow(
                    sender=sender,
                    receivers=receivers,
                    flow_type=flow_type
                )
                self.communication_flows.append(comm_flow)
                self._ensure_agents_registered([sender] + receivers)

            else:
                raise ValueError(f"Invalid communication flow format: {flow}")

    def _ensure_agents_registered(self, agents: List[BaseAgent]) -> None:
        """Ensure all agents in flows are registered with the agency."""
        for agent in agents:
            if agent.name not in self.agents:
                self.agents[agent.name] = agent
                logger.debug(f"Auto-registered agent from communication flow: {agent.name}")

    def _setup_agent_communication(self) -> None:
        """Setup communication tools for all agents."""
        for agent in self.agents.values():
            # Add send message tool to each agent
            try:
                agent.add_tool(self.send_message_tool.send_to_agent)
                logger.debug(f"Added send_message tool to agent: {agent.name}")
            except Exception as e:
                logger.warning(f"Failed to add send_message tool to {agent.name}: {e}")

    def __gt__(self, other: BaseAgent) -> "ArchonAgencyChainBuilder":
        """
        Enable the `>` operator syntax for creating communication flows.

        Example:
            agency = ceo_agent > developer_agent > tester_agent
        """
        return ArchonAgencyChainBuilder(self, [other])

    def add_communication_flow(
        self,
        sender: BaseAgent,
        receivers: Union[BaseAgent, List[BaseAgent]],
        flow_type: CommunicationFlowType = CommunicationFlowType.DIRECT,
        custom_tool_class: Optional[type] = None
    ) -> None:
        """
        Add a communication flow to the agency.

        Args:
            sender: The sending agent
            receivers: One or more receiving agents
            flow_type: Type of communication flow
            custom_tool_class: Optional custom tool class for this flow
        """
        if not isinstance(receivers, list):
            receivers = [receivers]

        flow = CommunicationFlow(
            sender=sender,
            receivers=receivers,
            flow_type=flow_type,
            custom_tool_class=custom_tool_class
        )

        self.communication_flows.append(flow)
        self._ensure_agents_registered([sender] + receivers)

        logger.info(f"Added communication flow: {sender.name} -> {[r.name for r in receivers]}")

    async def get_response(
        self,
        message: str,
        recipient_agent: Optional[Union[str, BaseAgent]] = None,
        context_override: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Get a response from the agency.

        Args:
            message: The input message
            recipient_agent: Target agent (uses first entry point if None)
            context_override: Additional context for the run
            **kwargs: Additional arguments passed to agent

        Returns:
            The agent's response
        """
        # Resolve recipient agent
        if recipient_agent is None:
            if not self.entry_points:
                raise ValueError("No entry point agents available")
            recipient_agent = self.entry_points[0]

        if isinstance(recipient_agent, str):
            if recipient_agent not in self.agents:
                raise ValueError(f"Agent '{recipient_agent}' not found in agency")
            recipient_agent = self.agents[recipient_agent]

        # Create dependencies
        deps = ArchonDependencies(
            request_id=str(uuid.uuid4()),
            user_id=context_override.get("user_id") if context_override else None,
            trace_id=str(uuid.uuid4()),
            context=context_override or {}
        )

        # Create or get thread context
        thread_id = kwargs.get("thread_id")
        if thread_id:
            thread_context = await self.thread_manager.get_thread(thread_id)
        else:
            thread_context = await self.thread_manager.create_thread(
                sender="external",
                recipient=recipient_agent.name
            )

        # Add thread context to dependencies
        deps.context.update({
            "thread_id": thread_context.thread_id,
            "agency_context": {
                "agency_name": self.config.name,
                "sender_agent": "external",
                "conversation_history": thread_context.get_recent_messages(10)
            }
        })

        try:
            # Execute the agent
            if hasattr(recipient_agent, 'run_with_confidence'):
                result, confidence = await recipient_agent.run_with_confidence(
                    message,
                    deps,
                    task_description=f"Agency message: {message[:100]}..."
                )
                logger.info(f"Agent {recipient_agent.name} completed with confidence: {confidence}")
            else:
                result = await recipient_agent.run(message, deps)
                logger.info(f"Agent {recipient_agent.name} completed")

            # Store message in thread
            await thread_context.add_message({
                "role": "user",
                "content": message,
                "sender": "external",
                "recipient": recipient_agent.name,
                "timestamp": datetime.utcnow().isoformat()
            })

            await thread_context.add_message({
                "role": "assistant",
                "content": str(result),
                "sender": recipient_agent.name,
                "recipient": "external",
                "timestamp": datetime.utcnow().isoformat()
            })

            return result

        except Exception as e:
            logger.error(f"Error getting response from {recipient_agent.name}: {e}")
            raise

    async def get_response_stream(
        self,
        message: str,
        recipient_agent: Optional[Union[str, BaseAgent]] = None,
        context_override: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncGenerator[Any, None]:
        """
        Get a streaming response from the agency.

        Args:
            message: The input message
            recipient_agent: Target agent (uses first entry point if None)
            context_override: Additional context for the run
            **kwargs: Additional arguments passed to agent

        Yields:
            Streaming response chunks
        """
        if not self.config.enable_streaming:
            logger.warning("Streaming not enabled, falling back to non-streaming response")
            result = await self.get_response(message, recipient_agent, context_override, **kwargs)
            yield {"type": "content", "content": result}
            return

        # Resolve recipient agent
        if recipient_agent is None:
            if not self.entry_points:
                raise ValueError("No entry point agents available")
            recipient_agent = self.entry_points[0]

        if isinstance(recipient_agent, str):
            if recipient_agent not in self.agents:
                raise ValueError(f"Agent '{recipient_agent}' not found in agency")
            recipient_agent = self.agents[recipient_agent]

        # Create dependencies
        deps = ArchonDependencies(
            request_id=str(uuid.uuid4()),
            user_id=context_override.get("user_id") if context_override else None,
            trace_id=str(uuid.uuid4()),
            context=context_override or {}
        )

        try:
            # Stream the response
            async for chunk in recipient_agent.run_stream(message, deps):
                yield chunk

        except Exception as e:
            logger.error(f"Error in streaming response from {recipient_agent.name}: {e}")
            yield {"type": "error", "error": str(e)}

    def get_agent(self, agent_name: str) -> BaseAgent:
        """Get an agent by name."""
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found in agency")
        return self.agents[agent_name]

    def list_agents(self) -> List[str]:
        """List all agent names in the agency."""
        return list(self.agents.keys())

    def get_communication_flows(self) -> List[Dict[str, Any]]:
        """Get all communication flows as a list of dictionaries."""
        return [
            {
                "id": flow.id,
                "sender": flow.sender.name,
                "receivers": [r.name for r in flow.receivers],
                "flow_type": flow.flow_type.value,
                "custom_tool_class": flow.custom_tool_class.__name__ if flow.custom_tool_class else None
            }
            for flow in self.communication_flows
        ]

    def get_agency_structure(self) -> Dict[str, Any]:
        """
        Get a ReactFlow-compatible structure of the agency.

        Returns:
            Dictionary with nodes and edges for visualization
        """
        nodes = []
        edges = []

        # Add agent nodes
        for i, agent in enumerate(self.agents.values()):
            is_entry_point = agent in self.entry_points
            nodes.append({
                "id": agent.name,
                "type": "agent",
                "position": {"x": i * 200, "y": 100},
                "data": {
                    "name": agent.name,
                    "type": agent.__class__.__name__,
                    "is_entry_point": is_entry_point,
                    "model": getattr(agent, 'model', 'unknown')
                }
            })

        # Add communication flow edges
        for flow in self.communication_flows:
            for receiver in flow.receivers:
                edges.append({
                    "id": f"{flow.sender.name}-{receiver.name}",
                    "source": flow.sender.name,
                    "target": receiver.name,
                    "type": "communication",
                    "data": {
                        "flow_type": flow.flow_type.value,
                        "flow_id": flow.id
                    }
                })

        return {
            "nodes": nodes,
            "edges": edges
        }

    async def create_conversation_thread(
        self,
        sender: str,
        recipient: str,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new conversation thread between agents.

        Args:
            sender: Sender agent name
            recipient: Recipient agent name
            initial_context: Initial context for the thread

        Returns:
            Thread ID
        """
        return await self.thread_manager.create_thread(
            sender=sender,
            recipient=recipient,
            initial_context=initial_context
        )

    async def get_conversation_history(self, thread_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get conversation history for a thread.

        Args:
            thread_id: Thread identifier
            limit: Maximum number of messages to retrieve

        Returns:
            List of messages
        """
        thread_context = await self.thread_manager.get_thread(thread_id)
        return thread_context.get_recent_messages(limit)

    async def send_agent_message(
        self,
        sender: str,
        recipient: str,
        message: str,
        thread_id: Optional[str] = None
    ) -> Any:
        """
        Send a message from one agent to another.

        Args:
            sender: Sender agent name
            recipient: Recipient agent name
            message: Message content
            thread_id: Optional thread ID for conversation continuity

        Returns:
            Response from recipient agent
        """
        if sender not in self.agents:
            raise ValueError(f"Sender agent '{sender}' not found")
        if recipient not in self.agents:
            raise ValueError(f"Recipient agent '{recipient}' not found")

        # Use the send message tool
        return await self.send_message_tool.send_to_agent(
            recipient_agent=recipient,
            message=message,
            sender_agent=sender,
            thread_id=thread_id
        )

    def visualize(self, output_file: str = "archon_agency_visualization.html") -> str:
        """
        Create a visual representation of the agency structure.

        Args:
            output_file: Path to save the HTML file

        Returns:
            Path to the generated file
        """
        # Import here to avoid circular dependencies
        from .archon_visualization import create_agency_visualization

        return create_agency_visualization(self, output_file)

    # Handoff Management Methods

    async def analyze_and_execute_handoff(
        self,
        message: str,
        task_description: str,
        current_agent: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Union[HandoffResult, Any]:
        """
        Analyze if a handoff is needed and execute it if beneficial.

        Args:
            message: The original message
            task_description: Description of the task
            current_agent: Name of the current agent
            context: Additional context

        Returns:
            HandoffResult if handoff occurred, original response otherwise
        """
        if not self.config.enable_handoffs or not self.config.auto_handoff_enabled:
            return None

        try:
            # Analyze handoff needs
            handoff_decision = await self.handoff_engine.analyze_handoff_needs(
                message, task_description, current_agent, context
            )

            if handoff_decision:
                logger.info(f"Auto-handoff triggered: {current_agent} -> {handoff_decision.target_agent}")
                return await self.execute_handoff(handoff_decision)
            else:
                logger.debug("No handoff needed")
                return None

        except Exception as e:
            logger.error(f"Error in auto-handoff analysis: {e}")
            return None

    async def execute_handoff(self, decision: HandoffDecision) -> HandoffResult:
        """
        Execute a handoff decision.

        Args:
            decision: The handoff decision to execute

        Returns:
            HandoffResult
        """
        try:
            logger.info(f"Executing handoff: {decision.source_agent} -> {decision.target_agent}")

            # Extract and preserve context
            handoff_context = await self.context_engine.extract_context(
                decision.context.sender_agent,
                decision.context.recipient_agent,
                decision.context.original_message,
                decision.context.task_description,
                decision.context.conversation_history,
                decision.context.dependencies,
                decision.context.metadata
            )

            # Update decision with enhanced context
            decision.context = handoff_context

            # Execute the handoff
            result = await self.handoff_engine.execute_handoff(decision)

            # Record for learning
            if self.config.enable_handoff_learning:
                await self.learning_engine.record_handoff_result(result)

            # Log handoff result
            if result.status.value == "completed":
                logger.info(f"Handoff completed successfully in {result.execution_time:.2f}s")
            else:
                logger.warning(f"Handoff failed: {result.error_message}")

            return result

        except Exception as e:
            logger.error(f"Error executing handoff: {e}")
            raise

    async def request_handoff(
        self,
        source_agent: str,
        target_agent: str,
        message: str,
        task_description: str,
        strategy: str = "sequential",
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
        from .agent_handoff_engine import HandoffStrategy

        strategy_enum = HandoffStrategy(strategy.lower())

        result = await self.handoff_engine.request_handoff(
            source_agent=source_agent,
            target_agent=target_agent,
            message=message,
            task_description=task_description,
            strategy=strategy_enum,
            context=context
        )

        # Record for learning
        if self.config.enable_handoff_learning:
            await self.learning_engine.record_handoff_result(result)

        return result

    async def get_handoff_recommendations(
        self,
        message: str,
        task_description: str,
        current_agent: str
    ) -> Dict[str, Any]:
        """
        Get handoff recommendations for a task.

        Args:
            message: The original message
            task_description: Description of the task
            current_agent: Name of the current agent

        Returns:
            Dictionary with recommendations
        """
        try:
            # Extract required capabilities
            required_capabilities = self.handoff_engine._extract_task_capabilities(
                task_description, message
            )

            # Find best agents
            agent_recommendations = []
            for agent_name, capability in self.handoff_engine.agent_capabilities.items():
                if agent_name != current_agent:
                    score = capability.get_capability_score(required_capabilities)
                    if score > 0.5:  # Only recommend good matches
                        agent_recommendations.append({
                            "agent": agent_name,
                            "score": score,
                            "capabilities": list(capability.capabilities),
                            "expertise": capability.expertise_level
                        })

            # Sort by score
            agent_recommendations.sort(key=lambda x: x["score"], reverse=True)

            # Get strategy recommendation
            strategy_rec = None
            if self.config.enable_handoff_learning:
                strategy_rec = await self.learning_engine.get_strategy_recommendation(
                    task_description, 0.7, current_agent
                )

            return {
                "recommended": len(agent_recommendations) > 0,
                "agents": agent_recommendations[:3],  # Top 3 recommendations
                "strategy_recommendation": strategy_rec,
                "required_capabilities": list(required_capabilities),
                "current_agent": current_agent
            }

        except Exception as e:
            logger.error(f"Error getting handoff recommendations: {e}")
            return {"error": str(e)}

    async def get_handoff_statistics(self) -> Dict[str, Any]:
        """Get comprehensive handoff statistics."""
        try:
            handoff_stats = self.handoff_engine.get_handoff_statistics()
            context_stats = self.context_engine.get_context_statistics()
            learning_insights = self.learning_engine.get_learning_insights()

            return {
                "handoff_engine": handoff_stats,
                "context_preservation": context_stats,
                "learning": learning_insights,
                "agency_config": {
                    "handoffs_enabled": self.config.enable_handoffs,
                    "learning_enabled": self.config.enable_handoff_learning,
                    "auto_handoff_enabled": self.config.auto_handoff_enabled,
                    "confidence_threshold": self.config.handoff_confidence_threshold
                }
            }

        except Exception as e:
            logger.error(f"Error getting handoff statistics: {e}")
            return {"error": str(e)}

    async def get_agent_handoff_history(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get handoff history for a specific agent."""
        return self.handoff_engine.get_agent_handoff_history(agent_name)

    async def run_learning_cycle(self) -> None:
        """Manually trigger a learning cycle."""
        if self.config.enable_handoff_learning:
            await self.learning_engine.run_learning_cycle()
        else:
            logger.warning("Handoff learning is disabled")

    async def _periodic_learning_cycle(self) -> None:
        """Periodic learning cycle running in the background."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.run_learning_cycle()
            except Exception as e:
                logger.error(f"Error in periodic learning cycle: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry


class ArchonAgencyChainBuilder:
    """
    Helper class for building communication chains using the `>` operator.

    Example:
        agency = ceo_agent > developer_agent > tester_agent
    """

    def __init__(self, agency: ArchonAgency, agents: List[BaseAgent]):
        self.agency = agency
        self.agents = agents

    def __gt__(self, other: BaseAgent) -> "ArchonAgencyChainBuilder":
        """Add another agent to the communication chain."""
        self.agents.append(other)
        return self

    def finalize(self) -> ArchonAgency:
        """
        Finalize the communication chain and return the agency.

        Creates a chain communication flow from the sequence of agents.
        """
        if len(self.agents) < 2:
            raise ValueError("Communication chain must have at least 2 agents")

        # Create chain flows
        for i in range(len(self.agents) - 1):
            self.agency.add_communication_flow(
                sender=self.agents[i],
                receivers=self.agents[i + 1],
                flow_type=CommunicationFlowType.CHAIN
            )

        return self.agency


# Helper functions for creating agencies with fluent syntax
def create_agency(
    *entry_point_agents: BaseAgent,
    communication_flows: Optional[List] = None,
    config: Optional[AgencyConfig] = None,
    **kwargs
) -> ArchonAgency:
    """
    Create a new Archon Agency with the specified configuration.

    Args:
        *entry_point_agents: Entry point agents for the agency
        communication_flows: Communication flows between agents
        config: Agency configuration
        **kwargs: Additional configuration options

    Returns:
        Configured ArchonAgency instance
    """
    return ArchonAgency(
        *entry_point_agents,
        communication_flows=communication_flows,
        config=config,
        **kwargs
    )