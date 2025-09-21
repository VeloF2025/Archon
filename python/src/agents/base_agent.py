"""
Base Agent class for all PydanticAI agents in the Archon system.

This provides common functionality and dependency injection for all agents.
MANDATORY: All agents must comply with ARCHON OPERATIONAL MANIFEST (MANIFEST.md)
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, Tuple, Optional, Callable, Dict, List
from datetime import datetime
import uuid

from pydantic import BaseModel
from pydantic_ai import Agent

# MANDATORY MANIFEST INTEGRATION
from .configs.MANIFEST_INTEGRATION import get_archon_manifest, enforce_manifest_compliance, get_manifest_system_prompt

# KAFKA INTEGRATION
def _get_kafka_service():
    """Lazy import of Kafka service to avoid circular imports"""
    try:
        from ..server.services.kafka_integration_service import get_kafka_service
        return get_kafka_service()
    except ImportError as e:
        logging.warning(f"Kafka integration not available: {e}")
        return None

# AGENCY INTEGRATION (Lazy loading to avoid circular imports)
def _get_agency_integration():
    """Lazy import of agency integration to avoid circular imports"""
    try:
        from .orchestration.archon_agency import ArchonAgency
        from .orchestration.archon_send_message import ArchonSendMessageTool
        return ArchonAgency, ArchonSendMessageTool
    except ImportError as e:
        logging.warning(f"Agency integration not available: {e}")
        return None, None

# CONFIDENCE INTEGRATION (Lazy loading to avoid circular imports)
def _get_confidence_integration():
    """Lazy import of confidence integration to avoid circular imports"""
    try:
        from .integration.confidence_integration import execute_agent_with_confidence, get_confidence_integration
        return execute_agent_with_confidence, get_confidence_integration
    except ImportError as e:
        logging.warning(f"Confidence integration not available: {e}")
        return None, None

CONFIDENCE_AVAILABLE = True
try:
    # Test if we can import the confidence modules
    from .deepconf.engine import ConfidenceScore
except ImportError as e:
    logging.warning(f"Confidence integration not available: {e}")
    CONFIDENCE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ArchonDependencies:
    """Base dependencies for all Archon agents."""

    request_id: str | None = None
    user_id: str | None = None
    trace_id: str | None = None
    # Additional fields for specialized agent compatibility
    agent_role: str = ""
    task_description: str = ""
    context: dict = None
    tool_permissions: list = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.tool_permissions is None:
            self.tool_permissions = []


# Type variables for generic agent typing
DepsT = TypeVar("DepsT", bound=ArchonDependencies)
OutputT = TypeVar("OutputT")


class BaseAgentOutput(BaseModel):
    """Base output model for all agent responses."""

    success: bool
    message: str
    data: dict[str, Any] | None = None
    errors: list[str] | None = None


class RateLimitHandler:
    """Handles OpenAI rate limiting with exponential backoff."""

    def __init__(self, max_retries: int = 5, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Minimum 100ms between requests

    async def execute_with_rate_limit(self, func, *args, progress_callback=None, **kwargs):
        """Execute a function with rate limiting protection."""
        retries = 0

        while retries <= self.max_retries:
            try:
                # Ensure minimum interval between requests
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                if time_since_last < self.min_request_interval:
                    await asyncio.sleep(self.min_request_interval - time_since_last)

                self.last_request_time = time.time()
                return await func(*args, **kwargs)

            except Exception as e:
                error_str = str(e).lower()
                full_error = str(e)

                logger.debug(f"Agent error caught: {full_error}")
                logger.debug(f"Error type: {type(e).__name__}")
                logger.debug(f"Error class: {e.__class__.__module__}.{e.__class__.__name__}")

                # Check for different types of rate limits
                is_rate_limit = (
                    "rate limit" in error_str
                    or "429" in error_str
                    or "request_limit" in error_str  # New: catch PydanticAI limits
                    or "exceed" in error_str
                )

                if is_rate_limit:
                    retries += 1
                    if retries > self.max_retries:
                        logger.debug(f"Max retries exceeded for rate limit: {full_error}")
                        if progress_callback:
                            await progress_callback({
                                "step": "ai_generation",
                                "log": f"❌ Rate limit exceeded after {self.max_retries} retries",
                            })
                        raise Exception(
                            f"Rate limit exceeded after {self.max_retries} retries: {full_error}"
                        )

                    # Extract wait time from error message if available
                    wait_time = self._extract_wait_time(full_error)
                    if wait_time is None:
                        # Use exponential backoff
                        wait_time = self.base_delay * (2 ** (retries - 1))

                    logger.info(
                        f"Rate limit hit. Type: {type(e).__name__}, Waiting {wait_time:.2f}s before retry {retries}/{self.max_retries}"
                    )

                    # Send progress update if callback provided
                    if progress_callback:
                        await progress_callback({
                            "step": "ai_generation",
                            "log": f"⏱️ Rate limit hit. Waiting {wait_time:.0f}s before retry {retries}/{self.max_retries}",
                        })

                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Non-rate-limit error, re-raise immediately
                    logger.debug(f"Non-rate-limit error, re-raising: {full_error}")
                    if progress_callback:
                        await progress_callback({
                            "step": "ai_generation",
                            "log": f"❌ Error: {str(e)}",
                        })
                    raise

        raise Exception(f"Failed after {self.max_retries} retries")

    def _extract_wait_time(self, error_message: str) -> float | None:
        """Extract wait time from OpenAI error message."""
        try:
            # Look for patterns like "Please try again in 1.242s"
            import re

            match = re.search(r"try again in (\d+(?:\.\d+)?)s", error_message)
            if match:
                return float(match.group(1))
        except:
            pass
        return None


class BaseAgent(ABC, Generic[DepsT, OutputT]):
    """
    Base class for all PydanticAI agents in the Archon system.

    Provides common functionality like:
    - Error handling and retries
    - Rate limiting protection
    - Logging and monitoring
    - Standard dependency injection
    - Common tools and utilities
    - Agency integration for inter-agent communication
    """

    def __init__(
        self,
        model: str = "openai:gpt-4o",
        name: str = None,
        retries: int = 3,
        enable_rate_limiting: bool = True,
        enable_confidence: bool = True,
        enable_agency: bool = True,
        **agent_kwargs,
    ):
        self.model = model
        self.name = name or self.__class__.__name__
        self.retries = retries
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_confidence = enable_confidence and CONFIDENCE_AVAILABLE
        self.enable_agency = enable_agency

        # MANDATORY: Enforce manifest compliance for all agents
        if not enforce_manifest_compliance(self.name, "initialization"):
            raise RuntimeError(f"MANIFEST COMPLIANCE FAILURE: {self.name} cannot initialize without manifest")

        # Initialize rate limiting
        if self.enable_rate_limiting:
            self.rate_limiter = RateLimitHandler(max_retries=retries)
        else:
            self.rate_limiter = None

        # Initialize the PydanticAI agent
        self._agent = self._create_agent(**agent_kwargs)

        # Setup logging
        self.logger = logging.getLogger(f"agents.{self.name}")
        
        # Initialize Kafka communication
        self.kafka_service = _get_kafka_service()
        self.kafka_enabled = self.kafka_service is not None and self.kafka_service.is_initialized

        # Initialize Agency integration
        self.agency = None  # Will be set when agent is added to an agency
        self.send_message_tool = None  # Will be initialized by agency
        self.agency_enabled = self.enable_agency
        
        # MANDATORY: Log manifest compliance
        confidence_status = "with confidence scoring" if self.enable_confidence else "without confidence scoring"
        kafka_status = "with Kafka messaging" if self.kafka_enabled else "without Kafka messaging"
        agency_status = "with Agency support" if self.agency_enabled else "without Agency support"
        self.logger.info(f"✅ {self.name} initialized with MANIFEST compliance {confidence_status} {kafka_status} {agency_status}")

    @abstractmethod
    def _create_agent(self, **kwargs) -> Agent:
        """Create and configure the PydanticAI agent. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent. Must be implemented by subclasses."""
        pass
    
    def get_manifest_enhanced_system_prompt(self) -> str:
        """
        MANDATORY: Get system prompt enhanced with manifest directives
        All agents MUST use this instead of get_system_prompt() directly
        """
        manifest_prefix = get_manifest_system_prompt()
        agent_specific_prompt = self.get_system_prompt()
        
        return f"{manifest_prefix}\n\n{agent_specific_prompt}\n\nMANDATE: Follow all ARCHON OPERATIONAL MANIFEST protocols."

    async def run(self, user_prompt: str, deps: DepsT) -> OutputT:
        """
        Run the agent with rate limiting protection.
        MANDATORY: All runs must enforce manifest compliance

        Args:
            user_prompt: The user's input prompt
            deps: Dependencies for the agent

        Returns:
            The agent's structured output
        """
        # MANDATORY: Enforce manifest compliance before execution
        if not enforce_manifest_compliance(self.name, "run_execution"):
            raise RuntimeError(f"MANIFEST COMPLIANCE FAILURE: {self.name} cannot execute without manifest")
            
        if self.rate_limiter:
            # Extract progress callback from deps if available
            progress_callback = getattr(deps, "progress_callback", None)
            return await self.rate_limiter.execute_with_rate_limit(
                self._run_agent, user_prompt, deps, progress_callback=progress_callback
            )
        else:
            return await self._run_agent(user_prompt, deps)
    
    async def run_with_confidence(self, user_prompt: str, deps: DepsT, task_description: Optional[str] = None) -> Tuple[OutputT, Optional["ConfidenceScore"]]:
        """
        Run the agent with confidence scoring enabled.
        
        This method provides the same functionality as run() but additionally
        calculates confidence scores and tracks uncertainty metrics.

        Args:
            user_prompt: The user's input prompt
            deps: Dependencies for the agent
            task_description: Optional detailed task description for better confidence analysis

        Returns:
            Tuple[OutputT, Optional[ConfidenceScore]]: Agent output and confidence score
        """
        # MANDATORY: Enforce manifest compliance before execution
        if not enforce_manifest_compliance(self.name, "run_execution"):
            raise RuntimeError(f"MANIFEST COMPLIANCE FAILURE: {self.name} cannot execute without manifest")
        
        if not self.enable_confidence:
            # Confidence disabled, run normally
            result = await self.run(user_prompt, deps)
            return result, None
        
        try:
            # Get confidence integration functions
            execute_with_confidence, _ = _get_confidence_integration()
            
            if execute_with_confidence:
                # Execute with confidence integration
                result, confidence_score = await execute_with_confidence(
                    self, user_prompt, deps, task_description
                )
                return result, confidence_score
            else:
                # Confidence not available, run normally
                result = await self.run(user_prompt, deps)
                return result, None
            
        except Exception as e:
            self.logger.error(f"Confidence-enabled execution failed: {e}")
            # Fallback to normal execution
            result = await self.run(user_prompt, deps)
            return result, None

    async def _run_agent(self, user_prompt: str, deps: DepsT) -> OutputT:
        """Internal method to run the agent."""
        start_time = time.time()
        
        # Publish task started event
        await self.publish_agent_event(
            "task_started", 
            {
                "prompt": user_prompt[:200] + "..." if len(user_prompt) > 200 else user_prompt,
                "dependencies": getattr(deps, "__dict__", {}),
                "start_time": datetime.utcnow().isoformat()
            }
        )
        
        try:
            # Add timeout to prevent hanging
            result = await asyncio.wait_for(
                self._agent.run(user_prompt, deps=deps),
                timeout=120.0,  # 2 minute timeout for agent operations
            )
            
            execution_time = time.time() - start_time
            
            # Publish task completed event
            await self.publish_agent_event(
                "task_completed",
                {
                    "execution_time_seconds": execution_time,
                    "success": True,
                    "end_time": datetime.utcnow().isoformat()
                }
            )
            
            # Publish analytics
            await self.publish_analytics(
                "execution_time",
                execution_time,
                {"status": "success"}
            )
            
            self.logger.info(f"Agent {self.name} completed successfully in {execution_time:.2f}s")
            # PydanticAI 0.8.x returns an AgentRunResult - extract the actual data
            try:
                # Based on PydanticAI documentation, the result should have an 'output' property
                # Example from docs: agent_run.output
                if hasattr(result, 'output'):
                    return result.output
                # Fallback attempts for different possible attributes
                elif hasattr(result, 'data'):
                    return result.data
                elif hasattr(result, 'content'):
                    return result.content
                elif hasattr(result, 'result'):
                    return result.result
                elif hasattr(result, 'value'):
                    return result.value
                else:
                    # Try to access the output property directly even if hasattr fails
                    try:
                        return result.output
                    except AttributeError:
                        pass
                    
                    # Log what we actually have
                    self.logger.warning(f"AgentRunResult type: {type(result).__name__}")
                    self.logger.warning(f"Available methods: {[attr for attr in dir(result) if not attr.startswith('_')]}")
                    
                    # Try calling str() on the result as final fallback
                    result_str = str(result)
                    if result_str and result_str != f"<{type(result).__name__} object at 0x":
                        return result_str
                    
                    # Give up and return a status message
                    return "Agent completed successfully but result format is unknown"
            except Exception as attr_error:
                self.logger.error(f"Result extraction failed: {attr_error}")
                return f"Agent completed with extraction error: {str(attr_error)}"
        except TimeoutError:
            execution_time = time.time() - start_time
            error_msg = f"Agent {self.name} timed out after 120 seconds"
            
            # Publish timeout event
            await self.publish_agent_event(
                "task_timeout",
                {
                    "execution_time_seconds": execution_time,
                    "timeout_seconds": 120.0,
                    "error": error_msg,
                    "end_time": datetime.utcnow().isoformat()
                },
                priority="high"
            )
            
            # Publish analytics
            await self.publish_analytics(
                "execution_time",
                execution_time,
                {"status": "timeout"}
            )
            
            self.logger.error(error_msg)
            raise Exception(f"Agent {self.name} operation timed out - taking too long to respond")
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            # Publish error event
            await self.publish_agent_event(
                "task_error",
                {
                    "execution_time_seconds": execution_time,
                    "error": error_msg,
                    "error_type": type(e).__name__,
                    "end_time": datetime.utcnow().isoformat()
                },
                priority="high"
            )
            
            # Publish analytics
            await self.publish_analytics(
                "execution_time",
                execution_time,
                {"status": "error", "error_type": type(e).__name__}
            )
            
            self.logger.error(f"Agent {self.name} failed: {error_msg}")
            raise

    def run_stream(self, user_prompt: str, deps: DepsT):
        """
        Run the agent with streaming output.

        Args:
            user_prompt: The user's input prompt
            deps: Dependencies for the agent

        Returns:
            Async context manager for streaming results
        """
        # Note: Rate limiting not supported for streaming to avoid complexity
        # The async context manager pattern doesn't work well with rate limiting
        self.logger.info(f"Starting streaming for agent {self.name}")
        # run_stream returns an async context manager directly, not a coroutine
        return self._agent.run_stream(user_prompt, deps=deps)

    def add_tool(self, func, **tool_kwargs):
        """
        Add a tool function to the agent.

        Args:
            func: The function to register as a tool
            **tool_kwargs: Additional arguments for the tool decorator
        """
        return self._agent.tool(**tool_kwargs)(func)

    def add_system_prompt_function(self, func):
        """
        Add a dynamic system prompt function to the agent.

        Args:
            func: The function to register as a system prompt
        """
        return self._agent.system_prompt(func)

    # Kafka Communication Methods
    
    async def publish_agent_event(
        self, 
        event_type: str, 
        data: dict[str, Any], 
        priority: str = "normal"
    ) -> bool:
        """
        Publish an agent event to Kafka for inter-agent communication
        
        Args:
            event_type: Type of event (e.g., "task_started", "task_completed", "error")
            data: Event data payload
            priority: Message priority (critical, high, normal, low, background)
            
        Returns:
            bool: True if published successfully, False otherwise
        """
        if not self.kafka_enabled:
            self.logger.debug(f"Kafka not enabled for {self.name}, event not published")
            return False
            
        try:
            from ..server.services.kafka_integration_service import MessagePriority
            
            # Map priority string to enum
            priority_map = {
                "critical": MessagePriority.CRITICAL,
                "high": MessagePriority.HIGH,
                "normal": MessagePriority.NORMAL,
                "low": MessagePriority.LOW,
                "background": MessagePriority.BACKGROUND
            }
            
            priority_enum = priority_map.get(priority, MessagePriority.NORMAL)
            
            # Enhance data with agent metadata
            enhanced_data = {
                **data,
                "agent_name": self.name,
                "agent_type": self.__class__.__name__,
                "timestamp": datetime.utcnow().isoformat(),
                "event_id": str(uuid.uuid4())
            }
            
            # Publish system event
            success = await self.kafka_service.publish_system_event(
                event_type=f"agent.{event_type}",
                data=enhanced_data,
                priority=priority_enum
            )
            
            if success:
                self.logger.debug(f"Published agent event: {event_type}")
            else:
                self.logger.warning(f"Failed to publish agent event: {event_type}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error publishing agent event {event_type}: {e}")
            return False

    async def send_agent_command(
        self, 
        target_agent_id: str, 
        command: str, 
        params: dict[str, Any] = None
    ) -> bool:
        """
        Send a command to another agent via Kafka
        
        Args:
            target_agent_id: ID of the target agent
            command: Command to send
            params: Command parameters
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.kafka_enabled:
            self.logger.debug(f"Kafka not enabled for {self.name}, command not sent")
            return False
            
        try:
            success = await self.kafka_service.publish_agent_command(
                agent_id=target_agent_id,
                command=command,
                params=params or {}
            )
            
            if success:
                self.logger.debug(f"Sent command to {target_agent_id}: {command}")
            else:
                self.logger.warning(f"Failed to send command to {target_agent_id}: {command}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending agent command {command} to {target_agent_id}: {e}")
            return False

    async def publish_analytics(
        self, 
        metric_type: str, 
        value: Any, 
        tags: dict[str, str] = None
    ) -> bool:
        """
        Publish analytics/metrics data to Kafka
        
        Args:
            metric_type: Type of metric (e.g., "execution_time", "success_rate")
            value: Metric value
            tags: Additional tags for the metric
            
        Returns:
            bool: True if published successfully, False otherwise
        """
        if not self.kafka_enabled:
            self.logger.debug(f"Kafka not enabled for {self.name}, analytics not published")
            return False
            
        try:
            # Enhance tags with agent metadata
            enhanced_tags = {
                "agent_name": self.name,
                "agent_type": self.__class__.__name__,
                **(tags or {})
            }
            
            success = await self.kafka_service.publish_analytics_event(
                metric_type=f"agent.{metric_type}",
                value=value,
                tags=enhanced_tags
            )
            
            if success:
                self.logger.debug(f"Published analytics: {metric_type} = {value}")
            else:
                self.logger.warning(f"Failed to publish analytics: {metric_type}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error publishing analytics {metric_type}: {e}")
            return False

    async def register_command_handler(self, command: str, handler_func: Callable):
        """
        Register a handler for incoming agent commands
        
        Args:
            command: Command name to handle
            handler_func: Async function to handle the command
        """
        if not self.kafka_enabled:
            self.logger.debug(f"Kafka not enabled for {self.name}, command handler not registered")
            return
            
        try:
            # Create a wrapper that handles the agent command event
            async def command_wrapper(event_data):
                try:
                    if (event_data.event_type == "agent_command" and 
                        event_data.data.get("command") == command):
                        
                        # Call the handler with the command parameters
                        await handler_func(event_data.data.get("params", {}))
                        
                        # Publish command completion event
                        await self.publish_agent_event(
                            "command_completed",
                            {
                                "command": command,
                                "original_event_id": event_data.event_id
                            }
                        )
                        
                except Exception as e:
                    self.logger.error(f"Error handling command {command}: {e}")
                    
                    # Publish command error event
                    await self.publish_agent_event(
                        "command_error",
                        {
                            "command": command,
                            "error": str(e),
                            "original_event_id": event_data.event_id
                        }
                    )
            
            # Register the event handler with Kafka service
            self.kafka_service.register_event_handler(f"agent_command_{self.name}", command_wrapper)
            
            self.logger.info(f"Registered command handler for {self.name}: {command}")
            
        except Exception as e:
            self.logger.error(f"Error registering command handler {command}: {e}")

    async def get_kafka_status(self) -> dict[str, Any]:
        """
        Get Kafka connection and messaging status for this agent
        
        Returns:
            dict: Status information
        """
        return {
            "agent_name": self.name,
            "kafka_enabled": self.kafka_enabled,
            "kafka_service_initialized": self.kafka_service is not None and self.kafka_service.is_initialized if self.kafka_service else False,
            "messaging_system_available": (
                self.kafka_service.messaging_system is not None 
                if self.kafka_service else False
            )
        }

    async def get_confidence_metrics(self, task_id: str) -> Optional[dict[str, Any]]:
        """
        Get confidence metrics for a specific task
        
        Args:
            task_id: Task identifier to get metrics for
            
        Returns:
            Optional[Dict[str, Any]]: Confidence metrics or None if not available
        """
        if not self.enable_confidence:
            return None
            
        try:
            _, get_integration = _get_confidence_integration()
            if get_integration:
                integration = get_integration()
                return await integration.get_confidence_metrics(task_id)
            return None
        except Exception as e:
            self.logger.error(f"Failed to get confidence metrics for task {task_id}: {e}")
            return None
    
    def cleanup_confidence_tracking(self, task_id: str) -> None:
        """
        Clean up confidence tracking for a completed task
        
        Args:
            task_id: Task identifier to clean up
        """
        if not self.enable_confidence:
            return
            
        try:
            _, get_integration = _get_confidence_integration()
            if get_integration:
                integration = get_integration()
                integration.cleanup_tracking(task_id)
        except Exception as e:
            self.logger.error(f"Failed to cleanup confidence tracking for task {task_id}: {e}")

    @property
    def agent(self) -> Agent:
        """Get the underlying PydanticAI agent instance."""
        return self._agent

    # Agency Integration Methods

    def set_agency(self, agency: Any) -> None:
        """
        Set the agency for this agent. Called when agent is added to an agency.

        Args:
            agency: The ArchonAgency instance
        """
        self.agency = agency
        if agency and hasattr(agency, 'send_message_tool'):
            self.send_message_tool = agency.send_message_tool
        self.logger.info(f"Agent {self.name} added to agency: {getattr(agency, 'config', {}).get('name', 'Unnamed')}")

    def is_in_agency(self) -> bool:
        """Check if this agent is part of an agency."""
        return self.agency is not None

    async def send_message_to_agent(
        self,
        recipient_agent_name: str,
        message: str,
        thread_id: Optional[str] = None,
        priority: str = "normal",
        metadata: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Send a message to another agent within the same agency.

        Args:
            recipient_agent_name: Name of the recipient agent
            message: Message content to send
            thread_id: Optional thread ID for conversation continuity
            priority: Message priority: low, normal, high, critical
            metadata: Additional metadata for the message
            timeout: Timeout in seconds for the response

        Returns:
            Response from the recipient agent

        Raises:
            ValueError: If agent not in agency or recipient not found
        """
        if not self.agency:
            raise ValueError(f"Agent {self.name} is not part of an agency")

        if not self.send_message_tool:
            raise ValueError(f"Agent {self.name} does not have send_message_tool configured")

        return await self.send_message_tool.send_to_agent(
            recipient_agent=recipient_agent_name,
            message=message,
            sender_agent=self.name,
            thread_id=thread_id,
            priority=priority,
            metadata=metadata,
            timeout=timeout
        )

    async def broadcast_message(
        self,
        message: str,
        recipient_agents: Optional[List[str]] = None,
        priority: str = "normal",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Broadcast a message to multiple agents within the agency.

        Args:
            message: Message content to send
            recipient_agents: List of recipient agent names (None for all agents)
            priority: Message priority
            metadata: Additional metadata for the message

        Returns:
            Dictionary mapping agent names to their responses

        Raises:
            ValueError: If agent not in agency
        """
        if not self.agency:
            raise ValueError(f"Agent {self.name} is not part of an agency")

        if not self.send_message_tool:
            raise ValueError(f"Agent {self.name} does not have send_message_tool configured")

        return await self.send_message_tool.broadcast_message(
            message=message,
            sender_agent=self.name,
            recipient_agents=recipient_agents,
            priority=priority,
            metadata=metadata
        )

    def get_available_agents(self) -> List[str]:
        """
        Get list of available agents in the same agency.

        Returns:
            List of agent names, empty list if not in agency
        """
        if not self.agency:
            return []

        return self.agency.list_agents()

    def get_communication_flows(self) -> List[Dict[str, Any]]:
        """
        Get communication flows from the agency.

        Returns:
            List of communication flow dictionaries, empty list if not in agency
        """
        if not self.agency:
            return []

        return self.agency.get_communication_flows()

    async def create_conversation_thread(
        self,
        recipient_agent_name: str,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new conversation thread with another agent.

        Args:
            recipient_agent_name: Name of the recipient agent
            initial_context: Initial context for the conversation

        Returns:
            Thread ID

        Raises:
            ValueError: If agent not in agency or recipient not found
        """
        if not self.agency:
            raise ValueError(f"Agent {self.name} is not part of an agency")

        if recipient_agent_name not in self.agency.agents:
            raise ValueError(f"Recipient agent '{recipient_agent_name}' not found in agency")

        return await self.agency.create_conversation_thread(
            sender=self.name,
            recipient=recipient_agent_name,
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

        Raises:
            ValueError: If agent not in agency
        """
        if not self.agency:
            raise ValueError(f"Agent {self.name} is not part of an agency")

        return await self.agency.get_conversation_history(thread_id, limit)

    async def get_agency_status(self) -> Dict[str, Any]:
        """
        Get status information about the agency.

        Returns:
            Dictionary with agency status information

        Raises:
            ValueError: If agent not in agency
        """
        if not self.agency:
            raise ValueError(f"Agent {self.name} is not part of an agency")

        try:
            thread_stats = await self.agency.thread_manager.get_thread_statistics()
            message_stats = self.send_message_tool.get_statistics() if self.send_message_tool else {}

            return {
                "agency_name": self.agency.config.name or "Unnamed",
                "total_agents": len(self.agency.agents),
                "entry_points": len(self.agency.entry_points),
                "communication_flows": len(self.agency.communication_flows),
                "thread_statistics": thread_stats,
                "message_statistics": message_stats,
                "persistence_enabled": self.agency.config.enable_persistence,
                "streaming_enabled": self.agency.config.enable_streaming
            }
        except Exception as e:
            self.logger.error(f"Error getting agency status: {e}")
            return {"error": str(e)}

    # Handoff Integration Methods

    async def request_handoff(
        self,
        target_agent_name: str,
        message: str,
        task_description: Optional[str] = None,
        strategy: str = "sequential",
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Request a handoff to another agent.

        Args:
            target_agent_name: Name of the target agent
            message: Original message that triggered handoff request
            task_description: Description of the task being handed off
            strategy: Handoff strategy to use
            context: Additional context for the handoff

        Returns:
            Response from target agent after handoff

        Raises:
            ValueError: If agent not in agency or target not found
        """
        if not self.agency:
            raise ValueError(f"Agent {self.name} is not part of an agency")

        if target_agent_name not in self.agency.agents:
            raise ValueError(f"Target agent '{target_agent_name}' not found in agency")

        # Use task description from current context if not provided
        if not task_description:
            task_description = f"Handoff from {self.name}: {message[:100]}..."

        # Request handoff through agency
        handoff_result = await self.agency.request_handoff(
            source_agent=self.name,
            target_agent=target_agent_name,
            message=message,
            task_description=task_description,
            strategy=strategy,
            context=context or {}
        )

        # Return the response content if successful
        if handoff_result.status.value == "completed":
            return handoff_result.response_content
        else:
            raise Exception(f"Handoff failed: {handoff_result.error_message}")

    async def get_handoff_recommendations(
        self,
        message: str,
        task_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get handoff recommendations for the current task.

        Args:
            message: Current message/task
            task_description: Optional task description

        Returns:
            Dictionary with handoff recommendations
        """
        if not self.agency:
            return {"error": "Agent not part of an agency"}

        if not task_description:
            task_description = message

        return await self.agency.get_handoff_recommendations(
            message=message,
            task_description=task_description,
            current_agent=self.name
        )

    async def should_handoff(
        self,
        message: str,
        task_description: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if a handoff should be considered for the current task.

        Args:
            message: Current message/task
            task_description: Optional task description
            context: Additional context

        Returns:
            True if handoff should be considered
        """
        try:
            recommendations = await self.get_handoff_recommendations(message, task_description)
            return recommendations.get("recommended", False)
        except Exception as e:
            self.logger.warning(f"Error checking handoff recommendation: {e}")
            return False
