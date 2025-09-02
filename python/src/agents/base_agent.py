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
from typing import Any, Generic, TypeVar, Tuple, Optional

from pydantic import BaseModel
from pydantic_ai import Agent

# MANDATORY MANIFEST INTEGRATION
from .configs.MANIFEST_INTEGRATION import get_archon_manifest, enforce_manifest_compliance, get_manifest_system_prompt

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
    """

    def __init__(
        self,
        model: str = "openai:gpt-4o",
        name: str = None,
        retries: int = 3,
        enable_rate_limiting: bool = True,
        enable_confidence: bool = True,
        **agent_kwargs,
    ):
        self.model = model
        self.name = name or self.__class__.__name__
        self.retries = retries
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_confidence = enable_confidence and CONFIDENCE_AVAILABLE

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
        
        # MANDATORY: Log manifest compliance
        confidence_status = "with confidence scoring" if self.enable_confidence else "without confidence scoring"
        self.logger.info(f"✅ {self.name} initialized with MANIFEST compliance {confidence_status}")

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
        try:
            # Add timeout to prevent hanging
            result = await asyncio.wait_for(
                self._agent.run(user_prompt, deps=deps),
                timeout=120.0,  # 2 minute timeout for agent operations
            )
            self.logger.info(f"Agent {self.name} completed successfully")
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
            self.logger.error(f"Agent {self.name} timed out after 120 seconds")
            raise Exception(f"Agent {self.name} operation timed out - taking too long to respond")
        except Exception as e:
            self.logger.error(f"Agent {self.name} failed: {str(e)}")
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
