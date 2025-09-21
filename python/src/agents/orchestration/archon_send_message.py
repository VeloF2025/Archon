"""
Archon SendMessage Tool - Enhanced inter-agent communication.

This module provides the SendMessage tool for agent-to-agent communication,
building upon Agency Swarm's communication patterns but enhanced for Archon's
enterprise requirements.

Key Features:
- Direct agent-to-agent messaging
- Thread-based conversation continuity
- Message validation and error handling
- Integration with existing Archon agents
- Support for both synchronous and asynchronous operations
- Enterprise-grade logging and monitoring
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field

# Lazy import to avoid circular dependency
def _get_archon_agency():
    """Lazy import of ArchonAgency to avoid circular imports"""
    try:
        from .archon_agency import ArchonAgency
        return ArchonAgency
    except ImportError:
        return None
from .archon_thread_manager import ThreadContext, ThreadMessage, ThreadStatus

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Priority levels for inter-agent messages."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class MessageStatus(Enum):
    """Status of inter-agent messages."""
    PENDING = "pending"
    DELIVERED = "delivered"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class AgentMessage:
    """Represents a message between agents."""
    message_id: str
    sender: str
    recipient: str
    content: str
    priority: MessagePriority = MessagePriority.NORMAL
    status: MessageStatus = MessageStatus.PENDING
    thread_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "priority": self.priority.value,
            "status": self.status.value,
            "thread_id": self.thread_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary."""
        return cls(
            message_id=data["message_id"],
            sender=data["sender"],
            recipient=data["recipient"],
            content=data["content"],
            priority=MessagePriority(data.get("priority", "normal")),
            status=MessageStatus(data.get("status", "pending")),
            thread_id=data.get("thread_id"),
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3)
        )


class SendMessageInput(BaseModel):
    """Input model for the SendMessage tool."""
    recipient_agent: str = Field(..., description="Name of the recipient agent")
    message: str = Field(..., description="Message content to send")
    thread_id: Optional[str] = Field(None, description="Optional thread ID for conversation continuity")
    priority: str = Field("normal", description="Message priority: low, normal, high, critical")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the message")
    timeout: Optional[float] = Field(None, description="Timeout in seconds for the response")


class SendMessageOutput(BaseModel):
    """Output model for the SendMessage tool."""
    success: bool
    message: str
    response_content: Optional[str] = None
    message_id: Optional[str] = None
    thread_id: Optional[str] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None


class ArchonSendMessageTool:
    """
    Enhanced SendMessage tool for inter-agent communication in Archon.

    This tool enables agents to send messages to other agents within the same agency,
    maintaining conversation threads and providing enterprise-grade reliability.
    """

    def __init__(self, agency: Any):  # Use Any to avoid circular import
        """
        Initialize the SendMessage tool.

        Args:
            agency: The agency instance this tool belongs to
        """
        self.agency = agency
        self.pending_messages: Dict[str, AgentMessage] = {}
        self.message_history: List[AgentMessage] = []

        logger.info(f"SendMessageTool initialized for agency: {agency.config.name or 'Unnamed'}")

    async def send_to_agent(
        self,
        recipient_agent: str,
        message: str,
        sender_agent: Optional[str] = None,
        thread_id: Optional[str] = None,
        priority: Union[MessagePriority, str] = MessagePriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> SendMessageOutput:
        """
        Send a message to another agent.

        Args:
            recipient_agent: Name of the recipient agent
            message: Message content to send
            sender_agent: Name of the sender agent (auto-detected if not provided)
            thread_id: Optional thread ID for conversation continuity
            priority: Message priority
            metadata: Additional metadata for the message
            timeout: Timeout in seconds for the response

        Returns:
            SendMessageOutput with the result
        """
        start_time = datetime.utcnow()

        try:
            # Validate recipient agent exists
            if recipient_agent not in self.agency.agents:
                return SendMessageOutput(
                    success=False,
                    message=f"Recipient agent '{recipient_agent}' not found in agency",
                    error="AGENT_NOT_FOUND"
                )

            # Auto-detect sender if not provided
            if not sender_agent:
                sender_agent = self._detect_sender_agent()
                if not sender_agent:
                    return SendMessageOutput(
                        success=False,
                        message="Could not determine sender agent",
                        error="SENDER_NOT_DETECTED"
                    )

            # Normalize priority
            if isinstance(priority, str):
                try:
                    priority = MessagePriority(priority.lower())
                except ValueError:
                    priority = MessagePriority.NORMAL

            # Create message
            agent_message = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender=sender_agent,
                recipient=recipient_agent,
                content=message,
                priority=priority,
                metadata=metadata or {}
            )

            # Set or create thread ID
            if not thread_id:
                thread_id = await self._get_or_create_thread(sender_agent, recipient_agent)
            agent_message.thread_id = thread_id

            # Add message to pending queue
            self.pending_messages[agent_message.message_id] = agent_message

            logger.info(f"Sending message from {sender_agent} to {recipient_agent} (thread: {thread_id})")

            # Process the message
            result = await self._process_message(agent_message, timeout)

            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()

            # Add to history
            agent_message.status = MessageStatus.COMPLETED if result.success else MessageStatus.FAILED
            self.message_history.append(agent_message)

            # Update thread with message and response
            await self._update_thread_conversation(agent_message, result)

            # Remove from pending
            self.pending_messages.pop(agent_message.message_id, None)

            return SendMessageOutput(
                success=result.success,
                message=result.message,
                response_content=result.response_content,
                message_id=agent_message.message_id,
                thread_id=thread_id,
                execution_time=execution_time,
                error=result.error
            )

        except Exception as e:
            logger.error(f"Error sending message from {sender_agent} to {recipient_agent}: {e}")
            execution_time = (datetime.utcnow() - start_time).total_seconds()

            return SendMessageOutput(
                success=False,
                message=f"Failed to send message: {str(e)}",
                execution_time=execution_time,
                error=str(e)
            )

    async def _process_message(self, message: AgentMessage, timeout: Optional[float]) -> SendMessageOutput:
        """
        Process a message and get response from recipient agent.

        Args:
            message: The message to process
            timeout: Optional timeout in seconds

        Returns:
            SendMessageOutput with the result
        """
        try:
            # Get recipient agent
            recipient_agent = self.agency.agents[message.recipient]

            # Create thread context for the agent
            try:
                thread_context = await self.agency.thread_manager.get_thread(message.thread_id)
            except ValueError:
                # Thread doesn't exist, create it
                thread_context = await self.agency.thread_manager.create_thread(
                    sender=message.sender,
                    recipient=message.recipient,
                    thread_id=message.thread_id
                )

            # Create dependencies with thread context
            from ...agents.base_agent import ArchonDependencies

            deps = ArchonDependencies(
                request_id=str(uuid.uuid4()),
                user_id=None,
                trace_id=str(uuid.uuid4()),
                context={
                    "thread_id": message.thread_id,
                    "agency_context": {
                        "agency_name": self.agency.config.name,
                        "sender_agent": message.sender,
                        "conversation_history": thread_context.get_recent_messages(10),
                        "message_priority": message.priority.value,
                        "message_metadata": message.metadata
                    }
                }
            )

            # Get response from recipient agent
            timeout = timeout or self.agency.config.default_timeout

            try:
                # Try to use confidence-enabled execution first
                if hasattr(recipient_agent, 'run_with_confidence'):
                    response, confidence = await asyncio.wait_for(
                        recipient_agent.run_with_confidence(
                            message.content,
                            deps,
                            task_description=f"Inter-agent message from {message.sender}: {message.content[:100]}..."
                        ),
                        timeout=timeout
                    )
                    logger.debug(f"Got response from {message.recipient} with confidence: {confidence}")
                else:
                    response = await asyncio.wait_for(
                        recipient_agent.run(message.content, deps),
                        timeout=timeout
                    )
                    logger.debug(f"Got response from {message.recipient}")

                # Convert response to string
                response_content = str(response) if response is not None else ""

                return SendMessageOutput(
                    success=True,
                    message=f"Successfully sent message to {message.recipient}",
                    response_content=response_content
                )

            except asyncio.TimeoutError:
                logger.warning(f"Timeout getting response from {message.recipient}")
                return SendMessageOutput(
                    success=False,
                    message=f"Timeout getting response from {message.recipient}",
                    error="TIMEOUT"
                )

            except Exception as e:
                logger.error(f"Error getting response from {message.recipient}: {e}")
                return SendMessageOutput(
                    success=False,
                    message=f"Error getting response from {message.recipient}: {str(e)}",
                    error=str(e)
                )

        except Exception as e:
            logger.error(f"Error processing message {message.message_id}: {e}")
            return SendMessageOutput(
                success=False,
                message=f"Error processing message: {str(e)}",
                error=str(e)
            )

    def _detect_sender_agent(self) -> Optional[str]:
        """
        Auto-detect the sender agent from the current context.

        Returns:
            Sender agent name or None if cannot be detected
        """
        # This is a simplified implementation
        # In a real implementation, this would use context from the current execution
        try:
            # Look at the thread manager's active threads
            active_threads = self.agency.thread_manager._active_threads
            if active_threads:
                # Return the most recently updated thread's sender
                latest_thread = max(active_threads.values(), key=lambda t: t.updated_at)
                return latest_thread.sender
        except Exception:
            pass

        # Alternative: return first entry point
        if self.agency.entry_points:
            return self.agency.entry_points[0].name

        return None

    async def _get_or_create_thread(self, sender: str, recipient: str) -> str:
        """
        Get existing thread or create new one for conversation continuity.

        Args:
            sender: Sender agent name
            recipient: Recipient agent name

        Returns:
            Thread ID
        """
        # Look for existing active thread between these agents
        try:
            existing_threads = await self.agency.thread_manager.list_threads(
                sender=sender,
                recipient=recipient,
                status=ThreadStatus.ACTIVE,
                active_only=True,
                limit=1
            )

            if existing_threads:
                return existing_threads[0].thread_id

        except Exception as e:
            logger.warning(f"Error looking for existing thread: {e}")

        # Create new thread
        return await self.agency.thread_manager.create_thread(sender, recipient)

    async def _update_thread_conversation(self, message: AgentMessage, result: SendMessageOutput) -> None:
        """
        Update the thread with the message and response.

        Args:
            message: The sent message
            result: The result of sending the message
        """
        try:
            thread_context = await self.agency.thread_manager.get_thread(message.thread_id)

            # Add sent message
            sent_message = ThreadMessage(
                role="user",
                content=message.content,
                sender=message.sender,
                recipient=message.recipient,
                timestamp=message.timestamp,
                metadata={
                    "message_id": message.message_id,
                    "priority": message.priority.value,
                    **message.metadata
                }
            )
            await thread_context.add_message_async(sent_message)

            # Add response if successful
            if result.success and result.response_content:
                response_message = ThreadMessage(
                    role="assistant",
                    content=result.response_content,
                    sender=message.recipient,
                    recipient=message.sender,
                    timestamp=datetime.utcnow().isoformat(),
                    metadata={
                        "original_message_id": message.message_id,
                        "execution_time": result.execution_time
                    }
                )
                await thread_context.add_message_async(response_message)

        except Exception as e:
            logger.error(f"Error updating thread conversation: {e}")

    def get_message_status(self, message_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a message.

        Args:
            message_id: Message identifier

        Returns:
            Message status dictionary or None if not found
        """
        # Check pending messages
        if message_id in self.pending_messages:
            message = self.pending_messages[message_id]
            return {
                "status": message.status.value,
                "pending": True,
                "retry_count": message.retry_count
            }

        # Check message history
        for message in self.message_history:
            if message.message_id == message_id:
                return {
                    "status": message.status.value,
                    "pending": False,
                    "retry_count": message.retry_count,
                    "completed_at": message.timestamp
                }

        return None

    def get_message_history(
        self,
        sender: Optional[str] = None,
        recipient: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get message history with optional filtering.

        Args:
            sender: Filter by sender
            recipient: Filter by recipient
            limit: Maximum number of messages

        Returns:
            List of message dictionaries
        """
        messages = self.message_history

        # Apply filters
        if sender:
            messages = [m for m in messages if m.sender == sender]
        if recipient:
            messages = [m for m in messages if m.recipient == recipient]

        # Sort by timestamp (newest first)
        messages.sort(key=lambda m: m.timestamp, reverse=True)

        return [msg.to_dict() for msg in messages[:limit]]

    async def broadcast_message(
        self,
        message: str,
        sender_agent: Optional[str] = None,
        recipient_agents: Optional[List[str]] = None,
        priority: Union[MessagePriority, str] = MessagePriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, SendMessageOutput]:
        """
        Broadcast a message to multiple agents.

        Args:
            message: Message content to send
            sender_agent: Name of the sender agent
            recipient_agents: List of recipient agent names
            priority: Message priority
            metadata: Additional metadata for the message

        Returns:
            Dictionary mapping agent names to their responses
        """
        if not recipient_agents:
            recipient_agents = list(self.agency.agents.keys())
            if sender_agent and sender_agent in recipient_agents:
                recipient_agents.remove(sender_agent)

        results = {}
        tasks = []

        # Create tasks for each recipient
        for recipient in recipient_agents:
            task = self.send_to_agent(
                recipient_agent=recipient,
                message=message,
                sender_agent=sender_agent,
                priority=priority,
                metadata=metadata
            )
            tasks.append((recipient, task))

        # Execute all tasks concurrently
        for recipient, task in tasks:
            try:
                result = await task
                results[recipient] = result
            except Exception as e:
                logger.error(f"Error broadcasting to {recipient}: {e}")
                results[recipient] = SendMessageOutput(
                    success=False,
                    message=f"Failed to broadcast to {recipient}: {str(e)}",
                    error=str(e)
                )

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get messaging statistics.

        Returns:
            Dictionary with messaging statistics
        """
        total_messages = len(self.message_history)
        pending_messages = len(self.pending_messages)

        status_counts = {}
        priority_counts = {}

        for message in self.message_history:
            status = message.status.value
            priority = message.priority.value

            status_counts[status] = status_counts.get(status, 0) + 1
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

        for message in self.pending_messages.values():
            status = message.status.value
            priority = message.priority.value

            status_counts[status] = status_counts.get(status, 0) + 1
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

        return {
            "total_messages": total_messages,
            "pending_messages": pending_messages,
            "status_distribution": status_counts,
            "priority_distribution": priority_counts,
            "available_agents": len(self.agency.agents),
            "entry_points": len(self.agency.entry_points)
        }


# PydanticAI tool function for direct integration
async def send_message_tool(
    recipient_agent: str,
    message: str,
    thread_id: Optional[str] = None,
    priority: str = "normal",
    metadata: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
    # Agency instance is injected by the tool wrapper
    agency: Any = None
) -> SendMessageOutput:
    """
    PydanticAI tool function for sending messages between agents.

    Args:
        recipient_agent: Name of the recipient agent
        message: Message content to send
        thread_id: Optional thread ID for conversation continuity
        priority: Message priority: low, normal, high, critical
        metadata: Additional metadata for the message
        timeout: Timeout in seconds for the response
        agency: Injected ArchonAgency instance

    Returns:
        SendMessageOutput with the result
    """
    if not agency:
        raise ValueError("Agency instance not provided to send_message_tool")

    send_tool = ArchonSendMessageTool(agency)
    return await send_tool.send_to_agent(
        recipient_agent=recipient_agent,
        message=message,
        thread_id=thread_id,
        priority=priority,
        metadata=metadata,
        timeout=timeout
    )