"""
Agency Communication MCP Tools for Archon MCP Server

This module provides MCP tools for agent communication patterns including:
- Sending messages between agents
- Managing conversation threads
- Broadcasting messages to multiple agents
- Handling agent communication flows
- Managing communication context and history

Integration with Phase 1 Agency Swarm communication system.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from mcp.server.fastmcp import Context, FastMCP

# Add the project root to Python path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import agency communication components
from src.agents.orchestration.archon_agency import ArchonAgency, CommunicationFlowType
from src.agents.orchestration.archon_thread_manager import ThreadContext
from src.server.config.logfire_config import mcp_logger

logger = logging.getLogger(__name__)


class CommunicationType(str, Enum):
    """Types of agent communication"""
    DIRECT = "direct"
    BROADCAST = "broadcast"
    CHAIN = "chain"
    MULTICAST = "multicast"
    ASYNC = "async"


class MessagePriority(str, Enum):
    """Message priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class AgentMessage:
    """Represents a message between agents"""
    message_id: str
    sender: str
    recipient: Union[str, List[str]]
    content: str
    message_type: CommunicationType
    priority: MessagePriority
    thread_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class CommunicationManager:
    """Manages agent communication patterns and message routing"""

    def __init__(self):
        self.active_agencies: Dict[str, ArchonAgency] = {}
        self.message_history: List[AgentMessage] = []
        self.communication_contexts: Dict[str, Dict[str, Any]] = {}
        self.active_threads: Dict[str, ThreadContext] = {}

    async def send_agent_message(
        self,
        sender: str,
        recipient: Union[str, List[str]],
        content: str,
        message_type: CommunicationType = CommunicationType.DIRECT,
        priority: MessagePriority = MessagePriority.MEDIUM,
        thread_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        agency_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send a message between agents"""
        message_id = str(uuid.uuid4())

        # Create message object
        message = AgentMessage(
            message_id=message_id,
            sender=sender,
            recipient=recipient,
            content=content,
            message_type=message_type,
            priority=priority,
            thread_id=thread_id,
            context=context or {},
            metadata={"agency_id": agency_id}
        )

        # Store in history
        self.message_history.append(message)

        try:
            # Route message based on type
            if message_type == CommunicationType.DIRECT:
                result = await self._send_direct_message(message, agency_id)
            elif message_type == CommunicationType.BROADCAST:
                result = await self._send_broadcast_message(message, agency_id)
            elif message_type == CommunicationType.CHAIN:
                result = await self._send_chain_message(message, agency_id)
            elif message_type == CommunicationType.MULTICAST:
                result = await self._send_multicast_message(message, agency_id)
            elif message_type == CommunicationType.ASYNC:
                result = await self._send_async_message(message, agency_id)
            else:
                raise ValueError(f"Unsupported communication type: {message_type}")

            # Store result in communication context
            if thread_id:
                await self._update_thread_context(thread_id, message, result)

            return {
                "success": True,
                "message_id": message_id,
                "result": result,
                "delivered_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to send message {message_id}: {e}")
            return {
                "success": False,
                "message_id": message_id,
                "error": str(e),
                "failed_at": datetime.utcnow().isoformat()
            }

    async def _send_direct_message(self, message: AgentMessage, agency_id: Optional[str]) -> Dict[str, Any]:
        """Send a direct message from one agent to another"""
        if isinstance(message.recipient, list):
            recipient = message.recipient[0]  # Use first recipient for direct
        else:
            recipient = message.recipient

        # Find agency to use
        agency = self._get_agency_for_communication(agency_id, message.sender, recipient)
        if not agency:
            raise ValueError(f"No agency found for communication between {message.sender} and {recipient}")

        # Send message through agency
        response = await agency.send_agent_message(
            sender=message.sender,
            recipient=recipient,
            message=message.content,
            thread_id=message.thread_id
        )

        return {
            "type": "direct_response",
            "recipient": recipient,
            "response": str(response),
            "agency_used": agency.config.name if agency.config else "unknown"
        }

    async def _send_broadcast_message(self, message: AgentMessage, agency_id: Optional[str]) -> Dict[str, Any]:
        """Broadcast a message to multiple recipients"""
        if not isinstance(message.recipient, list):
            recipients = [message.recipient]
        else:
            recipients = message.recipient

        results = {}
        agency_used = None

        for recipient in recipients:
            try:
                # Find agency for this communication
                agency = self._get_agency_for_communication(agency_id, message.sender, recipient)
                if agency:
                    agency_used = agency
                    response = await agency.send_agent_message(
                        sender=message.sender,
                        recipient=recipient,
                        message=message.content,
                        thread_id=message.thread_id
                    )
                    results[recipient] = {
                        "success": True,
                        "response": str(response)
                    }
                else:
                    results[recipient] = {
                        "success": False,
                        "error": f"No agency found for communication"
                    }

            except Exception as e:
                results[recipient] = {
                    "success": False,
                    "error": str(e)
                }

        return {
            "type": "broadcast_response",
            "total_recipients": len(recipients),
            "successful_deliveries": len([r for r in results.values() if r.get("success")]),
            "results": results,
            "agency_used": agency_used.config.name if agency_used else "unknown"
        }

    async def _send_chain_message(self, message: AgentMessage, agency_id: Optional[str]) -> Dict[str, Any]:
        """Send a chain message through a sequence of agents"""
        if not isinstance(message.recipient, list):
            recipients = [message.recipient]
        else:
            recipients = message.recipient

        if len(recipients) < 2:
            raise ValueError("Chain communication requires at least 2 recipients")

        # Find agency for chain communication
        agency = self._get_agency_for_communication(agency_id, message.sender, recipients[0])
        if not agency:
            raise ValueError(f"No agency found for chain communication")

        # Execute chain communication
        chain_results = []
        current_message = message.content
        current_sender = message.sender

        for i, recipient in enumerate(recipients):
            try:
                response = await agency.send_agent_message(
                    sender=current_sender,
                    recipient=recipient,
                    message=current_message,
                    thread_id=message.thread_id
                )

                chain_results.append({
                    "step": i + 1,
                    "sender": current_sender,
                    "recipient": recipient,
                    "response": str(response),
                    "success": True
                })

                # Update for next step in chain
                current_sender = recipient
                current_message = str(response)

            except Exception as e:
                chain_results.append({
                    "step": i + 1,
                    "sender": current_sender,
                    "recipient": recipient,
                    "error": str(e),
                    "success": False
                })
                break  # Stop chain on failure

        return {
            "type": "chain_response",
            "total_steps": len(recipients),
            "completed_steps": len(chain_results),
            "results": chain_results,
            "agency_used": agency.config.name if agency.config else "unknown"
        }

    async def _send_multicast_message(self, message: AgentMessage, agency_id: Optional[str]) -> Dict[str, Any]:
        """Send multicast message with optimized routing"""
        if not isinstance(message.recipient, list):
            recipients = [message.recipient]
        else:
            recipients = message.recipient

        # Group recipients by optimal agencies
        agency_groups = await self._group_recipients_by_agency(recipients, agency_id)

        # Send messages in parallel
        tasks = []
        for agency_id_rec, recips in agency_groups.items():
            task = self._send_to_agency_group(agency_id_rec, message.sender, recips, message.content, message.thread_id)
            tasks.append(task)

        # Execute all sends concurrently
        group_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        all_results = {}
        successful_count = 0

        for i, group_result in enumerate(group_results):
            if isinstance(group_result, Exception):
                logger.error(f"Multicast group {i} failed: {group_result}")
                continue

            all_results.update(group_result.get("results", {}))
            successful_count += group_result.get("successful_deliveries", 0)

        return {
            "type": "multicast_response",
            "total_recipients": len(recipients),
            "successful_deliveries": successful_count,
            "agency_groups": len(agency_groups),
            "results": all_results
        }

    async def _send_async_message(self, message: AgentMessage, agency_id: Optional[str]) -> Dict[str, Any]:
        """Send asynchronous message with callback handling"""
        if not isinstance(message.recipient, list):
            recipients = [message.recipient]
        else:
            recipients = message.recipient

        # Create async tasks for each recipient
        async_tasks = []
        for recipient in recipients:
            task = self._send_async_to_recipient(message.sender, recipient, message.content, message.thread_id, agency_id)
            async_tasks.append(task)

        # Execute with timeout and error handling
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*async_tasks, return_exceptions=True),
                timeout=30.0  # 30 second timeout for async messages
            )
        except asyncio.TimeoutError:
            return {
                "type": "async_response",
                "error": "Async message delivery timed out",
                "total_recipients": len(recipients),
                "successful_deliveries": 0
            }

        # Process results
        delivery_results = {}
        successful_count = 0

        for i, result in enumerate(results):
            recipient = recipients[i]
            if isinstance(result, Exception):
                delivery_results[recipient] = {
                    "success": False,
                    "error": str(result)
                }
            else:
                delivery_results[recipient] = {
                    "success": True,
                    "response": result.get("response", "No response")
                }
                successful_count += 1

        return {
            "type": "async_response",
            "total_recipients": len(recipients),
            "successful_deliveries": successful_count,
            "results": delivery_results
        }

    async def _send_to_agency_group(
        self,
        agency_id: str,
        sender: str,
        recipients: List[str],
        message: str,
        thread_id: Optional[str]
    ) -> Dict[str, Any]:
        """Send message to a group of recipients through the same agency"""
        if agency_id not in self.active_agencies:
            raise ValueError(f"Agency not found: {agency_id}")

        agency = self.active_agencies[agency_id]
        results = {}

        for recipient in recipients:
            try:
                response = await agency.send_agent_message(
                    sender=sender,
                    recipient=recipient,
                    message=message,
                    thread_id=thread_id
                )
                results[recipient] = {
                    "success": True,
                    "response": str(response)
                }
            except Exception as e:
                results[recipient] = {
                    "success": False,
                    "error": str(e)
                }

        return {
            "agency_id": agency_id,
            "results": results,
            "successful_deliveries": len([r for r in results.values() if r.get("success")])
        }

    async def _send_async_to_recipient(
        self,
        sender: str,
        recipient: str,
        message: str,
        thread_id: Optional[str],
        agency_id: Optional[str]
    ) -> Dict[str, Any]:
        """Send async message to a single recipient"""
        agency = self._get_agency_for_communication(agency_id, sender, recipient)
        if not agency:
            raise ValueError(f"No agency found for async communication")

        response = await agency.send_agent_message(
            sender=sender,
            recipient=recipient,
            message=message,
            thread_id=thread_id
        )

        return {
            "recipient": recipient,
            "response": str(response)
        }

    def _get_agency_for_communication(
        self,
        agency_id: Optional[str],
        sender: str,
        recipient: str
    ) -> Optional[ArchonAgency]:
        """Get the appropriate agency for a communication"""
        if agency_id and agency_id in self.active_agencies:
            return self.active_agencies[agency_id]

        # Find agency that has both sender and recipient
        for agency in self.active_agencies.values():
            if (sender in agency.agents and recipient in agency.agents):
                return agency

        # Return first available agency that has sender
        for agency in self.active_agencies.values():
            if sender in agency.agents:
                return agency

        return None

    async def _group_recipients_by_agency(
        self,
        recipients: List[str],
        preferred_agency_id: Optional[str]
    ) -> Dict[str, List[str]]:
        """Group recipients by optimal agency for multicast"""
        groups = {}

        if preferred_agency_id:
            # Try to use preferred agency first
            agency = self.active_agencies.get(preferred_agency_id)
            if agency:
                available_recipients = [r for r in recipients if r in agency.agents]
                if available_recipients:
                    groups[preferred_agency_id] = available_recipients

        # Group remaining recipients by available agencies
        remaining_recipients = [r for r in recipients if r not in groups.get(preferred_agency_id, [])]
        for agency_id, agency in self.active_agencies.items():
            if agency_id == preferred_agency_id:
                continue

            available_recipients = [r for r in remaining_recipients if r in agency.agents]
            if available_recipients:
                groups[agency_id] = available_recipients
                remaining_recipients = [r for r in remaining_recipients if r not in available_recipients]

        return groups

    async def _update_thread_context(
        self,
        thread_id: str,
        message: AgentMessage,
        result: Dict[str, Any]
    ) -> None:
        """Update thread context with communication results"""
        if thread_id not in self.communication_contexts:
            self.communication_contexts[thread_id] = {
                "thread_id": thread_id,
                "messages": [],
                "last_updated": datetime.utcnow()
            }

        context = self.communication_contexts[thread_id]
        context["messages"].append({
            "message_id": message.message_id,
            "sender": message.sender,
            "recipient": message.recipient,
            "content": message.content,
            "timestamp": message.timestamp.isoformat(),
            "result": result
        })
        context["last_updated"] = datetime.utcnow()

    async def get_conversation_history(
        self,
        thread_id: str,
        limit: int = 50,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a thread"""
        if thread_id not in self.communication_contexts:
            return []

        context = self.communication_contexts[thread_id]
        messages = context["messages"]

        # Apply limit
        if limit > 0:
            messages = messages[-limit:]

        # Format messages
        formatted_messages = []
        for msg in messages:
            formatted_msg = {
                "message_id": msg["message_id"],
                "sender": msg["sender"],
                "recipient": msg["recipient"],
                "content": msg["content"],
                "timestamp": msg["timestamp"]
            }

            if include_metadata:
                formatted_msg.update({
                    "result": msg["result"]
                })

            formatted_messages.append(formatted_msg)

        return formatted_messages

    async def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics and analytics"""
        total_messages = len(self.message_history)

        # Count by message type
        type_counts = {}
        for msg in self.message_history:
            msg_type = msg.message_type.value
            type_counts[msg_type] = type_counts.get(msg_type, 0) + 1

        # Count by priority
        priority_counts = {}
        for msg in self.message_history:
            priority = msg.priority.value
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

        # Recent activity (last hour)
        one_hour_ago = datetime.utcnow().timestamp() - 3600
        recent_messages = [
            msg for msg in self.message_history
            if msg.timestamp.timestamp() > one_hour_ago
        ]

        # Active threads
        active_threads = len([
            ctx for ctx in self.communication_contexts.values()
            if (datetime.utcnow() - ctx["last_updated"]).total_seconds() < 3600
        ])

        return {
            "total_messages": total_messages,
            "messages_by_type": type_counts,
            "messages_by_priority": priority_counts,
            "recent_messages_hour": len(recent_messages),
            "active_threads": active_threads,
            "active_agencies": len(self.active_agencies),
            "total_threads": len(self.communication_contexts)
        }

    def register_agency(self, agency_id: str, agency: ArchonAgency) -> None:
        """Register an agency for communication management"""
        self.active_agencies[agency_id] = agency
        logger.info(f"Registered agency for communication: {agency_id}")


# Global communication manager instance
_communication_manager = CommunicationManager()


def register_agency_communication_tools(mcp: FastMCP):
    """Register all agency communication tools with the MCP server."""

    @mcp.tool()
    async def archon_send_agent_message(
        ctx: Context,
        sender: str,
        recipient: Union[str, List[str]],
        message: str,
        message_type: str = "direct",
        priority: str = "medium",
        thread_id: Optional[str] = None,
        context: Optional[str] = None,
        agency_id: Optional[str] = None
    ) -> str:
        """
        Send a message between agents.

        Args:
            sender: Name of the sending agent
            recipient: Name or list of names of receiving agents
            message: Message content to send
            message_type: Type of communication (direct, broadcast, chain, multicast, async)
            priority: Message priority (low, medium, high, urgent)
            thread_id: Optional thread ID for conversation continuity
            context: Optional JSON string with additional context
            agency_id: Optional agency ID to scope communication

        Returns:
            JSON string with message delivery result
        """
        try:
            # Parse message type
            comm_type = CommunicationType(message_type.lower())

            # Parse priority
            msg_priority = MessagePriority(priority.lower())

            # Parse context if provided
            context_dict = {}
            if context:
                context_dict = json.loads(context)

            # Send message
            result = await _communication_manager.send_agent_message(
                sender=sender,
                recipient=recipient,
                content=message,
                message_type=comm_type,
                priority=msg_priority,
                thread_id=thread_id,
                context=context_dict,
                agency_id=agency_id
            )

            mcp_logger.info(f"Agent message sent: {sender} -> {recipient} ({message_type})")

            return json.dumps(result, indent=2)

        except ValueError as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)
        except json.JSONDecodeError as e:
            return json.dumps({
                "success": False,
                "error": f"Invalid JSON in context: {str(e)}"
            }, indent=2)
        except Exception as e:
            logger.error(f"Error sending agent message: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_get_conversation_history(
        ctx: Context,
        thread_id: str,
        limit: int = 50,
        include_metadata: bool = True
    ) -> str:
        """
        Get conversation history for a thread.

        Args:
            thread_id: Thread identifier
            limit: Maximum number of messages to retrieve
            include_metadata: Whether to include message metadata

        Returns:
            JSON string with conversation history
        """
        try:
            history = await _communication_manager.get_conversation_history(
                thread_id=thread_id,
                limit=limit,
                include_metadata=include_metadata
            )

            result = {
                "success": True,
                "thread_id": thread_id,
                "messages": history,
                "total_messages": len(history),
                "retrieved_at": datetime.utcnow().isoformat()
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_broadcast_message(
        ctx: Context,
        sender: str,
        recipients: List[str],
        message: str,
        priority: str = "medium",
        thread_id: Optional[str] = None
    ) -> str:
        """
        Broadcast a message to multiple agents.

        Args:
            sender: Name of the sending agent
            recipients: List of receiving agent names
            message: Message content to broadcast
            priority: Message priority (low, medium, high, urgent)
            thread_id: Optional thread ID

        Returns:
            JSON string with broadcast result
        """
        try:
            msg_priority = MessagePriority(priority.lower())

            result = await _communication_manager.send_agent_message(
                sender=sender,
                recipient=recipients,
                content=message,
                message_type=CommunicationType.BROADCAST,
                priority=msg_priority,
                thread_id=thread_id
            )

            result["action"] = "broadcast"
            mcp_logger.info(f"Broadcast message sent from {sender} to {len(recipients)} recipients")

            return json.dumps(result, indent=2)

        except ValueError as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)
        except Exception as e:
            logger.error(f"Error broadcasting message: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_send_chain_message(
        ctx: Context,
        sender: str,
        chain_recipients: List[str],
        message: str,
        thread_id: Optional[str] = None
    ) -> str:
        """
        Send a chain message through a sequence of agents.

        Args:
            sender: Name of the sending agent
            chain_recipients: List of agent names in chain order
            message: Initial message content
            thread_id: Optional thread ID

        Returns:
            JSON string with chain communication result
        """
        try:
            if len(chain_recipients) < 2:
                return json.dumps({
                    "success": False,
                    "error": "Chain communication requires at least 2 recipients"
                }, indent=2)

            result = await _communication_manager.send_agent_message(
                sender=sender,
                recipient=chain_recipients,
                content=message,
                message_type=CommunicationType.CHAIN,
                priority=MessagePriority.MEDIUM,
                thread_id=thread_id
            )

            result["action"] = "chain_communication"
            mcp_logger.info(f"Chain message sent from {sender} through {len(chain_recipients)} agents")

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error sending chain message: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_create_communication_flow(
        ctx: Context,
        flow_name: str,
        sender: str,
        receivers: Union[str, List[str]],
        flow_type: str = "direct",
        custom_rules: Optional[str] = None
    ) -> str:
        """
        Create a communication flow pattern between agents.

        Args:
            flow_name: Name for the communication flow
            sender: Sending agent name
            receivers: Receiving agent name(s)
            flow_type: Type of flow (direct, broadcast, chain)
            custom_rules: Optional JSON string with custom flow rules

        Returns:
            JSON string with flow creation result
        """
        try:
            # Parse flow type
            comm_flow_type = CommunicationFlowType(flow_type.lower())

            # Parse custom rules if provided
            rules_dict = {}
            if custom_rules:
                rules_dict = json.loads(custom_rules)

            flow_id = str(uuid.uuid4())

            # Create flow configuration
            flow_config = {
                "flow_id": flow_id,
                "flow_name": flow_name,
                "sender": sender,
                "receivers": [receivers] if isinstance(receivers, str) else receivers,
                "flow_type": comm_flow_type.value,
                "custom_rules": rules_dict,
                "created_at": datetime.utcnow().isoformat()
            }

            result = {
                "success": True,
                "flow_id": flow_id,
                "flow_config": flow_config,
                "message": f"Communication flow '{flow_name}' created successfully"
            }

            mcp_logger.info(f"Created communication flow: {flow_name} ({flow_id})")

            return json.dumps(result, indent=2)

        except ValueError as e:
            return json.dumps({
                "success": False,
                "error": f"Invalid flow type: {flow_type}"
            }, indent=2)
        except json.JSONDecodeError as e:
            return json.dumps({
                "success": False,
                "error": f"Invalid JSON in custom_rules: {str(e)}"
            }, indent=2)
        except Exception as e:
            logger.error(f"Error creating communication flow: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_get_communication_stats(
        ctx: Context,
        time_range_hours: int = 24,
        include_agencies: bool = True
    ) -> str:
        """
        Get communication statistics and analytics.

        Args:
            time_range_hours: Time range for statistics (default: 24 hours)
            include_agencies: Whether to include agency-specific statistics

        Returns:
            JSON string with communication statistics
        """
        try:
            stats = await _communication_manager.get_communication_stats()

            # Add time-based filtering
            cutoff_time = datetime.utcnow().timestamp() - (time_range_hours * 3600)
            time_filtered_messages = [
                msg for msg in _communication_manager.message_history
                if msg.timestamp.timestamp() > cutoff_time
            ]

            stats.update({
                "time_range_hours": time_range_hours,
                "messages_in_range": len(time_filtered_messages),
                "messages_per_hour": len(time_filtered_messages) / max(time_range_hours, 1)
            })

            # Add agency-specific stats if requested
            if include_agencies:
                agency_stats = {}
                for agency_id, agency in _communication_manager.active_agencies.items():
                    agency_messages = [
                        msg for msg in time_filtered_messages
                        if msg.metadata and msg.metadata.get("agency_id") == agency_id
                    ]
                    agency_stats[agency_id] = {
                        "messages": len(agency_messages),
                        "agents": len(agency.agents),
                        "communication_flows": len(agency.communication_flows)
                    }

                stats["agency_statistics"] = agency_stats

            result = {
                "success": True,
                "communication_statistics": stats,
                "generated_at": datetime.utcnow().isoformat()
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error getting communication stats: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_manage_communication_context(
        ctx: Context,
        action: str,
        thread_id: Optional[str] = None,
        context_data: Optional[str] = None
    ) -> str:
        """
        Manage communication contexts and thread metadata.

        Args:
            action: Action to perform (create, update, get, delete)
            thread_id: Thread ID to manage
            context_data: JSON string with context data (for create/update)

        Returns:
            JSON string with context management result
        """
        try:
            if action not in ["create", "update", "get", "delete"]:
                return json.dumps({
                    "success": False,
                    "error": f"Invalid action: {action}. Must be one of: create, update, get, delete"
                }, indent=2)

            result_data = {
                "action": action,
                "thread_id": thread_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            if action == "create":
                if not context_data:
                    return json.dumps({
                        "success": False,
                        "error": "context_data is required for create action"
                    }, indent=2)

                thread_id = thread_id or str(uuid.uuid4())
                context_dict = json.loads(context_data)

                _communication_manager.communication_contexts[thread_id] = {
                    "thread_id": thread_id,
                    "context_data": context_dict,
                    "messages": [],
                    "created_at": datetime.utcnow(),
                    "last_updated": datetime.utcnow()
                }

                result_data.update({
                    "thread_id": thread_id,
                    "message": "Communication context created successfully"
                })

            elif action == "update":
                if not thread_id or not context_data:
                    return json.dumps({
                        "success": False,
                        "error": "thread_id and context_data are required for update action"
                    }, indent=2)

                if thread_id not in _communication_manager.communication_contexts:
                    return json.dumps({
                        "success": False,
                        "error": f"Thread not found: {thread_id}"
                    }, indent=2)

                context_dict = json.loads(context_data)
                context = _communication_manager.communication_contexts[thread_id]
                context["context_data"].update(context_dict)
                context["last_updated"] = datetime.utcnow()

                result_data["message"] = "Communication context updated successfully"

            elif action == "get":
                if not thread_id:
                    return json.dumps({
                        "success": False,
                        "error": "thread_id is required for get action"
                    }, indent=2)

                context = _communication_manager.communication_contexts.get(thread_id)
                if not context:
                    return json.dumps({
                        "success": False,
                        "error": f"Thread not found: {thread_id}"
                    }, indent=2)

                result_data.update({
                    "context_data": context.get("context_data", {}),
                    "message_count": len(context.get("messages", [])),
                    "created_at": context.get("created_at").isoformat() if context.get("created_at") else None,
                    "last_updated": context.get("last_updated").isoformat() if context.get("last_updated") else None
                })

            elif action == "delete":
                if not thread_id:
                    return json.dumps({
                        "success": False,
                        "error": "thread_id is required for delete action"
                    }, indent=2)

                if thread_id in _communication_manager.communication_contexts:
                    del _communication_manager.communication_contexts[thread_id]
                    result_data["message"] = "Communication context deleted successfully"
                else:
                    result_data["message"] = f"Thread {thread_id} not found"

            result = {
                "success": True,
                "result": result_data
            }

            return json.dumps(result, indent=2)

        except json.JSONDecodeError as e:
            return json.dumps({
                "success": False,
                "error": f"Invalid JSON in context_data: {str(e)}"
            }, indent=2)
        except Exception as e:
            logger.error(f"Error managing communication context: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    # Log successful registration
    logger.info("✓ Agency communication tools registered")