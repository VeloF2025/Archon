"""
Agency Thread Management MCP Tools for Archon MCP Server

This module provides MCP tools for managing conversation threads including:
- Creating and managing conversation threads
- Thread context management
- Message history retrieval
- Thread lifecycle management
- Thread analytics and insights

Integration with Phase 1 Agency Swarm thread management system.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

from mcp.server.fastmcp import Context, FastMCP

# Add the project root to Python path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import thread management components
from src.agents.orchestration.archon_thread_manager import ArchonThreadManager, ThreadContext
from src.server.config.logfire_config import mcp_logger

logger = logging.getLogger(__name__)


class ThreadStatus(str, Enum):
    """Thread status states"""
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"
    CLOSED = "closed"


class ThreadType(str, Enum):
    """Types of conversation threads"""
    WORKFLOW = "workflow"
    COLLABORATION = "collaboration"
    DEBUGGING = "debugging"
    PLANNING = "planning"
    REVIEW = "review"
    GENERAL = "general"


@dataclass
class ThreadMetadata:
    """Metadata for conversation threads"""
    thread_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    thread_type: ThreadType = ThreadType.GENERAL
    status: ThreadStatus = ThreadStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    message_count: int = 0
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreadMessage:
    """Individual message in a conversation thread"""
    message_id: str
    thread_id: str
    role: str  # user, assistant, system
    sender: str
    recipient: Optional[str]
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThreadManager:
    """Enhanced thread management for agency conversations"""

    def __init__(self):
        self.base_thread_manager = ArchonThreadManager(enable_persistence=True)
        self.thread_metadata: Dict[str, ThreadMetadata] = {}
        self.thread_messages: Dict[str, List[ThreadMessage]] = defaultdict(list)
        self.thread_index: Dict[str, List[str]] = defaultdict(list)  # Various indexes for fast lookup

    async def create_thread(
        self,
        sender: str,
        recipient: str,
        initial_context: Optional[Dict[str, Any]] = None,
        thread_name: Optional[str] = None,
        thread_type: ThreadType = ThreadType.GENERAL,
        description: Optional[str] = None,
        expires_in_hours: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Create a new conversation thread"""
        # Create base thread using ArchonThreadManager
        thread_id = await self.base_thread_manager.create_thread(
            sender=sender,
            recipient=recipient,
            initial_context=initial_context or {}
        )

        # Create enhanced metadata
        metadata = ThreadMetadata(
            thread_id=thread_id,
            name=thread_name or f"Thread {thread_id[:8]}",
            thread_type=thread_type,
            description=description,
            participants=[sender, recipient],
            context=initial_context or {}
        )

        if expires_in_hours:
            metadata.expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)

        if tags:
            metadata.tags.extend(tags)

        self.thread_metadata[thread_id] = metadata

        # Update indexes
        self._update_indexes(thread_id, metadata)

        logger.info(f"Created thread: {thread_id} ({thread_type}) between {sender} and {recipient}")

        return thread_id

    async def add_message(
        self,
        thread_id: str,
        role: str,
        sender: str,
        content: str,
        recipient: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a message to a thread"""
        if thread_id not in self.thread_metadata:
            raise ValueError(f"Thread not found: {thread_id}")

        # Create message
        message_id = str(uuid.uuid4())
        message = ThreadMessage(
            message_id=message_id,
            thread_id=thread_id,
            role=role,
            sender=sender,
            recipient=recipient,
            content=content,
            metadata=metadata or {}
        )

        # Add to thread
        self.thread_messages[thread_id].append(message)

        # Update thread metadata
        metadata = self.thread_metadata[thread_id]
        metadata.message_count += 1
        metadata.updated_at = datetime.utcnow()

        # Add sender to participants if not already there
        if sender not in metadata.participants:
            metadata.participants.append(sender)

        if recipient and recipient not in metadata.participants:
            metadata.participants.append(recipient)

        # Also add to base thread manager for compatibility
        await self.base_thread_manager.add_message(thread_id, {
            "role": role,
            "content": content,
            "sender": sender,
            "recipient": recipient,
            "timestamp": datetime.utcnow().isoformat()
        })

        logger.debug(f"Added message to thread {thread_id}: {sender} -> {recipient or 'all'}")

        return message_id

    async def get_thread(self, thread_id: str) -> ThreadContext:
        """Get a thread context from base manager"""
        return await self.base_thread_manager.get_thread(thread_id)

    async def get_thread_messages(
        self,
        thread_id: str,
        limit: int = 50,
        offset: int = 0,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """Get messages from a thread with pagination"""
        if thread_id not in self.thread_messages:
            return []

        messages = self.thread_messages[thread_id]
        total_messages = len(messages)

        # Apply pagination
        start_idx = min(offset, total_messages)
        end_idx = min(start_idx + limit, total_messages)
        paginated_messages = messages[start_idx:end_idx]

        # Format messages
        formatted_messages = []
        for msg in paginated_messages:
            msg_data = {
                "message_id": msg.message_id,
                "role": msg.role,
                "sender": msg.sender,
                "recipient": msg.recipient,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }

            if include_metadata and msg.metadata:
                msg_data["metadata"] = msg.metadata

            formatted_messages.append(msg_data)

        return formatted_messages

    async def get_thread_metadata(self, thread_id: str) -> Optional[ThreadMetadata]:
        """Get thread metadata"""
        return self.thread_metadata.get(thread_id)

    async def list_threads(
        self,
        thread_type: Optional[ThreadType] = None,
        status: Optional[ThreadStatus] = None,
        participant: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 100,
        include_expired: bool = False
    ) -> List[Dict[str, Any]]:
        """List threads with filtering"""
        filtered_threads = []

        for thread_id, metadata in self.thread_metadata.items():
            # Apply filters
            if thread_type and metadata.thread_type != thread_type:
                continue
            if status and metadata.status != status:
                continue
            if participant and participant not in metadata.participants:
                continue
            if tag and tag not in metadata.tags:
                continue
            if not include_expired and metadata.expires_at and metadata.expires_at < datetime.utcnow():
                continue

            filtered_threads.append(metadata)

        # Sort by updated_at (most recent first)
        filtered_threads.sort(key=lambda m: m.updated_at, reverse=True)

        # Apply limit
        filtered_threads = filtered_threads[:limit]

        # Format for output
        formatted_threads = []
        for metadata in filtered_threads:
            thread_data = {
                "thread_id": metadata.thread_id,
                "name": metadata.name,
                "thread_type": metadata.thread_type.value,
                "status": metadata.status.value,
                "participants": metadata.participants,
                "message_count": metadata.message_count,
                "created_at": metadata.created_at.isoformat(),
                "updated_at": metadata.updated_at.isoformat()
            }

            if metadata.description:
                thread_data["description"] = metadata.description
            if metadata.expires_at:
                thread_data["expires_at"] = metadata.expires_at.isoformat()
            if metadata.tags:
                thread_data["tags"] = metadata.tags

            formatted_threads.append(thread_data)

        return formatted_threads

    async def update_thread_metadata(
        self,
        thread_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """Update thread metadata"""
        if thread_id not in self.thread_metadata:
            raise ValueError(f"Thread not found: {thread_id}")

        metadata = self.thread_metadata[thread_id]

        # Update fields
        if "name" in updates:
            metadata.name = updates["name"]
        if "description" in updates:
            metadata.description = updates["description"]
        if "status" in updates:
            metadata.status = ThreadStatus(updates["status"])
        if "thread_type" in updates:
            metadata.thread_type = ThreadType(updates["thread_type"])
        if "expires_in_hours" in updates:
            metadata.expires_at = datetime.utcnow() + timedelta(hours=updates["expires_in_hours"])
        if "tags" in updates:
            metadata.tags = updates["tags"]
        if "context" in updates:
            metadata.context.update(updates["context"])

        metadata.updated_at = datetime.utcnow()

        # Update indexes
        self._update_indexes(thread_id, metadata)

        logger.info(f"Updated thread metadata: {thread_id}")

    async def delete_thread(self, thread_id: str) -> None:
        """Delete a thread"""
        if thread_id not in self.thread_metadata:
            raise ValueError(f"Thread not found: {thread_id}")

        # Remove from storage
        del self.thread_metadata[thread_id]
        if thread_id in self.thread_messages:
            del self.thread_messages[thread_id]

        # Remove from indexes
        self._remove_from_indexes(thread_id)

        # Also remove from base manager
        try:
            await self.base_thread_manager.delete_thread(thread_id)
        except Exception as e:
            logger.warning(f"Failed to delete from base thread manager: {e}")

        logger.info(f"Deleted thread: {thread_id}")

    async def search_threads(
        self,
        query: str,
        search_content: bool = True,
        search_metadata: bool = True,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search threads by content or metadata"""
        query_lower = query.lower()
        matching_threads = []

        for thread_id, metadata in self.thread_metadata.items():
            match_score = 0

            # Search in metadata
            if search_metadata:
                if metadata.name and query_lower in metadata.name.lower():
                    match_score += 3
                if metadata.description and query_lower in metadata.description.lower():
                    match_score += 2
                if any(query_lower in tag.lower() for tag in metadata.tags):
                    match_score += 1

            # Search in message content
            if search_content and thread_id in self.thread_messages:
                for msg in self.thread_messages[thread_id]:
                    if query_lower in msg.content.lower():
                        match_score += 1

            if match_score > 0:
                # Create thread info with match score
                thread_info = {
                    "thread_id": thread_id,
                    "name": metadata.name,
                    "match_score": match_score,
                    "thread_type": metadata.thread_type.value,
                    "participants": metadata.participants,
                    "message_count": metadata.message_count,
                    "updated_at": metadata.updated_at.isoformat()
                }
                matching_threads.append((match_score, thread_info))

        # Sort by match score (highest first)
        matching_threads.sort(key=lambda x: x[0], reverse=True)

        # Return top matches
        result = [thread_info for _, thread_info in matching_threads[:limit]]

        return result

    async def get_thread_analytics(
        self,
        thread_id: Optional[str] = None,
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """Get thread analytics and statistics"""
        cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)

        analytics = {
            "time_range_hours": time_range_hours,
            "total_threads": len(self.thread_metadata),
            "total_messages": sum(len(messages) for messages in self.thread_messages.values()),
            "threads_by_type": defaultdict(int),
            "threads_by_status": defaultdict(int),
            "most_active_threads": [],
            "recent_activity": []
        }

        # Analyze threads
        for metadata in self.thread_metadata.values():
            analytics["threads_by_type"][metadata.thread_type.value] += 1
            analytics["threads_by_status"][metadata.status.value] += 1

            # Check for recent activity
            if metadata.updated_at > cutoff_time:
                analytics["recent_activity"].append({
                    "thread_id": metadata.thread_id,
                    "name": metadata.name,
                    "last_activity": metadata.updated_at.isoformat(),
                    "message_count": metadata.message_count
                })

        # Find most active threads (by message count)
        thread_activity = [
            (thread_id, len(messages))
            for thread_id, messages in self.thread_messages.items()
        ]
        thread_activity.sort(key=lambda x: x[1], reverse=True)

        for thread_id, message_count in thread_activity[:10]:  # Top 10
            metadata = self.thread_metadata.get(thread_id)
            if metadata:
                analytics["most_active_threads"].append({
                    "thread_id": thread_id,
                    "name": metadata.name,
                    "message_count": message_count,
                    "participants": len(metadata.participants)
                })

        # Sort recent activity by time
        analytics["recent_activity"].sort(key=lambda x: x["last_activity"], reverse=True)

        return analytics

    def _update_indexes(self, thread_id: str, metadata: ThreadMetadata) -> None:
        """Update search indexes for a thread"""
        # Index by type
        self.thread_index[f"type:{metadata.thread_type.value}"].append(thread_id)

        # Index by status
        self.thread_index[f"status:{metadata.status.value}"].append(thread_id)

        # Index by participants
        for participant in metadata.participants:
            self.thread_index[f"participant:{participant}"].append(thread_id)

        # Index by tags
        for tag in metadata.tags:
            self.thread_index[f"tag:{tag}"].append(thread_id)

    def _remove_from_indexes(self, thread_id: str) -> None:
        """Remove thread from all indexes"""
        for key in list(self.thread_index.keys()):
            if thread_id in self.thread_index[key]:
                self.thread_index[key].remove(thread_id)
                # Remove empty index lists
                if not self.thread_index[key]:
                    del self.thread_index[key]

    async def cleanup_expired_threads(self) -> int:
        """Clean up expired threads and return count of cleaned threads"""
        current_time = datetime.utcnow()
        expired_threads = []

        for thread_id, metadata in self.thread_metadata.items():
            if metadata.expires_at and metadata.expires_at < current_time:
                expired_threads.append(thread_id)

        for thread_id in expired_threads:
            await self.delete_thread(thread_id)

        logger.info(f"Cleaned up {len(expired_threads)} expired threads")
        return len(expired_threads)


# Global thread manager instance
_thread_manager = ThreadManager()


def register_agency_thread_tools(mcp: FastMCP):
    """Register all agency thread management tools with the MCP server."""

    @mcp.tool()
    async def archon_create_conversation_thread(
        ctx: Context,
        sender: str,
        recipient: str,
        initial_context: Optional[str] = None,
        thread_name: Optional[str] = None,
        thread_type: str = "general",
        description: Optional[str] = None,
        expires_in_hours: Optional[int] = None,
        tags: Optional[str] = None
    ) -> str:
        """
        Create a new conversation thread between agents.

        Args:
            sender: Name of the sending agent
            recipient: Name of the receiving agent
            initial_context: Optional JSON string with initial context
            thread_name: Optional name for the thread
            thread_type: Type of thread (workflow, collaboration, debugging, planning, review, general)
            description: Optional description of the thread purpose
            expires_in_hours: Optional hours until thread expires
            tags: Optional comma-separated list of tags

        Returns:
            JSON string with thread creation result
        """
        try:
            # Parse thread type
            thread_type_enum = ThreadType(thread_type.lower())

            # Parse context
            context_dict = {}
            if initial_context:
                context_dict = json.loads(initial_context)

            # Parse tags
            tag_list = []
            if tags:
                tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

            # Create thread
            thread_id = await _thread_manager.create_thread(
                sender=sender,
                recipient=recipient,
                initial_context=context_dict,
                thread_name=thread_name,
                thread_type=thread_type_enum,
                description=description,
                expires_in_hours=expires_in_hours,
                tags=tag_list
            )

            result = {
                "success": True,
                "thread_id": thread_id,
                "sender": sender,
                "recipient": recipient,
                "thread_type": thread_type,
                "created_at": datetime.utcnow().isoformat(),
                "message": f"Conversation thread '{thread_name or thread_id[:8]}' created successfully"
            }

            mcp_logger.info(f"Created conversation thread: {thread_id} between {sender} and {recipient}")

            return json.dumps(result, indent=2)

        except ValueError as e:
            return json.dumps({
                "success": False,
                "error": f"Invalid thread type: {thread_type}. Valid types: {', '.join(t.value for t in ThreadType)}"
            }, indent=2)
        except json.JSONDecodeError as e:
            return json.dumps({
                "success": False,
                "error": f"Invalid JSON in initial_context: {str(e)}"
            }, indent=2)
        except Exception as e:
            logger.error(f"Error creating conversation thread: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_add_thread_message(
        ctx: Context,
        thread_id: str,
        role: str,
        sender: str,
        content: str,
        recipient: Optional[str] = None,
        metadata: Optional[str] = None
    ) -> str:
        """
        Add a message to a conversation thread.

        Args:
            thread_id: ID of the thread to add message to
            role: Message role (user, assistant, system)
            sender: Name of the message sender
            content: Message content
            recipient: Optional name of the message recipient
            metadata: Optional JSON string with message metadata

        Returns:
            JSON string with message addition result
        """
        try:
            # Parse metadata
            metadata_dict = {}
            if metadata:
                metadata_dict = json.loads(metadata)

            # Add message
            message_id = await _thread_manager.add_message(
                thread_id=thread_id,
                role=role,
                sender=sender,
                content=content,
                recipient=recipient,
                metadata=metadata_dict
            )

            result = {
                "success": True,
                "message_id": message_id,
                "thread_id": thread_id,
                "sender": sender,
                "role": role,
                "added_at": datetime.utcnow().isoformat()
            }

            return json.dumps(result, indent=2)

        except ValueError as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)
        except json.JSONDecodeError as e:
            return json.dumps({
                "success": False,
                "error": f"Invalid JSON in metadata: {str(e)}"
            }, indent=2)
        except Exception as e:
            logger.error(f"Error adding thread message: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_get_thread_messages(
        ctx: Context,
        thread_id: str,
        limit: int = 50,
        offset: int = 0,
        include_metadata: bool = True
    ) -> str:
        """
        Get messages from a conversation thread.

        Args:
            thread_id: ID of the thread
            limit: Maximum number of messages to return
            offset: Number of messages to skip (for pagination)
            include_metadata: Whether to include message metadata

        Returns:
            JSON string with thread messages
        """
        try:
            messages = await _thread_manager.get_thread_messages(
                thread_id=thread_id,
                limit=limit,
                offset=offset,
                include_metadata=include_metadata
            )

            # Get thread metadata
            metadata = await _thread_manager.get_thread_metadata(thread_id)

            result = {
                "success": True,
                "thread_id": thread_id,
                "messages": messages,
                "total_messages": len(_thread_manager.thread_messages.get(thread_id, [])),
                "returned_messages": len(messages),
                "limit": limit,
                "offset": offset,
                "thread_info": {
                    "name": metadata.name if metadata else None,
                    "participants": metadata.participants if metadata else [],
                    "thread_type": metadata.thread_type.value if metadata else None
                } if metadata else None,
                "retrieved_at": datetime.utcnow().isoformat()
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error getting thread messages: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_list_threads(
        ctx: Context,
        thread_type: Optional[str] = None,
        status: Optional[str] = None,
        participant: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 100,
        include_expired: bool = False
    ) -> str:
        """
        List conversation threads with filtering options.

        Args:
            thread_type: Filter by thread type
            status: Filter by thread status
            participant: Filter by participant name
            tag: Filter by tag
            limit: Maximum number of threads to return
            include_expired: Whether to include expired threads

        Returns:
            JSON string with list of threads
        """
        try:
            # Parse filters
            thread_type_enum = None
            if thread_type:
                thread_type_enum = ThreadType(thread_type.lower())

            status_enum = None
            if status:
                status_enum = ThreadStatus(status.lower())

            # Get threads
            threads = await _thread_manager.list_threads(
                thread_type=thread_type_enum,
                status=status_enum,
                participant=participant,
                tag=tag,
                limit=limit,
                include_expired=include_expired
            )

            result = {
                "success": True,
                "threads": threads,
                "total_threads": len(threads),
                "filters": {
                    "thread_type": thread_type,
                    "status": status,
                    "participant": participant,
                    "tag": tag
                },
                "retrieved_at": datetime.utcnow().isoformat()
            }

            return json.dumps(result, indent=2)

        except ValueError as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)
        except Exception as e:
            logger.error(f"Error listing threads: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_update_thread_metadata(
        ctx: Context,
        thread_id: str,
        updates: str
    ) -> str:
        """
        Update conversation thread metadata.

        Args:
            thread_id: ID of the thread to update
            updates: JSON string with metadata updates

        Returns:
            JSON string with update result
        """
        try:
            # Parse updates
            update_dict = json.loads(updates)

            # Update thread metadata
            await _thread_manager.update_thread_metadata(thread_id, update_dict)

            result = {
                "success": True,
                "thread_id": thread_id,
                "updates_applied": list(update_dict.keys()),
                "updated_at": datetime.utcnow().isoformat()
            }

            return json.dumps(result, indent=2)

        except json.JSONDecodeError as e:
            return json.dumps({
                "success": False,
                "error": f"Invalid JSON in updates: {str(e)}"
            }, indent=2)
        except Exception as e:
            logger.error(f"Error updating thread metadata: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_delete_thread(
        ctx: Context,
        thread_id: str
    ) -> str:
        """
        Delete a conversation thread.

        Args:
            thread_id: ID of the thread to delete

        Returns:
            JSON string with deletion result
        """
        try:
            # Get thread info before deletion for response
            metadata = await _thread_manager.get_thread_metadata(thread_id)
            thread_name = metadata.name if metadata else thread_id

            # Delete thread
            await _thread_manager.delete_thread(thread_id)

            result = {
                "success": True,
                "thread_id": thread_id,
                "thread_name": thread_name,
                "deleted_at": datetime.utcnow().isoformat()
            }

            mcp_logger.info(f"Deleted conversation thread: {thread_id} ({thread_name})")

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error deleting thread: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_search_threads(
        ctx: Context,
        query: str,
        search_content: bool = True,
        search_metadata: bool = True,
        limit: int = 50
    ) -> str:
        """
        Search conversation threads by content or metadata.

        Args:
            query: Search query string
            search_content: Whether to search in message content
            search_metadata: Whether to search in thread metadata
            limit: Maximum number of results to return

        Returns:
            JSON string with search results
        """
        try:
            # Search threads
            results = await _thread_manager.search_threads(
                query=query,
                search_content=search_content,
                search_metadata=search_metadata,
                limit=limit
            )

            result = {
                "success": True,
                "query": query,
                "results": results,
                "total_results": len(results),
                "search_options": {
                    "search_content": search_content,
                    "search_metadata": search_metadata,
                    "limit": limit
                },
                "searched_at": datetime.utcnow().isoformat()
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error searching threads: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_get_thread_analytics(
        ctx: Context,
        thread_id: Optional[str] = None,
        time_range_hours: int = 24
    ) -> str:
        """
        Get conversation thread analytics and statistics.

        Args:
            thread_id: Optional filter by specific thread ID
            time_range_hours: Time range in hours for analytics (default: 24)

        Returns:
            JSON string with thread analytics
        """
        try:
            # Get analytics
            analytics = await _thread_manager.get_thread_analytics(
                thread_id=thread_id,
                time_range_hours=time_range_hours
            )

            # If specific thread requested, add detailed info
            if thread_id:
                metadata = await _thread_manager.get_thread_metadata(thread_id)
                messages = await _thread_manager.get_thread_messages(
                    thread_id=thread_id,
                    limit=1000,
                    include_metadata=False
                )

                if metadata:
                    analytics["thread_details"] = {
                        "thread_id": thread_id,
                        "name": metadata.name,
                        "description": metadata.description,
                        "thread_type": metadata.thread_type.value,
                        "status": metadata.status.value,
                        "participants": metadata.participants,
                        "message_count": len(messages),
                        "created_at": metadata.created_at.isoformat(),
                        "updated_at": metadata.updated_at.isoformat(),
                        "tags": metadata.tags
                    }

                    # Analyze message patterns
                    if messages:
                        senders = defaultdict(int)
                        roles = defaultdict(int)

                        for msg in messages:
                            senders[msg["sender"]] += 1
                            roles[msg["role"]] += 1

                        analytics["message_patterns"] = {
                            "senders": dict(senders),
                            "roles": dict(roles),
                            "average_messages_per_hour": len(messages) / max(time_range_hours, 1)
                        }

            result = {
                "success": True,
                "analytics": analytics,
                "generated_at": datetime.utcnow().isoformat()
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error getting thread analytics: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def archon_cleanup_expired_threads(
        ctx: Context
    ) -> str:
        """
        Clean up expired conversation threads.

        Returns:
            JSON string with cleanup results
        """
        try:
            # Clean up expired threads
            cleaned_count = await _thread_manager.cleanup_expired_threads()

            result = {
                "success": True,
                "cleaned_threads_count": cleaned_count,
                "remaining_threads": len(_thread_manager.thread_metadata),
                "cleaned_at": datetime.utcnow().isoformat()
            }

            mcp_logger.info(f"Cleaned up {cleaned_count} expired threads")

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error cleaning up expired threads: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    # Log successful registration
    logger.info("✓ Agency thread management tools registered")