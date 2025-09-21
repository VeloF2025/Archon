"""
Archon Thread Manager - Enhanced conversation persistence and isolation.

This module provides thread management for agent conversations, building upon
Agency Swarm's threading concepts but enhanced for Archon's enterprise requirements.

Key Features:
- Thread isolation for concurrent conversations
- Persistent conversation storage
- Conversation context management
- Integration with Archon's database systems
- Support for both synchronous and asynchronous operations
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ThreadStatus(Enum):
    """Status of a conversation thread."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    ERROR = "error"


@dataclass
class ThreadMessage:
    """Represents a single message in a conversation thread."""
    role: str  # "user", "assistant", "system"
    content: str
    sender: str  # Agent name or "external"
    recipient: str  # Agent name or "external"
    timestamp: str
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for storage."""
        return {
            "role": self.role,
            "content": self.content,
            "sender": self.sender,
            "recipient": self.recipient,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThreadMessage":
        """Create message from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            sender=data["sender"],
            recipient=data["recipient"],
            timestamp=data["timestamp"],
            message_id=data.get("message_id", str(uuid.uuid4())),
            metadata=data.get("metadata", {})
        )


@dataclass
class ThreadContext:
    """Context for a conversation thread."""
    thread_id: str
    sender: str
    recipient: str
    status: ThreadStatus = ThreadStatus.ACTIVE
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    messages: List[ThreadMessage] = field(default_factory=list)
    context_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: ThreadMessage) -> None:
        """Add a message to the thread."""
        self.messages.append(message)
        self.updated_at = datetime.utcnow().isoformat()

    async def add_message_async(self, message: ThreadMessage) -> None:
        """Asynchronously add a message to the thread."""
        self.messages.append(message)
        self.updated_at = datetime.utcnow().isoformat()

    def get_recent_messages(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent messages from the thread."""
        return [msg.to_dict() for msg in self.messages[-limit:]]

    def get_message_count(self) -> int:
        """Get total number of messages in the thread."""
        return len(self.messages)

    def update_context(self, context_data: Dict[str, Any]) -> None:
        """Update the thread context data."""
        self.context_data.update(context_data)
        self.updated_at = datetime.utcnow().isoformat()

    def set_status(self, status: ThreadStatus) -> None:
        """Set the thread status."""
        self.status = status
        self.updated_at = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert thread context to dictionary for storage."""
        return {
            "thread_id": self.thread_id,
            "sender": self.sender,
            "recipient": self.recipient,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [msg.to_dict() for msg in self.messages],
            "context_data": self.context_data,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThreadContext":
        """Create thread context from dictionary."""
        thread = cls(
            thread_id=data["thread_id"],
            sender=data["sender"],
            recipient=data["recipient"],
            status=ThreadStatus(data["status"]),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            context_data=data.get("context_data", {}),
            metadata=data.get("metadata", {})
        )

        # Load messages
        for msg_data in data.get("messages", []):
            thread.messages.append(ThreadMessage.from_dict(msg_data))

        return thread


class ThreadStorage:
    """Abstract base class for thread storage implementations."""

    async def save_thread(self, thread: ThreadContext) -> bool:
        """Save a thread to storage."""
        raise NotImplementedError

    async def load_thread(self, thread_id: str) -> Optional[ThreadContext]:
        """Load a thread from storage."""
        raise NotImplementedError

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread from storage."""
        raise NotImplementedError

    async def list_threads(
        self,
        sender: Optional[str] = None,
        recipient: Optional[str] = None,
        status: Optional[ThreadStatus] = None,
        limit: int = 100
    ) -> List[ThreadContext]:
        """List threads with optional filtering."""
        raise NotImplementedError

    async def cleanup_old_threads(self, days_old: int = 30) -> int:
        """Clean up threads older than specified days."""
        raise NotImplementedError


class InMemoryThreadStorage(ThreadStorage):
    """In-memory thread storage implementation for development and testing."""

    def __init__(self):
        self.threads: Dict[str, ThreadContext] = {}
        self._lock = threading.Lock()

    async def save_thread(self, thread: ThreadContext) -> bool:
        """Save a thread to memory."""
        with self._lock:
            self.threads[thread.thread_id] = thread
            logger.debug(f"Saved thread {thread.thread_id} to memory")
            return True

    async def load_thread(self, thread_id: str) -> Optional[ThreadContext]:
        """Load a thread from memory."""
        with self._lock:
            thread = self.threads.get(thread_id)
            if thread:
                logger.debug(f"Loaded thread {thread_id} from memory")
            return thread

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread from memory."""
        with self._lock:
            if thread_id in self.threads:
                del self.threads[thread_id]
                logger.debug(f"Deleted thread {thread_id} from memory")
                return True
            return False

    async def list_threads(
        self,
        sender: Optional[str] = None,
        recipient: Optional[str] = None,
        status: Optional[ThreadStatus] = None,
        limit: int = 100
    ) -> List[ThreadContext]:
        """List threads with optional filtering."""
        with self._lock:
            threads = list(self.threads.values())

            # Apply filters
            if sender:
                threads = [t for t in threads if t.sender == sender]
            if recipient:
                threads = [t for t in threads if t.recipient == recipient]
            if status:
                threads = [t for t in threads if t.status == status]

            # Sort by creation time (newest first)
            threads.sort(key=lambda t: t.created_at, reverse=True)

            return threads[:limit]

    async def cleanup_old_threads(self, days_old: int = 30) -> int:
        """Clean up threads older than specified days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        count = 0

        with self._lock:
            threads_to_delete = [
                thread for thread in self.threads.values()
                if datetime.fromisoformat(thread.created_at) < cutoff_date
            ]

            for thread in threads_to_delete:
                del self.threads[thread.thread_id]
                count += 1

        logger.info(f"Cleaned up {count} old threads from memory")
        return count


class DatabaseThreadStorage(ThreadStorage):
    """Database-backed thread storage for production use."""

    def __init__(self, db_connection_string: str):
        self.db_connection_string = db_connection_string
        self._connection_pool = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the database connection and tables."""
        # This would typically initialize connection to PostgreSQL/Supabase
        # For now, we'll implement a placeholder that logs the operations
        self._initialized = True
        logger.info("Database thread storage initialized (placeholder implementation)")

    async def save_thread(self, thread: ThreadContext) -> bool:
        """Save a thread to database."""
        if not self._initialized:
            await self.initialize()

        # Implementation would save to database
        logger.info(f"Saving thread {thread.thread_id} to database")
        return True

    async def load_thread(self, thread_id: str) -> Optional[ThreadContext]:
        """Load a thread from database."""
        if not self._initialized:
            await self.initialize()

        # Implementation would load from database
        logger.info(f"Loading thread {thread_id} from database")
        return None  # Placeholder

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread from database."""
        if not self._initialized:
            await self.initialize()

        # Implementation would delete from database
        logger.info(f"Deleting thread {thread_id} from database")
        return True

    async def list_threads(
        self,
        sender: Optional[str] = None,
        recipient: Optional[str] = None,
        status: Optional[ThreadStatus] = None,
        limit: int = 100
    ) -> List[ThreadContext]:
        """List threads from database with optional filtering."""
        if not self._initialized:
            await self.initialize()

        # Implementation would query database
        logger.info("Listing threads from database")
        return []

    async def cleanup_old_threads(self, days_old: int = 30) -> int:
        """Clean up old threads from database."""
        if not self._initialized:
            await self.initialize()

        # Implementation would clean up database
        logger.info(f"Cleaning up threads older than {days_old} days from database")
        return 0


class ArchonThreadManager:
    """
    Enhanced thread manager for Archon agency conversations.

    Provides thread isolation, persistence, and conversation context management.
    """

    def __init__(
        self,
        storage: Optional[ThreadStorage] = None,
        enable_persistence: bool = True,
        max_threads_per_agent: int = 1000,
        thread_timeout: int = 3600,  # 1 hour
        cleanup_interval: int = 300,  # 5 minutes
        enable_cleanup: bool = True
    ):
        """
        Initialize the thread manager.

        Args:
            storage: Thread storage implementation
            enable_persistence: Whether to persist threads
            max_threads_per_agent: Maximum threads per agent pair
            thread_timeout: Thread timeout in seconds
            cleanup_interval: Cleanup interval in seconds
            enable_cleanup: Whether to enable automatic cleanup
        """
        self.storage = storage or InMemoryThreadStorage()
        self.enable_persistence = enable_persistence
        self.max_threads_per_agent = max_threads_per_agent
        self.thread_timeout = thread_timeout
        self.cleanup_interval = cleanup_interval
        self.enable_cleanup = enable_cleanup

        # Active threads cache
        self._active_threads: Dict[str, ThreadContext] = {}
        self._lock = asyncio.Lock()

        # Thread cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None

        logger.info("Archon Thread Manager initialized")

    async def start(self) -> None:
        """Start the thread manager and background cleanup task."""
        if self.enable_cleanup:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Thread manager cleanup task started")

    async def stop(self) -> None:
        """Stop the thread manager and cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Persist all active threads
        if self.enable_persistence:
            await self._persist_all_threads()

        logger.info("Thread manager stopped")

    async def create_thread(
        self,
        sender: str,
        recipient: str,
        initial_context: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None
    ) -> str:
        """
        Create a new conversation thread.

        Args:
            sender: Sender identifier
            recipient: Recipient identifier
            initial_context: Initial context data
            thread_id: Optional specific thread ID

        Returns:
            Thread ID
        """
        thread_id = thread_id or str(uuid.uuid4())

        thread = ThreadContext(
            thread_id=thread_id,
            sender=sender,
            recipient=recipient,
            context_data=initial_context or {}
        )

        async with self._lock:
            # Check thread limit
            agent_pair_key = f"{sender}:{recipient}"
            existing_threads = [
                t for t in self._active_threads.values()
                if f"{t.sender}:{t.recipient}" == agent_pair_key
            ]

            if len(existing_threads) >= self.max_threads_per_agent:
                # Remove oldest thread
                oldest_thread = min(existing_threads, key=lambda t: t.created_at)
                del self._active_threads[oldest_thread.thread_id]
                logger.info(f"Removed oldest thread {oldest_thread.thread_id} due to limit")

            # Store thread
            self._active_threads[thread_id] = thread

        # Persist if enabled
        if self.enable_persistence:
            await self.storage.save_thread(thread)

        logger.info(f"Created thread {thread_id} between {sender} and {recipient}")
        return thread_id

    async def get_thread(self, thread_id: str) -> ThreadContext:
        """
        Get a thread by ID.

        Args:
            thread_id: Thread identifier

        Returns:
            Thread context

        Raises:
            ValueError: If thread not found
        """
        # Check active threads first
        async with self._lock:
            if thread_id in self._active_threads:
                thread = self._active_threads[thread_id]
                thread.updated_at = datetime.utcnow().isoformat()  # Update access time
                return thread

        # Try loading from storage
        if self.enable_persistence:
            thread = await self.storage.load_thread(thread_id)
            if thread:
                # Add to active threads
                async with self._lock:
                    self._active_threads[thread_id] = thread
                return thread

        raise ValueError(f"Thread {thread_id} not found")

    async def update_thread(
        self,
        thread_id: str,
        messages: Optional[List[ThreadMessage]] = None,
        context_data: Optional[Dict[str, Any]] = None,
        status: Optional[ThreadStatus] = None
    ) -> bool:
        """
        Update a thread.

        Args:
            thread_id: Thread identifier
            messages: Messages to add
            context_data: Context data to update
            status: New status

        Returns:
            True if successful, False if thread not found
        """
        try:
            thread = await self.get_thread(thread_id)

            if messages:
                for message in messages:
                    thread.add_message(message)

            if context_data:
                thread.update_context(context_data)

            if status:
                thread.set_status(status)

            # Persist if enabled
            if self.enable_persistence:
                await self.storage.save_thread(thread)

            return True

        except ValueError:
            logger.warning(f"Failed to update thread {thread_id}: thread not found")
            return False

    async def delete_thread(self, thread_id: str) -> bool:
        """
        Delete a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            True if successful, False if thread not found
        """
        try:
            # Remove from active threads
            async with self._lock:
                if thread_id in self._active_threads:
                    del self._active_threads[thread_id]

            # Remove from storage
            if self.enable_persistence:
                await self.storage.delete_thread(thread_id)

            logger.info(f"Deleted thread {thread_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting thread {thread_id}: {e}")
            return False

    async def list_threads(
        self,
        sender: Optional[str] = None,
        recipient: Optional[str] = None,
        status: Optional[ThreadStatus] = None,
        active_only: bool = False,
        limit: int = 100
    ) -> List[ThreadContext]:
        """
        List threads with optional filtering.

        Args:
            sender: Filter by sender
            recipient: Filter by recipient
            status: Filter by status
            active_only: Only return active threads
            limit: Maximum number of threads to return

        Returns:
            List of thread contexts
        """
        if active_only:
            # Return only active threads
            async with self._lock:
                threads = list(self._active_threads.values())
        else:
            # Return all threads from storage
            if self.enable_persistence:
                threads = await self.storage.list_threads(sender, recipient, status, limit)
            else:
                async with self._lock:
                    threads = list(self._active_threads.values())

        # Apply filters
        if sender:
            threads = [t for t in threads if t.sender == sender]
        if recipient:
            threads = [t for t in threads if t.recipient == recipient]
        if status:
            threads = [t for t in threads if t.status == status]

        # Sort by creation time (newest first)
        threads.sort(key=lambda t: t.created_at, reverse=True)

        return threads[:limit]

    async def get_thread_statistics(self) -> Dict[str, Any]:
        """
        Get thread management statistics.

        Returns:
            Dictionary with thread statistics
        """
        async with self._lock:
            active_count = len(self._active_threads)
            status_counts = {}
            for thread in self._active_threads.values():
                status = thread.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

        if self.enable_persistence:
            total_threads = await self.storage.list_threads()
            total_count = len(total_threads)
        else:
            total_count = active_count

        return {
            "active_threads": active_count,
            "total_threads": total_count,
            "status_distribution": status_counts,
            "max_threads_per_agent": self.max_threads_per_agent,
            "persistence_enabled": self.enable_persistence
        }

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop for expired threads."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired_threads()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in thread cleanup loop: {e}")

    async def _cleanup_expired_threads(self) -> None:
        """Clean up expired threads."""
        current_time = datetime.utcnow()
        expired_threads = []

        async with self._lock:
            for thread_id, thread in self._active_threads.items():
                thread_time = datetime.fromisoformat(thread.updated_at)
                if (current_time - thread_time).total_seconds() > self.thread_timeout:
                    expired_threads.append((thread_id, thread))

        # Remove expired threads
        for thread_id, thread in expired_threads:
            async with self._lock:
                if thread_id in self._active_threads:
                    del self._active_threads[thread_id]

            logger.debug(f"Removed expired thread {thread_id}")

        if expired_threads:
            logger.info(f"Cleaned up {len(expired_threads)} expired threads")

    async def _persist_all_threads(self) -> None:
        """Persist all active threads to storage."""
        if not self.enable_persistence:
            return

        async with self._lock:
            threads_to_persist = list(self._active_threads.values())

        for thread in threads_to_persist:
            try:
                await self.storage.save_thread(thread)
            except Exception as e:
                logger.error(f"Error persisting thread {thread.thread_id}: {e}")

    async def archive_thread(self, thread_id: str) -> bool:
        """
        Archive a thread (mark as completed and move to storage).

        Args:
            thread_id: Thread identifier

        Returns:
            True if successful
        """
        try:
            thread = await self.get_thread(thread_id)
            thread.set_status(ThreadStatus.COMPLETED)

            # Remove from active threads but keep in storage
            async with self._lock:
                if thread_id in self._active_threads:
                    del self._active_threads[thread_id]

            if self.enable_persistence:
                await self.storage.save_thread(thread)

            logger.info(f"Archived thread {thread_id}")
            return True

        except Exception as e:
            logger.error(f"Error archiving thread {thread_id}: {e}")
            return False


# Factory functions for creating thread managers
def create_in_memory_thread_manager(**kwargs) -> ArchonThreadManager:
    """Create a thread manager with in-memory storage."""
    return ArchonThreadManager(storage=InMemoryThreadStorage(), **kwargs)


def create_database_thread_manager(db_connection_string: str, **kwargs) -> ArchonThreadManager:
    """Create a thread manager with database storage."""
    storage = DatabaseThreadStorage(db_connection_string)
    return ArchonThreadManager(storage=storage, **kwargs)