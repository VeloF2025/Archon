"""
Context Preservation - Context transfer and management during handoffs.

This module handles the preservation and transfer of context during agent handoffs,
ensuring conversation continuity and maintaining important information.

Key Features:
- Context extraction and compression
- Context transfer between agents
- Conversation history preservation
- Metadata and state management
- Context validation and cleanup
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Union, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib

from .agent_handoff_engine import HandoffContext, AgentMessage
from ...agents.base_agent import ArchonDependencies

logger = logging.getLogger(__name__)


class ContextElementType(Enum):
    """Types of context elements."""
    CONVERSATION = "conversation"
    METADATA = "metadata"
    STATE = "state"
    KNOWLEDGE = "knowledge"
    REFERENCES = "references"
    DEPENDENCIES = "dependencies"
    CONFIDENCE = "confidence"


class ContextCompressionLevel(Enum):
    """Levels of context compression."""
    NONE = "none"           # No compression
    BASIC = "basic"         # Remove redundant information
    AGGRESSIVE = "aggressive"  # Keep only essential information
    SUMMARY = "summary"     # Convert to summary format


@dataclass
class ContextElement:
    """Represents a single element of context."""
    element_id: str
    element_type: ContextElementType
    content: Any
    importance: float  # 0-1, where 1 is most important
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert element to dictionary."""
        return {
            "element_id": self.element_id,
            "element_type": self.element_type.value,
            "content": self.content,
            "importance": self.importance,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "size_bytes": self.size_bytes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextElement":
        """Create element from dictionary."""
        return cls(
            element_id=data["element_id"],
            element_type=ContextElementType(data["element_type"]),
            content=data["content"],
            importance=data["importance"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            size_bytes=data.get("size_bytes", 0)
        )


@dataclass
class ContextPackage:
    """A complete context package for handoff."""
    package_id: str
    source_agent: str
    target_agent: str
    elements: List[ContextElement] = field(default_factory=list)
    conversation_summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert package to dictionary."""
        return {
            "package_id": self.package_id,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "elements": [elem.to_dict() for elem in self.elements],
            "conversation_summary": self.conversation_summary,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "size_bytes": self.size_bytes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextPackage":
        """Create package from dictionary."""
        return cls(
            package_id=data["package_id"],
            source_agent=data["source_agent"],
            target_agent=data["target_agent"],
            elements=[ContextElement.from_dict(elem) for elem in data["elements"]],
            conversation_summary=data.get("conversation_summary"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            size_bytes=data.get("size_bytes", 0)
        )


class ContextPreservationEngine:
    """
    Engine for preserving and transferring context during handoffs.

    This engine handles:
    - Context extraction from current agent state
    - Context packaging and compression
    - Context transfer to target agent
    - Context validation and cleanup
    """

    def __init__(self, max_context_size: int = 1024 * 1024):  # 1MB default
        """Initialize the context preservation engine."""
        self.max_context_size = max_context_size
        self.context_packages: Dict[str, ContextPackage] = {}
        self.compression_cache: Dict[str, ContextPackage] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.logger.info(f"ContextPreservationEngine initialized with max_size={max_context_size}")

    async def extract_context(
        self,
        source_agent: str,
        target_agent: str,
        message: str,
        task_description: str,
        conversation_history: List[Dict[str, Any]],
        dependencies: Optional[ArchonDependencies] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> HandoffContext:
        """
        Extract context from current agent state.

        Args:
            source_agent: Name of source agent
            target_agent: Name of target agent
            message: Original message
            task_description: Task description
            conversation_history: Conversation history
            dependencies: Agent dependencies
            metadata: Additional metadata

        Returns:
            HandoffContext with extracted information
        """
        try:
            # Create base context
            context = HandoffContext(
                context_id=str(uuid.uuid4()),
                original_message=message,
                task_description=task_description,
                sender_agent=source_agent,
                recipient_agent=target_agent,
                conversation_history=conversation_history[-10:],  # Keep recent history
                metadata=metadata or {},
                dependencies=dependencies
            )

            # Extract context elements
            elements = await self._extract_context_elements(
                message, task_description, conversation_history, dependencies, metadata
            )

            # Create context package
            package = await self._create_context_package(
                source_agent, target_agent, elements, conversation_history
            )

            # Store package reference in context
            context.metadata["context_package_id"] = package.package_id
            context.metadata["package_size"] = package.size_bytes

            self.logger.info(f"Extracted context for handoff: {source_agent} -> {target_agent}")
            return context

        except Exception as e:
            self.logger.error(f"Error extracting context: {e}")
            # Return minimal context on error
            return HandoffContext(
                context_id=str(uuid.uuid4()),
                original_message=message,
                task_description=task_description,
                sender_agent=source_agent,
                recipient_agent=target_agent,
                metadata={"error": str(e)}
            )

    async def _extract_context_elements(
        self,
        message: str,
        task_description: str,
        conversation_history: List[Dict[str, Any]],
        dependencies: Optional[ArchonDependencies],
        metadata: Optional[Dict[str, Any]]
    ) -> List[ContextElement]:
        """Extract individual context elements."""
        elements = []

        # Conversation element
        if conversation_history:
            conversation_element = ContextElement(
                element_id=str(uuid.uuid4()),
                element_type=ContextElementType.CONVERSATION,
                content=conversation_history[-5:],  # Keep last 5 messages
                importance=0.8
            )
            elements.append(conversation_element)

        # Metadata element
        if metadata:
            metadata_element = ContextElement(
                element_id=str(uuid.uuid4()),
                element_type=ContextElementType.METADATA,
                content=metadata,
                importance=0.6
            )
            elements.append(metadata_element)

        # State element from dependencies
        if dependencies and dependencies.context:
            state_element = ContextElement(
                element_id=str(uuid.uuid4()),
                element_type=ContextElementType.STATE,
                content=dependencies.context,
                importance=0.9
            )
            elements.append(state_element)

        # Task information element
        task_element = ContextElement(
            element_id=str(uuid.uuid4()),
            element_type=ContextElementType.KNOWLEDGE,
            content={
                "task_description": task_description,
                "original_message": message,
                "extracted_at": datetime.utcnow().isoformat()
            },
            importance=1.0
        )
        elements.append(task_element)

        # Calculate sizes
        for element in elements:
            element.size_bytes = len(json.dumps(element.content).encode('utf-8'))

        return elements

    async def _create_context_package(
        self,
        source_agent: str,
        target_agent: str,
        elements: List[ContextElement],
        conversation_history: List[Dict[str, Any]]
    ) -> ContextPackage:
        """Create a context package."""
        package = ContextPackage(
            package_id=str(uuid.uuid4()),
            source_agent=source_agent,
            target_agent=target_agent,
            elements=elements
        )

        # Generate conversation summary
        if conversation_history:
            package.conversation_summary = await self._generate_conversation_summary(conversation_history)

        # Calculate total size
        package.size_bytes = sum(elem.size_bytes for elem in elements)

        # Apply compression if needed
        if package.size_bytes > self.max_context_size:
            package = await self._compress_context_package(package)

        # Store package
        self.context_packages[package.package_id] = package

        self.logger.debug(f"Created context package: {package.package_id} ({package.size_bytes} bytes)")
        return package

    async def _generate_conversation_summary(self, conversation_history: List[Dict[str, Any]]) -> str:
        """Generate a summary of the conversation."""
        if not conversation_history:
            return "No conversation history available."

        # Simple summary generation - could be enhanced with AI
        messages = [msg.get("content", "") for msg in conversation_history[-5:]]
        total_messages = len(conversation_history)

        summary = f"Conversation summary ({total_messages} total messages):\n"
        summary += "Recent messages:\n"

        for i, msg in enumerate(messages[-3:], 1):  # Last 3 messages
            if len(msg) > 100:
                msg = msg[:100] + "..."
            summary += f"{i}. {msg}\n"

        return summary

    async def _compress_context_package(self, package: ContextPackage) -> ContextPackage:
        """Compress context package to fit size limits."""
        self.logger.info(f"Compressing context package: {package.package_id}")

        # Sort elements by importance
        elements_sorted = sorted(package.elements, key=lambda x: x.importance, reverse=True)

        # Keep most important elements
        compressed_elements = []
        current_size = 0

        for element in elements_sorted:
            if current_size + element.size_bytes <= self.max_context_size:
                compressed_elements.append(element)
                current_size += element.size_bytes
            else:
                # Try to compress content
                compressed_element = await self._compress_element(element, self.max_context_size - current_size)
                if compressed_element:
                    compressed_elements.append(compressed_element)
                    current_size += compressed_element.size_bytes
                break

        # Create compressed package
        compressed_package = ContextPackage(
            package_id=package.package_id,
            source_agent=package.source_agent,
            target_agent=package.target_agent,
            elements=compressed_elements,
            conversation_summary=package.conversation_summary,
            metadata={
                **package.metadata,
                "compressed": True,
                "original_size": package.size_bytes,
                "compression_ratio": current_size / package.size_bytes if package.size_bytes > 0 else 1.0
            },
            created_at=package.created_at,
            expires_at=package.expires_at,
            size_bytes=current_size
        )

        self.logger.info(f"Compressed package: {package.size_bytes} -> {current_size} bytes")
        return compressed_package

    async def _compress_element(self, element: ContextElement, max_size: int) -> Optional[ContextElement]:
        """Compress a single context element."""
        try:
            content_str = json.dumps(element.content)
            if len(content_str.encode('utf-8')) <= max_size:
                return element

            # Truncate or summarize content
            if element.element_type == ContextElementType.CONVERSATION:
                # Keep only the last 2 messages
                if isinstance(element.content, list) and len(element.content) > 2:
                    element.content = element.content[-2:]

            elif element.element_type == ContextElementType.METADATA:
                # Keep only essential metadata
                essential_keys = ["user_id", "project_id", "task_id"]
                if isinstance(element.content, dict):
                    element.content = {k: v for k, v in element.content.items() if k in essential_keys}

            # Recalculate size
            element.size_bytes = len(json.dumps(element.content).encode('utf-8'))

            return element if element.size_bytes <= max_size else None

        except Exception as e:
            self.logger.error(f"Error compressing element: {e}")
            return None

    async def transfer_context(
        self,
        context: HandoffContext,
        target_agent_name: str
    ) -> ContextPackage:
        """
        Transfer context to target agent.

        Args:
            context: Handoff context
            target_agent_name: Name of target agent

        Returns:
            Transferred context package
        """
        try:
            # Get context package
            package_id = context.metadata.get("context_package_id")
            if not package_id or package_id not in self.context_packages:
                raise ValueError("Context package not found")

            package = self.context_packages[package_id]

            # Validate package
            if package.expires_at and datetime.utcnow() > package.expires_at:
                raise ValueError("Context package has expired")

            # Update package metadata
            package.metadata["transferred_at"] = datetime.utcnow().isoformat()
            package.metadata["target_agent"] = target_agent_name

            self.logger.info(f"Transferred context package: {package_id} to {target_agent_name}")
            return package

        except Exception as e:
            self.logger.error(f"Error transferring context: {e}")
            raise

    async def reconstruct_dependencies(
        self,
        package: ContextPackage,
        target_agent_name: str
    ) -> ArchonDependencies:
        """
        Reconstruct dependencies for target agent from context package.

        Args:
            package: Context package
            target_agent_name: Name of target agent

        Returns:
            Reconstructed ArchonDependencies
        """
        try:
            # Find state element
            state_element = None
            for element in package.elements:
                if element.element_type == ContextElementType.STATE:
                    state_element = element
                    break

            # Create base dependencies
            context = {}
            if state_element:
                context.update(state_element.content)

            # Add package information
            context.update({
                "handoff_package_id": package.package_id,
                "source_agent": package.source_agent,
                "conversation_summary": package.conversation_summary
            })

            # Extract user and trace IDs from package
            user_id = package.metadata.get("user_id")
            trace_id = package.metadata.get("trace_id", str(uuid.uuid4()))

            deps = ArchonDependencies(
                request_id=str(uuid.uuid4()),
                user_id=user_id,
                trace_id=trace_id,
                context=context
            )

            self.logger.debug(f"Reconstructed dependencies for {target_agent_name}")
            return deps

        except Exception as e:
            self.logger.error(f"Error reconstructing dependencies: {e}")
            # Return basic dependencies on error
            return ArchonDependencies(
                request_id=str(uuid.uuid4()),
                user_id=None,
                trace_id=str(uuid.uuid4()),
                context={"handoff_error": str(e)}
            )

    async def cleanup_expired_contexts(self) -> None:
        """Clean up expired context packages."""
        try:
            expired_packages = []
            current_time = datetime.utcnow()

            for package_id, package in self.context_packages.items():
                if package.expires_at and current_time > package.expires_at:
                    expired_packages.append(package_id)

            for package_id in expired_packages:
                del self.context_packages[package_id]
                self.logger.debug(f"Cleaned up expired context package: {package_id}")

            if expired_packages:
                self.logger.info(f"Cleaned up {len(expired_packages)} expired context packages")

        except Exception as e:
            self.logger.error(f"Error cleaning up expired contexts: {e}")

    def get_context_statistics(self) -> Dict[str, Any]:
        """Get context preservation statistics."""
        total_packages = len(self.context_packages)
        total_size = sum(pkg.size_bytes for pkg in self.context_packages.values())

        element_type_counts = {}
        for package in self.context_packages.values():
            for element in package.elements:
                element_type = element.element_type.value
                element_type_counts[element_type] = element_type_counts.get(element_type, 0) + 1

        return {
            "total_packages": total_packages,
            "total_size_bytes": total_size,
            "average_package_size": total_size / total_packages if total_packages > 0 else 0,
            "element_type_distribution": element_type_counts,
            "max_context_size": self.max_context_size,
            "compression_enabled": True
        }

    async def validate_context_integrity(self, package_id: str) -> bool:
        """Validate the integrity of a context package."""
        try:
            if package_id not in self.context_packages:
                return False

            package = self.context_packages[package_id]

            # Check expiration
            if package.expires_at and datetime.utcnow() > package.expires_at:
                return False

            # Check element integrity
            for element in package.elements:
                if not element.content:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating context integrity: {e}")
            return False