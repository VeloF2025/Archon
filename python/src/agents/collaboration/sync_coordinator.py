"""
Real-time Synchronization Coordinator
Manages real-time synchronization of code changes across multiple developers
Handles operational transformation and maintains consistency
"""

from typing import Dict, List, Optional, Tuple, Any, Set, Deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections import deque, defaultdict
import asyncio
import json
import logging
import hashlib
import uuid

from pydantic import BaseModel, Field

from .conflict_resolver import CodeChange, ConflictResolver
from .awareness_engine import AwarenessEngine

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations for operational transformation"""
    INSERT = "insert"
    DELETE = "delete"
    RETAIN = "retain"
    REPLACE = "replace"


class SyncState(Enum):
    """Synchronization state for documents"""
    SYNCHRONIZED = "synchronized"
    PENDING = "pending"
    CONFLICTED = "conflicted"
    ERROR = "error"


@dataclass
class Operation:
    """
    Operational transformation operation
    Based on OT (Operational Transformation) principles
    """
    op_id: str
    op_type: OperationType
    position: int  # Character or line position
    length: int = 0  # For delete operations
    content: str = ""  # For insert/replace operations
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_id": self.op_id,
            "op_type": self.op_type.value,
            "position": self.position,
            "length": self.length,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Operation":
        return cls(
            op_id=data["op_id"],
            op_type=OperationType(data["op_type"]),
            position=data["position"],
            length=data.get("length", 0),
            content=data.get("content", ""),
            timestamp=datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00')),
            user_id=data.get("user_id", "")
        )


@dataclass
class DocumentState:
    """State of a document in collaborative editing"""
    file_path: str
    session_id: str
    content: str
    version: int = 0
    last_modified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checksum: str = ""
    sync_state: SyncState = SyncState.SYNCHRONIZED
    pending_operations: Deque[Operation] = field(default_factory=deque)
    applied_operations: List[Operation] = field(default_factory=list)
    active_editors: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate content checksum for integrity verification"""
        return hashlib.sha256(self.content.encode('utf-8')).hexdigest()[:16]
    
    def update_content(self, new_content: str) -> None:
        """Update document content and recalculate checksum"""
        self.content = new_content
        self.checksum = self._calculate_checksum()
        self.last_modified = datetime.now(timezone.utc)
        self.version += 1


@dataclass
class SyncMessage:
    """Message for synchronization between clients"""
    message_id: str
    session_id: str
    file_path: str
    message_type: str  # "operation", "ack", "sync_request", "sync_response"
    user_id: str
    timestamp: datetime
    operations: List[Operation] = field(default_factory=list)
    document_version: int = 0
    checksum: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "session_id": self.session_id,
            "file_path": self.file_path,
            "message_type": self.message_type,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "operations": [op.to_dict() for op in self.operations],
            "document_version": self.document_version,
            "checksum": self.checksum,
            "metadata": self.metadata
        }


class OperationalTransformer:
    """
    Operational Transformation engine for conflict-free synchronization
    Implements transform algorithms for concurrent editing operations
    """
    
    @staticmethod
    def transform_operations(
        op1: Operation, 
        op2: Operation, 
        priority: str = "timestamp"
    ) -> Tuple[Operation, Operation]:
        """
        Transform two concurrent operations to maintain consistency
        Returns (transformed_op1, transformed_op2)
        """
        # Create copies to avoid mutating originals
        t_op1 = Operation(
            op_id=op1.op_id,
            op_type=op1.op_type,
            position=op1.position,
            length=op1.length,
            content=op1.content,
            timestamp=op1.timestamp,
            user_id=op1.user_id
        )
        
        t_op2 = Operation(
            op_id=op2.op_id,
            op_type=op2.op_type,
            position=op2.position,
            length=op2.length,
            content=op2.content,
            timestamp=op2.timestamp,
            user_id=op2.user_id
        )
        
        # Transform based on operation types
        if op1.op_type == OperationType.INSERT and op2.op_type == OperationType.INSERT:
            t_op1, t_op2 = OperationalTransformer._transform_insert_insert(t_op1, t_op2, priority)
        elif op1.op_type == OperationType.INSERT and op2.op_type == OperationType.DELETE:
            t_op1, t_op2 = OperationalTransformer._transform_insert_delete(t_op1, t_op2)
        elif op1.op_type == OperationType.DELETE and op2.op_type == OperationType.INSERT:
            t_op2, t_op1 = OperationalTransformer._transform_insert_delete(t_op2, t_op1)
        elif op1.op_type == OperationType.DELETE and op2.op_type == OperationType.DELETE:
            t_op1, t_op2 = OperationalTransformer._transform_delete_delete(t_op1, t_op2)
        
        return t_op1, t_op2
    
    @staticmethod
    def _transform_insert_insert(
        op1: Operation, 
        op2: Operation, 
        priority: str
    ) -> Tuple[Operation, Operation]:
        """Transform two concurrent insert operations"""
        if op1.position <= op2.position:
            # op1 comes before op2, adjust op2 position
            op2.position += len(op1.content)
        elif op1.position > op2.position:
            # op2 comes before op1, adjust op1 position
            op1.position += len(op2.content)
        elif op1.position == op2.position:
            # Same position - use priority to decide order
            if priority == "timestamp":
                if op1.timestamp <= op2.timestamp:
                    op2.position += len(op1.content)
                else:
                    op1.position += len(op2.content)
            elif priority == "user_id":
                if op1.user_id < op2.user_id:
                    op2.position += len(op1.content)
                else:
                    op1.position += len(op2.content)
        
        return op1, op2
    
    @staticmethod
    def _transform_insert_delete(
        insert_op: Operation, 
        delete_op: Operation
    ) -> Tuple[Operation, Operation]:
        """Transform insert vs delete operations"""
        if insert_op.position <= delete_op.position:
            # Insert before delete range
            delete_op.position += len(insert_op.content)
        elif insert_op.position > delete_op.position + delete_op.length:
            # Insert after delete range
            insert_op.position -= delete_op.length
        else:
            # Insert within delete range - complex case
            # Adjust insert position to start of delete range
            insert_op.position = delete_op.position
        
        return insert_op, delete_op
    
    @staticmethod
    def _transform_delete_delete(
        op1: Operation, 
        op2: Operation
    ) -> Tuple[Operation, Operation]:
        """Transform two concurrent delete operations"""
        # Calculate ranges
        range1 = (op1.position, op1.position + op1.length)
        range2 = (op2.position, op2.position + op2.length)
        
        # Check for overlap
        if range1[1] <= range2[0]:
            # op1 before op2
            op2.position -= op1.length
        elif range2[1] <= range1[0]:
            # op2 before op1
            op1.position -= op2.length
        else:
            # Overlapping deletes - complex resolution
            start1, end1 = range1
            start2, end2 = range2
            
            # Calculate new ranges after removing overlap
            if start1 <= start2 and end1 >= end2:
                # op1 contains op2 - reduce op1 length
                op1.length -= (end2 - start2)
                op2.length = 0  # op2 becomes no-op
            elif start2 <= start1 and end2 >= end1:
                # op2 contains op1 - reduce op2 length
                op2.length -= (end1 - start1)
                op1.length = 0  # op1 becomes no-op
            else:
                # Partial overlap - calculate remaining ranges
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                overlap_length = overlap_end - overlap_start
                
                op1.length -= overlap_length
                op2.length -= overlap_length
                
                if start2 < start1:
                    op1.position = start2
        
        return op1, op2


class SyncCoordinator:
    """
    Real-time synchronization coordinator for collaborative editing
    Manages operational transformation, conflict resolution, and consistency
    """
    
    def __init__(
        self, 
        redis_client=None, 
        conflict_resolver: Optional[ConflictResolver] = None,
        awareness_engine: Optional[AwarenessEngine] = None
    ):
        self.redis = redis_client
        self.conflict_resolver = conflict_resolver or ConflictResolver(redis_client)
        self.awareness_engine = awareness_engine or AwarenessEngine(redis_client)
        
        # Document state management
        self.documents: Dict[str, DocumentState] = {}  # file_path -> DocumentState
        self.operation_history: Dict[str, List[Operation]] = defaultdict(list)
        self.pending_acks: Dict[str, Set[str]] = defaultdict(set)  # message_id -> user_ids
        
        # Operational transformer
        self.transformer = OperationalTransformer()
        
        # Configuration
        self.sync_timeout_seconds = 30
        self.max_pending_operations = 100
        self.operation_retention_hours = 24
        
        logger.info("SyncCoordinator initialized")
    
    async def initialize_document(
        self, 
        session_id: str,
        file_path: str, 
        initial_content: str,
        user_id: str
    ) -> DocumentState:
        """
        Initialize a document for collaborative editing
        """
        doc_key = f"{session_id}:{file_path}"
        
        if doc_key in self.documents:
            # Add user to existing document
            self.documents[doc_key].active_editors.add(user_id)
            return self.documents[doc_key]
        
        # Create new document state
        document = DocumentState(
            file_path=file_path,
            session_id=session_id,
            content=initial_content,
            active_editors={user_id}
        )
        
        self.documents[doc_key] = document
        
        logger.info(f"Initialized document {file_path} in session {session_id}")
        
        return document
    
    async def apply_operation(
        self, 
        session_id: str,
        file_path: str, 
        operation: Operation,
        user_id: str
    ) -> bool:
        """
        Apply an operation to a document with operational transformation
        """
        doc_key = f"{session_id}:{file_path}"
        
        if doc_key not in self.documents:
            logger.error(f"Document {file_path} not initialized in session {session_id}")
            return False
        
        document = self.documents[doc_key]
        operation.user_id = user_id
        
        # Add operation to pending queue
        document.pending_operations.append(operation)
        document.active_editors.add(user_id)
        
        try:
            # Transform against other pending operations
            transformed_ops = await self._transform_pending_operations(document)
            
            # Apply transformed operations to document
            for op in transformed_ops:
                await self._apply_single_operation(document, op)
            
            # Clear pending operations
            document.pending_operations.clear()
            
            # Broadcast sync message
            await self._broadcast_operations(session_id, file_path, transformed_ops)
            
            document.sync_state = SyncState.SYNCHRONIZED
            
            logger.debug(f"Applied operation {operation.op_id} to {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply operation {operation.op_id}: {e}")
            document.sync_state = SyncState.ERROR
            return False
    
    async def _transform_pending_operations(
        self, 
        document: DocumentState
    ) -> List[Operation]:
        """
        Transform all pending operations using operational transformation
        """
        if len(document.pending_operations) <= 1:
            return list(document.pending_operations)
        
        # Convert deque to list for easier manipulation
        operations = list(document.pending_operations)
        
        # Sort by timestamp for consistent ordering
        operations.sort(key=lambda op: op.timestamp)
        
        # Transform operations pairwise
        for i in range(len(operations)):
            for j in range(i + 1, len(operations)):
                op1, op2 = self.transformer.transform_operations(
                    operations[i], 
                    operations[j],
                    priority="timestamp"
                )
                operations[i] = op1
                operations[j] = op2
        
        return operations
    
    async def _apply_single_operation(
        self, 
        document: DocumentState, 
        operation: Operation
    ) -> None:
        """
        Apply a single operation to document content
        """
        content = document.content
        
        if operation.op_type == OperationType.INSERT:
            # Insert content at position
            new_content = (
                content[:operation.position] + 
                operation.content + 
                content[operation.position:]
            )
        elif operation.op_type == OperationType.DELETE:
            # Delete content from position
            end_pos = operation.position + operation.length
            new_content = content[:operation.position] + content[end_pos:]
        elif operation.op_type == OperationType.REPLACE:
            # Replace content at position
            end_pos = operation.position + operation.length
            new_content = (
                content[:operation.position] + 
                operation.content + 
                content[end_pos:]
            )
        else:
            # RETAIN - no change to content
            new_content = content
        
        # Update document
        document.update_content(new_content)
        document.applied_operations.append(operation)
        
        # Store in operation history
        doc_key = f"{document.session_id}:{document.file_path}"
        self.operation_history[doc_key].append(operation)
    
    async def _broadcast_operations(
        self, 
        session_id: str,
        file_path: str, 
        operations: List[Operation]
    ) -> None:
        """
        Broadcast operations to all session participants
        """
        if not self.redis or not operations:
            return
        
        doc_key = f"{session_id}:{file_path}"
        document = self.documents[doc_key]
        
        sync_message = SyncMessage(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            file_path=file_path,
            message_type="operation",
            user_id="system",
            timestamp=datetime.now(timezone.utc),
            operations=operations,
            document_version=document.version,
            checksum=document.checksum
        )
        
        try:
            channel = f"session:{session_id}:sync"
            await self.redis.publish(channel, json.dumps(sync_message.to_dict()))
            
            logger.debug(f"Broadcasted {len(operations)} operations for {file_path}")
        except Exception as e:
            logger.error(f"Failed to broadcast operations: {e}")
    
    async def handle_sync_message(self, message_data: Dict[str, Any]) -> bool:
        """
        Handle incoming synchronization message
        """
        try:
            sync_message = SyncMessage(
                message_id=message_data["message_id"],
                session_id=message_data["session_id"],
                file_path=message_data["file_path"],
                message_type=message_data["message_type"],
                user_id=message_data["user_id"],
                timestamp=datetime.fromisoformat(
                    message_data["timestamp"].replace('Z', '+00:00')
                ),
                operations=[
                    Operation.from_dict(op_data) 
                    for op_data in message_data.get("operations", [])
                ],
                document_version=message_data.get("document_version", 0),
                checksum=message_data.get("checksum", ""),
                metadata=message_data.get("metadata", {})
            )
            
            if sync_message.message_type == "operation":
                return await self._handle_operation_message(sync_message)
            elif sync_message.message_type == "sync_request":
                return await self._handle_sync_request(sync_message)
            elif sync_message.message_type == "ack":
                return await self._handle_ack_message(sync_message)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle sync message: {e}")
            return False
    
    async def _handle_operation_message(self, message: SyncMessage) -> bool:
        """Handle incoming operation message"""
        doc_key = f"{message.session_id}:{message.file_path}"
        
        if doc_key not in self.documents:
            # Request full sync for unknown document
            await self._request_full_sync(message.session_id, message.file_path)
            return False
        
        document = self.documents[doc_key]
        
        # Check version consistency
        if message.document_version != document.version + 1:
            logger.warning(
                f"Version mismatch for {message.file_path}: "
                f"expected {document.version + 1}, got {message.document_version}"
            )
            await self._request_full_sync(message.session_id, message.file_path)
            return False
        
        # Apply operations
        for operation in message.operations:
            await self._apply_single_operation(document, operation)
        
        # Verify checksum
        if message.checksum and message.checksum != document.checksum:
            logger.error(f"Checksum mismatch for {message.file_path}")
            document.sync_state = SyncState.CONFLICTED
            return False
        
        return True
    
    async def _handle_sync_request(self, message: SyncMessage) -> bool:
        """Handle full synchronization request"""
        doc_key = f"{message.session_id}:{message.file_path}"
        
        if doc_key not in self.documents:
            return False
        
        document = self.documents[doc_key]
        
        # Send full document state
        response_message = SyncMessage(
            message_id=str(uuid.uuid4()),
            session_id=message.session_id,
            file_path=message.file_path,
            message_type="sync_response",
            user_id="system",
            timestamp=datetime.now(timezone.utc),
            document_version=document.version,
            checksum=document.checksum,
            metadata={
                "content": document.content,
                "operations": [op.to_dict() for op in document.applied_operations[-50:]]
            }
        )
        
        try:
            channel = f"session:{message.session_id}:sync:{message.user_id}"
            await self.redis.publish(channel, json.dumps(response_message.to_dict()))
            return True
        except Exception as e:
            logger.error(f"Failed to send sync response: {e}")
            return False
    
    async def _handle_ack_message(self, message: SyncMessage) -> bool:
        """Handle acknowledgment message"""
        # Remove user from pending acks
        if message.message_id in self.pending_acks:
            self.pending_acks[message.message_id].discard(message.user_id)
            
            # If all users acknowledged, clean up
            if not self.pending_acks[message.message_id]:
                del self.pending_acks[message.message_id]
        
        return True
    
    async def _request_full_sync(self, session_id: str, file_path: str) -> None:
        """Request full synchronization for a document"""
        request_message = SyncMessage(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            file_path=file_path,
            message_type="sync_request",
            user_id="system",
            timestamp=datetime.now(timezone.utc)
        )
        
        try:
            channel = f"session:{session_id}:sync"
            await self.redis.publish(channel, json.dumps(request_message.to_dict()))
        except Exception as e:
            logger.error(f"Failed to request full sync: {e}")
    
    async def get_document_state(
        self, 
        session_id: str, 
        file_path: str
    ) -> Optional[DocumentState]:
        """Get current state of a document"""
        doc_key = f"{session_id}:{file_path}"
        return self.documents.get(doc_key)
    
    async def get_session_documents(self, session_id: str) -> List[DocumentState]:
        """Get all documents in a session"""
        return [
            doc for key, doc in self.documents.items() 
            if key.startswith(f"{session_id}:")
        ]
    
    async def cleanup_session(self, session_id: str) -> int:
        """Clean up documents and state for a session"""
        removed_count = 0
        
        # Remove documents
        keys_to_remove = [
            key for key in self.documents.keys() 
            if key.startswith(f"{session_id}:")
        ]
        
        for key in keys_to_remove:
            del self.documents[key]
            removed_count += 1
        
        # Clean up operation history
        for key in keys_to_remove:
            if key in self.operation_history:
                del self.operation_history[key]
        
        logger.info(f"Cleaned up {removed_count} documents for session {session_id}")
        
        return removed_count
    
    async def get_sync_statistics(
        self, 
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get synchronization statistics"""
        stats = {
            "total_documents": len(self.documents),
            "synchronized_documents": 0,
            "pending_documents": 0,
            "conflicted_documents": 0,
            "total_operations": 0,
            "active_sessions": set(),
            "sync_states": defaultdict(int)
        }
        
        for key, document in self.documents.items():
            if session_id and not key.startswith(f"{session_id}:"):
                continue
            
            stats["sync_states"][document.sync_state.value] += 1
            stats["active_sessions"].add(document.session_id)
            stats["total_operations"] += len(document.applied_operations)
            
            if document.sync_state == SyncState.SYNCHRONIZED:
                stats["synchronized_documents"] += 1
            elif document.sync_state == SyncState.PENDING:
                stats["pending_documents"] += 1
            elif document.sync_state == SyncState.CONFLICTED:
                stats["conflicted_documents"] += 1
        
        stats["active_sessions"] = len(stats["active_sessions"])
        stats["sync_states"] = dict(stats["sync_states"])
        
        return stats