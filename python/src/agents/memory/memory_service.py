#!/usr/bin/env python3
"""
Memory Service - Layered Memory Management for Archon+ Phase 4

Implements memory storage and retrieval with role-based access control:
- Global Memory: System-wide patterns, best practices
- Project Memory: Project-specific context, decisions  
- Job Memory: Current session/task context
- Runtime Memory: Immediate execution context

Performance requirements: <100ms query response time
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

from .memory_scopes import (
    MemoryLayerType, 
    AccessLevel, 
    RoleBasedAccessControl,
    MemoryScope
)

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """Individual memory entry with metadata"""
    entry_id: str
    content: Any
    memory_layer: MemoryLayerType
    source_agent: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance_score: float = 0.5  # 0.0 = low, 1.0 = high

@dataclass
class MemoryLayer:
    """Memory layer with storage and access management"""
    layer_type: MemoryLayerType
    entries: Dict[str, MemoryEntry] = field(default_factory=dict)
    max_entries: Optional[int] = None
    ttl_hours: Optional[int] = None
    persistent: bool = True
    storage_path: Optional[Path] = None

class MemoryService:
    """
    Layered memory service with role-based access control
    
    Provides persistent and in-memory storage across different memory layers
    with performance optimization and access control.
    """
    
    def __init__(self, base_storage_path: Optional[Path] = None):
        """
        Initialize memory service with layered storage
        
        Args:
            base_storage_path: Base directory for persistent storage
        """
        self.base_storage_path = base_storage_path or Path("python/src/agents/memory/storage")
        self.rbac = RoleBasedAccessControl()
        
        # Initialize memory layers based on PRP specifications
        self.layers: Dict[MemoryLayerType, MemoryLayer] = {
            MemoryLayerType.GLOBAL: MemoryLayer(
                layer_type=MemoryLayerType.GLOBAL,
                max_entries=10000,  # Large capacity for global patterns
                ttl_hours=None,     # Never expires
                persistent=True,
                storage_path=self.base_storage_path / "global"
            ),
            
            MemoryLayerType.PROJECT: MemoryLayer(
                layer_type=MemoryLayerType.PROJECT,
                max_entries=5000,   # Medium capacity for project context
                ttl_hours=24 * 30,  # 30 days
                persistent=True,
                storage_path=self.base_storage_path / "project"
            ),
            
            MemoryLayerType.JOB: MemoryLayer(
                layer_type=MemoryLayerType.JOB,
                max_entries=1000,   # Smaller capacity for job context
                ttl_hours=24 * 7,   # 7 days
                persistent=True,
                storage_path=self.base_storage_path / "job"
            ),
            
            MemoryLayerType.RUNTIME: MemoryLayer(
                layer_type=MemoryLayerType.RUNTIME,
                max_entries=500,    # Smallest capacity for runtime
                ttl_hours=1,        # 1 hour
                persistent=False,   # In-memory only
                storage_path=None
            )
        }
        
        # Create storage directories
        self._initialize_storage()
        
        # Load persistent memory layers
        self._load_persistent_layers()
        
        # Performance tracking
        self.query_times: List[float] = []
        self.max_query_time = 0.1  # 100ms requirement
        
    def _initialize_storage(self):
        """Create storage directories for persistent layers"""
        for layer in self.layers.values():
            if layer.persistent and layer.storage_path:
                layer.storage_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Initialized storage for {layer.layer_type.value} at {layer.storage_path}")
    
    def _load_persistent_layers(self):
        """Load data from persistent storage"""
        for layer_type, layer in self.layers.items():
            if not layer.persistent or not layer.storage_path:
                continue
                
            storage_file = layer.storage_path / f"{layer_type.value}_memory.json"
            if storage_file.exists():
                try:
                    with open(storage_file, 'r') as f:
                        data = json.load(f)
                    
                    # Reconstruct memory entries
                    for entry_id, entry_data in data.get("entries", {}).items():
                        entry = MemoryEntry(
                            entry_id=entry_id,
                            content=entry_data.get("content"),
                            memory_layer=MemoryLayerType(entry_data.get("memory_layer")),
                            source_agent=entry_data.get("source_agent", "unknown"),
                            created_at=entry_data.get("created_at", time.time()),
                            last_accessed=entry_data.get("last_accessed", time.time()),
                            access_count=entry_data.get("access_count", 0),
                            tags=entry_data.get("tags", []),
                            metadata=entry_data.get("metadata", {}),
                            importance_score=entry_data.get("importance_score", 0.5)
                        )
                        layer.entries[entry_id] = entry
                    
                    logger.info(f"Loaded {len(layer.entries)} entries for {layer_type.value} layer")
                    
                except Exception as e:
                    logger.error(f"Failed to load {layer_type.value} layer: {e}")
    
    def _save_persistent_layer(self, layer_type: MemoryLayerType):
        """Save a memory layer to persistent storage"""
        layer = self.layers[layer_type]
        if not layer.persistent or not layer.storage_path:
            return
            
        storage_file = layer.storage_path / f"{layer_type.value}_memory.json"
        
        try:
            # Serialize memory entries
            data = {
                "layer_type": layer_type.value,
                "last_saved": time.time(),
                "entries": {}
            }
            
            for entry_id, entry in layer.entries.items():
                data["entries"][entry_id] = {
                    "content": entry.content,
                    "memory_layer": entry.memory_layer.value,
                    "source_agent": entry.source_agent,
                    "created_at": entry.created_at,
                    "last_accessed": entry.last_accessed,
                    "access_count": entry.access_count,
                    "tags": entry.tags,
                    "metadata": entry.metadata,
                    "importance_score": entry.importance_score
                }
            
            with open(storage_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Saved {len(layer.entries)} entries for {layer_type.value} layer")
            
        except Exception as e:
            logger.error(f"Failed to save {layer_type.value} layer: {e}")
    
    def _cleanup_expired_entries(self, layer_type: MemoryLayerType):
        """Remove expired entries from a memory layer"""
        layer = self.layers[layer_type]
        if not layer.ttl_hours:
            return
            
        current_time = time.time()
        expired_entries = []
        
        for entry_id, entry in layer.entries.items():
            entry_age_hours = (current_time - entry.created_at) / 3600
            if entry_age_hours > layer.ttl_hours:
                expired_entries.append(entry_id)
        
        for entry_id in expired_entries:
            del layer.entries[entry_id]
            logger.debug(f"Removed expired entry {entry_id} from {layer_type.value} layer")
        
        if expired_entries:
            logger.info(f"Cleaned up {len(expired_entries)} expired entries from {layer_type.value}")
    
    def _enforce_capacity_limits(self, layer_type: MemoryLayerType):
        """Enforce maximum capacity limits using LRU eviction"""
        layer = self.layers[layer_type]
        if not layer.max_entries or len(layer.entries) <= layer.max_entries:
            return
            
        # Sort by last accessed time (LRU)
        sorted_entries = sorted(
            layer.entries.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Remove oldest entries
        entries_to_remove = len(layer.entries) - layer.max_entries
        for i in range(entries_to_remove):
            entry_id = sorted_entries[i][0]
            del layer.entries[entry_id]
            logger.debug(f"Evicted LRU entry {entry_id} from {layer_type.value} layer")
        
        logger.info(f"Evicted {entries_to_remove} entries from {layer_type.value} to enforce capacity")
    
    async def store(self, content: Any, memory_layer: MemoryLayerType, 
                   agent_role: str, entry_id: Optional[str] = None,
                   tags: Optional[List[str]] = None, 
                   metadata: Optional[Dict[str, Any]] = None,
                   importance_score: float = 0.5) -> Optional[str]:
        """
        Store content in a memory layer with access control
        
        Args:
            content: Data to store
            memory_layer: Target memory layer
            agent_role: Role of the storing agent
            entry_id: Optional custom entry ID
            tags: Optional tags for categorization
            metadata: Optional metadata
            importance_score: Importance score (0.0-1.0)
            
        Returns:
            Entry ID if successful, None otherwise
        """
        start_time = time.time()
        
        # Check access permissions
        access_result = self.rbac.validate_memory_access_request(
            agent_role, memory_layer, "write", content
        )
        
        if not access_result["allowed"]:
            logger.warning(f"Store denied: {access_result['reason']}")
            return None
        
        # Generate entry ID if not provided
        if not entry_id:
            entry_id = f"{agent_role}_{memory_layer.value}_{int(time.time() * 1000)}"
        
        # Create memory entry
        entry = MemoryEntry(
            entry_id=entry_id,
            content=content,
            memory_layer=memory_layer,
            source_agent=agent_role,
            tags=tags or [],
            metadata=metadata or {},
            importance_score=max(0.0, min(1.0, importance_score))
        )
        
        # Store in appropriate layer
        layer = self.layers[memory_layer]
        layer.entries[entry_id] = entry
        
        # Cleanup and maintenance
        self._cleanup_expired_entries(memory_layer)
        self._enforce_capacity_limits(memory_layer)
        
        # Save to persistent storage if needed
        if layer.persistent:
            self._save_persistent_layer(memory_layer)
        
        # Track performance
        query_time = time.time() - start_time
        self.query_times.append(query_time)
        
        if query_time > self.max_query_time:
            logger.warning(f"Store operation took {query_time:.3f}s (exceeds {self.max_query_time}s limit)")
        
        logger.debug(f"Stored entry {entry_id} in {memory_layer.value} layer")
        return entry_id
    
    async def retrieve(self, entry_id: str, memory_layer: MemoryLayerType,
                      agent_role: str) -> Optional[MemoryEntry]:
        """
        Retrieve a specific memory entry with access control
        
        Args:
            entry_id: ID of the entry to retrieve
            memory_layer: Memory layer to search
            agent_role: Role of the requesting agent
            
        Returns:
            MemoryEntry if found and accessible, None otherwise
        """
        start_time = time.time()
        
        # Check access permissions
        access_result = self.rbac.validate_memory_access_request(
            agent_role, memory_layer, "read"
        )
        
        if not access_result["allowed"]:
            logger.warning(f"Retrieve denied: {access_result['reason']}")
            return None
        
        # Get entry from layer
        layer = self.layers[memory_layer]
        entry = layer.entries.get(entry_id)
        
        if entry:
            # Update access tracking
            entry.last_accessed = time.time()
            entry.access_count += 1
            
            # Save updated access info if persistent
            if layer.persistent:
                self._save_persistent_layer(memory_layer)
        
        # Track performance
        query_time = time.time() - start_time
        self.query_times.append(query_time)
        
        if query_time > self.max_query_time:
            logger.warning(f"Retrieve operation took {query_time:.3f}s (exceeds {self.max_query_time}s limit)")
        
        return entry
    
    async def query(self, query_text: str, memory_layer: MemoryLayerType,
                   agent_role: str, limit: int = 10, 
                   tags_filter: Optional[List[str]] = None) -> List[MemoryEntry]:
        """
        Query memory layer for content matching criteria
        
        Args:
            query_text: Text to search for
            memory_layer: Memory layer to search
            agent_role: Role of the requesting agent
            limit: Maximum number of results
            tags_filter: Optional tags to filter by
            
        Returns:
            List of matching MemoryEntry objects
        """
        start_time = time.time()
        
        # Check access permissions
        access_result = self.rbac.validate_memory_access_request(
            agent_role, memory_layer, "read"
        )
        
        if not access_result["allowed"]:
            logger.warning(f"Query denied: {access_result['reason']}")
            return []
        
        layer = self.layers[memory_layer]
        matching_entries = []
        
        # Simple text-based search (can be enhanced with vector search later)
        query_lower = query_text.lower()
        
        for entry in layer.entries.values():
            # Check tags filter
            if tags_filter and not any(tag in entry.tags for tag in tags_filter):
                continue
            
            # Check content match
            content_str = str(entry.content).lower()
            if query_lower in content_str or any(tag.lower() in query_lower for tag in entry.tags):
                # Update access tracking
                entry.last_accessed = time.time()
                entry.access_count += 1
                matching_entries.append(entry)
        
        # Sort by relevance (importance score + access frequency)
        matching_entries.sort(
            key=lambda e: e.importance_score + (e.access_count / 100),
            reverse=True
        )
        
        # Limit results
        matching_entries = matching_entries[:limit]
        
        # Save updated access info if persistent
        if layer.persistent and matching_entries:
            self._save_persistent_layer(memory_layer)
        
        # Track performance
        query_time = time.time() - start_time
        self.query_times.append(query_time)
        
        if query_time > self.max_query_time:
            logger.warning(f"Query operation took {query_time:.3f}s (exceeds {self.max_query_time}s limit)")
        
        logger.debug(f"Query '{query_text}' found {len(matching_entries)} results in {memory_layer.value}")
        return matching_entries
    
    async def get_accessible_layers(self, agent_role: str) -> List[MemoryLayerType]:
        """Get all memory layers accessible to an agent role"""
        accessible_scopes = self.rbac.get_accessible_layers(agent_role)
        return list(accessible_scopes.keys())
    
    async def get_layer_stats(self, memory_layer: MemoryLayerType, 
                             agent_role: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a memory layer"""
        # Check access permissions
        access_result = self.rbac.validate_memory_access_request(
            agent_role, memory_layer, "read"
        )
        
        if not access_result["allowed"]:
            return None
        
        layer = self.layers[memory_layer]
        
        if not layer.entries:
            return {
                "layer_type": memory_layer.value,
                "total_entries": 0,
                "storage_size_bytes": 0,
                "avg_importance": 0.0,
                "most_accessed": None
            }
        
        # Calculate statistics
        total_entries = len(layer.entries)
        avg_importance = sum(e.importance_score for e in layer.entries.values()) / total_entries
        most_accessed = max(layer.entries.values(), key=lambda e: e.access_count)
        
        # Estimate storage size
        storage_size = sum(len(str(e.content)) for e in layer.entries.values())
        
        return {
            "layer_type": memory_layer.value,
            "total_entries": total_entries,
            "max_entries": layer.max_entries,
            "storage_size_bytes": storage_size,
            "avg_importance": round(avg_importance, 3),
            "most_accessed": {
                "entry_id": most_accessed.entry_id,
                "access_count": most_accessed.access_count,
                "source_agent": most_accessed.source_agent
            },
            "ttl_hours": layer.ttl_hours,
            "persistent": layer.persistent
        }
    
    async def clear_layer(self, memory_layer: MemoryLayerType, agent_role: str) -> bool:
        """Clear all entries from a memory layer (admin operation)"""
        # Check write access
        access_result = self.rbac.validate_memory_access_request(
            agent_role, memory_layer, "write"
        )
        
        if not access_result["allowed"]:
            logger.warning(f"Clear layer denied: {access_result['reason']}")
            return False
        
        layer = self.layers[memory_layer]
        entries_count = len(layer.entries)
        layer.entries.clear()
        
        # Save empty state if persistent
        if layer.persistent:
            self._save_persistent_layer(memory_layer)
        
        logger.info(f"Cleared {entries_count} entries from {memory_layer.value} layer")
        return True
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the memory service"""
        if not self.query_times:
            return {"avg_query_time": 0.0, "max_query_time": 0.0, "total_queries": 0}
        
        avg_time = sum(self.query_times) / len(self.query_times)
        max_time = max(self.query_times)
        
        return {
            "avg_query_time": round(avg_time, 4),
            "max_query_time": round(max_time, 4),
            "total_queries": len(self.query_times),
            "performance_target": self.max_query_time,
            "within_target": avg_time <= self.max_query_time
        }
    
    async def restart_session(self):
        """Restart session - clear runtime memory, reload persistent layers"""
        # Clear runtime memory
        runtime_layer = self.layers[MemoryLayerType.RUNTIME]
        runtime_entries_count = len(runtime_layer.entries)
        runtime_layer.entries.clear()
        
        # Reload persistent layers
        self._load_persistent_layers()
        
        logger.info(f"Session restarted - cleared {runtime_entries_count} runtime entries")
    
    async def store_memory(self, memory_id: str, data: Dict[str, Any]) -> bool:
        """
        Store memory data using simplified interface for SCWT compatibility
        
        Args:
            memory_id: Unique identifier for the memory entry
            data: Memory data to store
            
        Returns:
            True if storage was successful, False otherwise
        """
        try:
            # Extract metadata from data if available
            memory_layer = data.get('memory_layer', MemoryLayerType.RUNTIME)
            if isinstance(memory_layer, str):
                memory_layer = MemoryLayerType(memory_layer)
                
            agent_role = data.get('agent_role', 'system')
            content = data.get('content', data)
            tags = data.get('tags', [])
            metadata = data.get('metadata', {})
            importance_score = data.get('importance_score', 0.5)
            
            # Use the existing store method
            result_id = await self.store(
                content=content,
                memory_layer=memory_layer,
                agent_role=agent_role,
                entry_id=memory_id,
                tags=tags,
                metadata=metadata,
                importance_score=importance_score
            )
            
            return result_id is not None
            
        except Exception as e:
            logger.error(f"Failed to store memory {memory_id}: {e}")
            return False
    
    async def query_memories(self, query: str) -> List[Dict[str, Any]]:
        """
        Query memories across all accessible layers with simplified interface
        
        Args:
            query: Query string to search for
            
        Returns:
            List of memory dictionaries matching the query
        """
        try:
            all_results = []
            
            # Query all memory layers (using system role for broad access)
            for layer_type in self.layers.keys():
                try:
                    # Check if layer is accessible (using system role)
                    access_result = self.rbac.validate_memory_access_request(
                        'system', layer_type, 'read'
                    )
                    
                    if not access_result["allowed"]:
                        continue
                        
                    # Query the layer
                    results = await self.query(
                        query_text=query,
                        memory_layer=layer_type,
                        agent_role='system',
                        limit=50  # Higher limit for comprehensive search
                    )
                    
                    # Convert to dictionary format
                    for entry in results:
                        memory_dict = {
                            'memory_id': entry.entry_id,
                            'content': entry.content,
                            'memory_layer': entry.memory_layer.value,
                            'source_agent': entry.source_agent,
                            'created_at': entry.created_at,
                            'last_accessed': entry.last_accessed,
                            'access_count': entry.access_count,
                            'tags': entry.tags,
                            'metadata': entry.metadata,
                            'importance_score': entry.importance_score
                        }
                        all_results.append(memory_dict)
                        
                except Exception as e:
                    logger.warning(f"Error querying {layer_type.value} layer: {e}")
                    continue
            
            # Sort by importance and recency
            all_results.sort(
                key=lambda x: (x['importance_score'], x['last_accessed']), 
                reverse=True
            )
            
            return all_results[:20]  # Return top 20 results
            
        except Exception as e:
            logger.error(f"Failed to query memories: {e}")
            return []
    
    async def retrieve_memories(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve specific memory by ID across all layers
        
        Args:
            memory_id: Unique identifier of the memory to retrieve
            
        Returns:
            Memory dictionary if found, None otherwise
        """
        try:
            # Search across all layers for the memory ID
            for layer_type, layer in self.layers.items():
                if memory_id in layer.entries:
                    # Check access permissions (using system role)
                    access_result = self.rbac.validate_memory_access_request(
                        'system', layer_type, 'read'
                    )
                    
                    if not access_result["allowed"]:
                        continue
                        
                    entry = layer.entries[memory_id]
                    
                    # Update access tracking
                    entry.last_accessed = time.time()
                    entry.access_count += 1
                    
                    # Save updated access info if persistent
                    if layer.persistent:
                        self._save_persistent_layer(layer_type)
                    
                    # Return as dictionary
                    return {
                        'memory_id': entry.entry_id,
                        'content': entry.content,
                        'memory_layer': entry.memory_layer.value,
                        'source_agent': entry.source_agent,
                        'created_at': entry.created_at,
                        'last_accessed': entry.last_accessed,
                        'access_count': entry.access_count,
                        'tags': entry.tags,
                        'metadata': entry.metadata,
                        'importance_score': entry.importance_score
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            return None

# Factory function for easy access
def create_memory_service(storage_path: Optional[Path] = None) -> MemoryService:
    """Create a configured memory service instance"""
    return MemoryService(storage_path)