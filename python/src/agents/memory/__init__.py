"""
Memory Service Module for Archon+ Phase 4

Provides layered memory management with role-specific access control:
- Global Memory: System-wide patterns, best practices
- Project Memory: Project-specific context, decisions
- Job Memory: Current session/task context  
- Runtime Memory: Immediate execution context

Each agent role has specific permissions for different memory layers.
"""

from .memory_service import MemoryService, MemoryLayer, MemoryEntry, create_memory_service
from .memory_scopes import (
    MemoryScope, 
    RoleBasedAccessControl, 
    MemoryLayerType, 
    AccessLevel, 
    RoleConfiguration,
    create_role_configuration,
    get_default_rbac
)
from .adaptive_retriever import (
    AdaptiveRetriever,
    RetrievalStrategy,
    RetrievalStrategyType,
    BanditAlgorithm,
    PerformanceMetrics,
    create_adaptive_retriever
)
from .context_assembler import (
    ContextAssembler,
    ContextPack,
    ContentSection,
    create_context_assembler
)

__all__ = [
    # Core memory service
    'MemoryService',
    'MemoryLayer', 
    'MemoryEntry',
    'create_memory_service',
    
    # Role-based access control
    'MemoryScope',
    'RoleBasedAccessControl',
    'MemoryLayerType',
    'AccessLevel',
    'RoleConfiguration',
    'create_role_configuration',
    'get_default_rbac',
    
    # Adaptive retrieval
    'AdaptiveRetriever',
    'RetrievalStrategy',
    'RetrievalStrategyType',
    'BanditAlgorithm',
    'PerformanceMetrics',
    'create_adaptive_retriever',
    
    # Context assembly
    'ContextAssembler',
    'ContextPack',
    'ContentSection',
    'create_context_assembler'
]