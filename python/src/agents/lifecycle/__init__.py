"""
Agent Lifecycle Management v3.0
Based on Agent_Lifecycle_Management_PRP.md specifications

NLNH Protocol: Complete implementation following TDD approach
DGTS Enforcement: Real functionality, no fake implementations
"""

from .agent_v3 import AgentV3, AgentState, StateTransitionLog, KnowledgeItem
from .pool_manager import AgentPoolManager, PoolStatistics, PoolOptimizationResult, AgentSpec
from .project_analyzer import ProjectAnalyzer, ProjectAnalysis
from .agent_spawner import AgentSpawner

__all__ = [
    # Core agent lifecycle
    "AgentV3",
    "AgentState", 
    "StateTransitionLog",
    "KnowledgeItem",
    
    # Pool management
    "AgentPoolManager",
    "PoolStatistics",
    "PoolOptimizationResult", 
    "AgentSpec",
    
    # Project analysis
    "ProjectAnalyzer",
    "ProjectAnalysis",
    
    # Agent spawning
    "AgentSpawner"
]

# Version information
__version__ = "3.0.0"
__author__ = "Archon Development Team"
__description__ = "Intelligence-Tiered Adaptive Agent Management System"

# Module-level constants
AGENT_LIFECYCLE_VERSION = "3.0.0"
SUPPORTED_MODEL_TIERS = ["opus", "sonnet", "haiku"]
MAX_AGENTS_PER_TIER = {"opus": 2, "sonnet": 10, "haiku": 50}