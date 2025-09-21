"""
Real-time Collaboration Engine
Enables multi-developer AI-assisted collaborative coding
"""

from .session_manager import CollaborationSessionManager
from .conflict_resolver import ConflictResolver  
from .awareness_engine import AwarenessEngine
from .sync_coordinator import SyncCoordinator
from .collaborative_predictor import CollaborativePredictor

__all__ = [
    'CollaborationSessionManager',
    'ConflictResolver',
    'AwarenessEngine', 
    'SyncCoordinator',
    'CollaborativePredictor'
]