"""
Collaboration Service
Coordinates all real-time collaboration components
"""

from typing import Optional
import logging
import asyncio
from functools import lru_cache

from ...agents.collaboration.session_manager import CollaborationSessionManager
from ...agents.collaboration.conflict_resolver import ConflictResolver
from ...agents.collaboration.awareness_engine import AwarenessEngine
from ...agents.collaboration.sync_coordinator import SyncCoordinator
from ...agents.collaboration.collaborative_predictor import CollaborativePredictor
from ...agents.predictive_assistant.context_analyzer import ContextAnalyzer
from ...agents.predictive_assistant.suggestion_engine import SuggestionEngine
from ...agents.pattern_recognition.pattern_detector import PatternDetector
from ..config.config import settings

logger = logging.getLogger(__name__)


class CollaborationService:
    """
    Central service for managing real-time collaboration features
    Coordinates session management, conflict resolution, awareness, sync, and AI predictions
    """
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
        
        # Initialize collaboration components
        self.session_manager = CollaborationSessionManager(redis_client)
        self.conflict_resolver = ConflictResolver(redis_client)
        self.awareness_engine = AwarenessEngine(redis_client)
        self.sync_coordinator = SyncCoordinator(
            redis_client, 
            self.conflict_resolver, 
            self.awareness_engine
        )
        
        # Initialize AI components for collaborative predictions
        self.context_analyzer = ContextAnalyzer()
        self.suggestion_engine = SuggestionEngine()
        self.pattern_detector = PatternDetector()
        
        self.collaborative_predictor = CollaborativePredictor(
            self.context_analyzer,
            self.suggestion_engine,
            self.pattern_detector,
            self.awareness_engine,
            redis_client
        )
        
        # Background tasks
        self._cleanup_task = None
        self._running = False
        
        logger.info("CollaborationService initialized")
    
    async def start(self) -> None:
        """
        Start the collaboration service and background tasks
        """
        if self._running:
            logger.warning("CollaborationService already running")
            return
        
        self._running = True
        
        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("CollaborationService started")
    
    async def stop(self) -> None:
        """
        Stop the collaboration service and cleanup resources
        """
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("CollaborationService stopped")
    
    async def _cleanup_loop(self) -> None:
        """
        Background task for periodic cleanup of expired data
        """
        while self._running:
            try:
                # Cleanup expired conflicts
                expired_conflicts = await self.conflict_resolver.cleanup_expired_conflicts()
                if expired_conflicts > 0:
                    logger.debug(f"Cleaned up {expired_conflicts} expired conflicts")
                
                # Cleanup inactive developers
                inactive_devs = await self.awareness_engine.cleanup_inactive_developers()
                if inactive_devs > 0:
                    logger.debug(f"Updated {inactive_devs} inactive developers")
                
                # Sleep for 5 minutes before next cleanup
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait a minute on error
    
    async def health_check(self) -> dict:
        """
        Perform health check on all collaboration components
        """
        health = {
            "session_manager": True,
            "conflict_resolver": True,
            "awareness_engine": True,
            "sync_coordinator": True,
            "collaborative_predictor": True,
            "redis": False
        }
        
        try:
            # Check Redis connection
            if self.redis:
                await self.redis.ping()
                health["redis"] = True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
        
        return health


# Dependency injection for FastAPI
_collaboration_service: Optional[CollaborationService] = None


async def get_collaboration_service() -> CollaborationService:
    """
    FastAPI dependency for getting the collaboration service
    """
    global _collaboration_service
    
    if _collaboration_service is None:
        # Initialize Redis client if available
        redis_client = None
        try:
            import redis.asyncio as redis
            
            # Try to get Redis configuration from settings
            redis_host = getattr(settings, 'REDIS_HOST', 'localhost')
            redis_port = getattr(settings, 'REDIS_PORT', 6379)
            redis_db = getattr(settings, 'REDIS_DB', 0)
            
            redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True
            )
            
            # Test connection
            await redis_client.ping()
            logger.info("Connected to Redis for collaboration service")
            
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}. Collaboration will work without real-time features.")
            redis_client = None
        
        _collaboration_service = CollaborationService(redis_client)
        await _collaboration_service.start()
    
    return _collaboration_service


@lru_cache()
def get_collaboration_service_sync() -> CollaborationService:
    """
    Synchronous version for non-async contexts
    Note: This won't have Redis connectivity
    """
    return CollaborationService(redis_client=None)