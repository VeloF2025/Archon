"""
Collaboration Session Manager
Manages real-time collaborative coding sessions with multiple developers
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import asyncio
import uuid
from pydantic import BaseModel, Field
import json

from ...server.services.redis_service import RedisService
from ..predictive_assistant.predictor import CodePredictor

logger = logging.getLogger(__name__)


class Developer(BaseModel):
    """Represents a developer in a collaboration session"""
    user_id: str
    display_name: str
    avatar_url: Optional[str] = None
    role: str = "developer"  # developer, lead, reviewer
    permissions: Set[str] = Field(default_factory=lambda: {"read", "write"})
    cursor_position: Optional[Tuple[int, int]] = None
    active_file: Optional[str] = None
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    status: str = "active"  # active, away, disconnected


class CollaborationSession(BaseModel):
    """Represents a collaborative coding session"""
    session_id: str
    project_id: str
    name: str
    description: Optional[str] = None
    owner_id: str
    developers: Dict[str, Developer] = Field(default_factory=dict)
    active_files: Set[str] = Field(default_factory=set)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    settings: Dict[str, Any] = Field(default_factory=dict)
    status: str = "active"  # active, paused, ended


class CodeChange(BaseModel):
    """Represents a code change in a collaboration session"""
    change_id: str
    session_id: str
    user_id: str
    file_path: str
    change_type: str  # insert, delete, replace
    position: Tuple[int, int]
    old_content: str = ""
    new_content: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    applied: bool = False


class CollaborationSessionManager:
    """Manages collaborative coding sessions"""
    
    def __init__(self, redis_service: Optional[RedisService] = None):
        self.redis = redis_service or RedisService()
        self.active_sessions: Dict[str, CollaborationSession] = {}
        self.change_buffer: Dict[str, List[CodeChange]] = {}
        self.predictor = CodePredictor()
        
        # Session configuration
        self.max_developers_per_session = 10
        self.session_timeout = timedelta(hours=4)
        self.change_buffer_size = 1000
    
    async def create_session(
        self,
        project_id: str,
        owner_id: str,
        name: str,
        description: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> CollaborationSession:
        """
        Create a new collaboration session
        
        Args:
            project_id: Project identifier
            owner_id: Session owner user ID
            name: Session name
            description: Optional description
            settings: Session settings
            
        Returns:
            Created collaboration session
        """
        session_id = str(uuid.uuid4())
        
        # Create owner developer
        owner = Developer(
            user_id=owner_id,
            display_name=f"User-{owner_id}",
            role="lead",
            permissions={"read", "write", "admin"}
        )
        
        session = CollaborationSession(
            session_id=session_id,
            project_id=project_id,
            name=name,
            description=description,
            owner_id=owner_id,
            developers={owner_id: owner},
            settings=settings or self._get_default_settings()
        )
        
        # Store in memory and Redis
        self.active_sessions[session_id] = session
        await self._store_session_in_redis(session)
        
        # Initialize change buffer
        self.change_buffer[session_id] = []
        
        logger.info(f"Created collaboration session {session_id} for project {project_id}")
        return session
    
    async def join_session(
        self,
        session_id: str,
        user_id: str,
        display_name: str,
        avatar_url: Optional[str] = None
    ) -> bool:
        """
        Join an existing collaboration session
        
        Args:
            session_id: Session to join
            user_id: User identifier
            display_name: User display name
            avatar_url: Optional avatar URL
            
        Returns:
            True if joined successfully
        """
        session = await self._get_session(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found")
            return False
        
        # Check if session is full
        if len(session.developers) >= self.max_developers_per_session:
            logger.warning(f"Session {session_id} is full")
            return False
        
        # Check if user already in session
        if user_id in session.developers:
            # Update existing developer
            session.developers[user_id].status = "active"
            session.developers[user_id].last_activity = datetime.utcnow()
        else:
            # Add new developer
            developer = Developer(
                user_id=user_id,
                display_name=display_name,
                avatar_url=avatar_url,
                role="developer",
                permissions={"read", "write"}
            )
            session.developers[user_id] = developer
        
        session.last_activity = datetime.utcnow()
        
        # Update in storage
        self.active_sessions[session_id] = session
        await self._store_session_in_redis(session)
        
        # Notify other developers
        await self._broadcast_session_event(session_id, {
            "type": "developer_joined",
            "user_id": user_id,
            "display_name": display_name,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"User {user_id} joined session {session_id}")
        return True
    
    async def leave_session(
        self,
        session_id: str,
        user_id: str
    ) -> bool:
        """
        Leave a collaboration session
        
        Args:
            session_id: Session to leave
            user_id: User identifier
            
        Returns:
            True if left successfully
        """
        session = await self._get_session(session_id)
        if not session or user_id not in session.developers:
            return False
        
        # Mark as disconnected instead of removing
        session.developers[user_id].status = "disconnected"
        session.developers[user_id].last_activity = datetime.utcnow()
        session.last_activity = datetime.utcnow()
        
        # Update in storage
        await self._store_session_in_redis(session)
        
        # Notify other developers
        await self._broadcast_session_event(session_id, {
            "type": "developer_left",
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Check if session should be paused/ended
        active_developers = [
            d for d in session.developers.values() 
            if d.status == "active"
        ]
        
        if len(active_developers) == 0:
            session.status = "paused"
            await self._store_session_in_redis(session)
        
        logger.info(f"User {user_id} left session {session_id}")
        return True
    
    async def apply_code_change(
        self,
        session_id: str,
        user_id: str,
        file_path: str,
        change_type: str,
        position: Tuple[int, int],
        old_content: str = "",
        new_content: str = ""
    ) -> CodeChange:
        """
        Apply a code change to the collaboration session
        
        Args:
            session_id: Session identifier
            user_id: User making the change
            file_path: File being modified
            change_type: Type of change (insert, delete, replace)
            position: Position in file (line, column)
            old_content: Content being replaced/deleted
            new_content: New content being inserted
            
        Returns:
            Applied code change
        """
        change_id = str(uuid.uuid4())
        
        change = CodeChange(
            change_id=change_id,
            session_id=session_id,
            user_id=user_id,
            file_path=file_path,
            change_type=change_type,
            position=position,
            old_content=old_content,
            new_content=new_content
        )
        
        # Add to change buffer
        if session_id not in self.change_buffer:
            self.change_buffer[session_id] = []
        
        self.change_buffer[session_id].append(change)
        
        # Limit buffer size
        if len(self.change_buffer[session_id]) > self.change_buffer_size:
            self.change_buffer[session_id] = self.change_buffer[session_id][-self.change_buffer_size:]
        
        # Update session
        session = await self._get_session(session_id)
        if session:
            session.active_files.add(file_path)
            session.last_activity = datetime.utcnow()
            await self._store_session_in_redis(session)
        
        # Broadcast change to other developers
        await self._broadcast_code_change(change)
        
        # Store change in Redis for persistence
        await self._store_change_in_redis(change)
        
        logger.debug(f"Applied code change {change_id} in session {session_id}")
        return change
    
    async def get_collaborative_suggestions(
        self,
        session_id: str,
        user_id: str,
        code: str,
        cursor_position: Tuple[int, int],
        file_path: str
    ) -> List[Any]:
        """
        Get AI suggestions enhanced by collaborative context
        
        Args:
            session_id: Session identifier
            user_id: User requesting suggestions
            code: Current code
            cursor_position: Cursor position
            file_path: File path
            
        Returns:
            Enhanced suggestions with collaborative context
        """
        # Get base predictions
        base_suggestions = await self.predictor.predict(
            code=code,
            cursor_position=cursor_position,
            file_path=file_path
        )
        
        # Get collaborative context
        session = await self._get_session(session_id)
        if not session:
            return base_suggestions
        
        # Enhance with collaborative context
        enhanced_suggestions = []
        
        for suggestion in base_suggestions:
            # Add collaborative metadata
            suggestion.metadata["collaborative"] = True
            suggestion.metadata["session_id"] = session_id
            suggestion.metadata["other_developers"] = len([
                d for d in session.developers.values() 
                if d.status == "active" and d.user_id != user_id
            ])
            
            # Check if other developers are working on similar code
            similar_activity = await self._find_similar_activity(
                session_id, file_path, cursor_position
            )
            
            if similar_activity:
                suggestion.metadata["similar_activity"] = similar_activity
                suggestion.confidence *= 1.1  # Boost confidence
            
            enhanced_suggestions.append(suggestion)
        
        return enhanced_suggestions
    
    async def update_developer_cursor(
        self,
        session_id: str,
        user_id: str,
        file_path: str,
        cursor_position: Tuple[int, int]
    ):
        """
        Update developer's cursor position
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            file_path: Active file
            cursor_position: Cursor position
        """
        session = await self._get_session(session_id)
        if not session or user_id not in session.developers:
            return
        
        # Update developer info
        developer = session.developers[user_id]
        developer.cursor_position = cursor_position
        developer.active_file = file_path
        developer.last_activity = datetime.utcnow()
        
        # Update session
        session.last_activity = datetime.utcnow()
        await self._store_session_in_redis(session)
        
        # Broadcast cursor update
        await self._broadcast_session_event(session_id, {
            "type": "cursor_update",
            "user_id": user_id,
            "file_path": file_path,
            "cursor_position": cursor_position,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def get_session_state(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get complete session state
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session state or None if not found
        """
        session = await self._get_session(session_id)
        if not session:
            return None
        
        # Get recent changes
        recent_changes = self.change_buffer.get(session_id, [])[-50:]  # Last 50 changes
        
        return {
            "session": session.dict(),
            "recent_changes": [c.dict() for c in recent_changes],
            "active_developers": [
                d.dict() for d in session.developers.values() 
                if d.status == "active"
            ],
            "session_metrics": await self._calculate_session_metrics(session_id)
        }
    
    async def end_session(
        self,
        session_id: str,
        user_id: str
    ) -> bool:
        """
        End a collaboration session
        
        Args:
            session_id: Session to end
            user_id: User ending the session
            
        Returns:
            True if ended successfully
        """
        session = await self._get_session(session_id)
        if not session:
            return False
        
        # Check permissions
        if user_id != session.owner_id and "admin" not in session.developers[user_id].permissions:
            logger.warning(f"User {user_id} not authorized to end session {session_id}")
            return False
        
        # Update session status
        session.status = "ended"
        session.last_activity = datetime.utcnow()
        
        # Store final state
        await self._store_session_in_redis(session)
        
        # Notify all developers
        await self._broadcast_session_event(session_id, {
            "type": "session_ended",
            "ended_by": user_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Archive session data
        await self._archive_session(session_id)
        
        # Clean up
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        if session_id in self.change_buffer:
            del self.change_buffer[session_id]
        
        logger.info(f"Session {session_id} ended by user {user_id}")
        return True
    
    async def _get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Get session from memory or Redis"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Try to load from Redis
        session_data = await self.redis.get(f"collaboration:session:{session_id}")
        if session_data:
            session = CollaborationSession.parse_raw(session_data)
            self.active_sessions[session_id] = session
            return session
        
        return None
    
    async def _store_session_in_redis(self, session: CollaborationSession):
        """Store session in Redis"""
        await self.redis.set(
            f"collaboration:session:{session.session_id}",
            session.json(),
            expire=int(self.session_timeout.total_seconds())
        )
    
    async def _store_change_in_redis(self, change: CodeChange):
        """Store code change in Redis"""
        await self.redis.lpush(
            f"collaboration:changes:{change.session_id}",
            change.json()
        )
        
        # Limit stored changes
        await self.redis.ltrim(
            f"collaboration:changes:{change.session_id}",
            0,
            self.change_buffer_size - 1
        )
    
    async def _broadcast_session_event(self, session_id: str, event: Dict[str, Any]):
        """Broadcast session event to all developers"""
        await self.redis.publish(
            f"collaboration:session:{session_id}:events",
            json.dumps(event)
        )
    
    async def _broadcast_code_change(self, change: CodeChange):
        """Broadcast code change to session developers"""
        await self.redis.publish(
            f"collaboration:session:{change.session_id}:changes",
            change.json()
        )
    
    async def _find_similar_activity(
        self,
        session_id: str,
        file_path: str,
        cursor_position: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """Find similar activity by other developers"""
        session = await self._get_session(session_id)
        if not session:
            return []
        
        similar = []
        target_line = cursor_position[0]
        
        for dev in session.developers.values():
            if (dev.status == "active" and 
                dev.active_file == file_path and 
                dev.cursor_position):
                
                dev_line = dev.cursor_position[0]
                if abs(dev_line - target_line) <= 5:  # Within 5 lines
                    similar.append({
                        "user_id": dev.user_id,
                        "display_name": dev.display_name,
                        "cursor_position": dev.cursor_position,
                        "distance": abs(dev_line - target_line)
                    })
        
        return similar
    
    async def _calculate_session_metrics(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """Calculate session metrics"""
        session = await self._get_session(session_id)
        if not session:
            return {}
        
        changes = self.change_buffer.get(session_id, [])
        
        return {
            "total_developers": len(session.developers),
            "active_developers": len([
                d for d in session.developers.values() 
                if d.status == "active"
            ]),
            "total_changes": len(changes),
            "files_modified": len(session.active_files),
            "session_duration": (datetime.utcnow() - session.created_at).total_seconds(),
            "last_activity": session.last_activity.isoformat()
        }
    
    async def _archive_session(self, session_id: str):
        """Archive session data for future analysis"""
        session = await self._get_session(session_id)
        changes = self.change_buffer.get(session_id, [])
        
        archive_data = {
            "session": session.dict() if session else None,
            "changes": [c.dict() for c in changes],
            "archived_at": datetime.utcnow().isoformat()
        }
        
        await self.redis.set(
            f"collaboration:archive:{session_id}",
            json.dumps(archive_data),
            expire=86400 * 30  # Keep for 30 days
        )
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default session settings"""
        return {
            "auto_save": True,
            "conflict_resolution": "automatic",
            "ai_assistance": "enabled",
            "change_notifications": True,
            "cursor_sharing": True,
            "voice_chat": False,
            "screen_sharing": False
        }