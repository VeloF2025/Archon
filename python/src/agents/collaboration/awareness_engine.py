"""
Real-time Developer Awareness Engine
Tracks developer presence, cursor positions, selections, and activity status
Provides visual feedback for collaborative coding
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import logging
from collections import defaultdict

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class UserStatus(Enum):
    """Developer status in collaborative session"""
    ONLINE = "online"
    AWAY = "away"
    TYPING = "typing"
    SELECTING = "selecting"
    IDLE = "idle"
    OFFLINE = "offline"


class ActivityType(Enum):
    """Types of developer activities"""
    CURSOR_MOVE = "cursor_move"
    SELECTION_CHANGE = "selection_change"
    TEXT_EDIT = "text_edit"
    FILE_SWITCH = "file_switch"
    SCROLL = "scroll"
    FOCUS_CHANGE = "focus_change"
    TYPING_START = "typing_start"
    TYPING_STOP = "typing_stop"


@dataclass
class CursorPosition:
    """Represents a cursor position in code"""
    line: int
    column: int
    file_path: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "line": self.line,
            "column": self.column,
            "file_path": self.file_path,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class TextSelection:
    """Represents a text selection in code"""
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    file_path: str
    selected_text: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_line": self.start_line,
            "start_column": self.start_column,
            "end_line": self.end_line,
            "end_column": self.end_column,
            "file_path": self.file_path,
            "selected_text": self.selected_text,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class DeveloperActivity:
    """Represents developer activity in collaborative session"""
    user_id: str
    session_id: str
    activity_type: ActivityType
    timestamp: datetime
    file_path: str
    cursor_position: Optional[CursorPosition] = None
    selection: Optional[TextSelection] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "activity_type": self.activity_type.value,
            "timestamp": self.timestamp.isoformat(),
            "file_path": self.file_path,
            "cursor_position": self.cursor_position.to_dict() if self.cursor_position else None,
            "selection": self.selection.to_dict() if self.selection else None,
            "metadata": self.metadata
        }


@dataclass
class DeveloperPresence:
    """Current presence information for a developer"""
    user_id: str
    session_id: str
    display_name: str
    avatar_url: Optional[str] = None
    status: UserStatus = UserStatus.ONLINE
    current_file: Optional[str] = None
    cursor_position: Optional[CursorPosition] = None
    selection: Optional[TextSelection] = None
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    joined_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    color: str = "#007ACC"  # Visual indicator color
    typing: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "display_name": self.display_name,
            "avatar_url": self.avatar_url,
            "status": self.status.value,
            "current_file": self.current_file,
            "cursor_position": self.cursor_position.to_dict() if self.cursor_position else None,
            "selection": self.selection.to_dict() if self.selection else None,
            "last_activity": self.last_activity.isoformat(),
            "joined_at": self.joined_at.isoformat(),
            "color": self.color,
            "typing": self.typing
        }
    
    def is_active(self, threshold_minutes: int = 5) -> bool:
        """Check if developer is recently active"""
        threshold = datetime.now(timezone.utc) - timedelta(minutes=threshold_minutes)
        return self.last_activity > threshold


class AwarenessUpdate(BaseModel):
    """Update to developer awareness state"""
    user_id: str
    session_id: str
    cursor_position: Optional[Dict[str, Any]] = None
    selection: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    activity_type: str = "cursor_move"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AwarenessEngine:
    """
    Real-time developer awareness engine for collaborative coding
    Tracks cursors, selections, and activity across multiple developers
    """
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
        
        # Core state
        self.developer_presence: Dict[str, DeveloperPresence] = {}
        self.activity_history: List[DeveloperActivity] = []
        self.session_developers: Dict[str, Set[str]] = defaultdict(set)
        
        # Configuration
        self.activity_retention_hours = 24
        self.idle_timeout_minutes = 5
        self.away_timeout_minutes = 15
        self.max_history_per_user = 100
        
        # Color palette for developer indicators
        self.developer_colors = [
            "#007ACC", "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
            "#FECA57", "#FF9FF3", "#54A0FF", "#5F27CD", "#00D2D3"
        ]
        self.color_index = 0
        
        logger.info("AwarenessEngine initialized")
    
    async def join_session(
        self,
        user_id: str,
        session_id: str,
        display_name: str,
        avatar_url: Optional[str] = None
    ) -> DeveloperPresence:
        """
        Add developer to collaborative session awareness
        """
        # Assign unique color
        color = self.developer_colors[self.color_index % len(self.developer_colors)]
        self.color_index += 1
        
        presence = DeveloperPresence(
            user_id=user_id,
            session_id=session_id,
            display_name=display_name,
            avatar_url=avatar_url,
            color=color
        )
        
        # Store presence
        self.developer_presence[user_id] = presence
        self.session_developers[session_id].add(user_id)
        
        # Broadcast join event
        await self._broadcast_awareness_update(
            session_id, 
            user_id, 
            "user_joined",
            {"display_name": display_name, "color": color}
        )
        
        logger.info(f"Developer {display_name} ({user_id}) joined session {session_id}")
        
        return presence
    
    async def leave_session(self, user_id: str, session_id: str) -> bool:
        """
        Remove developer from collaborative session awareness
        """
        if user_id not in self.developer_presence:
            return False
        
        presence = self.developer_presence[user_id]
        presence.status = UserStatus.OFFLINE
        
        # Remove from session
        self.session_developers[session_id].discard(user_id)
        
        # Broadcast leave event
        await self._broadcast_awareness_update(
            session_id,
            user_id,
            "user_left",
            {"display_name": presence.display_name}
        )
        
        # Remove presence after broadcast
        del self.developer_presence[user_id]
        
        logger.info(f"Developer {presence.display_name} ({user_id}) left session {session_id}")
        
        return True
    
    async def update_awareness(self, update: AwarenessUpdate) -> bool:
        """
        Update developer awareness state (cursor, selection, activity)
        """
        if update.user_id not in self.developer_presence:
            logger.warning(f"Awareness update for unknown user: {update.user_id}")
            return False
        
        presence = self.developer_presence[update.user_id]
        current_time = datetime.now(timezone.utc)
        presence.last_activity = current_time
        
        # Update cursor position
        if update.cursor_position:
            presence.cursor_position = CursorPosition(
                line=update.cursor_position["line"],
                column=update.cursor_position["column"],
                file_path=update.cursor_position["file_path"],
                timestamp=current_time
            )
            presence.current_file = update.cursor_position["file_path"]
        
        # Update selection
        if update.selection:
            presence.selection = TextSelection(
                start_line=update.selection["start_line"],
                start_column=update.selection["start_column"],
                end_line=update.selection["end_line"],
                end_column=update.selection["end_column"],
                file_path=update.selection["file_path"],
                selected_text=update.selection.get("selected_text", ""),
                timestamp=current_time
            )
        
        # Update file if specified
        if update.file_path:
            presence.current_file = update.file_path
        
        # Update status based on activity
        activity_type = ActivityType(update.activity_type)
        if activity_type in [ActivityType.TEXT_EDIT, ActivityType.TYPING_START]:
            presence.status = UserStatus.TYPING
            presence.typing = True
        elif activity_type == ActivityType.TYPING_STOP:
            presence.status = UserStatus.ONLINE
            presence.typing = False
        elif activity_type == ActivityType.SELECTION_CHANGE:
            presence.status = UserStatus.SELECTING
        else:
            presence.status = UserStatus.ONLINE
        
        # Record activity
        activity = DeveloperActivity(
            user_id=update.user_id,
            session_id=update.session_id,
            activity_type=activity_type,
            timestamp=current_time,
            file_path=update.file_path or presence.current_file or "",
            cursor_position=presence.cursor_position,
            selection=presence.selection,
            metadata=update.metadata
        )
        
        await self._record_activity(activity)
        
        # Broadcast awareness update to session
        await self._broadcast_awareness_update(
            update.session_id,
            update.user_id,
            "awareness_update",
            {
                "presence": presence.to_dict(),
                "activity": activity.to_dict()
            }
        )
        
        return True
    
    async def _record_activity(self, activity: DeveloperActivity) -> None:
        """Record developer activity in history"""
        self.activity_history.append(activity)
        
        # Trim history to prevent memory growth
        if len(self.activity_history) > 1000:
            # Keep only recent activities
            cutoff_time = datetime.now(timezone.utc) - timedelta(
                hours=self.activity_retention_hours
            )
            self.activity_history = [
                a for a in self.activity_history 
                if a.timestamp > cutoff_time
            ]
    
    async def get_session_awareness(self, session_id: str) -> Dict[str, Any]:
        """
        Get complete awareness state for a session
        """
        if session_id not in self.session_developers:
            return {"developers": [], "active_files": []}
        
        developers = []
        active_files = set()
        
        for user_id in self.session_developers[session_id]:
            if user_id in self.developer_presence:
                presence = self.developer_presence[user_id]
                developers.append(presence.to_dict())
                
                if presence.current_file:
                    active_files.add(presence.current_file)
        
        return {
            "developers": developers,
            "active_files": list(active_files),
            "total_developers": len(developers),
            "active_developers": len([
                d for d in developers 
                if d["status"] in ["online", "typing", "selecting"]
            ])
        }
    
    async def get_file_awareness(
        self, 
        session_id: str, 
        file_path: str
    ) -> Dict[str, Any]:
        """
        Get awareness information for specific file
        """
        file_developers = []
        cursors = []
        selections = []
        
        if session_id not in self.session_developers:
            return {
                "developers": [],
                "cursors": [],
                "selections": [],
                "file_path": file_path
            }
        
        for user_id in self.session_developers[session_id]:
            if user_id not in self.developer_presence:
                continue
                
            presence = self.developer_presence[user_id]
            
            # Check if developer is active in this file
            if presence.current_file == file_path:
                file_developers.append({
                    "user_id": user_id,
                    "display_name": presence.display_name,
                    "color": presence.color,
                    "status": presence.status.value,
                    "typing": presence.typing
                })
                
                # Add cursor position
                if presence.cursor_position:
                    cursors.append({
                        **presence.cursor_position.to_dict(),
                        "user_id": user_id,
                        "display_name": presence.display_name,
                        "color": presence.color
                    })
                
                # Add selection
                if presence.selection:
                    selections.append({
                        **presence.selection.to_dict(),
                        "user_id": user_id,
                        "display_name": presence.display_name,
                        "color": presence.color
                    })
        
        return {
            "developers": file_developers,
            "cursors": cursors,
            "selections": selections,
            "file_path": file_path
        }
    
    async def get_developer_activity(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        hours: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get recent activity for a developer
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        activities = [
            activity.to_dict()
            for activity in self.activity_history
            if (activity.user_id == user_id and
                activity.timestamp > cutoff_time and
                (session_id is None or activity.session_id == session_id))
        ]
        
        return sorted(activities, key=lambda x: x["timestamp"], reverse=True)
    
    async def update_status(
        self, 
        user_id: str, 
        status: UserStatus
    ) -> bool:
        """
        Manually update developer status
        """
        if user_id not in self.developer_presence:
            return False
        
        presence = self.developer_presence[user_id]
        old_status = presence.status
        presence.status = status
        presence.last_activity = datetime.now(timezone.utc)
        
        # Special handling for typing status
        if status == UserStatus.TYPING:
            presence.typing = True
        elif old_status == UserStatus.TYPING:
            presence.typing = False
        
        # Broadcast status change
        await self._broadcast_awareness_update(
            presence.session_id,
            user_id,
            "status_change",
            {
                "old_status": old_status.value,
                "new_status": status.value,
                "display_name": presence.display_name
            }
        )
        
        return True
    
    async def cleanup_inactive_developers(self) -> int:
        """
        Clean up inactive developers and update their status
        """
        current_time = datetime.now(timezone.utc)
        updated_count = 0
        
        for user_id, presence in self.developer_presence.items():
            time_since_activity = current_time - presence.last_activity
            minutes_inactive = time_since_activity.total_seconds() / 60
            
            old_status = presence.status
            new_status = None
            
            if minutes_inactive > self.away_timeout_minutes:
                new_status = UserStatus.AWAY
            elif minutes_inactive > self.idle_timeout_minutes:
                new_status = UserStatus.IDLE
            elif presence.status in [UserStatus.AWAY, UserStatus.IDLE]:
                # Developer was inactive but might be active again
                continue
            
            if new_status and new_status != old_status:
                presence.status = new_status
                presence.typing = False  # Clear typing status
                updated_count += 1
                
                await self._broadcast_awareness_update(
                    presence.session_id,
                    user_id,
                    "status_change",
                    {
                        "old_status": old_status.value,
                        "new_status": new_status.value,
                        "display_name": presence.display_name,
                        "reason": "inactivity"
                    }
                )
        
        if updated_count > 0:
            logger.info(f"Updated status for {updated_count} inactive developers")
        
        return updated_count
    
    async def _broadcast_awareness_update(
        self,
        session_id: str,
        user_id: str,
        event_type: str,
        data: Dict[str, Any]
    ) -> None:
        """
        Broadcast awareness update to all session participants
        """
        if not self.redis:
            logger.debug(f"No Redis client - awareness update not broadcasted")
            return
        
        message = {
            "type": "awareness_update",
            "event": event_type,
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data
        }
        
        try:
            # Broadcast to session channel
            channel = f"session:{session_id}:awareness"
            await self.redis.publish(channel, json.dumps(message))
            
            logger.debug(f"Broadcasted {event_type} for {user_id} in session {session_id}")
        except Exception as e:
            logger.error(f"Failed to broadcast awareness update: {e}")
    
    async def get_awareness_statistics(
        self, 
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about developer awareness and activity
        """
        stats = {
            "total_developers": len(self.developer_presence),
            "online_developers": 0,
            "typing_developers": 0,
            "active_files": set(),
            "activity_counts": defaultdict(int),
            "status_distribution": defaultdict(int)
        }
        
        # Filter by session if specified
        developers = self.developer_presence.values()
        if session_id:
            developers = [
                p for p in developers 
                if p.session_id == session_id
            ]
            stats["session_id"] = session_id
        
        # Calculate statistics
        for presence in developers:
            stats["status_distribution"][presence.status.value] += 1
            
            if presence.status in [UserStatus.ONLINE, UserStatus.TYPING, UserStatus.SELECTING]:
                stats["online_developers"] += 1
            
            if presence.typing:
                stats["typing_developers"] += 1
            
            if presence.current_file:
                stats["active_files"].add(presence.current_file)
        
        # Count recent activity types
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        for activity in self.activity_history:
            if activity.timestamp > recent_cutoff:
                if not session_id or activity.session_id == session_id:
                    stats["activity_counts"][activity.activity_type.value] += 1
        
        # Convert sets to lists for JSON serialization
        stats["active_files"] = list(stats["active_files"])
        stats["activity_counts"] = dict(stats["activity_counts"])
        stats["status_distribution"] = dict(stats["status_distribution"])
        
        return stats