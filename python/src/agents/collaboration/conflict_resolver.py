"""
Real-time Collaboration Conflict Resolution
Handles merge conflicts and simultaneous edits in collaborative coding sessions
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import asyncio
import json
import logging
from difflib import SequenceMatcher
import hashlib

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Types of collaboration conflicts"""
    SIMULTANEOUS_EDIT = "simultaneous_edit"
    OVERLAPPING_CHANGES = "overlapping_changes"
    DELETION_CONFLICT = "deletion_conflict"
    INSERTION_CONFLICT = "insertion_conflict"
    SEMANTIC_CONFLICT = "semantic_conflict"
    LINE_MERGE_CONFLICT = "line_merge_conflict"


class ConflictResolutionStrategy(Enum):
    """Conflict resolution strategies"""
    MANUAL = "manual"
    AUTOMATIC_MERGE = "automatic_merge"
    FIRST_WINS = "first_wins"
    LAST_WINS = "last_wins"
    AI_ASSISTED = "ai_assisted"
    SEMANTIC_MERGE = "semantic_merge"


class MergeResult(Enum):
    """Results of merge operations"""
    SUCCESS = "success"
    CONFLICT = "conflict"
    ERROR = "error"
    REQUIRES_MANUAL = "requires_manual"


@dataclass
class CodeChange:
    """Represents a code change in collaborative editing"""
    change_id: str
    user_id: str
    session_id: str
    file_path: str
    start_line: int
    end_line: int
    start_column: int
    end_column: int
    old_content: str
    new_content: str
    timestamp: datetime
    operation: str  # "insert", "delete", "replace"
    context_hash: str = ""  # Hash of surrounding context
    
    def __post_init__(self):
        if not self.context_hash:
            # Generate context hash for conflict detection
            context = f"{self.file_path}:{self.start_line}:{self.old_content}"
            self.context_hash = hashlib.md5(context.encode()).hexdigest()[:8]


@dataclass
class Conflict:
    """Represents a merge conflict between collaborative changes"""
    conflict_id: str
    session_id: str
    conflict_type: ConflictType
    involved_changes: List[CodeChange]
    affected_lines: Tuple[int, int]
    file_path: str
    original_content: str
    conflicting_versions: List[str]
    timestamp: datetime
    resolution_strategy: Optional[ConflictResolutionStrategy] = None
    resolved: bool = False
    resolution: Optional[str] = None
    auto_resolvable: bool = False
    
    def __post_init__(self):
        # Determine if conflict can be automatically resolved
        self.auto_resolvable = self._can_auto_resolve()
    
    def _can_auto_resolve(self) -> bool:
        """Determine if conflict can be automatically resolved"""
        if self.conflict_type == ConflictType.INSERTION_CONFLICT:
            # Non-overlapping insertions can be auto-merged
            return True
        elif self.conflict_type == ConflictType.SIMULTANEOUS_EDIT:
            # Check if changes are identical
            if len(set(change.new_content for change in self.involved_changes)) == 1:
                return True
        return False


class ConflictResolutionRequest(BaseModel):
    """Request for conflict resolution"""
    conflict_id: str
    strategy: ConflictResolutionStrategy
    manual_resolution: Optional[str] = None
    user_preference: Optional[str] = None


class ConflictResolutionResult(BaseModel):
    """Result of conflict resolution"""
    conflict_id: str
    success: bool
    resolution: str
    merge_result: MergeResult
    applied_strategy: ConflictResolutionStrategy
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConflictResolver:
    """
    Advanced conflict resolution for real-time collaborative editing
    Handles various types of merge conflicts with multiple resolution strategies
    """
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.active_conflicts: Dict[str, Conflict] = {}
        self.resolution_history: List[ConflictResolutionResult] = []
        
        # Configuration
        self.auto_resolve_enabled = True
        self.similarity_threshold = 0.8  # For semantic conflict detection
        self.max_conflict_age_minutes = 30  # Auto-expire old conflicts
        
        logger.info("ConflictResolver initialized")
    
    async def detect_conflicts(
        self,
        incoming_change: CodeChange,
        existing_changes: List[CodeChange]
    ) -> List[Conflict]:
        """
        Detect conflicts between incoming change and existing changes
        """
        conflicts = []
        
        for existing_change in existing_changes:
            # Skip if same user or different file
            if (existing_change.user_id == incoming_change.user_id or
                existing_change.file_path != incoming_change.file_path):
                continue
            
            conflict = await self._analyze_conflict(incoming_change, existing_change)
            if conflict:
                conflicts.append(conflict)
                self.active_conflicts[conflict.conflict_id] = conflict
        
        return conflicts
    
    async def _analyze_conflict(
        self,
        change1: CodeChange,
        change2: CodeChange
    ) -> Optional[Conflict]:
        """
        Analyze two changes to determine if they conflict
        """
        # Check for line range overlap
        overlap = self._check_line_overlap(
            (change1.start_line, change1.end_line),
            (change2.start_line, change2.end_line)
        )
        
        if not overlap:
            # Check for semantic conflicts (changes that affect each other)
            if await self._check_semantic_conflict(change1, change2):
                return await self._create_conflict(
                    ConflictType.SEMANTIC_CONFLICT,
                    [change1, change2],
                    change1.file_path
                )
            return None
        
        # Determine conflict type based on operations and overlap
        if change1.operation == "delete" and change2.operation == "replace":
            conflict_type = ConflictType.DELETION_CONFLICT
        elif change1.operation == "insert" and change2.operation == "insert":
            conflict_type = ConflictType.INSERTION_CONFLICT
        elif self._are_changes_identical(change1, change2):
            # Same change by different users - not really a conflict
            return None
        else:
            conflict_type = ConflictType.OVERLAPPING_CHANGES
        
        return await self._create_conflict(
            conflict_type,
            [change1, change2],
            change1.file_path
        )
    
    def _check_line_overlap(
        self,
        range1: Tuple[int, int],
        range2: Tuple[int, int]
    ) -> bool:
        """Check if two line ranges overlap"""
        start1, end1 = range1
        start2, end2 = range2
        return not (end1 < start2 or end2 < start1)
    
    async def _check_semantic_conflict(
        self,
        change1: CodeChange,
        change2: CodeChange
    ) -> bool:
        """
        Check for semantic conflicts between changes
        (e.g., one changes function signature, another changes its usage)
        """
        # Simplified semantic analysis
        # In production, this would use AST analysis
        
        # Check if changes involve related code elements
        similarity = SequenceMatcher(
            None,
            change1.old_content,
            change2.old_content
        ).ratio()
        
        return similarity > self.similarity_threshold
    
    def _are_changes_identical(self, change1: CodeChange, change2: CodeChange) -> bool:
        """Check if two changes are identical"""
        return (
            change1.new_content == change2.new_content and
            change1.start_line == change2.start_line and
            change1.end_line == change2.end_line
        )
    
    async def _create_conflict(
        self,
        conflict_type: ConflictType,
        changes: List[CodeChange],
        file_path: str
    ) -> Conflict:
        """Create a conflict object from conflicting changes"""
        conflict_id = self._generate_conflict_id(changes)
        
        # Calculate affected line range
        all_lines = []
        for change in changes:
            all_lines.extend(range(change.start_line, change.end_line + 1))
        
        affected_lines = (min(all_lines), max(all_lines))
        
        # Extract conflicting versions
        conflicting_versions = [change.new_content for change in changes]
        
        # Use the first change's old content as original
        original_content = changes[0].old_content
        
        conflict = Conflict(
            conflict_id=conflict_id,
            session_id=changes[0].session_id,
            conflict_type=conflict_type,
            involved_changes=changes,
            affected_lines=affected_lines,
            file_path=file_path,
            original_content=original_content,
            conflicting_versions=conflicting_versions,
            timestamp=datetime.now(timezone.utc)
        )
        
        logger.info(
            f"Created conflict {conflict_id} of type {conflict_type.value} "
            f"in {file_path} affecting lines {affected_lines}"
        )
        
        return conflict
    
    def _generate_conflict_id(self, changes: List[CodeChange]) -> str:
        """Generate unique ID for conflict"""
        # Create deterministic ID based on changes
        change_data = "|".join([
            f"{c.user_id}:{c.timestamp.isoformat()}:{c.context_hash}"
            for c in sorted(changes, key=lambda x: x.timestamp)
        ])
        return hashlib.md5(change_data.encode()).hexdigest()[:12]
    
    async def resolve_conflict(
        self,
        conflict_id: str,
        strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.AUTOMATIC_MERGE,
        manual_resolution: Optional[str] = None
    ) -> ConflictResolutionResult:
        """
        Resolve a conflict using specified strategy
        """
        if conflict_id not in self.active_conflicts:
            return ConflictResolutionResult(
                conflict_id=conflict_id,
                success=False,
                resolution="",
                merge_result=MergeResult.ERROR,
                applied_strategy=strategy,
                metadata={"error": "Conflict not found"}
            )
        
        conflict = self.active_conflicts[conflict_id]
        
        try:
            if strategy == ConflictResolutionStrategy.MANUAL:
                result = await self._resolve_manual(conflict, manual_resolution)
            elif strategy == ConflictResolutionStrategy.AUTOMATIC_MERGE:
                result = await self._resolve_automatic_merge(conflict)
            elif strategy == ConflictResolutionStrategy.FIRST_WINS:
                result = await self._resolve_first_wins(conflict)
            elif strategy == ConflictResolutionStrategy.LAST_WINS:
                result = await self._resolve_last_wins(conflict)
            elif strategy == ConflictResolutionStrategy.AI_ASSISTED:
                result = await self._resolve_ai_assisted(conflict)
            elif strategy == ConflictResolutionStrategy.SEMANTIC_MERGE:
                result = await self._resolve_semantic_merge(conflict)
            else:
                result = ConflictResolutionResult(
                    conflict_id=conflict_id,
                    success=False,
                    resolution="",
                    merge_result=MergeResult.ERROR,
                    applied_strategy=strategy,
                    metadata={"error": "Unsupported resolution strategy"}
                )
            
            if result.success:
                conflict.resolved = True
                conflict.resolution = result.resolution
                conflict.resolution_strategy = strategy
                self.resolution_history.append(result)
                
                # Remove from active conflicts
                del self.active_conflicts[conflict_id]
            
            logger.info(
                f"Conflict {conflict_id} resolution attempt: "
                f"strategy={strategy.value}, success={result.success}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error resolving conflict {conflict_id}: {e}")
            return ConflictResolutionResult(
                conflict_id=conflict_id,
                success=False,
                resolution="",
                merge_result=MergeResult.ERROR,
                applied_strategy=strategy,
                metadata={"error": str(e)}
            )
    
    async def _resolve_manual(
        self,
        conflict: Conflict,
        manual_resolution: Optional[str]
    ) -> ConflictResolutionResult:
        """Resolve conflict with manual resolution"""
        if not manual_resolution:
            return ConflictResolutionResult(
                conflict_id=conflict.conflict_id,
                success=False,
                resolution="",
                merge_result=MergeResult.REQUIRES_MANUAL,
                applied_strategy=ConflictResolutionStrategy.MANUAL,
                metadata={"error": "Manual resolution required but not provided"}
            )
        
        return ConflictResolutionResult(
            conflict_id=conflict.conflict_id,
            success=True,
            resolution=manual_resolution,
            merge_result=MergeResult.SUCCESS,
            applied_strategy=ConflictResolutionStrategy.MANUAL
        )
    
    async def _resolve_automatic_merge(
        self,
        conflict: Conflict
    ) -> ConflictResolutionResult:
        """Attempt automatic three-way merge"""
        if not conflict.auto_resolvable:
            return ConflictResolutionResult(
                conflict_id=conflict.conflict_id,
                success=False,
                resolution="",
                merge_result=MergeResult.REQUIRES_MANUAL,
                applied_strategy=ConflictResolutionStrategy.AUTOMATIC_MERGE,
                metadata={"reason": "Not auto-resolvable"}
            )
        
        # Handle auto-resolvable conflicts
        if conflict.conflict_type == ConflictType.INSERTION_CONFLICT:
            # Merge all insertions
            merged_content = await self._merge_insertions(conflict)
        elif conflict.conflict_type == ConflictType.SIMULTANEOUS_EDIT:
            # Use first version if identical changes
            merged_content = conflict.conflicting_versions[0]
        else:
            return ConflictResolutionResult(
                conflict_id=conflict.conflict_id,
                success=False,
                resolution="",
                merge_result=MergeResult.REQUIRES_MANUAL,
                applied_strategy=ConflictResolutionStrategy.AUTOMATIC_MERGE
            )
        
        return ConflictResolutionResult(
            conflict_id=conflict.conflict_id,
            success=True,
            resolution=merged_content,
            merge_result=MergeResult.SUCCESS,
            applied_strategy=ConflictResolutionStrategy.AUTOMATIC_MERGE
        )
    
    async def _merge_insertions(self, conflict: Conflict) -> str:
        """Merge multiple insertion conflicts intelligently"""
        # Sort changes by line number and timestamp
        sorted_changes = sorted(
            conflict.involved_changes,
            key=lambda c: (c.start_line, c.timestamp)
        )
        
        # Combine insertions preserving order
        merged_lines = []
        for change in sorted_changes:
            merged_lines.append(change.new_content)
        
        return "\n".join(merged_lines)
    
    async def _resolve_first_wins(
        self,
        conflict: Conflict
    ) -> ConflictResolutionResult:
        """Resolve by taking the first change (earliest timestamp)"""
        earliest_change = min(
            conflict.involved_changes,
            key=lambda c: c.timestamp
        )
        
        return ConflictResolutionResult(
            conflict_id=conflict.conflict_id,
            success=True,
            resolution=earliest_change.new_content,
            merge_result=MergeResult.SUCCESS,
            applied_strategy=ConflictResolutionStrategy.FIRST_WINS,
            metadata={"winner": earliest_change.user_id}
        )
    
    async def _resolve_last_wins(
        self,
        conflict: Conflict
    ) -> ConflictResolutionResult:
        """Resolve by taking the last change (latest timestamp)"""
        latest_change = max(
            conflict.involved_changes,
            key=lambda c: c.timestamp
        )
        
        return ConflictResolutionResult(
            conflict_id=conflict.conflict_id,
            success=True,
            resolution=latest_change.new_content,
            merge_result=MergeResult.SUCCESS,
            applied_strategy=ConflictResolutionStrategy.LAST_WINS,
            metadata={"winner": latest_change.user_id}
        )
    
    async def _resolve_ai_assisted(
        self,
        conflict: Conflict
    ) -> ConflictResolutionResult:
        """
        Resolve conflict using AI assistance
        This would integrate with the knowledge graph and pattern recognition
        """
        # Placeholder for AI-assisted resolution
        # In production, this would:
        # 1. Analyze code context using AST
        # 2. Check against known patterns
        # 3. Use ML to suggest best merge
        # 4. Consider semantic meaning of changes
        
        # For now, fall back to automatic merge
        return await self._resolve_automatic_merge(conflict)
    
    async def _resolve_semantic_merge(
        self,
        conflict: Conflict
    ) -> ConflictResolutionResult:
        """
        Resolve conflict using semantic analysis of code changes
        """
        # Placeholder for semantic merge resolution
        # This would analyze the semantic meaning of changes and merge accordingly
        
        # For now, attempt automatic merge
        return await self._resolve_automatic_merge(conflict)
    
    async def get_active_conflicts(self, session_id: Optional[str] = None) -> List[Conflict]:
        """Get all active conflicts, optionally filtered by session"""
        conflicts = list(self.active_conflicts.values())
        
        if session_id:
            conflicts = [c for c in conflicts if c.session_id == session_id]
        
        return conflicts
    
    async def get_conflict_history(
        self,
        session_id: Optional[str] = None,
        limit: int = 50
    ) -> List[ConflictResolutionResult]:
        """Get conflict resolution history"""
        history = self.resolution_history[-limit:]
        
        # Filter by session if specified
        if session_id:
            # Note: We'd need to store session_id in resolution results
            # for this filtering to work properly
            pass
        
        return history
    
    async def cleanup_expired_conflicts(self) -> int:
        """Remove conflicts that have exceeded max age"""
        current_time = datetime.now(timezone.utc)
        expired_count = 0
        
        expired_ids = []
        for conflict_id, conflict in self.active_conflicts.items():
            age_minutes = (current_time - conflict.timestamp).total_seconds() / 60
            if age_minutes > self.max_conflict_age_minutes:
                expired_ids.append(conflict_id)
        
        for conflict_id in expired_ids:
            del self.active_conflicts[conflict_id]
            expired_count += 1
        
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired conflicts")
        
        return expired_count
    
    async def get_conflict_statistics(self) -> Dict[str, Any]:
        """Get statistics about conflicts and resolutions"""
        stats = {
            "active_conflicts": len(self.active_conflicts),
            "total_resolutions": len(self.resolution_history),
            "auto_resolvable_active": len([
                c for c in self.active_conflicts.values() if c.auto_resolvable
            ]),
            "conflict_types": {},
            "resolution_strategies": {},
            "success_rate": 0.0
        }
        
        # Count conflict types
        for conflict in self.active_conflicts.values():
            conflict_type = conflict.conflict_type.value
            stats["conflict_types"][conflict_type] = (
                stats["conflict_types"].get(conflict_type, 0) + 1
            )
        
        # Count resolution strategies and calculate success rate
        if self.resolution_history:
            successful = 0
            for result in self.resolution_history:
                strategy = result.applied_strategy.value
                stats["resolution_strategies"][strategy] = (
                    stats["resolution_strategies"].get(strategy, 0) + 1
                )
                if result.success:
                    successful += 1
            
            stats["success_rate"] = successful / len(self.resolution_history)
        
        return stats