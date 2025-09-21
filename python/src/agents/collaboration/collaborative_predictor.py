"""
Collaborative AI Predictor
Provides enhanced AI suggestions based on team context and collaborative patterns
Integrates with existing pattern recognition and predictive assistant systems
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import logging
from collections import defaultdict, Counter

from pydantic import BaseModel, Field

from ..predictive_assistant.context_analyzer import ContextAnalyzer, CodeContext
from ..predictive_assistant.suggestion_engine import SuggestionEngine, CodeSuggestion
from ..pattern_recognition.pattern_detector import PatternDetector
from .awareness_engine import AwarenessEngine, DeveloperPresence
from .session_manager import CollaborationSession

logger = logging.getLogger(__name__)


class CollaborativeContext(Enum):
    """Types of collaborative context"""
    TEAM_PATTERNS = "team_patterns"
    SHARED_CODING_STYLE = "shared_coding_style"
    COLLABORATIVE_REFACTORING = "collaborative_refactoring"
    PAIR_PROGRAMMING = "pair_programming"
    CODE_REVIEW_CONTEXT = "code_review_context"
    KNOWLEDGE_SHARING = "knowledge_sharing"


class PredictionScope(Enum):
    """Scope of collaborative predictions"""
    INDIVIDUAL = "individual"
    TEAM = "team" 
    SESSION = "session"
    PROJECT = "project"
    GLOBAL = "global"


@dataclass
class TeamPattern:
    """Pattern observed in team coding behavior"""
    pattern_id: str
    pattern_type: str
    description: str
    frequency: int
    team_members: Set[str]
    code_examples: List[str]
    confidence_score: float
    last_observed: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "frequency": self.frequency,
            "team_members": list(self.team_members),
            "code_examples": self.code_examples,
            "confidence_score": self.confidence_score,
            "last_observed": self.last_observed.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class CollaborativeSuggestion:
    """AI suggestion enhanced with collaborative context"""
    suggestion_id: str
    base_suggestion: CodeSuggestion
    collaborative_context: CollaborativeContext
    team_relevance_score: float
    similar_team_implementations: List[str]
    collaborative_metadata: Dict[str, Any]
    suggested_by_teammates: List[str] = field(default_factory=list)
    usage_by_team: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = self.base_suggestion.to_dict() if hasattr(self.base_suggestion, 'to_dict') else {}
        return {
            **base_dict,
            "suggestion_id": self.suggestion_id,
            "collaborative_context": self.collaborative_context.value,
            "team_relevance_score": self.team_relevance_score,
            "similar_team_implementations": self.similar_team_implementations,
            "collaborative_metadata": self.collaborative_metadata,
            "suggested_by_teammates": self.suggested_by_teammates,
            "usage_by_team": self.usage_by_team
        }


class CollaborativePredictionRequest(BaseModel):
    """Request for collaborative AI predictions"""
    session_id: str
    user_id: str
    file_path: str
    code_context: str
    cursor_position: Tuple[int, int]
    current_selection: Optional[str] = None
    prediction_scope: PredictionScope = PredictionScope.SESSION
    include_team_patterns: bool = True
    max_suggestions: int = 10


class CollaborativePredictor:
    """
    AI predictor enhanced with collaborative context and team patterns
    Provides suggestions based on team coding patterns and shared context
    """
    
    def __init__(
        self,
        context_analyzer: Optional[ContextAnalyzer] = None,
        suggestion_engine: Optional[SuggestionEngine] = None,
        pattern_detector: Optional[PatternDetector] = None,
        awareness_engine: Optional[AwarenessEngine] = None,
        redis_client=None
    ):
        self.context_analyzer = context_analyzer or ContextAnalyzer()
        self.suggestion_engine = suggestion_engine or SuggestionEngine()
        self.pattern_detector = pattern_detector or PatternDetector()
        self.awareness_engine = awareness_engine or AwarenessEngine(redis_client)
        self.redis = redis_client
        
        # Team pattern storage
        self.team_patterns: Dict[str, List[TeamPattern]] = defaultdict(list)
        self.coding_style_patterns: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.collaborative_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Cache for performance
        self.suggestion_cache: Dict[str, List[CollaborativeSuggestion]] = {}
        self.team_context_cache: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.cache_expiry_minutes = 5
        self.min_team_pattern_frequency = 3
        self.team_relevance_threshold = 0.6
        
        logger.info("CollaborativePredictor initialized")
    
    async def get_collaborative_suggestions(
        self,
        request: CollaborativePredictionRequest
    ) -> List[CollaborativeSuggestion]:
        """
        Get AI suggestions enhanced with collaborative context
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(request)
            
            # Check cache first
            if cache_key in self.suggestion_cache:
                cached_suggestions = self.suggestion_cache[cache_key]
                if self._is_cache_valid(cache_key):
                    logger.debug(f"Returning cached suggestions for {request.user_id}")
                    return cached_suggestions
            
            # Get base context analysis
            code_context = await self.context_analyzer.analyze_context(
                code=request.code_context,
                cursor_position=request.cursor_position,
                file_path=request.file_path
            )
            
            # Get base suggestions
            base_suggestions = await self.suggestion_engine.generate_suggestions(
                context=code_context,
                max_suggestions=request.max_suggestions * 2  # Get more for filtering
            )
            
            # Get collaborative context
            collaborative_context = await self._build_collaborative_context(
                request.session_id,
                request.user_id,
                code_context
            )
            
            # Enhance suggestions with collaborative data
            enhanced_suggestions = await self._enhance_with_collaboration(
                base_suggestions,
                collaborative_context,
                request
            )
            
            # Apply team filtering and ranking
            filtered_suggestions = await self._apply_team_filtering(
                enhanced_suggestions,
                collaborative_context,
                request.max_suggestions
            )
            
            # Cache results
            self.suggestion_cache[cache_key] = filtered_suggestions
            
            logger.info(
                f"Generated {len(filtered_suggestions)} collaborative suggestions "
                f"for {request.user_id} in session {request.session_id}"
            )
            
            return filtered_suggestions
            
        except Exception as e:
            logger.error(f"Error generating collaborative suggestions: {e}")
            return []
    
    async def _build_collaborative_context(
        self,
        session_id: str,
        user_id: str,
        code_context: CodeContext
    ) -> Dict[str, Any]:
        """
        Build collaborative context from team activity and patterns
        """
        context = {
            "session_id": session_id,
            "user_id": user_id,
            "team_members": [],
            "active_developers": [],
            "team_patterns": [],
            "shared_coding_style": {},
            "recent_team_activity": [],
            "similar_code_contexts": []
        }
        
        # Get session awareness
        session_awareness = await self.awareness_engine.get_session_awareness(session_id)
        context["team_members"] = [dev["user_id"] for dev in session_awareness.get("developers", [])]
        context["active_developers"] = [
            dev["user_id"] for dev in session_awareness.get("developers", [])
            if dev["status"] in ["online", "typing", "selecting"]
        ]
        
        # Get team patterns for this session/project
        if session_id in self.team_patterns:
            context["team_patterns"] = [
                pattern.to_dict() for pattern in self.team_patterns[session_id]
            ]
        
        # Get shared coding style patterns
        if session_id in self.coding_style_patterns:
            context["shared_coding_style"] = self.coding_style_patterns[session_id]
        
        # Get recent collaborative activity
        context["recent_team_activity"] = await self._get_recent_team_activity(
            session_id, hours=2
        )
        
        # Find similar code contexts from team history
        context["similar_code_contexts"] = await self._find_similar_team_contexts(
            code_context, session_id
        )
        
        return context
    
    async def _enhance_with_collaboration(
        self,
        base_suggestions: List[CodeSuggestion],
        collaborative_context: Dict[str, Any],
        request: CollaborativePredictionRequest
    ) -> List[CollaborativeSuggestion]:
        """
        Enhance base suggestions with collaborative context
        """
        enhanced_suggestions = []
        
        for suggestion in base_suggestions:
            # Calculate team relevance score
            team_relevance = await self._calculate_team_relevance(
                suggestion, collaborative_context
            )
            
            # Find similar team implementations
            similar_implementations = await self._find_similar_team_implementations(
                suggestion, collaborative_context
            )
            
            # Determine collaborative context type
            collab_context_type = await self._determine_collaborative_context(
                suggestion, collaborative_context, request
            )
            
            # Get teammates who used similar patterns
            suggested_by_teammates = await self._get_teammates_with_similar_usage(
                suggestion, collaborative_context
            )
            
            # Count team usage frequency
            usage_by_team = await self._count_team_usage(
                suggestion, collaborative_context
            )
            
            # Create enhanced suggestion
            enhanced_suggestion = CollaborativeSuggestion(
                suggestion_id=f"collab_{suggestion.suggestion_id if hasattr(suggestion, 'suggestion_id') else 'unknown'}",
                base_suggestion=suggestion,
                collaborative_context=collab_context_type,
                team_relevance_score=team_relevance,
                similar_team_implementations=similar_implementations,
                collaborative_metadata={
                    "team_size": len(collaborative_context["team_members"]),
                    "active_developers": len(collaborative_context["active_developers"]),
                    "similar_contexts_found": len(collaborative_context["similar_code_contexts"])
                },
                suggested_by_teammates=suggested_by_teammates,
                usage_by_team=usage_by_team
            )
            
            enhanced_suggestions.append(enhanced_suggestion)
        
        return enhanced_suggestions
    
    async def _calculate_team_relevance(
        self,
        suggestion: CodeSuggestion,
        collaborative_context: Dict[str, Any]
    ) -> float:
        """
        Calculate how relevant a suggestion is to the current team
        """
        relevance_score = 0.0
        max_score = 1.0
        
        # Base relevance from suggestion confidence
        if hasattr(suggestion, 'confidence'):
            relevance_score += suggestion.confidence * 0.3
        else:
            relevance_score += 0.3
        
        # Team pattern matching
        team_patterns = collaborative_context.get("team_patterns", [])
        if team_patterns:
            pattern_matches = 0
            for pattern in team_patterns:
                if self._suggestion_matches_pattern(suggestion, pattern):
                    pattern_matches += 1
            
            if pattern_matches > 0:
                relevance_score += min(pattern_matches / len(team_patterns), 0.4)
        
        # Coding style consistency
        coding_style = collaborative_context.get("shared_coding_style", {})
        if coding_style and self._matches_coding_style(suggestion, coding_style):
            relevance_score += 0.2
        
        # Similar context usage by team
        similar_contexts = collaborative_context.get("similar_code_contexts", [])
        if similar_contexts:
            relevance_score += min(len(similar_contexts) * 0.1, 0.1)
        
        return min(relevance_score, max_score)
    
    def _suggestion_matches_pattern(
        self, 
        suggestion: CodeSuggestion, 
        pattern: Dict[str, Any]
    ) -> bool:
        """
        Check if a suggestion matches a team pattern
        """
        # Simplified pattern matching
        # In production, this would use more sophisticated AST analysis
        
        suggestion_text = getattr(suggestion, 'content', '') or getattr(suggestion, 'text', '')
        pattern_examples = pattern.get("code_examples", [])
        
        for example in pattern_examples:
            if len(set(suggestion_text.lower().split()) & set(example.lower().split())) > 2:
                return True
        
        return False
    
    def _matches_coding_style(
        self, 
        suggestion: CodeSuggestion, 
        coding_style: Dict[str, Any]
    ) -> bool:
        """
        Check if suggestion matches team coding style
        """
        # Simple coding style checks
        # In production, this would be more comprehensive
        
        suggestion_text = getattr(suggestion, 'content', '') or getattr(suggestion, 'text', '')
        
        # Check indentation style
        if "indentation" in coding_style:
            preferred_indent = coding_style["indentation"]
            if preferred_indent == "spaces" and "\t" in suggestion_text:
                return False
            elif preferred_indent == "tabs" and "    " in suggestion_text:
                return False
        
        # Check naming conventions
        if "naming_convention" in coding_style:
            # Simplified check - would be more sophisticated in production
            return True
        
        return True
    
    async def _find_similar_team_implementations(
        self,
        suggestion: CodeSuggestion,
        collaborative_context: Dict[str, Any]
    ) -> List[str]:
        """
        Find similar implementations used by team members
        """
        similar_implementations = []
        
        # Search through collaborative history
        session_id = collaborative_context["session_id"]
        if session_id in self.collaborative_history:
            history = self.collaborative_history[session_id]
            
            suggestion_text = getattr(suggestion, 'content', '') or getattr(suggestion, 'text', '')
            suggestion_keywords = set(suggestion_text.lower().split())
            
            for entry in history:
                if "code_content" in entry:
                    entry_keywords = set(entry["code_content"].lower().split())
                    similarity = len(suggestion_keywords & entry_keywords) / max(len(suggestion_keywords), 1)
                    
                    if similarity > 0.3:
                        similar_implementations.append(entry["code_content"][:100])
        
        return similar_implementations[:5]  # Limit to top 5
    
    async def _determine_collaborative_context(
        self,
        suggestion: CodeSuggestion,
        collaborative_context: Dict[str, Any],
        request: CollaborativePredictionRequest
    ) -> CollaborativeContext:
        """
        Determine the type of collaborative context for the suggestion
        """
        active_devs = len(collaborative_context["active_developers"])
        
        if active_devs >= 2:
            return CollaborativeContext.PAIR_PROGRAMMING
        elif len(collaborative_context["team_patterns"]) > 0:
            return CollaborativeContext.TEAM_PATTERNS
        elif collaborative_context.get("shared_coding_style"):
            return CollaborativeContext.SHARED_CODING_STYLE
        elif len(collaborative_context["similar_code_contexts"]) > 0:
            return CollaborativeContext.KNOWLEDGE_SHARING
        else:
            return CollaborativeContext.TEAM_PATTERNS
    
    async def _get_teammates_with_similar_usage(
        self,
        suggestion: CodeSuggestion,
        collaborative_context: Dict[str, Any]
    ) -> List[str]:
        """
        Get teammates who have used similar code patterns
        """
        teammates = []
        
        # Search collaborative history for similar usage
        session_id = collaborative_context["session_id"]
        if session_id in self.collaborative_history:
            suggestion_text = getattr(suggestion, 'content', '') or getattr(suggestion, 'text', '')
            
            for entry in self.collaborative_history[session_id]:
                if (entry.get("user_id") != collaborative_context["user_id"] and
                    "code_content" in entry):
                    
                    if self._code_similarity(suggestion_text, entry["code_content"]) > 0.4:
                        if entry["user_id"] not in teammates:
                            teammates.append(entry["user_id"])
        
        return teammates[:3]  # Limit to top 3
    
    def _code_similarity(self, code1: str, code2: str) -> float:
        """
        Calculate similarity between two code snippets
        """
        words1 = set(code1.lower().split())
        words2 = set(code2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    async def _count_team_usage(
        self,
        suggestion: CodeSuggestion,
        collaborative_context: Dict[str, Any]
    ) -> int:
        """
        Count how many times team has used similar patterns
        """
        usage_count = 0
        
        session_id = collaborative_context["session_id"]
        if session_id in self.collaborative_history:
            suggestion_text = getattr(suggestion, 'content', '') or getattr(suggestion, 'text', '')
            
            for entry in self.collaborative_history[session_id]:
                if "code_content" in entry:
                    if self._code_similarity(suggestion_text, entry["code_content"]) > 0.5:
                        usage_count += 1
        
        return usage_count
    
    async def _apply_team_filtering(
        self,
        suggestions: List[CollaborativeSuggestion],
        collaborative_context: Dict[str, Any],
        max_suggestions: int
    ) -> List[CollaborativeSuggestion]:
        """
        Filter and rank suggestions based on team relevance
        """
        # Filter by minimum team relevance threshold
        relevant_suggestions = [
            s for s in suggestions 
            if s.team_relevance_score >= self.team_relevance_threshold
        ]
        
        # If not enough relevant suggestions, include lower-scored ones
        if len(relevant_suggestions) < max_suggestions // 2:
            additional_suggestions = [
                s for s in suggestions 
                if s not in relevant_suggestions
            ][:max_suggestions - len(relevant_suggestions)]
            relevant_suggestions.extend(additional_suggestions)
        
        # Sort by team relevance and usage
        def sort_key(suggestion):
            return (
                suggestion.team_relevance_score,
                suggestion.usage_by_team,
                len(suggestion.suggested_by_teammates)
            )
        
        sorted_suggestions = sorted(relevant_suggestions, key=sort_key, reverse=True)
        
        return sorted_suggestions[:max_suggestions]
    
    async def record_team_activity(
        self,
        session_id: str,
        user_id: str,
        activity_data: Dict[str, Any]
    ) -> None:
        """
        Record team activity for collaborative learning
        """
        activity_entry = {
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **activity_data
        }
        
        self.collaborative_history[session_id].append(activity_entry)
        
        # Limit history size
        if len(self.collaborative_history[session_id]) > 1000:
            self.collaborative_history[session_id] = self.collaborative_history[session_id][-500:]
        
        # Extract and update team patterns
        await self._update_team_patterns(session_id, activity_entry)
    
    async def _update_team_patterns(
        self,
        session_id: str,
        activity: Dict[str, Any]
    ) -> None:
        """
        Extract and update team patterns from activity
        """
        if "code_content" not in activity:
            return
        
        try:
            # Detect patterns in the code
            patterns = await self.pattern_detector.detect_patterns(
                activity["code_content"]
            )
            
            for pattern in patterns:
                # Find or create team pattern
                team_pattern = await self._find_or_create_team_pattern(
                    session_id, pattern
                )
                
                # Update pattern data
                team_pattern.frequency += 1
                team_pattern.team_members.add(activity["user_id"])
                team_pattern.last_observed = datetime.now(timezone.utc)
                
                # Add code example if not already present
                code_snippet = activity["code_content"][:200]
                if code_snippet not in team_pattern.code_examples:
                    team_pattern.code_examples.append(code_snippet)
                    
                    # Limit examples
                    if len(team_pattern.code_examples) > 10:
                        team_pattern.code_examples = team_pattern.code_examples[-10:]
                
                # Update confidence based on frequency and team size
                team_size = len(team_pattern.team_members)
                team_pattern.confidence_score = min(
                    team_pattern.frequency * 0.1 * team_size, 1.0
                )
        
        except Exception as e:
            logger.error(f"Error updating team patterns: {e}")
    
    async def _find_or_create_team_pattern(
        self,
        session_id: str,
        detected_pattern
    ) -> TeamPattern:
        """
        Find existing team pattern or create new one
        """
        pattern_type = getattr(detected_pattern, 'pattern_type', 'unknown')
        
        # Look for existing pattern
        for team_pattern in self.team_patterns[session_id]:
            if team_pattern.pattern_type == pattern_type:
                return team_pattern
        
        # Create new pattern
        new_pattern = TeamPattern(
            pattern_id=f"{session_id}_{pattern_type}_{len(self.team_patterns[session_id])}",
            pattern_type=pattern_type,
            description=getattr(detected_pattern, 'description', f"Team pattern: {pattern_type}"),
            frequency=0,
            team_members=set(),
            code_examples=[],
            confidence_score=0.0,
            last_observed=datetime.now(timezone.utc)
        )
        
        self.team_patterns[session_id].append(new_pattern)
        return new_pattern
    
    async def _get_recent_team_activity(
        self, 
        session_id: str, 
        hours: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Get recent team activity for context
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        recent_activity = []
        if session_id in self.collaborative_history:
            for entry in self.collaborative_history[session_id]:
                entry_time = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00'))
                if entry_time > cutoff_time:
                    recent_activity.append(entry)
        
        return sorted(recent_activity, key=lambda x: x["timestamp"], reverse=True)
    
    async def _find_similar_team_contexts(
        self,
        code_context: CodeContext,
        session_id: str
    ) -> List[Dict[str, Any]]:
        """
        Find similar code contexts used by team
        """
        similar_contexts = []
        
        if session_id not in self.collaborative_history:
            return similar_contexts
        
        current_scope = getattr(code_context, 'current_scope', {})
        current_imports = getattr(code_context, 'imports', [])
        
        for entry in self.collaborative_history[session_id]:
            if "context" in entry:
                entry_context = entry["context"]
                
                # Simple similarity check based on scope and imports
                similarity_score = 0.0
                
                if "scope" in entry_context and current_scope:
                    scope_similarity = len(
                        set(current_scope.keys()) & 
                        set(entry_context["scope"].keys())
                    ) / max(len(current_scope), 1)
                    similarity_score += scope_similarity * 0.5
                
                if "imports" in entry_context and current_imports:
                    import_similarity = len(
                        set(current_imports) & 
                        set(entry_context["imports"])
                    ) / max(len(current_imports), 1)
                    similarity_score += import_similarity * 0.5
                
                if similarity_score > 0.3:
                    similar_contexts.append({
                        "user_id": entry["user_id"],
                        "timestamp": entry["timestamp"],
                        "similarity_score": similarity_score,
                        "context": entry_context
                    })
        
        return sorted(similar_contexts, key=lambda x: x["similarity_score"], reverse=True)[:5]
    
    def _generate_cache_key(self, request: CollaborativePredictionRequest) -> str:
        """Generate cache key for request"""
        return f"{request.session_id}:{request.user_id}:{request.file_path}:{hash(request.code_context)}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        # Simple time-based validation
        # In production, would include content-based invalidation
        return True  # Simplified for now
    
    async def get_collaborative_statistics(
        self, 
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about collaborative predictions and patterns
        """
        stats = {
            "total_team_patterns": 0,
            "active_sessions": len(self.team_patterns),
            "collaborative_history_size": 0,
            "cache_hit_rate": 0.0,
            "top_team_patterns": [],
            "coding_style_consistency": {}
        }
        
        if session_id:
            # Session-specific stats
            if session_id in self.team_patterns:
                stats["total_team_patterns"] = len(self.team_patterns[session_id])
                stats["top_team_patterns"] = [
                    p.to_dict() for p in sorted(
                        self.team_patterns[session_id],
                        key=lambda x: x.frequency,
                        reverse=True
                    )[:5]
                ]
            
            if session_id in self.collaborative_history:
                stats["collaborative_history_size"] = len(self.collaborative_history[session_id])
        else:
            # Global stats
            for session_patterns in self.team_patterns.values():
                stats["total_team_patterns"] += len(session_patterns)
            
            for session_history in self.collaborative_history.values():
                stats["collaborative_history_size"] += len(session_history)
        
        return stats