"""
Adaptive Learning System
Intelligent learning system that adapts and improves from user interactions
Implements continuous learning, feedback processing, and model adaptation
"""

from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import logging
import numpy as np
import json
import pickle
import hashlib
from collections import defaultdict, Counter, deque
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """Learning modes for the adaptive system"""
    PASSIVE = "passive"  # Learn from observations only
    ACTIVE = "active"    # Request feedback and learn actively
    HYBRID = "hybrid"    # Combination of passive and active
    REINFORCEMENT = "reinforcement"  # Learn from rewards/penalties


class FeedbackType(Enum):
    """Types of feedback from users"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    CORRECTIVE = "corrective"
    EXPLICIT_RATING = "explicit_rating"


class LearningObjective(Enum):
    """Learning objectives for the system"""
    IMPROVE_SUGGESTIONS = "improve_suggestions"
    REDUCE_ERRORS = "reduce_errors"
    ADAPT_TO_PREFERENCES = "adapt_to_preferences"
    OPTIMIZE_PERFORMANCE = "optimize_performance"
    LEARN_PATTERNS = "learn_patterns"


@dataclass
class UserInteraction:
    """Represents a user interaction for learning"""
    interaction_id: str
    user_id: str
    timestamp: datetime
    context: Dict[str, Any]
    action_taken: str
    outcome: str
    feedback_type: Optional[FeedbackType] = None
    feedback_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "action_taken": self.action_taken,
            "outcome": self.outcome,
            "feedback_type": self.feedback_type.value if self.feedback_type else None,
            "feedback_score": self.feedback_score,
            "metadata": self.metadata
        }


@dataclass
class LearningPattern:
    """Represents a learned pattern from user interactions"""
    pattern_id: str
    pattern_type: str
    pattern_data: Dict[str, Any]
    confidence_score: float
    usage_count: int
    success_rate: float
    last_updated: datetime
    user_segments: List[str] = field(default_factory=list)
    context_conditions: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "pattern_data": self.pattern_data,
            "confidence_score": self.confidence_score,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "last_updated": self.last_updated.isoformat(),
            "user_segments": self.user_segments,
            "context_conditions": self.context_conditions
        }


@dataclass
class ModelAdaptation:
    """Represents an adaptation made to a model"""
    adaptation_id: str
    model_name: str
    adaptation_type: str  # "weight_update", "parameter_tune", "structure_change"
    changes: Dict[str, Any]
    performance_before: float
    performance_after: float
    applied_timestamp: datetime
    rollback_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "adaptation_id": self.adaptation_id,
            "model_name": self.model_name,
            "adaptation_type": self.adaptation_type,
            "changes": self.changes,
            "performance_before": self.performance_before,
            "performance_after": self.performance_after,
            "applied_timestamp": self.applied_timestamp.isoformat(),
            "rollback_data": self.rollback_data
        }


class UserFeedback(BaseModel):
    """User feedback for learning system"""
    interaction_id: str
    user_id: str
    feedback_type: str
    feedback_score: Optional[float] = None
    feedback_text: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class LearningConfiguration(BaseModel):
    """Configuration for adaptive learning"""
    learning_mode: str = "hybrid"
    learning_rate: float = 0.01
    feedback_weight: float = 1.0
    adaptation_threshold: float = 0.1
    pattern_confidence_threshold: float = 0.7
    max_adaptation_frequency: int = 10  # per hour
    enable_auto_rollback: bool = True


# Abstract base for learnable models
class LearnableModel(ABC):
    """Abstract base class for models that can be adapted"""
    
    @abstractmethod
    async def adapt(self, adaptations: List[ModelAdaptation]) -> bool:
        """Apply adaptations to the model"""
        pass
    
    @abstractmethod
    async def get_current_performance(self) -> float:
        """Get current model performance metric"""
        pass
    
    @abstractmethod
    async def rollback_adaptation(self, adaptation_id: str) -> bool:
        """Rollback a specific adaptation"""
        pass


class AdaptiveLearningSystem:
    """
    Intelligent adaptive learning system that learns from user interactions
    and continuously improves AI model performance and predictions
    """
    
    def __init__(
        self,
        config: Optional[LearningConfiguration] = None,
        storage_path: Optional[str] = None
    ):
        self.config = config or LearningConfiguration()
        self.storage_path = storage_path or "./adaptive_learning_data"
        
        # Core storage
        self.user_interactions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.learned_patterns: Dict[str, LearningPattern] = {}
        self.model_adaptations: List[ModelAdaptation] = []
        self.user_preferences: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Registered models for adaptation
        self.learnable_models: Dict[str, LearnableModel] = {}
        
        # Learning components
        self.pattern_extractor = PatternExtractor()
        self.feedback_processor = FeedbackProcessor()
        self.adaptation_engine = AdaptationEngine()
        self.performance_monitor = PerformanceMonitor()
        
        # Learning state
        self.learning_active = True
        self.last_adaptation_time = {}
        self.adaptation_queue: deque = deque()
        
        # Statistics
        self.learning_stats = {
            "total_interactions": 0,
            "total_patterns": 0,
            "successful_adaptations": 0,
            "failed_adaptations": 0,
            "average_feedback_score": 0.0
        }
        
        logger.info("AdaptiveLearningSystem initialized")
    
    async def register_learnable_model(
        self, 
        model_name: str, 
        model: LearnableModel
    ) -> bool:
        """Register a model that can be adapted through learning"""
        try:
            self.learnable_models[model_name] = model
            self.last_adaptation_time[model_name] = datetime.now(timezone.utc)
            
            logger.info(f"Registered learnable model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering model {model_name}: {e}")
            return False
    
    async def record_interaction(self, interaction: UserInteraction) -> bool:
        """Record a user interaction for learning"""
        try:
            # Store interaction
            self.user_interactions[interaction.user_id].append(interaction)
            self.learning_stats["total_interactions"] += 1
            
            # Extract patterns if enough interactions
            await self._extract_patterns_from_interaction(interaction)
            
            # Process immediate feedback if available
            if interaction.feedback_type is not None:
                await self._process_immediate_feedback(interaction)
            
            # Trigger adaptation if needed
            await self._check_adaptation_trigger(interaction)
            
            logger.debug(f"Recorded interaction: {interaction.interaction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording interaction: {e}")
            return False
    
    async def process_user_feedback(self, feedback: UserFeedback) -> bool:
        """Process explicit user feedback for continuous learning"""
        try:
            # Find related interaction
            interaction = await self._find_interaction(feedback.interaction_id, feedback.user_id)
            
            if not interaction:
                logger.warning(f"No interaction found for feedback: {feedback.interaction_id}")
                return False
            
            # Update interaction with feedback
            interaction.feedback_type = FeedbackType(feedback.feedback_type)
            interaction.feedback_score = feedback.feedback_score
            interaction.metadata.update(feedback.context)
            
            # Process feedback for learning
            learning_signal = await self.feedback_processor.process_feedback(
                feedback, interaction
            )
            
            # Update user preferences
            await self._update_user_preferences(feedback.user_id, learning_signal)
            
            # Update learned patterns
            await self._update_patterns_from_feedback(learning_signal)
            
            # Queue adaptation if significant feedback
            if learning_signal.get("significance", 0) > 0.5:
                await self._queue_adaptation(learning_signal)
            
            logger.info(f"Processed feedback for interaction: {feedback.interaction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return False
    
    async def get_adaptive_suggestions(
        self, 
        user_id: str, 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get suggestions adapted to user's learned preferences"""
        try:
            suggestions = []
            
            # Get user preferences
            user_prefs = self.user_preferences.get(user_id, {})
            
            # Find relevant learned patterns
            relevant_patterns = await self._find_relevant_patterns(user_id, context)
            
            # Generate adapted suggestions
            for pattern in relevant_patterns:
                if pattern.confidence_score >= self.config.pattern_confidence_threshold:
                    suggestion = await self._generate_adaptive_suggestion(
                        pattern, context, user_prefs
                    )
                    if suggestion:
                        suggestions.append(suggestion)
            
            # Sort by relevance and confidence
            suggestions.sort(
                key=lambda x: x.get("confidence", 0) * x.get("relevance", 0),
                reverse=True
            )
            
            return suggestions[:10]  # Top 10 suggestions
            
        except Exception as e:
            logger.error(f"Error generating adaptive suggestions: {e}")
            return []
    
    async def adapt_models(self, force_adaptation: bool = False) -> Dict[str, bool]:
        """Adapt registered models based on learned patterns"""
        adaptation_results = {}
        
        for model_name, model in self.learnable_models.items():
            try:
                # Check adaptation frequency limits
                if not force_adaptation and not await self._should_adapt_model(model_name):
                    continue
                
                # Generate adaptations
                adaptations = await self.adaptation_engine.generate_adaptations(
                    model_name, self.learned_patterns, self.user_preferences
                )
                
                if not adaptations:
                    continue
                
                # Get current performance baseline
                current_performance = await model.get_current_performance()
                
                # Apply adaptations
                success = await model.adapt(adaptations)
                
                if success:
                    # Monitor performance after adaptation
                    new_performance = await model.get_current_performance()
                    
                    # Record adaptation
                    adaptation_record = ModelAdaptation(
                        adaptation_id=f"adapt_{model_name}_{datetime.now().isoformat()}",
                        model_name=model_name,
                        adaptation_type="learned_adaptation",
                        changes={"adaptations": [a.to_dict() if hasattr(a, 'to_dict') else str(a) for a in adaptations]},
                        performance_before=current_performance,
                        performance_after=new_performance,
                        applied_timestamp=datetime.now(timezone.utc)
                    )
                    
                    self.model_adaptations.append(adaptation_record)
                    self.last_adaptation_time[model_name] = datetime.now(timezone.utc)
                    
                    # Check if adaptation was beneficial
                    if new_performance > current_performance:
                        self.learning_stats["successful_adaptations"] += 1
                        logger.info(f"Successful adaptation for {model_name}: {current_performance:.3f} → {new_performance:.3f}")
                    else:
                        self.learning_stats["failed_adaptations"] += 1
                        
                        # Auto-rollback if enabled and performance degraded significantly
                        if (self.config.enable_auto_rollback and 
                            current_performance - new_performance > 0.1):
                            await model.rollback_adaptation(adaptation_record.adaptation_id)
                            logger.warning(f"Auto-rolled back adaptation for {model_name} due to performance degradation")
                    
                    adaptation_results[model_name] = True
                else:
                    adaptation_results[model_name] = False
                    logger.error(f"Failed to apply adaptations to {model_name}")
                
            except Exception as e:
                logger.error(f"Error adapting model {model_name}: {e}")
                adaptation_results[model_name] = False
        
        return adaptation_results
    
    async def _extract_patterns_from_interaction(self, interaction: UserInteraction) -> None:
        """Extract learning patterns from user interaction"""
        try:
            patterns = await self.pattern_extractor.extract_patterns(
                interaction, self.user_interactions[interaction.user_id]
            )
            
            for pattern_data in patterns:
                pattern_id = pattern_data.get("pattern_id")
                
                if pattern_id in self.learned_patterns:
                    # Update existing pattern
                    pattern = self.learned_patterns[pattern_id]
                    pattern.usage_count += 1
                    pattern.last_updated = datetime.now(timezone.utc)
                    
                    # Update success rate
                    if interaction.outcome == "success":
                        pattern.success_rate = (pattern.success_rate * (pattern.usage_count - 1) + 1.0) / pattern.usage_count
                    else:
                        pattern.success_rate = (pattern.success_rate * (pattern.usage_count - 1) + 0.0) / pattern.usage_count
                    
                    # Update confidence based on usage and success
                    pattern.confidence_score = min(
                        pattern.success_rate * math.log(pattern.usage_count + 1) / 3,
                        1.0
                    )
                else:
                    # Create new pattern
                    pattern = LearningPattern(
                        pattern_id=pattern_id,
                        pattern_type=pattern_data.get("type", "unknown"),
                        pattern_data=pattern_data.get("data", {}),
                        confidence_score=0.5,
                        usage_count=1,
                        success_rate=1.0 if interaction.outcome == "success" else 0.0,
                        last_updated=datetime.now(timezone.utc),
                        user_segments=[interaction.user_id],
                        context_conditions=pattern_data.get("context", {})
                    )
                    
                    self.learned_patterns[pattern_id] = pattern
                    self.learning_stats["total_patterns"] += 1
            
        except Exception as e:
            logger.error(f"Error extracting patterns from interaction: {e}")
    
    async def _process_immediate_feedback(self, interaction: UserInteraction) -> None:
        """Process immediate feedback from interaction"""
        if interaction.feedback_score is not None:
            # Update average feedback score
            current_avg = self.learning_stats["average_feedback_score"]
            total_interactions = self.learning_stats["total_interactions"]
            
            new_avg = ((current_avg * (total_interactions - 1)) + interaction.feedback_score) / total_interactions
            self.learning_stats["average_feedback_score"] = new_avg
            
            # If feedback is significantly negative, flag for immediate attention
            if interaction.feedback_score < 0.3:
                await self._handle_negative_feedback(interaction)
    
    async def _handle_negative_feedback(self, interaction: UserInteraction) -> None:
        """Handle significantly negative feedback"""
        try:
            # Analyze what went wrong
            analysis = {
                "interaction_id": interaction.interaction_id,
                "user_id": interaction.user_id,
                "context": interaction.context,
                "action": interaction.action_taken,
                "feedback_score": interaction.feedback_score,
                "timestamp": interaction.timestamp.isoformat()
            }
            
            # Queue for immediate adaptation consideration
            self.adaptation_queue.append({
                "type": "negative_feedback_response",
                "priority": "high",
                "data": analysis,
                "timestamp": datetime.now(timezone.utc)
            })
            
            logger.warning(f"Negative feedback received for interaction {interaction.interaction_id}, queued for adaptation")
            
        except Exception as e:
            logger.error(f"Error handling negative feedback: {e}")
    
    async def _find_interaction(
        self, 
        interaction_id: str, 
        user_id: str
    ) -> Optional[UserInteraction]:
        """Find interaction by ID and user ID"""
        user_interactions = self.user_interactions.get(user_id, deque())
        
        for interaction in user_interactions:
            if interaction.interaction_id == interaction_id:
                return interaction
        
        return None
    
    async def _update_user_preferences(
        self, 
        user_id: str, 
        learning_signal: Dict[str, Any]
    ) -> None:
        """Update user preferences based on learning signal"""
        preferences = self.user_preferences[user_id]
        
        # Update preference weights
        for preference_key, signal_value in learning_signal.get("preferences", {}).items():
            current_value = preferences.get(preference_key, 0.5)
            
            # Exponential moving average update
            alpha = self.config.learning_rate
            updated_value = alpha * signal_value + (1 - alpha) * current_value
            preferences[preference_key] = updated_value
        
        # Update preference categories
        for category, category_data in learning_signal.get("categories", {}).items():
            if category not in preferences:
                preferences[category] = {}
            
            for key, value in category_data.items():
                current = preferences[category].get(key, 0.5)
                alpha = self.config.learning_rate
                preferences[category][key] = alpha * value + (1 - alpha) * current
    
    async def _update_patterns_from_feedback(self, learning_signal: Dict[str, Any]) -> None:
        """Update learned patterns based on feedback signal"""
        pattern_updates = learning_signal.get("pattern_updates", {})
        
        for pattern_id, update_data in pattern_updates.items():
            if pattern_id in self.learned_patterns:
                pattern = self.learned_patterns[pattern_id]
                
                # Update confidence based on feedback
                feedback_impact = update_data.get("confidence_delta", 0)
                pattern.confidence_score = max(0.0, min(1.0, pattern.confidence_score + feedback_impact))
                
                # Update success rate
                if "success_rate_delta" in update_data:
                    delta = update_data["success_rate_delta"]
                    pattern.success_rate = max(0.0, min(1.0, pattern.success_rate + delta))
                
                pattern.last_updated = datetime.now(timezone.utc)
    
    async def _queue_adaptation(self, learning_signal: Dict[str, Any]) -> None:
        """Queue an adaptation based on learning signal"""
        adaptation_request = {
            "type": "feedback_adaptation",
            "priority": "normal",
            "data": learning_signal,
            "timestamp": datetime.now(timezone.utc)
        }
        
        self.adaptation_queue.append(adaptation_request)
        logger.debug("Queued adaptation request based on learning signal")
    
    async def _find_relevant_patterns(
        self, 
        user_id: str, 
        context: Dict[str, Any]
    ) -> List[LearningPattern]:
        """Find learned patterns relevant to current context"""
        relevant_patterns = []
        
        for pattern in self.learned_patterns.values():
            # Check if pattern is relevant to user
            if user_id in pattern.user_segments or not pattern.user_segments:
                
                # Check context relevance
                relevance_score = await self._calculate_context_relevance(
                    pattern.context_conditions, context
                )
                
                if relevance_score > 0.5:
                    # Add relevance score to pattern for sorting
                    pattern.metadata = pattern.metadata or {}
                    pattern.metadata["relevance_score"] = relevance_score
                    relevant_patterns.append(pattern)
        
        return sorted(relevant_patterns, 
                     key=lambda p: p.metadata.get("relevance_score", 0) * p.confidence_score, 
                     reverse=True)
    
    async def _calculate_context_relevance(
        self, 
        pattern_context: Dict[str, Any], 
        current_context: Dict[str, Any]
    ) -> float:
        """Calculate relevance of pattern context to current context"""
        if not pattern_context or not current_context:
            return 0.5
        
        # Simple relevance calculation based on key overlap
        pattern_keys = set(pattern_context.keys())
        current_keys = set(current_context.keys())
        
        if not pattern_keys or not current_keys:
            return 0.5
        
        # Key overlap
        key_overlap = len(pattern_keys & current_keys) / len(pattern_keys | current_keys)
        
        # Value similarity for overlapping keys
        value_similarity = 0.0
        overlapping_keys = pattern_keys & current_keys
        
        if overlapping_keys:
            similarities = []
            for key in overlapping_keys:
                pattern_val = pattern_context[key]
                current_val = current_context[key]
                
                if pattern_val == current_val:
                    similarities.append(1.0)
                elif isinstance(pattern_val, str) and isinstance(current_val, str):
                    # String similarity
                    similarities.append(self._string_similarity(pattern_val, current_val))
                elif isinstance(pattern_val, (int, float)) and isinstance(current_val, (int, float)):
                    # Numeric similarity
                    max_val = max(abs(pattern_val), abs(current_val), 1)
                    similarities.append(1.0 - abs(pattern_val - current_val) / max_val)
                else:
                    similarities.append(0.5)  # Different types
            
            value_similarity = sum(similarities) / len(similarities)
        
        # Combined relevance score
        relevance = (key_overlap + value_similarity) / 2
        return relevance
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity"""
        if str1 == str2:
            return 1.0
        
        # Simple Jaccard similarity of words
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union
    
    async def _generate_adaptive_suggestion(
        self, 
        pattern: LearningPattern, 
        context: Dict[str, Any],
        user_prefs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate suggestion adapted to learned pattern and user preferences"""
        try:
            suggestion = {
                "suggestion_id": f"adaptive_{pattern.pattern_id}_{hash(str(context)) % 10000}",
                "type": "adaptive_suggestion",
                "content": pattern.pattern_data.get("content", ""),
                "confidence": pattern.confidence_score,
                "relevance": pattern.metadata.get("relevance_score", 0.5),
                "pattern_based": True,
                "pattern_id": pattern.pattern_id,
                "success_rate": pattern.success_rate,
                "usage_count": pattern.usage_count,
                "explanation": f"Based on learned pattern with {pattern.usage_count} uses and {pattern.success_rate:.1%} success rate"
            }
            
            # Adapt suggestion based on user preferences
            if user_prefs:
                # Adjust confidence based on user preference alignment
                preference_alignment = await self._calculate_preference_alignment(
                    suggestion, user_prefs
                )
                suggestion["confidence"] *= (1.0 + preference_alignment * 0.2)
                
                # Add user-specific adaptations
                suggestion["user_adapted"] = True
                suggestion["preference_alignment"] = preference_alignment
            
            return suggestion
            
        except Exception as e:
            logger.error(f"Error generating adaptive suggestion: {e}")
            return None
    
    async def _calculate_preference_alignment(
        self, 
        suggestion: Dict[str, Any], 
        user_prefs: Dict[str, Any]
    ) -> float:
        """Calculate how well suggestion aligns with user preferences"""
        alignment_score = 0.5  # Neutral baseline
        
        # Check preference categories
        for pref_category, prefs in user_prefs.items():
            if pref_category in suggestion:
                suggestion_value = suggestion[pref_category]
                
                if isinstance(prefs, dict):
                    for pref_key, pref_value in prefs.items():
                        if pref_key in str(suggestion_value).lower():
                            alignment_score += (pref_value - 0.5) * 0.1  # Max impact ±0.1 per preference
                elif isinstance(suggestion_value, str) and pref_category in suggestion_value.lower():
                    alignment_score += (prefs - 0.5) * 0.2
        
        return max(0.0, min(1.0, alignment_score))
    
    async def _should_adapt_model(self, model_name: str) -> bool:
        """Check if model should be adapted based on frequency limits"""
        last_adaptation = self.last_adaptation_time.get(model_name)
        if not last_adaptation:
            return True
        
        time_since_last = datetime.now(timezone.utc) - last_adaptation
        hours_since_last = time_since_last.total_seconds() / 3600
        
        # Check frequency limit
        if hours_since_last < (1.0 / self.config.max_adaptation_frequency):
            return False
        
        return True
    
    async def _check_adaptation_trigger(self, interaction: UserInteraction) -> None:
        """Check if interaction should trigger model adaptation"""
        # Trigger adaptation on significant negative feedback
        if (interaction.feedback_score is not None and 
            interaction.feedback_score < 0.3):
            await self._queue_adaptation({
                "trigger": "negative_feedback",
                "interaction": interaction.to_dict(),
                "significance": 0.8
            })
        
        # Trigger adaptation on pattern threshold reached
        user_interaction_count = len(self.user_interactions[interaction.user_id])
        if user_interaction_count > 0 and user_interaction_count % 50 == 0:  # Every 50 interactions
            await self._queue_adaptation({
                "trigger": "interaction_milestone",
                "user_id": interaction.user_id,
                "count": user_interaction_count,
                "significance": 0.6
            })
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about the learning process"""
        insights = {
            "learning_stats": self.learning_stats.copy(),
            "model_performance": {},
            "user_segments": {},
            "top_patterns": [],
            "adaptation_history": []
        }
        
        # Model performance
        for model_name in self.learnable_models.keys():
            try:
                performance = await self.learnable_models[model_name].get_current_performance()
                insights["model_performance"][model_name] = performance
            except Exception as e:
                insights["model_performance"][model_name] = f"Error: {e}"
        
        # User segments analysis
        for user_id, interactions in self.user_interactions.items():
            if interactions:
                avg_feedback = sum(
                    i.feedback_score for i in interactions 
                    if i.feedback_score is not None
                ) / max(sum(1 for i in interactions if i.feedback_score is not None), 1)
                
                insights["user_segments"][user_id] = {
                    "interaction_count": len(interactions),
                    "average_feedback": avg_feedback,
                    "preferences": self.user_preferences.get(user_id, {})
                }
        
        # Top patterns by confidence and usage
        top_patterns = sorted(
            self.learned_patterns.values(),
            key=lambda p: p.confidence_score * p.usage_count,
            reverse=True
        )[:10]
        
        insights["top_patterns"] = [p.to_dict() for p in top_patterns]
        
        # Recent adaptations
        insights["adaptation_history"] = [
            a.to_dict() for a in self.model_adaptations[-20:]  # Last 20 adaptations
        ]
        
        return insights
    
    async def save_learning_state(self, filepath: Optional[str] = None) -> bool:
        """Save learning state to disk"""
        try:
            save_path = filepath or f"{self.storage_path}/learning_state.json"
            
            state = {
                "learned_patterns": {k: v.to_dict() for k, v in self.learned_patterns.items()},
                "user_preferences": dict(self.user_preferences),
                "learning_stats": self.learning_stats,
                "model_adaptations": [a.to_dict() for a in self.model_adaptations],
                "config": {
                    "learning_mode": self.config.learning_mode,
                    "learning_rate": self.config.learning_rate,
                    "feedback_weight": self.config.feedback_weight,
                    "adaptation_threshold": self.config.adaptation_threshold,
                    "pattern_confidence_threshold": self.config.pattern_confidence_threshold
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Learning state saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving learning state: {e}")
            return False
    
    async def load_learning_state(self, filepath: Optional[str] = None) -> bool:
        """Load learning state from disk"""
        try:
            load_path = filepath or f"{self.storage_path}/learning_state.json"
            
            with open(load_path, 'r') as f:
                state = json.load(f)
            
            # Restore learned patterns
            for pattern_id, pattern_data in state.get("learned_patterns", {}).items():
                pattern = LearningPattern(
                    pattern_id=pattern_data["pattern_id"],
                    pattern_type=pattern_data["pattern_type"],
                    pattern_data=pattern_data["pattern_data"],
                    confidence_score=pattern_data["confidence_score"],
                    usage_count=pattern_data["usage_count"],
                    success_rate=pattern_data["success_rate"],
                    last_updated=datetime.fromisoformat(pattern_data["last_updated"]),
                    user_segments=pattern_data.get("user_segments", []),
                    context_conditions=pattern_data.get("context_conditions", {})
                )
                self.learned_patterns[pattern_id] = pattern
            
            # Restore user preferences
            self.user_preferences = defaultdict(dict, state.get("user_preferences", {}))
            
            # Restore stats
            self.learning_stats.update(state.get("learning_stats", {}))
            
            logger.info(f"Learning state loaded from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading learning state: {e}")
            return False


# Helper classes for learning components

class PatternExtractor:
    """Extracts patterns from user interactions"""
    
    async def extract_patterns(
        self, 
        interaction: UserInteraction, 
        interaction_history: deque
    ) -> List[Dict[str, Any]]:
        """Extract patterns from interaction and history"""
        patterns = []
        
        # Sequential pattern detection
        if len(interaction_history) >= 3:
            sequence_pattern = self._detect_sequence_pattern(interaction_history)
            if sequence_pattern:
                patterns.append(sequence_pattern)
        
        # Context pattern detection
        context_pattern = self._detect_context_pattern(interaction)
        if context_pattern:
            patterns.append(context_pattern)
        
        # Preference pattern detection
        preference_pattern = self._detect_preference_pattern(interaction, interaction_history)
        if preference_pattern:
            patterns.append(preference_pattern)
        
        return patterns
    
    def _detect_sequence_pattern(self, history: deque) -> Optional[Dict[str, Any]]:
        """Detect sequential patterns in user behavior"""
        if len(history) < 3:
            return None
        
        # Look for repeated action sequences
        recent_actions = [i.action_taken for i in list(history)[-5:]]
        action_sequence = " -> ".join(recent_actions)
        
        return {
            "pattern_id": f"sequence_{hashlib.md5(action_sequence.encode()).hexdigest()[:8]}",
            "type": "action_sequence",
            "data": {
                "sequence": recent_actions,
                "sequence_string": action_sequence
            },
            "context": {"sequence_length": len(recent_actions)}
        }
    
    def _detect_context_pattern(self, interaction: UserInteraction) -> Optional[Dict[str, Any]]:
        """Detect context-based patterns"""
        context_keys = list(interaction.context.keys())
        
        if len(context_keys) < 2:
            return None
        
        context_signature = "|".join(sorted(context_keys))
        
        return {
            "pattern_id": f"context_{hashlib.md5(context_signature.encode()).hexdigest()[:8]}",
            "type": "context_pattern",
            "data": {
                "context_keys": context_keys,
                "action": interaction.action_taken,
                "outcome": interaction.outcome
            },
            "context": interaction.context
        }
    
    def _detect_preference_pattern(
        self, 
        interaction: UserInteraction, 
        history: deque
    ) -> Optional[Dict[str, Any]]:
        """Detect user preference patterns"""
        if interaction.feedback_score is None:
            return None
        
        # Analyze feedback patterns
        recent_feedback = [
            i.feedback_score for i in history 
            if i.feedback_score is not None
        ][-10:]  # Last 10 feedback scores
        
        if len(recent_feedback) < 3:
            return None
        
        avg_feedback = sum(recent_feedback) / len(recent_feedback)
        
        return {
            "pattern_id": f"preference_{interaction.user_id}_{hashlib.md5(interaction.action_taken.encode()).hexdigest()[:8]}",
            "type": "preference_pattern",
            "data": {
                "action": interaction.action_taken,
                "average_feedback": avg_feedback,
                "feedback_count": len(recent_feedback)
            },
            "context": {"user_id": interaction.user_id}
        }


class FeedbackProcessor:
    """Processes user feedback for learning signals"""
    
    async def process_feedback(
        self, 
        feedback: UserFeedback, 
        interaction: UserInteraction
    ) -> Dict[str, Any]:
        """Process feedback and generate learning signals"""
        learning_signal = {
            "feedback_id": feedback.interaction_id,
            "significance": 0.5,
            "preferences": {},
            "categories": {},
            "pattern_updates": {}
        }
        
        # Calculate significance
        if feedback.feedback_score is not None:
            # Higher significance for extreme feedback
            significance = abs(feedback.feedback_score - 0.5) * 2
            learning_signal["significance"] = min(significance, 1.0)
        
        # Extract preference signals
        if feedback.feedback_type == "positive":
            learning_signal["preferences"] = self._extract_positive_preferences(
                interaction, feedback
            )
        elif feedback.feedback_type == "negative":
            learning_signal["preferences"] = self._extract_negative_preferences(
                interaction, feedback
            )
        
        # Generate pattern updates
        learning_signal["pattern_updates"] = self._generate_pattern_updates(
            interaction, feedback
        )
        
        return learning_signal
    
    def _extract_positive_preferences(
        self, 
        interaction: UserInteraction, 
        feedback: UserFeedback
    ) -> Dict[str, float]:
        """Extract positive preferences from feedback"""
        preferences = {}
        
        # Boost preference for action taken
        action_category = self._categorize_action(interaction.action_taken)
        preferences[action_category] = 0.7  # Boost positive actions
        
        # Extract contextual preferences
        for key, value in interaction.context.items():
            pref_key = f"context_{key}"
            preferences[pref_key] = 0.6
        
        return preferences
    
    def _extract_negative_preferences(
        self, 
        interaction: UserInteraction, 
        feedback: UserFeedback
    ) -> Dict[str, float]:
        """Extract negative preferences from feedback"""
        preferences = {}
        
        # Reduce preference for action taken
        action_category = self._categorize_action(interaction.action_taken)
        preferences[action_category] = 0.3  # Reduce negative actions
        
        # Reduce contextual preferences
        for key, value in interaction.context.items():
            pref_key = f"context_{key}"
            preferences[pref_key] = 0.4
        
        return preferences
    
    def _categorize_action(self, action: str) -> str:
        """Categorize action for preference learning"""
        action_lower = action.lower()
        
        if "suggest" in action_lower:
            return "suggestions"
        elif "predict" in action_lower:
            return "predictions"
        elif "refactor" in action_lower:
            return "refactoring"
        elif "analyze" in action_lower:
            return "analysis"
        else:
            return "general"
    
    def _generate_pattern_updates(
        self, 
        interaction: UserInteraction, 
        feedback: UserFeedback
    ) -> Dict[str, Dict[str, float]]:
        """Generate updates for existing patterns based on feedback"""
        updates = {}
        
        if feedback.feedback_score is not None:
            # Generate confidence delta based on feedback
            confidence_delta = (feedback.feedback_score - 0.5) * 0.1  # Max ±0.1 change
            
            # Apply to context-based patterns
            context_pattern_id = f"context_{hashlib.md5(str(interaction.context).encode()).hexdigest()[:8]}"
            updates[context_pattern_id] = {
                "confidence_delta": confidence_delta,
                "success_rate_delta": confidence_delta * 0.5
            }
        
        return updates


class AdaptationEngine:
    """Generates model adaptations based on learned patterns"""
    
    async def generate_adaptations(
        self, 
        model_name: str,
        learned_patterns: Dict[str, LearningPattern],
        user_preferences: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate adaptations for a model"""
        adaptations = []
        
        # Generate preference-based adaptations
        pref_adaptations = await self._generate_preference_adaptations(
            model_name, user_preferences
        )
        adaptations.extend(pref_adaptations)
        
        # Generate pattern-based adaptations
        pattern_adaptations = await self._generate_pattern_adaptations(
            model_name, learned_patterns
        )
        adaptations.extend(pattern_adaptations)
        
        return adaptations
    
    async def _generate_preference_adaptations(
        self, 
        model_name: str,
        user_preferences: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate adaptations based on user preferences"""
        adaptations = []
        
        # Aggregate preferences across users
        aggregated_prefs = self._aggregate_preferences(user_preferences)
        
        for pref_category, pref_value in aggregated_prefs.items():
            if abs(pref_value - 0.5) > 0.2:  # Significant preference
                adaptation = {
                    "type": "preference_weight_adjustment",
                    "category": pref_category,
                    "weight_adjustment": (pref_value - 0.5) * 0.4,  # Max ±0.2 adjustment
                    "confidence": abs(pref_value - 0.5)
                }
                adaptations.append(adaptation)
        
        return adaptations
    
    def _aggregate_preferences(
        self, 
        user_preferences: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Aggregate preferences across users"""
        aggregated = defaultdict(list)
        
        for user_id, prefs in user_preferences.items():
            for pref_key, pref_value in prefs.items():
                if isinstance(pref_value, (int, float)):
                    aggregated[pref_key].append(pref_value)
                elif isinstance(pref_value, dict):
                    # Handle nested preferences
                    for nested_key, nested_value in pref_value.items():
                        if isinstance(nested_value, (int, float)):
                            full_key = f"{pref_key}_{nested_key}"
                            aggregated[full_key].append(nested_value)
        
        # Calculate averages
        result = {}
        for key, values in aggregated.items():
            if values:
                result[key] = sum(values) / len(values)
        
        return result
    
    async def _generate_pattern_adaptations(
        self, 
        model_name: str,
        learned_patterns: Dict[str, LearningPattern]
    ) -> List[Dict[str, Any]]:
        """Generate adaptations based on learned patterns"""
        adaptations = []
        
        # Find high-confidence patterns
        high_conf_patterns = [
            p for p in learned_patterns.values()
            if p.confidence_score > 0.8 and p.usage_count > 10
        ]
        
        for pattern in high_conf_patterns:
            if pattern.pattern_type == "preference_pattern":
                adaptation = {
                    "type": "pattern_integration",
                    "pattern_id": pattern.pattern_id,
                    "pattern_data": pattern.pattern_data,
                    "confidence": pattern.confidence_score,
                    "usage_weight": min(pattern.usage_count / 100, 1.0)
                }
                adaptations.append(adaptation)
        
        return adaptations


class PerformanceMonitor:
    """Monitors model performance after adaptations"""
    
    def __init__(self):
        self.performance_history = defaultdict(list)
        self.baseline_performance = {}
    
    async def record_performance(
        self, 
        model_name: str, 
        performance: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record model performance"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        self.performance_history[model_name].append({
            "performance": performance,
            "timestamp": timestamp
        })
        
        # Maintain history limit
        if len(self.performance_history[model_name]) > 1000:
            self.performance_history[model_name] = self.performance_history[model_name][-500:]
    
    async def get_performance_trend(self, model_name: str, days: int = 7) -> Dict[str, Any]:
        """Get performance trend for a model"""
        if model_name not in self.performance_history:
            return {"trend": "no_data", "data": []}
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
        recent_data = [
            entry for entry in self.performance_history[model_name]
            if entry["timestamp"] > cutoff_time
        ]
        
        if len(recent_data) < 2:
            return {"trend": "insufficient_data", "data": recent_data}
        
        # Calculate trend
        performances = [entry["performance"] for entry in recent_data]
        first_half = performances[:len(performances)//2]
        second_half = performances[len(performances)//2:]
        
        if not first_half or not second_half:
            return {"trend": "insufficient_data", "data": recent_data}
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg > first_avg + 0.05:
            trend = "improving"
        elif second_avg < first_avg - 0.05:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "first_half_avg": first_avg,
            "second_half_avg": second_avg,
            "change": second_avg - first_avg,
            "data_points": len(recent_data)
        }


import math