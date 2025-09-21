#!/usr/bin/env python3
"""
Agent Learning & Adaptation System Module

This module provides comprehensive learning and adaptation capabilities for autonomous AI agents.
It implements reinforcement learning, meta-learning, adaptive behaviors, knowledge transfer,
and continuous improvement mechanisms to enable intelligent agent evolution.

Created: 2025-01-09
Author: Archon Enhancement System
Version: 7.0.0
"""

import asyncio
import json
import uuid
import math
import random
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import pickle
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LearningAlgorithm(Enum):
    """Types of learning algorithms"""
    Q_LEARNING = auto()
    DEEP_Q_NETWORK = auto()
    POLICY_GRADIENT = auto()
    ACTOR_CRITIC = auto()
    TEMPORAL_DIFFERENCE = auto()
    MONTE_CARLO = auto()
    MULTI_ARMED_BANDIT = auto()
    GENETIC_ALGORITHM = auto()
    NEURAL_EVOLUTION = auto()
    IMITATION_LEARNING = auto()
    TRANSFER_LEARNING = auto()
    META_LEARNING = auto()
    CONTINUAL_LEARNING = auto()
    ONLINE_LEARNING = auto()
    BATCH_LEARNING = auto()


class AdaptationType(Enum):
    """Types of agent adaptation"""
    BEHAVIORAL = auto()           # Behavior modification
    STRUCTURAL = auto()          # Architecture changes  
    PARAMETRIC = auto()          # Parameter tuning
    STRATEGIC = auto()           # Strategy adaptation
    REACTIVE = auto()            # Reactive adaptation
    PROACTIVE = auto()           # Proactive adaptation
    EVOLUTIONARY = auto()        # Evolutionary adaptation
    SOCIAL = auto()              # Social adaptation
    ENVIRONMENTAL = auto()       # Environmental adaptation
    COGNITIVE = auto()           # Cognitive adaptation


class LearningMode(Enum):
    """Learning operation modes"""
    EXPLORATION = auto()         # Exploration phase
    EXPLOITATION = auto()        # Exploitation phase
    EXPLORATION_EXPLOITATION = auto()  # Mixed mode
    TRANSFER = auto()            # Knowledge transfer
    CONSOLIDATION = auto()       # Memory consolidation
    ADAPTATION = auto()          # Active adaptation
    EVALUATION = auto()          # Performance evaluation


class KnowledgeType(Enum):
    """Types of knowledge"""
    PROCEDURAL = auto()          # How-to knowledge
    DECLARATIVE = auto()         # Factual knowledge
    EPISODIC = auto()           # Experience-based knowledge
    SEMANTIC = auto()           # Conceptual knowledge
    STRATEGIC = auto()          # Strategic knowledge
    CONTEXTUAL = auto()         # Context-specific knowledge
    CAUSAL = auto()             # Cause-effect knowledge
    PREDICTIVE = auto()         # Prediction knowledge


@dataclass
class LearningState:
    """Represents the state in learning process"""
    state_id: str
    features: Dict[str, float]
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_vector(self, feature_names: List[str]) -> List[float]:
        """Convert state to feature vector"""
        return [self.features.get(name, 0.0) for name in feature_names]
    
    def distance_to(self, other: 'LearningState', feature_names: List[str]) -> float:
        """Calculate distance to another state"""
        self_vector = self.to_vector(feature_names)
        other_vector = other.to_vector(feature_names)
        
        return math.sqrt(sum(
            (a - b) ** 2 for a, b in zip(self_vector, other_vector)
        ))


@dataclass
class LearningAction:
    """Represents an action in learning process"""
    action_id: str
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    expected_outcome: Optional[Dict[str, Any]] = None
    cost: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningExperience:
    """Represents a learning experience"""
    experience_id: str
    agent_id: str
    state: LearningState
    action: LearningAction
    reward: float
    next_state: Optional[LearningState]
    done: bool
    timestamp: datetime = field(default_factory=datetime.now)
    learning_value: float = 0.0  # Q-value, policy value, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeItem:
    """Represents a piece of knowledge"""
    knowledge_id: str
    knowledge_type: KnowledgeType
    content: Dict[str, Any]
    confidence: float = 0.8
    source: str = "learned"
    context: Dict[str, Any] = field(default_factory=dict)
    usage_count: int = 0
    success_rate: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    expiry: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if knowledge item is expired"""
        if not self.expiry:
            return False
        return datetime.now() > self.expiry
    
    def update_usage(self, success: bool) -> None:
        """Update usage statistics"""
        self.usage_count += 1
        self.last_used = datetime.now()
        
        # Update success rate using exponential moving average
        alpha = 0.1
        if self.usage_count == 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            self.success_rate = (
                alpha * (1.0 if success else 0.0) + 
                (1 - alpha) * self.success_rate
            )


@dataclass
class AdaptationEvent:
    """Represents an adaptation event"""
    event_id: str
    agent_id: str
    adaptation_type: AdaptationType
    trigger: str
    changes: Dict[str, Any]
    performance_before: float
    performance_after: Optional[float] = None
    success: Optional[bool] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningMetrics:
    """Metrics for learning and adaptation"""
    total_experiences: int = 0
    successful_experiences: int = 0
    average_reward: float = 0.0
    cumulative_reward: float = 0.0
    learning_rate_effective: float = 0.0
    exploration_ratio: float = 0.5
    adaptation_events: int = 0
    successful_adaptations: int = 0
    knowledge_items: int = 0
    knowledge_utilization: float = 0.0
    convergence_rate: float = 0.0
    transfer_success_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class BaseLearningAlgorithm(ABC):
    """Abstract base class for learning algorithms"""
    
    @abstractmethod
    async def learn(self, experience: LearningExperience) -> Dict[str, Any]:
        """Learn from experience"""
        pass
    
    @abstractmethod
    async def predict(self, state: LearningState) -> Tuple[LearningAction, float]:
        """Predict best action for state"""
        pass
    
    @abstractmethod
    async def update_policy(self, experiences: List[LearningExperience]) -> None:
        """Update learning policy"""
        pass
    
    @abstractmethod
    def get_learning_progress(self) -> Dict[str, float]:
        """Get learning progress metrics"""
        pass


class QLearningAlgorithm(BaseLearningAlgorithm):
    """Q-Learning algorithm implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.learning_rate = config.get('learning_rate', 0.1)
        self.discount_factor = config.get('discount_factor', 0.99)
        self.epsilon = config.get('epsilon', 0.1)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        
        # Q-table: (state_hash, action_id) -> Q-value
        self.q_table: Dict[Tuple[str, str], float] = defaultdict(float)
        self.state_action_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.episode_rewards: List[float] = []
        
    async def learn(self, experience: LearningExperience) -> Dict[str, Any]:
        """Learn from Q-learning experience"""
        try:
            state_hash = self._state_to_hash(experience.state)
            action_id = experience.action.action_id
            reward = experience.reward
            
            current_q = self.q_table[(state_hash, action_id)]
            
            if experience.next_state and not experience.done:
                next_state_hash = self._state_to_hash(experience.next_state)
                # Find max Q-value for next state
                next_q_values = [
                    self.q_table[(next_state_hash, aid)] 
                    for aid in self._get_available_actions(experience.next_state)
                ]
                max_next_q = max(next_q_values) if next_q_values else 0.0
            else:
                max_next_q = 0.0
            
            # Q-learning update rule
            target_q = reward + self.discount_factor * max_next_q
            updated_q = current_q + self.learning_rate * (target_q - current_q)
            
            self.q_table[(state_hash, action_id)] = updated_q
            self.state_action_counts[(state_hash, action_id)] += 1
            
            # Update experience with learned value
            experience.learning_value = updated_q
            
            return {
                'algorithm': 'Q-Learning',
                'q_value': updated_q,
                'temporal_difference': target_q - current_q,
                'exploration_rate': self.epsilon
            }
            
        except Exception as e:
            logger.error(f"Q-learning failed: {e}")
            return {'error': str(e)}
    
    async def predict(self, state: LearningState) -> Tuple[LearningAction, float]:
        """Predict best action using epsilon-greedy policy"""
        try:
            state_hash = self._state_to_hash(state)
            available_actions = self._get_available_actions(state)
            
            if not available_actions:
                # Create default action
                action = LearningAction(
                    action_id="default",
                    action_type="wait",
                    parameters={}
                )
                return action, 0.0
            
            # Epsilon-greedy action selection
            if random.random() < self.epsilon:
                # Exploration: random action
                action_id = random.choice(available_actions)
                q_value = self.q_table[(state_hash, action_id)]
            else:
                # Exploitation: best known action
                q_values = [(aid, self.q_table[(state_hash, aid)]) for aid in available_actions]
                action_id, q_value = max(q_values, key=lambda x: x[1])
            
            # Create action object
            action = LearningAction(
                action_id=action_id,
                action_type=action_id.split('_')[0] if '_' in action_id else action_id,
                parameters=state.context.get('action_params', {})
            )
            
            return action, q_value
            
        except Exception as e:
            logger.error(f"Q-learning prediction failed: {e}")
            return LearningAction("error", "error"), 0.0
    
    async def update_policy(self, experiences: List[LearningExperience]) -> None:
        """Update Q-learning policy"""
        try:
            # Batch update
            for experience in experiences:
                await self.learn(experience)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Calculate episode reward if experiences are from complete episode
            if experiences and experiences[-1].done:
                episode_reward = sum(exp.reward for exp in experiences)
                self.episode_rewards.append(episode_reward)
                
                # Keep only recent episodes
                if len(self.episode_rewards) > 1000:
                    self.episode_rewards = self.episode_rewards[-1000:]
            
        except Exception as e:
            logger.error(f"Q-learning policy update failed: {e}")
    
    def get_learning_progress(self) -> Dict[str, float]:
        """Get Q-learning progress metrics"""
        return {
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'average_episode_reward': sum(self.episode_rewards[-100:]) / len(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            'total_episodes': len(self.episode_rewards),
            'max_q_value': max(self.q_table.values()) if self.q_table else 0.0,
            'min_q_value': min(self.q_table.values()) if self.q_table else 0.0
        }
    
    def _state_to_hash(self, state: LearningState) -> str:
        """Convert state to hash for Q-table indexing"""
        # Simple feature-based hashing
        feature_str = "_".join(f"{k}_{v:.2f}" for k, v in sorted(state.features.items()))
        return hashlib.md5(feature_str.encode()).hexdigest()[:16] if 'hashlib' in globals() else feature_str[:16]
    
    def _get_available_actions(self, state: LearningState) -> List[str]:
        """Get available actions for state"""
        # Default action set - could be customized based on state
        return state.context.get('available_actions', [
            'move_forward', 'move_backward', 'turn_left', 'turn_right',
            'interact', 'wait', 'explore', 'exploit'
        ])


class PolicyGradientAlgorithm(BaseLearningAlgorithm):
    """Policy Gradient algorithm implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.learning_rate = config.get('learning_rate', 0.01)
        self.discount_factor = config.get('discount_factor', 0.99)
        
        # Policy parameters (simple linear policy)
        self.policy_params: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.baseline = 0.0  # For variance reduction
        self.episode_experiences: List[LearningExperience] = []
        
    async def learn(self, experience: LearningExperience) -> Dict[str, Any]:
        """Learn from policy gradient experience"""
        try:
            self.episode_experiences.append(experience)
            
            # Learn at end of episode
            if experience.done:
                await self._update_policy_gradient()
                policy_value = self._evaluate_policy_value(experience.state, experience.action)
                self.episode_experiences = []  # Reset for next episode
                
                return {
                    'algorithm': 'Policy Gradient',
                    'policy_value': policy_value,
                    'baseline': self.baseline
                }
            else:
                return {
                    'algorithm': 'Policy Gradient',
                    'status': 'experience_recorded'
                }
                
        except Exception as e:
            logger.error(f"Policy gradient learning failed: {e}")
            return {'error': str(e)}
    
    async def predict(self, state: LearningState) -> Tuple[LearningAction, float]:
        """Predict action using policy"""
        try:
            available_actions = self._get_available_actions(state)
            if not available_actions:
                return LearningAction("default", "wait"), 0.0
            
            # Calculate action probabilities
            action_probs = {}
            for action_id in available_actions:
                prob = self._calculate_action_probability(state, action_id)
                action_probs[action_id] = prob
            
            # Sample action based on probabilities
            actions, probs = zip(*action_probs.items())
            total_prob = sum(probs)
            
            if total_prob > 0:
                normalized_probs = [p / total_prob for p in probs]
                action_id = np.random.choice(actions, p=normalized_probs)
                confidence = action_probs[action_id]
            else:
                action_id = random.choice(available_actions)
                confidence = 1.0 / len(available_actions)
            
            action = LearningAction(
                action_id=action_id,
                action_type=action_id.split('_')[0] if '_' in action_id else action_id,
                parameters=state.context.get('action_params', {})
            )
            
            return action, confidence
            
        except Exception as e:
            logger.error(f"Policy gradient prediction failed: {e}")
            return LearningAction("error", "error"), 0.0
    
    async def update_policy(self, experiences: List[LearningExperience]) -> None:
        """Update policy gradient parameters"""
        try:
            # Group experiences by episodes
            episodes = []
            current_episode = []
            
            for exp in experiences:
                current_episode.append(exp)
                if exp.done:
                    episodes.append(current_episode)
                    current_episode = []
            
            # Update policy for each complete episode
            for episode in episodes:
                await self._update_policy_from_episode(episode)
                
        except Exception as e:
            logger.error(f"Policy gradient update failed: {e}")
    
    async def _update_policy_gradient(self) -> None:
        """Update policy using REINFORCE algorithm"""
        try:
            if not self.episode_experiences:
                return
            
            # Calculate returns (discounted rewards)
            returns = []
            G = 0
            for experience in reversed(self.episode_experiences):
                G = experience.reward + self.discount_factor * G
                returns.append(G)
            
            returns.reverse()
            
            # Update baseline (moving average of returns)
            episode_return = returns[0] if returns else 0
            self.baseline = 0.9 * self.baseline + 0.1 * episode_return
            
            # Update policy parameters
            for i, experience in enumerate(self.episode_experiences):
                advantage = returns[i] - self.baseline
                
                # Update policy parameters for this state-action pair
                state_key = self._state_to_key(experience.state)
                action_id = experience.action.action_id
                
                # Simple gradient update
                for feature_name, feature_value in experience.state.features.items():
                    gradient = advantage * feature_value
                    self.policy_params[action_id][feature_name] += self.learning_rate * gradient
                    
        except Exception as e:
            logger.error(f"Policy gradient update failed: {e}")
    
    async def _update_policy_from_episode(self, episode: List[LearningExperience]) -> None:
        """Update policy from complete episode"""
        self.episode_experiences = episode
        await self._update_policy_gradient()
    
    def _calculate_action_probability(self, state: LearningState, action_id: str) -> float:
        """Calculate probability of action given state"""
        # Linear policy: probability proportional to exp(theta^T * features)
        score = 0.0
        for feature_name, feature_value in state.features.items():
            weight = self.policy_params[action_id][feature_name]
            score += weight * feature_value
        
        return math.exp(score)
    
    def _evaluate_policy_value(self, state: LearningState, action: LearningAction) -> float:
        """Evaluate policy value for state-action pair"""
        return self._calculate_action_probability(state, action.action_id)
    
    def _state_to_key(self, state: LearningState) -> str:
        """Convert state to string key"""
        return "_".join(f"{k}_{v:.2f}" for k, v in sorted(state.features.items()))
    
    def _get_available_actions(self, state: LearningState) -> List[str]:
        """Get available actions for state"""
        return state.context.get('available_actions', [
            'move_forward', 'move_backward', 'turn_left', 'turn_right',
            'interact', 'wait', 'explore', 'exploit'
        ])
    
    def get_learning_progress(self) -> Dict[str, float]:
        """Get policy gradient progress metrics"""
        total_params = sum(len(params) for params in self.policy_params.values())
        avg_param_magnitude = 0.0
        
        if total_params > 0:
            all_params = [
                abs(param) for params in self.policy_params.values() 
                for param in params.values()
            ]
            avg_param_magnitude = sum(all_params) / len(all_params)
        
        return {
            'policy_size': len(self.policy_params),
            'total_parameters': total_params,
            'average_parameter_magnitude': avg_param_magnitude,
            'baseline_value': self.baseline,
            'learning_rate': self.learning_rate
        }


class KnowledgeBase:
    """Knowledge storage and retrieval system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.knowledge_items: Dict[str, KnowledgeItem] = {}
        self.knowledge_index: Dict[KnowledgeType, Set[str]] = defaultdict(set)
        self.context_index: Dict[str, Set[str]] = defaultdict(set)
        self.max_knowledge_items = config.get('max_items', 10000)
        
    async def store_knowledge(self, knowledge: KnowledgeItem) -> bool:
        """Store knowledge item"""
        try:
            # Check capacity
            if len(self.knowledge_items) >= self.max_knowledge_items:
                await self._cleanup_knowledge()
            
            # Store knowledge
            self.knowledge_items[knowledge.knowledge_id] = knowledge
            
            # Update indices
            self.knowledge_index[knowledge.knowledge_type].add(knowledge.knowledge_id)
            
            for context_key, context_value in knowledge.context.items():
                if isinstance(context_value, str):
                    self.context_index[f"{context_key}_{context_value}"].add(knowledge.knowledge_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Knowledge storage failed: {e}")
            return False
    
    async def retrieve_knowledge(self, knowledge_type: KnowledgeType = None,
                               context: Dict[str, Any] = None,
                               min_confidence: float = 0.5) -> List[KnowledgeItem]:
        """Retrieve knowledge items"""
        try:
            candidate_ids = set()
            
            # Filter by type
            if knowledge_type:
                candidate_ids = self.knowledge_index[knowledge_type].copy()
            else:
                candidate_ids = set(self.knowledge_items.keys())
            
            # Filter by context
            if context:
                context_matches = set()
                for key, value in context.items():
                    if isinstance(value, str):
                        context_key = f"{key}_{value}"
                        context_matches.update(self.context_index[context_key])
                
                if context_matches:
                    candidate_ids &= context_matches
            
            # Filter by confidence and expiry
            valid_knowledge = []
            for kid in candidate_ids:
                knowledge = self.knowledge_items.get(kid)
                if (knowledge and 
                    not knowledge.is_expired() and 
                    knowledge.confidence >= min_confidence):
                    valid_knowledge.append(knowledge)
            
            # Sort by confidence and usage
            valid_knowledge.sort(
                key=lambda k: (k.confidence, k.success_rate, k.usage_count),
                reverse=True
            )
            
            return valid_knowledge
            
        except Exception as e:
            logger.error(f"Knowledge retrieval failed: {e}")
            return []
    
    async def update_knowledge_usage(self, knowledge_id: str, success: bool) -> bool:
        """Update knowledge usage statistics"""
        try:
            if knowledge_id in self.knowledge_items:
                self.knowledge_items[knowledge_id].update_usage(success)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Knowledge usage update failed: {e}")
            return False
    
    async def _cleanup_knowledge(self) -> None:
        """Clean up old or low-value knowledge"""
        try:
            # Sort by value (combination of confidence, success rate, and recency)
            knowledge_list = list(self.knowledge_items.values())
            knowledge_list.sort(key=lambda k: (
                k.confidence * k.success_rate * 
                (1.0 if k.last_used and (datetime.now() - k.last_used).days < 30 else 0.5)
            ))
            
            # Remove lowest value items
            items_to_remove = knowledge_list[:len(knowledge_list) // 4]  # Remove 25%
            
            for knowledge in items_to_remove:
                await self._remove_knowledge(knowledge.knowledge_id)
                
        except Exception as e:
            logger.error(f"Knowledge cleanup failed: {e}")
    
    async def _remove_knowledge(self, knowledge_id: str) -> None:
        """Remove knowledge item and update indices"""
        if knowledge_id not in self.knowledge_items:
            return
        
        knowledge = self.knowledge_items[knowledge_id]
        
        # Remove from indices
        self.knowledge_index[knowledge.knowledge_type].discard(knowledge_id)
        
        for context_key, context_value in knowledge.context.items():
            if isinstance(context_value, str):
                self.context_index[f"{context_key}_{context_value}"].discard(knowledge_id)
        
        # Remove knowledge item
        del self.knowledge_items[knowledge_id]


class AdaptationEngine:
    """Engine for agent adaptation and evolution"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adaptation_threshold = config.get('adaptation_threshold', 0.1)
        self.adaptation_history: List[AdaptationEvent] = []
        self.performance_window = deque(maxlen=config.get('performance_window', 100))
        
    async def evaluate_adaptation_need(self, agent_id: str, 
                                     current_performance: float,
                                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate if agent needs adaptation"""
        try:
            self.performance_window.append(current_performance)
            
            if len(self.performance_window) < 10:
                return {'needs_adaptation': False, 'reason': 'insufficient_data'}
            
            # Calculate performance trend
            recent_performance = list(self.performance_window)[-10:]
            older_performance = list(self.performance_window)[:-10] if len(self.performance_window) > 10 else []
            
            recent_avg = sum(recent_performance) / len(recent_performance)
            
            if older_performance:
                older_avg = sum(older_performance) / len(older_performance)
                performance_change = recent_avg - older_avg
            else:
                performance_change = 0
            
            # Check adaptation triggers
            adaptation_needed = False
            trigger = None
            
            if performance_change < -self.adaptation_threshold:
                adaptation_needed = True
                trigger = "performance_decline"
            elif recent_avg < 0.3:  # Absolute poor performance
                adaptation_needed = True
                trigger = "poor_performance"
            elif context and context.get('environment_changed', False):
                adaptation_needed = True
                trigger = "environment_change"
            
            return {
                'needs_adaptation': adaptation_needed,
                'trigger': trigger,
                'current_performance': current_performance,
                'recent_average': recent_avg,
                'performance_change': performance_change,
                'suggested_adaptations': await self._suggest_adaptations(trigger, context) if adaptation_needed else []
            }
            
        except Exception as e:
            logger.error(f"Adaptation evaluation failed: {e}")
            return {'needs_adaptation': False, 'error': str(e)}
    
    async def execute_adaptation(self, agent_id: str, adaptation_type: AdaptationType,
                               changes: Dict[str, Any], context: Dict[str, Any] = None) -> AdaptationEvent:
        """Execute adaptation on agent"""
        try:
            # Get current performance
            current_performance = context.get('current_performance', 0.0) if context else 0.0
            
            # Create adaptation event
            event = AdaptationEvent(
                event_id=f"adapt_{uuid.uuid4().hex[:8]}",
                agent_id=agent_id,
                adaptation_type=adaptation_type,
                trigger=context.get('trigger', 'manual') if context else 'manual',
                changes=changes,
                performance_before=current_performance,
                metadata=context or {}
            )
            
            # Execute adaptation based on type
            success = await self._execute_adaptation_type(adaptation_type, changes)
            
            event.success = success
            self.adaptation_history.append(event)
            
            # Limit history size
            if len(self.adaptation_history) > 1000:
                self.adaptation_history = self.adaptation_history[-1000:]
            
            return event
            
        except Exception as e:
            logger.error(f"Adaptation execution failed: {e}")
            return AdaptationEvent(
                event_id="error",
                agent_id=agent_id,
                adaptation_type=adaptation_type,
                trigger="error",
                changes={},
                performance_before=0.0,
                success=False,
                metadata={'error': str(e)}
            )
    
    async def _suggest_adaptations(self, trigger: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Suggest adaptations based on trigger"""
        suggestions = []
        
        if trigger == "performance_decline":
            suggestions.extend([
                {
                    'type': AdaptationType.PARAMETRIC,
                    'changes': {'learning_rate': 'increase'},
                    'reason': 'Increase learning rate to adapt faster'
                },
                {
                    'type': AdaptationType.BEHAVIORAL,
                    'changes': {'exploration_rate': 'increase'},
                    'reason': 'Increase exploration to find better strategies'
                },
                {
                    'type': AdaptationType.STRATEGIC,
                    'changes': {'strategy': 'diversify'},
                    'reason': 'Try different strategic approaches'
                }
            ])
        
        elif trigger == "poor_performance":
            suggestions.extend([
                {
                    'type': AdaptationType.STRUCTURAL,
                    'changes': {'architecture': 'simplify'},
                    'reason': 'Simplify to reduce complexity'
                },
                {
                    'type': AdaptationType.PARAMETRIC,
                    'changes': {'reset_parameters': True},
                    'reason': 'Reset parameters to escape local minimum'
                }
            ])
        
        elif trigger == "environment_change":
            suggestions.extend([
                {
                    'type': AdaptationType.ENVIRONMENTAL,
                    'changes': {'recalibrate': True},
                    'reason': 'Recalibrate to new environment'
                },
                {
                    'type': AdaptationType.BEHAVIORAL,
                    'changes': {'exploration_rate': 'increase'},
                    'reason': 'Explore new environment characteristics'
                }
            ])
        
        return suggestions
    
    async def _execute_adaptation_type(self, adaptation_type: AdaptationType, changes: Dict[str, Any]) -> bool:
        """Execute specific adaptation type"""
        try:
            if adaptation_type == AdaptationType.PARAMETRIC:
                return await self._adapt_parameters(changes)
            elif adaptation_type == AdaptationType.BEHAVIORAL:
                return await self._adapt_behavior(changes)
            elif adaptation_type == AdaptationType.STRUCTURAL:
                return await self._adapt_structure(changes)
            elif adaptation_type == AdaptationType.STRATEGIC:
                return await self._adapt_strategy(changes)
            elif adaptation_type == AdaptationType.ENVIRONMENTAL:
                return await self._adapt_environment(changes)
            else:
                logger.warning(f"Unsupported adaptation type: {adaptation_type}")
                return False
                
        except Exception as e:
            logger.error(f"Adaptation execution failed: {e}")
            return False
    
    async def _adapt_parameters(self, changes: Dict[str, Any]) -> bool:
        """Adapt agent parameters"""
        # Implementation would modify agent parameters
        logger.info(f"Adapting parameters: {changes}")
        return True
    
    async def _adapt_behavior(self, changes: Dict[str, Any]) -> bool:
        """Adapt agent behavior"""
        # Implementation would modify agent behavior patterns
        logger.info(f"Adapting behavior: {changes}")
        return True
    
    async def _adapt_structure(self, changes: Dict[str, Any]) -> bool:
        """Adapt agent structure"""
        # Implementation would modify agent architecture
        logger.info(f"Adapting structure: {changes}")
        return True
    
    async def _adapt_strategy(self, changes: Dict[str, Any]) -> bool:
        """Adapt agent strategy"""
        # Implementation would modify agent strategy
        logger.info(f"Adapting strategy: {changes}")
        return True
    
    async def _adapt_environment(self, changes: Dict[str, Any]) -> bool:
        """Adapt to environment changes"""
        # Implementation would handle environment adaptation
        logger.info(f"Adapting to environment: {changes}")
        return True


class AgentLearningAdaptationSystem:
    """Main agent learning and adaptation system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.system_id = f"learn_sys_{uuid.uuid4().hex[:8]}"
        
        # Initialize components
        self.learning_algorithms: Dict[LearningAlgorithm, BaseLearningAlgorithm] = {}
        self._initialize_algorithms()
        
        self.knowledge_base = KnowledgeBase(config.get('knowledge', {}))
        self.adaptation_engine = AdaptationEngine(config.get('adaptation', {}))
        
        # System state
        self.active_learners: Dict[str, Dict[str, Any]] = {}
        self.learning_sessions: Dict[str, List[LearningExperience]] = {}
        self.metrics = LearningMetrics()
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.is_running = False
    
    def _initialize_algorithms(self) -> None:
        """Initialize learning algorithms"""
        try:
            self.learning_algorithms[LearningAlgorithm.Q_LEARNING] = QLearningAlgorithm(
                self.config.get('q_learning', {})
            )
            
            self.learning_algorithms[LearningAlgorithm.POLICY_GRADIENT] = PolicyGradientAlgorithm(
                self.config.get('policy_gradient', {})
            )
            
            logger.info(f"Initialized {len(self.learning_algorithms)} learning algorithms")
            
        except Exception as e:
            logger.error(f"Algorithm initialization failed: {e}")
            raise
    
    async def start(self) -> None:
        """Start the learning and adaptation system"""
        try:
            self.is_running = True
            
            # Start background tasks
            self.background_tasks.add(
                asyncio.create_task(self._learning_monitor())
            )
            
            self.background_tasks.add(
                asyncio.create_task(self._adaptation_monitor())
            )
            
            logger.info(f"Learning system {self.system_id} started")
            
        except Exception as e:
            logger.error(f"Learning system start failed: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the learning and adaptation system"""
        try:
            self.is_running = False
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            self.background_tasks.clear()
            
            logger.info(f"Learning system {self.system_id} stopped")
            
        except Exception as e:
            logger.error(f"Learning system stop failed: {e}")
    
    async def create_learner(self, agent_id: str, algorithm: LearningAlgorithm,
                           config: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new learning agent"""
        try:
            if algorithm not in self.learning_algorithms:
                raise ValueError(f"Unsupported learning algorithm: {algorithm}")
            
            learner_config = config or {}
            
            # Create learner
            self.active_learners[agent_id] = {
                'algorithm': algorithm,
                'algorithm_instance': self.learning_algorithms[algorithm],
                'config': learner_config,
                'created_at': datetime.now(),
                'last_update': datetime.now(),
                'total_experiences': 0,
                'performance_history': deque(maxlen=1000),
                'learning_mode': LearningMode.EXPLORATION
            }
            
            # Initialize learning session
            self.learning_sessions[agent_id] = []
            
            logger.info(f"Created learner {agent_id} with {algorithm.name}")
            return True
            
        except Exception as e:
            logger.error(f"Learner creation failed: {e}")
            return False
    
    async def learn_from_experience(self, agent_id: str, experience: LearningExperience) -> Dict[str, Any]:
        """Process learning experience for agent"""
        try:
            if agent_id not in self.active_learners:
                raise ValueError(f"Agent {agent_id} not found")
            
            learner = self.active_learners[agent_id]
            algorithm_instance = learner['algorithm_instance']
            
            # Learn from experience
            learning_result = await algorithm_instance.learn(experience)
            
            # Update learner state
            learner['total_experiences'] += 1
            learner['last_update'] = datetime.now()
            learner['performance_history'].append(experience.reward)
            
            # Store experience in session
            self.learning_sessions[agent_id].append(experience)
            
            # Update metrics
            self.metrics.total_experiences += 1
            if experience.reward > 0:
                self.metrics.successful_experiences += 1
            
            self.metrics.cumulative_reward += experience.reward
            self.metrics.average_reward = (
                self.metrics.cumulative_reward / self.metrics.total_experiences
            )
            
            # Store as knowledge if valuable
            if experience.reward > 0.5:  # Threshold for valuable experiences
                await self._store_experience_knowledge(experience)
            
            return learning_result
            
        except Exception as e:
            logger.error(f"Learning from experience failed: {e}")
            return {'error': str(e)}
    
    async def get_action_recommendation(self, agent_id: str, state: LearningState) -> Tuple[LearningAction, float]:
        """Get action recommendation for agent"""
        try:
            if agent_id not in self.active_learners:
                raise ValueError(f"Agent {agent_id} not found")
            
            learner = self.active_learners[agent_id]
            algorithm_instance = learner['algorithm_instance']
            
            # Get action from learning algorithm
            action, confidence = await algorithm_instance.predict(state)
            
            # Enhance with knowledge base
            relevant_knowledge = await self.knowledge_base.retrieve_knowledge(
                knowledge_type=KnowledgeType.PROCEDURAL,
                context=state.context
            )
            
            if relevant_knowledge:
                # Use knowledge to refine action
                knowledge_action = await self._apply_knowledge_to_action(
                    action, relevant_knowledge, state
                )
                if knowledge_action:
                    action = knowledge_action
                    confidence = min(confidence * 1.2, 1.0)  # Boost confidence
            
            return action, confidence
            
        except Exception as e:
            logger.error(f"Action recommendation failed: {e}")
            return LearningAction("error", "error"), 0.0
    
    async def evaluate_and_adapt(self, agent_id: str, performance: float,
                               context: Dict[str, Any] = None) -> Optional[AdaptationEvent]:
        """Evaluate performance and adapt if needed"""
        try:
            # Evaluate adaptation need
            adaptation_eval = await self.adaptation_engine.evaluate_adaptation_need(
                agent_id, performance, context
            )
            
            if not adaptation_eval.get('needs_adaptation', False):
                return None
            
            # Select best adaptation
            suggestions = adaptation_eval.get('suggested_adaptations', [])
            if not suggestions:
                return None
            
            # Execute first suggested adaptation
            suggestion = suggestions[0]
            adaptation_event = await self.adaptation_engine.execute_adaptation(
                agent_id=agent_id,
                adaptation_type=suggestion['type'],
                changes=suggestion['changes'],
                context=context
            )
            
            # Update metrics
            self.metrics.adaptation_events += 1
            if adaptation_event.success:
                self.metrics.successful_adaptations += 1
            
            return adaptation_event
            
        except Exception as e:
            logger.error(f"Evaluation and adaptation failed: {e}")
            return None
    
    async def transfer_knowledge(self, source_agent: str, target_agent: str,
                               knowledge_filter: Optional[Dict[str, Any]] = None) -> bool:
        """Transfer knowledge between agents"""
        try:
            if source_agent not in self.active_learners or target_agent not in self.active_learners:
                return False
            
            # Get source agent experiences
            source_experiences = self.learning_sessions.get(source_agent, [])
            if not source_experiences:
                return False
            
            # Filter valuable experiences
            valuable_experiences = [
                exp for exp in source_experiences
                if exp.reward > 0.3  # Threshold for valuable experiences
            ]
            
            if not valuable_experiences:
                return False
            
            # Apply knowledge filter if provided
            if knowledge_filter:
                # Implementation would filter based on criteria
                pass
            
            # Transfer to target agent
            target_algorithm = self.active_learners[target_agent]['algorithm_instance']
            
            transfer_count = 0
            for experience in valuable_experiences:
                # Modify experience for transfer (could include domain adaptation)
                transferred_experience = LearningExperience(
                    experience_id=f"transfer_{uuid.uuid4().hex[:8]}",
                    agent_id=target_agent,
                    state=experience.state,
                    action=experience.action,
                    reward=experience.reward * 0.8,  # Discount for transfer
                    next_state=experience.next_state,
                    done=experience.done,
                    metadata={**experience.metadata, 'transferred_from': source_agent}
                )
                
                await target_algorithm.learn(transferred_experience)
                transfer_count += 1
            
            # Update metrics
            if transfer_count > 0:
                self.metrics.transfer_success_rate = (
                    (self.metrics.transfer_success_rate * self.metrics.adaptation_events + 1.0) /
                    (self.metrics.adaptation_events + 1)
                )
            
            logger.info(f"Transferred {transfer_count} experiences from {source_agent} to {target_agent}")
            return transfer_count > 0
            
        except Exception as e:
            logger.error(f"Knowledge transfer failed: {e}")
            return False
    
    async def _store_experience_knowledge(self, experience: LearningExperience) -> None:
        """Store experience as knowledge"""
        try:
            knowledge = KnowledgeItem(
                knowledge_id=f"exp_knowledge_{uuid.uuid4().hex[:8]}",
                knowledge_type=KnowledgeType.EPISODIC,
                content={
                    'state_features': experience.state.features,
                    'action': {
                        'id': experience.action.action_id,
                        'type': experience.action.action_type,
                        'parameters': experience.action.parameters
                    },
                    'reward': experience.reward,
                    'success': experience.reward > 0
                },
                confidence=min(experience.reward, 1.0),
                context=experience.state.context,
                source="experience"
            )
            
            await self.knowledge_base.store_knowledge(knowledge)
            self.metrics.knowledge_items += 1
            
        except Exception as e:
            logger.error(f"Experience knowledge storage failed: {e}")
    
    async def _apply_knowledge_to_action(self, action: LearningAction, 
                                       knowledge: List[KnowledgeItem],
                                       state: LearningState) -> Optional[LearningAction]:
        """Apply knowledge to refine action"""
        try:
            # Find most relevant knowledge
            best_knowledge = None
            best_similarity = 0.0
            
            for knowledge_item in knowledge:
                if knowledge_item.knowledge_type != KnowledgeType.EPISODIC:
                    continue
                
                # Calculate similarity based on state features
                knowledge_features = knowledge_item.content.get('state_features', {})
                similarity = self._calculate_feature_similarity(state.features, knowledge_features)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_knowledge = knowledge_item
            
            # Apply knowledge if sufficiently similar
            if best_knowledge and best_similarity > 0.7:
                knowledge_action_info = best_knowledge.content.get('action', {})
                if knowledge_action_info.get('id') != action.action_id:
                    # Suggest different action based on knowledge
                    return LearningAction(
                        action_id=knowledge_action_info.get('id', action.action_id),
                        action_type=knowledge_action_info.get('type', action.action_type),
                        parameters=knowledge_action_info.get('parameters', action.parameters)
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Knowledge application failed: {e}")
            return None
    
    def _calculate_feature_similarity(self, features1: Dict[str, float], 
                                    features2: Dict[str, float]) -> float:
        """Calculate similarity between feature sets"""
        try:
            all_features = set(features1.keys()) | set(features2.keys())
            if not all_features:
                return 1.0
            
            similarity_sum = 0.0
            for feature in all_features:
                val1 = features1.get(feature, 0.0)
                val2 = features2.get(feature, 0.0)
                
                # Normalized difference
                max_val = max(abs(val1), abs(val2), 1.0)
                similarity = 1.0 - abs(val1 - val2) / max_val
                similarity_sum += similarity
            
            return similarity_sum / len(all_features)
            
        except Exception as e:
            logger.error(f"Feature similarity calculation failed: {e}")
            return 0.0
    
    def get_learner_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a learning agent"""
        if agent_id not in self.active_learners:
            return None
        
        learner = self.active_learners[agent_id]
        algorithm_instance = learner['algorithm_instance']
        
        # Get performance statistics
        performance_history = list(learner['performance_history'])
        avg_performance = sum(performance_history) / len(performance_history) if performance_history else 0.0
        
        return {
            'agent_id': agent_id,
            'algorithm': learner['algorithm'].name,
            'total_experiences': learner['total_experiences'],
            'average_performance': avg_performance,
            'recent_performance': performance_history[-10:] if performance_history else [],
            'learning_mode': learner['learning_mode'].name,
            'created_at': learner['created_at'],
            'last_update': learner['last_update'],
            'algorithm_progress': algorithm_instance.get_learning_progress()
        }
    
    def get_system_metrics(self) -> LearningMetrics:
        """Get system-wide learning metrics"""
        # Update knowledge metrics
        self.metrics.knowledge_items = len(self.knowledge_base.knowledge_items)
        
        # Calculate knowledge utilization
        total_usage = sum(
            item.usage_count for item in self.knowledge_base.knowledge_items.values()
        )
        self.metrics.knowledge_utilization = (
            total_usage / self.metrics.knowledge_items if self.metrics.knowledge_items > 0 else 0.0
        )
        
        self.metrics.last_updated = datetime.now()
        return self.metrics
    
    def list_active_learners(self) -> List[Dict[str, Any]]:
        """List all active learning agents"""
        return [
            {
                'agent_id': agent_id,
                'algorithm': info['algorithm'].name,
                'total_experiences': info['total_experiences'],
                'learning_mode': info['learning_mode'].name,
                'last_update': info['last_update']
            }
            for agent_id, info in self.active_learners.items()
        ]
    
    async def _learning_monitor(self) -> None:
        """Background task for monitoring learning progress"""
        while self.is_running:
            try:
                # Update learning rates and exploration
                for agent_id, learner in self.active_learners.items():
                    performance_history = list(learner['performance_history'])
                    
                    if len(performance_history) >= 50:
                        recent_performance = sum(performance_history[-10:]) / 10
                        older_performance = sum(performance_history[-50:-40]) / 10
                        
                        # Adjust learning mode based on performance trend
                        if recent_performance > older_performance * 1.1:
                            learner['learning_mode'] = LearningMode.EXPLOITATION
                        elif recent_performance < older_performance * 0.9:
                            learner['learning_mode'] = LearningMode.EXPLORATION
                        else:
                            learner['learning_mode'] = LearningMode.EXPLORATION_EXPLOITATION
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Learning monitoring failed: {e}")
                await asyncio.sleep(60)
    
    async def _adaptation_monitor(self) -> None:
        """Background task for monitoring adaptation needs"""
        while self.is_running:
            try:
                # Check for agents that might need adaptation
                for agent_id, learner in self.active_learners.items():
                    performance_history = list(learner['performance_history'])
                    
                    if len(performance_history) >= 20:
                        recent_avg = sum(performance_history[-10:]) / 10
                        
                        # Trigger adaptation evaluation for poorly performing agents
                        if recent_avg < 0.3:
                            await self.evaluate_and_adapt(agent_id, recent_avg, {
                                'trigger': 'monitoring',
                                'current_performance': recent_avg
                            })
                
                await asyncio.sleep(300)  # Monitor every 5 minutes
                
            except Exception as e:
                logger.error(f"Adaptation monitoring failed: {e}")
                await asyncio.sleep(300)


async def example_learning_adaptation_usage():
    """Comprehensive example of agent learning and adaptation system usage"""
    
    print("\n🧠 Agent Learning & Adaptation System Example")
    print("=" * 60)
    
    # Configuration
    config = {
        'q_learning': {
            'learning_rate': 0.1,
            'discount_factor': 0.99,
            'epsilon': 0.1,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01
        },
        'policy_gradient': {
            'learning_rate': 0.01,
            'discount_factor': 0.99
        },
        'knowledge': {
            'max_items': 1000
        },
        'adaptation': {
            'adaptation_threshold': 0.1,
            'performance_window': 100
        }
    }
    
    # Initialize system
    learning_system = AgentLearningAdaptationSystem(config)
    await learning_system.start()
    
    print(f"✅ Learning system {learning_system.system_id} started")
    
    try:
        # Example 1: Q-Learning Agent
        print("\n1. Q-Learning Agent Training")
        print("-" * 40)
        
        success = await learning_system.create_learner(
            agent_id="q_agent_1",
            algorithm=LearningAlgorithm.Q_LEARNING,
            config={'specialized_for': 'navigation'}
        )
        
        print(f"✅ Q-Learning agent created: {success}")
        
        # Simulate learning episodes
        for episode in range(5):
            episode_reward = 0.0
            state = LearningState(
                state_id=f"state_{episode}_0",
                features={'x': random.uniform(-5, 5), 'y': random.uniform(-5, 5), 'energy': 100.0},
                context={'available_actions': ['move_forward', 'turn_left', 'turn_right', 'wait']}
            )
            
            for step in range(10):
                # Get action recommendation
                action, confidence = await learning_system.get_action_recommendation("q_agent_1", state)
                
                # Simulate environment response
                reward = random.uniform(-1, 2) if action.action_id != 'wait' else -0.1
                episode_reward += reward
                
                # Create next state
                next_state = LearningState(
                    state_id=f"state_{episode}_{step+1}",
                    features={
                        'x': state.features['x'] + random.uniform(-0.5, 0.5),
                        'y': state.features['y'] + random.uniform(-0.5, 0.5),
                        'energy': max(0, state.features['energy'] - 1)
                    },
                    context=state.context
                )
                
                # Create experience
                experience = LearningExperience(
                    experience_id=f"exp_{episode}_{step}",
                    agent_id="q_agent_1",
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=(step == 9),
                    metadata={'episode': episode, 'step': step}
                )
                
                # Learn from experience
                learning_result = await learning_system.learn_from_experience("q_agent_1", experience)
                
                state = next_state
                
                if experience.done:
                    print(f"   Episode {episode}: Total reward = {episode_reward:.2f}")
        
        # Example 2: Policy Gradient Agent
        print("\n2. Policy Gradient Agent Training")
        print("-" * 40)
        
        success = await learning_system.create_learner(
            agent_id="pg_agent_1",
            algorithm=LearningAlgorithm.POLICY_GRADIENT
        )
        
        print(f"✅ Policy Gradient agent created: {success}")
        
        # Simulate learning episodes
        for episode in range(3):
            episode_experiences = []
            state = LearningState(
                state_id=f"pg_state_{episode}_0",
                features={'position': 0.0, 'velocity': 0.0, 'goal_distance': 10.0},
                context={'available_actions': ['accelerate', 'decelerate', 'maintain']}
            )
            
            for step in range(15):
                action, confidence = await learning_system.get_action_recommendation("pg_agent_1", state)
                
                # Simulate physics
                if action.action_id == 'accelerate':
                    velocity_change = 0.5
                    reward = -0.1  # Energy cost
                elif action.action_id == 'decelerate':
                    velocity_change = -0.3
                    reward = -0.05
                else:
                    velocity_change = 0.0
                    reward = -0.01
                
                new_velocity = max(-2, min(2, state.features['velocity'] + velocity_change))
                new_position = state.features['position'] + new_velocity
                new_goal_distance = abs(10.0 - new_position)
                
                # Reward for getting closer to goal
                if new_goal_distance < state.features['goal_distance']:
                    reward += 0.5
                
                # Bonus for reaching goal
                if new_goal_distance < 0.5:
                    reward += 10.0
                
                next_state = LearningState(
                    state_id=f"pg_state_{episode}_{step+1}",
                    features={'position': new_position, 'velocity': new_velocity, 'goal_distance': new_goal_distance},
                    context=state.context
                )
                
                experience = LearningExperience(
                    experience_id=f"pg_exp_{episode}_{step}",
                    agent_id="pg_agent_1",
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=(step == 14 or new_goal_distance < 0.5),
                    metadata={'episode': episode}
                )
                
                episode_experiences.append(experience)
                await learning_system.learn_from_experience("pg_agent_1", experience)
                
                state = next_state
                if experience.done:
                    break
            
            episode_reward = sum(exp.reward for exp in episode_experiences)
            print(f"   Episode {episode}: Total reward = {episode_reward:.2f}, Goal distance = {state.features['goal_distance']:.2f}")
        
        # Example 3: Knowledge Transfer
        print("\n3. Knowledge Transfer Between Agents")
        print("-" * 40)
        
        # Create third agent
        await learning_system.create_learner(
            agent_id="transfer_agent",
            algorithm=LearningAlgorithm.Q_LEARNING
        )
        
        # Transfer knowledge from q_agent_1 to transfer_agent
        transfer_success = await learning_system.transfer_knowledge(
            source_agent="q_agent_1",
            target_agent="transfer_agent"
        )
        
        print(f"✅ Knowledge transfer successful: {transfer_success}")
        
        # Example 4: Adaptation System
        print("\n4. Agent Adaptation")
        print("-" * 40)
        
        # Simulate poor performance to trigger adaptation
        poor_performances = [0.1, 0.15, 0.2, 0.05, 0.12]
        
        for i, performance in enumerate(poor_performances):
            adaptation_event = await learning_system.evaluate_and_adapt(
                agent_id="q_agent_1",
                performance=performance,
                context={'current_performance': performance, 'step': i}
            )
            
            if adaptation_event:
                print(f"   Adaptation triggered: {adaptation_event.adaptation_type.name}")
                print(f"   Changes: {adaptation_event.changes}")
                print(f"   Success: {adaptation_event.success}")
            else:
                print(f"   No adaptation needed for performance: {performance}")
        
        # Example 5: System Status and Metrics
        print("\n5. Learning System Status")
        print("-" * 40)
        
        active_learners = learning_system.list_active_learners()
        print(f"✅ Active learners: {len(active_learners)}")
        
        for learner_info in active_learners:
            print(f"   - {learner_info['agent_id']}: {learner_info['algorithm']}")
            print(f"     Experiences: {learner_info['total_experiences']}")
            print(f"     Mode: {learner_info['learning_mode']}")
            
            # Get detailed status
            status = learning_system.get_learner_status(learner_info['agent_id'])
            if status:
                print(f"     Avg Performance: {status['average_performance']:.3f}")
                print(f"     Recent Performance: {[f'{p:.2f}' for p in status['recent_performance'][-3:]]}")
        
        # Example 6: Global Metrics
        print("\n6. Global Learning Metrics")
        print("-" * 40)
        
        metrics = learning_system.get_system_metrics()
        print(f"✅ Total experiences: {metrics.total_experiences}")
        print(f"✅ Successful experiences: {metrics.successful_experiences}")
        print(f"✅ Average reward: {metrics.average_reward:.3f}")
        print(f"✅ Cumulative reward: {metrics.cumulative_reward:.2f}")
        print(f"✅ Knowledge items: {metrics.knowledge_items}")
        print(f"✅ Knowledge utilization: {metrics.knowledge_utilization:.3f}")
        print(f"✅ Adaptation events: {metrics.adaptation_events}")
        print(f"✅ Successful adaptations: {metrics.successful_adaptations}")
        print(f"✅ Transfer success rate: {metrics.transfer_success_rate:.3f}")
        
        # Allow background tasks to run briefly
        await asyncio.sleep(3)
        
    finally:
        # Cleanup
        await learning_system.stop()
        print(f"\n✅ Learning and adaptation system stopped successfully")


if __name__ == "__main__":
    asyncio.run(example_learning_adaptation_usage())