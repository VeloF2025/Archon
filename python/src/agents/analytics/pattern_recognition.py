"""
Pattern Recognition Module
Advanced ML-based pattern detection and analysis for conversation analytics
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
import numpy as np
import pandas as pd
from collections import defaultdict, deque, Counter
import re
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import networkx as nx
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of conversation patterns"""
    SEQUENTIAL_FLOW = "sequential_flow"
    PARALLEL_COLLABORATION = "parallel_collaboration"
    ERROR_SPIRAL = "error_spiral"
    HANDOFF_LOOP = "handoff_loop"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    TASK_DECOMPOSITION = "task_decomposition"
    CONSENSUS_BUILDING = "consensus_building"
    EXPERT_CONSULTATION = "expert_consultation"
    RAPID_PROTOTYPING = "rapid_prototyping"
    DEBUGGING_SESSION = "debugging_session"


class PatternSeverity(Enum):
    """Severity levels for patterns"""
    OPTIMAL = "optimal"
    NORMAL = "normal"
    SUBOPTIMAL = "suboptimal"
    PROBLEMATIC = "problematic"
    CRITICAL = "critical"


@dataclass
class PatternFeature:
    """Represents a feature extracted for pattern recognition"""
    feature_id: str
    name: str
    value: float
    description: str
    category: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConversationPattern:
    """Represents a detected conversation pattern"""
    pattern_id: str
    pattern_type: PatternType
    severity: PatternSeverity
    confidence: float
    description: str
    agents_involved: Set[str]
    time_window: Tuple[datetime, datetime]
    features: List[PatternFeature] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    impact_score: float = 0.0


@dataclass
class PatternCluster:
    """Represents a cluster of similar patterns"""
    cluster_id: str
    pattern_type: PatternType
    patterns: List[ConversationPattern] = field(default_factory=list)
    centroid: np.ndarray = field(default_factory=lambda: np.array([]))
    characteristics: Dict[str, Any] = field(default_factory=dict)


class ConversationPatternRecognizer:
    """
    Advanced pattern recognition for conversation analytics using ML techniques
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.pattern_history: List[ConversationPattern] = []
        self.pattern_clusters: Dict[str, PatternCluster] = {}
        self.feature_extractor = ConversationFeatureExtractor()
        self.ml_models = self._initialize_ml_models()
        self.pattern_thresholds = self._initialize_thresholds()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def _initialize_ml_models(self) -> Dict[str, Any]:
        """Initialize ML models for pattern recognition"""
        return {
            'clustering': {
                'dbscan': DBSCAN(eps=0.5, min_samples=3),
                'kmeans': KMeans(n_clusters=5, random_state=42)
            },
            'dimensionality_reduction': {
                'pca': PCA(n_components=0.95),
                'scaler': StandardScaler()
            },
            'text_analysis': {
                'tfidf': TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
            }
        }

    def _initialize_thresholds(self) -> Dict[str, float]:
        """Initialize pattern detection thresholds"""
        return {
            'sequential_flow_confidence': 0.7,
            'error_spiral_threshold': 0.3,
            'handoff_loop_threshold': 3,
            'parallel_collaboration_threshold': 2,
            'consensus_building_threshold': 0.8,
            'min_pattern_confidence': 0.6
        }

    async def detect_conversation_patterns(self,
                                         conversation_data: List[Dict[str, Any]]) -> List[ConversationPattern]:
        """
        Detect patterns in conversation data

        Args:
            conversation_data: List of conversation events

        Returns:
            List of detected patterns
        """
        try:
            # Extract features
            features = await self.feature_extractor.extract_features(conversation_data)

            # Apply pattern detection algorithms
            patterns = []

            # Sequential flow patterns
            sequential_patterns = await self._detect_sequential_flow_patterns(
                conversation_data, features
            )
            patterns.extend(sequential_patterns)

            # Parallel collaboration patterns
            parallel_patterns = await self._detect_parallel_collaboration_patterns(
                conversation_data, features
            )
            patterns.extend(parallel_patterns)

            # Error spiral patterns
            error_patterns = await self._detect_error_spiral_patterns(
                conversation_data, features
            )
            patterns.extend(error_patterns)

            # Handoff loop patterns
            handoff_patterns = await self._detect_handoff_loop_patterns(
                conversation_data, features
            )
            patterns.extend(handoff_patterns)

            # Knowledge sharing patterns
            knowledge_patterns = await self._detect_knowledge_sharing_patterns(
                conversation_data, features
            )
            patterns.extend(knowledge_patterns)

            # Filter by confidence
            filtered_patterns = [
                p for p in patterns
                if p.confidence >= self.pattern_thresholds['min_pattern_confidence']
            ]

            # Store patterns
            self.pattern_history.extend(filtered_patterns)

            # Update clusters
            await self._update_pattern_clusters()

            return filtered_patterns

        except Exception as e:
            logger.error(f"Error in conversation pattern detection: {e}")
            return []

    async def _detect_sequential_flow_patterns(self,
                                             conversation_data: List[Dict[str, Any]],
                                             features: List[PatternFeature]) -> List[ConversationPattern]:
        """Detect sequential flow patterns in conversations"""
        patterns = []

        # Group by session
        session_groups = defaultdict(list)
        for event in conversation_data:
            session_id = event.get('session_id')
            if session_id:
                session_groups[session_id].append(event)

        for session_id, events in session_groups.items():
            # Analyze temporal sequence
            sorted_events = sorted(events, key=lambda x: x.get('timestamp', datetime.utcnow()))

            # Calculate sequence metrics
            sequence_score = self._calculate_sequence_score(sorted_events)
            if sequence_score >= self.pattern_thresholds['sequential_flow_confidence']:
                pattern = ConversationPattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type=PatternType.SEQUENTIAL_FLOW,
                    severity=PatternSeverity.OPTIMAL,
                    confidence=sequence_score,
                    description="Sequential task execution pattern detected",
                    agents_involved=set(e.get('agent_id') for e in events),
                    time_window=(
                        min(e.get('timestamp', datetime.utcnow()) for e in events),
                        max(e.get('timestamp', datetime.utcnow()) for e in events)
                    ),
                    impact_score=sequence_score * 0.8,
                    metadata={
                        'session_id': session_id,
                        'event_count': len(events),
                        'sequence_metrics': self._calculate_sequence_metrics(sorted_events)
                    }
                )
                patterns.append(pattern)

        return patterns

    def _calculate_sequence_score(self, events: List[Dict[str, Any]]) -> float:
        """Calculate score for sequential pattern"""
        if len(events) < 3:
            return 0.0

        # Analyze agent sequence consistency
        agent_sequence = [e.get('agent_id') for e in events]
        agent_changes = sum(1 for i in range(1, len(agent_sequence))
                           if agent_sequence[i] != agent_sequence[i-1])

        # Calculate temporal consistency
        timestamps = [e.get('timestamp', datetime.utcnow()) for e in events]
        time_intervals = [(timestamps[i] - timestamps[i-1]).total_seconds()
                          for i in range(1, len(timestamps))]

        # Calculate sequence score
        agent_consistency = 1.0 - (agent_changes / len(events))
        temporal_consistency = 1.0 - (np.std(time_intervals) / np.mean(time_intervals) if np.mean(time_intervals) > 0 else 0)

        sequence_score = (agent_consistency * 0.6 + temporal_consistency * 0.4)
        return min(1.0, max(0.0, sequence_score))

    def _calculate_sequence_metrics(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate detailed sequence metrics"""
        if not events:
            return {}

        timestamps = [e.get('timestamp', datetime.utcnow()) for e in events]
        time_intervals = [(timestamps[i] - timestamps[i-1]).total_seconds()
                          for i in range(1, len(timestamps))]

        return {
            'total_events': len(events),
            'unique_agents': len(set(e.get('agent_id') for e in events)),
            'avg_interval': float(np.mean(time_intervals)) if time_intervals else 0,
            'interval_std': float(np.std(time_intervals)) if time_intervals else 0,
            'agent_changes': sum(1 for i in range(1, len(events))
                               if events[i].get('agent_id') != events[i-1].get('agent_id'))
        }

    async def _detect_parallel_collaboration_patterns(self,
                                                     conversation_data: List[Dict[str, Any]],
                                                     features: List[PatternFeature]) -> List[ConversationPattern]:
        """Detect parallel collaboration patterns"""
        patterns = []

        # Group by time windows
        time_windows = self._create_time_windows(conversation_data, window_minutes=5)

        for window_start, window_events in time_windows.items():
            # Check for parallel activity
            active_agents = set(e.get('agent_id') for e in window_events)
            unique_sessions = set(e.get('session_id') for e in window_events)

            if len(active_agents) >= self.pattern_thresholds['parallel_collaboration_threshold']:
                parallel_score = self._calculate_parallel_score(window_events)

                if parallel_score >= self.pattern_thresholds['sequential_flow_confidence']:
                    pattern = ConversationPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type=PatternType.PARALLEL_COLLABORATION,
                        severity=PatternSeverity.OPTIMAL,
                        confidence=parallel_score,
                        description="Parallel collaboration pattern detected",
                        agents_involved=active_agents,
                        time_window=(
                            window_start,
                            window_start + timedelta(minutes=5)
                        ),
                        impact_score=parallel_score * 0.9,
                        metadata={
                            'active_agents': len(active_agents),
                            'session_count': len(unique_sessions),
                            'event_count': len(window_events),
                            'parallel_metrics': self._calculate_parallel_metrics(window_events)
                        }
                    )
                    patterns.append(pattern)

        return patterns

    def _create_time_windows(self, events: List[Dict[str, Any]], window_minutes: int = 5) -> Dict[datetime, List[Dict[str, Any]]]:
        """Create time windows for analysis"""
        time_windows = {}
        window_delta = timedelta(minutes=window_minutes)

        # Get time range
        timestamps = [e.get('timestamp', datetime.utcnow()) for e in events]
        if not timestamps:
            return {}

        start_time = min(timestamps)
        end_time = max(timestamps)

        # Create windows
        current_window = start_time
        while current_window <= end_time:
            window_end = current_window + window_delta
            window_events = [
                e for e in events
                if current_window <= e.get('timestamp', datetime.utcnow()) < window_end
            ]

            if window_events:
                time_windows[current_window] = window_events

            current_window = window_end

        return time_windows

    def _calculate_parallel_score(self, events: List[Dict[str, Any]]) -> float:
        """Calculate score for parallel collaboration"""
        if len(events) < 2:
            return 0.0

        active_agents = set(e.get('agent_id') for e in events)
        sessions = set(e.get('session_id') for e in events)

        # Calculate parallel activity metrics
        agent_diversity = len(active_agents) / max(len(events), 1)
        session_diversity = len(sessions) / max(len(events), 1)

        # Calculate temporal overlap
        timestamps = [e.get('timestamp', datetime.utcnow()) for e in events]
        time_spread = (max(timestamps) - min(timestamps)).total_seconds()
        temporal_density = len(events) / max(time_spread / 60, 1)  # Events per minute

        # Calculate parallel score
        parallel_score = (agent_diversity * 0.4 + session_diversity * 0.3 + min(temporal_density / 10, 1.0) * 0.3)
        return min(1.0, max(0.0, parallel_score))

    def _calculate_parallel_metrics(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate detailed parallel collaboration metrics"""
        if not events:
            return {}

        active_agents = set(e.get('agent_id') for e in events)
        sessions = set(e.get('session_id') for e in events)
        timestamps = [e.get('timestamp', datetime.utcnow()) for e in events]

        return {
            'active_agents': len(active_agents),
            'unique_sessions': len(sessions),
            'event_count': len(events),
            'time_spread_seconds': (max(timestamps) - min(timestamps)).total_seconds(),
            'events_per_minute': len(events) / max((max(timestamps) - min(timestamps)).total_seconds() / 60, 1)
        }

    async def _detect_error_spiral_patterns(self,
                                          conversation_data: List[Dict[str, Any]],
                                          features: List[PatternFeature]) -> List[ConversationPattern]:
        """Detect error spiral patterns"""
        patterns = []

        # Group by agent
        agent_groups = defaultdict(list)
        for event in conversation_data:
            agent_id = event.get('agent_id')
            if agent_id:
                agent_groups[agent_id].append(event)

        for agent_id, events in agent_groups.items():
            # Analyze error concentration
            error_events = [e for e in events if e.get('event_type') == 'error']
            if len(error_events) / len(events) > self.pattern_thresholds['error_spiral_threshold']:
                spiral_score = self._calculate_error_spiral_score(events, error_events)

                if spiral_score >= self.pattern_thresholds['min_pattern_confidence']:
                    severity = PatternSeverity.CRITICAL if spiral_score > 0.8 else PatternSeverity.PROBLEMATIC

                    pattern = ConversationPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type=PatternType.ERROR_SPIRAL,
                        severity=severity,
                        confidence=spiral_score,
                        description="Error spiral pattern detected",
                        agents_involved={agent_id},
                        time_window=(
                            min(e.get('timestamp', datetime.utcnow()) for e in events),
                            max(e.get('timestamp', datetime.utcnow()) for e in events)
                        ),
                        impact_score=spiral_score,
                        metadata={
                            'agent_id': agent_id,
                            'error_count': len(error_events),
                            'total_events': len(events),
                            'error_rate': len(error_events) / len(events),
                            'spiral_metrics': self._calculate_error_spiral_metrics(events, error_events)
                        }
                    )
                    patterns.append(pattern)

        return patterns

    def _calculate_error_spiral_score(self, all_events: List[Dict[str, Any]],
                                    error_events: List[Dict[str, Any]]) -> float:
        """Calculate score for error spiral pattern"""
        if len(error_events) < 2:
            return 0.0

        # Calculate error concentration
        error_rate = len(error_events) / len(all_events)

        # Calculate error acceleration
        error_timestamps = sorted([e.get('timestamp', datetime.utcnow()) for e in error_events])
        error_intervals = [(error_timestamps[i] - error_timestamps[i-1]).total_seconds()
                           for i in range(1, len(error_timestamps))]

        if error_intervals:
            acceleration_factor = 1.0 - (np.mean(error_intervals) / max(np.std(error_intervals), 1))
        else:
            acceleration_factor = 0.0

        # Calculate spiral score
        spiral_score = (error_rate * 0.6 + acceleration_factor * 0.4)
        return min(1.0, max(0.0, spiral_score))

    def _calculate_error_spiral_metrics(self, all_events: List[Dict[str, Any]],
                                      error_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate detailed error spiral metrics"""
        if not error_events:
            return {}

        error_timestamps = sorted([e.get('timestamp', datetime.utcnow()) for e in error_events])
        error_intervals = [(error_timestamps[i] - error_timestamps[i-1]).total_seconds()
                           for i in range(1, len(error_timestamps))]

        return {
            'error_count': len(error_events),
            'total_events': len(all_events),
            'error_rate': len(error_events) / len(all_events),
            'avg_error_interval': float(np.mean(error_intervals)) if error_intervals else 0,
            'error_interval_std': float(np.std(error_intervals)) if error_intervals else 0,
            'time_span_minutes': (max(error_timestamps) - min(error_timestamps)).total_seconds() / 60
        }

    async def _detect_handoff_loop_patterns(self,
                                         conversation_data: List[Dict[str, Any]],
                                         features: List[PatternFeature]) -> List[ConversationPattern]:
        """Detect handoff loop patterns"""
        patterns = []

        # Build handoff graph
        handoff_graph = nx.DiGraph()
        handoff_events = [e for e in conversation_data if e.get('event_type') == 'handoff']

        for event in handoff_events:
            from_agent = event.get('agent_id')
            to_agent = event.get('target_agent_id')
            if from_agent and to_agent:
                handoff_graph.add_edge(from_agent, to_agent)

        # Detect cycles
        cycles = list(nx.simple_cycles(handoff_graph))

        for cycle in cycles:
            if len(cycle) >= self.pattern_thresholds['handoff_loop_threshold']:
                loop_score = self._calculate_handoff_loop_score(cycle, handoff_events)

                if loop_score >= self.pattern_thresholds['min_pattern_confidence']:
                    severity = PatternSeverity.PROBLEMATIC if loop_score > 0.7 else PatternSeverity.SUBOPTIMAL

                    pattern = ConversationPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type=PatternType.HANDOFF_LOOP,
                        severity=severity,
                        confidence=loop_score,
                        description="Handoff loop pattern detected",
                        agents_involved=set(cycle),
                        time_window=(
                            min(e.get('timestamp', datetime.utcnow()) for e in handoff_events),
                            max(e.get('timestamp', datetime.utcnow()) for e in handoff_events)
                        ),
                        impact_score=loop_score * 0.7,
                        metadata={
                            'cycle_length': len(cycle),
                            'handoff_count': len([e for e in handoff_events
                                                if e.get('agent_id') in cycle and e.get('target_agent_id') in cycle]),
                            'loop_agents': list(cycle),
                            'loop_metrics': self._calculate_handoff_loop_metrics(cycle, handoff_events)
                        }
                    )
                    patterns.append(pattern)

        return patterns

    def _calculate_handoff_loop_score(self, cycle: List[str], handoff_events: List[Dict[str, Any]]) -> float:
        """Calculate score for handoff loop pattern"""
        if len(cycle) < 3:
            return 0.0

        # Count handoffs in the cycle
        cycle_handoffs = [
            e for e in handoff_events
            if (e.get('agent_id') in cycle and e.get('target_agent_id') in cycle)
        ]

        # Calculate loop frequency
        loop_score = min(len(cycle_handoffs) / len(cycle), 1.0)
        return loop_score

    def _calculate_handoff_loop_metrics(self, cycle: List[str],
                                      handoff_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate detailed handoff loop metrics"""
        cycle_handoffs = [
            e for e in handoff_events
            if (e.get('agent_id') in cycle and e.get('target_agent_id') in cycle)
        ]

        if not cycle_handoffs:
            return {}

        return {
            'cycle_length': len(cycle),
            'handoff_count': len(cycle_handoffs),
            'avg_handoffs_per_agent': len(cycle_handoffs) / len(cycle),
            'agents_in_loop': len(cycle)
        }

    async def _detect_knowledge_sharing_patterns(self,
                                               conversation_data: List[Dict[str, Any]],
                                               features: List[PatternFeature]) -> List[ConversationPattern]:
        """Detect knowledge sharing patterns"""
        patterns = []

        # Look for knowledge-related events
        knowledge_events = [
            e for e in conversation_data
            if any(keyword in str(e.get('message', '')).lower()
                  for keyword in ['share', 'teach', 'learn', 'explain', 'knowledge', 'guide'])
        ]

        if len(knowledge_events) >= 3:
            knowledge_score = self._calculate_knowledge_sharing_score(knowledge_events)

            if knowledge_score >= self.pattern_thresholds['sequential_flow_confidence']:
                pattern = ConversationPattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type=PatternType.KNOWLEDGE_SHARING,
                    severity=PatternSeverity.OPTIMAL,
                    confidence=knowledge_score,
                    description="Knowledge sharing pattern detected",
                    agents_involved=set(e.get('agent_id') for e in knowledge_events),
                    time_window=(
                        min(e.get('timestamp', datetime.utcnow()) for e in knowledge_events),
                        max(e.get('timestamp', datetime.utcnow()) for e in knowledge_events)
                    ),
                    impact_score=knowledge_score * 0.8,
                    metadata={
                        'knowledge_events': len(knowledge_events),
                        'unique_agents': len(set(e.get('agent_id') for e in knowledge_events)),
                        'sharing_metrics': self._calculate_knowledge_sharing_metrics(knowledge_events)
                    }
                )
                patterns.append(pattern)

        return patterns

    def _calculate_knowledge_sharing_score(self, knowledge_events: List[Dict[str, Any]]) -> float:
        """Calculate score for knowledge sharing pattern"""
        if len(knowledge_events) < 3:
            return 0.0

        # Calculate knowledge sharing metrics
        unique_agents = len(set(e.get('agent_id') for e in knowledge_events))
        agent_diversity = unique_agents / len(knowledge_events)

        # Calculate message complexity (simple proxy)
        avg_message_length = np.mean([len(str(e.get('message', ''))) for e in knowledge_events])
        complexity_score = min(avg_message_length / 200, 1.0)  # Normalize to 200 chars

        # Calculate temporal distribution
        timestamps = [e.get('timestamp', datetime.utcnow()) for e in knowledge_events]
        time_spread = (max(timestamps) - min(timestamps)).total_seconds()
        distribution_score = min(time_spread / 300, 1.0)  # Normalize to 5 minutes

        # Calculate knowledge sharing score
        knowledge_score = (agent_diversity * 0.4 + complexity_score * 0.3 + distribution_score * 0.3)
        return min(1.0, max(0.0, knowledge_score))

    def _calculate_knowledge_sharing_metrics(self, knowledge_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate detailed knowledge sharing metrics"""
        if not knowledge_events:
            return {}

        unique_agents = set(e.get('agent_id') for e in knowledge_events)
        message_lengths = [len(str(e.get('message', ''))) for e in knowledge_events]
        timestamps = [e.get('timestamp', datetime.utcnow()) for e in knowledge_events]

        return {
            'knowledge_events': len(knowledge_events),
            'unique_agents': len(unique_agents),
            'avg_message_length': float(np.mean(message_lengths)),
            'max_message_length': float(np.max(message_lengths)),
            'time_span_minutes': (max(timestamps) - min(timestamps)).total_seconds() / 60
        }

    async def _update_pattern_clusters(self) -> None:
        """Update pattern clusters using ML clustering"""
        if len(self.pattern_history) < 10:
            return

        try:
            # Prepare feature matrix
            pattern_features = []
            pattern_types = []

            for pattern in self.pattern_history[-100:]:  # Last 100 patterns
                feature_vector = self._pattern_to_feature_vector(pattern)
                if feature_vector:
                    pattern_features.append(feature_vector)
                    pattern_types.append(pattern.pattern_type)

            if len(pattern_features) < 5:
                return

            # Apply clustering
            feature_matrix = np.array(pattern_features)
            scaled_features = self.ml_models['dimensionality_reduction']['scaler'].fit_transform(feature_matrix)

            # K-means clustering
            kmeans = self.ml_models['clustering']['kmeans']
            cluster_labels = kmeans.fit_predict(scaled_features)

            # Update pattern clusters
            self.pattern_clusters.clear()
            for i, label in enumerate(cluster_labels):
                cluster_id = f"cluster_{label}"
                pattern = self.pattern_history[-100:][i]

                if cluster_id not in self.pattern_clusters:
                    self.pattern_clusters[cluster_id] = PatternCluster(
                        cluster_id=cluster_id,
                        pattern_type=pattern.pattern_type,
                        patterns=[],
                        centroid=np.mean([pattern_features[j] for j, l in enumerate(cluster_labels) if l == label], axis=0)
                    )

                self.pattern_clusters[cluster_id].patterns.append(pattern)

            # Calculate cluster characteristics
            for cluster in self.pattern_clusters.values():
                cluster.characteristics = self._calculate_cluster_characteristics(cluster)

        except Exception as e:
            logger.error(f"Error updating pattern clusters: {e}")

    def _pattern_to_feature_vector(self, pattern: ConversationPattern) -> Optional[np.ndarray]:
        """Convert pattern to feature vector for ML processing"""
        try:
            features = [
                pattern.confidence,
                pattern.impact_score,
                len(pattern.agents_involved),
                (pattern.time_window[1] - pattern.time_window[0]).total_seconds(),
                len(pattern.features)
            ]

            # Add pattern type encoding
            pattern_type_encoding = {
                PatternType.SEQUENTIAL_FLOW: [1, 0, 0, 0, 0],
                PatternType.PARALLEL_COLLABORATION: [0, 1, 0, 0, 0],
                PatternType.ERROR_SPIRAL: [0, 0, 1, 0, 0],
                PatternType.HANDOFF_LOOP: [0, 0, 0, 1, 0],
                PatternType.KNOWLEDGE_SHARING: [0, 0, 0, 0, 1]
            }

            features.extend(pattern_type_encoding.get(pattern.pattern_type, [0, 0, 0, 0, 0]))

            return np.array(features)

        except Exception as e:
            logger.error(f"Error converting pattern to feature vector: {e}")
            return None

    def _calculate_cluster_characteristics(self, cluster: PatternCluster) -> Dict[str, Any]:
        """Calculate characteristics for a pattern cluster"""
        if not cluster.patterns:
            return {}

        confidences = [p.confidence for p in cluster.patterns]
        impact_scores = [p.impact_score for p in cluster.patterns]
        agent_counts = [len(p.agents_involved) for p in cluster.patterns]

        return {
            'pattern_count': len(cluster.patterns),
            'avg_confidence': float(np.mean(confidences)),
            'avg_impact_score': float(np.mean(impact_scores)),
            'avg_agents': float(np.mean(agent_counts)),
            'pattern_types': list(set(p.pattern_type.value for p in cluster.patterns)),
            'severity_distribution': {
                severity.value: sum(1 for p in cluster.patterns if p.severity == severity)
                for severity in PatternSeverity
            }
        }

    async def get_pattern_insights(self, time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get insights from detected patterns"""
        if time_range is None:
            time_range = timedelta(hours=24)

        cutoff_time = datetime.utcnow() - time_range
        recent_patterns = [
            p for p in self.pattern_history
            if p.time_window[1] > cutoff_time
        ]

        if not recent_patterns:
            return {"message": "No patterns detected in the specified time range"}

        # Analyze pattern distribution
        pattern_type_counts = Counter(p.pattern_type for p in recent_patterns)
        severity_counts = Counter(p.severity for p in recent_patterns)

        # Analyze clusters
        cluster_insights = {}
        for cluster_id, cluster in self.pattern_clusters.items():
            recent_cluster_patterns = [
                p for p in cluster.patterns
                if p.time_window[1] > cutoff_time
            ]

            if recent_cluster_patterns:
                cluster_insights[cluster_id] = {
                    'recent_patterns': len(recent_cluster_patterns),
                    'avg_confidence': float(np.mean([p.confidence for p in recent_cluster_patterns])),
                    'dominant_pattern_type': cluster.pattern_type.value,
                    'characteristics': cluster.characteristics
                }

        return {
            'total_patterns': len(recent_patterns),
            'pattern_distribution': {pt.value: count for pt, count in pattern_type_counts.items()},
            'severity_distribution': {sv.value: count for sv, count in severity_counts.items()},
            'cluster_insights': cluster_insights,
            'top_patterns': sorted(
                recent_patterns,
                key=lambda x: x.impact_score,
                reverse=True
            )[:5],
            'analysis_timestamp': datetime.utcnow().isoformat()
        }

    async def predict_emerging_patterns(self) -> List[Dict[str, Any]]:
        """Predict emerging patterns based on trend analysis"""
        if len(self.pattern_history) < 20:
            return []

        try:
            # Analyze pattern frequency trends
            recent_patterns = self.pattern_history[-50:]  # Last 50 patterns
            older_patterns = self.pattern_history[-100:-50]  # Previous 50 patterns

            recent_type_counts = Counter(p.pattern_type for p in recent_patterns)
            older_type_counts = Counter(p.pattern_type for p in older_patterns)

            # Identify emerging patterns
            emerging_patterns = []
            for pattern_type in PatternType:
                recent_count = recent_type_counts.get(pattern_type, 0)
                older_count = older_type_counts.get(pattern_type, 0)

                if recent_count > older_count * 1.5:  # 50% increase
                    emerging_patterns.append({
                        'pattern_type': pattern_type.value,
                        'growth_rate': (recent_count - older_count) / max(older_count, 1),
                        'recent_frequency': recent_count,
                        'prediction': f"Increasing trend in {pattern_type.value} patterns"
                    })

            return emerging_patterns

        except Exception as e:
            logger.error(f"Error predicting emerging patterns: {e}")
            return []


class ConversationFeatureExtractor:
    """Extracts features for pattern recognition"""

    def __init__(self):
        self.text_analyzer = TextAnalyticsProcessor()

    async def extract_features(self, conversation_data: List[Dict[str, Any]]) -> List[PatternFeature]:
        """Extract features from conversation data"""
        features = []

        # Temporal features
        temporal_features = self._extract_temporal_features(conversation_data)
        features.extend(temporal_features)

        # Agent interaction features
        interaction_features = self._extract_interaction_features(conversation_data)
        features.extend(interaction_features)

        # Content features
        content_features = await self._extract_content_features(conversation_data)
        features.extend(content_features)

        # Structural features
        structural_features = self._extract_structural_features(conversation_data)
        features.extend(structural_features)

        return features

    def _extract_temporal_features(self, conversation_data: List[Dict[str, Any]]) -> List[PatternFeature]:
        """Extract temporal features"""
        features = []
        timestamps = [e.get('timestamp', datetime.utcnow()) for e in conversation_data]

        if len(timestamps) < 2:
            return features

        time_intervals = [(timestamps[i] - timestamps[i-1]).total_seconds()
                          for i in range(1, len(timestamps))]

        features.extend([
            PatternFeature(
                feature_id=str(uuid.uuid4()),
                name="avg_interval",
                value=float(np.mean(time_intervals)),
                description="Average time between events",
                category="temporal"
            ),
            PatternFeature(
                feature_id=str(uuid.uuid4()),
                name="interval_std",
                value=float(np.std(time_intervals)),
                description="Standard deviation of time intervals",
                category="temporal"
            ),
            PatternFeature(
                feature_id=str(uuid.uuid4()),
                name="total_duration",
                value=float((max(timestamps) - min(timestamps)).total_seconds()),
                description="Total duration of conversation",
                category="temporal"
            )
        ])

        return features

    def _extract_interaction_features(self, conversation_data: List[Dict[str, Any]]) -> List[PatternFeature]:
        """Extract agent interaction features"""
        features = []

        agents = [e.get('agent_id') for e in conversation_data if e.get('agent_id')]
        unique_agents = set(agents)

        features.extend([
            PatternFeature(
                feature_id=str(uuid.uuid4()),
                name="agent_count",
                value=float(len(unique_agents)),
                description="Number of unique agents",
                category="interaction"
            ),
            PatternFeature(
                feature_id=str(uuid.uuid4()),
                name="agent_diversity",
                value=float(len(unique_agents) / max(len(agents), 1)),
                description="Agent diversity ratio",
                category="interaction"
            )
        ])

        return features

    async def _extract_content_features(self, conversation_data: List[Dict[str, Any]]) -> List[PatternFeature]:
        """Extract content-based features"""
        features = []

        messages = [str(e.get('message', '')) for e in conversation_data if e.get('message')]

        if not messages:
            return features

        # Average message length
        avg_length = np.mean([len(msg) for msg in messages])

        # Sentiment analysis
        sentiments = [self.text_analyzer.analyze_sentiment(msg) for msg in messages]
        avg_sentiment = np.mean(sentiments)

        # Topic diversity
        all_topics = []
        for msg in messages:
            topics = self.text_analyzer.extract_topics(msg)
            all_topics.extend(topics)

        topic_diversity = len(set(all_topics)) / max(len(all_topics), 1)

        features.extend([
            PatternFeature(
                feature_id=str(uuid.uuid4()),
                name="avg_message_length",
                value=float(avg_length),
                description="Average message length",
                category="content"
            ),
            PatternFeature(
                feature_id=str(uuid.uuid4()),
                name="avg_sentiment",
                value=float(avg_sentiment),
                description="Average sentiment score",
                category="content"
            ),
            PatternFeature(
                feature_id=str(uuid.uuid4()),
                name="topic_diversity",
                value=float(topic_diversity),
                description="Topic diversity ratio",
                category="content"
            )
        ])

        return features

    def _extract_structural_features(self, conversation_data: List[Dict[str, Any]]) -> List[PatternFeature]:
        """Extract structural features"""
        features = []

        # Event type distribution
        event_types = [e.get('event_type') for e in conversation_data if e.get('event_type')]
        type_counts = Counter(event_types)

        if type_counts:
            # Calculate entropy of event types
            total_events = sum(type_counts.values())
            entropy = -sum((count / total_events) * np.log2(count / total_events)
                          for count in type_counts.values())

            features.append(PatternFeature(
                feature_id=str(uuid.uuid4()),
                name="event_type_entropy",
                value=float(entropy),
                description="Entropy of event type distribution",
                category="structural"
            ))

        return features


# Factory function for creating pattern recognizer
def create_conversation_pattern_recognizer(config: Optional[Dict[str, Any]] = None) -> ConversationPatternRecognizer:
    """Factory function to create conversation pattern recognizer"""
    return ConversationPatternRecognizer(config)