"""
Conversation Analytics Engine
Core engine for comprehensive conversation analytics in Agency Swarm systems
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
from collections import defaultdict, deque
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ConversationEventType(Enum):
    """Types of conversation events"""
    MESSAGE = "message"
    HANDOFF = "handoff"
    COLLABORATION = "collaboration"
    ERROR = "error"
    SUCCESS = "success"
    TASK_ASSIGNED = "task_assigned"
    TASK_COMPLETED = "task_completed"
    INSIGHT_GENERATED = "insight_generated"
    PATTERN_DETECTED = "pattern_detected"


class AnalyticsMetricType(Enum):
    """Types of analytics metrics"""
    CONVERSATION_FREQUENCY = "conversation_frequency"
    RESPONSE_TIME = "response_time"
    COLLABORATION_SCORE = "collaboration_score"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    HANDOFF_EFFICIENCY = "handoff_efficiency"
    AGENT_PERFORMANCE = "agent_performance"
    SENTIMENT_SCORE = "sentiment_score"
    TOPIC_COVERAGE = "topic_coverage"
    KNOWLEDGE_INTEGRATION = "knowledge_integration"


class InsightSeverity(Enum):
    """Severity levels for analytics insights"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConversationEvent:
    """Represents a conversation event"""
    event_id: str
    event_type: ConversationEventType
    timestamp: datetime
    agent_id: str
    session_id: str
    message: Optional[str] = None
    target_agent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class ConversationThread:
    """Represents a conversation thread between agents"""
    thread_id: str
    session_id: str
    participants: Set[str] = field(default_factory=set)
    events: List[ConversationEvent] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    topic: Optional[str] = None
    sentiment_score: float = 0.0
    collaboration_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsMetric:
    """Represents an analytics metric"""
    metric_id: str
    metric_type: AnalyticsMetricType
    value: float
    timestamp: datetime
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    dimensions: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class AnalyticsInsight:
    """Represents an analytics insight"""
    insight_id: str
    title: str
    description: str
    severity: InsightSeverity
    category: str
    metrics: List[AnalyticsMetric] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationAnalyticsEngine:
    """
    Core engine for processing and analyzing conversation data
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.event_buffer: deque = deque(maxlen=10000)
        self.active_threads: Dict[str, ConversationThread] = {}
        self.metrics_cache: Dict[str, List[AnalyticsMetric]] = defaultdict(list)
        self.insights_cache: List[AnalyticsInsight] = []
        self.pattern_recognition_enabled = self.config.get('pattern_recognition_enabled', True)
        self.real_time_processing = self.config.get('real_time_processing', True)
        self.max_conversation_age = timedelta(hours=self.config.get('max_conversation_age_hours', 24))
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Initialize analytics processors
        self._initialize_processors()

    def _initialize_processors(self):
        """Initialize analytics processors"""
        self.text_analyzer = TextAnalyticsProcessor()
        self.pattern_detector = ConversationPatternDetector()
        self.performance_analyzer = ConversationPerformanceAnalyzer()
        self.insight_generator = ConversationInsightGenerator()

    async def process_conversation_event(self, event: ConversationEvent) -> None:
        """
        Process a conversation event in real-time

        Args:
            event: Conversation event to process
        """
        try:
            # Add to buffer
            self.event_buffer.append(event)

            # Update or create conversation thread
            await self._update_conversation_thread(event)

            # Extract metrics in real-time
            if self.real_time_processing:
                metrics = await self._extract_real_time_metrics(event)
                for metric in metrics:
                    self._cache_metric(metric)

            # Trigger pattern detection
            if self.pattern_recognition_enabled:
                asyncio.create_task(self._detect_patterns_async(event))

            logger.debug(f"Processed conversation event: {event.event_id}")

        except Exception as e:
            logger.error(f"Error processing conversation event {event.event_id}: {e}")

    async def _update_conversation_thread(self, event: ConversationEvent) -> None:
        """Update or create conversation thread for event"""
        thread_key = f"{event.session_id}_{event.agent_id}"

        if thread_key not in self.active_threads:
            thread = ConversationThread(
                thread_id=thread_key,
                session_id=event.session_id,
                participants={event.agent_id},
                start_time=event.timestamp
            )
            self.active_threads[thread_key] = thread
        else:
            thread = self.active_threads[thread_key]

        # Add event to thread
        thread.events.append(event)

        # Update thread metadata
        thread.participants.add(event.agent_id)
        if event.target_agent_id:
            thread.participants.add(event.target_agent_id)

        # Update thread end time
        thread.end_time = event.timestamp

        # Analyze thread sentiment and collaboration
        if event.message:
            thread.sentiment_score = self.text_analyzer.analyze_sentiment(event.message)
            thread.collaboration_score = self._calculate_collaboration_score(thread)

    def _calculate_collaboration_score(self, thread: ConversationThread) -> float:
        """Calculate collaboration score for a conversation thread"""
        if not thread.events:
            return 0.0

        # Factors for collaboration score
        participant_count = len(thread.participants)
        event_count = len(thread.events)
        message_events = sum(1 for e in thread.events if e.event_type == ConversationEventType.MESSAGE)
        handoff_events = sum(1 for e in thread.events if e.event_type == ConversationEventType.HANDOFF)

        # Score calculation (0.0 to 1.0)
        participation_score = min(participant_count / 5.0, 1.0)  # Normalize to max 5 participants
        activity_score = min(event_count / 10.0, 1.0)  # Normalize to 10 events
        interaction_score = min(message_events / max(event_count, 1), 1.0)
        handoff_efficiency = 1.0 - min(handoff_events / max(event_count, 1), 1.0)

        # Weighted average
        collaboration_score = (
            participation_score * 0.3 +
            activity_score * 0.2 +
            interaction_score * 0.3 +
            handoff_efficiency * 0.2
        )

        return max(0.0, min(1.0, collaboration_score))

    async def _extract_real_time_metrics(self, event: ConversationEvent) -> List[AnalyticsMetric]:
        """Extract real-time metrics from conversation event"""
        metrics = []

        # Response time metric
        if event.event_type == ConversationEventType.MESSAGE:
            response_time = self._calculate_response_time(event)
            if response_time is not None:
                metrics.append(AnalyticsMetric(
                    metric_id=str(uuid.uuid4()),
                    metric_type=AnalyticsMetricType.RESPONSE_TIME,
                    value=response_time,
                    timestamp=event.timestamp,
                    agent_id=event.agent_id,
                    session_id=event.session_id
                ))

        # Conversation frequency metric
        metrics.append(AnalyticsMetric(
            metric_id=str(uuid.uuid4()),
            metric_type=AnalyticsMetricType.CONVERSATION_FREQUENCY,
            value=1.0,
            timestamp=event.timestamp,
            agent_id=event.agent_id,
            session_id=event.session_id
        ))

        # Error rate metric
        if event.event_type == ConversationEventType.ERROR:
            metrics.append(AnalyticsMetric(
                metric_id=str(uuid.uuid4()),
                metric_type=AnalyticsMetricType.ERROR_RATE,
                value=1.0,
                timestamp=event.timestamp,
                agent_id=event.agent_id,
                session_id=event.session_id
            ))

        return metrics

    def _calculate_response_time(self, event: ConversationEvent) -> Optional[float]:
        """Calculate response time for a message event"""
        if not event.target_agent_id:
            return None

        # Find previous message from target agent
        thread_key = f"{event.session_id}_{event.target_agent_id}"
        if thread_key not in self.active_threads:
            return None

        thread = self.active_threads[thread_key]
        for prev_event in reversed(thread.events):
            if (prev_event.agent_id == event.target_agent_id and
                prev_event.event_type == ConversationEventType.MESSAGE and
                prev_event.timestamp < event.timestamp):
                return (event.timestamp - prev_event.timestamp).total_seconds()

        return None

    def _cache_metric(self, metric: AnalyticsMetric) -> None:
        """Cache analytics metric"""
        metric_key = f"{metric.metric_type.value}_{metric.agent_id or 'global'}"
        self.metrics_cache[metric_key].append(metric)

        # Keep only recent metrics (last 24 hours)
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.metrics_cache[metric_key] = [
            m for m in self.metrics_cache[metric_key]
            if m.timestamp > cutoff_time
        ]

    async def _detect_patterns_async(self, event: ConversationEvent) -> None:
        """Detect patterns in conversation asynchronously"""
        try:
            patterns = await self.pattern_detector.detect_patterns(event, self.active_threads)

            # Create pattern detected events
            for pattern in patterns:
                pattern_event = ConversationEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=ConversationEventType.PATTERN_DETECTED,
                    timestamp=datetime.utcnow(),
                    agent_id="analytics_engine",
                    session_id=event.session_id,
                    metadata={"pattern": pattern}
                )
                self.event_buffer.append(pattern_event)

        except Exception as e:
            logger.error(f"Error in pattern detection: {e}")

    async def generate_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive analytics for a session

        Args:
            session_id: Session ID to analyze

        Returns:
            Dictionary containing session analytics
        """
        try:
            # Get session threads
            session_threads = [
                thread for thread in self.active_threads.values()
                if thread.session_id == session_id
            ]

            if not session_threads:
                return {"error": "No conversation data found for session"}

            # Calculate session metrics
            session_metrics = await self._calculate_session_metrics(session_threads)

            # Generate insights
            insights = await self.insight_generator.generate_insights(
                session_threads, session_metrics
            )

            # Pattern analysis
            patterns = await self.pattern_detector.analyze_session_patterns(session_threads)

            return {
                "session_id": session_id,
                "metrics": session_metrics,
                "insights": insights,
                "patterns": patterns,
                "thread_count": len(session_threads),
                "total_events": sum(len(thread.events) for thread in session_threads),
                "participants": list(set().union(*[thread.participants for thread in session_threads])),
                "generated_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating session analytics for {session_id}: {e}")
            return {"error": str(e)}

    async def _calculate_session_metrics(self, threads: List[ConversationThread]) -> Dict[str, Any]:
        """Calculate comprehensive session metrics"""
        if not threads:
            return {}

        # Basic metrics
        total_events = sum(len(thread.events) for thread in threads)
        total_participants = len(set().union(*[thread.participants for thread in threads]))
        avg_thread_length = np.mean([len(thread.events) for thread in threads])

        # Time-based metrics
        session_duration = max(
            (thread.end_time or datetime.utcnow()) - (thread.start_time or datetime.utcnow())
            for thread in threads
        ).total_seconds()

        # Sentiment and collaboration
        avg_sentiment = np.mean([thread.sentiment_score for thread in threads])
        avg_collaboration = np.mean([thread.collaboration_score for thread in threads])

        # Event type distribution
        event_types = defaultdict(int)
        for thread in threads:
            for event in thread.events:
                event_types[event.event_type.value] += 1

        return {
            "total_events": total_events,
            "total_participants": total_participants,
            "avg_thread_length": float(avg_thread_length),
            "session_duration_seconds": session_duration,
            "avg_sentiment_score": float(avg_sentiment),
            "avg_collaboration_score": float(avg_collaboration),
            "event_type_distribution": dict(event_types),
            "thread_count": len(threads)
        }

    async def get_agent_performance_analytics(self, agent_id: str,
                                           time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get performance analytics for a specific agent

        Args:
            agent_id: Agent ID to analyze
            time_range: Time range for analysis (default: 24 hours)

        Returns:
            Dictionary containing agent performance analytics
        """
        if time_range is None:
            time_range = timedelta(hours=24)

        cutoff_time = datetime.utcnow() - time_range

        # Get agent metrics
        agent_metrics = []
        for metric_list in self.metrics_cache.values():
            for metric in metric_list:
                if (metric.agent_id == agent_id and
                    metric.timestamp > cutoff_time):
                    agent_metrics.append(metric)

        # Calculate performance metrics
        performance_data = await self.performance_analyzer.analyze_agent_performance(
            agent_id, agent_metrics, self.active_threads
        )

        return performance_data

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics dashboard data"""
        return {
            "active_sessions": len(set(thread.session_id for thread in self.active_threads.values())),
            "active_threads": len(self.active_threads),
            "total_events_today": len([e for e in self.event_buffer if e.timestamp.date() == datetime.utcnow().date()]),
            "buffer_size": len(self.event_buffer),
            "metrics_cache_size": sum(len(metrics) for metrics in self.metrics_cache.values()),
            "insights_count": len(self.insights_cache),
            "timestamp": datetime.utcnow().isoformat()
        }

    async def cleanup_old_data(self) -> None:
        """Clean up old conversation data"""
        cutoff_time = datetime.utcnow() - self.max_conversation_age

        # Clean old threads
        old_threads = [
            thread_id for thread_id, thread in self.active_threads.items()
            if (thread.end_time or datetime.utcnow()) < cutoff_time
        ]

        for thread_id in old_threads:
            del self.active_threads[thread_id]

        # Clean old events from buffer
        while self.event_buffer and self.event_buffer[0].timestamp < cutoff_time:
            self.event_buffer.popleft()

        # Clean old metrics
        for metric_key in list(self.metrics_cache.keys()):
            self.metrics_cache[metric_key] = [
                m for m in self.metrics_cache[metric_key]
                if m.timestamp > cutoff_time
            ]

        logger.info(f"Cleaned up {len(old_threads)} old threads and old data")


class TextAnalyticsProcessor:
    """Processor for text analytics in conversations"""

    def __init__(self):
        self.sentiment_keywords = {
            'positive': ['great', 'excellent', 'perfect', 'amazing', 'wonderful', 'fantastic', 'love', 'helpful'],
            'negative': ['error', 'failed', 'wrong', 'bad', 'terrible', 'awful', 'hate', 'useless']
        }

    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text (-1.0 to 1.0)"""
        if not text:
            return 0.0

        text_lower = text.lower()
        positive_count = sum(1 for keyword in self.sentiment_keywords['positive'] if keyword in text_lower)
        negative_count = sum(1 for keyword in self.sentiment_keywords['negative'] if keyword in text_lower)

        total_words = len(text.split())
        if total_words == 0:
            return 0.0

        sentiment = (positive_count - negative_count) / max(total_words / 10, 1)
        return max(-1.0, min(1.0, sentiment))

    def extract_topics(self, text: str) -> List[str]:
        """Extract topics from text"""
        # Simple keyword-based topic extraction
        topics = []
        text_lower = text.lower()

        topic_keywords = {
            'development': ['code', 'function', 'class', 'method', 'implementation'],
            'bug_fixing': ['bug', 'error', 'fix', 'issue', 'problem'],
            'design': ['design', 'ui', 'ux', 'interface', 'layout'],
            'testing': ['test', 'spec', 'assert', 'mock', 'verify'],
            'deployment': ['deploy', 'release', 'production', 'server'],
            'performance': ['performance', 'speed', 'optimization', 'memory']
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)

        return topics


class ConversationPatternDetector:
    """Detects patterns in conversation data"""

    def __init__(self):
        self.pattern_thresholds = {
            'rapid_fire': timedelta(seconds=5),
            'long_conversation': timedelta(minutes=30),
            'high_error_rate': 0.3,
            'frequent_handoffs': 5
        }

    async def detect_patterns(self, event: ConversationEvent,
                            threads: Dict[str, ConversationThread]) -> List[Dict[str, Any]]:
        """Detect patterns in conversation data"""
        patterns = []

        # Check for rapid fire messages
        rapid_fire_pattern = self._detect_rapid_fire_pattern(event, threads)
        if rapid_fire_pattern:
            patterns.append(rapid_fire_pattern)

        # Check for error patterns
        error_pattern = self._detect_error_pattern(event, threads)
        if error_pattern:
            patterns.append(error_pattern)

        # Check for handoff patterns
        handoff_pattern = self._detect_handoff_pattern(event, threads)
        if handoff_pattern:
            patterns.append(handoff_pattern)

        return patterns

    def _detect_rapid_fire_pattern(self, event: ConversationEvent,
                                  threads: Dict[str, ConversationThread]) -> Optional[Dict[str, Any]]:
        """Detect rapid fire message pattern"""
        if event.event_type != ConversationEventType.MESSAGE:
            return None

        thread_key = f"{event.session_id}_{event.agent_id}"
        if thread_key not in threads:
            return None

        thread = threads[thread_key]
        recent_events = [
            e for e in thread.events[-5:]  # Last 5 events
            if e.timestamp > event.timestamp - self.pattern_thresholds['rapid_fire']
        ]

        if len(recent_events) >= 3:  # 3+ events in rapid succession
            return {
                "pattern_type": "rapid_fire",
                "description": "Rapid sequence of messages detected",
                "severity": "medium",
                "event_count": len(recent_events),
                "time_window": self.pattern_thresholds['rapid_fire'].total_seconds()
            }

        return None

    def _detect_error_pattern(self, event: ConversationEvent,
                             threads: Dict[str, ConversationThread]) -> Optional[Dict[str, Any]]:
        """Detect error concentration pattern"""
        if event.event_type != ConversationEventType.ERROR:
            return None

        thread_key = f"{event.session_id}_{event.agent_id}"
        if thread_key not in threads:
            return None

        thread = threads[thread_key]
        recent_errors = [
            e for e in thread.events[-10:]  # Last 10 events
            if e.event_type == ConversationEventType.ERROR
        ]

        error_rate = len(recent_errors) / min(len(thread.events[-10:]), 10)
        if error_rate > self.pattern_thresholds['high_error_rate']:
            return {
                "pattern_type": "high_error_rate",
                "description": "High concentration of errors detected",
                "severity": "high",
                "error_rate": error_rate,
                "recent_errors": len(recent_errors)
            }

        return None

    def _detect_handoff_pattern(self, event: ConversationEvent,
                               threads: Dict[str, ConversationThread]) -> Optional[Dict[str, Any]]:
        """Detect frequent handoff pattern"""
        if event.event_type != ConversationEventType.HANDOFF:
            return None

        thread_key = f"{event.session_id}_{event.agent_id}"
        if thread_key not in threads:
            return None

        thread = threads[thread_key]
        recent_handoffs = [
            e for e in thread.events[-20:]  # Last 20 events
            if e.event_type == ConversationEventType.HANDOFF
        ]

        if len(recent_handoffs) >= self.pattern_thresholds['frequent_handoffs']:
            return {
                "pattern_type": "frequent_handoffs",
                "description": "Frequent agent handoffs detected",
                "severity": "medium",
                "handoff_count": len(recent_handoffs)
            }

        return None

    async def analyze_session_patterns(self, threads: List[ConversationThread]) -> Dict[str, Any]:
        """Analyze patterns across an entire session"""
        patterns = {
            "long_conversations": 0,
            "high_error_sessions": 0,
            "collaborative_sessions": 0,
            "session_patterns": []
        }

        for thread in threads:
            if thread.end_time and thread.start_time:
                duration = thread.end_time - thread.start_time
                if duration > self.pattern_thresholds['long_conversation']:
                    patterns["long_conversations"] += 1

            # Check error rate
            error_events = sum(1 for e in thread.events if e.event_type == ConversationEventType.ERROR)
            if len(thread.events) > 0:
                error_rate = error_events / len(thread.events)
                if error_rate > self.pattern_thresholds['high_error_rate']:
                    patterns["high_error_sessions"] += 1

            # Check collaboration
            if thread.collaboration_score > 0.7:
                patterns["collaborative_sessions"] += 1

        return patterns


class ConversationPerformanceAnalyzer:
    """Analyzes performance metrics for conversations"""

    async def analyze_agent_performance(self, agent_id: str,
                                     metrics: List[AnalyticsMetric],
                                     threads: Dict[str, ConversationThread]) -> Dict[str, Any]:
        """Analyze performance for a specific agent"""

        # Calculate response times
        response_times = [
            m.value for m in metrics
            if m.metric_type == AnalyticsMetricType.RESPONSE_TIME
        ]

        avg_response_time = np.mean(response_times) if response_times else 0

        # Calculate error rate
        error_events = [
            m for m in metrics
            if m.metric_type == AnalyticsMetricType.ERROR_RATE
        ]
        total_events = len([m for m in metrics if m.metric_type == AnalyticsMetricType.CONVERSATION_FREQUENCY])
        error_rate = len(error_events) / max(total_events, 1)

        # Get agent threads
        agent_threads = [
            thread for thread in threads.values()
            if agent_id in thread.participants
        ]

        # Calculate collaboration metrics
        collaboration_scores = [thread.collaboration_score for thread in agent_threads]
        avg_collaboration = np.mean(collaboration_scores) if collaboration_scores else 0

        return {
            "agent_id": agent_id,
            "avg_response_time": float(avg_response_time),
            "error_rate": float(error_rate),
            "avg_collaboration_score": float(avg_collaboration),
            "total_events": total_events,
            "threads_participated": len(agent_threads),
            "performance_score": self._calculate_performance_score(
                avg_response_time, error_rate, avg_collaboration
            )
        }

    def _calculate_performance_score(self, response_time: float,
                                   error_rate: float,
                                   collaboration_score: float) -> float:
        """Calculate overall performance score (0.0 to 1.0)"""
        # Normalize metrics
        response_score = max(0, 1 - (response_time / 300))  # 5 minutes max
        error_score = max(0, 1 - error_rate)
        collaboration_score = collaboration_score

        # Weighted average
        performance_score = (
            response_score * 0.4 +
            error_score * 0.4 +
            collaboration_score * 0.2
        )

        return max(0.0, min(1.0, performance_score))


class ConversationInsightGenerator:
    """Generates actionable insights from conversation analytics"""

    async def generate_insights(self, threads: List[ConversationThread],
                              metrics: Dict[str, Any]) -> List[AnalyticsInsight]:
        """Generate insights from conversation data"""
        insights = []

        # Performance insights
        performance_insights = self._generate_performance_insights(threads, metrics)
        insights.extend(performance_insights)

        # Collaboration insights
        collaboration_insights = self._generate_collaboration_insights(threads, metrics)
        insights.extend(collaboration_insights)

        # Pattern insights
        pattern_insights = self._generate_pattern_insights(threads, metrics)
        insights.extend(pattern_insights)

        return insights

    def _generate_performance_insights(self, threads: List[ConversationThread],
                                      metrics: Dict[str, Any]) -> List[AnalyticsInsight]:
        """Generate performance-related insights"""
        insights = []

        # High error rate insight
        if metrics.get('error_rate', 0) > 0.2:
            insight = AnalyticsInsight(
                insight_id=str(uuid.uuid4()),
                title="High Error Rate Detected",
                description=f"Error rate of {metrics['error_rate']:.2%} exceeds acceptable threshold",
                severity=InsightSeverity.HIGH,
                category="performance",
                recommendations=[
                    "Review error handling in agent implementations",
                    "Implement better input validation",
                    "Add retry mechanisms for failed operations"
                ]
            )
            insights.append(insight)

        # Slow response time insight
        if metrics.get('avg_response_time', 0) > 120:  # 2 minutes
            insight = AnalyticsInsight(
                insight_id=str(uuid.uuid4()),
                title="Slow Response Times",
                description=f"Average response time of {metrics['avg_response_time']:.1f}s is above optimal",
                severity=InsightSeverity.MEDIUM,
                category="performance",
                recommendations=[
                    "Optimize agent processing pipelines",
                    "Implement async processing where possible",
                    "Review and optimize computational complexity"
                ]
            )
            insights.append(insight)

        return insights

    def _generate_collaboration_insights(self, threads: List[ConversationThread],
                                        metrics: Dict[str, Any]) -> List[AnalyticsInsight]:
        """Generate collaboration-related insights"""
        insights = []

        # Low collaboration score
        if metrics.get('avg_collaboration_score', 1.0) < 0.5:
            insight = AnalyticsInsight(
                insight_id=str(uuid.uuid4()),
                title="Low Collaboration Effectiveness",
                description=f"Average collaboration score of {metrics['avg_collaboration_score']:.2f} indicates room for improvement",
                severity=InsightSeverity.MEDIUM,
                category="collaboration",
                recommendations=[
                    "Improve handoff mechanisms between agents",
                    "Enhance shared context management",
                    "Implement better communication protocols"
                ]
            )
            insights.append(insight)

        # High participant count
        if metrics.get('total_participants', 0) > 8:
            insight = AnalyticsInsight(
                insight_id=str(uuid.uuid4()),
                title="High Participant Count",
                description=f"Session has {metrics['total_participants']} participants, which may reduce efficiency",
                severity=InsightSeverity.LOW,
                category="collaboration",
                recommendations=[
                    "Consider breaking into smaller working groups",
                    "Implement role-based specialization",
                    "Optimize communication channels"
                ]
            )
            insights.append(insight)

        return insights

    def _generate_pattern_insights(self, threads: List[ConversationThread],
                                   metrics: Dict[str, Any]) -> List[AnalyticsInsight]:
        """Generate pattern-related insights"""
        insights = []

        # Long conversation patterns
        if metrics.get('long_conversations', 0) > 0:
            insight = AnalyticsInsight(
                insight_id=str(uuid.uuid4()),
                title="Extended Conversation Sessions",
                description=f"Detected {metrics['long_conversations']} conversations exceeding 30 minutes",
                severity=InsightSeverity.LOW,
                category="patterns",
                recommendations=[
                    "Analyze long conversations for optimization opportunities",
                    "Consider breaking complex tasks into smaller chunks",
                    "Implement progress checkpoints in long-running tasks"
                ]
            )
            insights.append(insight)

        return insights


# Factory function for creating analytics engine
def create_conversation_analytics_engine(config: Optional[Dict[str, Any]] = None) -> ConversationAnalyticsEngine:
    """Factory function to create conversation analytics engine"""
    return ConversationAnalyticsEngine(config)