"""
Conversation Insights Generator
Advanced insight generation and actionable recommendations for conversation analytics
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
from concurrent.futures import ThreadPoolExecutor
import re

from .conversation_analytics_engine import ConversationAnalyticsEngine, AnalyticsInsight, InsightSeverity
from .pattern_recognition import ConversationPatternRecognizer, ConversationPattern, PatternType
from .performance_analytics import ConversationPerformanceAnalyzer, PerformanceGrade
from .conversation_predictive_analytics import ConversationPredictiveAnalyzer, ConversationRiskLevel

logger = logging.getLogger(__name__)


class InsightCategory(Enum):
    """Categories of conversation insights"""
    PERFORMANCE = "performance"
    COLLABORATION = "collaboration"
    EFFICIENCY = "efficiency"
    QUALITY = "quality"
    RISK = "risk"
    OPTIMIZATION = "optimization"
    TRAINING = "training"
    RESOURCE = "resource"
    COMMUNICATION = "communication"
    KNOWLEDGE = "knowledge"


class InsightPriority(Enum):
    """Priority levels for insights"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ActionType(Enum):
    """Types of recommended actions"""
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"
    MONITOR = "monitor"
    INVESTIGATE = "investigate"


@dataclass
class InsightEvidence:
    """Evidence supporting an insight"""
    evidence_id: str
    evidence_type: str
    value: float
    description: str
    confidence: float
    source_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionableRecommendation:
    """Actionable recommendation for improvement"""
    recommendation_id: str
    title: str
    description: str
    action_type: ActionType
    priority: InsightPriority
    estimated_impact: float
    effort_required: float
    timeframe: str
    dependencies: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    implementation_steps: List[str] = field(default_factory=list)


@dataclass
class ConversationInsight:
    """Comprehensive conversation insight with evidence and recommendations"""
    insight_id: str
    title: str
    description: str
    category: InsightCategory
    priority: InsightPriority
    severity: InsightSeverity
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    evidence: List[InsightEvidence] = field(default_factory=list)
    recommendations: List[ActionableRecommendation] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_insights: List[str] = field(default_factory=list)


@dataclass
class InsightBundle:
    """Bundle of related insights"""
    bundle_id: str
    title: str
    description: str
    category: InsightCategory
    overall_priority: InsightPriority
    estimated_impact: float
    insights: List[ConversationInsight] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


class ConversationInsightGenerator:
    """
    Advanced insight generation for conversation analytics
    """

    def __init__(self,
                 analytics_engine: ConversationAnalyticsEngine,
                 pattern_recognizer: ConversationPatternRecognizer,
                 performance_analyzer: ConversationPerformanceAnalyzer,
                 predictive_analyzer: ConversationPredictiveAnalyzer,
                 config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.analytics_engine = analytics_engine
        self.pattern_recognizer = pattern_recognizer
        self.performance_analyzer = performance_analyzer
        self.predictive_analyzer = predictive_analyzer
        self.insight_history = deque(maxlen=5000)
        self.insight_templates = self._initialize_insight_templates()
        self.action_library = self._initialize_action_library()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def _initialize_insight_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize insight templates for different scenarios"""
        return {
            'high_error_rate': {
                'title': 'High Error Rate Detected',
                'description': 'Elevated error rates indicate systemic issues requiring attention',
                'category': InsightCategory.PERFORMANCE,
                'base_priority': InsightPriority.HIGH,
                'evidence_types': ['error_rate', 'error_frequency', 'error_patterns'],
                'recommendations': [
                    {
                        'title': 'Implement Error Handling Improvements',
                        'action_type': ActionType.SHORT_TERM,
                        'effort': 0.6,
                        'impact': 0.8,
                        'steps': ['Review error logs', 'Identify common failure patterns', 'Implement retry mechanisms']
                    },
                    {
                        'title': 'Add Input Validation',
                        'action_type': ActionType.IMMEDIATE,
                        'effort': 0.4,
                        'impact': 0.6,
                        'steps': ['Audit input validation', 'Add validation rules', 'Test edge cases']
                    }
                ]
            },
            'poor_collaboration': {
                'title': 'Collaboration Efficiency Issues',
                'description': 'Suboptimal collaboration patterns are affecting team performance',
                'category': InsightCategory.COLLABORATION,
                'base_priority': InsightPriority.MEDIUM,
                'evidence_types': ['collaboration_score', 'handoff_efficiency', 'participation_rate'],
                'recommendations': [
                    {
                        'title': 'Improve Communication Protocols',
                        'action_type': ActionType.MEDIUM_TERM,
                        'effort': 0.7,
                        'impact': 0.7,
                        'steps': ['Establish communication standards', 'Create shared vocabulary', 'Implement feedback loops']
                    },
                    {
                        'title': 'Enhance Context Sharing',
                        'action_type': ActionType.SHORT_TERM,
                        'effort': 0.5,
                        'impact': 0.6,
                        'steps': ['Document context requirements', 'Implement context transfer mechanisms', 'Train agents']
                    }
                ]
            },
            'response_time_degradation': {
                'title': 'Response Time Degradation',
                'description': 'Increasing response times are impacting user experience',
                'category': InsightCategory.PERFORMANCE,
                'base_priority': InsightPriority.HIGH,
                'evidence_types': ['response_time', 'processing_time', 'queue_length'],
                'recommendations': [
                    {
                        'title': 'Optimize Processing Pipelines',
                        'action_type': ActionType.MEDIUM_TERM,
                        'effort': 0.8,
                        'impact': 0.7,
                        'steps': ['Profile bottlenecks', 'Implement caching', 'Optimize algorithms']
                    },
                    {
                        'title': 'Scale Resources',
                        'action_type': ActionType.SHORT_TERM,
                        'effort': 0.6,
                        'impact': 0.5,
                        'steps': ['Assess resource utilization', 'Add computational resources', 'Monitor performance']
                    }
                ]
            },
            'knowledge_gaps': {
                'title': 'Knowledge Gaps Identified',
                'description': 'Agents are lacking critical knowledge for effective task completion',
                'category': InsightCategory.KNOWLEDGE,
                'base_priority': InsightPriority.MEDIUM,
                'evidence_types': ['knowledge_references', 'success_rate', 'information_requests'],
                'recommendations': [
                    {
                        'title': 'Expand Knowledge Base',
                        'action_type': ActionType.LONG_TERM,
                        'effort': 0.9,
                        'impact': 0.8,
                        'steps': ['Identify knowledge gaps', 'Create documentation', 'Implement training programs']
                    },
                    {
                        'title': 'Implement Knowledge Sharing',
                        'action_type': ActionType.MEDIUM_TERM,
                        'effort': 0.6,
                        'impact': 0.6,
                        'steps': ['Create knowledge sharing mechanisms', 'Establish best practices', 'Reward knowledge sharing']
                    }
                ]
            },
            'conflict_patterns': {
                'title': 'Conflict Patterns Detected',
                'description': 'Recurring conflicts are affecting team dynamics and productivity',
                'category': InsightCategory.RISK,
                'base_priority': InsightPriority.HIGH,
                'evidence_types': ['sentiment_analysis', 'disagreement_frequency', 'resolution_time'],
                'recommendations': [
                    {
                        'title': 'Implement Conflict Resolution',
                        'action_type': ActionType.IMMEDIATE,
                        'effort': 0.7,
                        'impact': 0.8,
                        'steps': ['Establish conflict protocols', 'Train on resolution techniques', 'Monitor interactions']
                    },
                    {
                        'title': 'Improve Communication Clarity',
                        'action_type': ActionType.SHORT_TERM,
                        'effort': 0.5,
                        'impact': 0.6,
                        'steps': ['Standardize communication', 'Reduce ambiguity', 'Increase feedback frequency']
                    }
                ]
            }
        }

    def _initialize_action_library(self) -> Dict[str, Dict[str, Any]]:
        """Initialize library of common actions and their characteristics"""
        return {
            'implement_caching': {
                'title': 'Implement Caching Mechanism',
                'effort': 0.6,
                'impact': 0.7,
                'timeframe': '2-4 weeks',
                'dependencies': ['performance_analysis'],
                'success_criteria': ['50% reduction in response time', '80% cache hit rate']
            },
            'add_monitoring': {
                'title': 'Add Monitoring and Alerting',
                'effort': 0.5,
                'impact': 0.6,
                'timeframe': '1-2 weeks',
                'dependencies': [],
                'success_criteria': ['Real-time alerts', 'Comprehensive dashboards', 'Historical tracking']
            },
            'optimize_workflows': {
                'title': 'Optimize Workflow Processes',
                'effort': 0.8,
                'impact': 0.7,
                'timeframe': '4-8 weeks',
                'dependencies': ['workflow_analysis'],
                'success_criteria': ['30% reduction in processing time', 'Higher success rates']
            },
            'provide_training': {
                'title': 'Provide Agent Training',
                'effort': 0.7,
                'impact': 0.6,
                'timeframe': '2-6 weeks',
                'dependencies': ['training_materials'],
                'success_criteria': ['Improved performance metrics', 'Higher confidence scores']
            },
            'scale_resources': {
                'title': 'Scale System Resources',
                'effort': 0.6,
                'impact': 0.5,
                'timeframe': '1-3 weeks',
                'dependencies': ['capacity_analysis'],
                'success_criteria': ['Reduced latency', 'Higher throughput', 'Better resource utilization']
            }
        }

    async def generate_insights(self, conversation_data: List[Dict[str, Any]],
                               time_range: Optional[Tuple[datetime, datetime]] = None) -> List[ConversationInsight]:
        """
        Generate comprehensive insights from conversation data

        Args:
            conversation_data: Raw conversation data
            time_range: Time range for analysis

        Returns:
            List of generated insights
        """
        try:
            if time_range is None:
                time_range = (datetime.utcnow() - timedelta(days=7), datetime.utcnow())

            logger.info(f"Generating insights for {len(conversation_data)} conversation events")

            # Extract features and patterns
            features = await self.analytics_engine.feature_extractor.extract_features(conversation_data)
            patterns = await self.pattern_recognizer.detect_conversation_patterns(conversation_data)

            # Generate category-specific insights
            insights = []

            # Performance insights
            performance_insights = await self._generate_performance_insights(
                conversation_data, features, patterns, time_range
            )
            insights.extend(performance_insights)

            # Collaboration insights
            collaboration_insights = await self._generate_collaboration_insights(
                conversation_data, features, patterns, time_range
            )
            insights.extend(collaboration_insights)

            # Efficiency insights
            efficiency_insights = await self._generate_efficiency_insights(
                conversation_data, features, patterns, time_range
            )
            insights.extend(efficiency_insights)

            # Risk insights
            risk_insights = await self._generate_risk_insights(
                conversation_data, features, patterns, time_range
            )
            insights.extend(risk_insights)

            # Knowledge insights
            knowledge_insights = await self._generate_knowledge_insights(
                conversation_data, features, patterns, time_range
            )
            insights.extend(knowledge_insights)

            # Remove duplicates and similar insights
            unique_insights = await self._deduplicate_insights(insights)

            # Store insights
            self.insight_history.extend(unique_insights)

            logger.info(f"Generated {len(unique_insights)} unique insights")
            return unique_insights

        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return []

    async def _generate_performance_insights(self, conversation_data: List[Dict[str, Any]],
                                          features: List[Any],
                                          patterns: List[ConversationPattern],
                                          time_range: Tuple[datetime, datetime]) -> List[ConversationInsight]:
        """Generate performance-related insights"""
        insights = []

        try:
            # Analyze error patterns
            error_events = [e for e in conversation_data if e.get('event_type') == 'error']
            total_events = len(conversation_data)

            if total_events > 0:
                error_rate = len(error_events) / total_events

                if error_rate > 0.15:  # 15% error threshold
                    insight = await self._create_insight_from_template(
                        'high_error_rate',
                        evidence=[
                            InsightEvidence(
                                evidence_id=str(uuid.uuid4()),
                                evidence_type='error_rate',
                                value=error_rate,
                                description=f"Error rate of {error_rate:.1%} exceeds acceptable threshold",
                                confidence=0.9,
                                source_data={'error_count': len(error_events), 'total_events': total_events}
                            )
                        ],
                        context={'time_range': time_range}
                    )
                    insights.append(insight)

            # Analyze response times
            response_times = [e.get('response_time', 0) for e in conversation_data if e.get('response_time')]
            if response_times:
                avg_response_time = np.mean(response_times)
                if avg_response_time > 120:  # 2 minutes threshold
                    insight = await self._create_insight_from_template(
                        'response_time_degradation',
                        evidence=[
                            InsightEvidence(
                                evidence_id=str(uuid.uuid4()),
                                evidence_type='response_time',
                                value=avg_response_time,
                                description=f"Average response time of {avg_response_time:.1f}s is above optimal",
                                confidence=0.8,
                                source_data={'avg_response_time': avg_response_time, 'sample_count': len(response_times)}
                            )
                        ],
                        context={'time_range': time_range}
                    )
                    insights.append(insight)

            # Analyze throughput
            if len(conversation_data) > 10:
                time_span = (time_range[1] - time_range[0]).total_seconds() / 3600  # hours
                throughput = len(conversation_data) / max(time_span, 1)

                if throughput < 5:  # Low throughput threshold
                    insight = ConversationInsight(
                        insight_id=str(uuid.uuid4()),
                        title="Low Throughput Detected",
                        description=f"System throughput of {throughput:.1f} events/hour is below expected levels",
                        category=InsightCategory.PERFORMANCE,
                        priority=InsightPriority.MEDIUM,
                        severity=InsightSeverity.MEDIUM,
                        confidence=0.7,
                        evidence=[
                            InsightEvidence(
                                evidence_id=str(uuid.uuid4()),
                                evidence_type='throughput',
                                value=throughput,
                                description=f"Low throughput indicates processing bottlenecks",
                                confidence=0.7,
                                source_data={'throughput': throughput, 'time_span_hours': time_span}
                            )
                        ],
                        recommendations=[
                            ActionableRecommendation(
                                recommendation_id=str(uuid.uuid4()),
                                title="Optimize Processing Pipelines",
                                description="Review and optimize agent processing workflows",
                                action_type=ActionType.MEDIUM_TERM,
                                priority=InsightPriority.MEDIUM,
                                estimated_impact=0.6,
                                effort_required=0.7,
                                timeframe="4-6 weeks",
                                implementation_steps=["Profile performance", "Identify bottlenecks", "Implement optimizations"]
                            )
                        ],
                        context={'time_range': time_range}
                    )
                    insights.append(insight)

        except Exception as e:
            logger.error(f"Error generating performance insights: {e}")

        return insights

    async def _generate_collaboration_insights(self, conversation_data: List[Dict[str, Any]],
                                            features: List[Any],
                                            patterns: List[ConversationPattern],
                                            time_range: Tuple[datetime, datetime]) -> List[ConversationInsight]:
        """Generate collaboration-related insights"""
        insights = []

        try:
            # Analyze collaboration patterns
            collab_patterns = [p for p in patterns if p.pattern_type == PatternType.PARALLEL_COLLABORATION]

            if collab_patterns:
                avg_collab_score = np.mean([p.confidence for p in collab_patterns])

                if avg_collab_score < 0.6:
                    insight = await self._create_insight_from_template(
                        'poor_collaboration',
                        evidence=[
                            InsightEvidence(
                                evidence_id=str(uuid.uuid4()),
                                evidence_type='collaboration_score',
                                value=avg_collab_score,
                                description=f"Average collaboration score of {avg_collab_score:.2f} indicates room for improvement",
                                confidence=0.8,
                                source_data={'avg_collaboration_score': avg_collab_score, 'pattern_count': len(collab_patterns)}
                            )
                        ],
                        context={'time_range': time_range}
                    )
                    insights.append(insight)

            # Analyze handoff efficiency
            handoff_events = [e for e in conversation_data if e.get('event_type') == 'handoff']
            if handoff_events:
                failed_handoffs = [e for e in handoff_events if e.get('success') == False]
                handoff_success_rate = 1 - (len(failed_handoffs) / len(handoff_events))

                if handoff_success_rate < 0.8:
                    insight = ConversationInsight(
                        insight_id=str(uuid.uuid4()),
                        title="Handoff Efficiency Issues",
                        description=f"Handoff success rate of {handoff_success_rate:.1%} is below target",
                        category=InsightCategory.COLLABORATION,
                        priority=InsightPriority.HIGH,
                        severity=InsightSeverity.MEDIUM,
                        confidence=0.8,
                        evidence=[
                            InsightEvidence(
                                evidence_id=str(uuid.uuid4()),
                                evidence_type='handoff_success_rate',
                                value=handoff_success_rate,
                                description=f"Low handoff success rate indicates coordination issues",
                                confidence=0.8,
                                source_data={'success_rate': handoff_success_rate, 'total_handoffs': len(handoff_events)}
                            )
                        ],
                        recommendations=[
                            ActionableRecommendation(
                                recommendation_id=str(uuid.uuid4()),
                                title="Improve Handoff Protocols",
                                description="Standardize and improve handoff processes between agents",
                                action_type=ActionType.SHORT_TERM,
                                priority=InsightPriority.HIGH,
                                estimated_impact=0.7,
                                effort_required=0.6,
                                timeframe="2-4 weeks",
                                implementation_steps=["Document handoff procedures", "Train agents", "Monitor effectiveness"]
                            )
                        ],
                        context={'time_range': time_range}
                    )
                    insights.append(insight)

            # Analyze participation patterns
            agent_participation = Counter(e.get('agent_id') for e in conversation_data if e.get('agent_id'))
            if agent_participation:
                min_participation = min(agent_participation.values())
                max_participation = max(agent_participation.values())
                participation_ratio = min_participation / max_participation if max_participation > 0 else 1.0

                if participation_ratio < 0.3:  # Highly uneven participation
                    insight = ConversationInsight(
                        insight_id=str(uuid.uuid4()),
                        title="Uneven Agent Participation",
                        description=f"Significant variation in agent participation detected (ratio: {participation_ratio:.2f})",
                        category=InsightCategory.COLLABORATION,
                        priority=InsightPriority.MEDIUM,
                        severity=InsightSeverity.LOW,
                        confidence=0.7,
                        evidence=[
                            InsightEvidence(
                                evidence_id=str(uuid.uuid4()),
                                evidence_type='participation_ratio',
                                value=participation_ratio,
                                description=f"Uneven participation may indicate workload imbalance",
                                confidence=0.7,
                                source_data={'participation_ratio': participation_ratio, 'agent_count': len(agent_participation)}
                            )
                        ],
                        recommendations=[
                            ActionableRecommendation(
                                recommendation_id=str(uuid.uuid4()),
                                title="Balance Workload Distribution",
                                description="Redistribute tasks to ensure more even agent participation",
                                action_type=ActionType.MEDIUM_TERM,
                                priority=InsightPriority.MEDIUM,
                                estimated_impact=0.5,
                                effort_required=0.6,
                                timeframe="3-5 weeks",
                                implementation_steps=["Analyze workload", "Redistribute tasks", "Monitor balance"]
                            )
                        ],
                        context={'time_range': time_range}
                    )
                    insights.append(insight)

        except Exception as e:
            logger.error(f"Error generating collaboration insights: {e}")

        return insights

    async def _generate_efficiency_insights(self, conversation_data: List[Dict[str, Any]],
                                          features: List[Any],
                                          patterns: List[ConversationPattern],
                                          time_range: Tuple[datetime, datetime]) -> List[ConversationInsight]:
        """Generate efficiency-related insights"""
        insights = []

        try:
            # Analyze task completion efficiency
            task_events = [e for e in conversation_data if e.get('event_type') in ['task_assigned', 'task_completed']]
            if len(task_events) >= 2:
                completed_tasks = [e for e in task_events if e.get('event_type') == 'task_completed']
                task_completion_rate = len(completed_tasks) / (len(task_events) / 2)  # Assuming paired events

                if task_completion_rate < 0.7:
                    insight = ConversationInsight(
                        insight_id=str(uuid.uuid4()),
                        title="Low Task Completion Rate",
                        description=f"Task completion rate of {task_completion_rate:.1%} indicates efficiency issues",
                        category=InsightCategory.EFFICIENCY,
                        priority=InsightPriority.HIGH,
                        severity=InsightSeverity.MEDIUM,
                        confidence=0.8,
                        evidence=[
                            InsightEvidence(
                                evidence_id=str(uuid.uuid4()),
                                evidence_type='task_completion_rate',
                                value=task_completion_rate,
                                description=f"Low completion rate suggests process inefficiencies",
                                confidence=0.8,
                                source_data={'completion_rate': task_completion_rate, 'total_tasks': len(task_events) // 2}
                            )
                        ],
                        recommendations=[
                            ActionableRecommendation(
                                recommendation_id=str(uuid.uuid4()),
                                title="Streamline Task Processes",
                                description="Optimize task assignment and completion workflows",
                                action_type=ActionType.MEDIUM_TERM,
                                priority=InsightPriority.HIGH,
                                estimated_impact=0.7,
                                effort_required=0.7,
                                timeframe="4-6 weeks",
                                implementation_steps=["Map current processes", "Identify bottlenecks", "Implement improvements"]
                            )
                        ],
                        context={'time_range': time_range}
                    )
                    insights.append(insight)

            # Analyze resource utilization
            if features:
                # Extract resource-related features
                resource_features = [f for f in features if hasattr(f, 'category') and f.category == 'resource']
                if resource_features:
                    avg_utilization = np.mean([f.value for f in resource_features])

                    if avg_utilization > 0.85:  # High utilization
                        insight = ConversationInsight(
                            insight_id=str(uuid.uuid4()),
                            title="High Resource Utilization",
                            description=f"Average resource utilization of {avg_utilization:.1%} approaches capacity limits",
                            category=InsightCategory.RESOURCE,
                            priority=InsightPriority.MEDIUM,
                            severity=InsightSeverity.MEDIUM,
                            confidence=0.7,
                            evidence=[
                                InsightEvidence(
                                    evidence_id=str(uuid.uuid4()),
                                    evidence_type='resource_utilization',
                                    value=avg_utilization,
                                    description=f"High utilization may lead to performance degradation",
                                    confidence=0.7,
                                    source_data={'avg_utilization': avg_utilization, 'feature_count': len(resource_features)}
                                )
                            ],
                            recommendations=[
                                ActionableRecommendation(
                                    recommendation_id=str(uuid.uuid4()),
                                    title="Scale Resources or Optimize Usage",
                                    description="Address high resource utilization to prevent bottlenecks",
                                    action_type=ActionType.SHORT_TERM,
                                    priority=InsightPriority.MEDIUM,
                                    estimated_impact=0.6,
                                    effort_required=0.5,
                                    timeframe="2-3 weeks",
                                    implementation_steps=["Assess resource needs", "Optimize usage", "Scale if necessary"]
                                )
                            ],
                            context={'time_range': time_range}
                        )
                        insights.append(insight)

        except Exception as e:
            logger.error(f"Error generating efficiency insights: {e}")

        return insights

    async def _generate_risk_insights(self, conversation_data: List[Dict[str, Any]],
                                    features: List[Any],
                                    patterns: List[ConversationPattern],
                                    time_range: Tuple[datetime, datetime]) -> List[ConversationInsight]:
        """Generate risk-related insights"""
        insights = []

        try:
            # Analyze conflict patterns
            conflict_patterns = [p for p in patterns if p.pattern_type == PatternType.ERROR_SPIRAL]

            if conflict_patterns:
                high_risk_patterns = [p for p in conflict_patterns if p.severity.value in ['high', 'critical']]

                if high_risk_patterns:
                    insight = await self._create_insight_from_template(
                        'conflict_patterns',
                        evidence=[
                            InsightEvidence(
                                evidence_id=str(uuid.uuid4()),
                                evidence_type='conflict_pattern_count',
                                value=len(high_risk_patterns),
                                description=f"Detected {len(high_risk_patterns)} high-risk conflict patterns",
                                confidence=0.8,
                                source_data={'high_risk_patterns': len(high_risk_patterns), 'total_patterns': len(patterns)}
                            )
                        ],
                        context={'time_range': time_range}
                    )
                    insights.append(insight)

            # Analyze sentiment trends
            sentiment_scores = [e.get('sentiment_score', 0) for e in conversation_data if e.get('sentiment_score')]
            if sentiment_scores:
                avg_sentiment = np.mean(sentiment_scores)
                sentiment_trend = np.polyfit(range(len(sentiment_scores)), sentiment_scores, 1)[0]

                if avg_sentiment < -0.3 or sentiment_trend < -0.01:  # Negative sentiment or declining trend
                    insight = ConversationInsight(
                        insight_id=str(uuid.uuid4()),
                        title="Negative Sentiment Detected",
                        description=f"Average sentiment of {avg_sentiment:.2f} with {'declining' if sentiment_trend < 0 else 'stable'} trend",
                        category=InsightCategory.RISK,
                        priority=InsightPriority.MEDIUM,
                        severity=InsightSeverity.MEDIUM,
                        confidence=0.7,
                        evidence=[
                            InsightEvidence(
                                evidence_id=str(uuid.uuid4()),
                                evidence_type='sentiment_analysis',
                                value=avg_sentiment,
                                description=f"Negative sentiment may indicate underlying issues",
                                confidence=0.7,
                                source_data={'avg_sentiment': avg_sentiment, 'trend': sentiment_trend, 'sample_count': len(sentiment_scores)}
                            )
                        ],
                        recommendations=[
                            ActionableRecommendation(
                                recommendation_id=str(uuid.uuid4()),
                                title="Address Sentiment Issues",
                                description="Investigate and address causes of negative sentiment",
                                action_type=ActionType.SHORT_TERM,
                                priority=InsightPriority.MEDIUM,
                                estimated_impact=0.6,
                                effort_required=0.5,
                                timeframe="2-3 weeks",
                                implementation_steps=["Identify root causes", "Implement improvements", "Monitor sentiment"]
                            )
                        ],
                        context={'time_range': time_range}
                    )
                    insights.append(insight)

        except Exception as e:
            logger.error(f"Error generating risk insights: {e}")

        return insights

    async def _generate_knowledge_insights(self, conversation_data: List[Dict[str, Any]],
                                        features: List[Any],
                                        patterns: List[ConversationPattern],
                                        time_range: Tuple[datetime, datetime]) -> List[ConversationInsight]:
        """Generate knowledge-related insights"""
        insights = []

        try:
            # Analyze knowledge sharing patterns
            knowledge_patterns = [p for p in patterns if p.pattern_type == PatternType.KNOWLEDGE_SHARING]

            if len(knowledge_patterns) < 2:  # Low knowledge sharing
                insight = await self._create_insight_from_template(
                    'knowledge_gaps',
                    evidence=[
                        InsightEvidence(
                            evidence_id=str(uuid.uuid4()),
                            evidence_type='knowledge_sharing_frequency',
                            value=len(knowledge_patterns),
                            description=f"Low knowledge sharing activity detected ({len(knowledge_patterns)} instances)",
                            confidence=0.7,
                            source_data={'knowledge_patterns': len(knowledge_patterns), 'total_patterns': len(patterns)}
                        )
                    ],
                    context={'time_range': time_range}
                )
                insights.append(insight)

            # Analyze information requests
            info_requests = [e for e in conversation_data if 'information_request' in str(e.get('message', '')).lower()]
            total_messages = len([e for e in conversation_data if e.get('message')])

            if total_messages > 0:
                info_request_rate = len(info_requests) / total_messages

                if info_request_rate > 0.2:  # High information request rate
                    insight = ConversationInsight(
                        insight_id=str(uuid.uuid4()),
                        title="High Information Request Rate",
                        description=f"Information request rate of {info_request_rate:.1%} suggests knowledge gaps",
                        category=InsightCategory.KNOWLEDGE,
                        priority=InsightPriority.MEDIUM,
                        severity=InsightSeverity.LOW,
                        confidence=0.7,
                        evidence=[
                            InsightEvidence(
                                evidence_id=str(uuid.uuid4()),
                                evidence_type='information_request_rate',
                                value=info_request_rate,
                                description=f"High request rate indicates insufficient knowledge accessibility",
                                confidence=0.7,
                                source_data={'request_rate': info_request_rate, 'total_messages': total_messages}
                            )
                        ],
                        recommendations=[
                            ActionableRecommendation(
                                recommendation_id=str(uuid.uuid4()),
                                title="Improve Knowledge Accessibility",
                                description="Enhance knowledge base and information retrieval systems",
                                action_type=ActionType.MEDIUM_TERM,
                                priority=InsightPriority.MEDIUM,
                                estimated_impact=0.6,
                                effort_required=0.7,
                                timeframe="3-5 weeks",
                                implementation_steps=["Audit knowledge base", "Improve search functionality", "Train agents"]
                            )
                        ],
                        context={'time_range': time_range}
                    )
                    insights.append(insight)

        except Exception as e:
            logger.error(f"Error generating knowledge insights: {e}")

        return insights

    async def _create_insight_from_template(self, template_name: str,
                                         evidence: List[InsightEvidence],
                                         context: Dict[str, Any]) -> ConversationInsight:
        """Create insight from predefined template"""
        template = self.insight_templates.get(template_name, {})

        # Generate recommendations from template
        recommendations = []
        for rec_data in template.get('recommendations', []):
            recommendation = ActionableRecommendation(
                recommendation_id=str(uuid.uuid4()),
                title=rec_data['title'],
                description=rec_data.get('description', ''),
                action_type=rec_data['action_type'],
                priority=template['base_priority'],
                estimated_impact=rec_data['impact'],
                effort_required=rec_data['effort'],
                timeframe=self._estimate_timeframe(rec_data['effort']),
                implementation_steps=rec_data['steps']
            )
            recommendations.append(recommendation)

        return ConversationInsight(
            insight_id=str(uuid.uuid4()),
            title=template['title'],
            description=template['description'],
            category=template['category'],
            priority=template['base_priority'],
            severity=InsightSeverity.MEDIUM,  # Default severity
            confidence=0.8,  # Default confidence
            evidence=evidence,
            recommendations=recommendations,
            context=context
        )

    def _estimate_timeframe(self, effort: float) -> str:
        """Estimate timeframe based on effort required"""
        if effort < 0.3:
            return "1-2 weeks"
        elif effort < 0.5:
            return "2-4 weeks"
        elif effort < 0.7:
            return "4-8 weeks"
        else:
            return "8+ weeks"

    async def _deduplicate_insights(self, insights: List[ConversationInsight]) -> List[ConversationInsight]:
        """Remove duplicate or very similar insights"""
        unique_insights = []
        insight_titles = set()

        for insight in insights:
            # Simple deduplication based on title similarity
            title_normalized = re.sub(r'[^\w\s]', '', insight.title.lower()).strip()

            if title_normalized not in insight_titles:
                unique_insights.append(insight)
                insight_titles.add(title_normalized)

        return unique_insights

    async def generate_insight_bundles(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> List[InsightBundle]:
        """Generate bundles of related insights"""
        if time_range is None:
            time_range = (datetime.utcnow() - timedelta(days=7), datetime.utcnow())

        # Get insights for time range
        range_insights = [
            insight for insight in self.insight_history
            if time_range[0] <= insight.timestamp <= time_range[1]
        ]

        if not range_insights:
            return []

        # Group insights by category
        category_groups = defaultdict(list)
        for insight in range_insights:
            category_groups[insight.category].append(insight)

        # Create bundles for each category with multiple insights
        bundles = []
        for category, insights in category_groups.items():
            if len(insights) >= 2:  # Only create bundles with 2+ insights
                bundle = InsightBundle(
                    bundle_id=str(uuid.uuid4()),
                    title=f"{category.value.title()} Insights Bundle",
                    description=f"Comprehensive insights related to {category.value}",
                    category=category,
                    insights=insights,
                    overall_priority=max(insight.priority for insight in insights),
                    estimated_impact=np.mean([rec.estimated_impact for insight in insights for rec in insight.recommendations])
                )
                bundles.append(bundle)

        return bundles

    async def get_insight_summary(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Generate summary of insights and recommendations"""
        if time_range is None:
            time_range = (datetime.utcnow() - timedelta(days=7), datetime.utcnow())

        try:
            # Get insights for time range
            range_insights = [
                insight for insight in self.insight_history
                if time_range[0] <= insight.timestamp <= time_range[1]
            ]

            if not range_insights:
                return {"message": "No insights found in the specified time range"}

            # Analyze by category
            category_analysis = {}
            for category in InsightCategory:
                category_insights = [i for i in range_insights if i.category == category]
                if category_insights:
                    category_analysis[category.value] = {
                        'count': len(category_insights),
                        'avg_confidence': np.mean([i.confidence for i in category_insights]),
                        'priority_distribution': {
                            priority.value: len([i for i in category_insights if i.priority == priority])
                            for priority in InsightPriority
                        }
                    }

            # Analyze recommendations
            all_recommendations = [rec for insight in range_insights for rec in insight.recommendations]
            if all_recommendations:
                rec_analysis = {
                    'total_recommendations': len(all_recommendations),
                    'avg_estimated_impact': np.mean([rec.estimated_impact for rec in all_recommendations]),
                    'avg_effort_required': np.mean([rec.effort_required for rec in all_recommendations]),
                    'action_type_distribution': {
                        action_type.value: len([rec for rec in all_recommendations if rec.action_type == action_type])
                        for action_type in ActionType
                    },
                    'top_recommendations': sorted(
                        all_recommendations,
                        key=lambda x: x.estimated_impact * x.priority.value == 'critical',
                        reverse=True
                    )[:5]
                }
            else:
                rec_analysis = {'total_recommendations': 0}

            return {
                'summary_period': {
                    'start': time_range[0].isoformat(),
                    'end': time_range[1].isoformat()
                },
                'total_insights': len(range_insights),
                'category_analysis': category_analysis,
                'recommendation_analysis': rec_analysis,
                'high_priority_insights': len([i for i in range_insights if i.priority in [InsightPriority.CRITICAL, InsightPriority.HIGH]]),
                'estimated_total_impact': sum(rec.estimated_impact for rec in all_recommendations),
                'generated_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating insight summary: {e}")
            return {"error": str(e)}

    async def track_insight_impact(self, insight_ids: List[str]) -> Dict[str, Any]:
        """Track the impact of implemented insights"""
        try:
            impact_tracking = {}

            for insight_id in insight_ids:
                # Find the insight
                insight = next((i for i in self.insight_history if i.insight_id == insight_id), None)
                if not insight:
                    continue

                # Calculate impact metrics (simplified)
                time_since_creation = (datetime.utcnow() - insight.timestamp).total_seconds() / 3600  # hours

                # Estimate impact based on recommendations and time
                estimated_impact = 0.0
                for rec in insight.recommendations:
                    # Decay impact over time if not implemented
                    time_factor = max(0, 1 - (time_since_creation / (24 * 30)))  # 30-day decay
                    estimated_impact += rec.estimated_impact * time_factor * 0.1  # 10% of estimated impact per hour

                impact_tracking[insight_id] = {
                    'insight_title': insight.title,
                    'category': insight.category.value,
                    'created_at': insight.timestamp.isoformat(),
                    'hours_since_creation': time_since_creation,
                    'estimated_potential_impact': estimated_impact,
                    'recommendations_count': len(insight.recommendations),
                    'implementation_status': 'not_tracked'  # Would be tracked in real system
                }

            return {
                'tracked_insights': len(impact_tracking),
                'total_estimated_impact': sum(t['estimated_potential_impact'] for t in impact_tracking.values()),
                'impact_details': impact_tracking,
                'generated_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error tracking insight impact: {e}")
            return {"error": str(e)}


# Factory function for creating insight generator
def create_conversation_insight_generator(
    analytics_engine: ConversationAnalyticsEngine,
    pattern_recognizer: ConversationPatternRecognizer,
    performance_analyzer: ConversationPerformanceAnalyzer,
    predictive_analyzer: ConversationPredictiveAnalyzer,
    config: Optional[Dict[str, Any]] = None
) -> ConversationInsightGenerator:
    """Factory function to create conversation insight generator"""
    return ConversationInsightGenerator(
        analytics_engine, pattern_recognizer, performance_analyzer, predictive_analyzer, config
    )