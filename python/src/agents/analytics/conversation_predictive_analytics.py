"""
Conversation Predictive Analytics
Enhanced predictive analytics specifically for conversation patterns and agent interactions
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
from sklearn.ensemble import RandomForestRegressor, IsolationForest, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report
from sklearn.cluster import KMeans, DBSCAN
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from .predictive_analytics import PredictiveAnalytics, ModelConfig, AlgorithmType, ModelType

logger = logging.getLogger(__name__)


class ConversationPredictionType(Enum):
    """Types of conversation-specific predictions"""
    COLLABORATION_EFFICIENCY = "collaboration_efficiency"
    HANDOFF_SUCCESS = "handoff_success"
    CONVERSATION_FLOW = "conversation_flow"
    AGENT_COMPATIBILITY = "agent_compatibility"
    KNOWLEDGE_TRANSFER = "knowledge_transfer"
    CONFLICT_PREDICTION = "conflict_prediction"
    TASK_DURATION = "task_duration"
    QUALITY_SCORE = "quality_score"
    ENGAGEMENT_PREDICTION = "engagement_prediction"
    RESOURCE_OPTIMIZATION = "resource_optimization"


class ConversationRiskLevel(Enum):
    """Risk levels for conversation predictions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConversationFeatures:
    """Features extracted from conversation data"""
    agent_id: str
    session_id: str
    timestamp: datetime
    response_time: float
    message_count: int
    error_count: int
    collaboration_score: float
    handoff_count: int
    sentiment_score: float
    topic_diversity: float
    participant_count: int
    knowledge_references: int
    context_switches: int
    success_rate: float
    engagement_level: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationPrediction:
    """Prediction result for conversation analytics"""
    prediction_id: str
    prediction_type: ConversationPredictionType
    predicted_value: float
    confidence: float
    risk_level: ConversationRiskLevel
    timestamp: datetime = field(default_factory=datetime.utcnow)
    features_used: List[str] = field(default_factory=list)
    contributing_factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCompatibilityScore:
    """Compatibility score between agents"""
    agent_a: str
    agent_b: str
    compatibility_score: float
    collaboration_efficiency: float
    communication_quality: float
    knowledge_synergy: float
    conflict_risk: float
    recommendations: List[str] = field(default_factory=list)


class ConversationPredictiveAnalyzer:
    """
    Advanced predictive analytics for conversation patterns and agent interactions
    """

    def __init__(self, base_analytics: Optional[PredictiveAnalytics] = None, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.base_analytics = base_analytics
        self.conversation_models = self._initialize_conversation_models()
        self.scalers = self._initialize_scalers()
        self.feature_cache = {}
        self.prediction_history = deque(maxlen=10000)
        self.compatibility_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=4)

    def _initialize_conversation_models(self) -> Dict[str, Any]:
        """Initialize conversation-specific ML models"""
        return {
            'collaboration_efficiency': {
                'model': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
                'features': ['response_time', 'message_count', 'sentiment_score', 'participant_count', 'knowledge_references'],
                'target_range': (0.0, 1.0)
            },
            'handoff_success': {
                'model': LogisticRegression(random_state=42),
                'features': ['context_switches', 'collaboration_score', 'knowledge_transfer', 'previous_handoffs'],
                'target_range': (0.0, 1.0)
            },
            'conversation_flow': {
                'model': RandomForestRegressor(n_estimators=50, random_state=42),
                'features': ['message_frequency', 'response_time_consistency', 'topic_coherence', 'engagement_level'],
                'target_range': (0.0, 1.0)
            },
            'task_duration': {
                'model': GradientBoostingRegressor(n_estimators=80, random_state=42),
                'features': ['complexity_score', 'participant_count', 'collaboration_score', 'knowledge_references'],
                'target_range': (0, float('inf'))
            },
            'quality_score': {
                'model': RandomForestRegressor(n_estimators=75, random_state=42),
                'features': ['success_rate', 'error_count', 'collaboration_score', 'sentiment_score'],
                'target_range': (0.0, 1.0)
            },
            'conflict_prediction': {
                'model': IsolationForest(contamination=0.1, random_state=42),
                'features': ['sentiment_variance', 'response_time_spikes', 'interruption_frequency', 'disagreement_indicators'],
                'target_range': (0.0, 1.0)
            }
        }

    def _initialize_scalers(self) -> Dict[str, Any]:
        """Initialize feature scalers"""
        return {
            'standard_scaler': StandardScaler(),
            'minmax_scaler': MinMaxScaler(),
            'robust_scaler': None  # Could add RobustScaler if needed
        }

    async def extract_conversation_features(self, conversation_data: List[Dict[str, Any]]) -> List[ConversationFeatures]:
        """Extract features from raw conversation data"""
        features_list = []

        try:
            # Group by agent-session pairs
            for event in conversation_data:
                features = ConversationFeatures(
                    agent_id=event.get('agent_id', 'unknown'),
                    session_id=event.get('session_id', 'unknown'),
                    timestamp=event.get('timestamp', datetime.utcnow()),
                    response_time=float(event.get('response_time', 0)),
                    message_count=int(event.get('message_count', 1)),
                    error_count=int(event.get('error_count', 0)),
                    collaboration_score=float(event.get('collaboration_score', 0.5)),
                    handoff_count=int(event.get('handoff_count', 0)),
                    sentiment_score=float(event.get('sentiment_score', 0.0)),
                    topic_diversity=float(event.get('topic_diversity', 0.0)),
                    participant_count=int(event.get('participant_count', 1)),
                    knowledge_references=int(event.get('knowledge_references', 0)),
                    context_switches=int(event.get('context_switches', 0)),
                    success_rate=float(event.get('success_rate', 1.0)),
                    engagement_level=float(event.get('engagement_level', 0.5)),
                    metadata=event.get('metadata', {})
                )

                # Calculate derived features
                features = await self._calculate_derived_features(features)
                features_list.append(features)

        except Exception as e:
            logger.error(f"Error extracting conversation features: {e}")

        return features_list

    async def _calculate_derived_features(self, features: ConversationFeatures) -> ConversationFeatures:
        """Calculate derived features"""
        try:
            # Message frequency (messages per minute)
            time_window_minutes = 5  # 5-minute window
            features.metadata['message_frequency'] = features.message_count / max(time_window_minutes, 1)

            # Response time consistency (coefficient of variation)
            if features.response_time > 0:
                features.metadata['response_time_consistency'] = 1.0  # Would calculate from historical data
            else:
                features.metadata['response_time_consistency'] = 0.5

            # Error rate
            features.metadata['error_rate'] = features.error_count / max(features.message_count, 1)

            # Collaboration efficiency
            if features.participant_count > 1:
                features.metadata['collaboration_efficiency'] = features.collaboration_score / features.participant_count
            else:
                features.metadata['collaboration_efficiency'] = features.collaboration_score

            # Knowledge transfer efficiency
            if features.knowledge_references > 0:
                features.metadata['knowledge_transfer'] = features.knowledge_references / features.message_count
            else:
                features.metadata['knowledge_transfer'] = 0.0

        except Exception as e:
            logger.error(f"Error calculating derived features: {e}")

        return features

    async def predict_collaboration_efficiency(self, features: ConversationFeatures) -> ConversationPrediction:
        """Predict collaboration efficiency for a conversation"""
        try:
            model_name = 'collaboration_efficiency'
            model_config = self.conversation_models[model_name]

            # Extract features
            feature_values = self._extract_feature_values(model_config['features'], features)

            # Scale features
            feature_values_scaled = self.scalers['standard_scaler'].transform([feature_values])

            # Make prediction
            model = model_config['model']
            predicted_value = model.predict(feature_values_scaled)[0]

            # Normalize to target range
            min_val, max_val = model_config['target_range']
            predicted_value = np.clip(predicted_value, min_val, max_val)

            # Calculate confidence
            confidence = self._calculate_prediction_confidence(model_name, feature_values_scaled)

            # Determine risk level
            risk_level = self._determine_risk_level(predicted_value, model_config['target_range'])

            # Generate recommendations
            recommendations = self._generate_collaboration_recommendations(predicted_value, features)

            # Create prediction
            prediction = ConversationPrediction(
                prediction_id=str(uuid.uuid4()),
                prediction_type=ConversationPredictionType.COLLABORATION_EFFICIENCY,
                predicted_value=float(predicted_value),
                confidence=confidence,
                risk_level=risk_level,
                features_used=model_config['features'],
                contributing_factors=self._identify_contributing_factors(feature_values, model_name),
                recommendations=recommendations,
                context={
                    'agent_id': features.agent_id,
                    'session_id': features.session_id,
                    'timestamp': features.timestamp.isoformat()
                }
            )

            self.prediction_history.append(prediction)
            return prediction

        except Exception as e:
            logger.error(f"Error predicting collaboration efficiency: {e}")
            return self._create_error_prediction(ConversationPredictionType.COLLABORATION_EFFICIENCY, features)

    async def predict_handoff_success(self, features: ConversationFeatures) -> ConversationPrediction:
        """Predict handoff success probability"""
        try:
            model_name = 'handoff_success'
            model_config = self.conversation_models[model_name]

            # Extract features
            feature_values = self._extract_feature_values(model_config['features'], features)

            # Make prediction
            model = model_config['model']
            predicted_proba = model.predict_proba([feature_values])[0]

            # Get success probability (class 1)
            predicted_value = predicted_proba[1] if len(predicted_proba) > 1 else 0.5

            # Calculate confidence
            confidence = max(predicted_proba) * 0.8  # Adjust for prediction confidence

            # Determine risk level
            risk_level = self._determine_risk_level(predicted_value, (0.0, 1.0))

            # Generate recommendations
            recommendations = self._generate_handoff_recommendations(predicted_value, features)

            # Create prediction
            prediction = ConversationPrediction(
                prediction_id=str(uuid.uuid4()),
                prediction_type=ConversationPredictionType.HANDOFF_SUCCESS,
                predicted_value=float(predicted_value),
                confidence=confidence,
                risk_level=risk_level,
                features_used=model_config['features'],
                contributing_factors=self._identify_contributing_factors(feature_values, model_name),
                recommendations=recommendations,
                context={
                    'agent_id': features.agent_id,
                    'session_id': features.session_id,
                    'timestamp': features.timestamp.isoformat()
                }
            )

            self.prediction_history.append(prediction)
            return prediction

        except Exception as e:
            logger.error(f"Error predicting handoff success: {e}")
            return self._create_error_prediction(ConversationPredictionType.HANDOFF_SUCCESS, features)

    async def predict_task_duration(self, features: ConversationFeatures, task_complexity: float = 0.5) -> ConversationPrediction:
        """Predict task completion duration"""
        try:
            model_name = 'task_duration'
            model_config = self.conversation_models[model_name]

            # Add complexity to features
            enhanced_features = features.metadata.copy()
            enhanced_features['complexity_score'] = task_complexity

            # Extract features
            feature_values = self._extract_feature_values(model_config['features'], enhanced_features)

            # Scale features
            feature_values_scaled = self.scalers['standard_scaler'].transform([feature_values])

            # Make prediction
            model = model_config['model']
            predicted_value = model.predict(feature_values_scaled)[0]

            # Ensure positive duration
            predicted_value = max(0, predicted_value)

            # Calculate confidence
            confidence = self._calculate_prediction_confidence(model_name, feature_values_scaled)

            # Determine risk level (longer duration = higher risk)
            if predicted_value > 3600:  # > 1 hour
                risk_level = ConversationRiskLevel.HIGH
            elif predicted_value > 1800:  # > 30 minutes
                risk_level = ConversationRiskLevel.MEDIUM
            else:
                risk_level = ConversationRiskLevel.LOW

            # Generate recommendations
            recommendations = self._generate_duration_recommendations(predicted_value, features)

            # Create prediction
            prediction = ConversationPrediction(
                prediction_id=str(uuid.uuid4()),
                prediction_type=ConversationPredictionType.TASK_DURATION,
                predicted_value=float(predicted_value),
                confidence=confidence,
                risk_level=risk_level,
                features_used=model_config['features'],
                contributing_factors=self._identify_contributing_factors(feature_values, model_name),
                recommendations=recommendations,
                context={
                    'agent_id': features.agent_id,
                    'session_id': features.session_id,
                    'task_complexity': task_complexity,
                    'timestamp': features.timestamp.isoformat()
                }
            )

            self.prediction_history.append(prediction)
            return prediction

        except Exception as e:
            logger.error(f"Error predicting task duration: {e}")
            return self._create_error_prediction(ConversationPredictionType.TASK_DURATION, features)

    async def predict_conflict_risk(self, features: ConversationFeatures) -> ConversationPrediction:
        """Predict conflict risk in conversations"""
        try:
            model_name = 'conflict_prediction'
            model_config = self.conversation_models[model_name]

            # Extract features
            feature_values = self._extract_feature_values(model_config['features'], features)

            # Make prediction
            model = model_config['model']
            prediction = model.predict([feature_values])[0]

            # For IsolationForest, -1 indicates anomaly (conflict)
            conflict_probability = 1.0 if prediction == -1 else 0.0

            # Calculate confidence
            confidence = self._calculate_prediction_confidence(model_name, [feature_values])

            # Determine risk level
            risk_level = ConversationRiskLevel.CRITICAL if conflict_probability > 0.8 else \
                        ConversationRiskLevel.HIGH if conflict_probability > 0.6 else \
                        ConversationRiskLevel.MEDIUM if conflict_probability > 0.3 else \
                        ConversationRiskLevel.LOW

            # Generate recommendations
            recommendations = self._generate_conflict_recommendations(conflict_probability, features)

            # Create prediction
            prediction = ConversationPrediction(
                prediction_id=str(uuid.uuid4()),
                prediction_type=ConversationPredictionType.CONFLICT_PREDICTION,
                predicted_value=conflict_probability,
                confidence=confidence,
                risk_level=risk_level,
                features_used=model_config['features'],
                contributing_factors=self._identify_contributing_factors(feature_values, model_name),
                recommendations=recommendations,
                context={
                    'agent_id': features.agent_id,
                    'session_id': features.session_id,
                    'timestamp': features.timestamp.isoformat()
                }
            )

            self.prediction_history.append(prediction)
            return prediction

        except Exception as e:
            logger.error(f"Error predicting conflict risk: {e}")
            return self._create_error_prediction(ConversationPredictionType.CONFLICT_PREDICTION, features)

    async def calculate_agent_compatibility(self, agent_a: str, agent_b: str,
                                         historical_data: Optional[List[Dict[str, Any]]] = None) -> AgentCompatibilityScore:
        """Calculate compatibility score between two agents"""
        try:
            # Check cache first
            cache_key = f"{min(agent_a, agent_b)}_{max(agent_a, agent_b)}"
            if cache_key in self.compatibility_cache:
                return self.compatibility_cache[cache_key]

            # Analyze historical interactions
            if historical_data:
                compatibility_data = self._analyze_historical_interactions(agent_a, agent_b, historical_data)
            else:
                # Use default/estimated compatibility
                compatibility_data = self._estimate_compatibility(agent_a, agent_b)

            # Calculate compatibility metrics
            collaboration_efficiency = self._calculate_collaboration_efficiency(compatibility_data)
            communication_quality = self._calculate_communication_quality(compatibility_data)
            knowledge_synergy = self._calculate_knowledge_synergy(compatibility_data)
            conflict_risk = self._calculate_conflict_risk(compatibility_data)

            # Overall compatibility score
            compatibility_score = (
                collaboration_efficiency * 0.4 +
                communication_quality * 0.3 +
                knowledge_synergy * 0.2 +
                (1 - conflict_risk) * 0.1
            )

            # Generate recommendations
            recommendations = self._generate_compatibility_recommendations(
                compatibility_score, collaboration_efficiency, communication_quality, conflict_risk
            )

            # Create compatibility score
            compatibility = AgentCompatibilityScore(
                agent_a=agent_a,
                agent_b=agent_b,
                compatibility_score=np.clip(compatibility_score, 0.0, 1.0),
                collaboration_efficiency=collaboration_efficiency,
                communication_quality=communication_quality,
                knowledge_synergy=knowledge_synergy,
                conflict_risk=conflict_risk,
                recommendations=recommendations
            )

            # Cache result
            self.compatibility_cache[cache_key] = compatibility

            return compatibility

        except Exception as e:
            logger.error(f"Error calculating agent compatibility: {e}")
            return AgentCompatibilityScore(
                agent_a=agent_a,
                agent_b=agent_b,
                compatibility_score=0.5,
                collaboration_efficiency=0.5,
                communication_quality=0.5,
                knowledge_synergy=0.5,
                conflict_risk=0.5,
                recommendations=["Unable to calculate compatibility due to insufficient data"]
            )

    def _analyze_historical_interactions(self, agent_a: str, agent_b: str,
                                      historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze historical interactions between agents"""
        interactions = [
            event for event in historical_data
            if ((event.get('agent_id') == agent_a and event.get('target_agent_id') == agent_b) or
                (event.get('agent_id') == agent_b and event.get('target_agent_id') == agent_a))
        ]

        return {
            'interaction_count': len(interactions),
            'avg_response_time': np.mean([e.get('response_time', 0) for e in interactions]) if interactions else 0,
            'success_rate': np.mean([e.get('success', 1) for e in interactions]) if interactions else 1.0,
            'collaboration_score': np.mean([e.get('collaboration_score', 0.5) for e in interactions]) if interactions else 0.5,
            'sentiment_score': np.mean([e.get('sentiment_score', 0.0) for e in interactions]) if interactions else 0.0
        }

    def _estimate_compatibility(self, agent_a: str, agent_b: str) -> Dict[str, Any]:
        """Estimate compatibility without historical data"""
        # Simple heuristic based on agent characteristics
        # In practice, this would use agent profiles, capabilities, etc.
        return {
            'interaction_count': 0,
            'avg_response_time': 120.0,  # Default 2 minutes
            'success_rate': 0.8,       # Default 80%
            'collaboration_score': 0.6, # Default 60%
            'sentiment_score': 0.0      # Neutral sentiment
        }

    def _calculate_collaboration_efficiency(self, data: Dict[str, Any]) -> float:
        """Calculate collaboration efficiency metric"""
        if data['interaction_count'] == 0:
            return 0.5  # Default neutral score

        # Combine multiple factors
        response_time_score = max(0, 1 - (data['avg_response_time'] / 300))  # Normalize to 5 minutes
        success_rate_score = data['success_rate']
        collaboration_score = data['collaboration_score']

        return (response_time_score * 0.3 + success_rate_score * 0.4 + collaboration_score * 0.3)

    def _calculate_communication_quality(self, data: Dict[str, Any]) -> float:
        """Calculate communication quality metric"""
        if data['interaction_count'] == 0:
            return 0.5

        # Based on sentiment and success rate
        sentiment_score = (data['sentiment_score'] + 1) / 2  # Convert from [-1,1] to [0,1]
        success_factor = data['success_rate']

        return (sentiment_score * 0.6 + success_factor * 0.4)

    def _calculate_knowledge_synergy(self, data: Dict[str, Any]) -> float:
        """Calculate knowledge synergy metric"""
        # This would require domain-specific knowledge about agent expertise
        # For now, use interaction frequency as a proxy
        interaction_factor = min(data['interaction_count'] / 10, 1.0)  # Normalize to 10 interactions
        return interaction_factor * 0.7 + 0.3  # Base 30% + interaction factor

    def _calculate_conflict_risk(self, data: Dict[str, Any]) -> float:
        """Calculate conflict risk metric"""
        if data['interaction_count'] == 0:
            return 0.3  # Low default risk

        # Risk based on negative sentiment and low success rate
        sentiment_risk = max(0, -data['sentiment_score'])  # Negative sentiment increases risk
        failure_risk = 1 - data['success_rate']

        return (sentiment_risk * 0.6 + failure_risk * 0.4)

    def _extract_feature_values(self, feature_names: List[str], data_source: Union[ConversationFeatures, Dict[str, Any]]) -> List[float]:
        """Extract feature values from data source"""
        feature_values = []

        for feature_name in feature_names:
            if isinstance(data_source, ConversationFeatures):
                # Extract from ConversationFeatures
                if hasattr(data_source, feature_name):
                    value = getattr(data_source, feature_name)
                elif feature_name in data_source.metadata:
                    value = data_source.metadata[feature_name]
                else:
                    value = 0.0  # Default value
            else:
                # Extract from dictionary
                value = data_source.get(feature_name, 0.0)

            feature_values.append(float(value))

        return feature_values

    def _calculate_prediction_confidence(self, model_name: str, features_scaled: np.ndarray) -> float:
        """Calculate prediction confidence score"""
        try:
            # Use model-specific confidence calculation
            model = self.conversation_models[model_name]['model']

            if hasattr(model, 'predict_proba'):
                # For classification models
                proba = model.predict_proba(features_scaled)
                confidence = np.max(proba[0]) if len(proba) > 0 else 0.5
            else:
                # For regression models, use feature quality and model performance
                feature_quality = self._assess_feature_quality(features_scaled[0])
                confidence = feature_quality * 0.8  # Base confidence on feature quality

            return min(1.0, max(0.0, confidence))

        except Exception as e:
            logger.error(f"Error calculating prediction confidence: {e}")
            return 0.5

    def _assess_feature_quality(self, features: List[float]) -> float:
        """Assess quality of input features"""
        try:
            # Check for missing or extreme values
            missing_count = sum(1 for f in features if f == 0.0)
            extreme_count = sum(1 for f in features if abs(f) > 3)  # More than 3 standard deviations

            quality_score = 1.0 - (missing_count * 0.1 + extreme_count * 0.05)
            return max(0.0, min(1.0, quality_score))

        except Exception:
            return 0.5

    def _determine_risk_level(self, value: float, value_range: Tuple[float, float]) -> ConversationRiskLevel:
        """Determine risk level based on predicted value"""
        min_val, max_val = value_range
        normalized_value = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5

        if normalized_value < 0.3:
            return ConversationRiskLevel.LOW
        elif normalized_value < 0.6:
            return ConversationRiskLevel.MEDIUM
        elif normalized_value < 0.8:
            return ConversationRiskLevel.HIGH
        else:
            return ConversationRiskLevel.CRITICAL

    def _identify_contributing_factors(self, feature_values: List[float], model_name: str) -> List[str]:
        """Identify factors contributing to the prediction"""
        contributing_factors = []

        try:
            model = self.conversation_models[model_name]['model']
            feature_names = self.conversation_models[model_name]['features']

            if hasattr(model, 'feature_importances_'):
                # Get feature importance
                importance_scores = model.feature_importances_

                # Identify top contributing features
                feature_importance_pairs = list(zip(feature_names, importance_scores, feature_values))
                feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

                # Generate descriptions for top factors
                for i, (feature, importance, value) in enumerate(feature_importance_pairs[:3]):
                    if importance > 0.1:  # Only include significant factors
                        contributing_factors.append(f"{feature.replace('_', ' ').title()}: {value:.2f}")

        except Exception as e:
            logger.error(f"Error identifying contributing factors: {e}")

        return contributing_factors

    def _generate_collaboration_recommendations(self, efficiency_score: float, features: ConversationFeatures) -> List[str]:
        """Generate recommendations for improving collaboration"""
        recommendations = []

        if efficiency_score < 0.6:
            recommendations.append("Consider reviewing communication protocols between agents")
            recommendations.append("Implement better context sharing mechanisms")

        if features.sentiment_score < -0.3:
            recommendations.append("Address negative sentiment in conversations")

        if features.response_time > 300:  # > 5 minutes
            recommendations.append("Optimize response times through better resource allocation")

        if features.participant_count > 5:
            recommendations.append("Consider breaking large groups into smaller teams")

        return recommendations[:3]

    def _generate_handoff_recommendations(self, success_probability: float, features: ConversationFeatures) -> List[str]:
        """Generate recommendations for improving handoff success"""
        recommendations = []

        if success_probability < 0.7:
            recommendations.append("Improve handoff documentation and context transfer")
            recommendations.append("Standardize handoff procedures between agents")

        if features.context_switches > 3:
            recommendations.append("Reduce unnecessary context switches in workflows")

        if features.collaboration_score < 0.5:
            recommendations.append("Enhance collaboration training for agents")

        return recommendations[:3]

    def _generate_duration_recommendations(self, predicted_duration: float, features: ConversationFeatures) -> List[str]:
        """Generate recommendations for task duration optimization"""
        recommendations = []

        if predicted_duration > 3600:  # > 1 hour
            recommendations.append("Consider breaking task into smaller sub-tasks")
            recommendations.append("Optimize workflow to reduce processing time")

        if features.participant_count < 2:
            recommendations.append("Consider adding more participants to complex tasks")

        if features.knowledge_references < 3:
            recommendations.append("Improve knowledge sharing to reduce task duration")

        return recommendations[:3]

    def _generate_conflict_recommendations(self, conflict_probability: float, features: ConversationFeatures) -> List[str]:
        """Generate recommendations for conflict mitigation"""
        recommendations = []

        if conflict_probability > 0.6:
            recommendations.append("Implement conflict resolution protocols")
            recommendations.append("Improve communication clarity between agents")

        if features.sentiment_score < -0.5:
            recommendations.append("Address negative sentiment immediately")

        recommendations.append("Establish clear decision-making processes")

        return recommendations[:3]

    def _generate_compatibility_recommendations(self, compatibility_score: float,
                                             collaboration_efficiency: float,
                                             communication_quality: float,
                                             conflict_risk: float) -> List[str]:
        """Generate recommendations for agent compatibility improvement"""
        recommendations = []

        if compatibility_score < 0.6:
            recommendations.append("Consider compatibility training for agent pair")
            recommendations.append("Review and optimize communication protocols")

        if collaboration_efficiency < 0.5:
            recommendations.append("Focus on improving collaborative workflows")

        if communication_quality < 0.5:
            recommendations.append("Enhance communication clarity and effectiveness")

        if conflict_risk > 0.5:
            recommendations.append("Implement conflict prevention strategies")

        return recommendations[:3]

    def _create_error_prediction(self, prediction_type: ConversationPredictionType,
                               features: ConversationFeatures) -> ConversationPrediction:
        """Create error prediction result"""
        return ConversationPrediction(
            prediction_id=str(uuid.uuid4()),
            prediction_type=prediction_type,
            predicted_value=0.0,
            confidence=0.0,
            risk_level=ConversationRiskLevel.MEDIUM,
            recommendations=["Prediction failed due to error"],
            context={
                'agent_id': features.agent_id,
                'session_id': features.session_id,
                'timestamp': features.timestamp.isoformat(),
                'error': 'Prediction processing failed'
            }
        )

    async def train_conversation_models(self, training_data: List[ConversationFeatures]) -> Dict[str, Dict[str, float]]:
        """Train conversation-specific models"""
        training_results = {}

        try:
            # Prepare training data
            df = self._prepare_training_dataframe(training_data)

            for model_name, model_config in self.conversation_models.items():
                logger.info(f"Training conversation model: {model_name}")

                # Prepare features and target
                X, y = self._prepare_model_data(model_name, df)

                if len(X) < 10:
                    logger.warning(f"Insufficient data for model {model_name}: {len(X)} samples")
                    continue

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # Scale features
                X_train_scaled = self.scalers['standard_scaler'].fit_transform(X_train)
                X_test_scaled = self.scalers['standard_scaler'].transform(X_test)

                # Train model
                model = model_config['model']
                model.fit(X_train_scaled, y_train)

                # Evaluate model
                performance = self._evaluate_conversation_model(model, X_test_scaled, y_test)
                training_results[model_name] = performance

                logger.info(f"Model {model_name} trained with accuracy: {performance.get('accuracy', 0):.3f}")

        except Exception as e:
            logger.error(f"Error training conversation models: {e}")

        return training_results

    def _prepare_training_dataframe(self, training_data: List[ConversationFeatures]) -> pd.DataFrame:
        """Prepare training data as DataFrame"""
        data_rows = []

        for features in training_data:
            row = {
                'agent_id': features.agent_id,
                'session_id': features.session_id,
                'timestamp': features.timestamp,
                'response_time': features.response_time,
                'message_count': features.message_count,
                'error_count': features.error_count,
                'collaboration_score': features.collaboration_score,
                'handoff_count': features.handoff_count,
                'sentiment_score': features.sentiment_score,
                'topic_diversity': features.topic_diversity,
                'participant_count': features.participant_count,
                'knowledge_references': features.knowledge_references,
                'context_switches': features.context_switches,
                'success_rate': features.success_rate,
                'engagement_level': features.engagement_level
            }

            # Add metadata
            row.update(features.metadata)
            data_rows.append(row)

        return pd.DataFrame(data_rows)

    def _prepare_model_data(self, model_name: str, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for specific model"""
        feature_names = self.conversation_models[model_name]['features']

        # Extract features
        X = df[feature_names].fillna(0).values

        # Determine target variable
        if model_name == 'collaboration_efficiency':
            y = df['collaboration_score'].fillna(0.5).values
        elif model_name == 'handoff_success':
            y = (df['success_rate'] > 0.8).astype(int).values  # Binary classification
        elif model_name == 'task_duration':
            y = df['response_time'].fillna(0).values
        elif model_name == 'quality_score':
            y = df['success_rate'].fillna(0.5).values
        elif model_name == 'conflict_prediction':
            y = (df['sentiment_score'] < -0.3).astype(int).values  # Binary classification
        else:
            y = df['collaboration_score'].fillna(0.5).values

        return X, y

    def _evaluate_conversation_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate conversation model performance"""
        try:
            y_pred = model.predict(X_test)

            if hasattr(model, 'predict_proba'):
                # Classification model
                y_pred_proba = model.predict_proba(X_test)
                accuracy = model.score(X_test, y_test)
                return {'accuracy': accuracy}
            else:
                # Regression model
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                return {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'accuracy': max(0, 1 - (mse / np.var(y_test))) if np.var(y_test) > 0 else 0
                }

        except Exception as e:
            logger.error(f"Error evaluating conversation model: {e}")
            return {'accuracy': 0.0}

    async def get_conversation_insights(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Generate comprehensive conversation insights"""
        if time_range is None:
            time_range = (datetime.utcnow() - timedelta(days=7), datetime.utcnow())

        try:
            # Filter predictions by time range
            range_predictions = [
                p for p in self.prediction_history
                if time_range[0] <= p.timestamp <= time_range[1]
            ]

            if not range_predictions:
                return {"message": "No conversation predictions found in the specified time range"}

            # Analyze by prediction type
            type_analysis = {}
            for pred_type in ConversationPredictionType:
                type_predictions = [p for p in range_predictions if p.prediction_type == pred_type]
                if type_predictions:
                    type_analysis[pred_type.value] = {
                        'count': len(type_predictions),
                        'avg_predicted_value': np.mean([p.predicted_value for p in type_predictions]),
                        'avg_confidence': np.mean([p.confidence for p in type_predictions]),
                        'risk_distribution': {
                            risk.value: len([p for p in type_predictions if p.risk_level == risk])
                            for risk in ConversationRiskLevel
                        }
                    }

            # High-risk predictions
            high_risk_predictions = [
                p for p in range_predictions
                if p.risk_level in [ConversationRiskLevel.HIGH, ConversationRiskLevel.CRITICAL]
            ]

            # Agent compatibility insights
            compatibility_insights = {}
            if self.compatibility_cache:
                all_scores = list(self.compatibility_cache.values())
                compatibility_insights = {
                    'total_pairs': len(all_scores),
                    'avg_compatibility': np.mean([s.compatibility_score for s in all_scores]),
                    'high_compatibility_pairs': len([s for s in all_scores if s.compatibility_score > 0.8]),
                    'low_compatibility_pairs': len([s for s in all_scores if s.compatibility_score < 0.4])
                }

            return {
                'analysis_period': {
                    'start': time_range[0].isoformat(),
                    'end': time_range[1].isoformat()
                },
                'total_predictions': len(range_predictions),
                'type_analysis': type_analysis,
                'high_risk_predictions': len(high_risk_predictions),
                'compatibility_insights': compatibility_insights,
                'top_recommendations': self._generate_top_insights(range_predictions),
                'generated_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating conversation insights: {e}")
            return {"error": str(e)}

    def _generate_top_insights(self, predictions: List[ConversationPrediction]) -> List[str]:
        """Generate top insights from predictions"""
        insights = []

        if not predictions:
            return insights

        # Collaboration efficiency insights
        collab_predictions = [p for p in predictions if p.prediction_type == ConversationPredictionType.COLLABORATION_EFFICIENCY]
        if collab_predictions:
            avg_efficiency = np.mean([p.predicted_value for p in collab_predictions])
            if avg_efficiency < 0.6:
                insights.append(f"Low collaboration efficiency detected (avg: {avg_efficiency:.1%})")
            elif avg_efficiency > 0.8:
                insights.append(f"High collaboration efficiency achieved (avg: {avg_efficiency:.1%})")

        # Conflict risk insights
        conflict_predictions = [p for p in predictions if p.prediction_type == ConversationPredictionType.CONFLICT_PREDICTION]
        if conflict_predictions:
            high_conflict = len([p for p in conflict_predictions if p.predicted_value > 0.6])
            if high_conflict > len(conflict_predictions) * 0.3:
                insights.append(f"High conflict risk in {high_conflict}/{len(conflict_predictions)} conversations")

        # Task duration insights
        duration_predictions = [p for p in predictions if p.prediction_type == ConversationPredictionType.TASK_DURATION]
        if duration_predictions:
            long_tasks = len([p for p in duration_predictions if p.predicted_value > 1800])  # > 30 minutes
            if long_tasks > len(duration_predictions) * 0.4:
                insights.append(f"Many tasks taking longer than 30 minutes ({long_tasks}/{len(duration_predictions)})")

        return insights[:5]


# Factory function for creating conversation predictive analyzer
def create_conversation_predictive_analyzer(base_analytics: Optional[PredictiveAnalytics] = None,
                                           config: Optional[Dict[str, Any]] = None) -> ConversationPredictiveAnalyzer:
    """Factory function to create conversation predictive analyzer"""
    return ConversationPredictiveAnalyzer(base_analytics, config)