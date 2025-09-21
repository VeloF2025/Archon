"""
Data Processing Module for Conversation Analytics

This module provides comprehensive data processing capabilities for conversation analytics,
including data transformation, cleaning, normalization, feature engineering, and
preparation for machine learning models.

Author: Archon AI System
Date: 2025-09-21
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class DataQualityIssue(Enum):
    """Data quality issue types"""
    MISSING_VALUES = "missing_values"
    DUPLICATE_ENTRIES = "duplicate_entries"
    OUT_OF_RANGE = "out_of_range"
    INVALID_FORMAT = "invalid_format"
    INCONSISTENT_TIMESTAMPS = "inconsistent_timestamps"
    MALFORMED_JSON = "malformed_json"
    EMPTY_MESSAGES = "empty_messages"
    INVALID_AGENTS = "invalid_agents"


@dataclass
class ProcessingResult:
    """Result of data processing operation"""
    processing_id: str
    status: ProcessingStatus
    input_records: int
    output_records: int
    quality_issues: Dict[DataQualityIssue, int] = field(default_factory=dict)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    """Comprehensive data quality report"""
    total_records: int
    valid_records: int
    quality_issues: Dict[DataQualityIssue, List[Dict[str, Any]]] = field(default_factory=dict)
    quality_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureVector:
    """Processed feature vector for ML models"""
    vector_id: str
    features: np.ndarray
    feature_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class DataProcessor(ABC):
    """Abstract base class for data processors"""

    def __init__(self, processor_id: str):
        self.processor_id = processor_id
        self.processing_count = 0
        self.total_processing_time = 0.0

    @abstractmethod
    async def process(self, data: Any) -> ProcessingResult:
        """Process data and return results"""
        pass

    @abstractmethod
    def validate_input(self, data: Any) -> bool:
        """Validate input data"""
        pass


class ConversationDataProcessor(DataProcessor):
    """Main conversation data processor"""

    def __init__(self, processor_id: str = "conversation_processor"):
        super().__init__(processor_id)
        self.text_cleaner = TextCleaner()
        self.normalizer = DataNormalizer()
        self.feature_extractor = BasicFeatureExtractor()
        self.quality_validator = DataQualityValidator()

    def validate_input(self, data: Union[List[Dict], Dict]) -> bool:
        """Validate conversation data input"""
        if isinstance(data, dict):
            return self._validate_single_conversation(data)
        elif isinstance(data, list):
            return all(self._validate_single_conversation(item) for item in data)
        return False

    def _validate_single_conversation(self, conversation: Dict) -> bool:
        """Validate single conversation entry"""
        required_fields = ['id', 'messages', 'timestamp', 'participants']
        return all(field in conversation for field in required_fields)

    async def process(self, data: Union[List[Dict], Dict]) -> ProcessingResult:
        """Process conversation data"""
        processing_start = datetime.now()
        processing_id = f"proc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.processing_count}"

        try:
            # Convert single item to list for uniform processing
            if isinstance(data, dict):
                data = [data]

            input_records = len(data)
            processed_data = []
            quality_issues = {}

            # Process each conversation
            for i, conversation in enumerate(data):
                try:
                    # Validate conversation structure
                    if not self._validate_single_conversation(conversation):
                        quality_issues.setdefault(DataQualityIssue.INVALID_FORMAT, []).append(i)
                        continue

                    # Clean and normalize data
                    cleaned_conversation = self._clean_conversation(conversation)

                    # Extract features
                    features = self._extract_features(cleaned_conversation)

                    # Validate quality
                    quality_result = self.quality_validator.validate(cleaned_conversation)

                    # Combine results
                    processed_item = {
                        **cleaned_conversation,
                        'features': features,
                        'quality_score': quality_result.quality_score,
                        'processing_metadata': {
                            'processing_id': processing_id,
                            'processed_at': datetime.now().isoformat(),
                            'quality_issues': quality_result.quality_issues
                        }
                    }

                    processed_data.append(processed_item)

                    # Aggregate quality issues
                    for issue_type, issues in quality_result.quality_issues.items():
                        quality_issues.setdefault(issue_type, []).extend(
                            [{'index': i, 'issue': issue} for issue in issues]
                        )

                except Exception as e:
                    logger.error(f"Error processing conversation {i}: {e}")
                    quality_issues.setdefault(DataQualityIssue.INVALID_FORMAT, []).append({
                        'index': i,
                        'error': str(e)
                    })

            # Calculate processing metrics
            processing_time = (datetime.now() - processing_start).total_seconds()
            output_records = len(processed_data)

            # Determine processing status
            if output_records == 0:
                status = ProcessingStatus.FAILED
            elif output_records == input_records:
                status = ProcessingStatus.COMPLETED
            else:
                status = ProcessingStatus.PARTIAL

            result = ProcessingResult(
                processing_id=processing_id,
                status=status,
                input_records=input_records,
                output_records=output_records,
                quality_issues={k: len(v) for k, v in quality_issues.items()},
                processing_time=processing_time,
                metadata={
                    'processor_id': self.processor_id,
                    'processing_date': processing_start.isoformat(),
                    'quality_summary': self._generate_quality_summary(quality_issues)
                }
            )

            self.processing_count += 1
            self.total_processing_time += processing_time

            logger.info(f"Processed {input_records} conversations in {processing_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return ProcessingResult(
                processing_id=processing_id,
                status=ProcessingStatus.FAILED,
                input_records=len(data) if isinstance(data, list) else 1,
                output_records=0,
                processing_time=(datetime.now() - processing_start).total_seconds(),
                error_message=str(e)
            )

    def _clean_conversation(self, conversation: Dict) -> Dict:
        """Clean conversation data"""
        cleaned = conversation.copy()

        # Clean messages
        if 'messages' in cleaned:
            processed_messages = []
            for msg in cleaned['messages']:
                if isinstance(msg, dict) and msg.get('content'):
                    processed_messages.append(self.text_cleaner.clean_text(msg['content']))
                elif isinstance(msg, str) and msg.strip():
                    processed_messages.append(self.text_cleaner.clean_text(msg))
            cleaned['messages'] = processed_messages

        # Normalize timestamps
        if 'timestamp' in cleaned:
            cleaned['timestamp'] = self.normalizer.normalize_timestamp(cleaned['timestamp'])

        # Clean participants
        if 'participants' in cleaned:
            cleaned['participants'] = [
                self.normalizer.normalize_agent_id(agent)
                for agent in cleaned['participants']
            ]

        return cleaned

    def _extract_features(self, conversation: Dict) -> Dict[str, Any]:
        """Extract features from conversation"""
        return self.feature_extractor.extract_features(conversation)

    def _generate_quality_summary(self, quality_issues: Dict) -> Dict[str, Any]:
        """Generate quality summary statistics"""
        total_issues = sum(len(issues) for issues in quality_issues.values())
        issue_types = list(quality_issues.keys())

        return {
            'total_issues': total_issues,
            'issue_types': issue_types,
            'most_common_issue': max(quality_issues.items(), key=lambda x: len(x[1]))[0] if quality_issues else None,
            'issue_distribution': {issue_type.value: len(issues) for issue_type, issues in quality_issues.items()}
        }


class TextCleaner:
    """Text cleaning utilities"""

    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.code_pattern = re.compile(r'```[\s\S]*?```|`[^`\n]+`')

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove URLs (optional - can be configured)
        # text = self.url_pattern.sub('[URL]', text)

        # Remove email addresses (optional)
        # text = self.email_pattern.sub('[EMAIL]', text)

        # Remove code blocks (optional - can be configured)
        # text = self.code_pattern.sub('[CODE]', text)

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"').replace("'", "'").replace("'", "'")

        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\@\#\$\%\^\&\*\+\=\~\`]', '', text)

        return text


class DataNormalizer:
    """Data normalization utilities"""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}

    def normalize_timestamp(self, timestamp: Union[str, datetime]) -> str:
        """Normalize timestamp to ISO format"""
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return dt.isoformat()
            except ValueError:
                # Try common formats
                formats = [
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%dT%H:%M:%S',
                    '%Y-%m-%dT%H:%M:%SZ',
                    '%Y-%m-%dT%H:%M:%S.%fZ'
                ]
                for fmt in formats:
                    try:
                        dt = datetime.strptime(timestamp, fmt)
                        return dt.isoformat()
                    except ValueError:
                        continue
                return timestamp
        elif isinstance(timestamp, datetime):
            return timestamp.isoformat()
        return str(timestamp)

    def normalize_agent_id(self, agent_id: str) -> str:
        """Normalize agent identifier"""
        if not agent_id:
            return "unknown_agent"

        # Remove special characters and normalize
        normalized = re.sub(r'[^\w\-_]', '_', str(agent_id))
        normalized = re.sub(r'_+', '_', normalized)
        normalized = normalized.strip('_')

        return normalized.lower()

    def normalize_numeric_values(self, values: List[float], method: str = 'standard') -> np.ndarray:
        """Normalize numeric values"""
        values_array = np.array(values).reshape(-1, 1)

        if method == 'standard':
            if 'standard' not in self.scalers:
                self.scalers['standard'] = StandardScaler()
                self.scalers['standard'].fit(values_array)
            return self.scalers['standard'].transform(values_array).flatten()
        elif method == 'minmax':
            if 'minmax' not in self.scalers:
                self.scalers['minmax'] = MinMaxScaler()
                self.scalers['minmax'].fit(values_array)
            return self.scalers['minmax'].transform(values_array).flatten()

        return values_array.flatten()


class BasicFeatureExtractor:
    """Basic feature extraction from conversations"""

    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )

    def extract_features(self, conversation: Dict) -> Dict[str, Any]:
        """Extract features from conversation"""
        features = {}

        # Basic conversation features
        features['message_count'] = len(conversation.get('messages', []))
        features['participant_count'] = len(conversation.get('participants', []))
        features['conversation_duration'] = self._calculate_duration(conversation)

        # Text features
        if 'messages' in conversation:
            all_text = ' '.join(conversation['messages'])
            features.update(self._extract_text_features(all_text))

        # Interaction features
        features.update(self._extract_interaction_features(conversation))

        # Temporal features
        features.update(self._extract_temporal_features(conversation))

        return features

    def _calculate_duration(self, conversation: Dict) -> float:
        """Calculate conversation duration in minutes"""
        messages = conversation.get('messages', [])
        if not messages:
            return 0.0

        # This is a simplified calculation - in practice, you'd need message timestamps
        return len(messages) * 2.0  # Assume 2 minutes per message as average

    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract text-based features"""
        features = {}

        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(text.split('.'))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        features['question_count'] = text.count('?')
        features['exclamation_count'] = text.count('!')
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0

        return features

    def _extract_interaction_features(self, conversation: Dict) -> Dict[str, Any]:
        """Extract interaction-based features"""
        features = {}

        participants = conversation.get('participants', [])
        messages = conversation.get('messages', [])

        features['participant_diversity'] = len(set(participants))
        features['message_per_participant'] = len(messages) / len(participants) if participants else 0
        features['is_multiparty'] = len(participants) > 2

        return features

    def _extract_temporal_features(self, conversation: Dict) -> Dict[str, Any]:
        """Extract temporal features"""
        features = {}

        timestamp = conversation.get('timestamp')
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                features['hour_of_day'] = dt.hour
                features['day_of_week'] = dt.weekday()
                features['is_weekend'] = dt.weekday() >= 5
                features['is_business_hours'] = 9 <= dt.hour <= 17
            except ValueError:
                pass

        return features


class DataQualityValidator:
    """Data quality validation"""

    def validate(self, conversation: Dict) -> DataQualityReport:
        """Validate conversation data quality"""
        issues = {}
        total_score = 100.0

        # Check for missing values
        missing_issues = self._check_missing_values(conversation)
        if missing_issues:
            issues[DataQualityIssue.MISSING_VALUES] = missing_issues
            total_score -= len(missing_issues) * 5

        # Check for empty messages
        empty_issues = self._check_empty_messages(conversation)
        if empty_issues:
            issues[DataQualityIssue.EMPTY_MESSAGES] = empty_issues
            total_score -= len(empty_issues) * 3

        # Check timestamp consistency
        timestamp_issues = self._check_timestamp_consistency(conversation)
        if timestamp_issues:
            issues[DataQualityIssue.INCONSISTENT_TIMESTAMPS] = timestamp_issues
            total_score -= len(timestamp_issues) * 10

        # Check agent validity
        agent_issues = self._check_agent_validity(conversation)
        if agent_issues:
            issues[DataQualityIssue.INVALID_AGENTS] = agent_issues
            total_score -= len(agent_issues) * 7

        # Ensure score doesn't go below 0
        quality_score = max(0.0, total_score)

        return DataQualityReport(
            total_records=1,
            valid_records=1 if quality_score > 70 else 0,
            quality_issues=issues,
            quality_score=quality_score,
            recommendations=self._generate_recommendations(issues)
        )

    def _check_missing_values(self, conversation: Dict) -> List[Dict]:
        """Check for missing required values"""
        issues = []
        required_fields = ['id', 'messages', 'timestamp', 'participants']

        for field in required_fields:
            if field not in conversation or not conversation[field]:
                issues.append({
                    'field': field,
                    'issue': 'missing_or_empty',
                    'severity': 'high'
                })

        return issues

    def _check_empty_messages(self, conversation: Dict) -> List[Dict]:
        """Check for empty or invalid messages"""
        issues = []
        messages = conversation.get('messages', [])

        for i, msg in enumerate(messages):
            if not msg or len(msg.strip()) < 3:
                issues.append({
                    'message_index': i,
                    'issue': 'empty_or_too_short',
                    'severity': 'medium'
                })

        return issues

    def _check_timestamp_consistency(self, conversation: Dict) -> List[Dict]:
        """Check timestamp consistency"""
        issues = []
        timestamp = conversation.get('timestamp')

        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

                # Handle timezone-aware comparison
                now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()

                # Check if timestamp is in the future
                if dt > now + timedelta(minutes=5):
                    issues.append({
                        'timestamp': timestamp,
                        'issue': 'future_timestamp',
                        'severity': 'high'
                    })

                # Check if timestamp is too old
                if dt < now - timedelta(days=365):
                    issues.append({
                        'timestamp': timestamp,
                        'issue': 'ancient_timestamp',
                        'severity': 'medium'
                    })

            except ValueError:
                issues.append({
                    'timestamp': timestamp,
                    'issue': 'invalid_format',
                    'severity': 'high'
                })

        return issues

    def _check_agent_validity(self, conversation: Dict) -> List[Dict]:
        """Check agent identifier validity"""
        issues = []
        participants = conversation.get('participants', [])

        for i, agent in enumerate(participants):
            if not agent or len(agent.strip()) < 2:
                issues.append({
                    'agent_index': i,
                    'agent_id': agent,
                    'issue': 'invalid_agent_id',
                    'severity': 'medium'
                })

        return issues

    def _generate_recommendations(self, issues: Dict[DataQualityIssue, List[Dict]]) -> List[str]:
        """Generate recommendations for quality improvement"""
        recommendations = []

        if DataQualityIssue.MISSING_VALUES in issues:
            recommendations.append("Ensure all required fields (id, messages, timestamp, participants) are present")

        if DataQualityIssue.EMPTY_MESSAGES in issues:
            recommendations.append("Filter out empty or very short messages")

        if DataQualityIssue.INCONSISTENT_TIMESTAMPS in issues:
            recommendations.append("Validate timestamp formats and check for future timestamps")

        if DataQualityIssue.INVALID_AGENTS in issues:
            recommendations.append("Validate agent identifiers and remove invalid entries")

        return recommendations


class FeatureEngineeringProcessor:
    """Advanced feature engineering for ML models"""

    def __init__(self):
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.feature_importance = {}

    def create_feature_vectors(self, conversations: List[Dict]) -> List[FeatureVector]:
        """Create feature vectors for ML models"""
        feature_vectors = []

        for conversation in conversations:
            try:
                # Extract base features
                base_features = self._extract_base_features(conversation)

                # Create interaction features
                interaction_features = self._create_interaction_features(base_features)

                # Create temporal features
                temporal_features = self._create_temporal_features(conversation)

                # Combine all features
                combined_features = {
                    **base_features,
                    **interaction_features,
                    **temporal_features
                }

                # Convert to numpy array
                feature_names = list(combined_features.keys())
                feature_array = np.array(list(combined_features.values()))

                # Handle NaN values
                feature_array = np.nan_to_num(feature_array, nan=0.0)

                feature_vector = FeatureVector(
                    vector_id=f"feat_{conversation.get('id', 'unknown')}",
                    features=feature_array,
                    feature_names=feature_names,
                    metadata={
                        'conversation_id': conversation.get('id'),
                        'participant_count': conversation.get('participant_count', 0),
                        'message_count': conversation.get('message_count', 0)
                    }
                )

                feature_vectors.append(feature_vector)

            except Exception as e:
                logger.error(f"Error creating feature vector for conversation {conversation.get('id')}: {e}")

        return feature_vectors

    def apply_dimensionality_reduction(self, feature_vectors: List[FeatureVector]) -> List[FeatureVector]:
        """Apply PCA for dimensionality reduction"""
        if len(feature_vectors) < 2:
            return feature_vectors

        # Stack all feature vectors
        features_matrix = np.vstack([fv.features for fv in feature_vectors])

        # Fit PCA if not already fitted
        if not hasattr(self.pca, 'components_'):
            features_matrix_reduced = self.pca.fit_transform(features_matrix)
        else:
            features_matrix_reduced = self.pca.transform(features_matrix)

        # Create reduced feature vectors
        reduced_vectors = []
        for i, fv in enumerate(feature_vectors):
            reduced_fv = FeatureVector(
                vector_id=f"{fv.vector_id}_pca",
                features=features_matrix_reduced[i],
                feature_names=[f"PC_{j}" for j in range(features_matrix_reduced.shape[1])],
                metadata={
                    **fv.metadata,
                    'original_dimensions': len(fv.features),
                    'reduced_dimensions': len(features_matrix_reduced[i]),
                    'variance_explained': self.pca.explained_variance_ratio_.sum()
                }
            )
            reduced_vectors.append(reduced_fv)

        return reduced_vectors

    def _extract_base_features(self, conversation: Dict) -> Dict[str, float]:
        """Extract base numerical features"""
        features = {}

        # Message statistics
        messages = conversation.get('messages', [])
        features['total_messages'] = len(messages)
        features['avg_message_length'] = np.mean([len(msg) for msg in messages]) if messages else 0
        features['max_message_length'] = max([len(msg) for msg in messages]) if messages else 0
        features['min_message_length'] = min([len(msg) for msg in messages]) if messages else 0

        # Participant statistics
        participants = conversation.get('participants', [])
        features['participant_count'] = len(participants)
        features['unique_participant_ratio'] = len(set(participants)) / len(participants) if participants else 0

        return features

    def _create_interaction_features(self, base_features: Dict[str, float]) -> Dict[str, float]:
        """Create interaction features"""
        features = {}

        # Message-participant interactions
        if base_features.get('participant_count', 0) > 0:
            features['messages_per_participant'] = base_features.get('total_messages', 0) / base_features['participant_count']

        # Length variability
        if base_features.get('avg_message_length', 0) > 0:
            features['length_variability'] = base_features.get('max_message_length', 0) / base_features['avg_message_length']

        return features

    def _create_temporal_features(self, conversation: Dict) -> Dict[str, float]:
        """Create temporal features"""
        features = {}

        timestamp = conversation.get('timestamp')
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                features['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
                features['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
                features['day_sin'] = np.sin(2 * np.pi * dt.weekday() / 7)
                features['day_cos'] = np.cos(2 * np.pi * dt.weekday() / 7)
                features['month_sin'] = np.sin(2 * np.pi * dt.month / 12)
                features['month_cos'] = np.cos(2 * np.pi * dt.month / 12)
            except ValueError:
                pass

        return features


# Factory function for creating data processors
def create_conversation_data_processor(processor_type: str = "default") -> DataProcessor:
    """Create a conversation data processor instance"""
    if processor_type == "default":
        return ConversationDataProcessor()
    else:
        raise ValueError(f"Unknown processor type: {processor_type}")


# Convenience functions for common processing tasks
async def process_conversation_batch(
    conversations: List[Dict],
    processor: Optional[DataProcessor] = None
) -> ProcessingResult:
    """Process a batch of conversations"""
    if processor is None:
        processor = create_conversation_data_processor()

    return await processor.process(conversations)


def create_feature_vectors_from_conversations(
    conversations: List[Dict],
    apply_pca: bool = False
) -> List[FeatureVector]:
    """Create feature vectors from conversations"""
    feature_engineer = FeatureEngineeringProcessor()
    feature_vectors = feature_engineer.create_feature_vectors(conversations)

    if apply_pca and len(feature_vectors) > 1:
        feature_vectors = feature_engineer.apply_dimensionality_reduction(feature_vectors)

    return feature_vectors


# Example usage and testing
if __name__ == "__main__":
    async def test_data_processing():
        """Test the data processing module"""
        # Sample conversation data
        sample_conversations = [
            {
                "id": "conv_001",
                "messages": [
                    "Hello, I need help with my project.",
                    "Sure, I can help you with that. What's the issue?",
                    "I'm having trouble with the database connection."
                ],
                "timestamp": "2025-09-21T10:30:00Z",
                "participants": ["agent_001", "agent_002"]
            },
            {
                "id": "conv_002",
                "messages": [
                    "Can you review my code?",
                    "Yes, let me take a look.",
                    "The code looks good overall."
                ],
                "timestamp": "2025-09-21T14:15:00Z",
                "participants": ["agent_003", "agent_004"]
            }
        ]

        # Test processing
        processor = create_conversation_data_processor()
        result = await processor.process(sample_conversations)

        print(f"Processing result: {result.status}")
        print(f"Input records: {result.input_records}")
        print(f"Output records: {result.output_records}")
        print(f"Processing time: {result.processing_time:.2f}s")

        # Test feature vector creation
        feature_vectors = create_feature_vectors_from_conversations(sample_conversations)
        print(f"Created {len(feature_vectors)} feature vectors")

        return result

    # Run the test
    asyncio.run(test_data_processing())