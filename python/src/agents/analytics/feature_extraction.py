"""
Feature Extraction Engine for Conversation Analytics

This module provides advanced feature extraction capabilities specifically designed
for conversation analytics, including linguistic features, interaction patterns,
sentiment analysis, and behavioral indicators.

Author: Archon AI System
Date: 2025-09-21
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import re
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Feature type enumeration"""
    LINGUISTIC = "linguistic"
    INTERACTION = "interaction"
    SENTIMENT = "sentiment"
    TEMPORAL = "temporal"
    STRUCTURAL = "structural"
    BEHAVIORAL = "behavioral"
    NETWORK = "network"
    SEMANTIC = "semantic"


class ExtractionMethod(Enum):
    """Feature extraction method enumeration"""
    STATISTICAL = "statistical"
    EMBEDDING = "embedding"
    RULE_BASED = "rule_based"
    ML_MODEL = "ml_model"
    HYBRID = "hybrid"


@dataclass
class FeatureDefinition:
    """Feature definition with metadata"""
    name: str
    feature_type: FeatureType
    extraction_method: ExtractionMethod
    description: str
    data_type: str = "float"
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    is_categorical: bool = False
    categories: Optional[List[str]] = None
    importance_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureExtractionResult:
    """Result of feature extraction"""
    conversation_id: str
    features: Dict[str, Any]
    feature_definitions: Dict[str, FeatureDefinition]
    extraction_time: float = 0.0
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class BatchFeatureExtractionResult:
    """Result of batch feature extraction"""
    results: List[FeatureExtractionResult]
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    total_extraction_time: float = 0.0
    average_confidence: float = 0.0
    feature_coverage: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors"""

    def __init__(self, extractor_id: str):
        self.extractor_id = extractor_id
        self.feature_definitions: Dict[str, FeatureDefinition] = {}
        self.extraction_count = 0
        self.total_extraction_time = 0.0

    @abstractmethod
    async def extract_features(self, conversation: Dict[str, Any]) -> FeatureExtractionResult:
        """Extract features from conversation"""
        pass

    @abstractmethod
    def get_feature_definitions(self) -> Dict[str, FeatureDefinition]:
        """Get feature definitions"""
        pass


class LinguisticFeatureExtractor(FeatureExtractor):
    """Linguistic feature extraction for conversation analysis"""

    def __init__(self):
        super().__init__("linguistic_extractor")
        self._initialize_feature_definitions()
        self._initialize_patterns()

    def _initialize_feature_definitions(self):
        """Initialize linguistic feature definitions"""
        self.feature_definitions = {
            "lexical_diversity": FeatureDefinition(
                name="lexical_diversity",
                feature_type=FeatureType.LINGUISTIC,
                extraction_method=ExtractionMethod.STATISTICAL,
                description="Ratio of unique words to total words",
                data_type="float",
                range_min=0.0,
                range_max=1.0,
                importance_score=0.8
            ),
            "avg_sentence_length": FeatureDefinition(
                name="avg_sentence_length",
                feature_type=FeatureType.LINGUISTIC,
                extraction_method=ExtractionMethod.STATISTICAL,
                description="Average number of words per sentence",
                data_type="float",
                range_min=1.0,
                range_max=100.0,
                importance_score=0.7
            ),
            "question_ratio": FeatureDefinition(
                name="question_ratio",
                feature_type=FeatureType.LINGUISTIC,
                extraction_method=ExtractionMethod.STATISTICAL,
                description="Ratio of questions to total statements",
                data_type="float",
                range_min=0.0,
                range_max=1.0,
                importance_score=0.6
            ),
            "exclamation_ratio": FeatureDefinition(
                name="exclamation_ratio",
                feature_type=FeatureType.LINGUISTIC,
                extraction_method=ExtractionMethod.STATISTICAL,
                description="Ratio of exclamations to total statements",
                data_type="float",
                range_min=0.0,
                range_max=1.0,
                importance_score=0.5
            ),
            "formality_score": FeatureDefinition(
                name="formality_score",
                feature_type=FeatureType.LINGUISTIC,
                extraction_method=ExtractionMethod.RULE_BASED,
                description="Measure of language formality",
                data_type="float",
                range_min=0.0,
                range_max=1.0,
                importance_score=0.7
            ),
            "readability_score": FeatureDefinition(
                name="readability_score",
                feature_type=FeatureType.LINGUISTIC,
                extraction_method=ExtractionMethod.STATISTICAL,
                description="Text readability complexity score",
                data_type="float",
                range_min=0.0,
                range_max=100.0,
                importance_score=0.6
            ),
            "technical_term_ratio": FeatureDefinition(
                name="technical_term_ratio",
                feature_type=FeatureType.LINGUISTIC,
                extraction_method=ExtractionMethod.RULE_BASED,
                description="Ratio of technical terms to total words",
                data_type="float",
                range_min=0.0,
                range_max=1.0,
                importance_score=0.8
            )
        }

    def _initialize_patterns(self):
        """Initialize linguistic patterns"""
        self.technical_terms = {
            'api', 'database', 'server', 'client', 'framework', 'library',
            'algorithm', 'function', 'method', 'class', 'object', 'variable',
            'parameter', 'argument', 'return', 'call', 'execute', 'process',
            'thread', 'async', 'sync', 'http', 'json', 'xml', 'sql', 'nosql'
        }

        self.formal_words = {
            'therefore', 'however', 'furthermore', 'moreover', 'consequently',
            'additionally', 'accordingly', 'subsequently', 'nevertheless',
            'notwithstanding', 'henceforth', 'hereby', 'wherein', 'whereas'
        }

        self.informal_words = {
            'hey', 'hi', 'yeah', 'yep', 'nope', 'gonna', 'wanna', 'gotta',
            'kinda', 'sorta', 'cool', 'awesome', 'great', 'thanks', 'thank'
        }

    async def extract_features(self, conversation: Dict[str, Any]) -> FeatureExtractionResult:
        """Extract linguistic features from conversation"""
        extraction_start = datetime.now()

        try:
            # Combine all text from conversation
            all_text = self._extract_all_text(conversation)

            # Extract features
            features = {}

            # Basic text statistics
            features.update(self._extract_basic_stats(all_text))

            # Lexical diversity
            features["lexical_diversity"] = self._calculate_lexical_diversity(all_text)

            # Sentence analysis
            features.update(self._extract_sentence_features(all_text))

            # Formality analysis
            features["formality_score"] = self._calculate_formality_score(all_text)

            # Readability score
            features["readability_score"] = self._calculate_readability_score(all_text)

            # Technical term analysis
            features["technical_term_ratio"] = self._calculate_technical_term_ratio(all_text)

            # Calculate extraction time
            extraction_time = (datetime.now() - extraction_start).total_seconds()

            return FeatureExtractionResult(
                conversation_id=conversation.get('id', 'unknown'),
                features=features,
                feature_definitions=self.feature_definitions,
                extraction_time=extraction_time,
                metadata={
                    'extractor_id': self.extractor_id,
                    'total_words': len(all_text.split()),
                    'total_characters': len(all_text)
                }
            )

        except Exception as e:
            logger.error(f"Error extracting linguistic features: {e}")
            extraction_time = (datetime.now() - extraction_start).total_seconds()

            return FeatureExtractionResult(
                conversation_id=conversation.get('id', 'unknown'),
                features={},
                feature_definitions=self.feature_definitions,
                extraction_time=extraction_time,
                errors=[str(e)]
            )

    def _extract_all_text(self, conversation: Dict[str, Any]) -> str:
        """Extract all text from conversation"""
        messages = conversation.get('messages', [])
        if isinstance(messages, list):
            # Handle string messages
            if all(isinstance(msg, str) for msg in messages):
                return ' '.join(messages)
            # Handle dict messages with content field
            elif all(isinstance(msg, dict) and 'content' in msg for msg in messages):
                return ' '.join(msg['content'] for msg in messages)

        return ""

    def _extract_basic_stats(self, text: str) -> Dict[str, Any]:
        """Extract basic text statistics"""
        if not text:
            return {"word_count": 0, "character_count": 0}

        words = text.split()
        return {
            "word_count": len(words),
            "character_count": len(text),
            "avg_word_length": np.mean([len(word) for word in words]) if words else 0,
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }

    def _calculate_lexical_diversity(self, text: str) -> float:
        """Calculate lexical diversity (Type-Token Ratio)"""
        if not text:
            return 0.0

        words = text.lower().split()
        unique_words = set(words)

        return len(unique_words) / len(words) if words else 0.0

    def _extract_sentence_features(self, text: str) -> Dict[str, Any]:
        """Extract sentence-level features"""
        if not text:
            return {"avg_sentence_length": 0, "question_ratio": 0, "exclamation_ratio": 0}

        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return {"avg_sentence_length": 0, "question_ratio": 0, "exclamation_ratio": 0}

        # Calculate average sentence length
        avg_length = np.mean([len(sentence.split()) for sentence in sentences])

        # Count questions and exclamations
        question_count = sum(1 for sentence in sentences if '?' in sentence)
        exclamation_count = sum(1 for sentence in sentences if '!' in sentence)

        return {
            "avg_sentence_length": avg_length,
            "question_ratio": question_count / len(sentences),
            "exclamation_ratio": exclamation_count / len(sentences)
        }

    def _calculate_formality_score(self, text: str) -> float:
        """Calculate language formality score"""
        if not text:
            return 0.5  # Neutral

        words = text.lower().split()
        if not words:
            return 0.5

        formal_count = sum(1 for word in words if word in self.formal_words)
        informal_count = sum(1 for word in words if word in self.informal_words)

        # Score between 0 (very informal) and 1 (very formal)
        total_markers = formal_count + informal_count
        if total_markers == 0:
            return 0.5

        return formal_count / total_markers

    def _calculate_readability_score(self, text: str) -> float:
        """Calculate simplified readability score"""
        if not text:
            return 50.0  # Medium readability

        sentences = [s.strip() for s in text.split('.') if s.strip()]
        words = text.split()

        if not sentences or not words:
            return 50.0

        avg_words_per_sentence = len(words) / len(sentences)
        avg_syllables_per_word = np.mean([self._count_syllables(word) for word in words])

        # Simplified Flesch Reading Ease formula
        score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
        return max(0, min(100, score))

    def _count_syllables(self, word: str) -> int:
        """Simple syllable counter"""
        word = word.lower()
        if not word:
            return 0

        vowels = 'aeiouy'
        syllable_count = 0
        prev_char_was_vowel = False

        for char in word:
            if char in vowels:
                if not prev_char_was_vowel:
                    syllable_count += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False

        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1

        return max(1, syllable_count)

    def _calculate_technical_term_ratio(self, text: str) -> float:
        """Calculate ratio of technical terms"""
        if not text:
            return 0.0

        words = text.lower().split()
        if not words:
            return 0.0

        technical_count = sum(1 for word in words if word in self.technical_terms)
        return technical_count / len(words)

    def get_feature_definitions(self) -> Dict[str, FeatureDefinition]:
        """Get linguistic feature definitions"""
        return self.feature_definitions


class InteractionFeatureExtractor(FeatureExtractor):
    """Interaction pattern feature extraction"""

    def __init__(self):
        super().__init__("interaction_extractor")
        self._initialize_feature_definitions()

    def _initialize_feature_definitions(self):
        """Initialize interaction feature definitions"""
        self.feature_definitions = {
            "turn_taking_balance": FeatureDefinition(
                name="turn_taking_balance",
                feature_type=FeatureType.INTERACTION,
                extraction_method=ExtractionMethod.STATISTICAL,
                description="Balance of conversation turns between participants",
                data_type="float",
                range_min=0.0,
                range_max=1.0,
                importance_score=0.9
            ),
            "response_time_avg": FeatureDefinition(
                name="response_time_avg",
                feature_type=FeatureType.INTERACTION,
                extraction_method=ExtractionMethod.STATISTICAL,
                description="Average response time between messages",
                data_type="float",
                range_min=0.0,
                range_max=3600.0,  # Up to 1 hour
                importance_score=0.7
            ),
            "interruption_count": FeatureDefinition(
                name="interruption_count",
                feature_type=FeatureType.INTERACTION,
                extraction_method=ExtractionMethod.RULE_BASED,
                description="Number of conversation interruptions",
                data_type="int",
                range_min=0,
                importance_score=0.6
            ),
            "topic_shift_frequency": FeatureDefinition(
                name="topic_shift_frequency",
                feature_type=FeatureType.INTERACTION,
                extraction_method=ExtractionMethod.RULE_BASED,
                description="Frequency of topic changes in conversation",
                data_type="float",
                range_min=0.0,
                range_max=1.0,
                importance_score=0.8
            ),
            "collaboration_index": FeatureDefinition(
                name="collaboration_index",
                feature_type=FeatureType.INTERACTION,
                extraction_method=ExtractionMethod.HYBRID,
                description="Measure of collaborative behavior",
                data_type="float",
                range_min=0.0,
                range_max=1.0,
                importance_score=0.9
            ),
            "dominance_ratio": FeatureDefinition(
                name="dominance_ratio",
                feature_type=FeatureType.INTERACTION,
                extraction_method=ExtractionMethod.STATISTICAL,
                description="Ratio of most active participant's contribution",
                data_type="float",
                range_min=0.0,
                range_max=1.0,
                importance_score=0.7
            ),
            "backchannel_frequency": FeatureDefinition(
                name="backchannel_frequency",
                feature_type=FeatureType.INTERACTION,
                extraction_method=ExtractionMethod.RULE_BASED,
                description="Frequency of backchannel responses",
                data_type="float",
                range_min=0.0,
                range_max=1.0,
                importance_score=0.5
            )
        }

    async def extract_features(self, conversation: Dict[str, Any]) -> FeatureExtractionResult:
        """Extract interaction features from conversation"""
        extraction_start = datetime.now()

        try:
            features = {}

            # Extract message structure
            message_structure = self._extract_message_structure(conversation)

            # Turn taking analysis
            features["turn_taking_balance"] = self._calculate_turn_balance(message_structure)

            # Response time analysis
            features.update(self._analyze_response_times(message_structure))

            # Interruption detection
            features["interruption_count"] = self._count_interruptions(message_structure)

            # Topic shift analysis
            features["topic_shift_frequency"] = self._calculate_topic_shifts(message_structure)

            # Collaboration index
            features["collaboration_index"] = self._calculate_collaboration_index(message_structure)

            # Dominance analysis
            features["dominance_ratio"] = self._calculate_dominance_ratio(message_structure)

            # Backchannel analysis
            features["backchannel_frequency"] = self._calculate_backchannel_frequency(message_structure)

            extraction_time = (datetime.now() - extraction_start).total_seconds()

            return FeatureExtractionResult(
                conversation_id=conversation.get('id', 'unknown'),
                features=features,
                feature_definitions=self.feature_definitions,
                extraction_time=extraction_time,
                metadata={
                    'extractor_id': self.extractor_id,
                    'total_messages': len(message_structure),
                    'participant_count': len(set(msg['participant'] for msg in message_structure))
                }
            )

        except Exception as e:
            logger.error(f"Error extracting interaction features: {e}")
            extraction_time = (datetime.now() - extraction_start).total_seconds()

            return FeatureExtractionResult(
                conversation_id=conversation.get('id', 'unknown'),
                features={},
                feature_definitions=self.feature_definitions,
                extraction_time=extraction_time,
                errors=[str(e)]
            )

    def _extract_message_structure(self, conversation: Dict[str, Any]) -> List[Dict]:
        """Extract structured message information"""
        messages = conversation.get('messages', [])
        participants = conversation.get('participants', [])

        if not messages:
            return []

        message_structure = []

        for i, message in enumerate(messages):
            # Determine participant for this message
            if isinstance(message, dict) and 'participant' in message:
                participant = message['participant']
                content = message.get('content', '')
            elif isinstance(message, str):
                # Alternate participants for simple string messages
                participant = participants[i % len(participants)] if participants else 'unknown'
                content = message
            else:
                continue

            message_structure.append({
                'index': i,
                'participant': participant,
                'content': content,
                'timestamp': self._estimate_timestamp(i, conversation)
            })

        return message_structure

    def _estimate_timestamp(self, message_index: int, conversation: Dict[str, Any]) -> datetime:
        """Estimate timestamp for message"""
        base_timestamp = conversation.get('timestamp')
        if base_timestamp:
            try:
                base_dt = datetime.fromisoformat(base_timestamp.replace('Z', '+00:00'))
                return base_dt + timedelta(minutes=message_index * 2)  # Assume 2 minutes per message
            except ValueError:
                pass

        return datetime.now() + timedelta(minutes=message_index * 2)

    def _calculate_turn_balance(self, message_structure: List[Dict]) -> float:
        """Calculate balance of conversation turns"""
        if not message_structure:
            return 0.5

        participant_counts = Counter(msg['participant'] for msg in message_structure)
        counts = list(participant_counts.values())

        if len(counts) <= 1:
            return 1.0  # Perfect balance with one participant

        # Calculate Gini coefficient for inequality
        counts_sorted = sorted(counts)
        n = len(counts_sorted)
        cumsum = np.cumsum(counts_sorted)
        gini = (2 * np.sum((np.arange(1, n + 1) * counts_sorted)) / (n * cumsum[-1])) - (n + 1) / n

        return 1.0 - gini  # Convert to balance score (1 = perfect balance)

    def _analyze_response_times(self, message_structure: List[Dict]) -> Dict[str, float]:
        """Analyze response times between messages"""
        if len(message_structure) < 2:
            return {"response_time_avg": 0.0, "response_time_std": 0.0}

        response_times = []
        for i in range(1, len(message_structure)):
            time_diff = (message_structure[i]['timestamp'] - message_structure[i-1]['timestamp']).total_seconds()
            if 0 < time_diff < 3600:  # Reasonable response times
                response_times.append(time_diff)

        if not response_times:
            return {"response_time_avg": 0.0, "response_time_std": 0.0}

        return {
            "response_time_avg": np.mean(response_times),
            "response_time_std": np.std(response_times)
        }

    def _count_interruptions(self, message_structure: List[Dict]) -> int:
        """Count conversation interruptions"""
        if len(message_structure) < 2:
            return 0

        interruptions = 0
        for i in range(1, len(message_structure)):
            current_participant = message_structure[i]['participant']
            previous_participant = message_structure[i-1]['participant']

            # Quick back-and-forth between same participants might indicate interruption
            if current_participant != previous_participant:
                # Check if this looks like an interruption (short response, incomplete thought)
                prev_content = message_structure[i-1]['content'].lower()
                curr_content = message_structure[i]['content'].lower()

                if (len(curr_content.split()) < 5 and
                    any(word in prev_content for word in ['but', 'however', 'although'])):
                    interruptions += 1

        return interruptions

    def _calculate_topic_shifts(self, message_structure: List[Dict]) -> float:
        """Calculate frequency of topic changes"""
        if len(message_structure) < 2:
            return 0.0

        # Simple topic shift detection based on keyword changes
        topic_keywords = ['project', 'code', 'bug', 'feature', 'test', 'deploy', 'design', 'api']
        topic_changes = 0

        for i in range(1, len(message_structure)):
            prev_content = message_structure[i-1]['content'].lower()
            curr_content = message_structure[i]['content'].lower()

            prev_topics = [kw for kw in topic_keywords if kw in prev_content]
            curr_topics = [kw for kw in topic_keywords if kw in curr_content]

            if prev_topics and curr_topics and not set(prev_topics) & set(curr_topics):
                topic_changes += 1

        return topic_changes / (len(message_structure) - 1)

    def _calculate_collaboration_index(self, message_structure: List[Dict]) -> float:
        """Calculate collaboration index based on interaction patterns"""
        if not message_structure:
            return 0.5

        collaboration_indicators = 0
        total_messages = len(message_structure)

        # Look for collaborative language patterns
        collaborative_phrases = [
            'let\'s', 'we should', 'how about', 'what do you think',
            'together', 'collaborate', 'team', 'partnership'
        ]

        for msg in message_structure:
            content = msg['content'].lower()
            if any(phrase in content for phrase in collaborative_phrases):
                collaboration_indicators += 1

        return collaboration_indicators / total_messages

    def _calculate_dominance_ratio(self, message_structure: List[Dict]) -> float:
        """Calculate dominance ratio of most active participant"""
        if not message_structure:
            return 0.5

        participant_counts = Counter(msg['participant'] for msg in message_structure)
        if not participant_counts:
            return 0.5

        max_count = max(participant_counts.values())
        total_messages = len(message_structure)

        return max_count / total_messages

    def _calculate_backchannel_frequency(self, message_structure: List[Dict]) -> float:
        """Calculate frequency of backchannel responses"""
        if len(message_structure) < 2:
            return 0.0

        backchannel_responses = 0
        total_responses = len(message_structure) - 1

        backchannel_indicators = [
            'yes', 'yeah', 'okay', 'got it', 'right', 'sure',
            'i see', 'understood', 'makes sense', 'agree'
        ]

        for i in range(1, len(message_structure)):
            content = message_structure[i]['content'].lower()
            if any(indicator in content for indicator in backchannel_indicators):
                # Check if it's a short response (likely backchannel)
                if len(content.split()) <= 3:
                    backchannel_responses += 1

        return backchannel_responses / total_responses if total_responses > 0 else 0.0

    def get_feature_definitions(self) -> Dict[str, FeatureDefinition]:
        """Get interaction feature definitions"""
        return self.feature_definitions


class SentimentFeatureExtractor(FeatureExtractor):
    """Simple sentiment feature extraction (rule-based)"""

    def __init__(self):
        super().__init__("sentiment_extractor")
        self._initialize_feature_definitions()
        self._initialize_sentiment_lexicons()

    def _initialize_feature_definitions(self):
        """Initialize sentiment feature definitions"""
        self.feature_definitions = {
            "sentiment_positive": FeatureDefinition(
                name="sentiment_positive",
                feature_type=FeatureType.SENTIMENT,
                extraction_method=ExtractionMethod.RULE_BASED,
                description="Positive sentiment score",
                data_type="float",
                range_min=0.0,
                range_max=1.0,
                importance_score=0.8
            ),
            "sentiment_negative": FeatureDefinition(
                name="sentiment_negative",
                feature_type=FeatureType.SENTIMENT,
                extraction_method=ExtractionMethod.RULE_BASED,
                description="Negative sentiment score",
                data_type="float",
                range_min=0.0,
                range_max=1.0,
                importance_score=0.8
            ),
            "sentiment_neutral": FeatureDefinition(
                name="sentiment_neutral",
                feature_type=FeatureType.SENTIMENT,
                extraction_method=ExtractionMethod.RULE_BASED,
                description="Neutral sentiment score",
                data_type="float",
                range_min=0.0,
                range_max=1.0,
                importance_score=0.6
            ),
            "sentiment_compound": FeatureDefinition(
                name="sentiment_compound",
                feature_type=FeatureType.SENTIMENT,
                extraction_method=ExtractionMethod.RULE_BASED,
                description="Compound sentiment score (-1 to 1)",
                data_type="float",
                range_min=-1.0,
                range_max=1.0,
                importance_score=0.9
            ),
            "emotion_frustration": FeatureDefinition(
                name="emotion_frustration",
                feature_type=FeatureType.SENTIMENT,
                extraction_method=ExtractionMethod.RULE_BASED,
                description="Frustration emotion score",
                data_type="float",
                range_min=0.0,
                range_max=1.0,
                importance_score=0.7
            ),
            "emotion_satisfaction": FeatureDefinition(
                name="emotion_satisfaction",
                feature_type=FeatureType.SENTIMENT,
                extraction_method=ExtractionMethod.RULE_BASED,
                description="Satisfaction emotion score",
                data_type="float",
                range_min=0.0,
                range_max=1.0,
                importance_score=0.7
            )
        }

    def _initialize_sentiment_lexicons(self):
        """Initialize simple sentiment lexicons"""
        self.positive_words = {
            'good', 'great', 'excellent', 'awesome', 'wonderful', 'fantastic',
            'amazing', 'perfect', 'love', 'like', 'happy', 'satisfied',
            'success', 'working', 'fixed', 'solved', 'helpful', 'thanks'
        }

        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike',
            'angry', 'frustrated', 'stuck', 'broken', 'error', 'fail',
            'problem', 'issue', 'difficult', 'confused', 'wrong'
        }

        self.frustration_words = {
            'stuck', 'frustrated', 'annoying', 'difficult', 'hard',
            'impossible', 'broken', 'error', 'fail', 'wrong', 'confused'
        }

        self.satisfaction_words = {
            'satisfied', 'happy', 'pleased', 'good', 'great', 'excellent',
            'working', 'fixed', 'solved', 'success', 'helpful', 'perfect'
        }

    async def extract_features(self, conversation: Dict[str, Any]) -> FeatureExtractionResult:
        """Extract sentiment features from conversation"""
        extraction_start = datetime.now()

        try:
            # Extract all text
            all_text = self._extract_all_text(conversation)
            words = all_text.lower().split()

            if not words:
                features = {
                    "sentiment_positive": 0.0,
                    "sentiment_negative": 0.0,
                    "sentiment_neutral": 1.0,
                    "sentiment_compound": 0.0,
                    "emotion_frustration": 0.0,
                    "emotion_satisfaction": 0.0
                }
            else:
                # Calculate sentiment scores
                positive_count = sum(1 for word in words if word in self.positive_words)
                negative_count = sum(1 for word in words if word in self.negative_words)
                frustration_count = sum(1 for word in words if word in self.frustration_words)
                satisfaction_count = sum(1 for word in words if word in self.satisfaction_words)

                total_sentiment_words = positive_count + negative_count
                total_words = len(words)

                features = {
                    "sentiment_positive": positive_count / total_words,
                    "sentiment_negative": negative_count / total_words,
                    "sentiment_neutral": (total_words - total_sentiment_words) / total_words,
                    "sentiment_compound": (positive_count - negative_count) / total_words,
                    "emotion_frustration": frustration_count / total_words,
                    "emotion_satisfaction": satisfaction_count / total_words
                }

            extraction_time = (datetime.now() - extraction_start).total_seconds()

            return FeatureExtractionResult(
                conversation_id=conversation.get('id', 'unknown'),
                features=features,
                feature_definitions=self.feature_definitions,
                extraction_time=extraction_time,
                metadata={
                    'extractor_id': self.extractor_id,
                    'total_words': len(words),
                    'sentiment_words': sum(1 for word in words if word in self.positive_words | self.negative_words)
                }
            )

        except Exception as e:
            logger.error(f"Error extracting sentiment features: {e}")
            extraction_time = (datetime.now() - extraction_start).total_seconds()

            return FeatureExtractionResult(
                conversation_id=conversation.get('id', 'unknown'),
                features={},
                feature_definitions=self.feature_definitions,
                extraction_time=extraction_time,
                errors=[str(e)]
            )

    def _extract_all_text(self, conversation: Dict[str, Any]) -> str:
        """Extract all text from conversation"""
        messages = conversation.get('messages', [])
        if isinstance(messages, list):
            if all(isinstance(msg, str) for msg in messages):
                return ' '.join(messages)
            elif all(isinstance(msg, dict) and 'content' in msg for msg in messages):
                return ' '.join(msg['content'] for msg in messages)
        return ""

    def get_feature_definitions(self) -> Dict[str, FeatureDefinition]:
        """Get sentiment feature definitions"""
        return self.feature_definitions


class ComprehensiveFeatureExtractor:
    """Comprehensive feature extraction combining multiple extractors"""

    def __init__(self):
        self.extractors = {
            'linguistic': LinguisticFeatureExtractor(),
            'interaction': InteractionFeatureExtractor(),
            'sentiment': SentimentFeatureExtractor()
        }

    async def extract_all_features(self, conversation: Dict[str, Any]) -> FeatureExtractionResult:
        """Extract all features using all extractors"""
        extraction_start = datetime.now()
        all_features = {}
        all_definitions = {}
        all_metadata = {}
        all_errors = []

        # Extract features from each extractor
        for extractor_name, extractor in self.extractors.items():
            try:
                result = await extractor.extract_features(conversation)
                all_features.update(result.features)
                all_definitions.update(result.feature_definitions)
                all_metadata[extractor_name] = result.metadata
                all_errors.extend(result.errors or [])
            except Exception as e:
                logger.error(f"Error in {extractor_name} extractor: {e}")
                all_errors.append(f"{extractor_name}: {str(e)}")

        extraction_time = (datetime.now() - extraction_start).total_seconds()

        return FeatureExtractionResult(
            conversation_id=conversation.get('id', 'unknown'),
            features=all_features,
            feature_definitions=all_definitions,
            extraction_time=extraction_time,
            metadata={
                'comprehensive_extractor': True,
                'extractor_metadata': all_metadata,
                'total_extractors': len(self.extractors),
                'successful_extractors': len(self.extractors) - len([e for e in all_errors if e])
            },
            errors=all_errors
        )

    async def extract_batch_features(self, conversations: List[Dict[str, Any]]) -> BatchFeatureExtractionResult:
        """Extract features from a batch of conversations"""
        batch_start = datetime.now()
        results = []
        all_errors = []

        # Process conversations in parallel
        tasks = [self.extract_all_features(conv) for conv in conversations]
        try:
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    error_msg = f"Conversation {i}: {str(result)}"
                    all_errors.append(error_msg)
                    logger.error(error_msg)
                else:
                    results.append(result)

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            all_errors.append(f"Batch processing: {str(e)}")

        # Calculate summary statistics
        total_time = (datetime.now() - batch_start).total_seconds()

        if results:
            # Calculate feature coverage
            feature_names = set()
            for result in results:
                feature_names.update(result.features.keys())

            feature_coverage = {}
            for feature_name in feature_names:
                available_count = sum(1 for result in results if feature_name in result.features)
                feature_coverage[feature_name] = available_count / len(results)

            # Calculate average confidence
            avg_confidence = np.mean([
                np.mean(list(result.confidence_scores.values())) if result.confidence_scores else 0.5
                for result in results
            ])

            # Summary statistics
            summary_stats = {
                'total_conversations': len(conversations),
                'successful_extractions': len(results),
                'failed_extractions': len(all_errors),
                'total_features': len(feature_names),
                'avg_features_per_conversation': np.mean([len(result.features) for result in results]),
                'avg_extraction_time': np.mean([result.extraction_time for result in results])
            }
        else:
            feature_coverage = {}
            avg_confidence = 0.0
            summary_stats = {
                'total_conversations': len(conversations),
                'successful_extractions': 0,
                'failed_extractions': len(all_errors),
                'total_features': 0,
                'avg_features_per_conversation': 0,
                'avg_extraction_time': 0
            }

        return BatchFeatureExtractionResult(
            results=results,
            summary_stats=summary_stats,
            total_extraction_time=total_time,
            average_confidence=avg_confidence,
            feature_coverage=feature_coverage,
            errors=all_errors
        )


# Factory functions
def create_feature_extractor(extractor_type: str = "comprehensive") -> FeatureExtractor:
    """Create a feature extractor instance"""
    if extractor_type == "linguistic":
        return LinguisticFeatureExtractor()
    elif extractor_type == "interaction":
        return InteractionFeatureExtractor()
    elif extractor_type == "sentiment":
        return SentimentFeatureExtractor()
    elif extractor_type == "comprehensive":
        return ComprehensiveFeatureExtractor()
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")


# Convenience functions
async def extract_conversation_features(
    conversation: Dict[str, Any],
    extractor_type: str = "comprehensive"
) -> FeatureExtractionResult:
    """Extract features from a single conversation"""
    extractor = create_feature_extractor(extractor_type)

    if isinstance(extractor, ComprehensiveFeatureExtractor):
        return await extractor.extract_all_features(conversation)
    else:
        return await extractor.extract_features(conversation)


async def extract_batch_conversation_features(
    conversations: List[Dict[str, Any]],
    extractor_type: str = "comprehensive"
) -> BatchFeatureExtractionResult:
    """Extract features from a batch of conversations"""
    extractor = create_feature_extractor(extractor_type)

    if isinstance(extractor, ComprehensiveFeatureExtractor):
        return await extractor.extract_batch_features(conversations)
    else:
        # For individual extractors, process manually
        results = []
        errors = []
        for conv in conversations:
            try:
                result = await extractor.extract_features(conv)
                results.append(result)
            except Exception as e:
                errors.append(str(e))
                logger.error(f"Error processing conversation: {e}")

        return BatchFeatureExtractionResult(
            results=results,
            summary_stats={
                'total_conversations': len(conversations),
                'successful_extractions': len(results),
                'failed_extractions': len(errors)
            },
            errors=errors
        )


# Example usage and testing
if __name__ == "__main__":
    async def test_feature_extraction():
        """Test the feature extraction module"""
        # Sample conversation data
        sample_conversations = [
            {
                "id": "conv_001",
                "messages": [
                    "Hello, I need help with my database connection issue.",
                    "Sure, I can help you with that. What's the specific problem?",
                    "The connection keeps timing out when I try to query large datasets."
                ],
                "timestamp": "2025-09-21T10:30:00Z",
                "participants": ["agent_001", "agent_002"]
            },
            {
                "id": "conv_002",
                "messages": [
                    "Great work on the API implementation!",
                    "Thanks! Let's collaborate on the next feature.",
                    "Awesome, I'm excited about this project."
                ],
                "timestamp": "2025-09-21T14:15:00Z",
                "participants": ["agent_003", "agent_004"]
            }
        ]

        # Test single conversation extraction
        print("Testing single conversation feature extraction...")
        result = await extract_conversation_features(sample_conversations[0])
        print(f"Extracted {len(result.features)} features")
        print(f"Extraction time: {result.extraction_time:.3f}s")

        # Test batch extraction
        print("\nTesting batch feature extraction...")
        batch_result = await extract_batch_conversation_features(sample_conversations)
        print(f"Processed {batch_result.summary_stats['successful_extractions']}/{batch_result.summary_stats['total_conversations']} conversations")
        print(f"Total extraction time: {batch_result.total_extraction_time:.3f}s")

        return batch_result

    # Run the test
    asyncio.run(test_feature_extraction())