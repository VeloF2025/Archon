"""
🚀 ARCHON ENHANCEMENT 2025 - PHASE 6: ADVANCED AI INTEGRATION
Multi-Modal AI Pipeline - Unified Intelligence Processing System

This module provides a comprehensive multi-modal AI pipeline that processes and integrates
various data types (text, images, audio, video) through unified intelligence models.
"""

import asyncio
import logging
import base64
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
import io
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Supported modality types in the pipeline."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TABULAR = "tabular"
    GRAPH = "graph"
    TIME_SERIES = "time_series"


class ProcessingStatus(Enum):
    """Processing status for multi-modal inputs."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class FusionStrategy(Enum):
    """Strategies for fusing multi-modal representations."""
    EARLY_FUSION = "early_fusion"
    LATE_FUSION = "late_fusion"
    HIERARCHICAL_FUSION = "hierarchical_fusion"
    ATTENTION_FUSION = "attention_fusion"
    CROSS_MODAL_ATTENTION = "cross_modal_attention"


@dataclass
class ModalInput:
    """Input data for a specific modality."""
    modality: ModalityType
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_config: Dict[str, Any] = field(default_factory=dict)
    content_hash: Optional[str] = None
    
    def __post_init__(self):
        if self.content_hash is None:
            self.content_hash = self._compute_content_hash()
    
    def _compute_content_hash(self) -> str:
        """Compute hash of input content for caching."""
        content_str = str(self.data) + str(self.metadata) + str(self.processing_config)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]


@dataclass
class ModalOutput:
    """Output representation for a processed modality."""
    modality: ModalityType
    embedding: np.ndarray
    features: Dict[str, Any]
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiModalResult:
    """Result of multi-modal processing."""
    input_hash: str
    modal_outputs: List[ModalOutput]
    fused_embedding: np.ndarray
    fusion_strategy: FusionStrategy
    overall_confidence: float
    processing_summary: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class BaseModalProcessor(ABC):
    """Abstract base class for modality-specific processors."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the processor with required models/resources."""
        pass
    
    @abstractmethod
    async def process(self, modal_input: ModalInput) -> ModalOutput:
        """Process input data and return modal representation."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Return the dimension of embeddings produced by this processor."""
        pass


class TextProcessor(BaseModalProcessor):
    """Advanced text processing with embeddings and NLP features."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embedding_model = None
        self.nlp_pipeline = None
        self.embedding_dim = config.get('embedding_dim', 768)
    
    async def initialize(self) -> None:
        """Initialize text processing models."""
        logger.info("Initializing text processor...")
        # Simulate model loading
        await asyncio.sleep(0.1)
        self.is_initialized = True
        logger.info("Text processor initialized")
    
    async def process(self, modal_input: ModalInput) -> ModalOutput:
        """Process text input and extract features."""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = asyncio.get_event_loop().time()
        text_data = modal_input.data
        
        # Text preprocessing
        cleaned_text = self._preprocess_text(text_data)
        
        # Generate embeddings (simulated)
        embedding = await self._generate_text_embedding(cleaned_text)
        
        # Extract NLP features
        features = await self._extract_nlp_features(cleaned_text)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        confidence = self._compute_confidence(features)
        
        return ModalOutput(
            modality=ModalityType.TEXT,
            embedding=embedding,
            features=features,
            confidence_score=confidence,
            processing_time=processing_time,
            metadata={
                'text_length': len(text_data),
                'cleaned_length': len(cleaned_text),
                'language_detected': features.get('language', 'unknown')
            }
        )
    
    def get_embedding_dimension(self) -> int:
        return self.embedding_dim
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text data."""
        # Basic text cleaning
        cleaned = text.strip().lower()
        return cleaned[:self.config.get('max_text_length', 10000)]
    
    async def _generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate text embeddings."""
        # Simulate embedding generation
        await asyncio.sleep(0.05)
        # Return normalized random embedding for demonstration
        embedding = np.random.normal(0, 1, self.embedding_dim)
        return embedding / np.linalg.norm(embedding)
    
    async def _extract_nlp_features(self, text: str) -> Dict[str, Any]:
        """Extract NLP features from text."""
        return {
            'word_count': len(text.split()),
            'char_count': len(text),
            'sentiment_score': np.random.uniform(-1, 1),
            'language': 'en',
            'readability_score': np.random.uniform(0, 100),
            'topics': ['technology', 'ai'] if 'ai' in text.lower() else ['general']
        }
    
    def _compute_confidence(self, features: Dict[str, Any]) -> float:
        """Compute confidence score for text processing."""
        word_count = features.get('word_count', 0)
        if word_count < 5:
            return 0.3
        elif word_count < 20:
            return 0.7
        else:
            return 0.95


class ImageProcessor(BaseModalProcessor):
    """Advanced image processing with computer vision."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.vision_model = None
        self.embedding_dim = config.get('embedding_dim', 2048)
    
    async def initialize(self) -> None:
        """Initialize image processing models."""
        logger.info("Initializing image processor...")
        await asyncio.sleep(0.1)
        self.is_initialized = True
        logger.info("Image processor initialized")
    
    async def process(self, modal_input: ModalInput) -> ModalOutput:
        """Process image input and extract visual features."""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = asyncio.get_event_loop().time()
        image_data = modal_input.data
        
        # Image preprocessing
        processed_image = await self._preprocess_image(image_data)
        
        # Generate visual embeddings
        embedding = await self._generate_visual_embedding(processed_image)
        
        # Extract visual features
        features = await self._extract_visual_features(processed_image)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        confidence = self._compute_visual_confidence(features)
        
        return ModalOutput(
            modality=ModalityType.IMAGE,
            embedding=embedding,
            features=features,
            confidence_score=confidence,
            processing_time=processing_time,
            metadata={
                'image_format': modal_input.metadata.get('format', 'unknown'),
                'resolution': features.get('resolution', (0, 0))
            }
        )
    
    def get_embedding_dimension(self) -> int:
        return self.embedding_dim
    
    async def _preprocess_image(self, image_data: Any) -> np.ndarray:
        """Preprocess image data."""
        # Simulate image preprocessing
        await asyncio.sleep(0.02)
        # Return simulated preprocessed image
        return np.random.random((224, 224, 3))
    
    async def _generate_visual_embedding(self, image: np.ndarray) -> np.ndarray:
        """Generate visual embeddings from image."""
        await asyncio.sleep(0.1)
        embedding = np.random.normal(0, 1, self.embedding_dim)
        return embedding / np.linalg.norm(embedding)
    
    async def _extract_visual_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract visual features from image."""
        return {
            'resolution': image.shape[:2],
            'dominant_colors': ['blue', 'red', 'green'],
            'detected_objects': ['person', 'car'] if np.random.random() > 0.5 else ['building'],
            'scene_category': 'outdoor' if np.random.random() > 0.5 else 'indoor',
            'quality_score': np.random.uniform(0.5, 1.0),
            'brightness': np.mean(image),
            'contrast': np.std(image)
        }
    
    def _compute_visual_confidence(self, features: Dict[str, Any]) -> float:
        """Compute confidence score for image processing."""
        quality_score = features.get('quality_score', 0.5)
        num_objects = len(features.get('detected_objects', []))
        return min(0.95, quality_score * 0.7 + (num_objects * 0.1))


class AudioProcessor(BaseModalProcessor):
    """Advanced audio processing with speech and acoustic analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.audio_model = None
        self.speech_recognizer = None
        self.embedding_dim = config.get('embedding_dim', 512)
    
    async def initialize(self) -> None:
        """Initialize audio processing models."""
        logger.info("Initializing audio processor...")
        await asyncio.sleep(0.1)
        self.is_initialized = True
        logger.info("Audio processor initialized")
    
    async def process(self, modal_input: ModalInput) -> ModalOutput:
        """Process audio input and extract acoustic features."""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = asyncio.get_event_loop().time()
        audio_data = modal_input.data
        
        # Audio preprocessing
        processed_audio = await self._preprocess_audio(audio_data)
        
        # Generate audio embeddings
        embedding = await self._generate_audio_embedding(processed_audio)
        
        # Extract acoustic features
        features = await self._extract_acoustic_features(processed_audio)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        confidence = self._compute_audio_confidence(features)
        
        return ModalOutput(
            modality=ModalityType.AUDIO,
            embedding=embedding,
            features=features,
            confidence_score=confidence,
            processing_time=processing_time,
            metadata={
                'duration': features.get('duration', 0),
                'sample_rate': modal_input.metadata.get('sample_rate', 44100)
            }
        )
    
    def get_embedding_dimension(self) -> int:
        return self.embedding_dim
    
    async def _preprocess_audio(self, audio_data: Any) -> np.ndarray:
        """Preprocess audio data."""
        await asyncio.sleep(0.03)
        # Return simulated preprocessed audio
        return np.random.random(16000)  # 1 second at 16kHz
    
    async def _generate_audio_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Generate audio embeddings."""
        await asyncio.sleep(0.05)
        embedding = np.random.normal(0, 1, self.embedding_dim)
        return embedding / np.linalg.norm(embedding)
    
    async def _extract_acoustic_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """Extract acoustic features from audio."""
        return {
            'duration': len(audio) / 16000,  # Assuming 16kHz sample rate
            'average_volume': np.mean(np.abs(audio)),
            'peak_frequency': np.random.uniform(100, 8000),
            'speech_detected': np.random.random() > 0.3,
            'transcription': 'Hello world' if np.random.random() > 0.5 else '',
            'speaker_count': np.random.randint(1, 4),
            'background_noise_level': np.random.uniform(0, 0.3)
        }
    
    def _compute_audio_confidence(self, features: Dict[str, Any]) -> float:
        """Compute confidence score for audio processing."""
        volume = features.get('average_volume', 0)
        noise_level = features.get('background_noise_level', 1)
        return min(0.95, (volume * 2) - noise_level)


class MultiModalFuser:
    """Fusion engine for combining multi-modal representations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fusion_strategies = {
            FusionStrategy.EARLY_FUSION: self._early_fusion,
            FusionStrategy.LATE_FUSION: self._late_fusion,
            FusionStrategy.HIERARCHICAL_FUSION: self._hierarchical_fusion,
            FusionStrategy.ATTENTION_FUSION: self._attention_fusion,
            FusionStrategy.CROSS_MODAL_ATTENTION: self._cross_modal_attention
        }
    
    async def fuse_modalities(
        self, 
        modal_outputs: List[ModalOutput], 
        strategy: FusionStrategy
    ) -> Tuple[np.ndarray, float]:
        """Fuse multiple modal representations into unified embedding."""
        if not modal_outputs:
            raise ValueError("No modal outputs provided for fusion")
        
        fusion_func = self.fusion_strategies.get(strategy)
        if not fusion_func:
            raise ValueError(f"Unsupported fusion strategy: {strategy}")
        
        fused_embedding, confidence = await fusion_func(modal_outputs)
        return fused_embedding, confidence
    
    async def _early_fusion(self, modal_outputs: List[ModalOutput]) -> Tuple[np.ndarray, float]:
        """Concatenate embeddings at feature level."""
        embeddings = [output.embedding for output in modal_outputs]
        confidences = [output.confidence_score for output in modal_outputs]
        
        # Concatenate embeddings
        fused_embedding = np.concatenate(embeddings)
        
        # Weighted average confidence
        weights = np.array([len(emb) for emb in embeddings])
        weights = weights / weights.sum()
        overall_confidence = np.average(confidences, weights=weights)
        
        return fused_embedding, float(overall_confidence)
    
    async def _late_fusion(self, modal_outputs: List[ModalOutput]) -> Tuple[np.ndarray, float]:
        """Average embeddings after individual processing."""
        # Normalize all embeddings to same dimension
        max_dim = max(len(output.embedding) for output in modal_outputs)
        normalized_embeddings = []
        
        for output in modal_outputs:
            emb = output.embedding
            if len(emb) < max_dim:
                # Pad with zeros
                padded = np.zeros(max_dim)
                padded[:len(emb)] = emb
                normalized_embeddings.append(padded)
            elif len(emb) > max_dim:
                # Truncate
                normalized_embeddings.append(emb[:max_dim])
            else:
                normalized_embeddings.append(emb)
        
        # Weighted average by confidence
        confidences = [output.confidence_score for output in modal_outputs]
        weights = np.array(confidences) / sum(confidences)
        
        fused_embedding = np.average(normalized_embeddings, weights=weights, axis=0)
        overall_confidence = np.mean(confidences)
        
        return fused_embedding, float(overall_confidence)
    
    async def _hierarchical_fusion(self, modal_outputs: List[ModalOutput]) -> Tuple[np.ndarray, float]:
        """Hierarchical fusion with modality grouping."""
        # Group by modality type for hierarchical processing
        modality_groups = {}
        for output in modal_outputs:
            modality = output.modality
            if modality not in modality_groups:
                modality_groups[modality] = []
            modality_groups[modality].append(output)
        
        # Process each group
        group_embeddings = []
        group_confidences = []
        
        for modality, outputs in modality_groups.items():
            if len(outputs) == 1:
                group_embeddings.append(outputs[0].embedding)
                group_confidences.append(outputs[0].confidence_score)
            else:
                # Fuse within group
                group_emb, group_conf = await self._late_fusion(outputs)
                group_embeddings.append(group_emb)
                group_confidences.append(group_conf)
        
        # Fuse across groups
        final_outputs = [
            ModalOutput(
                modality=list(modality_groups.keys())[i],
                embedding=emb,
                features={},
                confidence_score=conf,
                processing_time=0
            )
            for i, (emb, conf) in enumerate(zip(group_embeddings, group_confidences))
        ]
        
        return await self._late_fusion(final_outputs)
    
    async def _attention_fusion(self, modal_outputs: List[ModalOutput]) -> Tuple[np.ndarray, float]:
        """Attention-based fusion with learned weights."""
        embeddings = [output.embedding for output in modal_outputs]
        confidences = [output.confidence_score for output in modal_outputs]
        
        # Compute attention weights based on confidence and embedding magnitude
        attention_scores = []
        for emb, conf in zip(embeddings, confidences):
            # Attention score combines confidence and embedding energy
            energy = np.linalg.norm(emb)
            attention_score = conf * energy
            attention_scores.append(attention_score)
        
        # Softmax normalization
        attention_scores = np.array(attention_scores)
        attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores))
        
        # Apply attention to create fused representation
        max_dim = max(len(emb) for emb in embeddings)
        attended_embeddings = []
        
        for i, (emb, weight) in enumerate(zip(embeddings, attention_weights)):
            # Normalize dimension and apply attention
            if len(emb) != max_dim:
                normalized = np.zeros(max_dim)
                normalized[:len(emb)] = emb[:max_dim] if len(emb) > max_dim else emb
            else:
                normalized = emb
            
            attended_embeddings.append(normalized * weight)
        
        fused_embedding = np.sum(attended_embeddings, axis=0)
        overall_confidence = np.average(confidences, weights=attention_weights)
        
        return fused_embedding, float(overall_confidence)
    
    async def _cross_modal_attention(self, modal_outputs: List[ModalOutput]) -> Tuple[np.ndarray, float]:
        """Cross-modal attention for inter-modality interactions."""
        if len(modal_outputs) < 2:
            return modal_outputs[0].embedding, modal_outputs[0].confidence_score
        
        embeddings = [output.embedding for output in modal_outputs]
        confidences = [output.confidence_score for output in modal_outputs]
        
        # Normalize all embeddings to same dimension
        max_dim = max(len(emb) for emb in embeddings)
        normalized_embeddings = []
        
        for emb in embeddings:
            if len(emb) != max_dim:
                normalized = np.zeros(max_dim)
                normalized[:len(emb)] = emb[:max_dim] if len(emb) > max_dim else emb
            else:
                normalized = emb
            normalized_embeddings.append(normalized)
        
        # Cross-modal attention matrix
        embeddings_matrix = np.array(normalized_embeddings)
        attention_matrix = np.dot(embeddings_matrix, embeddings_matrix.T)
        
        # Apply softmax row-wise
        for i in range(len(attention_matrix)):
            attention_matrix[i] = np.exp(attention_matrix[i]) / np.sum(np.exp(attention_matrix[i]))
        
        # Compute cross-attended representations
        attended_embeddings = np.dot(attention_matrix, embeddings_matrix)
        
        # Final fusion with original embeddings
        alpha = 0.3  # Balance between original and cross-attended
        final_embeddings = alpha * embeddings_matrix + (1 - alpha) * attended_embeddings
        
        # Aggregate to single representation
        confidence_weights = np.array(confidences) / sum(confidences)
        fused_embedding = np.average(final_embeddings, weights=confidence_weights, axis=0)
        overall_confidence = np.mean(confidences)
        
        return fused_embedding, float(overall_confidence)


class MultiModalAIPipeline:
    """Main orchestrator for multi-modal AI processing pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processors: Dict[ModalityType, BaseModalProcessor] = {}
        self.fuser = MultiModalFuser(config.get('fusion', {}))
        self.cache: Dict[str, MultiModalResult] = {}
        self.max_cache_size = config.get('max_cache_size', 1000)
        self.processing_timeout = config.get('processing_timeout', 30.0)
        self.background_tasks: Dict[str, asyncio.Task] = {}
        
        # Initialize processors
        self._initialize_processors()
    
    def _initialize_processors(self):
        """Initialize modality-specific processors."""
        processor_configs = self.config.get('processors', {})
        
        # Text processor
        if ModalityType.TEXT not in self.processors:
            text_config = processor_configs.get('text', {})
            self.processors[ModalityType.TEXT] = TextProcessor(text_config)
        
        # Image processor  
        if ModalityType.IMAGE not in self.processors:
            image_config = processor_configs.get('image', {})
            self.processors[ModalityType.IMAGE] = ImageProcessor(image_config)
        
        # Audio processor
        if ModalityType.AUDIO not in self.processors:
            audio_config = processor_configs.get('audio', {})
            self.processors[ModalityType.AUDIO] = AudioProcessor(audio_config)
    
    async def initialize(self) -> None:
        """Initialize all processors."""
        logger.info("Initializing multi-modal AI pipeline...")
        
        initialization_tasks = []
        for processor in self.processors.values():
            if not processor.is_initialized:
                initialization_tasks.append(processor.initialize())
        
        if initialization_tasks:
            await asyncio.gather(*initialization_tasks)
        
        logger.info("Multi-modal AI pipeline initialized successfully")
    
    async def process_multimodal_input(
        self, 
        inputs: List[ModalInput], 
        fusion_strategy: FusionStrategy = FusionStrategy.LATE_FUSION
    ) -> MultiModalResult:
        """Process multi-modal input and return unified result."""
        if not inputs:
            raise ValueError("No inputs provided for processing")
        
        # Generate input hash for caching
        input_hash = self._compute_input_hash(inputs)
        
        # Check cache
        if input_hash in self.cache:
            logger.info(f"Returning cached result for input hash: {input_hash}")
            return self.cache[input_hash]
        
        # Process each modality
        modal_outputs = []
        processing_tasks = []
        
        for modal_input in inputs:
            processor = self.processors.get(modal_input.modality)
            if processor:
                task = asyncio.create_task(
                    self._process_with_timeout(processor, modal_input)
                )
                processing_tasks.append(task)
            else:
                logger.warning(f"No processor available for modality: {modal_input.modality}")
        
        # Wait for all processing to complete
        try:
            modal_outputs = await asyncio.gather(*processing_tasks, return_exceptions=True)
            
            # Filter out exceptions and log errors
            valid_outputs = []
            for i, output in enumerate(modal_outputs):
                if isinstance(output, Exception):
                    logger.error(f"Processing failed for input {i}: {output}")
                else:
                    valid_outputs.append(output)
            
            modal_outputs = valid_outputs
            
        except Exception as e:
            logger.error(f"Error during modal processing: {e}")
            raise
        
        if not modal_outputs:
            raise RuntimeError("All modal processing failed")
        
        # Fuse modalities
        fused_embedding, overall_confidence = await self.fuser.fuse_modalities(
            modal_outputs, fusion_strategy
        )
        
        # Create result
        result = MultiModalResult(
            input_hash=input_hash,
            modal_outputs=modal_outputs,
            fused_embedding=fused_embedding,
            fusion_strategy=fusion_strategy,
            overall_confidence=overall_confidence,
            processing_summary={
                'num_inputs': len(inputs),
                'num_successful_outputs': len(modal_outputs),
                'modalities_processed': [output.modality.value for output in modal_outputs],
                'total_processing_time': sum(output.processing_time for output in modal_outputs),
                'fusion_strategy': fusion_strategy.value
            }
        )
        
        # Cache result
        self._cache_result(input_hash, result)
        
        return result
    
    async def _process_with_timeout(
        self, 
        processor: BaseModalProcessor, 
        modal_input: ModalInput
    ) -> ModalOutput:
        """Process modal input with timeout."""
        try:
            return await asyncio.wait_for(
                processor.process(modal_input), 
                timeout=self.processing_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Processing timeout for {modal_input.modality}")
            # Return minimal output with low confidence
            return ModalOutput(
                modality=modal_input.modality,
                embedding=np.zeros(processor.get_embedding_dimension()),
                features={'error': 'timeout'},
                confidence_score=0.0,
                processing_time=self.processing_timeout
            )
    
    def _compute_input_hash(self, inputs: List[ModalInput]) -> str:
        """Compute hash for input combination."""
        hash_components = []
        for modal_input in sorted(inputs, key=lambda x: x.modality.value):
            hash_components.append(modal_input.content_hash)
        
        combined_hash = ''.join(hash_components)
        return hashlib.sha256(combined_hash.encode()).hexdigest()[:16]
    
    def _cache_result(self, input_hash: str, result: MultiModalResult):
        """Cache processing result with size limit."""
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entries (simple LRU)
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
            del self.cache[oldest_key]
        
        self.cache[input_hash] = result
    
    async def process_batch(
        self, 
        batch_inputs: List[List[ModalInput]], 
        fusion_strategy: FusionStrategy = FusionStrategy.LATE_FUSION,
        max_concurrent: int = 5
    ) -> List[MultiModalResult]:
        """Process batch of multi-modal inputs concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(inputs):
            async with semaphore:
                return await self.process_multimodal_input(inputs, fusion_strategy)
        
        tasks = [process_single(inputs) for inputs in batch_inputs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed for item {i}: {result}")
                # Create error result
                error_result = MultiModalResult(
                    input_hash=f"error_{i}",
                    modal_outputs=[],
                    fused_embedding=np.array([]),
                    fusion_strategy=fusion_strategy,
                    overall_confidence=0.0,
                    processing_summary={'error': str(result)}
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def stream_process(
        self, 
        input_stream: AsyncGenerator[List[ModalInput], None],
        fusion_strategy: FusionStrategy = FusionStrategy.LATE_FUSION
    ) -> AsyncGenerator[MultiModalResult, None]:
        """Process streaming multi-modal inputs."""
        async for inputs in input_stream:
            try:
                result = await self.process_multimodal_input(inputs, fusion_strategy)
                yield result
            except Exception as e:
                logger.error(f"Stream processing error: {e}")
                # Yield error result
                error_result = MultiModalResult(
                    input_hash="stream_error",
                    modal_outputs=[],
                    fused_embedding=np.array([]),
                    fusion_strategy=fusion_strategy,
                    overall_confidence=0.0,
                    processing_summary={'error': str(e)}
                )
                yield error_result
    
    def get_processor_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about processors."""
        stats = {}
        for modality, processor in self.processors.items():
            stats[modality.value] = {
                'is_initialized': processor.is_initialized,
                'embedding_dimension': processor.get_embedding_dimension(),
                'config': processor.config
            }
        return stats
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.max_cache_size,
            'cache_hit_keys': list(self.cache.keys())
        }
    
    async def clear_cache(self) -> None:
        """Clear the result cache."""
        self.cache.clear()
        logger.info("Multi-modal pipeline cache cleared")
    
    async def shutdown(self) -> None:
        """Shutdown the pipeline and cleanup resources."""
        logger.info("Shutting down multi-modal AI pipeline...")
        
        # Cancel background tasks
        for task in self.background_tasks.values():
            task.cancel()
        
        # Wait for cancellation
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks.values(), return_exceptions=True)
        
        # Clear cache
        await self.clear_cache()
        
        logger.info("Multi-modal AI pipeline shutdown complete")


# Example usage and testing functions
async def example_multimodal_processing():
    """Example of multi-modal processing."""
    # Pipeline configuration
    config = {
        'processors': {
            'text': {'embedding_dim': 768, 'max_text_length': 5000},
            'image': {'embedding_dim': 2048},
            'audio': {'embedding_dim': 512}
        },
        'fusion': {},
        'max_cache_size': 100,
        'processing_timeout': 10.0
    }
    
    # Initialize pipeline
    pipeline = MultiModalAIPipeline(config)
    await pipeline.initialize()
    
    # Create sample inputs
    text_input = ModalInput(
        modality=ModalityType.TEXT,
        data="This is a sample text about artificial intelligence and machine learning.",
        metadata={'language': 'en'}
    )
    
    image_input = ModalInput(
        modality=ModalityType.IMAGE,
        data="base64_encoded_image_data",
        metadata={'format': 'jpeg', 'width': 224, 'height': 224}
    )
    
    audio_input = ModalInput(
        modality=ModalityType.AUDIO,
        data="audio_waveform_data",
        metadata={'sample_rate': 16000, 'duration': 5.0}
    )
    
    # Process multi-modal input
    inputs = [text_input, image_input, audio_input]
    result = await pipeline.process_multimodal_input(
        inputs, 
        fusion_strategy=FusionStrategy.ATTENTION_FUSION
    )
    
    logger.info(f"Processing complete. Overall confidence: {result.overall_confidence:.3f}")
    logger.info(f"Fused embedding dimension: {len(result.fused_embedding)}")
    logger.info(f"Processing summary: {result.processing_summary}")
    
    # Test batch processing
    batch_inputs = [inputs, inputs[:2], [text_input]]  # Different combinations
    batch_results = await pipeline.process_batch(
        batch_inputs, 
        max_concurrent=2
    )
    
    logger.info(f"Batch processing complete. Processed {len(batch_results)} items")
    
    # Cleanup
    await pipeline.shutdown()


if __name__ == "__main__":
    asyncio.run(example_multimodal_processing())