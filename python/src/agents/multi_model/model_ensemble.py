"""
Multi-Model Intelligence Fusion - Model Ensemble Architecture
Intelligently routes tasks across multiple AI providers for optimal performance.
"""

import asyncio
import time
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import openai
import anthropic
import google.generativeai as genai
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
import aiohttp

from ..monitoring.metrics import (
    track_agent_execution,
    agent_cost_dollars,
    agent_execution_total
)

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported AI model providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    META = "meta"
    COHERE = "cohere"
    MISTRAL = "mistral"


class TaskType(Enum):
    """Different types of tasks for optimal model selection."""
    CODING = "coding"
    CREATIVE_WRITING = "creative_writing" 
    ANALYSIS = "analysis"
    SIMPLE_QUERY = "simple_query"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    REASONING = "reasoning"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"
    DEBUGGING = "debugging"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: ModelProvider
    model_name: str
    cost_per_1k_tokens: float
    max_tokens: int
    context_window: int
    strengths: List[TaskType]
    performance_score: float = 0.8
    availability: bool = True
    api_key_env: Optional[str] = None
    endpoint_url: Optional[str] = None


@dataclass
class TaskRequest:
    """Request for model execution."""
    prompt: str
    task_type: TaskType
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    cost_budget: Optional[float] = None
    quality_requirement: float = 0.8
    urgency: str = "normal"  # low, normal, high, critical
    system_prompt: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class ModelResponse:
    """Response from a model execution."""
    content: str
    provider: ModelProvider
    model_name: str
    tokens_used: int
    cost: float
    response_time: float
    quality_score: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance tracking for models."""
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    avg_quality_score: float = 0.0
    cost_efficiency: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


class ModelEnsemble:
    """
    Multi-provider model ensemble with intelligent routing.
    """
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.performance_metrics: Dict[str, PerformanceMetrics] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.load_balancer_weights: Dict[str, float] = {}
        
        # Initialize model configurations
        self._setup_model_configs()
        
        # Performance tracking
        self.response_times: Dict[str, List[float]] = {}
        self.quality_scores: Dict[str, List[float]] = {}
        self.cost_tracking: Dict[str, List[float]] = {}
        
        # Circuit breaker pattern for failing models
        self.circuit_breakers: Dict[str, Dict] = {}
        
        # Task type routing preferences
        self.routing_preferences = {
            TaskType.CODING: [
                "anthropic:claude-3-sonnet-20240229",
                "openai:gpt-4-turbo-preview", 
                "anthropic:claude-3-opus-20240229"
            ],
            TaskType.CREATIVE_WRITING: [
                "anthropic:claude-3-opus-20240229",
                "openai:gpt-4-turbo-preview",
                "google:gemini-pro"
            ],
            TaskType.ANALYSIS: [
                "openai:gpt-4-turbo-preview",
                "anthropic:claude-3-sonnet-20240229",
                "google:gemini-pro"
            ],
            TaskType.SIMPLE_QUERY: [
                "anthropic:claude-3-haiku-20240307",
                "openai:gpt-3.5-turbo",
                "google:gemini-pro"
            ],
            TaskType.CODE_REVIEW: [
                "anthropic:claude-3-sonnet-20240229",
                "openai:gpt-4-turbo-preview"
            ],
            TaskType.DOCUMENTATION: [
                "anthropic:claude-3-sonnet-20240229",
                "openai:gpt-4-turbo-preview"
            ]
        }
    
    def _setup_model_configs(self):
        """Initialize all supported model configurations."""
        
        # Anthropic Models
        self.model_configs.update({
            "anthropic:claude-3-opus-20240229": ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-opus-20240229",
                cost_per_1k_tokens=15.0,  # Input cost
                max_tokens=4096,
                context_window=200000,
                strengths=[TaskType.CREATIVE_WRITING, TaskType.REASONING, TaskType.ANALYSIS],
                performance_score=0.95,
                api_key_env="ANTHROPIC_API_KEY"
            ),
            "anthropic:claude-3-sonnet-20240229": ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-sonnet-20240229", 
                cost_per_1k_tokens=3.0,
                max_tokens=4096,
                context_window=200000,
                strengths=[TaskType.CODING, TaskType.ANALYSIS, TaskType.CODE_REVIEW],
                performance_score=0.90,
                api_key_env="ANTHROPIC_API_KEY"
            ),
            "anthropic:claude-3-haiku-20240307": ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-haiku-20240307",
                cost_per_1k_tokens=0.25,
                max_tokens=4096,
                context_window=200000,
                strengths=[TaskType.SIMPLE_QUERY, TaskType.SUMMARIZATION],
                performance_score=0.85,
                api_key_env="ANTHROPIC_API_KEY"
            ),
        })
        
        # OpenAI Models
        self.model_configs.update({
            "openai:gpt-4-turbo-preview": ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4-turbo-preview",
                cost_per_1k_tokens=10.0,
                max_tokens=4096,
                context_window=128000,
                strengths=[TaskType.ANALYSIS, TaskType.REASONING, TaskType.CODING],
                performance_score=0.92,
                api_key_env="OPENAI_API_KEY"
            ),
            "openai:gpt-4": ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4",
                cost_per_1k_tokens=30.0,
                max_tokens=8192,
                context_window=8192,
                strengths=[TaskType.ANALYSIS, TaskType.REASONING],
                performance_score=0.90,
                api_key_env="OPENAI_API_KEY"
            ),
            "openai:gpt-3.5-turbo": ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-3.5-turbo",
                cost_per_1k_tokens=0.5,
                max_tokens=4096,
                context_window=16385,
                strengths=[TaskType.SIMPLE_QUERY, TaskType.SUMMARIZATION],
                performance_score=0.80,
                api_key_env="OPENAI_API_KEY"
            ),
        })
        
        # Google Models
        self.model_configs.update({
            "google:gemini-pro": ModelConfig(
                provider=ModelProvider.GOOGLE,
                model_name="gemini-pro",
                cost_per_1k_tokens=0.5,
                max_tokens=8192,
                context_window=32768,
                strengths=[TaskType.ANALYSIS, TaskType.CREATIVE_WRITING],
                performance_score=0.85,
                api_key_env="GOOGLE_API_KEY"
            ),
            "google:gemini-pro-vision": ModelConfig(
                provider=ModelProvider.GOOGLE,
                model_name="gemini-pro-vision",
                cost_per_1k_tokens=0.25,
                max_tokens=4096,
                context_window=16384,
                strengths=[TaskType.ANALYSIS],
                performance_score=0.80,
                api_key_env="GOOGLE_API_KEY"
            ),
        })
    
    async def route_task(self, request: TaskRequest) -> Tuple[str, str]:
        """
        Intelligently route task to optimal model.
        Returns (model_id, reasoning).
        """
        # Get candidate models for task type
        candidates = self.routing_preferences.get(request.task_type, [])
        
        if not candidates:
            # Fall back to general-purpose models
            candidates = [
                "anthropic:claude-3-sonnet-20240229",
                "openai:gpt-4-turbo-preview",
                "google:gemini-pro"
            ]
        
        # Score each candidate
        scores = {}
        for model_id in candidates:
            if model_id not in self.model_configs:
                continue
                
            config = self.model_configs[model_id]
            if not config.availability:
                continue
                
            # Calculate composite score
            score = await self._calculate_model_score(
                model_id, request
            )
            scores[model_id] = score
        
        if not scores:
            # Emergency fallback
            return "anthropic:claude-3-haiku-20240307", "emergency_fallback"
        
        # Select best model
        best_model = max(scores.keys(), key=lambda k: scores[k])
        reasoning = f"scored_{scores[best_model]:.3f}"
        
        return best_model, reasoning
    
    async def _calculate_model_score(
        self,
        model_id: str,
        request: TaskRequest
    ) -> float:
        """Calculate composite score for model selection."""
        config = self.model_configs[model_id]
        
        # Base performance score
        score = config.performance_score
        
        # Task type suitability bonus
        if request.task_type in config.strengths:
            score += 0.15
        
        # Cost efficiency factor
        if request.cost_budget:
            estimated_cost = self._estimate_cost(model_id, request)
            if estimated_cost <= request.cost_budget:
                score += 0.1
            else:
                score -= 0.2  # Penalize over-budget models
        
        # Historical performance
        metrics = await self._get_performance_metrics(model_id)
        if metrics:
            score += metrics.success_rate * 0.1
            score += (1 - min(metrics.avg_response_time / 10.0, 1)) * 0.05
            score += metrics.avg_quality_score * 0.1
        
        # Urgency adjustment
        if request.urgency == "critical":
            # Prefer fastest models for critical tasks
            score += (1 - min(config.cost_per_1k_tokens / 30.0, 1)) * 0.1
        elif request.urgency == "low":
            # Prefer cost-efficient models for low priority
            score += (1 - config.cost_per_1k_tokens / 30.0) * 0.2
        
        # Circuit breaker check
        if self._is_circuit_open(model_id):
            score *= 0.1  # Heavy penalty for failing models
        
        return max(score, 0)
    
    def _estimate_cost(self, model_id: str, request: TaskRequest) -> float:
        """Estimate cost for request."""
        config = self.model_configs[model_id]
        
        # Rough token estimation (input + output)
        input_tokens = len(request.prompt) // 4  # ~4 chars per token
        output_tokens = request.max_tokens or 1000
        total_tokens = input_tokens + output_tokens
        
        return (total_tokens / 1000) * config.cost_per_1k_tokens
    
    async def execute_task(self, request: TaskRequest) -> ModelResponse:
        """Execute task using optimal model."""
        start_time = time.time()
        
        try:
            # Route to optimal model
            model_id, routing_reason = await self.route_task(request)
            config = self.model_configs[model_id]
            
            logger.info(
                f"Routing {request.task_type.value} task to {model_id} "
                f"(reason: {routing_reason})"
            )
            
            # Execute with selected model
            response = await self._execute_with_model(model_id, request)
            
            # Track performance
            await self._track_performance(model_id, response, start_time)
            
            # Update metrics
            track_agent_execution(
                agent_type=f"multi_model_{config.provider.value}",
                tier=config.model_name,
                status="success" if response.success else "failure",
                duration=response.response_time,
                cost=response.cost
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            
            # Return error response
            return ModelResponse(
                content="",
                provider=ModelProvider.ANTHROPIC,  # Default
                model_name="unknown",
                tokens_used=0,
                cost=0.0,
                response_time=time.time() - start_time,
                quality_score=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_with_model(
        self,
        model_id: str,
        request: TaskRequest
    ) -> ModelResponse:
        """Execute request with specific model."""
        config = self.model_configs[model_id]
        start_time = time.time()
        
        try:
            if config.provider == ModelProvider.ANTHROPIC:
                return await self._execute_anthropic(config, request, start_time)
            elif config.provider == ModelProvider.OPENAI:
                return await self._execute_openai(config, request, start_time)
            elif config.provider == ModelProvider.GOOGLE:
                return await self._execute_google(config, request, start_time)
            else:
                raise ValueError(f"Unsupported provider: {config.provider}")
                
        except Exception as e:
            response_time = time.time() - start_time
            
            # Update circuit breaker
            self._record_failure(model_id)
            
            return ModelResponse(
                content="",
                provider=config.provider,
                model_name=config.model_name,
                tokens_used=0,
                cost=0.0,
                response_time=response_time,
                quality_score=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_anthropic(
        self,
        config: ModelConfig,
        request: TaskRequest,
        start_time: float
    ) -> ModelResponse:
        """Execute with Anthropic model."""
        import os
        
        client = anthropic.Anthropic(
            api_key=os.getenv(config.api_key_env)
        )
        
        messages = [{"role": "user", "content": request.prompt}]
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})
        
        response = await asyncio.to_thread(
            client.messages.create,
            model=config.model_name,
            max_tokens=request.max_tokens or 1000,
            temperature=request.temperature,
            messages=messages
        )
        
        response_time = time.time() - start_time
        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        cost = (tokens_used / 1000) * config.cost_per_1k_tokens
        
        # Calculate quality score (placeholder - could use actual metrics)
        quality_score = min(0.9 + (len(response.content[0].text) / 1000) * 0.1, 1.0)
        
        return ModelResponse(
            content=response.content[0].text,
            provider=config.provider,
            model_name=config.model_name,
            tokens_used=tokens_used,
            cost=cost,
            response_time=response_time,
            quality_score=quality_score,
            success=True,
            metadata={
                "usage": response.usage.__dict__,
                "model": response.model,
                "stop_reason": response.stop_reason
            }
        )
    
    async def _execute_openai(
        self,
        config: ModelConfig,
        request: TaskRequest,
        start_time: float
    ) -> ModelResponse:
        """Execute with OpenAI model."""
        import os
        
        client = openai.AsyncOpenAI(
            api_key=os.getenv(config.api_key_env)
        )
        
        messages = [{"role": "user", "content": request.prompt}]
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})
        
        response = await client.chat.completions.create(
            model=config.model_name,
            messages=messages,
            max_tokens=request.max_tokens or 1000,
            temperature=request.temperature
        )
        
        response_time = time.time() - start_time
        tokens_used = response.usage.total_tokens
        cost = (tokens_used / 1000) * config.cost_per_1k_tokens
        
        quality_score = min(0.9 + (len(response.choices[0].message.content) / 1000) * 0.1, 1.0)
        
        return ModelResponse(
            content=response.choices[0].message.content,
            provider=config.provider,
            model_name=config.model_name,
            tokens_used=tokens_used,
            cost=cost,
            response_time=response_time,
            quality_score=quality_score,
            success=True,
            metadata={
                "usage": response.usage.__dict__,
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason
            }
        )
    
    async def _execute_google(
        self,
        config: ModelConfig,
        request: TaskRequest,
        start_time: float
    ) -> ModelResponse:
        """Execute with Google model."""
        import os
        
        genai.configure(api_key=os.getenv(config.api_key_env))
        model = genai.GenerativeModel(config.model_name)
        
        prompt = request.prompt
        if request.system_prompt:
            prompt = f"{request.system_prompt}\n\n{prompt}"
        
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=request.max_tokens or 1000,
                temperature=request.temperature
            )
        )
        
        response_time = time.time() - start_time
        
        # Google doesn't provide token usage in free tier
        estimated_tokens = len(response.text) // 4
        cost = (estimated_tokens / 1000) * config.cost_per_1k_tokens
        
        quality_score = min(0.85 + (len(response.text) / 1000) * 0.1, 1.0)
        
        return ModelResponse(
            content=response.text,
            provider=config.provider,
            model_name=config.model_name,
            tokens_used=estimated_tokens,
            cost=cost,
            response_time=response_time,
            quality_score=quality_score,
            success=True,
            metadata={
                "candidates_count": len(response.candidates),
                "safety_ratings": response.candidates[0].safety_ratings if response.candidates else []
            }
        )
    
    async def _get_performance_metrics(self, model_id: str) -> Optional[PerformanceMetrics]:
        """Get historical performance metrics for model."""
        if model_id in self.performance_metrics:
            return self.performance_metrics[model_id]
        
        # Try to load from Redis if available
        if self.redis_client:
            try:
                data = self.redis_client.get(f"model_metrics:{model_id}")
                if data:
                    return PerformanceMetrics(**json.loads(data))
            except Exception as e:
                logger.warning(f"Failed to load metrics from Redis: {e}")
        
        return None
    
    async def _track_performance(
        self,
        model_id: str,
        response: ModelResponse,
        start_time: float
    ):
        """Track performance metrics for model."""
        if model_id not in self.performance_metrics:
            self.performance_metrics[model_id] = PerformanceMetrics()
        
        metrics = self.performance_metrics[model_id]
        
        # Update metrics
        metrics.total_requests += 1
        if not response.success:
            metrics.failed_requests += 1
        
        metrics.success_rate = (
            (metrics.total_requests - metrics.failed_requests) / metrics.total_requests
        )
        
        # Update averages (exponential moving average)
        alpha = 0.1  # Smoothing factor
        if response.success:
            metrics.avg_response_time = (
                alpha * response.response_time + 
                (1 - alpha) * metrics.avg_response_time
            )
            metrics.avg_quality_score = (
                alpha * response.quality_score +
                (1 - alpha) * metrics.avg_quality_score
            )
            
            # Cost efficiency (quality per dollar)
            if response.cost > 0:
                efficiency = response.quality_score / response.cost
                metrics.cost_efficiency = (
                    alpha * efficiency + (1 - alpha) * metrics.cost_efficiency
                )
        
        metrics.last_updated = datetime.now()
        
        # Store in Redis if available
        if self.redis_client:
            try:
                data = {
                    "success_rate": metrics.success_rate,
                    "avg_response_time": metrics.avg_response_time,
                    "avg_quality_score": metrics.avg_quality_score,
                    "cost_efficiency": metrics.cost_efficiency,
                    "total_requests": metrics.total_requests,
                    "failed_requests": metrics.failed_requests,
                    "last_updated": metrics.last_updated.isoformat()
                }
                self.redis_client.setex(
                    f"model_metrics:{model_id}",
                    3600 * 24,  # 24 hours
                    json.dumps(data)
                )
            except Exception as e:
                logger.warning(f"Failed to store metrics in Redis: {e}")
    
    def _is_circuit_open(self, model_id: str) -> bool:
        """Check if circuit breaker is open for model."""
        if model_id not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[model_id]
        
        # Check if circuit should be reset
        if breaker["state"] == "open" and breaker["next_attempt"] < time.time():
            breaker["state"] = "half_open"
        
        return breaker["state"] == "open"
    
    def _record_failure(self, model_id: str):
        """Record failure for circuit breaker."""
        if model_id not in self.circuit_breakers:
            self.circuit_breakers[model_id] = {
                "state": "closed",
                "failure_count": 0,
                "next_attempt": 0,
                "failure_threshold": 5,
                "timeout": 60  # seconds
            }
        
        breaker = self.circuit_breakers[model_id]
        breaker["failure_count"] += 1
        
        if breaker["failure_count"] >= breaker["failure_threshold"]:
            breaker["state"] = "open"
            breaker["next_attempt"] = time.time() + breaker["timeout"]
            logger.warning(f"Circuit breaker opened for {model_id}")
    
    def _record_success(self, model_id: str):
        """Record success for circuit breaker."""
        if model_id in self.circuit_breakers:
            breaker = self.circuit_breakers[model_id]
            breaker["state"] = "closed"
            breaker["failure_count"] = 0
    
    async def get_model_status(self) -> Dict[str, Dict]:
        """Get status of all models."""
        status = {}
        
        for model_id, config in self.model_configs.items():
            metrics = await self._get_performance_metrics(model_id)
            circuit_open = self._is_circuit_open(model_id)
            
            status[model_id] = {
                "provider": config.provider.value,
                "model_name": config.model_name,
                "available": config.availability and not circuit_open,
                "cost_per_1k_tokens": config.cost_per_1k_tokens,
                "performance_score": config.performance_score,
                "circuit_breaker_open": circuit_open,
                "metrics": {
                    "success_rate": metrics.success_rate if metrics else 0,
                    "avg_response_time": metrics.avg_response_time if metrics else 0,
                    "avg_quality_score": metrics.avg_quality_score if metrics else 0,
                    "total_requests": metrics.total_requests if metrics else 0,
                } if metrics else None
            }
        
        return status