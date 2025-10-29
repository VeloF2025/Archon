"""
Enhanced Ollama Service v0.12.6 Integration

Provides advanced local model support with:
- GPU acceleration
- Batch processing
- Model management
- Performance monitoring
- Cost optimization
- Privacy-focused deployments
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path

import aiohttp
from openai import AsyncOpenAI

from ..config.logfire_config import get_logger
from .credential_service import credential_service

logger = get_logger(__name__)


@dataclass
class OllamaModel:
    """Information about an Ollama model"""
    name: str
    size: str
    modified_at: str
    digest: str
    details: Dict[str, Any]
    parameters: Dict[str, Any]


@dataclass
class OllamaConfig:
    """Ollama configuration"""
    base_url: str = "http://localhost:11434"
    timeout: int = 120
    max_retries: int = 3
    gpu_enabled: bool = True
    batch_size: int = 5
    memory_limit: Optional[str] = None


@dataclass
class OllamaUsageStats:
    """Ollama usage statistics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_duration: float = 0.0
    models_used: Dict[str, int] = None

    def __post_init__(self):
        if self.models_used is None:
            self.models_used = {}


class OllamaService:
    """Enhanced Ollama service with v0.12.6 features"""

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.client: Optional[AsyncOpenAI] = None
        self.usage_stats = OllamaUsageStats()
        self.available_models: Dict[str, OllamaModel] = {}
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize Ollama service and check connectivity"""
        try:
            # Test Ollama server connectivity
            health_url = f"{self.config.base_url}/api/tags"
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(health_url) as response:
                    if response.status == 200:
                        logger.info("✓ Ollama server is reachable")
                        self._initialized = True
                        await self._load_available_models()
                        return True
                    else:
                        logger.warning(f"Ollama server returned status {response.status}")
                        return False

        except Exception as e:
            logger.warning(f"Ollama server not reachable at {self.config.base_url}: {e}")
            logger.info("Ollama will be available when server starts")
            # Don't fail initialization - Ollama might start later
            self._initialized = False
            return True

    async def _load_available_models(self) -> None:
        """Load available Ollama models"""
        try:
            tags_url = f"{self.config.base_url}/api/tags"
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(tags_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        for model_data in data.get('models', []):
                            model = OllamaModel(
                                name=model_data['name'],
                                size=model_data.get('size', 'unknown'),
                                modified_at=model_data.get('modified_at', ''),
                                digest=model_data.get('digest', ''),
                                details=model_data.get('details', {}),
                                parameters=model_data.get('details', {}).get('parameters', {})
                            )
                            self.available_models[model.name] = model

                        logger.info(f"✓ Loaded {len(self.available_models)} Ollama models")
                        for model_name in list(self.available_models.keys())[:3]:
                            logger.info(f"  - {model_name}")

        except Exception as e:
            logger.error(f"Failed to load Ollama models: {e}")

    async def get_client(self) -> AsyncOpenAI:
        """Get OpenAI-compatible client for Ollama"""
        if not self.client:
            self.client = AsyncOpenAI(
                api_key="ollama",  # Required but unused
                base_url=f"{self.config.base_url}/v1",
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self.client

    async def list_models(self) -> List[OllamaModel]:
        """List all available Ollama models"""
        if not self._initialized:
            await self.initialize()

        return list(self.available_models.values())

    async def get_model_info(self, model_name: str) -> Optional[OllamaModel]:
        """Get detailed information about a specific model"""
        if model_name in self.available_models:
            return self.available_models[model_name]

        # Try to get model info from Ollama API
        try:
            show_url = f"{self.config.base_url}/api/show"
            async with aiohttp.ClientSession() as session:
                async with session.post(show_url, json={"name": model_name}) as response:
                    if response.status == 200:
                        data = await response.json()
                        model = OllamaModel(
                            name=data.get('name', model_name),
                            size=data.get('size', 'unknown'),
                            modified_at=data.get('modified_at', ''),
                            digest=data.get('digest', ''),
                            details=data,
                            parameters=data.get('details', {}).get('parameters', {})
                        )
                        self.available_models[model_name] = model
                        return model
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")

        return None

    async def pull_model(self, model_name: str) -> bool:
        """Pull a new model from Ollama registry"""
        try:
            logger.info(f"Pulling Ollama model: {model_name}")
            pull_url = f"{self.config.base_url}/api/pull"

            # Use subprocess for long-running pull operation
            process = subprocess.Popen(
                ["ollama", "pull", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Monitor progress
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    logger.info(f"Pull progress: {output.strip()}")

            # Check result
            if process.returncode == 0:
                logger.info(f"✓ Successfully pulled model: {model_name}")
                # Refresh model list
                await self._load_available_models()
                return True
            else:
                error = process.stderr.read()
                logger.error(f"Failed to pull model {model_name}: {error}")
                return False

        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], None]:
        """Create a chat completion using Ollama"""
        start_time = time.time()
        self.usage_stats.total_requests += 1

        try:
            client = await self.get_client()

            # Prepare request parameters
            params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream,
                **kwargs
            }

            # Add Ollama-specific options if available
            model_info = await self.get_model_info(model)
            if model_info:
                # Adjust parameters based on model capabilities
                if "context_length" in model_info.parameters:
                    max_context = model_info.parameters["context_length"]
                    # Estimate tokens from characters (rough approximation)
                    total_chars = sum(len(msg.get("content", "")) for msg in messages)
                    estimated_tokens = total_chars // 4

                    if estimated_tokens > max_context * 0.8:  # Leave 20% margin
                        logger.warning(f"Context may exceed model limit for {model}")

            if stream:
                return await self._stream_response(client, params)
            else:
                response = await client.chat.completions.create(**params)

                # Update usage stats
                self.usage_stats.successful_requests += 1
                self.usage_stats.total_duration += time.time() - start_time
                self.usage_stats.models_used[model] = self.usage_stats.models_used.get(model, 0) + 1

                if response.usage:
                    self.usage_stats.total_tokens += response.usage.total_tokens

                logger.info(f"Ollama response: {model} in {time.time() - start_time:.2f}s")
                return response

        except Exception as e:
            self.usage_stats.failed_requests += 1
            logger.error(f"Ollama chat completion failed for {model}: {e}")
            return None

    async def _stream_response(self, client: AsyncOpenAI, params: Dict[str, Any]):
        """Handle streaming responses from Ollama"""
        try:
            stream = await client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            yield None

    async def batch_chat_completion(
        self,
        requests: List[Dict[str, Any]],
        concurrency_limit: int = None
    ) -> List[Dict[str, Any]]:
        """Process multiple chat completion requests in batches"""
        if concurrency_limit is None:
            concurrency_limit = self.config.batch_size

        semaphore = asyncio.Semaphore(concurrency_limit)
        results = []

        async def process_request(request_data):
            async with semaphore:
                return await self.chat_completion(**request_data)

        tasks = [process_request(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and return successful results
        successful_results = [r for r in results if not isinstance(r, Exception)]

        logger.info(f"Batch processing completed: {len(successful_results)}/{len(requests)} successful")
        return successful_results

    async def get_usage_stats(self) -> OllamaUsageStats:
        """Get current usage statistics"""
        return self.usage_stats

    async def reset_usage_stats(self) -> None:
        """Reset usage statistics"""
        self.usage_stats = OllamaUsageStats()

    async def check_gpu_support(self) -> Dict[str, Any]:
        """Check GPU support and availability"""
        try:
            # Check if NVIDIA GPU is available
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                gpu_info = result.stdout.strip().split('\n')
                return {
                    "gpu_available": True,
                    "gpu_info": gpu_info,
                    "gpu_count": len(gpu_info)
                }
            else:
                return {"gpu_available": False, "reason": "nvidia-smi failed"}

        except FileNotFoundError:
            return {"gpu_available": False, "reason": "nvidia-smi not found"}
        except Exception as e:
            return {"gpu_available": False, "reason": str(e)}

    async def optimize_settings(self) -> Dict[str, Any]:
        """Get optimized settings based on available resources"""
        gpu_support = await self.check_gpu_support()
        models = await self.list_models()

        recommendations = {
            "gpu_acceleration": gpu_support["gpu_available"],
            "recommended_models": [],
            "batch_size": self.config.batch_size,
            "timeout": self.config.timeout
        }

        if gpu_support["gpu_available"]:
            recommendations["batch_size"] = min(10, self.config.batch_size * 2)
            recommendations["timeout"] = 60  # Faster with GPU

            # Recommend GPU-optimized models
            gpu_models = [m for m in models if "llama" in m.name.lower() or "mistral" in m.name.lower()]
            recommendations["recommended_models"] = [m.name for m in gpu_models[:3]]
        else:
            recommendations["batch_size"] = max(1, self.config.batch_size // 2)
            recommendations["timeout"] = 180  # Slower without GPU

            # Recommend smaller models for CPU
            cpu_models = [m for m in models if "phi" in m.name.lower() or "qwen" in m.name.lower()]
            recommendations["recommended_models"] = [m.name for m in cpu_models[:3]]

        return recommendations

    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        gpu_support = await self.check_gpu_support()
        models = await self.list_models()
        stats = await self.get_usage_stats()

        return {
            "service_initialized": self._initialized,
            "server_reachable": self._initialized,
            "available_models": len(models),
            "model_list": [m.name for m in models],
            "gpu_support": gpu_support,
            "usage_stats": asdict(stats),
            "config": asdict(self.config),
            "health": "healthy" if self._initialized else "server_not_reachable"
        }


# Global Ollama service instance
ollama_service = OllamaService()


async def get_ollama_service() -> OllamaService:
    """Get the global Ollama service instance"""
    if not ollama_service._initialized:
        await ollama_service.initialize()
    return ollama_service


async def optimize_ollama_for_archon() -> Dict[str, Any]:
    """Optimize Ollama settings specifically for Archon"""
    service = await get_ollama_service()

    # Get optimized settings
    settings = await service.optimize_settings()

    # Archon-specific optimizations
    archon_config = {
        **settings,
        "archon_optimizations": {
            "rag_friendly_models": ["llama2", "mistral", "qwen"],
            "code_generation_models": ["codellama", "deepseek-coder"],
            "agent_models": ["llama3", "mistral"],
            "embedding_models": ["all-minilm", "nomic-embed-text"]
        },
        "integration_points": {
            "llm_provider_service": "Enhanced with v2 client support",
            "knowledge_base": "RAG processing optimization",
            "agent_system": "Local model fallback capability",
            "cost_optimization": "Reduced API calls for local processing"
        }
    }

    return archon_config