"""
LLM Provider Service

Provides a unified interface for creating OpenAI-compatible clients for different LLM providers.
Supports OpenAI, Ollama, Google Gemini API, and Gemini CLI.
Includes intelligent task routing for optimal cost and performance.
"""

import time
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import openai

from ..config.logfire_config import get_logger
from .credential_service import credential_service
from .gemini_cli_service import (
    GeminiCLIService,
    GeminiTask,
    TaskType,
    TaskPriority,
    get_gemini_cli_service
)

logger = get_logger(__name__)


class LLMProvider(Enum):
    """Available LLM providers"""
    OPENAI = "openai"
    OLLAMA = "ollama"
    GOOGLE_API = "google"  # Gemini API
    GEMINI_CLI = "gemini_cli"  # Gemini CLI for multimodal and large context


class TaskCharacteristics:
    """Characteristics of a task for routing decisions"""
    
    def __init__(
        self,
        requires_multimodal: bool = False,
        has_images: bool = False,
        has_pdfs: bool = False,
        context_size: int = 0,
        priority: str = "normal",
        requires_streaming: bool = False,
        requires_function_calling: bool = False,
        is_embedding: bool = False
    ):
        self.requires_multimodal = requires_multimodal
        self.has_images = has_images
        self.has_pdfs = has_pdfs
        self.context_size = context_size
        self.priority = priority
        self.requires_streaming = requires_streaming
        self.requires_function_calling = requires_function_calling
        self.is_embedding = is_embedding

# Settings cache with TTL
_settings_cache: dict[str, tuple[Any, float]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes


def _get_cached_settings(key: str) -> Any | None:
    """Get cached settings if not expired."""
    if key in _settings_cache:
        value, timestamp = _settings_cache[key]
        if time.time() - timestamp < _CACHE_TTL_SECONDS:
            return value
        else:
            # Expired, remove from cache
            del _settings_cache[key]
    return None


def _set_cached_settings(key: str, value: Any) -> None:
    """Cache settings with current timestamp."""
    _settings_cache[key] = (value, time.time())


@asynccontextmanager
async def get_llm_client(provider: str | None = None, use_embedding_provider: bool = False):
    """
    Create an async OpenAI-compatible client based on the configured provider.

    This context manager handles client creation for different LLM providers
    that support the OpenAI API format.

    Args:
        provider: Override provider selection
        use_embedding_provider: Use the embedding-specific provider if different

    Yields:
        openai.AsyncOpenAI: An OpenAI-compatible client configured for the selected provider
    """
    client = None

    try:
        # Get provider configuration from database settings
        if provider:
            # Explicit provider requested - get minimal config
            provider_name = provider
            api_key = await credential_service._get_provider_api_key(provider)

            # Check cache for rag_settings
            cache_key = "rag_strategy_settings"
            rag_settings = _get_cached_settings(cache_key)
            if rag_settings is None:
                rag_settings = await credential_service.get_credentials_by_category("rag_strategy")
                _set_cached_settings(cache_key, rag_settings)
                logger.debug("Fetched and cached rag_strategy settings")
            else:
                logger.debug("Using cached rag_strategy settings")

            base_url = credential_service._get_provider_base_url(provider, rag_settings)
        else:
            # Get configured provider from database
            service_type = "embedding" if use_embedding_provider else "llm"

            # Check cache for provider config
            cache_key = f"provider_config_{service_type}"
            provider_config = _get_cached_settings(cache_key)
            if provider_config is None:
                provider_config = await credential_service.get_active_provider(service_type)
                _set_cached_settings(cache_key, provider_config)
                logger.debug(f"Fetched and cached {service_type} provider config")
            else:
                logger.debug(f"Using cached {service_type} provider config")

            provider_name = provider_config["provider"]
            api_key = provider_config["api_key"]
            base_url = provider_config["base_url"]

        logger.info(f"Creating LLM client for provider: {provider_name}")

        if provider_name == "openai":
            if not api_key:
                raise ValueError("OpenAI API key not found")

            client = openai.AsyncOpenAI(api_key=api_key)
            logger.info("OpenAI client created successfully")

        elif provider_name == "ollama":
            # Ollama requires an API key in the client but doesn't actually use it
            client = openai.AsyncOpenAI(
                api_key="ollama",  # Required but unused by Ollama
                base_url=base_url or "http://localhost:11434/v1",
            )
            logger.info(f"Ollama client created successfully with base URL: {base_url}")

        elif provider_name == "google":
            if not api_key:
                raise ValueError("Google API key not found")

            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=base_url or "https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            logger.info("Google Gemini client created successfully")

        elif provider_name == "gemini_cli":
            # Gemini CLI doesn't use the standard OpenAI client
            # It's handled separately through the GeminiCLIService
            logger.info("Gemini CLI provider selected - use execute_with_gemini_cli() instead")
            yield None  # Special case for Gemini CLI
            
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")

        if provider_name != "gemini_cli":
            yield client

    except Exception as e:
        logger.error(
            f"Error creating LLM client for provider {provider_name if 'provider_name' in locals() else 'unknown'}: {e}"
        )
        raise
    finally:
        # Cleanup if needed
        pass


async def route_llm_task(task_characteristics: TaskCharacteristics) -> Tuple[LLMProvider, str]:
    """Intelligently route a task to the optimal LLM provider
    
    This function implements the hybrid LLM strategy to optimize cost and performance
    by routing tasks to the most appropriate provider based on their characteristics.
    
    Args:
        task_characteristics: Characteristics of the task to route
        
    Returns:
        Tuple[LLMProvider, str]: (selected_provider, reason_for_selection)
    """
    # Check if Gemini CLI is available for multimodal/large context tasks
    gemini_cli = await get_gemini_cli_service()
    gemini_available = gemini_cli._initialized if gemini_cli else False
    
    # Priority 1: Multimodal tasks -> Gemini CLI (if available)
    if task_characteristics.requires_multimodal and gemini_available:
        can_execute, reason = await gemini_cli.can_execute(TaskType.MULTIMODAL)
        if can_execute:
            return LLMProvider.GEMINI_CLI, "Multimodal task routed to Gemini CLI"
    
    # Priority 2: Large context (>128K tokens) -> Gemini CLI
    if task_characteristics.context_size > 128000 and gemini_available:
        can_execute, reason = await gemini_cli.can_execute(TaskType.LARGE_CONTEXT)
        if can_execute:
            return LLMProvider.GEMINI_CLI, f"Large context ({task_characteristics.context_size} tokens) routed to Gemini CLI"
    
    # Priority 3: High-priority or streaming tasks -> OpenAI
    if task_characteristics.priority == "high" or task_characteristics.requires_streaming:
        return LLMProvider.OPENAI, "High priority/streaming task routed to OpenAI"
    
    # Priority 4: Function calling -> OpenAI (best support)
    if task_characteristics.requires_function_calling:
        return LLMProvider.OPENAI, "Function calling task routed to OpenAI"
    
    # Priority 5: Embeddings -> Use existing embedding provider
    if task_characteristics.is_embedding:
        provider_config = await credential_service.get_active_provider("embedding")
        return LLMProvider(provider_config["provider"]), "Embedding task routed to configured provider"
    
    # Priority 6: Default routing based on availability and cost
    if gemini_available:
        # Check if we have budget remaining for Gemini CLI
        can_execute, reason = await gemini_cli.can_execute(TaskType.GENERAL)
        if can_execute:
            return LLMProvider.GEMINI_CLI, "General task routed to Gemini CLI (cost optimization)"
    
    # Fallback: Use configured default provider
    provider_config = await credential_service.get_active_provider("llm")
    return LLMProvider(provider_config["provider"]), "Routed to default configured provider"


async def execute_with_gemini_cli(
    prompt: str,
    files: Optional[List[str]] = None,
    task_type: TaskType = TaskType.GENERAL,
    priority: TaskPriority = TaskPriority.NORMAL
) -> Dict[str, Any]:
    """Execute a task using Gemini CLI
    
    Args:
        prompt: The prompt to execute
        files: Optional list of files for multimodal processing
        task_type: Type of task
        priority: Task priority
        
    Returns:
        Dict containing the result
    """
    gemini_cli = await get_gemini_cli_service()
    
    if not gemini_cli._initialized:
        raise ValueError("Gemini CLI is not available or not initialized")
    
    task = GeminiTask(
        type=task_type,
        priority=priority,
        prompt=prompt,
        files=files or [],
        cache_key=gemini_cli._generate_cache_key(prompt, files) if files else None
    )
    
    return await gemini_cli.execute_task(task)


async def get_llm_usage_stats() -> Dict[str, Any]:
    """Get usage statistics for all LLM providers
    
    Returns:
        Dict containing usage stats for each provider
    """
    stats = {}
    
    # Get Gemini CLI stats if available
    gemini_cli = await get_gemini_cli_service()
    if gemini_cli and gemini_cli._initialized:
        stats["gemini_cli"] = await gemini_cli.get_usage_stats()
    
    # TODO: Add stats for other providers (OpenAI, etc.)
    # This would require implementing token tracking for each provider
    
    return stats


async def get_embedding_model(provider: str | None = None) -> str:
    """
    Get the configured embedding model based on the provider.

    Args:
        provider: Override provider selection

    Returns:
        str: The embedding model to use
    """
    try:
        # Get provider configuration
        if provider:
            # Explicit provider requested
            provider_name = provider
            # Get custom model from settings if any
            cache_key = "rag_strategy_settings"
            rag_settings = _get_cached_settings(cache_key)
            if rag_settings is None:
                rag_settings = await credential_service.get_credentials_by_category("rag_strategy")
                _set_cached_settings(cache_key, rag_settings)
            custom_model = rag_settings.get("EMBEDDING_MODEL", "")
        else:
            # Get configured provider from database
            cache_key = "provider_config_embedding"
            provider_config = _get_cached_settings(cache_key)
            if provider_config is None:
                provider_config = await credential_service.get_active_provider("embedding")
                _set_cached_settings(cache_key, provider_config)
            provider_name = provider_config["provider"]
            custom_model = provider_config["embedding_model"]

        # Use custom model if specified
        if custom_model:
            return custom_model

        # Return provider-specific defaults
        if provider_name == "openai":
            return "text-embedding-3-small"
        elif provider_name == "ollama":
            # Ollama default embedding model
            return "nomic-embed-text"
        elif provider_name == "google":
            # Google's embedding model
            return "text-embedding-004"
        else:
            # Fallback to OpenAI's model
            return "text-embedding-3-small"

    except Exception as e:
        logger.error(f"Error getting embedding model: {e}")
        # Fallback to OpenAI default
        return "text-embedding-3-small"
