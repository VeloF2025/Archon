"""
Utilities for server services.

This file provides backward compatibility for imports that previously used
utils module functions. It re-exports functions from the main utils module.
"""

# Re-export all functions from the main utils module
from ...utils import (
    # Threading functions
    initialize_threading_service,
    get_utils_threading_service,
    get_threading_service,
    ProcessingMode,
    ThreadingConfig,
    RateLimitConfig,
    # Client functions
    get_supabase_client,
    # Embedding functions
    create_embedding,
    create_embeddings_batch,
    create_embedding_async,
    create_embeddings_batch_async,
    get_openai_client,
    # Contextual embedding functions
    generate_contextual_embedding,
    generate_contextual_embedding_async,
    generate_contextual_embeddings_batch,
    process_chunk_with_context,
    process_chunk_with_context_async,
    # Source management functions
    extract_source_summary,
    generate_source_title_and_metadata,
    update_source_info,
)

__all__ = [
    # Threading functions
    "initialize_threading_service",
    "get_utils_threading_service",
    "get_threading_service",
    "ProcessingMode",
    "ThreadingConfig",
    "RateLimitConfig",
    # Client functions
    "get_supabase_client",
    # Embedding functions
    "create_embedding",
    "create_embeddings_batch",
    "create_embedding_async",
    "create_embeddings_batch_async",
    "get_openai_client",
    # Contextual embedding functions
    "generate_contextual_embedding",
    "generate_contextual_embedding_async",
    "generate_contextual_embeddings_batch",
    "process_chunk_with_context",
    "process_chunk_with_context_async",
    # Source management functions
    "extract_source_summary",
    "generate_source_title_and_metadata",
    "update_source_info",
]