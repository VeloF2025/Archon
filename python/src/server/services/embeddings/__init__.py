"""
Embedding Services

Handles all embedding-related operations.
"""

from .contextual_embedding_service import (
    generate_contextual_embedding_sync as generate_contextual_embedding,
    generate_contextual_embeddings_batch_sync as generate_contextual_embeddings_batch,
    process_chunk_with_context_sync as process_chunk_with_context,
    generate_contextual_embedding_async,
    generate_contextual_embeddings_batch_async,
    process_chunk_with_context_async,
)
from .embedding_service import (
    create_embedding_sync as create_embedding,
    create_embedding as create_embedding_async,
    create_embeddings_batch_sync as create_embeddings_batch,
    create_embeddings_batch as create_embeddings_batch_async,
    get_openai_client
)

__all__ = [
    # Embedding functions
    "create_embedding",
    "create_embedding_async",
    "create_embeddings_batch",
    "create_embeddings_batch_async",
    "get_openai_client",
    # Contextual embedding functions
    "generate_contextual_embedding",
    "generate_contextual_embedding_async",
    "generate_contextual_embeddings_batch",
    "generate_contextual_embeddings_batch_async",
    "process_chunk_with_context",
    "process_chunk_with_context_async",
]
