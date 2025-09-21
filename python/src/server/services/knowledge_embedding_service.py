"""
Knowledge Embedding Service - Vector embeddings for agent knowledge management

This service provides vector embedding generation and similarity search capabilities
for agent knowledge management using pgvector in Supabase.
"""

import logging
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass, field
import asyncio
import numpy as np

from sentence_transformers import SentenceTransformer
from supabase import Client
import httpx

from ...database.agent_models import AgentType

logger = logging.getLogger(__name__)

# Embedding model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, efficient model for knowledge embeddings
EMBEDDING_DIMENSION = 384  # Dimension of the embedding vectors

@dataclass
class KnowledgeItem:
    """Represents a piece of knowledge with its embedding"""
    id: UUID = field(default_factory=uuid4)
    agent_id: UUID = None
    content: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    relevance_score: float = 0.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None

@dataclass
class KnowledgeContext:
    """Context for knowledge retrieval and storage"""
    agent_id: UUID
    task_id: UUID
    query: str
    max_results: int = 5
    similarity_threshold: float = 0.7
    include_global: bool = True  # Include global knowledge in search

class KnowledgeEmbeddingService:
    """Service for managing agent knowledge with vector embeddings"""
    
    def __init__(self, supabase_client: Client = None):
        """
        Initialize the knowledge embedding service
        
        Args:
            supabase_client: Supabase client for database operations
        """
        self.supabase = supabase_client
        self.embedding_model = None
        self.knowledge_cache: Dict[UUID, List[KnowledgeItem]] = {}
        self._initialize_embedding_model()
        
        logger.info("Initialized Knowledge Embedding Service")
    
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model for embeddings"""
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fallback to random embeddings if model fails to load
            self.embedding_model = None
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text:
            return [0.0] * EMBEDDING_DIMENSION
        
        try:
            if self.embedding_model:
                # Generate real embedding using the model
                embedding = self.embedding_model.encode(text, convert_to_numpy=True)
                return embedding.tolist()
            else:
                # Fallback: Generate deterministic pseudo-embedding based on text hash
                hash_obj = hashlib.sha256(text.encode())
                hash_bytes = hash_obj.digest()
                
                # Convert hash to floats and normalize
                values = []
                for i in range(0, len(hash_bytes), 2):
                    if i + 1 < len(hash_bytes):
                        value = (hash_bytes[i] + hash_bytes[i+1]) / 510.0  # Normalize to [0, 1]
                        values.append(value * 2 - 1)  # Scale to [-1, 1]
                
                # Pad or truncate to match embedding dimension
                if len(values) < EMBEDDING_DIMENSION:
                    values.extend([0.0] * (EMBEDDING_DIMENSION - len(values)))
                else:
                    values = values[:EMBEDDING_DIMENSION]
                
                return values
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * EMBEDDING_DIMENSION
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Normalize to [0, 1] range
            return (similarity + 1) / 2
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    async def store_knowledge(
        self,
        agent_id: UUID,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> KnowledgeItem:
        """
        Store a piece of knowledge with its embedding
        
        Args:
            agent_id: ID of the agent storing the knowledge
            content: The knowledge content
            context: Context information about the knowledge
            metadata: Additional metadata
            
        Returns:
            Created KnowledgeItem
        """
        # Generate embedding for the content
        embedding = self.generate_embedding(content)
        
        # Create knowledge item
        knowledge_item = KnowledgeItem(
            agent_id=agent_id,
            content=content,
            context=context or {},
            embedding=embedding,
            metadata=metadata or {}
        )
        
        # Store in database if available
        if self.supabase:
            try:
                result = self.supabase.table("archon_agent_knowledge").insert({
                    "id": str(knowledge_item.id),
                    "agent_id": str(agent_id),
                    "content": content,
                    "context": json.dumps(context or {}),
                    "embedding": embedding,
                    "metadata": json.dumps(metadata or {}),
                    "created_at": knowledge_item.created_at.isoformat()
                }).execute()
                
                if result.data:
                    logger.info(f"Stored knowledge item {knowledge_item.id} for agent {agent_id}")
            except Exception as e:
                logger.error(f"Failed to store knowledge in database: {e}")
        
        # Update cache
        if agent_id not in self.knowledge_cache:
            self.knowledge_cache[agent_id] = []
        self.knowledge_cache[agent_id].append(knowledge_item)
        
        return knowledge_item
    
    async def retrieve_relevant_knowledge(
        self,
        context: KnowledgeContext
    ) -> List[KnowledgeItem]:
        """
        Retrieve relevant knowledge based on similarity search
        
        Args:
            context: Knowledge retrieval context
            
        Returns:
            List of relevant knowledge items sorted by relevance
        """
        # Generate embedding for the query
        query_embedding = self.generate_embedding(context.query)
        
        relevant_items = []
        
        # Search in database if available
        if self.supabase:
            try:
                # Build the query
                query = self.supabase.table("archon_agent_knowledge").select("*")
                
                # Filter by agent if not including global knowledge
                if not context.include_global:
                    query = query.eq("agent_id", str(context.agent_id))
                
                result = query.execute()
                
                if result.data:
                    for item_data in result.data:
                        # Calculate similarity
                        item_embedding = item_data.get("embedding", [])
                        similarity = self.calculate_similarity(query_embedding, item_embedding)
                        
                        # Filter by similarity threshold
                        if similarity >= context.similarity_threshold:
                            knowledge_item = KnowledgeItem(
                                id=UUID(item_data["id"]),
                                agent_id=UUID(item_data["agent_id"]),
                                content=item_data["content"],
                                context=json.loads(item_data.get("context", "{}")),
                                embedding=item_embedding,
                                metadata=json.loads(item_data.get("metadata", "{}")),
                                relevance_score=similarity
                            )
                            relevant_items.append(knowledge_item)
                            
            except Exception as e:
                logger.error(f"Failed to retrieve knowledge from database: {e}")
        
        # Also search in cache
        cache_items = []
        if context.agent_id in self.knowledge_cache:
            for item in self.knowledge_cache[context.agent_id]:
                similarity = self.calculate_similarity(query_embedding, item.embedding)
                if similarity >= context.similarity_threshold:
                    item.relevance_score = similarity
                    cache_items.append(item)
        
        # Merge and deduplicate
        all_items = relevant_items + cache_items
        unique_items = {item.id: item for item in all_items}.values()
        
        # Sort by relevance score
        sorted_items = sorted(unique_items, key=lambda x: x.relevance_score, reverse=True)
        
        # Limit results
        limited_items = sorted_items[:context.max_results]
        
        # Update access metrics
        for item in limited_items:
            item.access_count += 1
            item.last_accessed = datetime.now()
        
        logger.info(f"Retrieved {len(limited_items)} relevant knowledge items for query: {context.query[:50]}...")
        
        return limited_items
    
    async def update_knowledge_relevance(
        self,
        knowledge_id: UUID,
        feedback_score: float
    ):
        """
        Update knowledge relevance based on usage feedback
        
        Args:
            knowledge_id: ID of the knowledge item
            feedback_score: Feedback score (0-1, higher is better)
        """
        try:
            if self.supabase:
                # Update relevance in database
                result = self.supabase.table("archon_agent_knowledge").update({
                    "relevance_score": feedback_score,
                    "last_accessed": datetime.now().isoformat()
                }).eq("id", str(knowledge_id)).execute()
                
                if result.data:
                    logger.debug(f"Updated relevance for knowledge {knowledge_id}: {feedback_score}")
                    
        except Exception as e:
            logger.error(f"Failed to update knowledge relevance: {e}")
    
    async def get_agent_knowledge_summary(self, agent_id: UUID) -> Dict[str, Any]:
        """
        Get summary of an agent's knowledge base
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Summary statistics and top knowledge items
        """
        summary = {
            "total_items": 0,
            "avg_relevance": 0.0,
            "total_accesses": 0,
            "top_items": [],
            "recent_items": []
        }
        
        try:
            if self.supabase:
                # Get all knowledge for the agent
                result = self.supabase.table("archon_agent_knowledge").select("*").eq(
                    "agent_id", str(agent_id)
                ).execute()
                
                if result.data:
                    items = result.data
                    summary["total_items"] = len(items)
                    
                    if items:
                        # Calculate average relevance
                        relevances = [item.get("relevance_score", 0) for item in items]
                        summary["avg_relevance"] = sum(relevances) / len(relevances)
                        
                        # Calculate total accesses
                        summary["total_accesses"] = sum(item.get("access_count", 0) for item in items)
                        
                        # Get top items by relevance
                        sorted_by_relevance = sorted(
                            items,
                            key=lambda x: x.get("relevance_score", 0),
                            reverse=True
                        )
                        summary["top_items"] = [
                            {
                                "id": item["id"],
                                "content": item["content"][:100],
                                "relevance": item.get("relevance_score", 0)
                            }
                            for item in sorted_by_relevance[:5]
                        ]
                        
                        # Get recent items
                        sorted_by_date = sorted(
                            items,
                            key=lambda x: x.get("created_at", ""),
                            reverse=True
                        )
                        summary["recent_items"] = [
                            {
                                "id": item["id"],
                                "content": item["content"][:100],
                                "created_at": item.get("created_at", "")
                            }
                            for item in sorted_by_date[:5]
                        ]
                        
        except Exception as e:
            logger.error(f"Failed to get knowledge summary: {e}")
        
        return summary
    
    async def prune_low_relevance_knowledge(
        self,
        agent_id: UUID,
        threshold: float = 0.3,
        min_age_days: int = 30
    ) -> int:
        """
        Remove low-relevance knowledge items that haven't been accessed recently
        
        Args:
            agent_id: ID of the agent
            threshold: Relevance score threshold below which to prune
            min_age_days: Minimum age in days before pruning
            
        Returns:
            Number of items pruned
        """
        pruned_count = 0
        
        try:
            if self.supabase:
                # Calculate cutoff date
                from datetime import timedelta
                cutoff_date = datetime.now() - timedelta(days=min_age_days)
                
                # Delete low-relevance old items
                result = self.supabase.table("archon_agent_knowledge").delete().eq(
                    "agent_id", str(agent_id)
                ).lt("relevance_score", threshold).lt(
                    "created_at", cutoff_date.isoformat()
                ).execute()
                
                if result.data:
                    pruned_count = len(result.data)
                    logger.info(f"Pruned {pruned_count} low-relevance knowledge items for agent {agent_id}")
                    
        except Exception as e:
            logger.error(f"Failed to prune knowledge: {e}")
        
        return pruned_count

# Global instance
_knowledge_embedding_service = None

def get_knowledge_embedding_service(supabase_client: Client = None) -> KnowledgeEmbeddingService:
    """Get or create the global knowledge embedding service instance"""
    global _knowledge_embedding_service
    if _knowledge_embedding_service is None:
        _knowledge_embedding_service = KnowledgeEmbeddingService(supabase_client)
    return _knowledge_embedding_service