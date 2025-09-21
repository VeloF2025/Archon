"""
Pattern Storage Service with Vector Embeddings
Stores patterns in PostgreSQL with pgvector for similarity search
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from pydantic import BaseModel, Field

from ...server.utils import get_supabase_client
from ...server.services.embeddings import create_embedding

logger = logging.getLogger(__name__)


class StoredPattern(BaseModel):
    """Pattern stored in database with embeddings"""
    id: str
    name: str
    category: str
    language: str
    signature: str
    embedding: List[float]
    examples: List[Dict[str, Any]]
    frequency: int
    confidence: float
    effectiveness_score: float = 0.0
    usage_count: int = 0
    last_used: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class PatternStorage:
    """Manages pattern persistence and retrieval with vector search"""
    
    def __init__(self):
        self.supabase = get_supabase_client()
        self.embedding_dimension = 1536  # OpenAI embedding dimension
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Ensure pattern storage tables exist"""
        # Table creation would normally be done via migration
        # This is just for documentation of schema
        """
        CREATE TABLE IF NOT EXISTS code_patterns (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            language TEXT NOT NULL,
            signature TEXT UNIQUE NOT NULL,
            embedding vector(1536),
            examples JSONB DEFAULT '[]',
            frequency INTEGER DEFAULT 1,
            confidence REAL DEFAULT 0.0,
            effectiveness_score REAL DEFAULT 0.0,
            usage_count INTEGER DEFAULT 0,
            last_used TIMESTAMP,
            metadata JSONB DEFAULT '{}',
            is_antipattern BOOLEAN DEFAULT FALSE,
            performance_impact TEXT,
            suggested_alternative TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_patterns_embedding 
        ON code_patterns USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        
        CREATE INDEX IF NOT EXISTS idx_patterns_language 
        ON code_patterns(language);
        
        CREATE INDEX IF NOT EXISTS idx_patterns_category 
        ON code_patterns(category);
        """
        pass
    
    async def store_pattern(self, pattern: Dict[str, Any]) -> bool:
        """
        Store a detected pattern with embedding
        
        Args:
            pattern: Pattern data to store
            
        Returns:
            Success status
        """
        try:
            # Generate embedding for pattern
            pattern_text = self._pattern_to_text(pattern)
            embedding = await self._generate_embedding(pattern_text)
            
            # Prepare data for storage
            pattern_data = {
                "id": pattern["id"],
                "name": pattern["name"],
                "category": pattern["category"],
                "language": pattern["language"],
                "signature": pattern["signature"],
                "embedding": embedding,
                "examples": json.dumps(pattern.get("examples", [])),
                "frequency": pattern.get("frequency", 1),
                "confidence": pattern.get("confidence", 0.0),
                "metadata": json.dumps(pattern.get("metadata", {})),
                "is_antipattern": pattern.get("is_antipattern", False),
                "performance_impact": pattern.get("performance_impact"),
                "suggested_alternative": pattern.get("suggested_alternative"),
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Check if pattern exists
            existing = self.supabase.table("code_patterns").select("id").eq(
                "signature", pattern["signature"]
            ).execute()
            
            if existing.data:
                # Update existing pattern
                result = self.supabase.table("code_patterns").update({
                    "frequency": pattern_data["frequency"],
                    "confidence": pattern_data["confidence"],
                    "examples": pattern_data["examples"],
                    "updated_at": pattern_data["updated_at"]
                }).eq("signature", pattern["signature"]).execute()
                
                logger.info(f"Updated pattern: {pattern['name']}")
            else:
                # Insert new pattern
                result = self.supabase.table("code_patterns").insert(pattern_data).execute()
                logger.info(f"Stored new pattern: {pattern['name']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing pattern: {e}")
            return False
    
    async def search_similar_patterns(
        self, 
        query: str, 
        language: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[StoredPattern]:
        """
        Search for similar patterns using vector similarity
        
        Args:
            query: Search query text
            language: Filter by language
            category: Filter by category
            limit: Maximum results
            threshold: Similarity threshold
            
        Returns:
            List of similar patterns
        """
        try:
            # Generate embedding for query
            query_embedding = await self._generate_embedding(query)
            
            # Build query
            query_builder = self.supabase.table("code_patterns").select("*")
            
            # Add filters
            if language:
                query_builder = query_builder.eq("language", language)
            if category:
                query_builder = query_builder.eq("category", category)
            
            # Execute vector similarity search
            # Note: This would use pgvector's <-> operator in production
            # For now, we'll do a regular query and filter in memory
            result = query_builder.execute()
            
            if not result.data:
                return []
            
            # Calculate similarities and filter
            patterns_with_similarity = []
            for pattern in result.data:
                if pattern.get("embedding"):
                    similarity = self._cosine_similarity(
                        query_embedding,
                        pattern["embedding"]
                    )
                    if similarity >= threshold:
                        pattern["similarity"] = similarity
                        patterns_with_similarity.append(pattern)
            
            # Sort by similarity and limit
            patterns_with_similarity.sort(key=lambda x: x["similarity"], reverse=True)
            patterns_with_similarity = patterns_with_similarity[:limit]
            
            # Convert to StoredPattern objects
            stored_patterns = []
            for p in patterns_with_similarity:
                stored_patterns.append(StoredPattern(
                    id=p["id"],
                    name=p["name"],
                    category=p["category"],
                    language=p["language"],
                    signature=p["signature"],
                    embedding=p["embedding"],
                    examples=json.loads(p["examples"]) if isinstance(p["examples"], str) else p["examples"],
                    frequency=p["frequency"],
                    confidence=p["confidence"],
                    effectiveness_score=p.get("effectiveness_score", 0.0),
                    usage_count=p.get("usage_count", 0),
                    last_used=p.get("last_used"),
                    metadata=json.loads(p["metadata"]) if isinstance(p["metadata"], str) else p["metadata"],
                    created_at=p["created_at"],
                    updated_at=p["updated_at"]
                ))
            
            return stored_patterns
            
        except Exception as e:
            logger.error(f"Error searching patterns: {e}")
            return []
    
    async def get_pattern_by_id(self, pattern_id: str) -> Optional[StoredPattern]:
        """Get a specific pattern by ID"""
        try:
            result = self.supabase.table("code_patterns").select("*").eq(
                "id", pattern_id
            ).execute()
            
            if result.data:
                p = result.data[0]
                return StoredPattern(
                    id=p["id"],
                    name=p["name"],
                    category=p["category"],
                    language=p["language"],
                    signature=p["signature"],
                    embedding=p["embedding"],
                    examples=json.loads(p["examples"]) if isinstance(p["examples"], str) else p["examples"],
                    frequency=p["frequency"],
                    confidence=p["confidence"],
                    effectiveness_score=p.get("effectiveness_score", 0.0),
                    usage_count=p.get("usage_count", 0),
                    last_used=p.get("last_used"),
                    metadata=json.loads(p["metadata"]) if isinstance(p["metadata"], str) else p["metadata"],
                    created_at=p["created_at"],
                    updated_at=p["updated_at"]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching pattern {pattern_id}: {e}")
            return None
    
    async def update_pattern_usage(self, pattern_id: str, effective: bool):
        """
        Update pattern usage statistics
        
        Args:
            pattern_id: Pattern identifier
            effective: Whether the pattern was effective
        """
        try:
            pattern = await self.get_pattern_by_id(pattern_id)
            if not pattern:
                return
            
            # Update usage count and effectiveness
            new_usage_count = pattern.usage_count + 1
            new_effectiveness = pattern.effectiveness_score
            
            if effective:
                # Weighted average for effectiveness
                new_effectiveness = (
                    (pattern.effectiveness_score * pattern.usage_count + 1.0) /
                    new_usage_count
                )
            else:
                new_effectiveness = (
                    (pattern.effectiveness_score * pattern.usage_count) /
                    new_usage_count
                )
            
            # Update in database
            self.supabase.table("code_patterns").update({
                "usage_count": new_usage_count,
                "effectiveness_score": new_effectiveness,
                "last_used": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", pattern_id).execute()
            
            logger.info(f"Updated usage for pattern {pattern_id}: count={new_usage_count}, effectiveness={new_effectiveness:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating pattern usage: {e}")
    
    async def get_top_patterns(
        self,
        language: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 20
    ) -> List[StoredPattern]:
        """
        Get top patterns by effectiveness and usage
        
        Args:
            language: Filter by language
            category: Filter by category
            limit: Maximum results
            
        Returns:
            List of top patterns
        """
        try:
            query_builder = self.supabase.table("code_patterns").select("*")
            
            # Add filters
            if language:
                query_builder = query_builder.eq("language", language)
            if category:
                query_builder = query_builder.eq("category", category)
            
            # Order by effectiveness and usage
            query_builder = query_builder.order(
                "effectiveness_score", desc=True
            ).order(
                "usage_count", desc=True
            ).limit(limit)
            
            result = query_builder.execute()
            
            if not result.data:
                return []
            
            # Convert to StoredPattern objects
            patterns = []
            for p in result.data:
                patterns.append(StoredPattern(
                    id=p["id"],
                    name=p["name"],
                    category=p["category"],
                    language=p["language"],
                    signature=p["signature"],
                    embedding=p.get("embedding", []),
                    examples=json.loads(p["examples"]) if isinstance(p["examples"], str) else p["examples"],
                    frequency=p["frequency"],
                    confidence=p["confidence"],
                    effectiveness_score=p.get("effectiveness_score", 0.0),
                    usage_count=p.get("usage_count", 0),
                    last_used=p.get("last_used"),
                    metadata=json.loads(p["metadata"]) if isinstance(p["metadata"], str) else p["metadata"],
                    created_at=p["created_at"],
                    updated_at=p["updated_at"]
                ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error fetching top patterns: {e}")
            return []
    
    async def get_antipatterns(self, language: Optional[str] = None) -> List[StoredPattern]:
        """Get all detected anti-patterns"""
        try:
            query_builder = self.supabase.table("code_patterns").select("*").eq(
                "is_antipattern", True
            )
            
            if language:
                query_builder = query_builder.eq("language", language)
            
            result = query_builder.execute()
            
            if not result.data:
                return []
            
            # Convert to StoredPattern objects
            patterns = []
            for p in result.data:
                patterns.append(StoredPattern(
                    id=p["id"],
                    name=p["name"],
                    category=p["category"],
                    language=p["language"],
                    signature=p["signature"],
                    embedding=p.get("embedding", []),
                    examples=json.loads(p["examples"]) if isinstance(p["examples"], str) else p["examples"],
                    frequency=p["frequency"],
                    confidence=p["confidence"],
                    effectiveness_score=p.get("effectiveness_score", 0.0),
                    usage_count=p.get("usage_count", 0),
                    last_used=p.get("last_used"),
                    metadata=json.loads(p["metadata"]) if isinstance(p["metadata"], str) else p["metadata"],
                    created_at=p["created_at"],
                    updated_at=p["updated_at"]
                ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error fetching antipatterns: {e}")
            return []
    
    def _pattern_to_text(self, pattern: Dict[str, Any]) -> str:
        """Convert pattern to text for embedding generation"""
        text_parts = [
            f"Pattern: {pattern['name']}",
            f"Category: {pattern['category']}",
            f"Language: {pattern['language']}"
        ]
        
        if pattern.get("metadata", {}).get("description"):
            text_parts.append(f"Description: {pattern['metadata']['description']}")
        
        if pattern.get("suggested_alternative"):
            text_parts.append(f"Alternative: {pattern['suggested_alternative']}")
        
        if pattern.get("examples"):
            text_parts.append(f"Examples: {json.dumps(pattern['examples'][:2])}")
        
        return " ".join(text_parts)
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text"""
        try:
            # Use embedding service to generate embedding
            embedding = await create_embedding(text)
            if embedding:
                return embedding
            
            # Fallback to random embedding for testing
            logger.warning("Using random embedding as fallback")
            return np.random.randn(self.embedding_dimension).tolist()
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return random embedding as fallback
            return np.random.randn(self.embedding_dimension).tolist()
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def export_patterns_to_json(self, filepath: str):
        """Export all patterns to JSON file"""
        try:
            result = self.supabase.table("code_patterns").select("*").execute()
            
            if result.data:
                export_data = {
                    "patterns": result.data,
                    "exported_at": datetime.utcnow().isoformat(),
                    "total_count": len(result.data)
                }
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                logger.info(f"Exported {len(result.data)} patterns to {filepath}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error exporting patterns: {e}")
            return False