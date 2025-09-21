"""
Semantic Similarity Engine
Advanced semantic similarity analysis for code using multiple similarity metrics
Provides intelligent code matching, duplicate detection, and relationship analysis
"""

from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import asyncio
import logging
import numpy as np
import json
import hashlib
from collections import defaultdict, Counter
import math

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SimilarityType(Enum):
    """Types of similarity analysis"""
    SYNTACTIC = "syntactic"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    FUNCTIONAL = "functional"
    CONTEXTUAL = "contextual"
    BEHAVIORAL = "behavioral"


class SimilarityScope(Enum):
    """Scope of similarity analysis"""
    TOKEN_LEVEL = "token_level"
    LINE_LEVEL = "line_level"
    FUNCTION_LEVEL = "function_level"
    CLASS_LEVEL = "class_level"
    FILE_LEVEL = "file_level"
    MODULE_LEVEL = "module_level"


class MatchType(Enum):
    """Types of matches found"""
    EXACT_DUPLICATE = "exact_duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    FUNCTIONAL_EQUIVALENT = "functional_equivalent"
    STRUCTURAL_SIMILAR = "structural_similar"
    SEMANTIC_RELATED = "semantic_related"
    REFACTOR_CANDIDATE = "refactor_candidate"


@dataclass
class CodeSegment:
    """Represents a segment of code for similarity analysis"""
    segment_id: str
    content: str
    file_path: str
    start_line: int
    end_line: int
    segment_type: str  # "function", "class", "method", "block"
    language: str = "python"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "content": self.content,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "segment_type": self.segment_type,
            "language": self.language,
            "metadata": self.metadata
        }


@dataclass
class SimilarityMatch:
    """Represents a similarity match between code segments"""
    match_id: str
    source_segment: CodeSegment
    target_segment: CodeSegment
    similarity_score: float
    similarity_type: SimilarityType
    match_type: MatchType
    confidence: float
    explanation: str
    similarity_metrics: Dict[str, float]
    alignment_data: Optional[Dict[str, Any]] = None
    refactor_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "source_segment": self.source_segment.to_dict(),
            "target_segment": self.target_segment.to_dict(),
            "similarity_score": self.similarity_score,
            "similarity_type": self.similarity_type.value,
            "match_type": self.match_type.value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "similarity_metrics": self.similarity_metrics,
            "alignment_data": self.alignment_data,
            "refactor_suggestions": self.refactor_suggestions
        }


@dataclass
class SemanticVector:
    """Semantic vector representation of code"""
    vector_id: str
    segment_id: str
    embedding: np.ndarray
    vector_type: str  # "token", "ast", "semantic", "hybrid"
    dimensionality: int
    normalization: str = "l2"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector_id": self.vector_id,
            "segment_id": self.segment_id,
            "embedding": self.embedding.tolist() if isinstance(self.embedding, np.ndarray) else self.embedding,
            "vector_type": self.vector_type,
            "dimensionality": self.dimensionality,
            "normalization": self.normalization,
            "metadata": self.metadata
        }


class SimilaritySearchRequest(BaseModel):
    """Request for similarity search"""
    query_code: str
    search_scope: str = "function_level"
    similarity_types: List[str] = Field(default=["semantic", "structural"])
    threshold: float = 0.7
    max_results: int = 10
    include_explanations: bool = True
    file_filters: List[str] = Field(default_factory=list)
    language: str = "python"


class SemanticSimilarityEngine:
    """
    Advanced semantic similarity engine for code analysis
    Provides multiple similarity metrics and intelligent code matching
    """
    
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        
        # Storage
        self.code_segments: Dict[str, CodeSegment] = {}
        self.semantic_vectors: Dict[str, SemanticVector] = {}
        self.similarity_cache: Dict[str, List[SimilarityMatch]] = {}
        self.segment_index: Dict[str, List[str]] = defaultdict(list)  # file_path -> segment_ids
        
        # Similarity calculators
        self.similarity_calculators = {
            SimilarityType.SYNTACTIC: SyntacticSimilarityCalculator(),
            SimilarityType.SEMANTIC: SemanticSimilarityCalculator(),
            SimilarityType.STRUCTURAL: StructuralSimilarityCalculator(),
            SimilarityType.FUNCTIONAL: FunctionalSimilarityCalculator(),
            SimilarityType.CONTEXTUAL: ContextualSimilarityCalculator(),
            SimilarityType.BEHAVIORAL: BehavioralSimilarityCalculator()
        }
        
        # Configuration
        self.default_threshold = 0.7
        self.exact_match_threshold = 0.95
        self.near_duplicate_threshold = 0.85
        self.semantic_threshold = 0.75
        
        logger.info("SemanticSimilarityEngine initialized")
    
    async def add_code_segment(
        self, 
        segment: CodeSegment,
        generate_vectors: bool = True
    ) -> bool:
        """
        Add a code segment for similarity analysis
        """
        try:
            # Store segment
            self.code_segments[segment.segment_id] = segment
            
            # Update index
            self.segment_index[segment.file_path].append(segment.segment_id)
            
            # Generate semantic vectors
            if generate_vectors:
                await self._generate_semantic_vectors(segment)
            
            logger.debug(f"Added code segment: {segment.segment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding code segment: {e}")
            return False
    
    async def find_similar_code(
        self, 
        request: SimilaritySearchRequest
    ) -> List[SimilarityMatch]:
        """
        Find code segments similar to the query
        """
        try:
            # Create temporary segment for query
            query_segment = CodeSegment(
                segment_id="query_temp",
                content=request.query_code,
                file_path="query",
                start_line=0,
                end_line=len(request.query_code.split('\n')),
                segment_type="query",
                language=request.language
            )
            
            # Generate vectors for query
            await self._generate_semantic_vectors(query_segment)
            
            # Find candidates
            candidates = await self._find_similarity_candidates(
                query_segment, 
                request
            )
            
            # Calculate similarities
            matches = []
            for candidate_id in candidates:
                candidate_segment = self.code_segments[candidate_id]
                
                # Calculate similarity for each requested type
                for similarity_type_str in request.similarity_types:
                    try:
                        similarity_type = SimilarityType(similarity_type_str)
                        match = await self._calculate_similarity(
                            query_segment,
                            candidate_segment,
                            similarity_type
                        )
                        
                        if match and match.similarity_score >= request.threshold:
                            matches.append(match)
                            
                    except ValueError:
                        logger.warning(f"Unknown similarity type: {similarity_type_str}")
                        continue
            
            # Remove duplicates and sort
            unique_matches = self._deduplicate_matches(matches)
            sorted_matches = sorted(
                unique_matches, 
                key=lambda m: m.similarity_score, 
                reverse=True
            )
            
            # Apply limit
            result_matches = sorted_matches[:request.max_results]
            
            logger.info(f"Found {len(result_matches)} similar code segments")
            
            return result_matches
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    async def detect_duplicates(
        self, 
        scope: SimilarityScope = SimilarityScope.FUNCTION_LEVEL,
        threshold: float = 0.85
    ) -> List[SimilarityMatch]:
        """
        Detect duplicate and near-duplicate code segments
        """
        duplicates = []
        segment_ids = list(self.code_segments.keys())
        
        # Compare all pairs
        for i in range(len(segment_ids)):
            for j in range(i + 1, len(segment_ids)):
                segment1 = self.code_segments[segment_ids[i]]
                segment2 = self.code_segments[segment_ids[j]]
                
                # Skip if different scope
                if not self._matches_scope(segment1, segment2, scope):
                    continue
                
                # Calculate syntactic similarity for duplicates
                match = await self._calculate_similarity(
                    segment1, 
                    segment2, 
                    SimilarityType.SYNTACTIC
                )
                
                if match and match.similarity_score >= threshold:
                    # Classify match type
                    if match.similarity_score >= self.exact_match_threshold:
                        match.match_type = MatchType.EXACT_DUPLICATE
                    else:
                        match.match_type = MatchType.NEAR_DUPLICATE
                    
                    duplicates.append(match)
        
        return sorted(duplicates, key=lambda m: m.similarity_score, reverse=True)
    
    async def analyze_code_relationships(
        self, 
        file_paths: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze relationships between code segments in specified files
        """
        relationships = {
            "files": file_paths,
            "similarity_matrix": {},
            "clusters": [],
            "refactor_opportunities": [],
            "architectural_insights": []
        }
        
        # Get segments for specified files
        relevant_segments = []
        for file_path in file_paths:
            segment_ids = self.segment_index.get(file_path, [])
            for segment_id in segment_ids:
                if segment_id in self.code_segments:
                    relevant_segments.append(self.code_segments[segment_id])
        
        if len(relevant_segments) < 2:
            return relationships
        
        # Build similarity matrix
        similarity_matrix = {}
        for i, segment1 in enumerate(relevant_segments):
            for j, segment2 in enumerate(relevant_segments):
                if i != j:
                    # Calculate multiple similarity types
                    similarities = {}
                    for sim_type in [SimilarityType.SEMANTIC, SimilarityType.STRUCTURAL]:
                        match = await self._calculate_similarity(segment1, segment2, sim_type)
                        if match:
                            similarities[sim_type.value] = match.similarity_score
                    
                    if similarities:
                        key = f"{segment1.segment_id}-{segment2.segment_id}"
                        similarity_matrix[key] = similarities
        
        relationships["similarity_matrix"] = similarity_matrix
        
        # Identify clusters
        clusters = await self._identify_code_clusters(relevant_segments, similarity_matrix)
        relationships["clusters"] = clusters
        
        # Find refactoring opportunities
        refactor_ops = await self._identify_refactor_opportunities(relevant_segments, similarity_matrix)
        relationships["refactor_opportunities"] = refactor_ops
        
        return relationships
    
    async def _generate_semantic_vectors(self, segment: CodeSegment) -> None:
        """Generate semantic vectors for a code segment"""
        try:
            # Token-based vector
            token_vector = await self._generate_token_vector(segment)
            self.semantic_vectors[f"{segment.segment_id}_token"] = token_vector
            
            # AST-based vector
            ast_vector = await self._generate_ast_vector(segment)
            self.semantic_vectors[f"{segment.segment_id}_ast"] = ast_vector
            
            # Semantic embedding vector
            semantic_vector = await self._generate_semantic_embedding(segment)
            self.semantic_vectors[f"{segment.segment_id}_semantic"] = semantic_vector
            
            # Hybrid vector (combination)
            hybrid_vector = await self._generate_hybrid_vector(segment, [token_vector, ast_vector, semantic_vector])
            self.semantic_vectors[f"{segment.segment_id}_hybrid"] = hybrid_vector
            
        except Exception as e:
            logger.error(f"Error generating semantic vectors: {e}")
    
    async def _generate_token_vector(self, segment: CodeSegment) -> SemanticVector:
        """Generate token-based vector representation"""
        # Tokenize code
        tokens = self._tokenize_code(segment.content, segment.language)
        
        # Create vocabulary-based vector (simplified TF-IDF style)
        vocab = set(tokens)
        vector_size = min(512, len(vocab) * 2)  # Reasonable size
        
        # Hash-based embedding for tokens
        embedding = np.zeros(vector_size)
        for token in tokens:
            token_hash = hash(token.lower()) % vector_size
            embedding[token_hash] += 1.0
        
        # Normalize
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
        
        return SemanticVector(
            vector_id=f"{segment.segment_id}_token",
            segment_id=segment.segment_id,
            embedding=embedding,
            vector_type="token",
            dimensionality=vector_size,
            metadata={"token_count": len(tokens), "unique_tokens": len(vocab)}
        )
    
    async def _generate_ast_vector(self, segment: CodeSegment) -> SemanticVector:
        """Generate AST-based structural vector"""
        try:
            if segment.language == "python":
                import ast
                tree = ast.parse(segment.content)
                
                # Extract AST features
                features = {
                    "node_types": [],
                    "depth": 0,
                    "complexity": 0
                }
                
                for node in ast.walk(tree):
                    features["node_types"].append(type(node).__name__)
                
                # Create feature vector
                unique_nodes = list(set(features["node_types"]))
                vector_size = max(256, len(unique_nodes))
                
                embedding = np.zeros(vector_size)
                for i, node_type in enumerate(unique_nodes):
                    count = features["node_types"].count(node_type)
                    embedding[i % vector_size] = count
                
                # Normalize
                if np.linalg.norm(embedding) > 0:
                    embedding = embedding / np.linalg.norm(embedding)
                
                return SemanticVector(
                    vector_id=f"{segment.segment_id}_ast",
                    segment_id=segment.segment_id,
                    embedding=embedding,
                    vector_type="ast",
                    dimensionality=vector_size,
                    metadata={"node_count": len(features["node_types"]), "unique_nodes": len(unique_nodes)}
                )
                
        except Exception as e:
            logger.warning(f"AST parsing failed: {e}")
        
        # Fallback to simple structural features
        return self._generate_simple_structural_vector(segment)
    
    def _generate_simple_structural_vector(self, segment: CodeSegment) -> SemanticVector:
        """Generate simple structural vector as fallback"""
        # Basic structural features
        features = {
            "lines": len(segment.content.split('\n')),
            "indentation_levels": len(set(len(line) - len(line.lstrip()) for line in segment.content.split('\n'))),
            "functions": segment.content.count('def '),
            "classes": segment.content.count('class '),
            "conditionals": segment.content.count('if ') + segment.content.count('elif '),
            "loops": segment.content.count('for ') + segment.content.count('while '),
            "exceptions": segment.content.count('try') + segment.content.count('except')
        }
        
        embedding = np.array(list(features.values()), dtype=float)
        
        # Pad to reasonable size
        if len(embedding) < 256:
            embedding = np.pad(embedding, (0, 256 - len(embedding)))
        
        # Normalize
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
        
        return SemanticVector(
            vector_id=f"{segment.segment_id}_ast",
            segment_id=segment.segment_id,
            embedding=embedding,
            vector_type="ast",
            dimensionality=len(embedding),
            metadata=features
        )
    
    async def _generate_semantic_embedding(self, segment: CodeSegment) -> SemanticVector:
        """Generate semantic embedding using language models"""
        # Simplified semantic embedding - in production would use CodeBERT, etc.
        
        # Extract semantic features
        semantic_features = {
            "imports": len([line for line in segment.content.split('\n') if line.strip().startswith(('import ', 'from '))]),
            "docstrings": segment.content.count('"""') + segment.content.count("'''"),
            "comments": len([line for line in segment.content.split('\n') if line.strip().startswith('#')]),
            "string_literals": segment.content.count('"') + segment.content.count("'"),
            "numeric_literals": len([token for token in segment.content.split() if token.replace('.', '').isdigit()]),
            "boolean_operations": segment.content.count(' and ') + segment.content.count(' or ') + segment.content.count(' not '),
            "comparisons": segment.content.count('==') + segment.content.count('!=') + segment.content.count('<=') + segment.content.count('>='),
            "assignments": segment.content.count(' = ') - segment.content.count(' == ')
        }
        
        # Create semantic hash-based embedding
        embedding_size = 768  # Common transformer size
        embedding = np.zeros(embedding_size)
        
        # Hash-based semantic features
        words = segment.content.lower().split()
        for word in words:
            if word.isalpha():  # Only alphabetic words
                word_hash = hash(word) % embedding_size
                embedding[word_hash] += 1.0
        
        # Add structural semantic features
        for i, (feature_name, value) in enumerate(semantic_features.items()):
            embedding[i % embedding_size] += value * 0.1
        
        # Normalize
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
        
        return SemanticVector(
            vector_id=f"{segment.segment_id}_semantic",
            segment_id=segment.segment_id,
            embedding=embedding,
            vector_type="semantic",
            dimensionality=embedding_size,
            metadata=semantic_features
        )
    
    async def _generate_hybrid_vector(
        self, 
        segment: CodeSegment, 
        component_vectors: List[SemanticVector]
    ) -> SemanticVector:
        """Generate hybrid vector combining multiple representations"""
        # Combine vectors with weights
        weights = [0.3, 0.3, 0.4]  # token, ast, semantic
        
        # Ensure all vectors have same dimensionality
        target_dim = max(vec.dimensionality for vec in component_vectors)
        
        combined_embedding = np.zeros(target_dim)
        total_weight = 0.0
        
        for vector, weight in zip(component_vectors, weights):
            if vector.embedding is not None:
                # Pad or truncate to target dimension
                vec_embedding = vector.embedding[:target_dim]
                if len(vec_embedding) < target_dim:
                    vec_embedding = np.pad(vec_embedding, (0, target_dim - len(vec_embedding)))
                
                combined_embedding += vec_embedding * weight
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            combined_embedding = combined_embedding / total_weight
        
        if np.linalg.norm(combined_embedding) > 0:
            combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
        
        return SemanticVector(
            vector_id=f"{segment.segment_id}_hybrid",
            segment_id=segment.segment_id,
            embedding=combined_embedding,
            vector_type="hybrid",
            dimensionality=target_dim,
            metadata={"component_weights": dict(zip(["token", "ast", "semantic"], weights))}
        )
    
    def _tokenize_code(self, code: str, language: str) -> List[str]:
        """Tokenize code into meaningful tokens"""
        import re
        
        # Remove comments and strings for tokenization
        if language == "python":
            code_no_comments = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
            code_no_strings = re.sub(r'["\'].*?["\']', 'STRING_LITERAL', code_no_comments, flags=re.DOTALL)
        else:
            code_no_strings = code
        
        # Extract tokens
        tokens = re.findall(r'\b\w+\b|[^\w\s]', code_no_strings)
        
        # Filter and normalize
        filtered_tokens = []
        for token in tokens:
            token = token.strip().lower()
            if token and len(token) > 1:
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    async def _find_similarity_candidates(
        self, 
        query_segment: CodeSegment, 
        request: SimilaritySearchRequest
    ) -> List[str]:
        """Find candidate segments for similarity comparison"""
        candidates = []
        
        # Apply file filters
        if request.file_filters:
            for file_path in request.file_filters:
                if file_path in self.segment_index:
                    candidates.extend(self.segment_index[file_path])
        else:
            # Include all segments
            candidates = list(self.code_segments.keys())
        
        # Filter by scope
        scope = SimilarityScope(request.search_scope)
        filtered_candidates = []
        
        for candidate_id in candidates:
            candidate = self.code_segments[candidate_id]
            if self._matches_scope(query_segment, candidate, scope):
                filtered_candidates.append(candidate_id)
        
        return filtered_candidates
    
    def _matches_scope(
        self, 
        segment1: CodeSegment, 
        segment2: CodeSegment, 
        scope: SimilarityScope
    ) -> bool:
        """Check if segments match the specified scope"""
        if scope == SimilarityScope.TOKEN_LEVEL:
            return True
        elif scope == SimilarityScope.LINE_LEVEL:
            return True
        elif scope == SimilarityScope.FUNCTION_LEVEL:
            return (segment1.segment_type in ["function", "method"] and 
                   segment2.segment_type in ["function", "method"])
        elif scope == SimilarityScope.CLASS_LEVEL:
            return (segment1.segment_type == "class" and segment2.segment_type == "class")
        elif scope == SimilarityScope.FILE_LEVEL:
            return True
        else:
            return True
    
    async def _calculate_similarity(
        self,
        segment1: CodeSegment,
        segment2: CodeSegment,
        similarity_type: SimilarityType
    ) -> Optional[SimilarityMatch]:
        """Calculate similarity between two code segments"""
        try:
            calculator = self.similarity_calculators[similarity_type]
            
            similarity_score = await calculator.calculate_similarity(
                segment1, 
                segment2, 
                self.semantic_vectors
            )
            
            if similarity_score < 0.1:  # Skip very low similarity
                return None
            
            # Determine match type
            match_type = self._determine_match_type(similarity_score, similarity_type)
            
            # Generate explanation
            explanation = self._generate_similarity_explanation(
                segment1, segment2, similarity_score, similarity_type
            )
            
            # Get detailed metrics
            detailed_metrics = await calculator.get_detailed_metrics(segment1, segment2)
            
            # Generate refactor suggestions if applicable
            refactor_suggestions = []
            if match_type in [MatchType.NEAR_DUPLICATE, MatchType.FUNCTIONAL_EQUIVALENT]:
                refactor_suggestions = self._generate_refactor_suggestions(segment1, segment2, similarity_score)
            
            match_id = f"{segment1.segment_id}_{segment2.segment_id}_{similarity_type.value}"
            
            return SimilarityMatch(
                match_id=match_id,
                source_segment=segment1,
                target_segment=segment2,
                similarity_score=similarity_score,
                similarity_type=similarity_type,
                match_type=match_type,
                confidence=min(similarity_score + 0.1, 1.0),
                explanation=explanation,
                similarity_metrics=detailed_metrics,
                refactor_suggestions=refactor_suggestions
            )
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return None
    
    def _determine_match_type(self, similarity_score: float, similarity_type: SimilarityType) -> MatchType:
        """Determine the type of match based on similarity score"""
        if similarity_score >= self.exact_match_threshold:
            return MatchType.EXACT_DUPLICATE
        elif similarity_score >= self.near_duplicate_threshold:
            return MatchType.NEAR_DUPLICATE
        elif similarity_score >= self.semantic_threshold:
            if similarity_type == SimilarityType.FUNCTIONAL:
                return MatchType.FUNCTIONAL_EQUIVALENT
            elif similarity_type == SimilarityType.STRUCTURAL:
                return MatchType.STRUCTURAL_SIMILAR
            else:
                return MatchType.SEMANTIC_RELATED
        else:
            return MatchType.REFACTOR_CANDIDATE
    
    def _generate_similarity_explanation(
        self,
        segment1: CodeSegment,
        segment2: CodeSegment,
        similarity_score: float,
        similarity_type: SimilarityType
    ) -> str:
        """Generate explanation for similarity match"""
        explanations = {
            SimilarityType.SYNTACTIC: f"Code segments have {similarity_score:.1%} syntactic similarity in token structure",
            SimilarityType.SEMANTIC: f"Code segments share {similarity_score:.1%} semantic meaning and purpose",
            SimilarityType.STRUCTURAL: f"Code segments have {similarity_score:.1%} structural similarity in organization",
            SimilarityType.FUNCTIONAL: f"Code segments perform {similarity_score:.1%} functionally equivalent operations",
            SimilarityType.CONTEXTUAL: f"Code segments appear in {similarity_score:.1%} similar contexts",
            SimilarityType.BEHAVIORAL: f"Code segments exhibit {similarity_score:.1%} similar behavioral patterns"
        }
        
        return explanations.get(similarity_type, f"Similarity score: {similarity_score:.1%}")
    
    def _generate_refactor_suggestions(
        self,
        segment1: CodeSegment,
        segment2: CodeSegment,
        similarity_score: float
    ) -> List[str]:
        """Generate refactoring suggestions for similar code"""
        suggestions = []
        
        if similarity_score >= self.near_duplicate_threshold:
            suggestions.append("Consider extracting common functionality into a shared function")
            suggestions.append("Evaluate parameters that differ between implementations")
        
        if segment1.segment_type == "function" and segment2.segment_type == "function":
            suggestions.append("Consider creating a higher-order function or template")
        
        if segment1.file_path != segment2.file_path:
            suggestions.append("Consider moving similar functions to a common module")
        
        return suggestions
    
    def _deduplicate_matches(self, matches: List[SimilarityMatch]) -> List[SimilarityMatch]:
        """Remove duplicate matches"""
        seen = set()
        unique_matches = []
        
        for match in matches:
            # Create key for duplicate detection
            key = (
                match.source_segment.segment_id,
                match.target_segment.segment_id,
                match.similarity_type.value
            )
            
            # Also check reverse
            reverse_key = (
                match.target_segment.segment_id,
                match.source_segment.segment_id,
                match.similarity_type.value
            )
            
            if key not in seen and reverse_key not in seen:
                seen.add(key)
                unique_matches.append(match)
        
        return unique_matches
    
    async def _identify_code_clusters(
        self,
        segments: List[CodeSegment],
        similarity_matrix: Dict[str, Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """Identify clusters of similar code segments"""
        clusters = []
        
        # Simple clustering based on similarity threshold
        clustered_segments = set()
        cluster_threshold = 0.7
        
        for i, segment1 in enumerate(segments):
            if segment1.segment_id in clustered_segments:
                continue
            
            cluster = {
                "cluster_id": f"cluster_{i}",
                "segments": [segment1.segment_id],
                "center_segment": segment1.segment_id,
                "similarity_scores": []
            }
            
            # Find similar segments
            for segment2 in segments:
                if segment2.segment_id == segment1.segment_id:
                    continue
                
                key = f"{segment1.segment_id}-{segment2.segment_id}"
                reverse_key = f"{segment2.segment_id}-{segment1.segment_id}"
                
                similarities = similarity_matrix.get(key) or similarity_matrix.get(reverse_key)
                if similarities:
                    max_similarity = max(similarities.values())
                    if max_similarity >= cluster_threshold:
                        cluster["segments"].append(segment2.segment_id)
                        cluster["similarity_scores"].append(max_similarity)
                        clustered_segments.add(segment2.segment_id)
            
            if len(cluster["segments"]) > 1:
                clustered_segments.add(segment1.segment_id)
                clusters.append(cluster)
        
        return clusters
    
    async def _identify_refactor_opportunities(
        self,
        segments: List[CodeSegment],
        similarity_matrix: Dict[str, Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """Identify refactoring opportunities based on similarity analysis"""
        opportunities = []
        
        duplicate_threshold = 0.8
        
        for key, similarities in similarity_matrix.items():
            segment_ids = key.split('-')
            if len(segment_ids) == 2:
                max_similarity = max(similarities.values())
                
                if max_similarity >= duplicate_threshold:
                    segment1_id, segment2_id = segment_ids
                    segment1 = next((s for s in segments if s.segment_id == segment1_id), None)
                    segment2 = next((s for s in segments if s.segment_id == segment2_id), None)
                    
                    if segment1 and segment2:
                        opportunity = {
                            "opportunity_id": f"refactor_{len(opportunities)}",
                            "type": "extract_common_function",
                            "segments": [segment1_id, segment2_id],
                            "similarity_score": max_similarity,
                            "description": f"Extract common logic from similar functions in {segment1.file_path} and {segment2.file_path}",
                            "estimated_effort": "medium",
                            "potential_benefits": ["Reduce code duplication", "Improve maintainability", "Single point of change"]
                        }
                        opportunities.append(opportunity)
        
        return opportunities
    
    async def get_similarity_statistics(self) -> Dict[str, Any]:
        """Get statistics about similarity analysis"""
        stats = {
            "total_segments": len(self.code_segments),
            "total_vectors": len(self.semantic_vectors),
            "cache_size": len(self.similarity_cache),
            "segment_types": Counter(segment.segment_type for segment in self.code_segments.values()),
            "file_distribution": {
                file_path: len(segment_ids) 
                for file_path, segment_ids in self.segment_index.items()
            }
        }
        
        return stats


# Similarity calculator classes

class SimilarityCalculatorBase:
    """Base class for similarity calculators"""
    
    async def calculate_similarity(
        self,
        segment1: CodeSegment,
        segment2: CodeSegment,
        vectors: Dict[str, SemanticVector]
    ) -> float:
        raise NotImplementedError
    
    async def get_detailed_metrics(
        self,
        segment1: CodeSegment,
        segment2: CodeSegment
    ) -> Dict[str, float]:
        return {}


class SyntacticSimilarityCalculator(SimilarityCalculatorBase):
    """Calculate syntactic similarity using token-based comparison"""
    
    async def calculate_similarity(
        self,
        segment1: CodeSegment,
        segment2: CodeSegment,
        vectors: Dict[str, SemanticVector]
    ) -> float:
        # Get token vectors
        vector1_key = f"{segment1.segment_id}_token"
        vector2_key = f"{segment2.segment_id}_token"
        
        if vector1_key in vectors and vector2_key in vectors:
            vec1 = vectors[vector1_key].embedding
            vec2 = vectors[vector2_key].embedding
            
            # Cosine similarity
            return self._cosine_similarity(vec1, vec2)
        
        # Fallback to simple token comparison
        tokens1 = set(segment1.content.lower().split())
        tokens2 = set(segment2.content.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        # Ensure same dimensions
        min_dim = min(len(vec1), len(vec2))
        vec1 = vec1[:min_dim]
        vec2 = vec2[:min_dim]
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def get_detailed_metrics(
        self,
        segment1: CodeSegment,
        segment2: CodeSegment
    ) -> Dict[str, float]:
        tokens1 = segment1.content.lower().split()
        tokens2 = segment2.content.lower().split()
        
        return {
            "token_jaccard": len(set(tokens1) & set(tokens2)) / len(set(tokens1) | set(tokens2)) if tokens1 or tokens2 else 0.0,
            "token_overlap": len(set(tokens1) & set(tokens2)) / min(len(set(tokens1)), len(set(tokens2))) if tokens1 and tokens2 else 0.0,
            "length_ratio": min(len(tokens1), len(tokens2)) / max(len(tokens1), len(tokens2)) if tokens1 and tokens2 else 0.0
        }


class SemanticSimilarityCalculator(SimilarityCalculatorBase):
    """Calculate semantic similarity using embeddings"""
    
    async def calculate_similarity(
        self,
        segment1: CodeSegment,
        segment2: CodeSegment,
        vectors: Dict[str, SemanticVector]
    ) -> float:
        # Use semantic vectors
        vector1_key = f"{segment1.segment_id}_semantic"
        vector2_key = f"{segment2.segment_id}_semantic"
        
        if vector1_key in vectors and vector2_key in vectors:
            vec1 = vectors[vector1_key].embedding
            vec2 = vectors[vector2_key].embedding
            
            return self._cosine_similarity(vec1, vec2)
        
        return 0.0
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        min_dim = min(len(vec1), len(vec2))
        vec1 = vec1[:min_dim]
        vec2 = vec2[:min_dim]
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class StructuralSimilarityCalculator(SimilarityCalculatorBase):
    """Calculate structural similarity using AST-based comparison"""
    
    async def calculate_similarity(
        self,
        segment1: CodeSegment,
        segment2: CodeSegment,
        vectors: Dict[str, SemanticVector]
    ) -> float:
        # Use AST vectors
        vector1_key = f"{segment1.segment_id}_ast"
        vector2_key = f"{segment2.segment_id}_ast"
        
        if vector1_key in vectors and vector2_key in vectors:
            vec1 = vectors[vector1_key].embedding
            vec2 = vectors[vector2_key].embedding
            
            return self._cosine_similarity(vec1, vec2)
        
        return 0.0
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        min_dim = min(len(vec1), len(vec2))
        vec1 = vec1[:min_dim]
        vec2 = vec2[:min_dim]
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class FunctionalSimilarityCalculator(SimilarityCalculatorBase):
    """Calculate functional similarity based on behavior"""
    
    async def calculate_similarity(
        self,
        segment1: CodeSegment,
        segment2: CodeSegment,
        vectors: Dict[str, SemanticVector]
    ) -> float:
        # Simplified functional similarity
        # In production, would analyze input/output behavior
        
        # Check for similar function signatures
        if segment1.segment_type == "function" and segment2.segment_type == "function":
            return self._compare_function_signatures(segment1.content, segment2.content)
        
        return 0.0
    
    def _compare_function_signatures(self, code1: str, code2: str) -> float:
        """Compare function signatures"""
        import re
        
        # Extract function definitions
        func_pattern = r'def\s+(\w+)\s*\((.*?)\):'
        
        match1 = re.search(func_pattern, code1)
        match2 = re.search(func_pattern, code2)
        
        if not match1 or not match2:
            return 0.0
        
        # Compare parameter counts
        params1 = [p.strip() for p in match1.group(2).split(',') if p.strip()]
        params2 = [p.strip() for p in match2.group(2).split(',') if p.strip()]
        
        if len(params1) == len(params2):
            return 0.7  # Similar parameter count
        else:
            return 0.3  # Different parameter count


class ContextualSimilarityCalculator(SimilarityCalculatorBase):
    """Calculate contextual similarity based on usage context"""
    
    async def calculate_similarity(
        self,
        segment1: CodeSegment,
        segment2: CodeSegment,
        vectors: Dict[str, SemanticVector]
    ) -> float:
        # Simple contextual similarity based on imports and surrounding context
        similarity = 0.0
        
        # Compare imports
        imports1 = [line for line in segment1.content.split('\n') if line.strip().startswith(('import ', 'from '))]
        imports2 = [line for line in segment2.content.split('\n') if line.strip().startswith(('import ', 'from '))]
        
        if imports1 or imports2:
            common_imports = len(set(imports1) & set(imports2))
            total_imports = len(set(imports1) | set(imports2))
            
            if total_imports > 0:
                similarity += (common_imports / total_imports) * 0.5
        
        # Compare file paths (similar directories might have similar context)
        path1_parts = segment1.file_path.split('/')
        path2_parts = segment2.file_path.split('/')
        
        common_path_parts = len(set(path1_parts) & set(path2_parts))
        total_path_parts = len(set(path1_parts) | set(path2_parts))
        
        if total_path_parts > 0:
            similarity += (common_path_parts / total_path_parts) * 0.5
        
        return min(similarity, 1.0)


class BehavioralSimilarityCalculator(SimilarityCalculatorBase):
    """Calculate behavioral similarity based on execution patterns"""
    
    async def calculate_similarity(
        self,
        segment1: CodeSegment,
        segment2: CodeSegment,
        vectors: Dict[str, SemanticVector]
    ) -> float:
        # Simplified behavioral similarity
        # In production, would analyze execution traces or patterns
        
        # Look for similar control structures
        control_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'with']
        
        patterns1 = []
        patterns2 = []
        
        for keyword in control_keywords:
            patterns1.append(segment1.content.count(keyword))
            patterns2.append(segment2.content.count(keyword))
        
        # Calculate pattern similarity
        if sum(patterns1) == 0 and sum(patterns2) == 0:
            return 0.5  # Both have no control structures
        
        # Normalize patterns
        norm1 = sum(patterns1)
        norm2 = sum(patterns2)
        
        if norm1 > 0:
            patterns1 = [p / norm1 for p in patterns1]
        if norm2 > 0:
            patterns2 = [p / norm2 for p in patterns2]
        
        # Calculate cosine similarity of control patterns
        dot_product = sum(p1 * p2 for p1, p2 in zip(patterns1, patterns2))
        norm1 = math.sqrt(sum(p * p for p in patterns1))
        norm2 = math.sqrt(sum(p * p for p in patterns2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)