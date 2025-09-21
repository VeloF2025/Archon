"""
🚀 ARCHON ENHANCEMENT 2025 - PHASE 6: ADVANCED AI INTEGRATION  
Knowledge Graph Intelligence - Advanced Graph-Based Knowledge Management

This module provides a comprehensive knowledge graph intelligence system with advanced
capabilities including entity linking, relation extraction, semantic reasoning,
graph embeddings, and intelligent knowledge discovery with enterprise-grade scalability.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict, Counter, deque
import hashlib
import uuid
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Knowledge graph entity types."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    CONCEPT = "concept"
    DOCUMENT = "document"
    PRODUCT = "product"
    SERVICE = "service"
    TECHNOLOGY = "technology"
    PROCESS = "process"
    SKILL = "skill"
    TOOL = "tool"
    PROJECT = "project"
    DATASET = "dataset"
    MODEL = "model"
    UNKNOWN = "unknown"


class RelationType(Enum):
    """Knowledge graph relation types."""
    # Basic relations
    IS_A = "is_a"
    PART_OF = "part_of"
    HAS_PROPERTY = "has_property"
    LOCATED_IN = "located_in"
    WORKS_FOR = "works_for"
    CREATED_BY = "created_by"
    USES = "uses"
    DEPENDS_ON = "depends_on"
    
    # Semantic relations
    SIMILAR_TO = "similar_to"
    RELATED_TO = "related_to"
    OPPOSITE_OF = "opposite_of"
    CAUSES = "causes"
    ENABLES = "enables"
    REQUIRES = "requires"
    
    # Temporal relations
    HAPPENS_BEFORE = "happens_before"
    HAPPENS_AFTER = "happens_after"
    HAPPENS_DURING = "happens_during"
    
    # Social relations
    COLLABORATES_WITH = "collaborates_with"
    REPORTS_TO = "reports_to"
    MENTORS = "mentors"
    COMPETES_WITH = "competes_with"


class ConfidenceLevel(Enum):
    """Confidence levels for knowledge assertions."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CERTAIN = "certain"


@dataclass
class KnowledgeEntity:
    """Knowledge graph entity representation."""
    entity_id: str
    name: str
    entity_type: EntityType
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    confidence: float = 1.0
    source: str = "manual"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeRelation:
    """Knowledge graph relation representation."""
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType
    confidence: float
    properties: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    source: str = "manual"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgePath:
    """Path between entities in knowledge graph."""
    path_id: str
    entities: List[str]  # Entity IDs
    relations: List[str]  # Relation IDs
    path_length: int
    total_confidence: float
    semantic_similarity: float = 0.0
    explanation: str = ""


@dataclass
class SemanticQuery:
    """Semantic query for knowledge graph."""
    query_id: str
    query_text: str
    entity_mentions: List[str] = field(default_factory=list)
    relation_mentions: List[str] = field(default_factory=list)
    intent: str = "search"
    constraints: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Result of semantic query."""
    result_id: str
    query_id: str
    entities: List[KnowledgeEntity]
    relations: List[KnowledgeRelation]
    paths: List[KnowledgePath] = field(default_factory=list)
    confidence: float = 0.0
    explanation: str = ""
    reasoning_steps: List[str] = field(default_factory=list)
    processing_time: float = 0.0


@dataclass
class ReasoningRule:
    """Rule for semantic reasoning."""
    rule_id: str
    name: str
    premise_pattern: Dict[str, Any]
    conclusion_pattern: Dict[str, Any]
    confidence_factor: float
    priority: int = 0
    is_active: bool = True


class BaseEntityLinker(ABC):
    """Abstract base class for entity linking."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the entity linker."""
        pass
    
    @abstractmethod
    async def link_entities(self, text: str, entities: List[KnowledgeEntity]) -> List[Tuple[str, str, float]]:
        """Link text mentions to knowledge entities."""
        pass


class BaseRelationExtractor(ABC):
    """Abstract base class for relation extraction."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the relation extractor."""
        pass
    
    @abstractmethod
    async def extract_relations(self, text: str, entities: List[KnowledgeEntity]) -> List[KnowledgeRelation]:
        """Extract relations from text given entities."""
        pass


class SimpleEntityLinker(BaseEntityLinker):
    """Simple rule-based entity linker."""
    
    async def initialize(self) -> None:
        """Initialize the entity linker."""
        logger.info("Initializing simple entity linker...")
        await asyncio.sleep(0.1)
        self.is_initialized = True
        logger.info("Simple entity linker initialized")
    
    async def link_entities(self, text: str, entities: List[KnowledgeEntity]) -> List[Tuple[str, str, float]]:
        """Link text mentions to knowledge entities."""
        if not self.is_initialized:
            await self.initialize()
        
        links = []
        text_lower = text.lower()
        
        for entity in entities:
            # Check exact name match
            if entity.name.lower() in text_lower:
                confidence = 0.9
                links.append((entity.name, entity.entity_id, confidence))
            
            # Check alias matches
            for alias in entity.aliases:
                if alias.lower() in text_lower:
                    confidence = 0.8
                    links.append((alias, entity.entity_id, confidence))
        
        return links


class PatternBasedRelationExtractor(BaseRelationExtractor):
    """Pattern-based relation extraction."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.relation_patterns = {
            RelationType.WORKS_FOR: [
                r"(\w+) works for (\w+)",
                r"(\w+) is employed by (\w+)",
                r"(\w+) is an employee of (\w+)"
            ],
            RelationType.CREATED_BY: [
                r"(\w+) created (\w+)",
                r"(\w+) developed (\w+)",
                r"(\w+) built (\w+)"
            ],
            RelationType.USES: [
                r"(\w+) uses (\w+)",
                r"(\w+) utilizes (\w+)",
                r"(\w+) leverages (\w+)"
            ],
            RelationType.LOCATED_IN: [
                r"(\w+) is located in (\w+)",
                r"(\w+) is based in (\w+)",
                r"(\w+) is situated in (\w+)"
            ]
        }
    
    async def initialize(self) -> None:
        """Initialize the relation extractor."""
        logger.info("Initializing pattern-based relation extractor...")
        await asyncio.sleep(0.1)
        self.is_initialized = True
        logger.info("Pattern-based relation extractor initialized")
    
    async def extract_relations(self, text: str, entities: List[KnowledgeEntity]) -> List[KnowledgeRelation]:
        """Extract relations from text given entities."""
        if not self.is_initialized:
            await self.initialize()
        
        relations = []
        entity_name_to_id = {entity.name.lower(): entity.entity_id for entity in entities}
        
        # Add aliases to mapping
        for entity in entities:
            for alias in entity.aliases:
                entity_name_to_id[alias.lower()] = entity.entity_id
        
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                import re
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    source_name = match.group(1).lower()
                    target_name = match.group(2).lower()
                    
                    source_id = entity_name_to_id.get(source_name)
                    target_id = entity_name_to_id.get(target_name)
                    
                    if source_id and target_id and source_id != target_id:
                        relation = KnowledgeRelation(
                            relation_id=f"rel_{uuid.uuid4().hex[:8]}",
                            source_entity_id=source_id,
                            target_entity_id=target_id,
                            relation_type=relation_type,
                            confidence=0.8,
                            evidence=[match.group(0)],
                            source="pattern_extraction"
                        )
                        relations.append(relation)
        
        return relations


class KnowledgeGraphStore:
    """Knowledge graph storage and indexing system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # In-memory storage (in production, would use graph database)
        self.entities: Dict[str, KnowledgeEntity] = {}
        self.relations: Dict[str, KnowledgeRelation] = {}
        
        # Indexes for fast lookups
        self.entity_name_index: Dict[str, Set[str]] = defaultdict(set)
        self.entity_type_index: Dict[EntityType, Set[str]] = defaultdict(set)
        self.relation_type_index: Dict[RelationType, Set[str]] = defaultdict(set)
        self.adjacency_list: Dict[str, Set[str]] = defaultdict(set)  # entity_id -> set of relation_ids
        
        # Caching
        self.path_cache: Dict[str, List[KnowledgePath]] = {}
        self.max_cache_size = config.get('max_cache_size', 1000)
    
    async def add_entity(self, entity: KnowledgeEntity) -> None:
        """Add entity to knowledge graph."""
        self.entities[entity.entity_id] = entity
        
        # Update indexes
        self.entity_name_index[entity.name.lower()].add(entity.entity_id)
        for alias in entity.aliases:
            self.entity_name_index[alias.lower()].add(entity.entity_id)
        
        self.entity_type_index[entity.entity_type].add(entity.entity_id)
        
        logger.debug(f"Added entity: {entity.name} ({entity.entity_id})")
    
    async def add_relation(self, relation: KnowledgeRelation) -> None:
        """Add relation to knowledge graph."""
        self.relations[relation.relation_id] = relation
        
        # Update indexes
        self.relation_type_index[relation.relation_type].add(relation.relation_id)
        self.adjacency_list[relation.source_entity_id].add(relation.relation_id)
        self.adjacency_list[relation.target_entity_id].add(relation.relation_id)
        
        # Clear path cache as graph structure changed
        self.path_cache.clear()
        
        logger.debug(f"Added relation: {relation.source_entity_id} -> {relation.target_entity_id}")
    
    async def get_entity(self, entity_id: str) -> Optional[KnowledgeEntity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)
    
    async def get_relation(self, relation_id: str) -> Optional[KnowledgeRelation]:
        """Get relation by ID."""
        return self.relations.get(relation_id)
    
    async def find_entities_by_name(self, name: str) -> List[KnowledgeEntity]:
        """Find entities by name (case-insensitive)."""
        entity_ids = self.entity_name_index.get(name.lower(), set())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]
    
    async def find_entities_by_type(self, entity_type: EntityType) -> List[KnowledgeEntity]:
        """Find entities by type."""
        entity_ids = self.entity_type_index.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]
    
    async def get_entity_relations(self, entity_id: str, relation_type: Optional[RelationType] = None) -> List[KnowledgeRelation]:
        """Get all relations for an entity, optionally filtered by type."""
        relation_ids = self.adjacency_list.get(entity_id, set())
        relations = [self.relations[rid] for rid in relation_ids if rid in self.relations]
        
        if relation_type:
            relations = [rel for rel in relations if rel.relation_type == relation_type]
        
        return relations
    
    async def get_neighbors(self, entity_id: str, max_distance: int = 1) -> List[Tuple[KnowledgeEntity, int]]:
        """Get neighboring entities within max_distance."""
        if max_distance < 1:
            return []
        
        visited = set()
        neighbors = []
        queue = deque([(entity_id, 0)])
        
        while queue:
            current_id, distance = queue.popleft()
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            if distance > 0:  # Don't include the source entity
                entity = self.entities.get(current_id)
                if entity:
                    neighbors.append((entity, distance))
            
            if distance < max_distance:
                # Add neighbors to queue
                relation_ids = self.adjacency_list.get(current_id, set())
                for relation_id in relation_ids:
                    relation = self.relations.get(relation_id)
                    if relation:
                        # Add both source and target entities
                        if relation.source_entity_id not in visited:
                            queue.append((relation.source_entity_id, distance + 1))
                        if relation.target_entity_id not in visited:
                            queue.append((relation.target_entity_id, distance + 1))
        
        return neighbors
    
    async def find_path(self, source_id: str, target_id: str, max_length: int = 5) -> List[KnowledgePath]:
        """Find paths between two entities."""
        cache_key = f"{source_id}_{target_id}_{max_length}"
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        paths = []
        visited = set()
        
        # BFS to find shortest paths
        queue = deque([([source_id], [], 0.0)])
        
        while queue:
            current_path, relation_path, confidence = queue.popleft()
            current_id = current_path[-1]
            
            if len(current_path) > max_length:
                continue
            
            if current_id == target_id and len(current_path) > 1:
                # Found a path
                path = KnowledgePath(
                    path_id=f"path_{uuid.uuid4().hex[:8]}",
                    entities=current_path,
                    relations=relation_path,
                    path_length=len(current_path) - 1,
                    total_confidence=confidence / len(current_path) if current_path else 0.0
                )
                paths.append(path)
                continue
            
            # Explore neighbors
            relation_ids = self.adjacency_list.get(current_id, set())
            for relation_id in relation_ids:
                relation = self.relations.get(relation_id)
                if not relation:
                    continue
                
                # Determine next entity
                next_entity_id = None
                if relation.source_entity_id == current_id:
                    next_entity_id = relation.target_entity_id
                elif relation.target_entity_id == current_id:
                    next_entity_id = relation.source_entity_id
                
                if next_entity_id and next_entity_id not in current_path:
                    new_path = current_path + [next_entity_id]
                    new_relation_path = relation_path + [relation_id]
                    new_confidence = confidence + relation.confidence
                    
                    queue.append((new_path, new_relation_path, new_confidence))
        
        # Cache results
        if len(self.path_cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.path_cache))
            del self.path_cache[oldest_key]
        
        self.path_cache[cache_key] = paths
        return paths
    
    async def get_subgraph(self, entity_ids: List[str], include_neighbors: bool = True) -> Dict[str, Any]:
        """Get subgraph containing specified entities."""
        subgraph_entities = {}
        subgraph_relations = {}
        
        # Collect entities
        all_entity_ids = set(entity_ids)
        
        if include_neighbors:
            # Add immediate neighbors
            for entity_id in entity_ids:
                neighbors = await self.get_neighbors(entity_id, max_distance=1)
                for neighbor_entity, _ in neighbors:
                    all_entity_ids.add(neighbor_entity.entity_id)
        
        # Get entities
        for entity_id in all_entity_ids:
            entity = self.entities.get(entity_id)
            if entity:
                subgraph_entities[entity_id] = entity
        
        # Get relations between entities in subgraph
        for entity_id in all_entity_ids:
            relations = await self.get_entity_relations(entity_id)
            for relation in relations:
                if (relation.source_entity_id in all_entity_ids and 
                    relation.target_entity_id in all_entity_ids):
                    subgraph_relations[relation.relation_id] = relation
        
        return {
            'entities': subgraph_entities,
            'relations': subgraph_relations,
            'stats': {
                'entity_count': len(subgraph_entities),
                'relation_count': len(subgraph_relations)
            }
        }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        entity_type_counts = {}
        for entity_type in EntityType:
            count = len(self.entity_type_index.get(entity_type, set()))
            if count > 0:
                entity_type_counts[entity_type.value] = count
        
        relation_type_counts = {}
        for relation_type in RelationType:
            count = len(self.relation_type_index.get(relation_type, set()))
            if count > 0:
                relation_type_counts[relation_type.value] = count
        
        return {
            'total_entities': len(self.entities),
            'total_relations': len(self.relations),
            'entity_types': entity_type_counts,
            'relation_types': relation_type_counts,
            'avg_entity_degree': sum(len(relations) for relations in self.adjacency_list.values()) / len(self.entities) if self.entities else 0,
            'cache_size': len(self.path_cache)
        }


class SemanticReasoner:
    """Semantic reasoning engine for knowledge graphs."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reasoning_rules: Dict[str, ReasoningRule] = {}
        self.inference_cache: Dict[str, List[KnowledgeRelation]] = {}
        self.max_reasoning_depth = config.get('max_reasoning_depth', 3)
    
    async def add_reasoning_rule(self, rule: ReasoningRule) -> None:
        """Add a reasoning rule."""
        self.reasoning_rules[rule.rule_id] = rule
        # Clear inference cache when rules change
        self.inference_cache.clear()
        logger.info(f"Added reasoning rule: {rule.name}")
    
    async def infer_relations(self, graph_store: KnowledgeGraphStore) -> List[KnowledgeRelation]:
        """Infer new relations using reasoning rules."""
        inferred_relations = []
        
        # Apply reasoning rules iteratively
        for depth in range(self.max_reasoning_depth):
            new_relations = await self._apply_rules_once(graph_store, depth)
            
            if not new_relations:
                break  # No new inferences
            
            # Add new relations to graph for next iteration
            for relation in new_relations:
                await graph_store.add_relation(relation)
            
            inferred_relations.extend(new_relations)
        
        logger.info(f"Inferred {len(inferred_relations)} new relations")
        return inferred_relations
    
    async def _apply_rules_once(self, graph_store: KnowledgeGraphStore, depth: int) -> List[KnowledgeRelation]:
        """Apply reasoning rules once."""
        new_relations = []
        
        for rule in self.reasoning_rules.values():
            if not rule.is_active:
                continue
            
            # Apply rule to generate new relations
            rule_inferences = await self._apply_single_rule(graph_store, rule, depth)
            new_relations.extend(rule_inferences)
        
        return new_relations
    
    async def _apply_single_rule(self, graph_store: KnowledgeGraphStore, rule: ReasoningRule, depth: int) -> List[KnowledgeRelation]:
        """Apply a single reasoning rule."""
        inferences = []
        
        # Simple rule application (in production, would be more sophisticated)
        premise = rule.premise_pattern
        conclusion = rule.conclusion_pattern
        
        # Example rule: if A works_for B and B part_of C, then A works_for C
        if (premise.get('relation1') == 'works_for' and 
            premise.get('relation2') == 'part_of' and
            conclusion.get('relation') == 'works_for'):
            
            # Find all works_for relations
            works_for_relations = []
            part_of_relations = []
            
            for relation in graph_store.relations.values():
                if relation.relation_type == RelationType.WORKS_FOR:
                    works_for_relations.append(relation)
                elif relation.relation_type == RelationType.PART_OF:
                    part_of_relations.append(relation)
            
            # Find matching patterns
            for wf_rel in works_for_relations:
                for po_rel in part_of_relations:
                    if wf_rel.target_entity_id == po_rel.source_entity_id:
                        # A works_for B, B part_of C => A works_for C
                        
                        # Check if this relation already exists
                        existing = await self._relation_exists(
                            graph_store, wf_rel.source_entity_id, po_rel.target_entity_id, RelationType.WORKS_FOR
                        )
                        
                        if not existing:
                            inferred_relation = KnowledgeRelation(
                                relation_id=f"inf_{uuid.uuid4().hex[:8]}",
                                source_entity_id=wf_rel.source_entity_id,
                                target_entity_id=po_rel.target_entity_id,
                                relation_type=RelationType.WORKS_FOR,
                                confidence=min(wf_rel.confidence, po_rel.confidence) * rule.confidence_factor,
                                source="inference",
                                evidence=[wf_rel.relation_id, po_rel.relation_id],
                                metadata={'rule_id': rule.rule_id, 'inference_depth': depth}
                            )
                            inferences.append(inferred_relation)
        
        return inferences
    
    async def _relation_exists(self, graph_store: KnowledgeGraphStore, source_id: str, target_id: str, relation_type: RelationType) -> bool:
        """Check if a relation already exists."""
        source_relations = await graph_store.get_entity_relations(source_id, relation_type)
        
        for relation in source_relations:
            if (relation.target_entity_id == target_id or 
                relation.source_entity_id == target_id):
                return True
        
        return False
    
    async def explain_reasoning(self, relation: KnowledgeRelation, graph_store: KnowledgeGraphStore) -> str:
        """Generate explanation for inferred relation."""
        if relation.source != "inference":
            return f"Direct relation from {relation.source}"
        
        explanation_parts = []
        
        # Get evidence relations
        for evidence_id in relation.evidence:
            evidence_relation = await graph_store.get_relation(evidence_id)
            if evidence_relation:
                source_entity = await graph_store.get_entity(evidence_relation.source_entity_id)
                target_entity = await graph_store.get_entity(evidence_relation.target_entity_id)
                
                if source_entity and target_entity:
                    explanation_parts.append(
                        f"{source_entity.name} {evidence_relation.relation_type.value.replace('_', ' ')} {target_entity.name}"
                    )
        
        rule_id = relation.metadata.get('rule_id')
        rule = self.reasoning_rules.get(rule_id) if rule_id else None
        rule_name = rule.name if rule else "unknown rule"
        
        base_explanation = " and ".join(explanation_parts)
        return f"Inferred using {rule_name}: {base_explanation}"


class QueryProcessor:
    """Semantic query processor for knowledge graphs."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.entity_linker = SimpleEntityLinker(config.get('entity_linking', {}))
        self.query_cache: Dict[str, QueryResult] = {}
        self.max_cache_size = config.get('max_cache_size', 500)
    
    async def initialize(self) -> None:
        """Initialize query processor components."""
        await self.entity_linker.initialize()
        logger.info("Query processor initialized")
    
    async def process_query(self, query: SemanticQuery, graph_store: KnowledgeGraphStore) -> QueryResult:
        """Process semantic query against knowledge graph."""
        if not self.entity_linker.is_initialized:
            await self.initialize()
        
        start_time = asyncio.get_event_loop().time()
        
        # Check cache
        cache_key = self._generate_query_cache_key(query)
        if cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            logger.info(f"Returning cached query result: {query.query_id}")
            return cached_result
        
        result = QueryResult(
            result_id=f"result_{uuid.uuid4().hex[:8]}",
            query_id=query.query_id
        )
        
        # Extract entities from query
        query_entities = await self._extract_query_entities(query, graph_store)
        
        # Process based on query intent
        if query.intent == "search":
            await self._process_search_query(query, graph_store, result, query_entities)
        elif query.intent == "explore":
            await self._process_exploration_query(query, graph_store, result, query_entities)
        elif query.intent == "path":
            await self._process_path_query(query, graph_store, result, query_entities)
        else:
            await self._process_general_query(query, graph_store, result, query_entities)
        
        # Calculate processing time and confidence
        processing_time = asyncio.get_event_loop().time() - start_time
        result.processing_time = processing_time
        result.confidence = self._calculate_result_confidence(result)
        
        # Generate explanation
        result.explanation = await self._generate_query_explanation(query, result, graph_store)
        
        # Cache result
        self._cache_query_result(cache_key, result)
        
        return result
    
    async def _extract_query_entities(self, query: SemanticQuery, graph_store: KnowledgeGraphStore) -> List[KnowledgeEntity]:
        """Extract entities mentioned in query."""
        # Get all entities for linking
        all_entities = list(graph_store.entities.values())
        
        # Link entities in query text
        entity_links = await self.entity_linker.link_entities(query.query_text, all_entities)
        
        # Resolve to actual entities
        linked_entities = []
        for mention, entity_id, confidence in entity_links:
            entity = await graph_store.get_entity(entity_id)
            if entity:
                linked_entities.append(entity)
        
        return linked_entities
    
    async def _process_search_query(self, query: SemanticQuery, graph_store: KnowledgeGraphStore, result: QueryResult, query_entities: List[KnowledgeEntity]) -> None:
        """Process search query."""
        result.entities = query_entities
        
        # Get relations involving query entities
        for entity in query_entities:
            entity_relations = await graph_store.get_entity_relations(entity.entity_id)
            result.relations.extend(entity_relations)
        
        result.reasoning_steps.append(f"Found {len(query_entities)} entities matching query")
        result.reasoning_steps.append(f"Retrieved {len(result.relations)} related relations")
    
    async def _process_exploration_query(self, query: SemanticQuery, graph_store: KnowledgeGraphStore, result: QueryResult, query_entities: List[KnowledgeEntity]) -> None:
        """Process exploration query."""
        result.entities = query_entities
        
        # Explore neighborhood of query entities
        for entity in query_entities:
            neighbors = await graph_store.get_neighbors(entity.entity_id, max_distance=2)
            
            for neighbor_entity, distance in neighbors:
                if neighbor_entity not in result.entities:
                    result.entities.append(neighbor_entity)
        
        # Get relations among all discovered entities
        entity_ids = {entity.entity_id for entity in result.entities}
        for entity_id in entity_ids:
            entity_relations = await graph_store.get_entity_relations(entity_id)
            for relation in entity_relations:
                if (relation.source_entity_id in entity_ids and 
                    relation.target_entity_id in entity_ids):
                    if relation not in result.relations:
                        result.relations.append(relation)
        
        result.reasoning_steps.append(f"Explored neighborhood of {len(query_entities)} seed entities")
        result.reasoning_steps.append(f"Discovered {len(result.entities)} total entities")
    
    async def _process_path_query(self, query: SemanticQuery, graph_store: KnowledgeGraphStore, result: QueryResult, query_entities: List[KnowledgeEntity]) -> None:
        """Process path-finding query."""
        if len(query_entities) >= 2:
            # Find paths between first two entities
            source = query_entities[0]
            target = query_entities[1]
            
            paths = await graph_store.find_path(source.entity_id, target.entity_id)
            result.paths = paths[:5]  # Limit to top 5 paths
            
            # Include entities and relations from paths
            entity_ids = set()
            relation_ids = set()
            
            for path in result.paths:
                entity_ids.update(path.entities)
                relation_ids.update(path.relations)
            
            # Get entities and relations
            for entity_id in entity_ids:
                entity = await graph_store.get_entity(entity_id)
                if entity and entity not in result.entities:
                    result.entities.append(entity)
            
            for relation_id in relation_ids:
                relation = await graph_store.get_relation(relation_id)
                if relation and relation not in result.relations:
                    result.relations.append(relation)
            
            result.reasoning_steps.append(f"Found {len(result.paths)} paths between {source.name} and {target.name}")
        else:
            result.reasoning_steps.append("Path query requires at least 2 entities")
    
    async def _process_general_query(self, query: SemanticQuery, graph_store: KnowledgeGraphStore, result: QueryResult, query_entities: List[KnowledgeEntity]) -> None:
        """Process general query."""
        # Default to search behavior
        await self._process_search_query(query, graph_store, result, query_entities)
    
    def _calculate_result_confidence(self, result: QueryResult) -> float:
        """Calculate confidence score for query result."""
        if not result.entities and not result.relations:
            return 0.0
        
        entity_confidences = [entity.confidence for entity in result.entities]
        relation_confidences = [relation.confidence for relation in result.relations]
        
        all_confidences = entity_confidences + relation_confidences
        
        if not all_confidences:
            return 0.0
        
        return np.mean(all_confidences)
    
    async def _generate_query_explanation(self, query: SemanticQuery, result: QueryResult, graph_store: KnowledgeGraphStore) -> str:
        """Generate explanation for query result."""
        explanation_parts = [
            f"Query: '{query.query_text}' (Intent: {query.intent})",
            f"Found {len(result.entities)} entities and {len(result.relations)} relations"
        ]
        
        if result.paths:
            explanation_parts.append(f"Discovered {len(result.paths)} paths")
        
        explanation_parts.extend(result.reasoning_steps)
        
        return " | ".join(explanation_parts)
    
    def _generate_query_cache_key(self, query: SemanticQuery) -> str:
        """Generate cache key for query."""
        key_components = [
            query.query_text,
            query.intent,
            str(sorted(query.entity_mentions)),
            str(sorted(query.relation_mentions)),
            str(sorted(query.constraints.items()))
        ]
        
        combined_key = '|'.join(key_components)
        return hashlib.sha256(combined_key.encode()).hexdigest()[:16]
    
    def _cache_query_result(self, cache_key: str, result: QueryResult) -> None:
        """Cache query result."""
        if len(self.query_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        self.query_cache[cache_key] = result


class KnowledgeGraphIntelligence:
    """Main orchestrator for knowledge graph intelligence system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.graph_store = KnowledgeGraphStore(config.get('storage', {}))
        self.entity_linker = SimpleEntityLinker(config.get('entity_linking', {}))
        self.relation_extractor = PatternBasedRelationExtractor(config.get('relation_extraction', {}))
        self.reasoner = SemanticReasoner(config.get('reasoning', {}))
        self.query_processor = QueryProcessor(config.get('query_processing', {}))
        
        # Knowledge extraction statistics
        self.extraction_stats = {
            'documents_processed': 0,
            'entities_extracted': 0,
            'relations_extracted': 0,
            'inferences_made': 0
        }
    
    async def initialize(self) -> None:
        """Initialize knowledge graph intelligence system."""
        logger.info("Initializing Knowledge Graph Intelligence system...")
        
        # Initialize all components
        await self.entity_linker.initialize()
        await self.relation_extractor.initialize()
        await self.query_processor.initialize()
        
        # Add default reasoning rules
        await self._setup_default_reasoning_rules()
        
        logger.info("Knowledge Graph Intelligence system initialized successfully")
    
    async def _setup_default_reasoning_rules(self) -> None:
        """Setup default reasoning rules."""
        # Transitivity rule for works_for and part_of
        transitivity_rule = ReasoningRule(
            rule_id="transitivity_works_for",
            name="Transitivity of Employment",
            premise_pattern={
                'relation1': 'works_for',
                'relation2': 'part_of'
            },
            conclusion_pattern={
                'relation': 'works_for'
            },
            confidence_factor=0.8,
            priority=1
        )
        
        await self.reasoner.add_reasoning_rule(transitivity_rule)
    
    async def extract_knowledge_from_text(self, text: str, source: str = "text_extraction") -> Dict[str, int]:
        """Extract knowledge from text document."""
        extraction_result = {
            'entities_added': 0,
            'relations_added': 0
        }
        
        # First pass: extract entities (simplified approach)
        entities = await self._extract_entities_from_text(text, source)
        
        # Add entities to graph
        for entity in entities:
            await self.graph_store.add_entity(entity)
            extraction_result['entities_added'] += 1
        
        # Second pass: extract relations
        relations = await self.relation_extractor.extract_relations(text, entities)
        
        # Add relations to graph
        for relation in relations:
            await self.graph_store.add_relation(relation)
            extraction_result['relations_added'] += 1
        
        # Update statistics
        self.extraction_stats['documents_processed'] += 1
        self.extraction_stats['entities_extracted'] += extraction_result['entities_added']
        self.extraction_stats['relations_extracted'] += extraction_result['relations_added']
        
        logger.info(f"Extracted {extraction_result['entities_added']} entities and {extraction_result['relations_added']} relations from text")
        return extraction_result
    
    async def _extract_entities_from_text(self, text: str, source: str) -> List[KnowledgeEntity]:
        """Extract entities from text (simplified NER)."""
        entities = []
        
        # Simple entity extraction based on patterns
        import re
        
        # Extract person names (capitalized words)
        person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        person_matches = re.findall(person_pattern, text)
        
        for name in set(person_matches):  # Remove duplicates
            entity = KnowledgeEntity(
                entity_id=f"person_{uuid.uuid4().hex[:8]}",
                name=name,
                entity_type=EntityType.PERSON,
                confidence=0.8,
                source=source
            )
            entities.append(entity)
        
        # Extract organizations (words ending with Inc, Corp, LLC, etc.)
        org_pattern = r'\b[A-Z][a-zA-Z\s]+(Inc|Corp|LLC|Company|Ltd)\b'
        org_matches = re.findall(org_pattern, text)
        
        for match in set(org_matches):
            org_name = match[0] + match[1] if isinstance(match, tuple) else match
            entity = KnowledgeEntity(
                entity_id=f"org_{uuid.uuid4().hex[:8]}",
                name=org_name,
                entity_type=EntityType.ORGANIZATION,
                confidence=0.7,
                source=source
            )
            entities.append(entity)
        
        # Extract technology terms (common tech keywords)
        tech_keywords = [
            'Python', 'JavaScript', 'React', 'Angular', 'Vue', 'Node.js',
            'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP', 'TensorFlow',
            'PyTorch', 'Machine Learning', 'Artificial Intelligence',
            'Deep Learning', 'Neural Networks', 'API', 'REST', 'GraphQL'
        ]
        
        for keyword in tech_keywords:
            if keyword.lower() in text.lower():
                entity = KnowledgeEntity(
                    entity_id=f"tech_{keyword.lower().replace(' ', '_')}",
                    name=keyword,
                    entity_type=EntityType.TECHNOLOGY,
                    confidence=0.9,
                    source=source
                )
                entities.append(entity)
        
        return entities
    
    async def perform_reasoning(self) -> List[KnowledgeRelation]:
        """Perform semantic reasoning to infer new knowledge."""
        logger.info("Starting semantic reasoning...")
        
        inferred_relations = await self.reasoner.infer_relations(self.graph_store)
        
        # Update statistics
        self.extraction_stats['inferences_made'] += len(inferred_relations)
        
        logger.info(f"Semantic reasoning complete. Inferred {len(inferred_relations)} new relations")
        return inferred_relations
    
    async def query_knowledge(self, query_text: str, intent: str = "search") -> QueryResult:
        """Query the knowledge graph with natural language."""
        query = SemanticQuery(
            query_id=f"query_{uuid.uuid4().hex[:8]}",
            query_text=query_text,
            intent=intent
        )
        
        result = await self.query_processor.process_query(query, self.graph_store)
        
        logger.info(f"Processed query: '{query_text}' -> {len(result.entities)} entities, {len(result.relations)} relations")
        return result
    
    async def discover_insights(self, entity_name: str) -> Dict[str, Any]:
        """Discover insights about an entity."""
        # Find entity
        entities = await self.graph_store.find_entities_by_name(entity_name)
        
        if not entities:
            return {'error': f'Entity "{entity_name}" not found'}
        
        entity = entities[0]  # Use first match
        
        insights = {
            'entity': {
                'name': entity.name,
                'type': entity.entity_type.value,
                'aliases': entity.aliases,
                'properties': entity.properties
            },
            'connections': {},
            'paths': {},
            'statistics': {}
        }
        
        # Analyze connections
        relations = await self.graph_store.get_entity_relations(entity.entity_id)
        
        connection_types = defaultdict(list)
        for relation in relations:
            # Determine if entity is source or target
            if relation.source_entity_id == entity.entity_id:
                target_entity = await self.graph_store.get_entity(relation.target_entity_id)
                if target_entity:
                    connection_types[relation.relation_type.value].append({
                        'entity': target_entity.name,
                        'direction': 'outgoing',
                        'confidence': relation.confidence
                    })
            else:
                source_entity = await self.graph_store.get_entity(relation.source_entity_id)
                if source_entity:
                    connection_types[relation.relation_type.value].append({
                        'entity': source_entity.name,
                        'direction': 'incoming',
                        'confidence': relation.confidence
                    })
        
        insights['connections'] = dict(connection_types)
        
        # Find interesting paths
        neighbors = await self.graph_store.get_neighbors(entity.entity_id, max_distance=2)
        
        for neighbor_entity, distance in neighbors[:5]:  # Limit to top 5
            paths = await self.graph_store.find_path(entity.entity_id, neighbor_entity.entity_id, max_length=3)
            if paths:
                insights['paths'][neighbor_entity.name] = {
                    'shortest_path_length': min(path.path_length for path in paths),
                    'path_count': len(paths),
                    'best_confidence': max(path.total_confidence for path in paths)
                }
        
        # Statistics
        insights['statistics'] = {
            'total_connections': len(relations),
            'connection_types': len(connection_types),
            'neighborhood_size': len(neighbors),
            'centrality_estimate': len(relations) / max(1, len(self.graph_store.entities))
        }
        
        return insights
    
    async def get_graph_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of knowledge graph."""
        stats = await self.graph_store.get_statistics()
        
        # Add extraction statistics
        stats['extraction_stats'] = self.extraction_stats.copy()
        
        # Add component status
        stats['component_status'] = {
            'entity_linker': self.entity_linker.is_initialized,
            'relation_extractor': self.relation_extractor.is_initialized,
            'query_processor': self.query_processor.entity_linker.is_initialized,
            'reasoning_rules_count': len(self.reasoner.reasoning_rules)
        }
        
        # Top entities by degree
        entity_degrees = {}
        for entity_id in self.graph_store.entities.keys():
            degree = len(self.graph_store.adjacency_list.get(entity_id, set()))
            entity_degrees[entity_id] = degree
        
        top_entities = sorted(entity_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        
        stats['top_entities'] = []
        for entity_id, degree in top_entities:
            entity = self.graph_store.entities.get(entity_id)
            if entity:
                stats['top_entities'].append({
                    'name': entity.name,
                    'type': entity.entity_type.value,
                    'degree': degree
                })
        
        return stats
    
    async def export_subgraph(self, entity_names: List[str], include_neighbors: bool = True) -> Dict[str, Any]:
        """Export subgraph containing specified entities."""
        # Find entities by name
        entity_ids = []
        for name in entity_names:
            entities = await self.graph_store.find_entities_by_name(name)
            entity_ids.extend([entity.entity_id for entity in entities])
        
        if not entity_ids:
            return {'error': 'No matching entities found'}
        
        # Get subgraph
        subgraph = await self.graph_store.get_subgraph(entity_ids, include_neighbors)
        
        # Convert to serializable format
        export_data = {
            'entities': [],
            'relations': [],
            'metadata': subgraph['stats']
        }
        
        for entity in subgraph['entities'].values():
            export_data['entities'].append({
                'id': entity.entity_id,
                'name': entity.name,
                'type': entity.entity_type.value,
                'aliases': entity.aliases,
                'properties': entity.properties,
                'confidence': entity.confidence,
                'source': entity.source
            })
        
        for relation in subgraph['relations'].values():
            export_data['relations'].append({
                'id': relation.relation_id,
                'source': relation.source_entity_id,
                'target': relation.target_entity_id,
                'type': relation.relation_type.value,
                'confidence': relation.confidence,
                'properties': relation.properties,
                'evidence': relation.evidence
            })
        
        return export_data
    
    async def clear_cache(self) -> None:
        """Clear all caches."""
        self.graph_store.path_cache.clear()
        self.query_processor.query_cache.clear()
        self.reasoner.inference_cache.clear()
        logger.info("All caches cleared")
    
    async def shutdown(self) -> None:
        """Shutdown knowledge graph intelligence system."""
        logger.info("Shutting down Knowledge Graph Intelligence system...")
        await self.clear_cache()
        logger.info("Knowledge Graph Intelligence system shutdown complete")


# Example usage and demonstration
async def example_knowledge_graph_intelligence():
    """Example of knowledge graph intelligence system."""
    config = {
        'storage': {'max_cache_size': 1000},
        'entity_linking': {},
        'relation_extraction': {},
        'reasoning': {'max_reasoning_depth': 2},
        'query_processing': {'max_cache_size': 500}
    }
    
    # Initialize system
    kg_intelligence = KnowledgeGraphIntelligence(config)
    await kg_intelligence.initialize()
    
    # Sample knowledge extraction
    sample_text = """
    John Smith works for Google Inc and uses Python for machine learning projects.
    Google Inc is a technology company that created TensorFlow. 
    Mary Johnson also works for Google Inc and specializes in Deep Learning.
    TensorFlow is used for Neural Networks development.
    Google Inc is part of Alphabet Inc.
    """
    
    # Extract knowledge from text
    extraction_result = await kg_intelligence.extract_knowledge_from_text(
        sample_text, source="example_document"
    )
    logger.info(f"Knowledge extraction result: {extraction_result}")
    
    # Perform reasoning to infer new relations
    inferred_relations = await kg_intelligence.perform_reasoning()
    logger.info(f"Inferred {len(inferred_relations)} new relations")
    
    # Query the knowledge graph
    query_result = await kg_intelligence.query_knowledge("John Smith", intent="explore")
    logger.info(f"Query result: {len(query_result.entities)} entities found")
    
    # Discover insights about an entity
    insights = await kg_intelligence.discover_insights("John Smith")
    logger.info(f"Insights for John Smith: {len(insights.get('connections', {}))} connection types")
    
    # Get graph summary
    summary = await kg_intelligence.get_graph_summary()
    logger.info(f"Graph summary: {summary['total_entities']} entities, {summary['total_relations']} relations")
    
    # Export subgraph
    subgraph = await kg_intelligence.export_subgraph(["John Smith", "Google Inc"], include_neighbors=True)
    logger.info(f"Exported subgraph: {subgraph['metadata']}")
    
    # Cleanup
    await kg_intelligence.shutdown()


if __name__ == "__main__":
    asyncio.run(example_knowledge_graph_intelligence())