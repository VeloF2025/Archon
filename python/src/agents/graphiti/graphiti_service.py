#!/usr/bin/env python3
"""
Graphiti Service - Temporal Knowledge Graphs with Kuzu Database

Implements temporal knowledge graph operations for Archon+ Phase 4:
- Entity ingestion and management (code_functions, agents, projects, requirements)
- Relationship discovery and tracking (calls, implements, validates, references)
- Temporal tracking (creation_time, modification_time, access_frequency)
- Confidence propagation through relationship paths

Uses Kuzu embedded graph database for cost-effective, high-performance graph operations.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import hashlib

# Kuzu imports (will need to install: pip install kuzu)
try:
    import kuzu
    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False
    kuzu = None

logger = logging.getLogger(__name__)

class EntityType(Enum):
    """Types of entities in the knowledge graph"""
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    CONCEPT = "concept"
    AGENT = "agent"
    PROJECT = "project"
    REQUIREMENT = "requirement"
    PATTERN = "pattern"
    DOCUMENT = "document"

class RelationshipType(Enum):
    """Types of relationships between entities"""
    CALLS = "calls"
    IMPLEMENTS = "implements"
    VALIDATES = "validates"
    REFERENCES = "references"
    DEPENDS_ON = "depends_on"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    EVOLVED_FROM = "evolved_from"

@dataclass
class GraphEntity:
    """Entity in the temporal knowledge graph"""
    entity_id: str
    entity_type: EntityType
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    modification_time: float = field(default_factory=time.time)
    access_frequency: int = 0
    confidence_score: float = 1.0  # 0.0 to 1.0
    importance_weight: float = 0.5  # 0.0 to 1.0
    tags: List[str] = field(default_factory=list)

@dataclass
class GraphRelationship:
    """Relationship between entities with temporal data"""
    relationship_id: str
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    confidence: float = 1.0
    creation_time: float = field(default_factory=time.time)
    modification_time: float = field(default_factory=time.time)
    access_frequency: int = 0
    temporal_data: Dict[str, Any] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)

class GraphitiService:
    """
    Main service for temporal knowledge graph operations using Kuzu
    
    Provides entity management, relationship tracking, temporal queries,
    and confidence propagation for the Archon+ knowledge system.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize Graphiti service with Kuzu database
        
        Args:
            db_path: Path to Kuzu database directory
        """
        if not KUZU_AVAILABLE:
            raise ImportError("Kuzu library not available. Install with: pip install kuzu")
        
        self.db_path = db_path or Path("python/src/agents/graphiti/storage/kuzu_db")
        
        # If db_path is a directory, create a database file inside it
        if self.db_path.suffix != '.db':
            self.db_path.mkdir(parents=True, exist_ok=True)
            self.db_file = self.db_path / "graphiti.db"
        else:
            # db_path is already a file path
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.db_file = self.db_path
        
        # Initialize Kuzu database
        self.database = kuzu.Database(str(self.db_file))
        self.connection = kuzu.Connection(self.database)
        
        # Initialize schema
        self._initialize_schema()
        
        # Performance tracking
        self.query_times: List[float] = []
        self.entity_cache: Dict[str, GraphEntity] = {}
        
        logger.info(f"Initialized Graphiti service with Kuzu database at {self.db_path}")
    
    def _initialize_schema(self):
        """Initialize Kuzu database schema for temporal knowledge graphs"""
        try:
            # Install and load JSON extension
            self.connection.execute("INSTALL JSON;")
            self.connection.execute("LOAD EXTENSION JSON;")
            
            # Create Entity node table
            self.connection.execute("""
                CREATE NODE TABLE IF NOT EXISTS Entity(
                    entity_id STRING,
                    entity_type STRING,
                    name STRING,
                    attributes JSON,
                    creation_time DOUBLE,
                    modification_time DOUBLE,
                    access_frequency INT64,
                    confidence_score DOUBLE,
                    importance_weight DOUBLE,
                    tags STRING[],
                    PRIMARY KEY (entity_id)
                )
            """)
            
            # Create Relationship edge table
            self.connection.execute("""
                CREATE REL TABLE IF NOT EXISTS Relationship(
                    FROM Entity TO Entity,
                    relationship_id STRING,
                    relationship_type STRING,
                    confidence DOUBLE,
                    creation_time DOUBLE,
                    modification_time DOUBLE,
                    access_frequency INT64,
                    temporal_data JSON,
                    attributes JSON
                )
            """)
            
            logger.info("Kuzu schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kuzu schema: {e}")
            raise
    
    async def add_entity(self, entity: GraphEntity) -> bool:
        """
        Add or update an entity in the knowledge graph
        
        Args:
            entity: GraphEntity to add
            
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        
        try:
            # Validate entity before processing
            is_valid, validation_errors = self.validate_entity(entity)
            if not is_valid:
                logger.error(f"Entity validation failed for {entity.entity_id}: {validation_errors}")
                return False
            
            # Convert attributes to JSON string for Kuzu
            attributes_json = json.dumps(entity.attributes)
            
            # Check if entity exists first (Kuzu-compatible upsert)
            exists_query = "MATCH (e:Entity) WHERE e.entity_id = $entity_id RETURN e.entity_id"
            result = self.connection.execute(exists_query, {'entity_id': entity.entity_id})
            
            try:
                entity_exists = result.has_next()
            except:
                entity_exists = False
            
            if entity_exists:
                # Update existing entity
                query = """
                    MATCH (e:Entity)
                    WHERE e.entity_id = $entity_id
                    SET e.entity_type = $entity_type,
                        e.name = $name,
                        e.attributes = $attributes,
                        e.modification_time = $modification_time,
                        e.access_frequency = $access_frequency,
                        e.confidence_score = $confidence_score,
                        e.importance_weight = $importance_weight,
                        e.tags = $tags
                """
                params = {
                    'entity_id': entity.entity_id,
                    'entity_type': entity.entity_type.value,
                    'name': entity.name,
                    'attributes': attributes_json,
                    'modification_time': entity.modification_time,
                    'access_frequency': entity.access_frequency,
                    'confidence_score': entity.confidence_score,
                    'importance_weight': entity.importance_weight,
                    'tags': entity.tags
                }
            else:
                # Create new entity
                query = """
                    CREATE (e:Entity {
                        entity_id: $entity_id,
                        entity_type: $entity_type,
                        name: $name,
                        attributes: $attributes,
                        creation_time: $creation_time,
                        modification_time: $modification_time,
                        access_frequency: $access_frequency,
                        confidence_score: $confidence_score,
                        importance_weight: $importance_weight,
                        tags: $tags
                    })
                """
                params = {
                    'entity_id': entity.entity_id,
                    'entity_type': entity.entity_type.value,
                    'name': entity.name,
                    'attributes': attributes_json,
                    'creation_time': entity.creation_time,
                    'modification_time': entity.modification_time,
                    'access_frequency': entity.access_frequency,
                    'confidence_score': entity.confidence_score,
                    'importance_weight': entity.importance_weight,
                    'tags': entity.tags
                }
            
            self.connection.execute(query, params)
            
            # In Kuzu, transactions are auto-committed by default, but let's be explicit
            # Kuzu handles commits automatically for single operations
            
            # Update cache
            self.entity_cache[entity.entity_id] = entity
            
            # Track performance
            query_time = time.time() - start_time
            self.query_times.append(query_time)
            
            logger.debug(f"Added entity {entity.entity_id} ({entity.entity_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add entity {entity.entity_id}: {e}")
            return False
    
    async def add_relationship(self, relationship: GraphRelationship) -> bool:
        """
        Add or update a relationship in the knowledge graph
        
        Args:
            relationship: GraphRelationship to add
            
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        
        try:
            # Validate relationship before processing
            is_valid, validation_errors = self.validate_relationship(relationship)
            if not is_valid:
                logger.error(f"Relationship validation failed for {relationship.relationship_id}: {validation_errors}")
                return False
            
            # Ensure source and target entities exist
            source_exists = await self.entity_exists(relationship.source_id)
            target_exists = await self.entity_exists(relationship.target_id)
            
            if not (source_exists and target_exists):
                logger.warning(f"Cannot create relationship {relationship.relationship_id}: "
                             f"source_exists={source_exists}, target_exists={target_exists}")
                return False
            
            # Convert temporal data and attributes to JSON
            temporal_json = json.dumps(relationship.temporal_data)
            attributes_json = json.dumps(relationship.attributes)
            
            # For Kuzu, we need to create relationships differently
            # First ensure we have both entities, then create the relationship
            query = """
                MATCH (source:Entity), (target:Entity)
                WHERE source.entity_id = $source_id AND target.entity_id = $target_id
                CREATE (source)-[:Relationship {
                    relationship_id: $relationship_id,
                    relationship_type: $relationship_type,
                    confidence: $confidence,
                    creation_time: $creation_time,
                    modification_time: $modification_time,
                    access_frequency: $access_frequency,
                    temporal_data: $temporal_data,
                    attributes: $attributes
                }]->(target)
            """
            
            params = {
                'source_id': relationship.source_id,
                'target_id': relationship.target_id,
                'relationship_id': relationship.relationship_id,
                'relationship_type': relationship.relationship_type.value,
                'confidence': relationship.confidence,
                'creation_time': relationship.creation_time,
                'modification_time': relationship.modification_time,
                'access_frequency': relationship.access_frequency,
                'temporal_data': temporal_json,
                'attributes': attributes_json
            }
            
            self.connection.execute(query, params)
            
            # In Kuzu, transactions are auto-committed by default
            # Kuzu handles commits automatically for single operations
            
            # Track performance
            query_time = time.time() - start_time
            self.query_times.append(query_time)
            
            logger.debug(f"Added relationship {relationship.relationship_id} "
                        f"({relationship.source_id} -> {relationship.target_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add relationship {relationship.relationship_id}: {e}")
            return False
    
    async def entity_exists(self, entity_id: str) -> bool:
        """Check if an entity exists in the graph"""
        try:
            # Check cache first
            if entity_id in self.entity_cache:
                return True
            
            query = "MATCH (e:Entity) WHERE e.entity_id = $entity_id RETURN e.entity_id"
            result = self.connection.execute(query, {'entity_id': entity_id})
            
            try:
                exists = result.has_next()
            except:
                exists = False
            
            return exists
            
        except Exception as e:
            logger.debug(f"Error checking entity existence {entity_id}: {e}")
            return False
    
    async def get_entity(self, entity_id: str) -> Optional[GraphEntity]:
        """
        Retrieve an entity by ID
        
        Args:
            entity_id: ID of entity to retrieve
            
        Returns:
            GraphEntity if found, None otherwise
        """
        start_time = time.time()
        
        try:
            # Check cache first
            if entity_id in self.entity_cache:
                entity = self.entity_cache[entity_id]
                entity.access_frequency += 1
                await self._update_entity_access(entity_id, entity.access_frequency)
                return entity
            
            query = """
                MATCH (e:Entity)
                WHERE e.entity_id = $entity_id
                RETURN e.entity_id, e.entity_type, e.name, e.attributes,
                       e.creation_time, e.modification_time, e.access_frequency,
                       e.confidence_score, e.importance_weight, e.tags
            """
            
            result = self.connection.execute(query, {'entity_id': entity_id})
            
            if not result.has_next():
                return None
            
            row = result.get_next()
            
            # Parse attributes JSON
            attributes = json.loads(row[3]) if row[3] else {}
            
            # Convert string value back to EntityType enum
            try:
                entity_type = EntityType(row[1])
            except ValueError:
                # Fallback for invalid enum values
                entity_type = EntityType.CONCEPT
            
            entity = GraphEntity(
                entity_id=row[0],
                entity_type=entity_type,
                name=row[2],
                attributes=attributes,
                creation_time=row[4],
                modification_time=row[5],
                access_frequency=row[6] + 1,  # Increment access
                confidence_score=row[7],
                importance_weight=row[8],
                tags=row[9] if row[9] else []
            )
            
            # Update cache and access frequency
            self.entity_cache[entity_id] = entity
            await self._update_entity_access(entity_id, entity.access_frequency)
            
            # Track performance
            query_time = time.time() - start_time
            self.query_times.append(query_time)
            
            return entity
            
        except Exception as e:
            logger.error(f"Failed to get entity {entity_id}: {e}")
            return None
    
    async def _update_entity_access(self, entity_id: str, access_frequency: int):
        """Update entity access frequency"""
        try:
            query = """
                MATCH (e:Entity)
                WHERE e.entity_id = $entity_id
                SET e.access_frequency = $access_frequency,
                    e.modification_time = $modification_time
            """
            
            params = {
                'entity_id': entity_id,
                'access_frequency': access_frequency,
                'modification_time': time.time()
            }
            
            self.connection.execute(query, params)
            
        except Exception as e:
            logger.debug(f"Failed to update entity access {entity_id}: {e}")
    
    async def query_temporal(self, entity_type: Optional[Union[EntityType, str]] = None,
                           time_window: Optional[str] = None,
                           pattern: Optional[str] = None,
                           limit: int = 50) -> List[GraphEntity]:
        """
        Query entities with temporal filtering
        
        Args:
            entity_type: Filter by entity type
            time_window: Time window (e.g., "24h", "7d", "30d")
            pattern: Pattern to match (e.g., "evolution", "trending")
            limit: Maximum number of results
            
        Returns:
            List of matching GraphEntity objects
        """
        start_time = time.time()
        
        try:
            # Build query conditions
            conditions = []
            params = {}
            
            if entity_type:
                conditions.append("e.entity_type = $entity_type")
                # Handle both EntityType enum and string input
                if isinstance(entity_type, EntityType):
                    params['entity_type'] = entity_type.value
                elif isinstance(entity_type, str):
                    # Validate string is a valid enum value
                    try:
                        EntityType(entity_type)  # Just validate, don't store
                        params['entity_type'] = entity_type
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid entity_type string: {entity_type}, ignoring filter")
                        conditions.pop()  # Remove the condition we just added
                else:
                    logger.warning(f"Invalid entity_type type: {type(entity_type)}, ignoring filter")
                    conditions.pop()  # Remove the condition we just added
            
            if time_window:
                # Parse time window
                window_seconds = self._parse_time_window(time_window)
                if window_seconds:
                    cutoff_time = time.time() - window_seconds
                    conditions.append("e.creation_time >= $cutoff_time")
                    params['cutoff_time'] = cutoff_time
            
            # Build query
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            if pattern == "evolution":
                # Find entities that have evolved (high modification frequency)
                order_clause = "ORDER BY e.creation_time DESC, e.modification_time DESC"
            elif pattern == "trending":
                # Find trending entities (high recent access)
                order_clause = "ORDER BY e.access_frequency DESC, e.importance_weight DESC"
            else:
                # Default ordering
                order_clause = "ORDER BY e.importance_weight DESC, e.confidence_score DESC"
            
            query = f"""
                MATCH (e:Entity)
                WHERE {where_clause}
                RETURN e.entity_id, e.entity_type, e.name, e.attributes,
                       e.creation_time, e.modification_time, e.access_frequency,
                       e.confidence_score, e.importance_weight, e.tags
                {order_clause}
                LIMIT $limit
            """
            
            params['limit'] = limit
            
            result = self.connection.execute(query, params)
            entities = []
            
            while result.has_next():
                try:
                    row = result.get_next()
                    # Kuzu returns tuples, we need to access them properly
                    # The values are returned in the order we specified in RETURN
                    attributes = json.loads(row[3]) if row[3] else {}
                    
                    # Convert string value back to EntityType enum
                    try:
                        entity_type = EntityType(row[1])
                    except (ValueError, TypeError):
                        # Fallback for invalid enum values or None
                        logger.warning(f"Invalid entity_type value: {row[1]}, using CONCEPT fallback")
                        entity_type = EntityType.CONCEPT
                    
                    entity = GraphEntity(
                        entity_id=row[0],
                        entity_type=entity_type,
                        name=row[2],
                        attributes=attributes,
                        creation_time=row[4],
                        modification_time=row[5],
                        access_frequency=row[6],
                        confidence_score=row[7],
                        importance_weight=row[8],
                        tags=row[9] if row[9] else []
                    )
                    entities.append(entity)
                    
                except Exception as e:
                    logger.debug(f"Error processing row: {e}")
                    continue
            
            # Track performance
            query_time = time.time() - start_time
            self.query_times.append(query_time)
            
            logger.debug(f"Temporal query returned {len(entities)} entities")
            return entities
            
        except Exception as e:
            logger.error(f"Temporal query failed: {e}")
            return []
    
    def _parse_time_window(self, time_window: str) -> Optional[int]:
        """Parse time window string to seconds"""
        try:
            if time_window.endswith('h'):
                return int(time_window[:-1]) * 3600
            elif time_window.endswith('d'):
                return int(time_window[:-1]) * 86400
            elif time_window.endswith('w'):
                return int(time_window[:-1]) * 604800
            else:
                return int(time_window)  # Assume seconds
        except:
            return None
    
    def propagate_confidence(self, source_confidence: float, relationship_confidence: float) -> float:
        """
        Propagate confidence scores through relationship paths (SCWT compatible)
        
        Args:
            source_confidence: Confidence score from source entity (0.0-1.0)
            relationship_confidence: Confidence score of the relationship (0.0-1.0)
            
        Returns:
            Propagated confidence score
        """
        try:
            # Validate inputs
            source_confidence = max(0.0, min(1.0, source_confidence))
            relationship_confidence = max(0.0, min(1.0, relationship_confidence))
            
            # Propagation formula: weighted average with decay
            decay_factor = 0.8  # Confidence decays through relationships
            
            # Calculate propagated confidence as weighted product with decay
            propagated_confidence = source_confidence * relationship_confidence * decay_factor
            
            # Ensure result is in valid range
            result = max(0.0, min(1.0, propagated_confidence))
            
            logger.debug(f"Propagated confidence: {source_confidence:.3f} * {relationship_confidence:.3f} * {decay_factor:.3f} = {result:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Confidence propagation failed: {e}")
            return 0.0
    
    def propagate_confidence_entities(self, source_entity: GraphEntity,
                            target_entity: GraphEntity,
                            relationship: GraphRelationship) -> float:
        """
        Propagate confidence scores through relationship paths using entities
        
        Args:
            source_entity: Source entity with confidence
            target_entity: Target entity to update
            relationship: Relationship connecting them
            
        Returns:
            Updated confidence score for target entity
        """
        try:
            # Use the SCWT-compatible method
            source_confidence = source_entity.confidence_score
            relationship_confidence = relationship.confidence
            
            # Calculate propagated confidence
            propagated_confidence = self.propagate_confidence(source_confidence, relationship_confidence)
            
            # Update target entity confidence (weighted combination with existing)
            current_confidence = target_entity.confidence_score
            weight = 0.3  # Weight of new evidence
            
            new_confidence = (
                (1 - weight) * current_confidence +
                weight * propagated_confidence
            )
            
            # Ensure confidence stays in valid range
            updated_confidence = max(0.0, min(1.0, new_confidence))
            
            # Update target entity
            target_entity.confidence_score = updated_confidence
            target_entity.modification_time = time.time()
            
            # Note: In synchronous version, we don't update the entity in database
            # Just return the calculated confidence for testing purposes
            # In production, this would need async database update
            
            logger.debug(f"Propagated confidence from {source_entity.entity_id} to "
                        f"{target_entity.entity_id}: {current_confidence:.3f} -> {updated_confidence:.3f}")
            
            return updated_confidence
            
        except Exception as e:
            logger.error(f"Confidence propagation failed: {e}")
            return target_entity.confidence_score
    
    async def get_related_entities(self, entity_id: str, 
                                 relationship_types: Optional[List[RelationshipType]] = None,
                                 max_depth: int = 2) -> List[Tuple[GraphEntity, GraphRelationship]]:
        """
        Get entities related to a given entity
        
        Args:
            entity_id: Source entity ID
            relationship_types: Filter by relationship types
            max_depth: Maximum relationship depth to traverse
            
        Returns:
            List of (entity, relationship) tuples
        """
        try:
            # Build relationship type filter
            type_filter = ""
            params = {'entity_id': entity_id}
            
            if relationship_types:
                type_values = [rt.value for rt in relationship_types]
                type_filter = "AND r.relationship_type IN $relationship_types"
                params['relationship_types'] = type_values
            
            # Kuzu syntax for finding related entities via relationships
            # Use explicit direction to avoid bidirectional issues
            query = f"""
                MATCH (source:Entity)
                WHERE source.entity_id = $entity_id
                MATCH (source)-[r:Relationship]->(target:Entity)
                WHERE source.entity_id <> target.entity_id {type_filter}
                RETURN DISTINCT target.entity_id, target.entity_type, target.name,
                       target.attributes, target.creation_time, target.modification_time,
                       target.access_frequency, target.confidence_score,
                       target.importance_weight, target.tags,
                       r.relationship_id, r.relationship_type, r.confidence
                LIMIT 100
            """
            
            result = self.connection.execute(query, params)
            related = []
            
            while result.has_next():
                try:
                    row = result.get_next()
                    # Create entity
                    attributes = json.loads(row[3]) if row[3] else {}
                    
                    # Convert string value back to EntityType enum
                    try:
                        entity_type = EntityType(row[1])
                    except (ValueError, TypeError):
                        # Fallback for invalid enum values or None
                        logger.warning(f"Invalid entity_type value: {row[1]}, using CONCEPT fallback")
                        entity_type = EntityType.CONCEPT
                    
                    entity = GraphEntity(
                        entity_id=row[0],
                        entity_type=entity_type,
                        name=row[2],
                        attributes=attributes,
                        creation_time=row[4],
                            modification_time=row[5],
                            access_frequency=row[6],
                            confidence_score=row[7],
                            importance_weight=row[8],
                            tags=row[9] if row[9] else []
                        )
                    
                    # Create minimal relationship info
                    # Handle RelationshipType enum safely
                    try:
                        rel_type = RelationshipType(row[11])
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid relationship_type value: {row[11]}, using RELATED_TO fallback")
                        rel_type = RelationshipType.RELATED_TO
                        
                    relationship = GraphRelationship(
                        relationship_id=row[10],
                        source_id=entity_id,
                        target_id=row[0],
                        relationship_type=rel_type,
                        confidence=row[12]
                    )
                    
                    related.append((entity, relationship))
                        
                except Exception as e:
                    logger.debug(f"Error processing related entity row: {e}")
                    continue
            
            return related
            
        except Exception as e:
            logger.error(f"Failed to get related entities for {entity_id}: {e}")
            return []
    
    async def add_entities_batch(self, entities: List[GraphEntity]) -> Dict[str, Any]:
        """
        Add multiple entities in batch for better performance
        
        Args:
            entities: List of GraphEntity objects to add
            
        Returns:
            Dict with success/failure counts and details
        """
        start_time = time.time()
        results = {'succeeded': [], 'failed': []}
        
        for entity in entities:
            try:
                success = await self.add_entity(entity)
                if success:
                    results['succeeded'].append(entity.entity_id)
                else:
                    results['failed'].append({
                        'entity_id': entity.entity_id,
                        'error': 'Failed to add entity (unknown error)'
                    })
            except Exception as e:
                results['failed'].append({
                    'entity_id': entity.entity_id,
                    'error': str(e)
                })
                logger.error(f"Batch add failed for entity {entity.entity_id}: {e}")
        
        batch_time = time.time() - start_time
        logger.info(f"Batch add completed: {len(results['succeeded'])} succeeded, "
                   f"{len(results['failed'])} failed in {batch_time:.3f}s")
        
        return {
            'succeeded_count': len(results['succeeded']),
            'failed_count': len(results['failed']),
            'succeeded_entities': results['succeeded'],
            'failed_entities': results['failed'],
            'batch_time': batch_time
        }
    
    async def add_relationships_batch(self, relationships: List[GraphRelationship]) -> Dict[str, Any]:
        """
        Add multiple relationships in batch for better performance
        
        Args:
            relationships: List of GraphRelationship objects to add
            
        Returns:
            Dict with success/failure counts and details
        """
        start_time = time.time()
        results = {'succeeded': [], 'failed': []}
        
        for relationship in relationships:
            try:
                success = await self.add_relationship(relationship)
                if success:
                    results['succeeded'].append(relationship.relationship_id)
                else:
                    results['failed'].append({
                        'relationship_id': relationship.relationship_id,
                        'error': 'Failed to add relationship (unknown error)'
                    })
            except Exception as e:
                results['failed'].append({
                    'relationship_id': relationship.relationship_id,
                    'error': str(e)
                })
                logger.error(f"Batch add failed for relationship {relationship.relationship_id}: {e}")
        
        batch_time = time.time() - start_time
        logger.info(f"Relationship batch add completed: {len(results['succeeded'])} succeeded, "
                   f"{len(results['failed'])} failed in {batch_time:.3f}s")
        
        return {
            'succeeded_count': len(results['succeeded']),
            'failed_count': len(results['failed']),
            'succeeded_relationships': results['succeeded'],
            'failed_relationships': results['failed'],
            'batch_time': batch_time
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the Graphiti service"""
        if not self.query_times:
            return {"avg_query_time": 0.0, "max_query_time": 0.0, "total_queries": 0}
        
        return {
            "avg_query_time": sum(self.query_times) / len(self.query_times),
            "max_query_time": max(self.query_times),
            "total_queries": len(self.query_times),
            "cached_entities": len(self.entity_cache),
            "db_path": str(self.db_path)
        }
    
    def validate_entity(self, entity: GraphEntity) -> Tuple[bool, List[str]]:
        """
        Validate entity data before adding to graph
        
        Args:
            entity: GraphEntity to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        if not entity.entity_id or not isinstance(entity.entity_id, str):
            errors.append("entity_id must be a non-empty string")
        
        if not entity.name or not isinstance(entity.name, str):
            errors.append("name must be a non-empty string")
        
        if not isinstance(entity.entity_type, EntityType):
            errors.append("entity_type must be an EntityType enum")
        
        # Check numeric constraints
        if not (0.0 <= entity.confidence_score <= 1.0):
            errors.append("confidence_score must be between 0.0 and 1.0")
        
        if not (0.0 <= entity.importance_weight <= 1.0):
            errors.append("importance_weight must be between 0.0 and 1.0")
        
        if entity.access_frequency < 0:
            errors.append("access_frequency must be non-negative")
        
        # Check timestamps
        if entity.creation_time <= 0:
            errors.append("creation_time must be positive")
        
        if entity.modification_time <= 0:
            errors.append("modification_time must be positive")
        
        # Check attributes type
        if not isinstance(entity.attributes, dict):
            errors.append("attributes must be a dictionary")
        
        if not isinstance(entity.tags, list):
            errors.append("tags must be a list")
        
        return len(errors) == 0, errors
    
    def validate_relationship(self, relationship: GraphRelationship) -> Tuple[bool, List[str]]:
        """
        Validate relationship data before adding to graph
        
        Args:
            relationship: GraphRelationship to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        if not relationship.relationship_id or not isinstance(relationship.relationship_id, str):
            errors.append("relationship_id must be a non-empty string")
        
        if not relationship.source_id or not isinstance(relationship.source_id, str):
            errors.append("source_id must be a non-empty string")
        
        if not relationship.target_id or not isinstance(relationship.target_id, str):
            errors.append("target_id must be a non-empty string")
        
        if not isinstance(relationship.relationship_type, RelationshipType):
            errors.append("relationship_type must be a RelationshipType enum")
        
        # Check that source and target are different
        if relationship.source_id == relationship.target_id:
            errors.append("source_id and target_id must be different")
        
        # Check numeric constraints
        if not (0.0 <= relationship.confidence <= 1.0):
            errors.append("confidence must be between 0.0 and 1.0")
        
        if relationship.access_frequency < 0:
            errors.append("access_frequency must be non-negative")
        
        # Check timestamps
        if relationship.creation_time <= 0:
            errors.append("creation_time must be positive")
        
        if relationship.modification_time <= 0:
            errors.append("modification_time must be positive")
        
        # Check data types
        if not isinstance(relationship.temporal_data, dict):
            errors.append("temporal_data must be a dictionary")
        
        if not isinstance(relationship.attributes, dict):
            errors.append("attributes must be a dictionary")
        
        return len(errors) == 0, errors
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the Graphiti service
        
        Returns:
            Dict with health status and diagnostics
        """
        health_status = {
            'status': 'healthy',
            'checks': {},
            'timestamp': time.time()
        }
        
        try:
            # Check database connection
            test_query = "MATCH (e:Entity) RETURN COUNT(e) as count"
            result = self.connection.execute(test_query)
            
            try:
                if result.has_next():
                    row = result.get_next()
                    entity_count = row[0] if row else 0
                else:
                    entity_count = 0
            except:
                entity_count = 0
            
            health_status['checks']['database'] = {
                'status': 'healthy',
                'entity_count': entity_count
            }
            
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['checks']['database'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Check cache status
        health_status['checks']['cache'] = {
            'status': 'healthy',
            'cached_entities': len(self.entity_cache)
        }
        
        # Check performance metrics
        if self.query_times:
            avg_query_time = sum(self.query_times) / len(self.query_times)
            health_status['checks']['performance'] = {
                'status': 'healthy' if avg_query_time < 1.0 else 'warning',
                'avg_query_time': avg_query_time,
                'total_queries': len(self.query_times)
            }
        else:
            health_status['checks']['performance'] = {
                'status': 'healthy',
                'message': 'No queries executed yet'
            }
        
        return health_status

    async def close(self):
        """Close database connections"""
        try:
            if hasattr(self, 'connection') and self.connection:
                # Kuzu connections are automatically managed
                pass
            logger.info("Graphiti service closed")
        except Exception as e:
            logger.error(f"Error closing Graphiti service: {e}")

# Factory function
def create_graphiti_service(db_path: Optional[Path] = None) -> GraphitiService:
    """Create a configured Graphiti service instance"""
    return GraphitiService(db_path)