"""
Graphiti Temporal Knowledge Graphs for Archon+ Phase 4

Provides temporal knowledge graph operations using Kuzu database:
- Entity extraction and ingestion from code/docs/interactions
- Relationship discovery through static analysis and runtime observation
- Temporal queries for entity evolution and pattern detection
- Confidence propagation through relationship paths

Uses Kuzu embedded graph database for cost-effective operation.
"""

from .graphiti_service import (
    GraphitiService, 
    GraphEntity, 
    GraphRelationship, 
    EntityType, 
    RelationshipType,
    create_graphiti_service
)
from .entity_extractor import (
    EntityExtractor,
    ExtractionResult,
    CodeAnalyzer,
    DocumentAnalyzer,
    create_entity_extractor
)

__all__ = [
    # Core Graphiti service
    'GraphitiService',
    'GraphEntity', 
    'GraphRelationship',
    'EntityType',
    'RelationshipType',
    'create_graphiti_service',
    
    # Entity extraction
    'EntityExtractor',
    'ExtractionResult',
    'CodeAnalyzer',
    'DocumentAnalyzer',
    'create_entity_extractor',
    
    # Future components
    # 'TemporalQueryEngine',
    # 'ConfidencePropagator'
]