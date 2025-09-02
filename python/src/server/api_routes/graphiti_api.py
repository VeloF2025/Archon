"""
Graphiti API Routes - Knowledge Graph Operations

Provides REST API endpoints for interacting with the Graphiti knowledge graph:
- Get graph data with filtering
- Query entities and relationships
- Temporal filtering and analytics
- Health monitoring
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# Avoid importing through agents.__init__ which requires pydantic_ai
try:
    from ...agents.graphiti.graphiti_service import (
        GraphitiService,
        GraphEntity,
        GraphRelationship,
        EntityType,
        RelationshipType,
        create_graphiti_service
    )
except ImportError:
    # Fallback if pydantic_ai is not available
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from src.agents.graphiti.graphiti_service import (
        GraphitiService,
        GraphEntity,
        GraphRelationship,
        EntityType,
        RelationshipType,
        create_graphiti_service
    )

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/graphiti", tags=["graphiti"])

# Global service instance (initialized on first use)
_graphiti_service: Optional[GraphitiService] = None

def get_graphiti_service() -> GraphitiService:
    """Get or create the Graphiti service instance"""
    global _graphiti_service
    if _graphiti_service is None:
        _graphiti_service = create_graphiti_service()
    return _graphiti_service

# Request/Response Models

class GraphFilters(BaseModel):
    """Filters for graph data queries"""
    entity_types: Optional[List[str]] = Field(default=None, description="Filter by entity types")
    relationship_types: Optional[List[str]] = Field(default=None, description="Filter by relationship types")
    time_window: Optional[str] = Field(default=None, description="Time window (e.g., '24h', '7d', '30d')")
    search_term: Optional[str] = Field(default=None, description="Search term for entity names/tags")
    confidence_threshold: Optional[float] = Field(default=0.0, description="Minimum confidence score")
    importance_threshold: Optional[float] = Field(default=0.0, description="Minimum importance weight")
    limit: Optional[int] = Field(default=100, description="Maximum number of results")

class GraphNode(BaseModel):
    """Graph node representation"""
    id: str
    label: str
    type: str
    properties: Dict[str, Any]
    position: Optional[Dict[str, float]] = None

class GraphEdge(BaseModel):
    """Graph edge representation"""
    id: str
    source: str
    target: str
    type: str
    properties: Dict[str, Any]

class GraphMetadata(BaseModel):
    """Graph metadata"""
    total_entities: int
    total_relationships: int
    entity_types: List[str]
    relationship_types: List[str]
    last_updated: float
    query_time: float

class GraphData(BaseModel):
    """Complete graph data response"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metadata: GraphMetadata

class TemporalFilterRequest(BaseModel):
    """Temporal filter parameters"""
    start_time: float
    end_time: float
    granularity: str = Field(default="hour", description="Time granularity (hour, day, week)")
    entity_type: Optional[str] = None
    pattern: Optional[str] = Field(default=None, description="Pattern to match (evolution, trending)")

class EntitySearchRequest(BaseModel):
    """Entity search parameters"""
    query: str
    entity_types: Optional[List[str]] = None
    limit: int = Field(default=20, description="Maximum results")

# API Endpoints

@router.get("/health")
async def health_check():
    """Get Graphiti service health status"""
    try:
        service = get_graphiti_service()
        health_status = await service.health_check()
        return health_status
    except Exception as e:
        logger.error(f"Graphiti health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Graphiti service unhealthy: {str(e)}")

@router.get("/graph-data", response_model=GraphData)
async def get_graph_data(
    entity_types: Optional[str] = Query(None, description="Comma-separated entity types"),
    relationship_types: Optional[str] = Query(None, description="Comma-separated relationship types"),
    time_window: Optional[str] = Query(None, description="Time window (24h, 7d, 30d)"),
    search_term: Optional[str] = Query(None, description="Search term"),
    confidence_threshold: float = Query(0.0, description="Minimum confidence"),
    importance_threshold: float = Query(0.0, description="Minimum importance"),
    limit: int = Query(100, description="Max results"),
):
    """
    Get graph data from knowledge base sources and chunks
    
    Transforms knowledge base data into graph visualization format
    """
    start_time = time.time()
    
    try:
        from ..utils import get_supabase_client
        
        supabase = get_supabase_client()
        
        # Get knowledge sources as nodes
        sources_result = supabase.table('archon_sources').select(
            'source_id, source_display_name, source_url, metadata, created_at'
        ).limit(limit).execute()
        
        nodes = []
        entity_ids = set()
        
        # Convert sources to nodes
        for source in sources_result.data:
            source_id = source['source_id']
            metadata = source.get('metadata', {})
            source_type = metadata.get('source_type', 'document')
            
            # Map source types to visual categories
            node_type = 'document'
            source_url = source.get('source_url') or ''
            if 'github' in source_url.lower():
                node_type = 'project'
            elif 'docs' in source_url.lower():
                node_type = 'concept'
            elif source_type == 'url':
                node_type = 'module'
            
            node = GraphNode(
                id=source_id,
                label=source['source_display_name'] or 'Unknown Source',
                type=node_type,
                properties={
                    "url": source.get('source_url', ''),
                    "created_at": source.get('created_at', ''),
                    "source_type": source_type,
                    "confidence_score": 0.9,  # High confidence for KB sources
                    "metadata": metadata
                }
            )
            nodes.append(node)
            entity_ids.add(source_id)
        
        # Get chunks and create relationships
        edges = []
        relationship_types = set()
        
        if entity_ids:
            chunks_result = supabase.table('archon_crawled_pages').select(
                'source_id, url, content, metadata'
            ).in_('source_id', list(entity_ids)).limit(limit * 2).execute()
            
            # Group chunks by source and create internal relationships
            source_chunks = {}
            for chunk in chunks_result.data:
                source_id = chunk['source_id']
                if source_id not in source_chunks:
                    source_chunks[source_id] = []
                source_chunks[source_id].append(chunk)
            
            # Create edges between related sources (same domain/type)
            sources_by_domain = {}
            for source in sources_result.data:
                url = source.get('source_url', '')
                if url:
                    domain = url.split('/')[2] if '://' in url else 'unknown'
                    if domain not in sources_by_domain:
                        sources_by_domain[domain] = []
                    sources_by_domain[domain].append(source['source_id'])
            
            # Create relationships between sources from same domain
            for domain, source_ids in sources_by_domain.items():
                if len(source_ids) > 1:
                    for i, source_id in enumerate(source_ids):
                        for j, related_id in enumerate(source_ids):
                            if i != j and source_id in entity_ids and related_id in entity_ids:
                                relationship_types.add('related_to')
                                edge = GraphEdge(
                                    id=f"{source_id}_{related_id}",
                                    source=source_id,
                                    target=related_id,
                                    type='related_to',
                                    properties={
                                        "confidence": 0.7,
                                        "relationship_type": "same_domain"
                                    }
                                )
                                edges.append(edge)
                                break  # Only one edge per source to avoid clutter
        
        # Apply search filter if provided
        if search_term:
            search_lower = search_term.lower()
            nodes = [n for n in nodes if search_lower in n.label.lower()]
            # Filter edges to only include nodes that passed the search
            node_ids = {n.id for n in nodes}
            edges = [e for e in edges if e.source in node_ids and e.target in node_ids]
        
        # Apply confidence threshold filter
        if confidence_threshold > 0:
            nodes = [n for n in nodes if n.properties.get('confidence_score', 0.9) >= confidence_threshold]
            node_ids = {n.id for n in nodes}
            edges = [e for e in edges if e.source in node_ids and e.target in node_ids]
        
        query_time = time.time() - start_time
        
        metadata = GraphMetadata(
            total_entities=len(nodes),
            total_relationships=len(edges),
            entity_types=list(set(n.type for n in nodes)),
            relationship_types=list(relationship_types),
            last_updated=time.time(),
            query_time=query_time
        )
        
        logger.info(f"Generated graph with {len(nodes)} nodes and {len(edges)} edges in {query_time:.3f}s")
        
        return GraphData(nodes=nodes, edges=edges, metadata=metadata)
    
    except Exception as e:
        logger.error(f"Failed to retrieve graph data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve graph data: {str(e)}")

@router.post("/temporal-filter", response_model=GraphData)
# @require_credentials  # TODO: Add authentication when available
async def apply_temporal_filter(request: TemporalFilterRequest):
    """
    Apply temporal filtering to graph data
    
    Filters entities and relationships by time range with specified granularity
    """
    try:
        service = get_graphiti_service()
        
        # Calculate time window from timestamps
        window_seconds = request.end_time - request.start_time
        
        if window_seconds <= 86400:  # 1 day
            time_window = f"{int(window_seconds/3600)}h"
        elif window_seconds <= 604800:  # 1 week
            time_window = f"{int(window_seconds/86400)}d"
        else:
            time_window = f"{int(window_seconds/604800)}w"
        
        # Query with temporal constraints
        entities = await service.query_temporal(
            entity_type=request.entity_type,
            time_window=time_window,
            pattern=request.pattern,
            limit=100
        )
        
        # Further filter by exact time range
        filtered_entities = [
            e for e in entities
            if request.start_time <= e.creation_time <= request.end_time
        ]
        
        # Convert to graph format (similar to get_graph_data)
        nodes = []
        entity_ids = set()
        
        for entity in filtered_entities:
            entity_ids.add(entity.entity_id)
            
            node = GraphNode(
                id=entity.entity_id,
                label=entity.name,
                type=entity.entity_type.value,
                properties={
                    "entity_type": entity.entity_type.value,
                    "name": entity.name,
                    "confidence_score": entity.confidence_score,
                    "importance_weight": entity.importance_weight,
                    "creation_time": entity.creation_time,
                    "tags": entity.tags
                }
            )
            nodes.append(node)
        
        # Get relationships
        edges = []
        relationship_types = set()
        
        for entity in filtered_entities:
            related = await service.get_related_entities(entity.entity_id)
            
            for related_entity, relationship in related:
                if (related_entity.entity_id in entity_ids and
                    request.start_time <= relationship.creation_time <= request.end_time):
                    
                    relationship_types.add(relationship.relationship_type.value)
                    
                    edge = GraphEdge(
                        id=relationship.relationship_id,
                        source=relationship.source_id,
                        target=relationship.target_id,
                        type=relationship.relationship_type.value,
                        properties={
                            "confidence": relationship.confidence,
                            "creation_time": relationship.creation_time
                        }
                    )
                    edges.append(edge)
        
        metadata = GraphMetadata(
            total_entities=len(nodes),
            total_relationships=len(edges),
            entity_types=list(set(e.type for e in nodes)),
            relationship_types=list(relationship_types),
            last_updated=time.time(),
            query_time=0.0
        )
        
        return GraphData(nodes=nodes, edges=edges, metadata=metadata)
        
    except Exception as e:
        logger.error(f"Temporal filter failed: {e}")
        raise HTTPException(status_code=500, detail=f"Temporal filter failed: {str(e)}")

@router.get("/entity/{entity_id}")
# @require_credentials  # TODO: Add authentication when available
async def get_entity_details(entity_id: str):
    """Get detailed information about a specific entity"""
    try:
        service = get_graphiti_service()
        entity = await service.get_entity(entity_id)
        
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")
        
        # Get related entities
        related = await service.get_related_entities(entity_id)
        
        return {
            "entity": {
                "entity_id": entity.entity_id,
                "entity_type": entity.entity_type.value,
                "name": entity.name,
                "attributes": entity.attributes,
                "creation_time": entity.creation_time,
                "modification_time": entity.modification_time,
                "access_frequency": entity.access_frequency,
                "confidence_score": entity.confidence_score,
                "importance_weight": entity.importance_weight,
                "tags": entity.tags
            },
            "related_entities": [
                {
                    "entity": {
                        "entity_id": related_entity.entity_id,
                        "name": related_entity.name,
                        "entity_type": related_entity.entity_type.value,
                        "confidence_score": related_entity.confidence_score
                    },
                    "relationship": {
                        "relationship_type": relationship.relationship_type.value,
                        "confidence": relationship.confidence
                    }
                }
                for related_entity, relationship in related
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get entity details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get entity: {str(e)}")


@router.get("/entities/{entity_id}/related")
# @require_credentials  # TODO: Add authentication when available
async def get_related_entities(entity_id: str):
    """Get entities related to a specific entity"""
    try:
        service = get_graphiti_service()
        
        # Get related entities with relationships
        related = await service.get_related_entities(entity_id)
        
        if not related:
            return {"related_entities": []}
        
        # Transform to match frontend expectations
        related_entities = []
        for related_entity, relationship in related:
            # Determine relationship direction based on entity IDs
            direction = 'outgoing'  # Default
            if hasattr(relationship, 'source_id') and hasattr(relationship, 'target_id'):
                if relationship.source_id != entity_id:
                    direction = 'incoming'
            
            related_entities.append({
                "entity": {
                    "id": related_entity.entity_id,
                    "type": related_entity.entity_type.value,
                    "label": related_entity.name,
                    "name": related_entity.name,
                    "properties": {
                        "confidence_score": related_entity.confidence_score,
                        "importance_weight": related_entity.importance_weight,
                        "access_frequency": related_entity.access_frequency,
                        "tags": related_entity.tags,
                        **related_entity.attributes
                    },
                    "created_at": related_entity.creation_time,
                    "updated_at": related_entity.modification_time
                },
                "relationship_type": relationship.relationship_type.value,
                "confidence": relationship.confidence,
                "direction": direction
            })
        
        return {"related_entities": related_entities}
        
    except Exception as e:
        logger.error(f"Failed to get related entities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get related entities: {str(e)}")


@router.post("/search")
# @require_credentials  # TODO: Add authentication when available
async def search_entities(request: EntitySearchRequest):
    """Search entities by name or tags"""
    try:
        service = get_graphiti_service()
        
        # Query entities with search filtering
        entities = await service.query_temporal(limit=request.limit * 2)  # Get more for filtering
        
        # Filter by search query
        search_lower = request.query.lower()
        filtered_entities = [
            e for e in entities
            if (search_lower in e.name.lower() or
                any(search_lower in tag.lower() for tag in e.tags))
        ]
        
        # Filter by entity types if specified
        if request.entity_types:
            filtered_entities = [
                e for e in filtered_entities
                if e.entity_type.value in request.entity_types
            ]
        
        # Limit results
        filtered_entities = filtered_entities[:request.limit]
        
        return {
            "entities": [
                {
                    "entity_id": entity.entity_id,
                    "entity_type": entity.entity_type.value,
                    "name": entity.name,
                    "confidence_score": entity.confidence_score,
                    "importance_weight": entity.importance_weight,
                    "tags": entity.tags,
                    "creation_time": entity.creation_time
                }
                for entity in filtered_entities
            ],
            "total": len(filtered_entities)
        }
        
    except Exception as e:
        logger.error(f"Entity search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/stats")
# @require_credentials  # TODO: Add authentication when available
async def get_graph_statistics():
    """Get comprehensive graph statistics and analytics"""
    try:
        service = get_graphiti_service()
        
        # Get performance stats
        perf_stats = service.get_performance_stats()
        
        # Get entity counts by type
        entity_type_counts = {}
        for entity_type in EntityType:
            entities = await service.query_temporal(entity_type=entity_type, limit=1000)
            entity_type_counts[entity_type.value] = len(entities)
        
        # Get recent activity (entities created in last 24h)
        recent_entities = await service.query_temporal(time_window="24h", limit=100)
        
        # Calculate trending entities (high access frequency)
        trending_entities = await service.query_temporal(pattern="trending", limit=10)
        
        return {
            "performance": perf_stats,
            "entity_counts": entity_type_counts,
            "total_entities": sum(entity_type_counts.values()),
            "recent_activity": {
                "entities_24h": len(recent_entities),
                "recent_entities": [
                    {
                        "entity_id": e.entity_id,
                        "name": e.name,
                        "entity_type": e.entity_type.value,
                        "creation_time": e.creation_time
                    }
                    for e in recent_entities[:5]
                ]
            },
            "trending": [
                {
                    "entity_id": e.entity_id,
                    "name": e.name,
                    "entity_type": e.entity_type.value,
                    "access_frequency": e.access_frequency,
                    "importance_weight": e.importance_weight
                }
                for e in trending_entities
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get graph statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@router.get("/available-actions")
async def get_available_actions():
    """Get list of available UI actions for the Graph Explorer"""
    return {
        "actions": [
            {
                "name": "zoom_in",
                "label": "Zoom In",
                "description": "Zoom into the graph visualization",
                "shortcut": "+"
            },
            {
                "name": "zoom_out", 
                "label": "Zoom Out",
                "description": "Zoom out of the graph visualization",
                "shortcut": "-"
            },
            {
                "name": "pan",
                "label": "Pan",
                "description": "Pan around the graph by dragging",
                "shortcut": "drag"
            },
            {
                "name": "filter_by_type",
                "label": "Filter by Type",
                "description": "Filter entities by their type",
                "shortcut": "F"
            },
            {
                "name": "temporal_filter",
                "label": "Time Filter",
                "description": "Filter by time range",
                "shortcut": "T"
            },
            {
                "name": "search",
                "label": "Search",
                "description": "Search entities by name or tags",
                "shortcut": "Ctrl+F"
            },
            {
                "name": "export_graph",
                "label": "Export Graph",
                "description": "Export graph data as JSON",
                "shortcut": "Ctrl+E"
            },
            {
                "name": "reset_layout",
                "label": "Reset Layout",
                "description": "Reset graph to default layout",
                "shortcut": "R"
            },
            {
                "name": "toggle_labels",
                "label": "Toggle Labels",
                "description": "Show/hide node labels",
                "shortcut": "L"
            },
            {
                "name": "fullscreen",
                "label": "Fullscreen",
                "description": "Enter fullscreen mode",
                "shortcut": "F11"
            }
        ]
    }