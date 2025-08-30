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
# @require_credentials  # TODO: Add authentication when available
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
    Get graph data with optional filtering
    
    Returns nodes and edges formatted for visualization
    """
    start_time = time.time()
    
    try:
        service = get_graphiti_service()
        
        # Parse comma-separated filters
        entity_type_list = entity_types.split(',') if entity_types else None
        relationship_type_list = relationship_types.split(',') if relationship_types else None
        
        # Query entities with temporal filtering
        entities = await service.query_temporal(
            entity_type=entity_type_list[0] if entity_type_list else None,
            time_window=time_window,
            limit=limit
        )
        
        # Filter by search term if provided
        if search_term:
            search_lower = search_term.lower()
            entities = [
                e for e in entities 
                if search_lower in e.name.lower() or 
                any(search_lower in tag.lower() for tag in e.tags)
            ]
        
        # Filter by confidence and importance thresholds
        entities = [
            e for e in entities 
            if e.confidence_score >= confidence_threshold and 
            e.importance_weight >= importance_threshold
        ]
        
        # Convert entities to nodes
        nodes = []
        entity_ids = set()
        
        for entity in entities:
            entity_ids.add(entity.entity_id)
            
            # Create node with position for layout
            node = GraphNode(
                id=entity.entity_id,
                label=entity.name,
                type=entity.entity_type.value,
                properties={
                    "entity_type": entity.entity_type.value,
                    "name": entity.name,
                    "attributes": entity.attributes,
                    "creation_time": entity.creation_time,
                    "modification_time": entity.modification_time,
                    "access_frequency": entity.access_frequency,
                    "confidence_score": entity.confidence_score,
                    "importance_weight": entity.importance_weight,
                    "tags": entity.tags
                }
            )
            nodes.append(node)
        
        # Get relationships between filtered entities
        edges = []
        all_entity_types = set()
        all_relationship_types = set()
        
        for entity in entities:
            all_entity_types.add(entity.entity_type.value)
            
            # Get related entities for this entity
            related = await service.get_related_entities(entity.entity_id, max_depth=1)
            
            for related_entity, relationship in related:
                # Only include if target entity is in our filtered set
                if related_entity.entity_id in entity_ids:
                    all_relationship_types.add(relationship.relationship_type.value)
                    
                    edge = GraphEdge(
                        id=relationship.relationship_id,
                        source=relationship.source_id,
                        target=relationship.target_id,
                        type=relationship.relationship_type.value,
                        properties={
                            "relationship_type": relationship.relationship_type.value,
                            "confidence": relationship.confidence,
                            "creation_time": relationship.creation_time,
                            "modification_time": relationship.modification_time,
                            "access_frequency": relationship.access_frequency,
                            "temporal_data": relationship.temporal_data,
                            "attributes": relationship.attributes
                        }
                    )
                    edges.append(edge)
        
        # Create metadata
        query_time = time.time() - start_time
        metadata = GraphMetadata(
            total_entities=len(nodes),
            total_relationships=len(edges),
            entity_types=list(all_entity_types),
            relationship_types=list(all_relationship_types),
            last_updated=time.time(),
            query_time=query_time
        )
        
        return GraphData(
            nodes=nodes,
            edges=edges,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Failed to get graph data: {e}")
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