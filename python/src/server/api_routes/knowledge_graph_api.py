"""
Knowledge Graph API endpoints with GraphQL support
Provides comprehensive graph operations and queries
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import logging
from datetime import datetime
import json

from ..services.knowledge_graph_service import KnowledgeGraphService, get_knowledge_graph_service
from ...agents.knowledge_graph.query_engine import QueryResult
from ...agents.knowledge_graph.knowledge_ingestion import KnowledgeConcept, KnowledgeRelation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge-graph", tags=["Knowledge Graph"])


class GraphNodeRequest(BaseModel):
    """Request for creating a graph node"""
    labels: List[str]
    properties: Dict[str, Any]
    generate_embedding: bool = True


class GraphRelationshipRequest(BaseModel):
    """Request for creating a relationship"""
    source_id: str
    target_id: str
    relationship_type: str
    properties: Optional[Dict[str, Any]] = None
    strength: float = 1.0


class GraphQueryRequest(BaseModel):
    """Request for querying the graph"""
    query: str
    parameters: Optional[Dict[str, Any]] = None
    query_type: str = "cypher"  # "cypher", "natural", "template", "semantic"
    limit: int = 100


class IngestCodeRequest(BaseModel):
    """Request for ingesting code into the graph"""
    code: str
    language: str = "python"
    source_file: str
    project_id: Optional[str] = None


class IngestDocumentRequest(BaseModel):
    """Request for ingesting documentation"""
    content: str
    doc_type: str  # "README", "API", "TUTORIAL", etc.
    source: str
    project_id: Optional[str] = None


class DiscoverRelationshipsRequest(BaseModel):
    """Request for discovering relationships"""
    concept_id: str
    max_depth: int = 2
    min_strength: float = 0.5


class GraphVisualizationRequest(BaseModel):
    """Request for graph visualization data"""
    center_node_id: Optional[str] = None
    depth: int = 2
    max_nodes: int = 100
    include_orphans: bool = False


@router.post("/nodes", response_model=Dict[str, Any])
async def create_node(
    request: GraphNodeRequest,
    kg_service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
) -> Dict[str, Any]:
    """Create a new node in the knowledge graph"""
    try:
        node = await kg_service.create_node(
            labels=request.labels,
            properties=request.properties,
            generate_embedding=request.generate_embedding
        )
        
        return {
            "success": True,
            "node": node.dict() if hasattr(node, 'dict') else node,
            "message": f"Node created with ID: {node.id if hasattr(node, 'id') else node.get('id')}"
        }
    except Exception as e:
        logger.error(f"Error creating node: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/relationships", response_model=Dict[str, Any])
async def create_relationship(
    request: GraphRelationshipRequest,
    kg_service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
) -> Dict[str, Any]:
    """Create a relationship between nodes"""
    try:
        relationship = await kg_service.create_relationship(
            source_id=request.source_id,
            target_id=request.target_id,
            relationship_type=request.relationship_type,
            properties=request.properties,
            strength=request.strength
        )
        
        return {
            "success": True,
            "relationship": relationship.dict() if hasattr(relationship, 'dict') else relationship,
            "message": "Relationship created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=Dict[str, Any])
async def query_graph(
    request: GraphQueryRequest,
    kg_service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
) -> Dict[str, Any]:
    """Execute a query on the knowledge graph"""
    try:
        result = await kg_service.query(
            query=request.query,
            parameters=request.parameters,
            query_type=request.query_type
        )
        
        return {
            "success": True,
            "query": request.query,
            "query_type": request.query_type,
            "results": result.results if hasattr(result, 'results') else result,
            "count": result.count if hasattr(result, 'count') else len(result),
            "execution_time": result.execution_time if hasattr(result, 'execution_time') else 0.0,
            "visualizable": result.visualizable if hasattr(result, 'visualizable') else False
        }
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/code", response_model=Dict[str, Any])
async def ingest_code(
    request: IngestCodeRequest,
    kg_service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
) -> Dict[str, Any]:
    """Ingest code into the knowledge graph"""
    try:
        result = await kg_service.ingest_code(
            code=request.code,
            language=request.language,
            source_file=request.source_file,
            project_id=request.project_id
        )
        
        return result
    except Exception as e:
        logger.error(f"Error ingesting code: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/document", response_model=Dict[str, Any])
async def ingest_document(
    request: IngestDocumentRequest,
    kg_service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
) -> Dict[str, Any]:
    """Ingest documentation into the knowledge graph"""
    try:
        result = await kg_service.ingest_documentation(
            content=request.content,
            doc_type=request.doc_type,
            source=request.source,
            project_id=request.project_id
        )
        
        return result
    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/discover-relationships", response_model=Dict[str, Any])
async def discover_relationships(
    request: DiscoverRelationshipsRequest,
    kg_service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
) -> Dict[str, Any]:
    """Discover relationships for a concept"""
    try:
        relationships = await kg_service.discover_relationships(
            concept_id=request.concept_id,
            max_depth=request.max_depth,
            min_strength=request.min_strength
        )
        
        return {
            "success": True,
            "concept_id": request.concept_id,
            "discovered": relationships,
            "count": len(relationships)
        }
    except Exception as e:
        logger.error(f"Error discovering relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nodes/{node_id}", response_model=Dict[str, Any])
async def get_node(
    node_id: str,
    include_neighbors: bool = Query(False),
    depth: int = Query(1, ge=1, le=5),
    kg_service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
) -> Dict[str, Any]:
    """Get a node by ID with optional neighbors"""
    try:
        node = await kg_service.get_node(node_id)
        
        if not node:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
        
        result = {
            "success": True,
            "node": node
        }
        
        if include_neighbors:
            neighbors = await kg_service.get_neighbors(node_id, depth)
            result["neighbors"] = neighbors
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting node: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search", response_model=Dict[str, Any])
async def search_nodes(
    query: str = Query(..., description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(20, ge=1, le=100),
    use_semantic: bool = Query(False, description="Use semantic search"),
    kg_service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
) -> Dict[str, Any]:
    """Search for nodes in the knowledge graph"""
    try:
        if use_semantic:
            results = await kg_service.semantic_search(query, limit)
        else:
            results = await kg_service.search_nodes(query, category, limit)
        
        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results),
            "search_type": "semantic" if use_semantic else "keyword"
        }
    except Exception as e:
        logger.error(f"Error searching nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/path/{source_id}/{target_id}", response_model=Dict[str, Any])
async def find_shortest_path(
    source_id: str,
    target_id: str,
    max_depth: int = Query(5, ge=1, le=10),
    kg_service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
) -> Dict[str, Any]:
    """Find shortest path between two nodes"""
    try:
        path = await kg_service.find_shortest_path(source_id, target_id, max_depth)
        
        if not path:
            return {
                "success": False,
                "message": f"No path found between {source_id} and {target_id} within depth {max_depth}"
            }
        
        return {
            "success": True,
            "source_id": source_id,
            "target_id": target_id,
            "path": path,
            "length": len(path.get("nodes", [])) - 1 if path else 0
        }
    except Exception as e:
        logger.error(f"Error finding path: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/visualize", response_model=Dict[str, Any])
async def get_visualization_data(
    request: GraphVisualizationRequest,
    kg_service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
) -> Dict[str, Any]:
    """Get graph data formatted for visualization"""
    try:
        data = await kg_service.get_visualization_data(
            center_node_id=request.center_node_id,
            depth=request.depth,
            max_nodes=request.max_nodes,
            include_orphans=request.include_orphans
        )
        
        return {
            "success": True,
            "data": data,
            "node_count": len(data.get("nodes", [])),
            "edge_count": len(data.get("edges", []))
        }
    except Exception as e:
        logger.error(f"Error getting visualization data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics", response_model=Dict[str, Any])
async def get_graph_statistics(
    kg_service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
) -> Dict[str, Any]:
    """Get comprehensive graph statistics"""
    try:
        stats = await kg_service.get_statistics()
        
        return {
            "success": True,
            "statistics": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_graph(
    analysis_type: str = Query("comprehensive", enum=["patterns", "centrality", "communities", "comprehensive"]),
    kg_service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
) -> Dict[str, Any]:
    """Analyze the knowledge graph structure"""
    try:
        analysis = await kg_service.analyze_graph(analysis_type)
        
        return {
            "success": True,
            "analysis_type": analysis_type,
            "results": analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error analyzing graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations", response_model=Dict[str, Any])
async def get_query_recommendations(
    context: Optional[str] = Query(None, description="Context for recommendations"),
    kg_service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
) -> Dict[str, Any]:
    """Get recommended queries based on context"""
    try:
        recommendations = await kg_service.get_query_recommendations(context)
        
        return {
            "success": True,
            "context": context,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export", response_model=Dict[str, Any])
async def export_subgraph(
    center_node_id: str,
    depth: int = Query(2, ge=1, le=5),
    format: str = Query("json", enum=["json", "cypher", "graphml"]),
    kg_service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
) -> Dict[str, Any]:
    """Export a subgraph centered on a node"""
    try:
        export_data = await kg_service.export_subgraph(
            center_node_id=center_node_id,
            depth=depth,
            format=format
        )
        
        return {
            "success": True,
            "center_node_id": center_node_id,
            "depth": depth,
            "format": format,
            "data": export_data
        }
    except Exception as e:
        logger.error(f"Error exporting subgraph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/nodes/{node_id}", response_model=Dict[str, Any])
async def delete_node(
    node_id: str,
    kg_service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
) -> Dict[str, Any]:
    """Delete a node and its relationships"""
    try:
        success = await kg_service.delete_node(node_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
        
        return {
            "success": True,
            "message": f"Node {node_id} and its relationships deleted"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting node: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=Dict[str, Any])
async def health_check(
    kg_service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
) -> Dict[str, Any]:
    """Check knowledge graph service health"""
    try:
        health = await kg_service.health_check()
        
        return {
            "success": True,
            "status": "healthy" if health else "unhealthy",
            "services": {
                "neo4j": health.get("neo4j", False) if isinstance(health, dict) else health,
                "kafka": health.get("kafka", False) if isinstance(health, dict) else False,
                "redis": health.get("redis", False) if isinstance(health, dict) else False
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error checking health: {e}")
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }