"""
Pattern Recognition API Routes
Endpoints for pattern detection, analysis, and recommendations
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Body, Query
from pydantic import BaseModel, Field
from datetime import datetime

from ..utils import get_supabase_client
from ...agents.pattern_recognition import (
    PatternDetector,
    PatternStorage,
    PatternAnalyzer,
    PatternRecommender
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/patterns", tags=["Pattern Recognition"])


# Request/Response Models
class PatternDetectionRequest(BaseModel):
    """Request for pattern detection"""
    code: str = Field(description="Source code to analyze")
    language: str = Field(default="python", description="Programming language")
    project_id: Optional[str] = Field(default=None, description="Associated project ID")
    detect_antipatterns: bool = Field(default=True, description="Include antipattern detection")


class PatternRecommendationRequest(BaseModel):
    """Request for pattern recommendations"""
    context: str = Field(description="Code context or description")
    language: str = Field(description="Programming language")
    requirements: Optional[List[str]] = Field(default=None, description="Specific requirements")
    avoid_antipatterns: bool = Field(default=True, description="Exclude antipatterns")
    limit: int = Field(default=5, description="Maximum recommendations")


class RefactoringRequest(BaseModel):
    """Request for refactoring recommendations"""
    code: str = Field(description="Code to refactor")
    language: str = Field(description="Programming language")
    focus: Optional[str] = Field(default=None, description="Focus area: performance, maintainability, security")


class PatternFeedbackRequest(BaseModel):
    """Feedback on pattern usefulness"""
    pattern_id: str = Field(description="Pattern ID")
    useful: bool = Field(description="Was the pattern useful?")
    effective: bool = Field(description="Was the pattern effective?")
    context: Optional[str] = Field(default=None, description="Usage context")


class PatternSearchRequest(BaseModel):
    """Search for similar patterns"""
    query: str = Field(description="Search query")
    language: Optional[str] = Field(default=None, description="Filter by language")
    category: Optional[str] = Field(default=None, description="Filter by category")
    limit: int = Field(default=10, description="Maximum results")
    threshold: float = Field(default=0.7, description="Similarity threshold")


# Initialize services
detector = PatternDetector()
storage = PatternStorage()
analyzer = PatternAnalyzer(storage)
recommender = PatternRecommender(storage, analyzer)


@router.post("/detect")
async def detect_patterns(request: PatternDetectionRequest):
    """
    Detect patterns in provided code
    
    Returns:
        List of detected patterns with confidence scores
    """
    try:
        logger.info(f"Detecting patterns in {request.language} code")
        
        # Detect patterns
        patterns = await detector.detect_patterns(
            code=request.code,
            language=request.language
        )
        
        # Store patterns if project associated
        if request.project_id:
            for pattern in patterns:
                pattern_dict = pattern.dict()
                pattern_dict["project_id"] = request.project_id
                await storage.store_pattern(pattern_dict)
        
        # Analyze patterns
        analysis_results = []
        for pattern in patterns[:5]:  # Analyze top 5 patterns
            analysis = await analyzer.analyze_pattern(pattern.id)
            if analysis:
                analysis_results.append(analysis.dict())
        
        return {
            "success": True,
            "patterns": [p.dict() for p in patterns],
            "analysis": analysis_results,
            "summary": {
                "total_patterns": len(patterns),
                "antipatterns": len([p for p in patterns if p.is_antipattern]),
                "languages": list(set(p.language for p in patterns)),
                "categories": list(set(p.category for p in patterns))
            }
        }
        
    except Exception as e:
        logger.error(f"Error detecting patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend")
async def recommend_patterns(request: PatternRecommendationRequest):
    """
    Get pattern recommendations based on context
    
    Returns:
        Recommended patterns with relevance scores
    """
    try:
        logger.info(f"Generating pattern recommendations for {request.language}")
        
        # Get recommendations
        recommendations = await recommender.recommend_patterns(
            context=request.context,
            language=request.language,
            requirements=request.requirements,
            avoid_antipatterns=request.avoid_antipatterns,
            limit=request.limit
        )
        
        return {
            "success": True,
            "recommendations": [r.dict() for r in recommendations],
            "total": len(recommendations)
        }
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refactor")
async def get_refactoring_recommendations(request: RefactoringRequest):
    """
    Get pattern recommendations for code refactoring
    
    Returns:
        Refactoring recommendations with antipattern replacements
    """
    try:
        logger.info(f"Generating refactoring recommendations for {request.language}")
        
        # Detect current patterns
        current_patterns = await detector.detect_patterns(
            code=request.code,
            language=request.language
        )
        
        # Get refactoring recommendations
        recommendations = await recommender.recommend_for_refactoring(
            code=request.code,
            language=request.language,
            focus=request.focus
        )
        
        # Identify issues
        antipatterns = [p for p in current_patterns if p.is_antipattern]
        
        return {
            "success": True,
            "current_patterns": [p.dict() for p in current_patterns],
            "antipatterns_found": [p.dict() for p in antipatterns],
            "recommendations": [r.dict() for r in recommendations],
            "refactoring_priority": "high" if len(antipatterns) > 2 else "medium" if antipatterns else "low"
        }
        
    except Exception as e:
        logger.error(f"Error generating refactoring recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_patterns(request: PatternSearchRequest):
    """
    Search for similar patterns
    
    Returns:
        Similar patterns with similarity scores
    """
    try:
        logger.info(f"Searching for patterns: {request.query}")
        
        # Search patterns
        patterns = await storage.search_similar_patterns(
            query=request.query,
            language=request.language,
            category=request.category,
            limit=request.limit,
            threshold=request.threshold
        )
        
        return {
            "success": True,
            "patterns": [p.dict() for p in patterns],
            "total": len(patterns)
        }
        
    except Exception as e:
        logger.error(f"Error searching patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/top")
async def get_top_patterns(
    language: Optional[str] = Query(None, description="Filter by language"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(20, description="Maximum results")
):
    """
    Get top patterns by effectiveness and usage
    
    Returns:
        Most effective and frequently used patterns
    """
    try:
        logger.info("Fetching top patterns")
        
        # Get top patterns
        patterns = await storage.get_top_patterns(
            language=language,
            category=category,
            limit=limit
        )
        
        return {
            "success": True,
            "patterns": [p.dict() for p in patterns],
            "total": len(patterns)
        }
        
    except Exception as e:
        logger.error(f"Error fetching top patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/antipatterns")
async def get_antipatterns(
    language: Optional[str] = Query(None, description="Filter by language")
):
    """
    Get all detected antipatterns
    
    Returns:
        List of antipatterns with alternatives
    """
    try:
        logger.info("Fetching antipatterns")
        
        # Get antipatterns
        antipatterns = await storage.get_antipatterns(language=language)
        
        return {
            "success": True,
            "antipatterns": [p.dict() for p in antipatterns],
            "total": len(antipatterns),
            "by_language": {}  # Group by language
        }
        
    except Exception as e:
        logger.error(f"Error fetching antipatterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights")
async def get_pattern_insights(
    language: Optional[str] = Query(None, description="Filter by language"),
    timeframe_days: int = Query(30, description="Analysis timeframe in days")
):
    """
    Get aggregated insights about patterns
    
    Returns:
        Pattern statistics, trends, and recommendations
    """
    try:
        logger.info("Generating pattern insights")
        
        # Get insights
        insights = await analyzer.get_pattern_insights(
            language=language,
            timeframe_days=timeframe_days
        )
        
        return {
            "success": True,
            "insights": insights,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pattern/{pattern_id}")
async def get_pattern_details(pattern_id: str):
    """
    Get detailed information about a specific pattern
    
    Returns:
        Pattern details with analysis and examples
    """
    try:
        logger.info(f"Fetching pattern details: {pattern_id}")
        
        # Get pattern
        pattern = await storage.get_pattern_by_id(pattern_id)
        if not pattern:
            raise HTTPException(status_code=404, detail="Pattern not found")
        
        # Get analysis
        analysis = await analyzer.analyze_pattern(pattern_id)
        
        # Get complementary patterns
        complementary = await recommender.recommend_complementary_patterns(
            pattern_id=pattern_id,
            limit=3
        )
        
        # Get example
        example = await recommender.get_pattern_example(pattern_id)
        
        return {
            "success": True,
            "pattern": pattern.dict(),
            "analysis": analysis.dict() if analysis else None,
            "complementary_patterns": [c.dict() for c in complementary],
            "example": example
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching pattern details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_pattern_feedback(request: PatternFeedbackRequest):
    """
    Submit feedback on pattern usefulness
    
    Returns:
        Confirmation of feedback submission
    """
    try:
        logger.info(f"Submitting feedback for pattern: {request.pattern_id}")
        
        # Update pattern usage statistics
        await storage.update_pattern_usage(
            pattern_id=request.pattern_id,
            effective=request.effective
        )
        
        # Update detector learning
        await detector.learn_from_feedback(
            pattern_id=request.pattern_id,
            useful=request.useful
        )
        
        return {
            "success": True,
            "message": "Feedback submitted successfully",
            "pattern_id": request.pattern_id
        }
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/relationships")
async def get_pattern_relationships():
    """
    Get relationships between patterns
    
    Returns:
        Pattern relationship graph
    """
    try:
        logger.info("Analyzing pattern relationships")
        
        # Analyze relationships
        relationships = await analyzer.analyze_pattern_relationships()
        
        # Convert to graph format
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        # Add nodes
        seen_nodes = set()
        for rel in relationships:
            if rel.pattern1_id not in seen_nodes:
                pattern = await storage.get_pattern_by_id(rel.pattern1_id)
                if pattern:
                    graph_data["nodes"].append({
                        "id": rel.pattern1_id,
                        "name": pattern.name,
                        "category": pattern.category
                    })
                    seen_nodes.add(rel.pattern1_id)
            
            if rel.pattern2_id not in seen_nodes:
                pattern = await storage.get_pattern_by_id(rel.pattern2_id)
                if pattern:
                    graph_data["nodes"].append({
                        "id": rel.pattern2_id,
                        "name": pattern.name,
                        "category": pattern.category
                    })
                    seen_nodes.add(rel.pattern2_id)
            
            # Add edge
            graph_data["edges"].append({
                "source": rel.pattern1_id,
                "target": rel.pattern2_id,
                "type": rel.relationship_type,
                "strength": rel.strength
            })
        
        return {
            "success": True,
            "relationships": [r.dict() for r in relationships],
            "graph": graph_data,
            "total_relationships": len(relationships)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_patterns(
    filepath: str = Body(default="/tmp/patterns_export.json", description="Export file path")
):
    """
    Export all patterns to JSON file
    
    Returns:
        Export confirmation
    """
    try:
        logger.info(f"Exporting patterns to {filepath}")
        
        # Export patterns
        success = await storage.export_patterns_to_json(filepath)
        
        if success:
            return {
                "success": True,
                "message": f"Patterns exported to {filepath}",
                "filepath": filepath
            }
        else:
            raise HTTPException(status_code=500, detail="Export failed")
        
    except Exception as e:
        logger.error(f"Error exporting patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))