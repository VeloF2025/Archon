"""
Phase 6 Learning Service
Provides learning and improvement endpoints for Docker-based agent system
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx

logger = logging.getLogger(__name__)

# Create router for learning endpoints
router = APIRouter(prefix="/agents", tags=["learning"])

# In-memory storage for learning data (in production, use database)
learning_storage = {
    "execution_history": [],
    "performance_metrics": {},
    "patterns": {},
    "improvements": {}
}

class LearningRecord(BaseModel):
    """Learning record from agent execution"""
    timestamp: str
    agent_role: str
    invocation: Dict[str, Any]
    context: Dict[str, Any]
    session_id: str

class LearningUpdate(BaseModel):
    """Performance update from agent execution"""
    agent_id: str
    agent_type: str
    performance: Dict[str, float]
    result: Dict[str, Any]

class EmbeddingRequest(BaseModel):
    """Request to create embedding for knowledge base"""
    text: str
    metadata: Dict[str, Any]

@router.post("/learn")
async def store_learning_data(record: LearningRecord):
    """
    Store learning data from agent execution.
    This enables agents to learn from past executions.
    """
    try:
        # Store in memory
        learning_storage["execution_history"].append(record.dict())
        
        # Extract patterns
        agent_role = record.agent_role
        if agent_role not in learning_storage["patterns"]:
            learning_storage["patterns"][agent_role] = []
        
        # Store pattern for this agent type
        pattern = {
            "timestamp": record.timestamp,
            "context": record.context,
            "invocation": record.invocation
        }
        learning_storage["patterns"][agent_role].append(pattern)
        
        # Keep only last 100 patterns per agent
        if len(learning_storage["patterns"][agent_role]) > 100:
            learning_storage["patterns"][agent_role] = learning_storage["patterns"][agent_role][-100:]
        
        logger.info(f"Stored learning data for {agent_role}")
        
        # Also store in knowledge base if available
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    "http://localhost:8181/api/knowledge-items/crawl",
                    json={
                        "url": f"agent-execution://{record.session_id}",
                        "knowledge_type": "agent_pattern",
                        "tags": [agent_role, "phase6", "learning"],
                        "content": json.dumps(record.dict())
                    },
                    timeout=5.0
                )
        except Exception as e:
            logger.debug(f"Could not store in knowledge base: {e}")
        
        return {"status": "success", "message": "Learning data stored"}
        
    except Exception as e:
        logger.error(f"Failed to store learning data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update")
async def update_performance_metrics(update: LearningUpdate):
    """
    Update performance metrics for an agent.
    Tracks improvement over time.
    """
    try:
        agent_type = update.agent_type
        
        # Initialize metrics if needed
        if agent_type not in learning_storage["performance_metrics"]:
            learning_storage["performance_metrics"][agent_type] = {
                "total_executions": 0,
                "successful_executions": 0,
                "average_execution_time": 0,
                "success_rate": 0,
                "performance_scores": []
            }
        
        metrics = learning_storage["performance_metrics"][agent_type]
        
        # Update metrics
        metrics["total_executions"] += 1
        
        if update.result.get("status") == "completed":
            metrics["successful_executions"] += 1
        
        # Update success rate
        metrics["success_rate"] = metrics["successful_executions"] / metrics["total_executions"]
        
        # Update average execution time
        exec_time = update.performance.get("last_execution_time", 0)
        metrics["average_execution_time"] = (
            (metrics["average_execution_time"] * (metrics["total_executions"] - 1) + exec_time) 
            / metrics["total_executions"]
        )
        
        # Store performance score
        if "performance_score" in update.result:
            metrics["performance_scores"].append(update.result["performance_score"])
            # Keep only last 100 scores
            if len(metrics["performance_scores"]) > 100:
                metrics["performance_scores"] = metrics["performance_scores"][-100:]
        
        # Calculate improvements
        if metrics["total_executions"] > 10:
            recent_scores = metrics["performance_scores"][-10:]
            older_scores = metrics["performance_scores"][-20:-10] if len(metrics["performance_scores"]) > 10 else []
            
            if recent_scores and older_scores:
                recent_avg = sum(recent_scores) / len(recent_scores)
                older_avg = sum(older_scores) / len(older_scores)
                improvement = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
                
                learning_storage["improvements"][agent_type] = {
                    "improvement_percentage": improvement,
                    "recent_average": recent_avg,
                    "previous_average": older_avg,
                    "trend": "improving" if improvement > 0 else "declining"
                }
        
        logger.info(f"Updated performance metrics for {agent_type}")
        
        return {
            "status": "success",
            "metrics": metrics,
            "improvements": learning_storage["improvements"].get(agent_type, {})
        }
        
    except Exception as e:
        logger.error(f"Failed to update performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/patterns/{agent_role}")
async def get_agent_patterns(agent_role: str, limit: int = 5):
    """
    Get successful patterns for an agent type.
    Used to provide learning context.
    """
    patterns = learning_storage["patterns"].get(agent_role, [])
    
    # Return most recent successful patterns
    recent_patterns = patterns[-limit:] if patterns else []
    
    return {
        "agent_role": agent_role,
        "patterns": recent_patterns,
        "total_patterns": len(patterns)
    }

@router.get("/performance/{agent_type}")
async def get_performance_metrics(agent_type: str):
    """
    Get performance metrics for an agent type.
    Shows improvement over time.
    """
    metrics = learning_storage["performance_metrics"].get(agent_type, {})
    improvements = learning_storage["improvements"].get(agent_type, {})
    
    return {
        "agent_type": agent_type,
        "metrics": metrics,
        "improvements": improvements
    }

@router.post("/create_embedding")
async def create_embedding(request: EmbeddingRequest):
    """
    Create embedding for agent execution data.
    This makes the knowledge searchable.
    """
    try:
        # In production, this would use an embedding service
        # For now, return a mock embedding
        import hashlib
        
        # Create a simple hash-based "embedding"
        text_hash = hashlib.sha256(request.text.encode()).hexdigest()
        
        # Generate mock embedding (in reality, use OpenAI or similar)
        embedding = [float(int(text_hash[i:i+2], 16)) / 255 for i in range(0, min(64, len(text_hash)), 2)]
        
        # Pad to standard size
        while len(embedding) < 1536:
            embedding.append(0.0)
        
        return {
            "embedding": embedding[:1536],
            "metadata": request.metadata
        }
        
    except Exception as e:
        logger.error(f"Failed to create embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/learning/summary")
async def get_learning_summary():
    """
    Get overall learning system summary.
    Shows what the system has learned.
    """
    summary = {
        "total_executions": len(learning_storage["execution_history"]),
        "agent_types_tracked": list(learning_storage["patterns"].keys()),
        "performance_metrics": {
            agent: {
                "success_rate": metrics.get("success_rate", 0),
                "avg_execution_time": metrics.get("average_execution_time", 0),
                "total_executions": metrics.get("total_executions", 0)
            }
            for agent, metrics in learning_storage["performance_metrics"].items()
        },
        "improvements": learning_storage["improvements"],
        "top_performing_agents": sorted(
            [
                (agent, metrics.get("success_rate", 0))
                for agent, metrics in learning_storage["performance_metrics"].items()
            ],
            key=lambda x: x[1],
            reverse=True
        )[:5]
    }
    
    return summary

@router.post("/reset")
async def reset_learning_data():
    """
    Reset learning data (for testing).
    WARNING: This clears all learning history.
    """
    global learning_storage
    learning_storage = {
        "execution_history": [],
        "performance_metrics": {},
        "patterns": {},
        "improvements": {}
    }
    
    return {"status": "success", "message": "Learning data reset"}

# Helper function to analyze patterns
def analyze_patterns(agent_role: str) -> Dict[str, Any]:
    """
    Analyze patterns for an agent to identify what works best.
    """
    patterns = learning_storage["patterns"].get(agent_role, [])
    
    if not patterns:
        return {"message": "No patterns available"}
    
    # Analyze common contexts
    contexts = [p["context"] for p in patterns]
    
    # Find common keys in contexts
    common_keys = set()
    for ctx in contexts:
        common_keys.update(ctx.keys())
    
    analysis = {
        "total_patterns": len(patterns),
        "common_context_keys": list(common_keys),
        "recent_patterns": patterns[-3:],
        "pattern_frequency": len(patterns) / max(1, learning_storage["performance_metrics"].get(agent_role, {}).get("total_executions", 1))
    }
    
    return analysis