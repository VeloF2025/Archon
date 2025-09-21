#!/usr/bin/env python3
"""
Knowledge Feedback Service - Bidirectional Knowledge Sharing System

This service enables project-specific agents to share their learnings with the
global Archon knowledge base, and retrieve relevant knowledge for their tech stack.

CRITICAL: Part of the hardcoded Archon enhancement for project-specific agent learning.
"""

import json
import asyncio
import hashlib
import httpx
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeFeedbackService:
    """
    Manages bidirectional knowledge flow between project agents and global knowledge base.
    """
    
    def __init__(self, 
                 project_path: str,
                 archon_api: str = "http://localhost:8181",
                 knowledge_api: str = "http://localhost:3737"):
        """
        Initialize the knowledge feedback service for a specific project.
        
        Args:
            project_path: Path to the project directory
            archon_api: Archon API endpoint
            knowledge_api: Knowledge base API endpoint
        """
        self.project_path = Path(project_path)
        self.archon_api = archon_api
        self.knowledge_api = knowledge_api
        
        # Load project configuration
        self.project_config = self._load_project_config()
        self.project_id = self.project_config.get('project_id', self.project_path.name)
        self.tech_stack = self.project_config.get('tech_stack', {})
        
        # Knowledge categories for filtering
        self.knowledge_categories = {
            'code_patterns': 'Successful code implementations',
            'error_solutions': 'Error fixes and workarounds',
            'performance_optimizations': 'Performance improvements',
            'architecture_decisions': 'Architectural patterns that work',
            'test_strategies': 'Effective testing approaches',
            'security_fixes': 'Security vulnerability resolutions',
            'integration_patterns': 'Service/API integration patterns',
            'deployment_configs': 'Deployment configurations that work'
        }
        
        logger.info(f"Knowledge Feedback Service initialized for project: {self.project_id}")
    
    def _load_project_config(self) -> Dict:
        """Load project-specific agent configuration."""
        config_path = self.project_path / '.archon' / 'project_agents.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    async def submit_knowledge(self,
                              agent_id: str,
                              category: str,
                              title: str,
                              content: Any,
                              metadata: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Submit new knowledge from a project agent to the global knowledge base.
        
        Args:
            agent_id: ID of the project-specific agent submitting knowledge
            category: Knowledge category (from self.knowledge_categories)
            title: Brief title/summary of the knowledge
            content: The actual knowledge content (code, solution, pattern, etc.)
            metadata: Additional metadata about the knowledge
        
        Returns:
            Tuple of (success, knowledge_id or error_message)
        """
        try:
            # Generate unique ID for this knowledge item
            knowledge_id = self._generate_knowledge_id(agent_id, title, content)
            
            # Check if this knowledge already exists
            if await self._knowledge_exists(knowledge_id):
                logger.info(f"Knowledge {knowledge_id} already exists, updating quality score")
                return await self._update_knowledge_quality(knowledge_id)
            
            # Prepare knowledge entry
            knowledge_entry = {
                'knowledge_id': knowledge_id,
                'project_id': self.project_id,
                'agent_id': agent_id,
                'category': category,
                'title': title,
                'content': content if isinstance(content, str) else json.dumps(content, indent=2),
                'metadata': {
                    'tech_stack': self.tech_stack,
                    'timestamp': datetime.now().isoformat(),
                    'project_path': str(self.project_path),
                    'quality_score': 1.0,  # Initial quality score
                    'usage_count': 0,
                    'success_rate': 1.0,
                    **(metadata or {})
                },
                'tags': self._generate_tags(category, title, content)
            }
            
            # Submit to knowledge base
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.knowledge_api}/api/knowledge/submit",
                    json=knowledge_entry
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Successfully submitted knowledge: {knowledge_id}")
                    
                    # Store local reference
                    await self._store_local_reference(knowledge_id, knowledge_entry)
                    
                    return True, knowledge_id
                else:
                    error_msg = f"Failed to submit knowledge: {response.status_code}"
                    logger.error(error_msg)
                    return False, error_msg
                    
        except Exception as e:
            error_msg = f"Error submitting knowledge: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    async def query_knowledge(self,
                            query: str,
                            category: Optional[str] = None,
                            tech_filters: Optional[List[str]] = None,
                            limit: int = 5) -> List[Dict]:
        """
        Query relevant knowledge from the global knowledge base.
        
        Args:
            query: Search query
            category: Optional category filter
            tech_filters: Optional technology stack filters
            limit: Maximum number of results
        
        Returns:
            List of relevant knowledge items
        """
        try:
            # Build query parameters
            params = {
                'query': query,
                'limit': limit,
                'project_context': self.project_id
            }
            
            if category:
                params['category'] = category
            
            # Add tech stack filters
            if tech_filters or self.tech_stack:
                filters = tech_filters or list(self.tech_stack.get('frameworks', []))
                params['tech_filters'] = ','.join(filters)
            
            # Query knowledge base
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.knowledge_api}/api/knowledge/query",
                    params=params
                )
                
                if response.status_code == 200:
                    results = response.json().get('results', [])
                    
                    # Score results based on relevance to project
                    scored_results = self._score_results(results)
                    
                    # Sort by relevance score
                    scored_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
                    
                    logger.info(f"Found {len(scored_results)} relevant knowledge items for query: {query}")
                    return scored_results[:limit]
                else:
                    logger.error(f"Failed to query knowledge: {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error querying knowledge: {str(e)}")
            return []
    
    async def report_usage(self,
                          knowledge_id: str,
                          success: bool,
                          execution_time: Optional[float] = None,
                          error_message: Optional[str] = None) -> bool:
        """
        Report usage of a knowledge item to update its quality score.
        
        Args:
            knowledge_id: ID of the knowledge item used
            success: Whether the usage was successful
            execution_time: Optional execution time in seconds
            error_message: Optional error message if failed
        
        Returns:
            True if usage report was submitted successfully
        """
        try:
            usage_report = {
                'knowledge_id': knowledge_id,
                'project_id': self.project_id,
                'success': success,
                'execution_time': execution_time,
                'error_message': error_message,
                'timestamp': datetime.now().isoformat()
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.knowledge_api}/api/knowledge/report_usage",
                    json=usage_report
                )
                
                if response.status_code == 200:
                    logger.info(f"Successfully reported usage for knowledge: {knowledge_id}")
                    return True
                else:
                    logger.error(f"Failed to report usage: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error reporting usage: {str(e)}")
            return False
    
    async def get_recommendations(self,
                                 agent_id: str,
                                 task_type: str,
                                 limit: int = 3) -> List[Dict]:
        """
        Get knowledge recommendations for a specific agent and task type.
        
        Args:
            agent_id: ID of the project-specific agent
            task_type: Type of task being performed
            limit: Maximum number of recommendations
        
        Returns:
            List of recommended knowledge items
        """
        try:
            # Build recommendation request
            request_data = {
                'project_id': self.project_id,
                'agent_id': agent_id,
                'task_type': task_type,
                'tech_stack': self.tech_stack,
                'limit': limit
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.knowledge_api}/api/knowledge/recommendations",
                    json=request_data
                )
                
                if response.status_code == 200:
                    recommendations = response.json().get('recommendations', [])
                    logger.info(f"Got {len(recommendations)} recommendations for {agent_id}")
                    return recommendations
                else:
                    logger.error(f"Failed to get recommendations: {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return []
    
    def _generate_knowledge_id(self, agent_id: str, title: str, content: Any) -> str:
        """Generate a unique ID for a knowledge item."""
        content_str = str(content) if not isinstance(content, str) else content
        hash_input = f"{self.project_id}:{agent_id}:{title}:{content_str[:500]}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    async def _knowledge_exists(self, knowledge_id: str) -> bool:
        """Check if a knowledge item already exists."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.knowledge_api}/api/knowledge/{knowledge_id}"
                )
                return response.status_code == 200
        except:
            return False
    
    async def _update_knowledge_quality(self, knowledge_id: str) -> Tuple[bool, str]:
        """Update the quality score of existing knowledge."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.knowledge_api}/api/knowledge/{knowledge_id}/quality",
                    json={'increment': 0.1}
                )
                
                if response.status_code == 200:
                    return True, knowledge_id
                else:
                    return False, f"Failed to update quality: {response.status_code}"
        except Exception as e:
            return False, str(e)
    
    def _generate_tags(self, category: str, title: str, content: Any) -> List[str]:
        """Generate tags for a knowledge item."""
        tags = [category, self.project_id]
        
        # Add tech stack tags
        if self.tech_stack:
            tags.extend(self.tech_stack.get('frameworks', []))
            tags.extend(self.tech_stack.get('languages', []))
        
        # Add keywords from title
        title_words = title.lower().split()
        tags.extend([w for w in title_words if len(w) > 3])
        
        # Remove duplicates and return
        return list(set(tags))
    
    def _score_results(self, results: List[Dict]) -> List[Dict]:
        """Score knowledge results based on relevance to project."""
        scored_results = []
        
        for result in results:
            score = 0.0
            result_metadata = result.get('metadata', {})
            
            # Score based on tech stack match
            result_tech = result_metadata.get('tech_stack', {})
            if result_tech:
                frameworks_match = len(set(result_tech.get('frameworks', [])) & 
                                      set(self.tech_stack.get('frameworks', [])))
                languages_match = len(set(result_tech.get('languages', [])) & 
                                     set(self.tech_stack.get('languages', [])))
                score += (frameworks_match * 0.3) + (languages_match * 0.2)
            
            # Score based on quality and usage
            score += result_metadata.get('quality_score', 0) * 0.2
            score += min(result_metadata.get('usage_count', 0) / 100, 1.0) * 0.1
            score += result_metadata.get('success_rate', 0) * 0.2
            
            # Add relevance score to result
            result['relevance_score'] = min(score, 1.0)
            scored_results.append(result)
        
        return scored_results
    
    async def _store_local_reference(self, knowledge_id: str, entry: Dict):
        """Store a local reference to submitted knowledge."""
        local_kb_path = self.project_path / '.archon' / 'knowledge_submissions.json'
        local_kb_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing submissions
        submissions = {}
        if local_kb_path.exists():
            with open(local_kb_path, 'r') as f:
                submissions = json.load(f)
        
        # Add new submission
        submissions[knowledge_id] = {
            'title': entry['title'],
            'category': entry['category'],
            'timestamp': entry['metadata']['timestamp'],
            'agent_id': entry['agent_id']
        }
        
        # Save updated submissions
        with open(local_kb_path, 'w') as f:
            json.dump(submissions, f, indent=2)
    
    async def sync_knowledge(self) -> Dict[str, Any]:
        """
        Synchronize project knowledge with global knowledge base.
        
        Returns:
            Synchronization report
        """
        try:
            logger.info(f"Starting knowledge synchronization for project: {self.project_id}")
            
            # Get local submissions
            local_kb_path = self.project_path / '.archon' / 'knowledge_submissions.json'
            local_submissions = {}
            if local_kb_path.exists():
                with open(local_kb_path, 'r') as f:
                    local_submissions = json.load(f)
            
            # Sync with global knowledge base
            sync_report = {
                'project_id': self.project_id,
                'local_count': len(local_submissions),
                'synced': 0,
                'failed': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            for knowledge_id in local_submissions:
                if await self._knowledge_exists(knowledge_id):
                    sync_report['synced'] += 1
                else:
                    sync_report['failed'] += 1
            
            logger.info(f"Knowledge sync complete: {sync_report}")
            return sync_report
            
        except Exception as e:
            error_msg = f"Error during knowledge sync: {str(e)}"
            logger.error(error_msg)
            return {'error': error_msg}


# CLI Interface for testing
async def main():
    """CLI interface for testing knowledge feedback service."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python knowledge_feedback_service.py <project_path> [command] [args...]")
        print("Commands:")
        print("  submit <agent_id> <category> <title> <content>")
        print("  query <search_query> [category]")
        print("  recommend <agent_id> <task_type>")
        print("  sync")
        sys.exit(1)
    
    project_path = sys.argv[1]
    service = KnowledgeFeedbackService(project_path)
    
    if len(sys.argv) < 3:
        # Default: show project info
        print(f"Project: {service.project_id}")
        print(f"Tech Stack: {service.tech_stack}")
        return
    
    command = sys.argv[2]
    
    if command == "submit" and len(sys.argv) >= 7:
        success, result = await service.submit_knowledge(
            agent_id=sys.argv[3],
            category=sys.argv[4],
            title=sys.argv[5],
            content=sys.argv[6]
        )
        print(f"Submit result: {success}, {result}")
    
    elif command == "query" and len(sys.argv) >= 4:
        category = sys.argv[4] if len(sys.argv) > 4 else None
        results = await service.query_knowledge(sys.argv[3], category=category)
        print(f"Found {len(results)} results:")
        for r in results:
            print(f"  - {r.get('title')} (score: {r.get('relevance_score', 0):.2f})")
    
    elif command == "recommend" and len(sys.argv) >= 5:
        recommendations = await service.get_recommendations(sys.argv[3], sys.argv[4])
        print(f"Recommendations for {sys.argv[3]}:")
        for r in recommendations:
            print(f"  - {r.get('title')}")
    
    elif command == "sync":
        report = await service.sync_knowledge()
        print(f"Sync report: {json.dumps(report, indent=2)}")
    
    else:
        print(f"Unknown command or missing arguments: {command}")


if __name__ == "__main__":
    asyncio.run(main())