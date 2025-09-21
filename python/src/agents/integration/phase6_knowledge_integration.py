"""
Phase 6 Knowledge Base Integration
Connects parallel Claude sub-agents to Archon's knowledge base for learning and improvement
"""

import asyncio
import json
import logging
import httpx
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

class Phase6KnowledgeIntegration:
    """
    Integrates Phase 6 parallel Claude agents with Archon's knowledge base.
    Enables learning, pattern recognition, and continuous improvement.
    """
    
    def __init__(self, 
                 archon_api: str = "http://localhost:8181",
                 mcp_server: str = "http://localhost:8051", 
                 agents_service: str = "http://localhost:8052"):
        
        self.archon_api = archon_api
        self.mcp_server = mcp_server
        self.agents_service = agents_service
        
        # Knowledge base collections
        self.knowledge_collections = {
            "agent_patterns": "Successful execution patterns by agent type",
            "code_examples": "Working code examples from successful runs",
            "error_solutions": "Common errors and their solutions",
            "performance_optimizations": "Performance improvements discovered",
            "architectural_decisions": "Architecture patterns that work",
            "test_patterns": "Effective testing strategies",
            "security_fixes": "Security vulnerabilities and fixes"
        }
        
        logger.info("Phase 6 Knowledge Integration initialized with Archon knowledge base")
    
    async def store_agent_execution(self, 
                                   agent_role: str,
                                   task: str,
                                   result: Dict,
                                   context: Dict) -> bool:
        """
        Store agent execution in Archon's knowledge base for learning.
        This is what makes agents improve over time.
        Enhanced with project-specific context for better learning.
        """
        
        try:
            # Extract project-specific information
            project_info = await self._get_project_info(context.get("project_path"))
            
            # Create knowledge entry with enhanced metadata
            knowledge_entry = {
                "source_type": "agent_execution",
                "source_name": f"phase6_{agent_role}",
                "content": {
                    "agent_role": agent_role,
                    "task": task,
                    "result": result,
                    "context": context,
                    "timestamp": datetime.now().isoformat(),
                    "success": result.get("status") == "completed",
                    "execution_time": result.get("execution_time", 0),
                    "performance_score": result.get("performance_score", 0)
                },
                "metadata": {
                    "phase": 6,
                    "agent_type": self._get_agent_type(agent_role),
                    "project": context.get("project", "archon"),
                    "project_type": project_info.get("type", "unknown"),
                    "tech_stack": project_info.get("tech_stack", {}),
                    "tags": self._generate_tags(agent_role, task, result),
                    "quality_score": self._calculate_quality_score(result),
                    "reusable": self._is_pattern_reusable(result, context)
                }
            }
            
            # Store in knowledge base via API
            async with httpx.AsyncClient() as client:
                # First, create embedding for searchability
                embedding_response = await client.post(
                    f"{self.agents_service}/agents/create_embedding",
                    json={
                        "text": f"{agent_role}: {task}\n{json.dumps(result, indent=2)}",
                        "metadata": knowledge_entry["metadata"]
                    }
                )
                
                if embedding_response.status_code == 200:
                    embedding = embedding_response.json().get("embedding")
                    knowledge_entry["embedding"] = embedding
                
                # Store in knowledge base
                response = await client.post(
                    f"{self.archon_api}/api/knowledge/store",
                    json=knowledge_entry
                )
                
                if response.status_code == 200:
                    logger.info(f"Stored {agent_role} execution in knowledge base")
                    
                    # Also store specific patterns
                    await self._extract_and_store_patterns(agent_role, task, result)
                    return True
                else:
                    logger.error(f"Failed to store in knowledge base: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error storing agent execution: {e}")
            return False
    
    async def retrieve_agent_knowledge(self, 
                                      agent_role: str,
                                      task_description: str,
                                      project_context: Optional[Dict] = None,
                                      limit: int = 5) -> List[Dict]:
        """
        Retrieve relevant knowledge for an agent from Archon's knowledge base.
        Enhanced with project-context-aware queries for better relevance.
        This provides learning context for better performance.
        """
        
        try:
            # Build enhanced search filters with project context
            filters = {
                "agent_role": agent_role,
                "success": True  # Only successful patterns
            }
            
            # Add project-specific filters if available
            if project_context:
                if 'tech_stack' in project_context:
                    filters['tech_stack'] = project_context['tech_stack']
                if 'project_type' in project_context:
                    filters['project_type'] = project_context['project_type']
            
            # Search knowledge base for relevant patterns
            async with httpx.AsyncClient() as client:
                # Use RAG search to find relevant knowledge
                search_response = await client.post(
                    f"{self.archon_api}/api/knowledge/search",
                    json={
                        "query": f"{agent_role} {task_description}",
                        "filters": filters,
                        "boost_fields": ["tech_stack", "project_type"],  # Prioritize matching tech
                        "limit": limit
                    }
                )
                
                if search_response.status_code == 200:
                    results = search_response.json().get("results", [])
                    
                    # Extract relevant patterns
                    knowledge = []
                    for result in results:
                        knowledge.append({
                            "pattern": result.get("content", {}).get("task"),
                            "solution": result.get("content", {}).get("result"),
                            "context": result.get("content", {}).get("context"),
                            "performance_score": result.get("content", {}).get("performance_score", 0)
                        })
                    
                    logger.info(f"Retrieved {len(knowledge)} knowledge items for {agent_role}")
                    return knowledge
                else:
                    logger.warning(f"Could not retrieve knowledge: {search_response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {e}")
            return []
    
    async def _extract_and_store_patterns(self, 
                                         agent_role: str,
                                         task: str,
                                         result: Dict):
        """
        Extract specific patterns from execution and store them.
        This builds the pattern library for each agent type.
        """
        
        patterns_to_store = []
        
        # Extract patterns based on agent role
        if agent_role == "python_backend_coder":
            # Store API patterns, database patterns, etc.
            if "fastapi" in task.lower():
                patterns_to_store.append({
                    "type": "api_pattern",
                    "pattern": "FastAPI endpoint implementation",
                    "code": result.get("output", ""),
                    "tags": ["fastapi", "api", "backend"]
                })
        
        elif agent_role == "security_auditor":
            # Store security patterns
            if "vulnerability" in str(result).lower():
                patterns_to_store.append({
                    "type": "security_pattern", 
                    "pattern": "Security vulnerability detection",
                    "findings": result.get("output", ""),
                    "tags": ["security", "vulnerability", "audit"]
                })
        
        elif agent_role == "test_generator":
            # Store testing patterns
            if "test" in task.lower():
                patterns_to_store.append({
                    "type": "test_pattern",
                    "pattern": "Test generation strategy",
                    "tests": result.get("output", ""),
                    "coverage": result.get("coverage", 0),
                    "tags": ["testing", "coverage", "quality"]
                })
        
        # Store patterns in knowledge base
        for pattern in patterns_to_store:
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"{self.archon_api}/api/knowledge/patterns/store",
                        json={
                            "agent_role": agent_role,
                            "pattern": pattern,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
            except Exception as e:
                logger.debug(f"Could not store pattern: {e}")
    
    async def get_learning_context(self, 
                                  agent_role: str,
                                  task: str) -> str:
        """
        Get learning context from knowledge base for an agent.
        This is injected into the agent's prompt for better performance.
        """
        
        # Retrieve relevant knowledge
        knowledge_items = await self.retrieve_agent_knowledge(agent_role, task)
        
        if not knowledge_items:
            return "No previous patterns found in knowledge base."
        
        # Build learning context
        context = f"Learning from Archon Knowledge Base ({len(knowledge_items)} relevant patterns):\n\n"
        
        for i, item in enumerate(knowledge_items[:3], 1):  # Top 3 most relevant
            context += f"{i}. Previous Success Pattern:\n"
            context += f"   Task: {item['pattern']}\n"
            
            if item['performance_score'] > 0.8:
                context += f"   Performance: Excellent ({item['performance_score']:.2f})\n"
            
            # Add relevant solution snippets
            if item.get('solution'):
                solution_preview = str(item['solution'])[:200]
                context += f"   Approach: {solution_preview}...\n"
            
            context += "\n"
        
        # Add specific tips based on agent role
        role_tips = {
            "python_backend_coder": "Tip: Use async/await patterns and connection pooling for better performance.",
            "security_auditor": "Tip: Focus on input validation and authentication flows first.",
            "test_generator": "Tip: Parameterized tests and fixtures improve maintainability.",
            "system_architect": "Tip: Consider microservices for complex domains.",
            "devops_engineer": "Tip: Container orchestration with Kubernetes scales better."
        }
        
        if agent_role in role_tips:
            context += f"\n{role_tips[agent_role]}\n"
        
        return context
    
    async def analyze_performance_trends(self, agent_role: str) -> Dict:
        """
        Analyze performance trends for an agent type from knowledge base.
        Identifies what's working and what needs improvement.
        """
        
        try:
            async with httpx.AsyncClient() as client:
                # Query performance metrics from knowledge base
                response = await client.post(
                    f"{self.archon_api}/api/knowledge/analytics",
                    json={
                        "agent_role": agent_role,
                        "metrics": ["execution_time", "success_rate", "performance_score"],
                        "time_range": "last_7_days"
                    }
                )
                
                if response.status_code == 200:
                    analytics = response.json()
                    
                    trends = {
                        "agent_role": agent_role,
                        "avg_execution_time": analytics.get("avg_execution_time", 0),
                        "success_rate": analytics.get("success_rate", 0),
                        "performance_trend": analytics.get("trend", "stable"),
                        "top_patterns": analytics.get("top_patterns", []),
                        "common_errors": analytics.get("common_errors", []),
                        "recommendations": self._generate_recommendations(analytics)
                    }
                    
                    return trends
                else:
                    return {"error": "Could not retrieve analytics"}
                    
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, analytics: Dict) -> List[str]:
        """Generate recommendations based on performance analytics"""
        
        recommendations = []
        
        if analytics.get("success_rate", 0) < 0.8:
            recommendations.append("Success rate below 80% - review error patterns")
        
        if analytics.get("avg_execution_time", 0) > 30:
            recommendations.append("Execution time high - consider optimization")
        
        common_errors = analytics.get("common_errors", [])
        if common_errors:
            recommendations.append(f"Address recurring error: {common_errors[0]}")
        
        return recommendations
    
    def _get_agent_type(self, agent_role: str) -> str:
        """Map agent role to Claude Code subagent type"""
        
        mappings = {
            "python_backend_coder": "code-implementer",
            "security_auditor": "security-auditor",
            "test_generator": "test-coverage-validator",
            "system_architect": "system-architect",
            "devops_engineer": "devops-automation"
        }
        
        return mappings.get(agent_role, "general-purpose")
    
    def _generate_tags(self, agent_role: str, task: str, result: Dict) -> List[str]:
        """Generate searchable tags for knowledge base entry"""
        
        tags = [agent_role, "phase6", self._get_agent_type(agent_role)]
        
        # Add task-based tags
        task_lower = task.lower()
        if "api" in task_lower:
            tags.append("api")
        if "test" in task_lower:
            tags.append("testing")
        if "security" in task_lower:
            tags.append("security")
        if "database" in task_lower:
            tags.append("database")
        
        # Add result-based tags
        if result.get("status") == "completed":
            tags.append("successful")
        
        if result.get("performance_score", 0) > 0.9:
            tags.append("high_performance")
        
        return tags
    
    async def _get_project_info(self, project_path: Optional[str]) -> Dict:
        """Get project-specific information for context"""
        if not project_path:
            return {}
        
        project_info = {
            "type": "unknown",
            "tech_stack": {}
        }
        
        try:
            path = Path(project_path)
            
            # Check for package.json (Node.js projects)
            package_json = path / "package.json"
            if package_json.exists():
                with open(package_json, 'r') as f:
                    pkg_data = json.load(f)
                    deps = {**pkg_data.get('dependencies', {}), **pkg_data.get('devDependencies', {})}
                    
                    # Detect frameworks
                    frameworks = []
                    if 'react' in deps:
                        frameworks.append('React')
                    if 'vue' in deps:
                        frameworks.append('Vue')
                    if 'angular' in deps:
                        frameworks.append('Angular')
                    if 'next' in deps:
                        frameworks.append('Next.js')
                    
                    project_info['type'] = 'javascript'
                    project_info['tech_stack']['frameworks'] = frameworks
                    project_info['tech_stack']['languages'] = ['JavaScript', 'TypeScript']
            
            # Check for requirements.txt (Python projects)
            requirements = path / "requirements.txt"
            if requirements.exists():
                project_info['type'] = 'python'
                project_info['tech_stack']['languages'] = ['Python']
                
                with open(requirements, 'r') as f:
                    content = f.read().lower()
                    frameworks = []
                    if 'django' in content:
                        frameworks.append('Django')
                    if 'flask' in content:
                        frameworks.append('Flask')
                    if 'fastapi' in content:
                        frameworks.append('FastAPI')
                    project_info['tech_stack']['frameworks'] = frameworks
        
        except Exception as e:
            logger.warning(f"Could not extract project info: {e}")
        
        return project_info
    
    def _calculate_quality_score(self, result: Dict) -> float:
        """Calculate quality score for a result"""
        score = 0.0
        
        # Success gives base score
        if result.get("status") == "completed":
            score += 0.5
        
        # Fast execution adds to score
        exec_time = result.get("execution_time", float('inf'))
        if exec_time < 5:
            score += 0.3
        elif exec_time < 15:
            score += 0.2
        elif exec_time < 30:
            score += 0.1
        
        # Performance score contribution
        perf_score = result.get("performance_score", 0)
        score += perf_score * 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _is_pattern_reusable(self, result: Dict, context: Dict) -> bool:
        """Determine if a pattern is reusable across projects"""
        
        # Successful patterns are more likely reusable
        if result.get("status") != "completed":
            return False
        
        # Check if pattern is generic enough
        task = context.get("task", "")
        
        # Common patterns that are usually reusable
        reusable_patterns = [
            "api", "database", "authentication", "validation",
            "error handling", "logging", "testing", "deployment",
            "optimization", "security", "caching", "monitoring"
        ]
        
        task_lower = task.lower() if task else ""
        for pattern in reusable_patterns:
            if pattern in task_lower:
                return True
        
        # High-quality patterns are likely reusable
        if self._calculate_quality_score(result) > 0.8:
            return True
        
        return False


# Integration with Phase 6 Task Execution
class EnhancedPhase6Executor:
    """
    Enhanced Phase 6 executor that uses Archon's knowledge base for learning.
    """
    
    def __init__(self):
        self.knowledge = Phase6KnowledgeIntegration()
    
    async def execute_with_learning(self, 
                                   agent_role: str,
                                   task: str,
                                   context: Dict) -> Dict:
        """
        Execute agent task with knowledge base learning context.
        """
        
        # Get learning context from knowledge base
        learning_context = await self.knowledge.get_learning_context(agent_role, task)
        
        # Create enhanced prompt with learning
        enhanced_prompt = f"""
{learning_context}

Current Task:
{task}

Context:
{json.dumps(context, indent=2)}

Apply the successful patterns from the knowledge base to complete this task efficiently.
"""
        
        # Create Task tool invocation with learning
        task_invocation = {
            "tool": "Task",
            "tool_input": {
                "subagent_type": self.knowledge._get_agent_type(agent_role),
                "description": f"{agent_role} with knowledge base learning",
                "prompt": enhanced_prompt
            }
        }
        
        # Execute (in production, this would be actual Task tool call)
        result = {
            "status": "completed",
            "output": f"Executed with learning context",
            "execution_time": 5.2,
            "performance_score": 0.92
        }
        
        # Store execution in knowledge base for future learning
        await self.knowledge.store_agent_execution(agent_role, task, result, context)
        
        return result


# Demonstration
async def demonstrate_knowledge_integration():
    """Show how Phase 6 uses Archon's knowledge base for learning"""
    
    executor = EnhancedPhase6Executor()
    
    # Execute task with learning
    result = await executor.execute_with_learning(
        agent_role="python_backend_coder",
        task="Implement user authentication API with JWT",
        context={"project": "archon", "framework": "fastapi"}
    )
    
    print("Task executed with knowledge base learning:")
    print(f"Status: {result['status']}")
    print(f"Performance: {result['performance_score']}")
    
    # Analyze trends
    trends = await executor.knowledge.analyze_performance_trends("python_backend_coder")
    print(f"\nPerformance Trends for python_backend_coder:")
    print(f"Success Rate: {trends.get('success_rate', 0):.2%}")
    print(f"Recommendations: {trends.get('recommendations', [])}")


if __name__ == "__main__":
    asyncio.run(demonstrate_knowledge_integration())