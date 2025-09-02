"""
Context Enricher for Archon Phase 3
Knowledge base integration for prompt enhancement
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timedelta
import json

try:
    import httpx
    from supabase import create_client, Client
except ImportError:
    # Handle missing dependencies gracefully
    httpx = None
    create_client = None
    Client = None

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeItem:
    """Represents a piece of knowledge from the knowledge base"""
    id: str
    content: str
    source: str
    relevance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieved_at: datetime = field(default_factory=datetime.now)


@dataclass
class ContextEnrichmentResult:
    """Result of context enrichment process"""
    original_context: Dict[str, Any]
    enriched_context: Dict[str, Any]
    knowledge_items: List[KnowledgeItem] = field(default_factory=list)
    confidence_score: float = 0.0
    sources_used: List[str] = field(default_factory=list)
    processing_time_ms: int = 0


class ContextEnricher:
    """
    Context enricher that integrates with knowledge base for prompt enhancement.
    
    Features:
    - RAG-based context retrieval
    - Project-specific knowledge integration
    - Agent role context adaptation
    - Caching for performance
    - Confidence scoring
    """

    def __init__(self,
                 supabase_url: Optional[str] = None,
                 supabase_key: Optional[str] = None,
                 mcp_server_url: Optional[str] = None,
                 cache_ttl_minutes: int = 30):
        """
        Initialize context enricher.
        
        Args:
            supabase_url: Supabase URL for knowledge base
            supabase_key: Supabase service key
            mcp_server_url: MCP server URL for tool calls
            cache_ttl_minutes: Cache time-to-live in minutes
        """
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.mcp_server_url = mcp_server_url or "http://localhost:8051"
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        
        # Initialize Supabase client if available
        self.supabase: Optional[Client] = None
        if supabase_url and supabase_key and create_client:
            try:
                self.supabase = create_client(supabase_url, supabase_key)
                logger.info("Supabase client initialized for context enrichment")
            except Exception as e:
                logger.warning(f"Failed to initialize Supabase client: {e}")
        
        # Initialize HTTP client for MCP calls
        self.http_client = httpx.AsyncClient() if httpx else None
        
        # Context cache
        self._context_cache: Dict[str, ContextEnrichmentResult] = {}
        
        # Knowledge extraction patterns
        self.extraction_patterns = {
            "project_type": [
                r"\b(?:react|vue|angular|svelte)\b",
                r"\b(?:fastapi|django|flask|express)\b",
                r"\b(?:python|javascript|typescript|java|c#|go|rust)\b"
            ],
            "technologies": [
                r"\b(?:postgresql|mysql|mongodb|redis|elasticsearch)\b",
                r"\b(?:docker|kubernetes|aws|azure|gcp)\b",
                r"\b(?:jest|vitest|pytest|junit|mocha)\b"
            ],
            "requirements": [
                r"(?:implement|create|build|develop|design)\s+(.+?)(?:\.|$)",
                r"(?:need|require|want)\s+(.+?)(?:\.|$)",
                r"(?:should|must|have to)\s+(.+?)(?:\.|$)"
            ]
        }
        
        logger.info("ContextEnricher initialized")

    async def enrich_context(self,
                             prompt: str,
                             agent_role: Optional[str] = None,
                             project_type: Optional[str] = None,
                             existing_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enrich context with knowledge base information.
        
        Args:
            prompt: The prompt to enrich context for
            agent_role: Target agent role for context adaptation
            project_type: Project type for relevant knowledge
            existing_context: Existing context to build upon
            
        Returns:
            Enriched context dictionary
        """
        start_time = datetime.now()
        
        try:
            # Initialize base context
            base_context = existing_context.copy() if existing_context else {}
            
            # Check cache first
            cache_key = self._generate_cache_key(prompt, agent_role, project_type)
            if cache_key in self._context_cache:
                cached_result = self._context_cache[cache_key]
                if datetime.now() - cached_result.enriched_context.get('cached_at', datetime.min) < self.cache_ttl:
                    logger.debug("Using cached context enrichment")
                    return cached_result.enriched_context
            
            # Extract key information from prompt
            extracted_info = self._extract_information_from_prompt(prompt)
            
            # Merge with existing context
            base_context.update(extracted_info)
            base_context['agent_role'] = agent_role
            base_context['project_type'] = project_type or base_context.get('project_type')
            
            # Retrieve relevant knowledge
            knowledge_items = await self._retrieve_relevant_knowledge(prompt, base_context)
            
            # Process knowledge items
            processed_knowledge = self._process_knowledge_items(knowledge_items, agent_role)
            
            # Build enriched context
            enriched_context = self._build_enriched_context(
                base_context,
                processed_knowledge,
                knowledge_items
            )
            
            # Add metadata
            enriched_context['cached_at'] = datetime.now()
            enriched_context['enrichment_source'] = 'knowledge_base'
            
            # Calculate confidence score
            confidence = self._calculate_enrichment_confidence(knowledge_items, base_context)
            enriched_context['enrichment_confidence'] = confidence
            
            # Cache result
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            result = ContextEnrichmentResult(
                original_context=existing_context or {},
                enriched_context=enriched_context,
                knowledge_items=knowledge_items,
                confidence_score=confidence,
                sources_used=[item.source for item in knowledge_items],
                processing_time_ms=processing_time
            )
            
            if cache_key:
                self._context_cache[cache_key] = result
            
            logger.info(f"Context enriched with {len(knowledge_items)} knowledge items (confidence: {confidence:.2f})")
            return enriched_context
            
        except Exception as e:
            logger.error(f"Context enrichment failed: {e}")
            # Return base context on failure
            return existing_context or {}

    def _extract_information_from_prompt(self, prompt: str) -> Dict[str, Any]:
        """Extract structured information from prompt using patterns."""
        extracted = {
            'technologies': [],
            'requirements': [],
            'constraints': [],
            'domain_keywords': []
        }
        
        prompt_lower = prompt.lower()
        
        # Extract technologies
        for pattern in self.extraction_patterns['technologies']:
            matches = re.findall(pattern, prompt_lower, re.IGNORECASE)
            extracted['technologies'].extend(matches)
        
        # Extract requirements
        for pattern in self.extraction_patterns['requirements']:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            extracted['requirements'].extend([match.strip() for match in matches])
        
        # Extract domain keywords
        # Simple keyword extraction based on common technical terms
        tech_keywords = re.findall(r'\b(?:api|database|frontend|backend|authentication|authorization|testing|deployment|security|performance|scalability|microservice|container|cloud|serverless)\b', prompt_lower)
        extracted['domain_keywords'] = list(set(tech_keywords))
        
        # Clean up lists
        for key in extracted:
            if isinstance(extracted[key], list):
                extracted[key] = list(set(extracted[key]))  # Remove duplicates
                extracted[key] = [item for item in extracted[key] if item.strip()]  # Remove empty items
        
        return extracted

    async def _retrieve_relevant_knowledge(self, 
                                          prompt: str, 
                                          context: Dict[str, Any]) -> List[KnowledgeItem]:
        """Retrieve relevant knowledge from knowledge base."""
        knowledge_items = []
        
        try:
            # Method 1: Try MCP server for knowledge retrieval
            if self.http_client:
                mcp_knowledge = await self._retrieve_via_mcp(prompt, context)
                knowledge_items.extend(mcp_knowledge)
            
            # Method 2: Direct Supabase query if available
            if self.supabase and len(knowledge_items) < 5:
                supabase_knowledge = await self._retrieve_via_supabase(prompt, context)
                knowledge_items.extend(supabase_knowledge)
            
            # Method 3: Fallback to built-in knowledge patterns
            if len(knowledge_items) < 3:
                builtin_knowledge = self._generate_builtin_knowledge(prompt, context)
                knowledge_items.extend(builtin_knowledge)
                
        except Exception as e:
            logger.error(f"Knowledge retrieval failed: {e}")
            # Always provide fallback knowledge
            knowledge_items = self._generate_builtin_knowledge(prompt, context)
        
        # Sort by relevance score and limit results
        knowledge_items.sort(key=lambda x: x.relevance_score, reverse=True)
        return knowledge_items[:10]  # Limit to top 10 items

    async def _retrieve_via_mcp(self, prompt: str, context: Dict[str, Any]) -> List[KnowledgeItem]:
        """Retrieve knowledge via MCP server."""
        knowledge_items = []
        
        if not self.http_client:
            return knowledge_items
        
        try:
            # Prepare search query
            search_terms = []
            search_terms.extend(context.get('domain_keywords', []))
            search_terms.extend(context.get('technologies', []))
            
            if context.get('agent_role'):
                search_terms.append(context['agent_role'])
            
            search_query = ' '.join(search_terms[:5])  # Limit search terms
            
            if not search_query.strip():
                search_query = prompt[:100]  # Use first 100 chars of prompt
            
            # Call MCP RAG query tool
            mcp_request = {
                "query": search_query,
                "match_count": 5
            }
            
            response = await self.http_client.post(
                f"{self.mcp_server_url}/tools/archon:perform_rag_query",
                json=mcp_request,
                timeout=10.0
            )
            
            if response.status_code == 200:
                result = response.json()
                documents = result.get('documents', [])
                
                for doc in documents:
                    knowledge_items.append(KnowledgeItem(
                        id=doc.get('id', 'mcp_unknown'),
                        content=doc.get('content', ''),
                        source=doc.get('source', 'MCP Knowledge Base'),
                        relevance_score=doc.get('similarity_score', 0.5),
                        metadata=doc.get('metadata', {})
                    ))
                    
                logger.debug(f"Retrieved {len(knowledge_items)} items via MCP")
                
        except Exception as e:
            logger.debug(f"MCP knowledge retrieval failed: {e}")
        
        return knowledge_items

    async def _retrieve_via_supabase(self, prompt: str, context: Dict[str, Any]) -> List[KnowledgeItem]:
        """Retrieve knowledge directly from Supabase."""
        knowledge_items = []
        
        if not self.supabase:
            return knowledge_items
        
        try:
            # Simple text search in documents table
            search_terms = ' '.join(context.get('domain_keywords', [])[:3])
            if not search_terms:
                search_terms = prompt[:50]
            
            # Query documents table
            response = self.supabase.table('documents').select('*').text_search('content', search_terms).limit(5).execute()
            
            for doc in response.data:
                knowledge_items.append(KnowledgeItem(
                    id=doc.get('id', 'supabase_unknown'),
                    content=doc.get('content', ''),
                    source=doc.get('source_url', 'Supabase Knowledge Base'),
                    relevance_score=0.7,  # Default relevance for direct queries
                    metadata={
                        'chunk_index': doc.get('chunk_index'),
                        'source_id': doc.get('source_id')
                    }
                ))
                
            logger.debug(f"Retrieved {len(knowledge_items)} items via Supabase")
            
        except Exception as e:
            logger.debug(f"Supabase knowledge retrieval failed: {e}")
        
        return knowledge_items

    def _generate_builtin_knowledge(self, prompt: str, context: Dict[str, Any]) -> List[KnowledgeItem]:
        """Generate built-in knowledge based on prompt analysis."""
        knowledge_items = []
        
        # Built-in knowledge patterns based on common scenarios
        builtin_knowledge = {
            "react": {
                "content": "React best practices: Use functional components with hooks, implement proper error boundaries, follow component composition patterns, use TypeScript for type safety, implement proper state management with useState/useReducer or external libraries like Redux/Zustand.",
                "relevance": 0.8
            },
            "fastapi": {
                "content": "FastAPI development: Use Pydantic models for request/response validation, implement proper dependency injection, use async/await for I/O operations, add comprehensive error handling, implement proper authentication/authorization, use SQLAlchemy for database operations.",
                "relevance": 0.8
            },
            "testing": {
                "content": "Testing best practices: Write unit tests for all business logic, use integration tests for API endpoints, mock external dependencies, achieve >90% code coverage, test error scenarios and edge cases, use descriptive test names that explain the scenario being tested.",
                "relevance": 0.7
            },
            "security": {
                "content": "Security considerations: Implement proper input validation, use parameterized queries to prevent SQL injection, implement authentication and authorization, use HTTPS for all communications, sanitize user inputs, implement rate limiting, log security events.",
                "relevance": 0.9
            },
            "performance": {
                "content": "Performance optimization: Implement proper caching strategies, optimize database queries with indexes, use pagination for large datasets, implement lazy loading, minimize bundle sizes, profile and optimize hot paths, use CDN for static assets.",
                "relevance": 0.7
            },
            "database": {
                "content": "Database best practices: Use proper indexing strategies, implement connection pooling, use transactions for data consistency, normalize data appropriately, implement proper backup and recovery procedures, monitor query performance.",
                "relevance": 0.8
            }
        }
        
        # Match built-in knowledge to prompt content
        prompt_lower = prompt.lower()
        for keyword, knowledge in builtin_knowledge.items():
            if keyword in prompt_lower or keyword in ' '.join(context.get('technologies', [])):
                knowledge_items.append(KnowledgeItem(
                    id=f"builtin_{keyword}",
                    content=knowledge['content'],
                    source="Archon Built-in Knowledge",
                    relevance_score=knowledge['relevance'],
                    metadata={"type": "builtin", "keyword": keyword}
                ))
        
        # Add general development knowledge if no specific matches
        if not knowledge_items:
            knowledge_items.append(KnowledgeItem(
                id="builtin_general",
                content="General development principles: Follow SOLID principles, implement proper error handling, write clean and maintainable code, use version control effectively, document important decisions, implement comprehensive testing, follow security best practices.",
                source="Archon Built-in Knowledge",
                relevance_score=0.5,
                metadata={"type": "builtin", "keyword": "general"}
            ))
        
        return knowledge_items

    def _process_knowledge_items(self, 
                                knowledge_items: List[KnowledgeItem],
                                agent_role: Optional[str]) -> Dict[str, Any]:
        """Process knowledge items for context integration."""
        processed = {
            "relevant_patterns": [],
            "best_practices": [],
            "examples": [],
            "constraints": [],
            "technologies": []
        }
        
        for item in knowledge_items:
            content = item.content.lower()
            
            # Categorize content based on keywords
            if any(word in content for word in ['pattern', 'approach', 'method']):
                processed["relevant_patterns"].append(item.content[:200])
            
            if any(word in content for word in ['best practice', 'should', 'recommend']):
                processed["best_practices"].append(item.content[:200])
            
            if any(word in content for word in ['example', 'implementation', 'code']):
                processed["examples"].append(item.content[:300])
            
            if any(word in content for word in ['constraint', 'limitation', 'requirement']):
                processed["constraints"].append(item.content[:150])
            
            # Extract technology mentions
            tech_mentions = re.findall(r'\b(?:react|vue|angular|python|javascript|typescript|fastapi|django|postgresql|mongodb|redis|docker|kubernetes|aws|azure)\b', content)
            processed["technologies"].extend(tech_mentions)
        
        # Clean up and deduplicate
        for key in processed:
            if isinstance(processed[key], list):
                processed[key] = list(set(processed[key]))  # Remove duplicates
                processed[key] = [item for item in processed[key] if item.strip()]  # Remove empty
        
        return processed

    def _build_enriched_context(self,
                               base_context: Dict[str, Any],
                               processed_knowledge: Dict[str, Any],
                               knowledge_items: List[KnowledgeItem]) -> Dict[str, Any]:
        """Build the final enriched context."""
        enriched = base_context.copy()
        
        # Add processed knowledge
        enriched.update(processed_knowledge)
        
        # Add context injections list for tracking
        enriched["injected_context"] = []
        
        # Add specific context based on knowledge
        if processed_knowledge["best_practices"]:
            enriched["injected_context"].append("Best practices from knowledge base")
            enriched["project_best_practices"] = processed_knowledge["best_practices"][:3]
        
        if processed_knowledge["relevant_patterns"]:
            enriched["injected_context"].append("Relevant patterns and approaches")
            enriched["implementation_patterns"] = processed_knowledge["relevant_patterns"][:2]
        
        if processed_knowledge["examples"]:
            enriched["injected_context"].append("Code examples and implementations")
            enriched["reference_examples"] = processed_knowledge["examples"][:2]
        
        # Add source tracking
        enriched["sources"] = list(set(item.source for item in knowledge_items))
        
        # Add knowledge metadata
        enriched["knowledge_items_count"] = len(knowledge_items)
        enriched["avg_relevance_score"] = sum(item.relevance_score for item in knowledge_items) / len(knowledge_items) if knowledge_items else 0.0
        
        return enriched

    def _calculate_enrichment_confidence(self,
                                       knowledge_items: List[KnowledgeItem],
                                       context: Dict[str, Any]) -> float:
        """Calculate confidence score for context enrichment."""
        if not knowledge_items:
            return 0.0
        
        # Base confidence on average relevance score
        avg_relevance = sum(item.relevance_score for item in knowledge_items) / len(knowledge_items)
        
        # Boost confidence based on number of sources
        source_diversity = len(set(item.source for item in knowledge_items))
        diversity_boost = min(source_diversity * 0.1, 0.3)
        
        # Boost confidence based on context richness
        context_richness = len([v for v in context.values() if v and v != []])
        richness_boost = min(context_richness * 0.05, 0.2)
        
        final_confidence = min(avg_relevance + diversity_boost + richness_boost, 1.0)
        return final_confidence

    def _generate_cache_key(self,
                           prompt: str,
                           agent_role: Optional[str],
                           project_type: Optional[str]) -> str:
        """Generate cache key for context enrichment."""
        key_parts = [
            prompt[:100],  # First 100 chars
            str(agent_role),
            str(project_type)
        ]
        return "|".join(key_parts)

    async def get_project_context(self, project_id: str) -> Dict[str, Any]:
        """Get specific project context from knowledge base."""
        context = {
            "project_id": project_id,
            "technologies": [],
            "architecture": "unknown",
            "constraints": [],
            "recent_activity": []
        }
        
        try:
            if self.http_client:
                # Try to get project info via MCP
                response = await self.http_client.post(
                    f"{self.mcp_server_url}/tools/archon:manage_project",
                    json={"action": "get", "project_id": project_id},
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    project_data = response.json()
                    context.update({
                        "name": project_data.get("name", "Unknown"),
                        "description": project_data.get("description", ""),
                        "technologies": project_data.get("technologies", []),
                        "status": project_data.get("status", "unknown")
                    })
                    
        except Exception as e:
            logger.debug(f"Failed to retrieve project context: {e}")
        
        return context

    def clear_cache(self):
        """Clear the context cache."""
        self._context_cache.clear()
        logger.info("Context enrichment cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._context_cache),
            "cache_ttl_minutes": self.cache_ttl.total_seconds() / 60,
            "oldest_entry": min(
                (result.enriched_context.get('cached_at', datetime.now()) 
                 for result in self._context_cache.values()),
                default=datetime.now()
            ).isoformat() if self._context_cache else None
        }