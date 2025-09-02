#!/usr/bin/env python3
"""
REF Tools MCP Client for Archon+ Phase 3
Integrates with REF Tools Model Context Protocol for enhanced documentation and context
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import httpx
import os
from urllib.parse import urlparse, urljoin

logger = logging.getLogger(__name__)

class REFToolsError(Exception):
    """Custom exception for REF Tools operations"""
    pass

class ContentType(Enum):
    DOCUMENTATION = "documentation"
    API_REFERENCE = "api_reference"
    TUTORIAL = "tutorial"
    EXAMPLE_CODE = "example_code"
    BEST_PRACTICES = "best_practices"

@dataclass
class ContentSection:
    """Individual content section from REF Tools"""
    title: str
    content: str
    section_type: ContentType
    confidence: float  # 0.0-1.0 relevance confidence
    source_url: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentResult:
    """Result from document search"""
    url: str
    title: str
    summary: str
    sections: List[ContentSection] = field(default_factory=list)
    relevance_score: float = 0.0
    last_updated: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContentResult:
    """Result from URL content extraction"""
    url: str
    title: str
    content: str
    sections: List[ContentSection] = field(default_factory=list)
    extraction_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContextPack:
    """Enhanced context pack for prompt enhancement"""
    pack_id: str
    topic: str
    documents: List[DocumentResult] = field(default_factory=list)
    extracted_content: List[ContentResult] = field(default_factory=list)
    total_relevance: float = 0.0
    generation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class REFToolsClient:
    """MCP client for REF Tools integration"""
    
    def __init__(self, 
                 mcp_url: str = "http://localhost:8051",
                 timeout: float = 30.0,
                 max_retries: int = 3):
        self.mcp_url = mcp_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.session_id = str(uuid.uuid4())
        self._client = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._client:
            await self._client.aclose()
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to MCP server with retries"""
        if not self._client:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        
        url = f"{self.mcp_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.max_retries):
            try:
                response = await self._client.request(method, url, **kwargs)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    raise REFToolsError(f"REF Tools endpoint not found: {url}")
                elif response.status_code >= 500:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    raise REFToolsError(f"REF Tools server error: {response.status_code}")
                else:
                    raise REFToolsError(f"REF Tools request failed: {response.status_code} - {response.text}")
                    
            except httpx.RequestError as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise REFToolsError(f"REF Tools connection failed: {str(e)}")
        
        raise REFToolsError(f"REF Tools request failed after {self.max_retries} attempts")
    
    async def _create_mcp_session(self) -> str:
        """Create a new MCP session and return session ID"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        # Initialize MCP session
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "archon-ref-tools-client",
                    "version": "1.0.0"
                }
            },
            "id": f"init_{int(time.time())}"
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.mcp_url}/mcp", 
                                       json=init_request, 
                                       headers=headers)
            
            if response.status_code != 200:
                raise REFToolsError(f"Failed to initialize MCP session: {response.status_code}")
            
            # Extract session ID from response headers (FastMCP uses 'mcp-session-id')
            session_id = response.headers.get("mcp-session-id", str(uuid.uuid4()))
            
            # Send initialized notification
            initialized_request = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }
            
            headers["mcp-session-id"] = session_id
            await client.post(f"{self.mcp_url}/mcp", 
                            json=initialized_request, 
                            headers=headers)
            
            return session_id

    async def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP tool using proper JSON-RPC protocol"""
        # Create session first
        session_id = await self._create_mcp_session()
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "mcp-session-id": session_id
        }
        
        # Make tool call request
        tool_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": f"{tool_name}_{int(time.time())}"
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.mcp_url}/mcp", 
                                       json=tool_request, 
                                       headers=headers)
            
            if response.status_code == 200:
                # Handle Server-Sent Events response format from FastMCP
                response_text = response.text
                
                # Parse SSE format (event: message\ndata: {...})
                if response_text.startswith("event:"):
                    lines = response_text.strip().split('\n')
                    for line in lines:
                        if line.startswith('data: '):
                            json_data = line[6:]  # Remove 'data: ' prefix
                            result = json.loads(json_data)
                            
                            if "result" in result:
                                # Handle both string and dict results from MCP tool
                                tool_result = result["result"]
                                if isinstance(tool_result, str):
                                    return json.loads(tool_result)
                                else:
                                    return tool_result
                            elif "error" in result:
                                raise REFToolsError(f"MCP tool error: {result['error']}")
                            break
                else:
                    # Try regular JSON parsing as fallback
                    try:
                        result = response.json()
                        if "result" in result:
                            # Handle both string and dict results from MCP tool
                            tool_result = result["result"]
                            if isinstance(tool_result, str):
                                return json.loads(tool_result)
                            else:
                                return tool_result
                        elif "error" in result:
                            raise REFToolsError(f"MCP tool error: {result['error']}")
                    except json.JSONDecodeError:
                        pass
            
            raise REFToolsError(f"MCP tool call failed: {response.status_code} - {response.text}")
    
    async def health_check(self) -> bool:
        """Check if REF Tools MCP server is available"""
        try:
            # Use proper MCP JSON-RPC protocol to call ref_tools_health
            tool_result = await self._call_mcp_tool("ref_tools_health", {})
            
            # Handle different response formats
            if isinstance(tool_result, dict):
                # Check for structured content first
                if "structuredContent" in tool_result:
                    import json
                    health_data = json.loads(tool_result["structuredContent"]["result"])
                    return health_data.get("available", False)
                # Check for direct result
                elif "available" in tool_result:
                    return tool_result.get("available", False)
                # Check for content field
                elif "content" in tool_result:
                    content = tool_result["content"]
                    if isinstance(content, str) and "available" in content.lower():
                        return "true" in content.lower()
                    elif isinstance(content, dict):
                        return content.get("available", False)
            elif isinstance(tool_result, str):
                # Parse string response
                return "available" in tool_result.lower() and "true" in tool_result.lower()
            
            # If we get any response, consider it healthy
            return True
            
        except Exception as e:
            logger.warning(f"REF Tools health check failed: {e}")
            return False
    
    async def search_documentation(self, 
                                 query: str,
                                 sources: Optional[List[str]] = None,
                                 limit: int = 10,
                                 content_types: Optional[List[ContentType]] = None) -> List[DocumentResult]:
        """Search documentation using REF Tools"""
        start_time = time.time()
        
        try:
            search_params = {
                "query": query,
                "limit": limit,
                "session_id": self.session_id
            }
            
            if sources:
                search_params["sources"] = ",".join(sources)
                
            if content_types:
                search_params["content_types"] = [ct.value for ct in content_types]
            
            # Use MCP tool instead of HTTP endpoint
            tool_result = await self._call_mcp_tool("search_documentation", search_params)
            
            # Parse structured response - handle multiple formats
            results = []
            search_data = None
            
            # Try different response formats
            if isinstance(tool_result, dict):
                if "structuredContent" in tool_result:
                    import json
                    search_data = json.loads(tool_result["structuredContent"]["result"])
                elif "content" in tool_result:
                    content = tool_result["content"]
                    if isinstance(content, str):
                        try:
                            import json
                            search_data = json.loads(content)
                        except:
                            # If content is not JSON, treat as text
                            search_data = {"success": True, "results": [{"title": "Documentation", "content": content, "url": "", "relevance": 0.8}]}
                    elif isinstance(content, dict):
                        search_data = content
                else:
                    # Direct result format
                    search_data = tool_result
            elif isinstance(tool_result, str):
                try:
                    import json
                    search_data = json.loads(tool_result)
                except:
                    # Treat as single text result
                    search_data = {"success": True, "results": [{"title": "Documentation", "content": tool_result, "url": "", "relevance": 0.8}]}
            
            if search_data and search_data.get("success", True):
                for item in search_data.get("results", []):
                    result = DocumentResult(
                        url=item.get("url", ""),
                        title=item.get("title", "Documentation"),
                        summary=item.get("content", item.get("summary", "")),
                        sections=[],  # Simplified - no sections parsing for now
                        relevance_score=item.get("relevance", item.get("score", 0.8)),
                        last_updated=None,
                        metadata={}
                    )
                    results.append(result)
            
            logger.info(f"Found {len(results)} documentation results for query: {query[:50]}...")
            return results
            
        except REFToolsError as e:
            logger.error(f"Documentation search failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in documentation search: {e}")
            return []
    
    async def read_url_content(self, 
                             url: str,
                             extract_sections: Optional[List[str]] = None,
                             content_type: Optional[ContentType] = None) -> ContentResult:
        """Read and extract specific content sections from URL"""
        start_time = time.time()
        
        try:
            extract_params = {
                "url": url,
                "session_id": self.session_id
            }
            
            if extract_sections:
                extract_params["extract_sections"] = extract_sections
                
            if content_type:
                extract_params["content_type"] = content_type.value
            
            response = await self._make_request(
                "POST",
                "/api/content/extract", 
                json=extract_params
            )
            
            sections = []
            for section_data in response.get("sections", []):
                section = ContentSection(
                    title=section_data.get("title", ""),
                    content=section_data.get("content", ""),
                    section_type=ContentType(section_data.get("type", "documentation")),
                    confidence=section_data.get("confidence", 0.5),
                    source_url=url,
                    metadata=section_data.get("metadata", {})
                )
                sections.append(section)
            
            result = ContentResult(
                url=url,
                title=response.get("title", ""),
                content=response.get("content", ""),
                sections=sections,
                extraction_time=time.time() - start_time,
                success=response.get("success", True),
                error_message=response.get("error"),
                metadata=response.get("metadata", {})
            )
            
            logger.info(f"Extracted content from {url} - {len(sections)} sections")
            return result
            
        except REFToolsError as e:
            logger.error(f"Content extraction failed for {url}: {e}")
            return ContentResult(
                url=url,
                title="",
                content="",
                extraction_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error in content extraction: {e}")
            return ContentResult(
                url=url,
                title="",
                content="",
                extraction_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    async def create_context_pack(self, 
                                topic: str,
                                query_terms: List[str],
                                sources: Optional[List[str]] = None,
                                max_documents: int = 5,
                                extract_content: bool = True) -> ContextPack:
        """Create comprehensive context pack for a topic"""
        start_time = time.time()
        pack_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Creating context pack for topic: {topic}")
            
            # Search for relevant documents
            documents = []
            extracted_content = []
            
            for query in query_terms:
                search_results = await self.search_documentation(
                    query=query,
                    sources=sources,
                    limit=max_documents // len(query_terms) + 1
                )
                documents.extend(search_results[:max_documents // len(query_terms)])
            
            # Remove duplicates based on URL
            unique_documents = {}
            for doc in documents:
                if doc.url not in unique_documents:
                    unique_documents[doc.url] = doc
                elif doc.relevance_score > unique_documents[doc.url].relevance_score:
                    unique_documents[doc.url] = doc
            
            documents = list(unique_documents.values())[:max_documents]
            
            # Extract content from top documents if requested
            if extract_content:
                for doc in documents[:3]:  # Extract from top 3 documents
                    if doc.url:
                        content_result = await self.read_url_content(doc.url)
                        if content_result.success:
                            extracted_content.append(content_result)
            
            # Calculate total relevance
            total_relevance = sum(doc.relevance_score for doc in documents) / len(documents) if documents else 0.0
            
            context_pack = ContextPack(
                pack_id=pack_id,
                topic=topic,
                documents=documents,
                extracted_content=extracted_content,
                total_relevance=total_relevance,
                generation_time=time.time() - start_time,
                metadata={
                    "query_terms": query_terms,
                    "sources": sources,
                    "document_count": len(documents),
                    "extracted_count": len(extracted_content),
                    "session_id": self.session_id
                }
            )
            
            logger.info(f"Created context pack {pack_id} - {len(documents)} docs, {len(extracted_content)} extracted")
            return context_pack
            
        except Exception as e:
            logger.error(f"Context pack creation failed: {e}")
            return ContextPack(
                pack_id=pack_id,
                topic=topic,
                generation_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    async def enhance_knowledge_search(self,
                                     keywords: List[str],
                                     project_context: Optional[str] = None,
                                     agent_role: Optional[str] = None) -> List[str]:
        """Enhanced knowledge search for prompt enhancement"""
        try:
            # Build context-aware query
            query_parts = keywords[:]
            
            if project_context:
                query_parts.append(f"project:{project_context}")
                
            if agent_role:
                query_parts.append(f"role:{agent_role}")
            
            query = " ".join(query_parts)
            
            # Search documentation
            doc_results = await self.search_documentation(
                query=query,
                limit=5,
                content_types=[ContentType.BEST_PRACTICES, ContentType.DOCUMENTATION]
            )
            
            # Extract key insights
            insights = []
            for doc in doc_results:
                if doc.summary:
                    insights.append(doc.summary)
                
                for section in doc.sections[:2]:  # Top 2 sections per document
                    if section.confidence > 0.6:  # Only high-confidence content
                        insights.append(f"{section.title}: {section.content[:200]}...")
            
            return insights[:10]  # Return top 10 insights
            
        except Exception as e:
            logger.error(f"Enhanced knowledge search failed: {e}")
            return []
    
    async def get_validation_evidence(self,
                                    validation_point: str,
                                    code_context: Optional[str] = None) -> List[str]:
        """Get validation evidence from documentation"""
        try:
            query_terms = [validation_point]
            
            if code_context:
                query_terms.append(code_context)
            
            # Search for validation-specific content
            doc_results = await self.search_documentation(
                query=" ".join(query_terms),
                limit=3,
                content_types=[ContentType.BEST_PRACTICES, ContentType.DOCUMENTATION]
            )
            
            evidence = []
            for doc in doc_results:
                for section in doc.sections:
                    if any(keyword in section.content.lower() 
                          for keyword in ["valid", "correct", "best", "recommend", "should"]):
                        evidence.append(f"From {doc.title}: {section.content[:150]}...")
                        break
            
            return evidence
            
        except Exception as e:
            logger.error(f"Validation evidence search failed: {e}")
            return []
    
    def export_context_pack(self, context_pack: ContextPack) -> str:
        """Export context pack as JSON"""
        return json.dumps({
            "pack_id": context_pack.pack_id,
            "topic": context_pack.topic,
            "total_relevance": context_pack.total_relevance,
            "generation_time": context_pack.generation_time,
            "documents": [
                {
                    "url": doc.url,
                    "title": doc.title,
                    "summary": doc.summary,
                    "relevance_score": doc.relevance_score,
                    "sections": [
                        {
                            "title": section.title,
                            "content": section.content[:300] + "..." if len(section.content) > 300 else section.content,
                            "type": section.section_type.value,
                            "confidence": section.confidence
                        }
                        for section in doc.sections
                    ]
                }
                for doc in context_pack.documents
            ],
            "extracted_content": [
                {
                    "url": content.url,
                    "title": content.title,
                    "success": content.success,
                    "sections_count": len(content.sections)
                }
                for content in context_pack.extracted_content
            ],
            "metadata": context_pack.metadata
        }, indent=2)

# Utility functions for integration with prompt enhancer
async def get_enhanced_context_for_prompt(query: str, 
                                        agent_role: str = None,
                                        project_type: str = None) -> List[str]:
    """Get enhanced context for prompt enhancement (standalone function)"""
    try:
        async with REFToolsClient() as client:
            # Check if REF Tools is available
            if not await client.health_check():
                logger.warning("REF Tools not available, returning empty context")
                return []
            
            # Use direct search for better result count
            search_results = await client.search_documentation(
                query=query,
                limit=10  # Get more results
            )
            
            # Convert search results to context strings
            context_strings = []
            for result in search_results[:8]:  # Take top 8 results
                if hasattr(result, 'title') and hasattr(result, 'summary'):
                    # DocumentResult object
                    title = result.title or "Documentation"
                    content = result.summary or ""
                    if content:
                        # Clean and truncate content for context
                        clean_content = content.replace('\n', ' ').strip()[:200]
                        context_strings.append(f"From {title}: {clean_content}...")
                elif isinstance(result, dict):
                    title = result.get("title", "Documentation")
                    content = result.get("content", result.get("summary", ""))
                    if content:
                        # Clean and truncate content for context
                        clean_content = content.replace('\n', ' ').strip()[:200]
                        context_strings.append(f"From {title}: {clean_content}...")
                elif isinstance(result, str):
                    # If result is a string, use it directly (truncated)
                    context_strings.append(result[:250] + "..." if len(result) > 250 else result)
            
            return context_strings
            
    except Exception as e:
        logger.error(f"Enhanced context retrieval failed: {e}")
        return []

# Example usage and testing
if __name__ == "__main__":
    async def test_ref_tools():
        async with REFToolsClient() as client:
            # Health check
            is_healthy = await client.health_check()
            print(f"REF Tools Health: {'OK' if is_healthy else 'FAILED'}")
            
            if not is_healthy:
                print("REF Tools not available - skipping tests")
                return
            
            # Test documentation search
            print("\n=== DOCUMENTATION SEARCH ===")
            results = await client.search_documentation(
                query="React TypeScript best practices",
                limit=3
            )
            
            for i, result in enumerate(results):
                print(f"\n{i+1}. {result.title}")
                print(f"   URL: {result.url}")
                print(f"   Score: {result.relevance_score:.2f}")
                print(f"   Summary: {result.summary[:100]}...")
                print(f"   Sections: {len(result.sections)}")
            
            # Test content extraction
            if results:
                print("\n=== CONTENT EXTRACTION ===")
                content_result = await client.read_url_content(results[0].url)
                print(f"Success: {content_result.success}")
                print(f"Title: {content_result.title}")
                print(f"Content Length: {len(content_result.content)}")
                print(f"Sections: {len(content_result.sections)}")
                print(f"Extraction Time: {content_result.extraction_time:.3f}s")
            
            # Test context pack creation
            print("\n=== CONTEXT PACK CREATION ===")
            context_pack = await client.create_context_pack(
                topic="React form validation",
                query_terms=["React forms", "input validation", "TypeScript forms"],
                max_documents=2,
                extract_content=True
            )
            
            print(f"Pack ID: {context_pack.pack_id}")
            print(f"Topic: {context_pack.topic}")
            print(f"Documents: {len(context_pack.documents)}")
            print(f"Extracted: {len(context_pack.extracted_content)}")
            print(f"Total Relevance: {context_pack.total_relevance:.2f}")
            print(f"Generation Time: {context_pack.generation_time:.3f}s")
            
            # Test enhanced knowledge search
            print("\n=== ENHANCED KNOWLEDGE SEARCH ===")
            insights = await client.enhance_knowledge_search(
                keywords=["error handling", "TypeScript"],
                project_context="react_app",
                agent_role="code_implementer"
            )
            
            for i, insight in enumerate(insights[:3]):
                print(f"{i+1}. {insight[:150]}...")
    
    asyncio.run(test_ref_tools())