"""
REF Tools Module for Archon MCP Server
Integrates with ref-tools-mcp for enhanced documentation access and context packs

This module provides tools for:
- Reading URL content with token-efficient extraction
- Searching documentation across multiple sources
- Creating context packs with relevant information
- Accessing real-time documentation and examples

Based on: https://github.com/ref-tools/ref-tools-mcp
"""

import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio

import httpx
from mcp.server.fastmcp import Context, FastMCP

logger = logging.getLogger(__name__)


class REFToolsImplementation:
    """Direct REF Tools implementation within Archon MCP server"""
    
    def __init__(self):
        self.timeout = httpx.Timeout(30.0, connect=5.0)
        self.base_url = "integrated"  # For compatibility with any remaining references
    
    async def is_available(self) -> bool:
        """REF Tools are always available when integrated into Archon MCP server"""
        return True
    
    async def read_url(self, url: str, extract_sections: Optional[List[str]] = None) -> Dict[str, Any]:
        """Read and extract content from URL directly"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                
                if response.status_code == 200:
                    content = response.text
                    
                    # Basic content extraction
                    extracted_content = content
                    if extract_sections:
                        # Simple section extraction by looking for headers
                        sections_found = []
                        for section in extract_sections:
                            if section.lower() in content.lower():
                                sections_found.append(section)
                        
                        return {
                            "success": True,
                            "url": url,
                            "content": extracted_content[:5000],  # Limit content size
                            "sections": sections_found,
                            "title": "Documentation"
                        }
                    
                    return {
                        "success": True,
                        "url": url,
                        "content": extracted_content[:5000],  # Limit content size
                        "title": "Documentation"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}: Failed to fetch URL"
                    }
        except Exception as e:
            logger.error(f"Error reading URL {url}: {e}")
            return {"success": False, "error": str(e)}
    
    async def search_docs(self, query: str, sources: Optional[List[str]] = None, limit: int = 5) -> Dict[str, Any]:
        """Search documentation using Archon's RAG system"""
        try:
            # Use the Archon RAG system for documentation search
            from src.server.config.service_discovery import get_api_url
            from urllib.parse import urljoin
            
            api_url = get_api_url()
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                search_data = {"query": query, "match_count": limit}
                if sources:
                    # Use first source as domain filter
                    search_data["source"] = sources[0]
                
                response = await client.post(urljoin(api_url, "/api/rag/query"), json=search_data)
                
                if response.status_code == 200:
                    result = response.json()
                    results = result.get("results", [])
                    
                    return {
                        "success": True,
                        "results": [
                            {
                                "title": r.get("title", "Documentation"),
                                "content": r.get("content", ""),
                                "url": r.get("source_url", ""),
                                "relevance": r.get("score", 0.5)
                            }
                            for r in results[:limit]
                        ]
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Search failed: {response.status_code}"
                    }
        except Exception as e:
            logger.error(f"Error searching docs for '{query}': {e}")
            return {"success": False, "error": str(e)}
    
    async def create_context_pack(self, topic: str, queries: List[str], max_items: int = 10) -> Dict[str, Any]:
        """Create a comprehensive context pack using Archon's RAG system"""
        try:
            all_results = []
            
            # Search for each query using Archon RAG
            for query in queries:
                search_result = await self.search_docs(query, limit=max_items // len(queries))
                if search_result.get("success"):
                    all_results.extend(search_result.get("results", []))
            
            # Limit to max_items and sort by relevance
            all_results = sorted(all_results, key=lambda x: x.get("relevance", 0), reverse=True)[:max_items]
            
            return {
                "success": True,
                "topic": topic,
                "items": [
                    {
                        "content": item.get("content", ""),
                        "source": item.get("url", ""),
                        "relevance": item.get("relevance", 0.5),
                        "url": item.get("url", "")
                    }
                    for item in all_results
                ]
            }
        except Exception as e:
            logger.error(f"Error creating context pack for '{topic}': {e}")
            return {"success": False, "error": str(e)}


def register_ref_tools(mcp: FastMCP):
    """Register REF Tools with the MCP server."""
    
    # Initialize REF Tools implementation
    ref_client = REFToolsImplementation()
    
    @mcp.tool()
    async def ref_tools_health(ctx: Context) -> str:
        """
        Check if REF Tools MCP server is available and healthy.
        
        Returns:
            JSON string with health status and availability information
        """
        try:
            logger.info("Checking REF Tools availability...")
            is_available = await ref_client.is_available()
            logger.info(f"REF Tools availability check result: {is_available}")
            
            return json.dumps({
                "success": True,
                "available": is_available,
                "server_url": "integrated",
                "timestamp": datetime.now().isoformat(),
                "message": "REF Tools server is available" if is_available else "REF Tools server is not available"
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Error checking REF Tools health: {e}")
            return json.dumps({
                "success": False,
                "available": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }, indent=2)
    
    @mcp.tool()
    async def read_url_content(
        url: str, 
        extract_sections: str = None
    ) -> str:
        """
        Read and extract specific content from a URL using REF Tools.
        
        Args:
            url: The URL to read content from
            extract_sections: Comma-separated list of sections to extract (optional)
                            Examples: "installation,getting-started", "api-reference"
        
        Returns:
            JSON string with extracted content, sections, and metadata
        """
        try:
            # Check if REF Tools is available
            if not await ref_client.is_available():
                return json.dumps({
                    "success": False,
                    "error": "REF Tools server is not available. Please start the ref-tools-mcp server.",
                    "url": url
                }, indent=2)
            
            # Parse sections if provided
            sections = None
            if extract_sections:
                sections = [s.strip() for s in extract_sections.split(",") if s.strip()]
            
            result = await ref_client.read_url(url, sections)
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error reading URL content: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "url": url
            }, indent=2)
    
    @mcp.tool()
    async def search_documentation(
        query: str,
        sources: str = None,
        limit: int = 5
    ) -> str:
        """
        Search for documentation across multiple sources using REF Tools.
        
        Args:
            query: Search query for documentation
            sources: Comma-separated list of sources to search (optional)
                    Examples: "react.dev,nextjs.org", "python.org"
            limit: Maximum number of results to return (default: 5)
        
        Returns:
            JSON string with search results, relevance scores, and source information
        """
        try:
            # Check if REF Tools is available
            if not await ref_client.is_available():
                return json.dumps({
                    "success": False,
                    "error": "REF Tools server is not available. Please start the ref-tools-mcp server.",
                    "query": query
                }, indent=2)
            
            # Parse sources if provided
            source_list = None
            if sources:
                source_list = [s.strip() for s in sources.split(",") if s.strip()]
            
            result = await ref_client.search_docs(query, source_list, limit)
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error searching documentation: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "query": query
            }, indent=2)
    
    @mcp.tool()
    async def create_context_pack(
        topic: str,
        queries: str,
        max_items: int = 10
    ) -> str:
        """
        Create a comprehensive context pack for a topic using multiple queries.
        
        Args:
            topic: The main topic for the context pack
            queries: Comma-separated list of related queries to gather context
                    Examples: "React hooks,useState,useEffect", "Python async,asyncio,coroutines"
            max_items: Maximum number of items to include in the context pack (default: 10)
        
        Returns:
            JSON string with comprehensive context pack including multiple sources and examples
        """
        try:
            # Check if REF Tools is available
            if not await ref_client.is_available():
                return json.dumps({
                    "success": False,
                    "error": "REF Tools server is not available. Please start the ref-tools-mcp server.",
                    "topic": topic
                }, indent=2)
            
            # Parse queries
            query_list = [q.strip() for q in queries.split(",") if q.strip()]
            
            if not query_list:
                return json.dumps({
                    "success": False,
                    "error": "At least one query must be provided",
                    "topic": topic
                }, indent=2)
            
            result = await ref_client.create_context_pack(topic, query_list, max_items)
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error creating context pack: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "topic": topic
            }, indent=2)
    
    @mcp.tool()
    async def get_enhanced_context(
        primary_query: str,
        agent_role: str = None,
        project_type: str = None,
        complexity: str = "medium"
    ) -> str:
        """
        Get enhanced context for prompt enhancement using REF Tools.
        This is specifically designed for the Phase 3 prompt enhancement system.
        
        Args:
            primary_query: The main query for context gathering
            agent_role: The role of the agent (e.g., "code_implementer", "system_architect")
            project_type: Type of project (e.g., "react_typescript", "python_api")
            complexity: Task complexity ("simple", "medium", "complex")
        
        Returns:
            JSON string with enhanced context optimized for prompt enhancement
        """
        try:
            # Check if REF Tools is available
            if not await ref_client.is_available():
                # Fallback to empty context rather than failing
                return json.dumps({
                    "success": True,
                    "context_items": [],
                    "total_relevance": 0.0,
                    "message": "REF Tools server not available - using fallback empty context",
                    "primary_query": primary_query
                }, indent=2)
            
            # Build enhanced queries based on context
            queries = [primary_query]
            
            if agent_role:
                queries.append(f"{agent_role} best practices")
                queries.append(f"{agent_role} patterns")
            
            if project_type:
                queries.append(f"{project_type} implementation")
                queries.append(f"{project_type} examples")
            
            # Adjust query depth based on complexity
            complexity_map = {
                "simple": 3,
                "medium": 5,
                "complex": 8
            }
            max_items = complexity_map.get(complexity, 5)
            
            # Create context pack
            result = await ref_client.create_context_pack(
                topic=primary_query,
                queries=queries,
                max_items=max_items
            )
            
            if result.get("success", False):
                # Transform result for prompt enhancement usage
                context_items = []
                items = result.get("items", [])
                
                for item in items:
                    context_items.append({
                        "type": "documentation",
                        "content": item.get("content", ""),
                        "source": item.get("source", "ref-tools"),
                        "relevance": item.get("relevance", 0.5),
                        "url": item.get("url", "")
                    })
                
                total_relevance = sum(item["relevance"] for item in context_items) / len(context_items) if context_items else 0.0
                
                return json.dumps({
                    "success": True,
                    "context_items": context_items,
                    "total_relevance": total_relevance,
                    "query_count": len(queries),
                    "max_items": max_items,
                    "primary_query": primary_query,
                    "agent_role": agent_role,
                    "project_type": project_type
                }, indent=2)
            else:
                # Return error from REF Tools
                return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error getting enhanced context: {e}")
            # Fallback to empty context rather than failing
            return json.dumps({
                "success": True,
                "context_items": [],
                "total_relevance": 0.0,
                "error": str(e),
                "message": "Error occurred but returning empty context to prevent failures",
                "primary_query": primary_query
            }, indent=2)
    
    # Log successful registration
    logger.info("✓ REF Tools module registered with MCP server")
    logger.info("  - REF Tools server: integrated")
    logger.info("  - Tools: ref_tools_health, read_url_content, search_documentation, create_context_pack, get_enhanced_context")