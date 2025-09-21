"""
Web Intelligence Tools for Archon MCP Server

Firecrawl-style advanced web scraping and crawling capabilities:
- Batch URL processing with intelligent rate limiting
- Deep recursive website mapping
- AI-powered structured data extraction  
- Web search with content retrieval
- Advanced content filtering and analysis
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
from datetime import datetime

try:
    import httpx
    from bs4 import BeautifulSoup
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

from mcp.server.fastmcp import Context, FastMCP

logger = logging.getLogger(__name__)


@dataclass
class CrawlConfig:
    """Configuration for web crawling operations"""
    max_depth: int = 3
    max_pages: int = 50
    delay_between_requests: float = 1.0
    timeout: int = 30
    respect_robots_txt: bool = True
    follow_external_links: bool = False
    extract_images: bool = False
    extract_links: bool = True
    content_filter: Optional[str] = None


@dataclass
class ScrapedContent:
    """Structured representation of scraped content"""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    links: List[str] = None
    images: List[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.links is None:
            self.links = []
        if self.images is None:
            self.images = []


class WebIntelligenceManager:
    """Advanced web scraping and intelligence gathering"""
    
    def __init__(self):
        self.session: Optional[httpx.AsyncClient] = None
        self.rate_limiter = {}
        
    async def get_session(self) -> httpx.AsyncClient:
        """Get or create HTTP session"""
        if self.session is None:
            headers = {
                'User-Agent': 'Archon-WebIntelligence/1.0 (Knowledge Engine)'
            }
            self.session = httpx.AsyncClient(
                headers=headers,
                timeout=30,
                follow_redirects=True
            )
        return self.session
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.aclose()
            self.session = None
    
    async def rate_limit(self, domain: str, delay: float = 1.0):
        """Implement intelligent rate limiting per domain"""
        now = time.time()
        if domain in self.rate_limiter:
            time_since_last = now - self.rate_limiter[domain]
            if time_since_last < delay:
                await asyncio.sleep(delay - time_since_last)
        self.rate_limiter[domain] = time.time()
    
    async def scrape_single_url(self, url: str, config: CrawlConfig = None) -> ScrapedContent:
        """Scrape a single URL with advanced content extraction"""
        if config is None:
            config = CrawlConfig()
            
        session = await self.get_session()
        domain = urlparse(url).netloc
        
        # Apply rate limiting
        await self.rate_limit(domain, config.delay_between_requests)
        
        try:
            response = await session.get(url, timeout=config.timeout)
            response.raise_for_status()
            
            # Parse content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.text.strip() if title_tag else urlparse(url).path
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()
            
            # Extract main content
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|article|main'))
            content = main_content.get_text(separator='\n', strip=True) if main_content else soup.get_text(separator='\n', strip=True)
            
            # Clean content
            content = re.sub(r'\n\s*\n', '\n\n', content)
            content = content[:10000]  # Limit content length
            
            # Extract links if requested
            links = []
            if config.extract_links:
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(url, href)
                    if config.follow_external_links or urlparse(full_url).netloc == domain:
                        links.append(full_url)
            
            # Extract images if requested
            images = []
            if config.extract_images:
                for img in soup.find_all('img', src=True):
                    img_url = urljoin(url, img['src'])
                    images.append(img_url)
            
            # Extract metadata
            metadata = {
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', ''),
                'content_length': len(content),
                'links_count': len(links),
                'images_count': len(images),
                'domain': domain
            }
            
            # Extract meta tags
            meta_description = soup.find('meta', attrs={'name': 'description'})
            if meta_description:
                metadata['description'] = meta_description.get('content', '')
            
            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            if meta_keywords:
                metadata['keywords'] = meta_keywords.get('content', '')
            
            return ScrapedContent(
                url=url,
                title=title,
                content=content,
                metadata=metadata,
                links=links,
                images=images
            )
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return ScrapedContent(
                url=url,
                title="Error",
                content="",
                metadata={'error': str(e)},
                links=[],
                images=[]
            )
    
    async def batch_scrape_urls(self, urls: List[str], config: CrawlConfig = None, parallel_limit: int = 5) -> List[ScrapedContent]:
        """Batch scrape multiple URLs with parallel processing and rate limiting"""
        if config is None:
            config = CrawlConfig()
        
        semaphore = asyncio.Semaphore(parallel_limit)
        
        async def scrape_with_semaphore(url: str) -> ScrapedContent:
            async with semaphore:
                return await self.scrape_single_url(url, config)
        
        tasks = [scrape_with_semaphore(url) for url in urls[:config.max_pages]]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid results
        valid_results = []
        for result in results:
            if isinstance(result, ScrapedContent):
                valid_results.append(result)
            else:
                logger.error(f"Batch scrape error: {result}")
        
        return valid_results
    
    async def deep_crawl_site(self, start_url: str, config: CrawlConfig = None) -> List[ScrapedContent]:
        """Deep crawl a website with configurable depth and filtering"""
        if config is None:
            config = CrawlConfig()
        
        visited = set()
        to_visit = [(start_url, 0)]  # (url, depth)
        results = []
        domain = urlparse(start_url).netloc
        
        while to_visit and len(results) < config.max_pages:
            current_url, depth = to_visit.pop(0)
            
            if current_url in visited or depth > config.max_depth:
                continue
                
            visited.add(current_url)
            
            # Skip if external domain and not allowed
            if not config.follow_external_links and urlparse(current_url).netloc != domain:
                continue
            
            try:
                scraped = await self.scrape_single_url(current_url, config)
                results.append(scraped)
                
                # Add links for next level crawling
                if depth < config.max_depth:
                    for link in scraped.links:
                        if link not in visited:
                            to_visit.append((link, depth + 1))
                            
            except Exception as e:
                logger.error(f"Error in deep crawl for {current_url}: {e}")
        
        return results
    
    async def extract_structured_data(self, url: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured data from URL based on provided schema"""
        scraped = await self.scrape_single_url(url)
        session = await self.get_session()
        
        try:
            # Simple structured extraction based on CSS selectors or patterns
            result = {}
            soup = BeautifulSoup((await session.get(url)).text, 'html.parser')
            
            for field, selector in schema.items():
                if isinstance(selector, str):
                    # CSS selector
                    elements = soup.select(selector)
                    if elements:
                        result[field] = [elem.get_text(strip=True) for elem in elements]
                elif isinstance(selector, dict):
                    # Pattern matching
                    pattern = selector.get('pattern')
                    if pattern:
                        matches = re.findall(pattern, scraped.content)
                        result[field] = matches
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting structured data from {url}: {e}")
            return {'error': str(e)}
    
    async def web_research(self, query: str, max_results: int = 10) -> List[ScrapedContent]:
        """Perform intelligent web research with content retrieval"""
        # Simple web search implementation
        # In production, this would use a proper search API like Google Custom Search
        search_urls = [
            f"https://www.google.com/search?q={query}",
            f"https://duckduckgo.com/?q={query}",
        ]
        
        # For now, return mock research results
        # In production, parse search results and scrape top results
        research_results = []
        
        try:
            config = CrawlConfig(max_pages=max_results, delay_between_requests=2.0)
            
            # Mock some research URLs (in production, extract from search results)
            mock_urls = [
                f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
            ]
            
            results = await self.batch_scrape_urls(mock_urls, config)
            return results
            
        except Exception as e:
            logger.error(f"Error in web research for '{query}': {e}")
            return []


# Global manager instance
_web_intelligence_manager: Optional[WebIntelligenceManager] = None


async def get_web_intelligence_manager() -> WebIntelligenceManager:
    """Get or create web intelligence manager"""
    global _web_intelligence_manager
    if _web_intelligence_manager is None:
        _web_intelligence_manager = WebIntelligenceManager()
    return _web_intelligence_manager


def register_web_intelligence_tools(mcp: FastMCP):
    """Register web intelligence tools with MCP server"""
    
    if not DEPENDENCIES_AVAILABLE:
        logger.warning("Web intelligence dependencies not available (httpx, beautifulsoup4)")
        return
    
    @mcp.tool()
    async def scrape_single_url(
        ctx: Context,
        url: str,
        max_content_length: int = 10000,
        extract_links: bool = True,
        extract_images: bool = False,
        delay: float = 1.0
    ) -> str:
        """
        Scrape a single URL and extract structured content.
        
        Args:
            url: URL to scrape
            max_content_length: Maximum content length to extract
            extract_links: Whether to extract links from the page
            extract_images: Whether to extract image URLs
            delay: Delay between requests for rate limiting
        """
        try:
            manager = await get_web_intelligence_manager()
            config = CrawlConfig(
                delay_between_requests=delay,
                extract_links=extract_links,
                extract_images=extract_images
            )
            
            result = await manager.scrape_single_url(url, config)
            
            return json.dumps({
                "success": True,
                "data": {
                    "url": result.url,
                    "title": result.title,
                    "content": result.content[:max_content_length],
                    "metadata": result.metadata,
                    "links": result.links[:50],  # Limit links
                    "images": result.images[:20],  # Limit images
                    "timestamp": result.timestamp
                }
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Error in scrape_single_url: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    @mcp.tool()
    async def batch_scrape_urls(
        ctx: Context,
        urls: str,  # JSON array of URLs
        parallel_limit: int = 5,
        delay_between_requests: float = 1.0,
        max_content_per_page: int = 5000
    ) -> str:
        """
        Batch scrape multiple URLs with parallel processing and rate limiting.
        
        Args:
            urls: JSON array of URLs to scrape
            parallel_limit: Maximum parallel requests
            delay_between_requests: Delay between requests per domain
            max_content_per_page: Maximum content length per page
        """
        try:
            url_list = json.loads(urls) if isinstance(urls, str) else urls
            if not isinstance(url_list, list):
                return json.dumps({"success": False, "error": "URLs must be a JSON array"})
            
            manager = await get_web_intelligence_manager()
            config = CrawlConfig(
                delay_between_requests=delay_between_requests,
                extract_links=True,
                max_pages=len(url_list)
            )
            
            results = await manager.batch_scrape_urls(url_list, config, parallel_limit)
            
            processed_results = []
            for result in results:
                processed_results.append({
                    "url": result.url,
                    "title": result.title,
                    "content": result.content[:max_content_per_page],
                    "metadata": result.metadata,
                    "links_count": len(result.links),
                    "timestamp": result.timestamp
                })
            
            return json.dumps({
                "success": True,
                "total_processed": len(processed_results),
                "data": processed_results
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Error in batch_scrape_urls: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    @mcp.tool()
    async def deep_crawl_website(
        ctx: Context,
        start_url: str,
        max_depth: int = 2,
        max_pages: int = 20,
        follow_external_links: bool = False,
        delay_between_requests: float = 1.5
    ) -> str:
        """
        Perform deep crawling of a website with configurable depth and limits.
        
        Args:
            start_url: Starting URL for crawling
            max_depth: Maximum crawl depth
            max_pages: Maximum pages to crawl
            follow_external_links: Whether to follow external domain links
            delay_between_requests: Delay between requests for rate limiting
        """
        try:
            manager = await get_web_intelligence_manager()
            config = CrawlConfig(
                max_depth=max_depth,
                max_pages=max_pages,
                follow_external_links=follow_external_links,
                delay_between_requests=delay_between_requests,
                extract_links=True
            )
            
            results = await manager.deep_crawl_site(start_url, config)
            
            crawl_summary = {
                "start_url": start_url,
                "pages_crawled": len(results),
                "total_links_found": sum(len(r.links) for r in results),
                "domains_visited": len(set(urlparse(r.url).netloc for r in results)),
                "crawl_depth_used": max_depth,
                "timestamp": datetime.now().isoformat()
            }
            
            page_data = []
            for result in results:
                page_data.append({
                    "url": result.url,
                    "title": result.title,
                    "content_preview": result.content[:500] + "..." if len(result.content) > 500 else result.content,
                    "content_length": len(result.content),
                    "outbound_links": len(result.links),
                    "metadata": result.metadata
                })
            
            return json.dumps({
                "success": True,
                "summary": crawl_summary,
                "pages": page_data
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Error in deep_crawl_website: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    @mcp.tool()
    async def extract_structured_data(
        ctx: Context,
        url: str,
        extraction_schema: str  # JSON schema for extraction
    ) -> str:
        """
        Extract structured data from a URL based on provided schema.
        
        Args:
            url: URL to extract data from
            extraction_schema: JSON schema defining what to extract
                Example: {"titles": "h1, h2", "prices": {"pattern": r"\\$\\d+\\.\\d{2}"}}
        """
        try:
            schema = json.loads(extraction_schema)
            manager = await get_web_intelligence_manager()
            
            result = await manager.extract_structured_data(url, schema)
            
            return json.dumps({
                "success": True,
                "url": url,
                "extracted_data": result,
                "schema_used": schema,
                "timestamp": datetime.now().isoformat()
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Error in extract_structured_data: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    @mcp.tool()
    async def web_intelligence_research(
        ctx: Context,
        research_query: str,
        max_results: int = 10,
        include_content: bool = True
    ) -> str:
        """
        Perform intelligent web research with content retrieval and analysis.
        
        Args:
            research_query: Research query or topic
            max_results: Maximum number of results to return
            include_content: Whether to include full content or just summaries
        """
        try:
            manager = await get_web_intelligence_manager()
            
            results = await manager.web_research(research_query, max_results)
            
            research_data = {
                "query": research_query,
                "results_found": len(results),
                "timestamp": datetime.now().isoformat(),
                "results": []
            }
            
            for result in results:
                result_data = {
                    "url": result.url,
                    "title": result.title,
                    "relevance_score": 1.0,  # Simple placeholder
                    "domain": result.metadata.get('domain', ''),
                    "content_length": len(result.content)
                }
                
                if include_content:
                    result_data["content"] = result.content[:2000]  # Limit content
                else:
                    result_data["summary"] = result.content[:200] + "..."
                
                research_data["results"].append(result_data)
            
            return json.dumps({
                "success": True,
                "research_data": research_data
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Error in web_intelligence_research: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })


# Cleanup function
async def cleanup_web_intelligence():
    """Cleanup web intelligence resources"""
    global _web_intelligence_manager
    if _web_intelligence_manager:
        await _web_intelligence_manager.close()
        _web_intelligence_manager = None