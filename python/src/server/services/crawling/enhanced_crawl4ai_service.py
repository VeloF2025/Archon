"""
Enhanced Crawl4AI Service with Advanced Features

This module provides advanced web crawling capabilities including:
- LLM-powered semantic extraction strategies
- Adaptive crawling with information foraging
- Enhanced structured data extraction  
- Anti-detection stealth mode
- Intelligent content filtering and caching
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Callable, Union
from urllib.parse import urlparse, urljoin
import time

from crawl4ai import (
    AsyncWebCrawler, 
    CrawlerRunConfig, 
    CacheMode,
    BrowserConfig
)
from crawl4ai.extraction_strategy import (
    LLMExtractionStrategy,
    CssExtractionStrategy,
    JsonCssExtractionStrategy
)

from ....config.logfire_config import get_logger, safe_logfire_info, safe_logfire_error

logger = get_logger(__name__)


class AdvancedExtractionStrategies:
    """Collection of advanced extraction strategies for different content types."""
    
    @staticmethod
    def get_semantic_content_strategy(content_type: str = "documentation") -> LLMExtractionStrategy:
        """Get LLM-powered semantic extraction strategy."""
        if content_type == "documentation":
            system_prompt = """
            You are an expert at extracting structured information from documentation pages.
            Extract the main content, code examples, API references, and key concepts.
            Focus on technical accuracy and completeness.
            """
            
            user_prompt = """
            Extract the following from this documentation page:
            1. Main topic/title
            2. Key concepts and definitions
            3. Code examples with explanations
            4. API methods and parameters
            5. Important warnings or notes
            6. Related links or references
            
            Format as clean, structured content suitable for RAG systems.
            """
        
        elif content_type == "article":
            system_prompt = """
            You are an expert at extracting key information from articles and blog posts.
            Focus on main ideas, arguments, and supporting evidence.
            """
            
            user_prompt = """
            Extract the following from this article:
            1. Main thesis or argument
            2. Key supporting points
            3. Important data or statistics
            4. Conclusions and implications
            5. Author's recommendations
            
            Maintain the logical flow and structure of the content.
            """
        
        else:  # general content
            system_prompt = """
            You are an expert at extracting meaningful content from web pages.
            Focus on the most important and relevant information.
            """
            
            user_prompt = """
            Extract the main content from this page, focusing on:
            1. Primary information or message
            2. Key details and facts
            3. Any structured data or lists
            4. Important contextual information
            
            Present in a clear, well-organized format.
            """
        
        return LLMExtractionStrategy(
            provider="openai/gpt-4o-mini",  # Cost-effective model for extraction
            api_token=None,  # Will use environment variable
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format={"type": "text"}
        )
    
    @staticmethod
    def get_structured_data_strategy() -> JsonCssExtractionStrategy:
        """Extract structured data like articles, products, etc."""
        schema = {
            "name": "structured_content",
            "baseSelector": "body",
            "fields": [
                {
                    "name": "title",
                    "selector": "h1, title, .title, [role='heading'][aria-level='1']",
                    "type": "text"
                },
                {
                    "name": "description", 
                    "selector": "meta[name='description'], .description, .summary",
                    "type": "text",
                    "attribute": "content"
                },
                {
                    "name": "headings",
                    "selector": "h2, h3, h4, h5, h6",
                    "type": "list",
                    "fields": [
                        {"name": "text", "type": "text"},
                        {"name": "level", "type": "attribute", "attribute": "tagName"}
                    ]
                },
                {
                    "name": "code_blocks",
                    "selector": "pre code, .highlight code, .codehilite code",
                    "type": "list",
                    "fields": [
                        {"name": "language", "type": "attribute", "attribute": "class"},
                        {"name": "code", "type": "text"}
                    ]
                },
                {
                    "name": "links",
                    "selector": "a[href]",
                    "type": "list", 
                    "fields": [
                        {"name": "text", "type": "text"},
                        {"name": "href", "type": "attribute", "attribute": "href"}
                    ]
                }
            ]
        }
        return JsonCssExtractionStrategy(schema)


class AdaptiveCrawlingEngine:
    """Intelligent crawling engine with adaptive stopping and quality assessment."""
    
    def __init__(self):
        self.content_quality_threshold = 0.7
        self.min_content_length = 200
        self.max_similar_pages = 3
        self.visited_content_hashes = set()
    
    def assess_content_quality(self, content: str, url: str) -> float:
        """Assess the quality and uniqueness of crawled content."""
        if not content or len(content) < self.min_content_length:
            return 0.0
        
        # Basic quality indicators
        score = 0.0
        
        # Length score (normalized)
        length_score = min(len(content) / 2000, 1.0) * 0.3
        score += length_score
        
        # Structure score (headings, paragraphs, etc.)
        structure_indicators = ['#', '##', '###', '\n\n', '```']
        structure_score = sum(1 for indicator in structure_indicators if indicator in content)
        structure_score = min(structure_score / len(structure_indicators), 1.0) * 0.3
        score += structure_score
        
        # Content type score
        technical_indicators = ['function', 'class', 'import', 'def', 'const', 'var', 'API', 'documentation']
        tech_score = sum(1 for indicator in technical_indicators if indicator.lower() in content.lower())
        tech_score = min(tech_score / 3, 1.0) * 0.4  # Higher weight for technical content
        score += tech_score
        
        logger.info(f"Content quality assessment for {url}: {score:.2f} (length: {length_score:.2f}, structure: {structure_score:.2f}, tech: {tech_score:.2f})")
        return score
    
    def should_continue_crawling(self, crawled_results: List[Dict], target_pages: int) -> bool:
        """Determine if crawling should continue based on adaptive criteria."""
        if len(crawled_results) >= target_pages:
            return False
        
        # Quality-based stopping
        high_quality_pages = sum(1 for result in crawled_results 
                               if result.get('quality_score', 0) >= self.content_quality_threshold)
        
        if high_quality_pages >= max(3, target_pages * 0.5):  # At least 50% high quality or 3 pages
            logger.info(f"Adaptive stopping: Found {high_quality_pages} high-quality pages, stopping early")
            return False
        
        return True


class EnhancedCrawl4AIService:
    """Enhanced Crawl4AI service with advanced features."""
    
    def __init__(self, enable_stealth: bool = True, max_concurrent: int = 5):
        self.adaptive_engine = AdaptiveCrawlingEngine()
        self.extraction_strategies = AdvancedExtractionStrategies()
        self.enable_stealth = enable_stealth
        self.max_concurrent = max_concurrent
        self.crawler = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize_crawler()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_crawler()
    
    async def initialize_crawler(self):
        """Initialize the enhanced crawler with optimal configuration."""
        browser_config = BrowserConfig(
            headless=True,
            viewport_width=1920,
            viewport_height=1080,
            # Anti-detection features
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36" if self.enable_stealth else None,
            # Performance optimizations
            java_script_enabled=True,
            cookies_enabled=True,
            ignore_https_errors=True,
            extra_args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-extensions",
                "--disable-plugins",
                "--disable-images" if not self.enable_stealth else "",  # Load images only in stealth mode
            ] if self.enable_stealth else None
        )
        
        self.crawler = AsyncWebCrawler(
            config=browser_config,
            verbose=False  # Reduce noise in logs
        )
        await self.crawler.start()
        logger.info("Enhanced Crawl4AI crawler initialized with stealth mode: %s", self.enable_stealth)
    
    async def close_crawler(self):
        """Close the crawler and cleanup resources."""
        if self.crawler:
            await self.crawler.close()
            logger.info("Enhanced Crawl4AI crawler closed")
    
    async def crawl_with_adaptive_strategy(
        self,
        urls: Union[str, List[str]],
        content_type: str = "documentation",
        max_pages: int = 10,
        use_llm_extraction: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Crawl URLs with adaptive strategies and intelligent stopping.
        
        Args:
            urls: Single URL or list of URLs to crawl
            content_type: Type of content for semantic extraction ("documentation", "article", "general")
            max_pages: Maximum pages to crawl
            use_llm_extraction: Whether to use LLM-powered semantic extraction
            progress_callback: Optional progress callback
            
        Returns:
            List of enhanced crawling results with quality scores
        """
        if isinstance(urls, str):
            urls = [urls]
        
        results = []
        total_urls = len(urls)
        
        # Determine extraction strategy
        extraction_strategy = None
        if use_llm_extraction:
            try:
                extraction_strategy = self.extraction_strategies.get_semantic_content_strategy(content_type)
                logger.info(f"Using LLM extraction strategy for content type: {content_type}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM extraction strategy: {e}")
                # Fallback to structured data extraction
                extraction_strategy = self.extraction_strategies.get_structured_data_strategy()
        else:
            extraction_strategy = self.extraction_strategies.get_structured_data_strategy()
        
        # Process URLs with adaptive crawling
        for i, url in enumerate(urls):
            # Check adaptive stopping criteria
            if not self.adaptive_engine.should_continue_crawling(results, max_pages):
                logger.info(f"Adaptive crawling stopped early at {len(results)} pages")
                break
            
            if progress_callback:
                await progress_callback(f"Crawling {url}", int((i / total_urls) * 100))
            
            try:
                result = await self._crawl_single_url_enhanced(url, extraction_strategy, content_type)
                if result and result.get('success'):
                    # Assess content quality
                    quality_score = self.adaptive_engine.assess_content_quality(
                        result.get('markdown', ''), url
                    )
                    result['quality_score'] = quality_score
                    result['extraction_method'] = 'llm' if use_llm_extraction else 'structured'
                    results.append(result)
                    
                    logger.info(f"Successfully crawled {url} with quality score: {quality_score:.2f}")
            
            except Exception as e:
                logger.error(f"Error crawling {url}: {e}")
                continue
        
        logger.info(f"Enhanced crawling completed: {len(results)} pages crawled from {total_urls} URLs")
        return results
    
    async def _crawl_single_url_enhanced(
        self, 
        url: str, 
        extraction_strategy,
        content_type: str
    ) -> Dict[str, Any]:
        """Crawl a single URL with enhanced configuration."""
        
        # Configure crawl settings based on content type
        if content_type == "documentation":
            wait_for = "main, article, .content, .markdown, .doc"
            page_timeout = 30000
            delay_before_return = 1.0
        else:
            wait_for = "body"
            page_timeout = 20000
            delay_before_return = 0.5
        
        crawl_config = CrawlerRunConfig(
            cache_mode=CacheMode.ENABLED,
            # Extraction strategy
            extraction_strategy=extraction_strategy,
            # Performance settings
            page_timeout=page_timeout,
            delay_before_return_html=delay_before_return,
            wait_for=wait_for,
            # Content processing
            process_iframes=True,
            remove_overlay_elements=True,
            scan_full_page=True,
            # Anti-detection
            simulate_user=self.enable_stealth,
            magic=self.enable_stealth,  # Enable Crawl4AI's anti-detection magic
            # Advanced features
            screenshot=False,  # Save bandwidth
            pdf=False,  # Don't generate PDFs
            word_count_threshold=50,  # Minimum word count
        )
        
        try:
            start_time = time.time()
            result = await self.crawler.arun(url=url, config=crawl_config)
            crawl_time = time.time() - start_time
            
            if not result.success:
                logger.warning(f"Crawl failed for {url}: {result.error_message}")
                return {
                    "success": False,
                    "url": url,
                    "error": result.error_message,
                    "crawl_time": crawl_time
                }
            
            # Enhanced result processing
            processed_result = {
                "success": True,
                "url": url,
                "title": result.title or "Untitled",
                "markdown": result.markdown,
                "html": result.html,
                "links": result.links,
                "crawl_time": crawl_time,
                "content_length": len(result.markdown) if result.markdown else 0,
                "extracted_content": result.extracted_content if hasattr(result, 'extracted_content') else None,
                "metadata": {
                    "status_code": getattr(result, 'status_code', None),
                    "response_headers": getattr(result, 'response_headers', {}),
                    "page_load_time": crawl_time,
                    "extraction_strategy": type(extraction_strategy).__name__
                }
            }
            
            logger.info(f"Enhanced crawl completed for {url} in {crawl_time:.2f}s")
            return processed_result
            
        except Exception as e:
            logger.error(f"Exception during enhanced crawl of {url}: {e}")
            return {
                "success": False,
                "url": url,
                "error": str(e),
                "crawl_time": 0
            }
    
    async def batch_crawl_with_concurrency(
        self,
        urls: List[str],
        content_type: str = "documentation",
        max_concurrent: int = None,
        use_llm_extraction: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch crawl URLs with optimal concurrency and advanced features.
        
        Args:
            urls: List of URLs to crawl
            content_type: Content type for extraction strategy
            max_concurrent: Maximum concurrent requests (defaults to instance setting)
            use_llm_extraction: Whether to use LLM extraction
            progress_callback: Progress callback function
            
        Returns:
            List of crawl results
        """
        if not urls:
            return []
        
        max_concurrent = max_concurrent or self.max_concurrent
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def crawl_with_semaphore(url: str, index: int) -> Dict[str, Any]:
            async with semaphore:
                if progress_callback:
                    await progress_callback(f"Crawling {url}", int((index / len(urls)) * 100))
                
                # Use the adaptive strategy for each URL
                results = await self.crawl_with_adaptive_strategy(
                    [url], 
                    content_type=content_type,
                    max_pages=1,
                    use_llm_extraction=use_llm_extraction
                )
                return results[0] if results else {"success": False, "url": url, "error": "No result returned"}
        
        # Execute all crawls concurrently
        tasks = [crawl_with_semaphore(url, i) for i, url in enumerate(urls)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception in batch crawl for {urls[i]}: {result}")
                processed_results.append({
                    "success": False,
                    "url": urls[i],
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        successful_count = sum(1 for r in processed_results if r.get('success'))
        logger.info(f"Batch crawl completed: {successful_count}/{len(urls)} successful")
        
        return processed_results


# Convenience function for easy integration
async def create_enhanced_crawler(enable_stealth: bool = True, max_concurrent: int = 5) -> EnhancedCrawl4AIService:
    """Create and initialize an enhanced Crawl4AI service."""
    service = EnhancedCrawl4AIService(enable_stealth=enable_stealth, max_concurrent=max_concurrent)
    await service.initialize_crawler()
    return service