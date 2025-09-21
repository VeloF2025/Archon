"""
Enhanced Crawling Strategy with Advanced Crawl4AI Features

This strategy leverages the enhanced Crawl4AI service for superior web crawling:
- LLM-powered semantic content extraction
- Adaptive crawling with quality assessment
- Anti-detection stealth mode
- Structured data extraction
- Intelligent stopping algorithms
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable, Awaitable
from urllib.parse import urlparse

from ..enhanced_crawl4ai_service import EnhancedCrawl4AIService
from .....config.logfire_config import get_logger, safe_logfire_info, safe_logfire_error

logger = get_logger(__name__)


class EnhancedCrawlStrategy:
    """Enhanced crawling strategy using advanced Crawl4AI features."""
    
    def __init__(self, enable_stealth: bool = True, max_concurrent: int = 10):
        self.enable_stealth = enable_stealth
        self.max_concurrent = max_concurrent
        self.enhanced_service = None
    
    async def _ensure_service_initialized(self):
        """Ensure the enhanced service is initialized."""
        if not self.enhanced_service:
            self.enhanced_service = EnhancedCrawl4AIService(
                enable_stealth=self.enable_stealth,
                max_concurrent=self.max_concurrent
            )
            await self.enhanced_service.initialize_crawler()
    
    def _detect_content_type(self, url: str) -> str:
        """Detect content type based on URL patterns."""
        url_lower = url.lower()
        
        # Documentation sites
        doc_indicators = [
            'docs.', 'doc.', 'documentation', 'api.', 'developer.',
            'guide', 'tutorial', 'manual', '/docs/', '/api/',
            'readme', 'wiki', 'gitbook', 'docusaurus', 'mkdocs'
        ]
        
        if any(indicator in url_lower for indicator in doc_indicators):
            return "documentation"
        
        # Article/blog indicators  
        article_indicators = [
            'blog', 'article', 'post', 'news', 'medium.com',
            'substack', 'dev.to', 'hashnode'
        ]
        
        if any(indicator in url_lower for indicator in article_indicators):
            return "article"
        
        return "general"
    
    async def enhanced_single_page_crawl(
        self,
        url: str,
        use_llm_extraction: bool = True,
        content_type: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Crawl a single page with enhanced features.
        
        Args:
            url: URL to crawl
            use_llm_extraction: Whether to use LLM-powered extraction
            content_type: Content type override ("documentation", "article", "general")
            progress_callback: Progress callback function
            
        Returns:
            Enhanced crawl result with quality scoring
        """
        await self._ensure_service_initialized()
        
        # Auto-detect content type if not provided
        if not content_type:
            content_type = self._detect_content_type(url)
            logger.info(f"Auto-detected content type '{content_type}' for {url}")
        
        try:
            results = await self.enhanced_service.crawl_with_adaptive_strategy(
                urls=[url],
                content_type=content_type,
                max_pages=1,
                use_llm_extraction=use_llm_extraction,
                progress_callback=progress_callback
            )
            
            if results:
                result = results[0]
                logger.info(f"Enhanced single page crawl successful for {url}, quality: {result.get('quality_score', 0):.2f}")
                return result
            else:
                return {
                    "success": False,
                    "url": url,
                    "error": "No content extracted"
                }
        
        except Exception as e:
            safe_logfire_error(f"Enhanced single page crawl failed for {url}: {e}")
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }
    
    async def enhanced_batch_crawl(
        self,
        urls: List[str],
        use_llm_extraction: bool = True,
        content_type: Optional[str] = None,
        max_concurrent: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
        start_progress: int = 0,
        end_progress: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Crawl multiple URLs with enhanced batch processing.
        
        Args:
            urls: List of URLs to crawl
            use_llm_extraction: Whether to use LLM extraction
            content_type: Content type for all URLs
            max_concurrent: Maximum concurrent requests
            progress_callback: Progress callback
            start_progress: Starting progress percentage
            end_progress: Ending progress percentage
            
        Returns:
            List of enhanced crawl results
        """
        if not urls:
            return []
        
        await self._ensure_service_initialized()
        
        # Auto-detect content type from first URL if not provided
        if not content_type:
            content_type = self._detect_content_type(urls[0])
            logger.info(f"Auto-detected content type '{content_type}' for batch crawl")
        
        # Create progress wrapper
        async def batch_progress_callback(message: str, percentage: int):
            if progress_callback:
                # Map percentage to the specified range
                mapped_percentage = start_progress + int((percentage / 100) * (end_progress - start_progress))
                await progress_callback("crawling", mapped_percentage, message)
        
        try:
            results = await self.enhanced_service.batch_crawl_with_concurrency(
                urls=urls,
                content_type=content_type,
                max_concurrent=max_concurrent or self.max_concurrent,
                use_llm_extraction=use_llm_extraction,
                progress_callback=batch_progress_callback
            )
            
            successful_count = sum(1 for r in results if r.get('success'))
            avg_quality = sum(r.get('quality_score', 0) for r in results if r.get('success')) / max(successful_count, 1)
            
            logger.info(f"Enhanced batch crawl completed: {successful_count}/{len(urls)} successful, avg quality: {avg_quality:.2f}")
            
            return results
        
        except Exception as e:
            safe_logfire_error(f"Enhanced batch crawl failed: {e}")
            return [{"success": False, "url": url, "error": str(e)} for url in urls]
    
    async def enhanced_recursive_crawl(
        self,
        start_urls: List[str],
        max_depth: int = 3,
        max_pages: int = 50,
        use_llm_extraction: bool = True,
        content_type: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        start_progress: int = 0,
        end_progress: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Enhanced recursive crawling with adaptive stopping.
        
        Args:
            start_urls: Starting URLs for recursive crawl
            max_depth: Maximum crawl depth
            max_pages: Maximum pages to crawl
            use_llm_extraction: Whether to use LLM extraction
            content_type: Content type for extraction
            progress_callback: Progress callback
            start_progress: Starting progress percentage
            end_progress: Ending progress percentage
            
        Returns:
            List of crawled results with quality scoring
        """
        if not start_urls:
            return []
        
        await self._ensure_service_initialized()
        
        # Auto-detect content type
        if not content_type:
            content_type = self._detect_content_type(start_urls[0])
            logger.info(f"Auto-detected content type '{content_type}' for recursive crawl")
        
        crawled_urls = set()
        all_results = []
        urls_to_crawl = [(url, 0) for url in start_urls]  # (url, depth)
        
        async def recursive_progress_callback(message: str, percentage: int):
            if progress_callback:
                mapped_percentage = start_progress + int((percentage / 100) * (end_progress - start_progress))
                await progress_callback("crawling", mapped_percentage, message)
        
        try:
            while urls_to_crawl and len(all_results) < max_pages:
                current_batch = []
                remaining_urls = []
                
                # Collect URLs for current batch (same depth level)
                current_depth = urls_to_crawl[0][1] if urls_to_crawl else 0
                
                for url, depth in urls_to_crawl:
                    if url in crawled_urls:
                        continue
                    
                    if depth == current_depth and len(current_batch) < self.max_concurrent:
                        current_batch.append(url)
                        crawled_urls.add(url)
                    else:
                        remaining_urls.append((url, depth))
                
                if not current_batch:
                    break
                
                # Crawl current batch
                await recursive_progress_callback(
                    f"Crawling depth {current_depth}: {len(current_batch)} pages",
                    int((len(all_results) / max_pages) * 100)
                )
                
                batch_results = await self.enhanced_batch_crawl(
                    urls=current_batch,
                    use_llm_extraction=use_llm_extraction,
                    content_type=content_type,
                    max_concurrent=self.max_concurrent,
                    progress_callback=None  # Handle progress at this level
                )
                
                # Process results and extract new URLs for next depth
                for result in batch_results:
                    if result.get('success'):
                        all_results.append(result)
                        
                        # Extract internal links for next depth (if not at max depth)
                        if current_depth < max_depth and result.get('links'):
                            base_domain = urlparse(result['url']).netloc
                            
                            for link in result['links']:
                                if isinstance(link, dict):
                                    link_url = link.get('href', '')
                                else:
                                    link_url = str(link)
                                
                                # Filter for internal links
                                if link_url and not link_url.startswith(('http://other', 'https://other')):
                                    # Make absolute URL if needed
                                    if link_url.startswith('/'):
                                        link_url = f"https://{base_domain}{link_url}"
                                    
                                    # Check if it's same domain
                                    try:
                                        link_domain = urlparse(link_url).netloc
                                        if link_domain == base_domain and link_url not in crawled_urls:
                                            remaining_urls.append((link_url, current_depth + 1))
                                    except:
                                        continue  # Skip invalid URLs
                
                urls_to_crawl = remaining_urls
                
                # Adaptive stopping check
                if not self.enhanced_service.adaptive_engine.should_continue_crawling(all_results, max_pages):
                    logger.info("Adaptive stopping triggered in recursive crawl")
                    break
            
            successful_count = len(all_results)
            avg_quality = sum(r.get('quality_score', 0) for r in all_results) / max(successful_count, 1)
            
            logger.info(f"Enhanced recursive crawl completed: {successful_count} pages, avg quality: {avg_quality:.2f}")
            
            return all_results
        
        except Exception as e:
            safe_logfire_error(f"Enhanced recursive crawl failed: {e}")
            return all_results  # Return what we have so far
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.enhanced_service:
            await self.enhanced_service.close_crawler()
            self.enhanced_service = None


# Integration wrapper for existing CrawlingService
class EnhancedCrawlIntegration:
    """Integration wrapper to upgrade existing crawling service with enhanced features."""
    
    @staticmethod
    async def upgrade_crawling_service(crawling_service, enable_llm_extraction: bool = False):
        """
        Upgrade an existing CrawlingService instance with enhanced capabilities.
        
        Args:
            crawling_service: Existing CrawlingService instance
            enable_llm_extraction: Whether to enable LLM extraction by default
        """
        # Add enhanced strategy
        crawling_service.enhanced_strategy = EnhancedCrawlStrategy()
        crawling_service.enable_llm_extraction = enable_llm_extraction
        
        # Add enhanced methods
        async def enhanced_crawl_single_page(url: str, retry_count: int = 3):
            try:
                result = await crawling_service.enhanced_strategy.enhanced_single_page_crawl(
                    url=url,
                    use_llm_extraction=crawling_service.enable_llm_extraction,
                    progress_callback=None
                )
                return result
            except Exception as e:
                # Fallback to original method
                logger.warning(f"Enhanced crawl failed for {url}, falling back to original: {e}")
                return await crawling_service.single_page_strategy.crawl_single_page(
                    url, crawling_service.url_handler.transform_github_url,
                    crawling_service.site_config.is_documentation_site, retry_count
                )
        
        async def enhanced_crawl_batch_with_progress(
            urls: List[str],
            max_concurrent: int = None,
            progress_callback=None,
            start_progress: int = 15,
            end_progress: int = 60
        ):
            try:
                results = await crawling_service.enhanced_strategy.enhanced_batch_crawl(
                    urls=urls,
                    use_llm_extraction=crawling_service.enable_llm_extraction,
                    max_concurrent=max_concurrent,
                    progress_callback=progress_callback,
                    start_progress=start_progress,
                    end_progress=end_progress
                )
                return results
            except Exception as e:
                # Fallback to original method
                logger.warning(f"Enhanced batch crawl failed, falling back to original: {e}")
                return await crawling_service.batch_strategy.crawl_batch_with_progress(
                    urls, crawling_service.url_handler.transform_github_url,
                    crawling_service.site_config.is_documentation_site,
                    max_concurrent, progress_callback, start_progress, end_progress
                )
        
        # Replace methods with enhanced versions
        crawling_service.enhanced_crawl_single_page = enhanced_crawl_single_page
        crawling_service.enhanced_crawl_batch_with_progress = enhanced_crawl_batch_with_progress
        
        logger.info("CrawlingService upgraded with enhanced Crawl4AI capabilities")