#!/usr/bin/env python
"""Re-crawl all failed sources in the knowledge base"""

import asyncio
import httpx
import json
from datetime import datetime

# List of failed sources to re-crawl
FAILED_SOURCES = [
    {
        "url": "https://docs.anthropic.com/en/docs/claude-code/overview",
        "name": "Anthropic Documentation - Claude Code Overview",
        "max_pages": 50
    },
    {
        "url": "https://github.com/coleam00/Archon",
        "name": "GitHub - coleam00/Archon",
        "max_pages": 30
    },
    {
        "url": "https://docs.anthropic.com/en/docs/build-with-claude/model-context-protocol",
        "name": "Anthropic Documentation - MCP",
        "max_pages": 50
    },
    {
        "url": "https://github.com/upstash/context7",
        "name": "GitHub - upstash/context7",
        "max_pages": 30
    },
    {
        "url": "https://github.com/ref-tools/ref-tools-mcp",
        "name": "GitHub - ref-tools/ref-tools-mcp",
        "max_pages": 30
    },
    {
        "url": "https://github.com/getzep/graphiti",
        "name": "GitHub - getzep/graphiti",
        "max_pages": 30
    },
    {
        "url": "https://github.com/coleam00/context-engineering-intro",
        "name": "GitHub - context-engineering-intro",
        "max_pages": 30
    },
    {
        "url": "https://api-docs.deepseek.com/",
        "name": "DeepSeek API Documentation",
        "max_pages": 50
    },
    {
        "url": "https://huggingface.co/papers/2508.15260",
        "name": "Huggingface Paper",
        "max_pages": 10
    },
    {
        "url": "https://docs.browsermcp.io/welcome",
        "name": "BrowserMCP Documentation",
        "max_pages": 50
    },
    {
        "url": "https://github.com/BrowserMCP/mcp",
        "name": "GitHub - BrowserMCP/mcp",
        "max_pages": 30
    },
    {
        "url": "https://github.com/dl-ezo/claude-code-sub-agents",
        "name": "GitHub - claude-code-sub-agents",
        "max_pages": 30
    },
    {
        "url": "https://github.com/wshobson/agents",
        "name": "GitHub - wshobson/agents",
        "max_pages": 30
    },
    {
        "url": "https://github.com/sapientinc/HRM",
        "name": "GitHub - sapientinc/HRM",
        "max_pages": 30
    }
]

async def crawl_source(url: str, name: str, max_pages: int = 30):
    """Crawl a single source"""
    print(f"\n{'='*60}")
    print(f"Crawling: {name}")
    print(f"URL: {url}")
    print(f"Max pages: {max_pages}")
    print(f"{'='*60}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # Call the crawl API endpoint
            response = await client.post(
                "http://localhost:8181/api/knowledge-items/crawl",
                json={
                    "url": url,
                    "max_pages": max_pages,
                    "progress_id": f"recrawl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"[SUCCESS] Crawl initiated for {name}")
                print(f"   Task ID: {result.get('task_id', 'Unknown')}")
                print(f"   Progress ID: {result.get('progress_id', 'Unknown')}")
                return True
            else:
                print(f"[FAILED] {name}")
                print(f"   Status: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                return False
                
        except Exception as e:
            print(f"[ERROR] crawling {name}: {str(e)}")
            return False

async def main():
    """Re-crawl all failed sources"""
    print("\n" + "="*80)
    print("RE-CRAWLING FAILED KNOWLEDGE BASE SOURCES")
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    successful = 0
    failed = 0
    
    # Process sources with delays to avoid overwhelming the system
    for source in FAILED_SOURCES:
        success = await crawl_source(
            url=source["url"],
            name=source["name"],
            max_pages=source["max_pages"]
        )
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Wait between crawls to avoid rate limiting
        print("[WAIT] Waiting 5 seconds before next crawl...")
        await asyncio.sleep(5)
    
    # Print summary
    print("\n" + "="*80)
    print("RE-CRAWL SUMMARY")
    print("="*80)
    print(f"Total sources: {len(FAILED_SOURCES)}")
    print(f"Successfully initiated: {successful}")
    print(f"Failed to initiate: {failed}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print("\nNOTE: Crawling happens asynchronously in the background.")
    print("Check the Archon UI or logs to monitor progress.")
    print("Run the chunking status check again in a few minutes to verify.")

if __name__ == "__main__":
    asyncio.run(main())