#!/usr/bin/env python
"""Fix remaining failed sources"""

import httpx
import asyncio
from datetime import datetime

REMAINING_SOURCES = [
    {
        "url": "https://huggingface.co/papers/2508.15260",
        "name": "Huggingface Paper",
        "max_pages": 3
    },
    {
        "url": "https://docs.browsermcp.io/",  # Try root URL
        "name": "BrowserMCP Documentation",
        "max_pages": 10
    },
    {
        "url": "https://github.com/BrowserMCP/mcp",
        "name": "GitHub - BrowserMCP/mcp",
        "max_pages": 5
    },
    {
        "url": "https://github.com/dl-ezo/claude-code-sub-agents",
        "name": "GitHub - claude-code-sub-agents",
        "max_pages": 5
    },
    {
        "url": "https://github.com/wshobson/agents",
        "name": "GitHub - wshobson/agents",
        "max_pages": 5
    },
    {
        "url": "https://github.com/sapientinc/HRM",
        "name": "GitHub - sapientinc/HRM", 
        "max_pages": 5
    }
]

async def crawl_source(url: str, name: str, max_pages: int):
    """Crawl a source with improved settings"""
    
    print(f"\n{'='*60}")
    print(f"Crawling: {name}")
    print(f"URL: {url}")
    print(f"Max pages: {max_pages}")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            progress_id = f"fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            response = await client.post(
                "http://localhost:8181/api/knowledge-items/crawl",
                json={
                    "url": url,
                    "max_pages": max_pages,
                    "progress_id": progress_id,
                    "wait_for": "domcontentloaded",
                    "delay": 2000,
                    "timeout": 30000
                }
            )
            
            if response.status_code == 200:
                print(f"[SUCCESS] Crawl initiated")
                return True
            else:
                print(f"[FAILED] Status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            return False

async def main():
    """Fix remaining sources"""
    print("\n" + "="*80)
    print("FIXING REMAINING FAILED SOURCES")
    print("="*80)
    
    successful = 0
    failed = 0
    
    for source in REMAINING_SOURCES:
        success = await crawl_source(
            url=source["url"],
            name=source["name"],
            max_pages=source["max_pages"]
        )
        
        if success:
            successful += 1
        else:
            failed += 1
        
        print("[WAIT] Waiting 15 seconds...")
        await asyncio.sleep(15)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total: {len(REMAINING_SOURCES)}")
    print(f"Success: {successful}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    asyncio.run(main())