#!/usr/bin/env python
"""Fix GitHub crawling by using simpler approach"""

import httpx
import asyncio
import json
from datetime import datetime

# Problematic GitHub sources - let's try with smaller page limits and delays
GITHUB_SOURCES = [
    {
        "url": "https://github.com/coleam00/Archon",
        "name": "GitHub - coleam00/Archon",
        "max_pages": 5  # Start small
    },
    {
        "url": "https://github.com/upstash/context7", 
        "name": "GitHub - upstash/context7",
        "max_pages": 5
    },
    {
        "url": "https://github.com/ref-tools/ref-tools-mcp",
        "name": "GitHub - ref-tools/ref-tools-mcp",
        "max_pages": 5
    },
    {
        "url": "https://github.com/getzep/graphiti",
        "name": "GitHub - getzep/graphiti",
        "max_pages": 5
    },
    {
        "url": "https://github.com/coleam00/context-engineering-intro",
        "name": "GitHub - context-engineering-intro",
        "max_pages": 5
    }
]

async def delete_failed_source(source_id: str):
    """Delete a failed source before re-crawling"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # First, delete the source from the database
            import os
            from supabase import create_client
            
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
            
            if supabase_url and supabase_key:
                supabase = create_client(supabase_url, supabase_key)
                
                # Delete any orphaned chunks first
                supabase.table('archon_crawled_pages').delete().eq('source_id', source_id).execute()
                
                # Delete the source
                supabase.table('archon_sources').delete().eq('source_id', source_id).execute()
                
                print(f"[DELETED] Source {source_id}")
                return True
        except Exception as e:
            print(f"[WARNING] Could not delete source {source_id}: {e}")
            return False

async def crawl_with_retry(url: str, name: str, max_pages: int = 5, max_retries: int = 3):
    """Crawl with retries and better error handling"""
    
    print(f"\n{'='*60}")
    print(f"Crawling: {name}")
    print(f"URL: {url}")
    print(f"Max pages: {max_pages}")
    print(f"{'='*60}")
    
    # Check if source exists and delete if it has no chunks
    import os
    from supabase import create_client
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
    
    if supabase_url and supabase_key:
        supabase = create_client(supabase_url, supabase_key)
        
        # Find existing source
        existing = supabase.table('archon_sources').select('source_id').eq('source_url', url).execute()
        if existing.data:
            source_id = existing.data[0]['source_id']
            
            # Check if it has chunks
            chunks = supabase.table('archon_crawled_pages').select('id').eq('source_id', source_id).limit(1).execute()
            if not chunks.data:
                print(f"[INFO] Deleting empty source {source_id} before re-crawl")
                await delete_failed_source(source_id)
    
    for attempt in range(max_retries):
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                # Use a unique progress ID for tracking
                progress_id = f"github_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{attempt}"
                
                response = await client.post(
                    "http://localhost:8181/api/knowledge-items/crawl",
                    json={
                        "url": url,
                        "max_pages": max_pages,
                        "progress_id": progress_id,
                        "wait_for": "domcontentloaded",  # Don't wait for everything
                        "delay": 2000,  # Wait 2 seconds between pages
                        "timeout": 30000  # 30 second timeout per page
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"[SUCCESS] Attempt {attempt + 1}: Crawl initiated")
                    print(f"   Task ID: {result.get('task_id', 'Unknown')}")
                    
                    # Wait a bit to let it process
                    await asyncio.sleep(30)
                    
                    # Check if chunks were created
                    if supabase_url and supabase_key:
                        supabase = create_client(supabase_url, supabase_key)
                        
                        # Find the source
                        source_check = supabase.table('archon_sources').select('source_id').eq('source_url', url).execute()
                        if source_check.data:
                            source_id = source_check.data[0]['source_id']
                            chunks = supabase.table('archon_crawled_pages').select('id').eq('source_id', source_id).execute()
                            chunk_count = len(chunks.data)
                            
                            if chunk_count > 0:
                                print(f"[VERIFIED] {chunk_count} chunks created!")
                                return True
                            else:
                                print(f"[WARNING] No chunks created yet, will retry...")
                    
                    return True
                else:
                    print(f"[FAILED] Attempt {attempt + 1}: Status {response.status_code}")
                    
            except Exception as e:
                print(f"[ERROR] Attempt {attempt + 1}: {str(e)}")
        
        if attempt < max_retries - 1:
            wait_time = (attempt + 1) * 10
            print(f"[RETRY] Waiting {wait_time} seconds before retry...")
            await asyncio.sleep(wait_time)
    
    return False

async def main():
    """Fix GitHub crawling issues"""
    print("\n" + "="*80)
    print("FIXING GITHUB CRAWLING ISSUES")
    print("Strategy: Smaller batches, retries, better error handling")
    print("="*80)
    
    successful = 0
    failed = 0
    
    for source in GITHUB_SOURCES:
        success = await crawl_with_retry(
            url=source["url"],
            name=source["name"],
            max_pages=source["max_pages"]
        )
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Longer wait between sources
        print("[WAIT] Waiting 20 seconds before next source...")
        await asyncio.sleep(20)
    
    print("\n" + "="*80)
    print("GITHUB FIX SUMMARY")
    print("="*80)
    print(f"Total sources: {len(GITHUB_SOURCES)}")
    print(f"Successfully initiated: {successful}")
    print(f"Failed: {failed}")
    print("="*80)

if __name__ == "__main__":
    # Set environment variables if running outside Docker
    import os
    if not os.getenv('SUPABASE_URL'):
        print("[INFO] Loading environment variables...")
        from dotenv import load_dotenv
        load_dotenv('C:/Jarvis/AI Workspace/Archon/.env')
    
    asyncio.run(main())