#!/usr/bin/env python
"""Fix the final 3 sources with specific strategies"""

import httpx
import asyncio
from datetime import datetime

# The actual 3 that are still failing
FINAL_SOURCES = [
    {
        "url": "https://github.com/coleam00/context-engineering-intro",  # Remove the subdirectory
        "name": "GitHub - context-engineering-intro",
        "source_id": "a04bdb36f4f29bfe",
        "max_pages": 10
    },
    {
        "url": "https://huggingface.co/papers/2508.15260",
        "name": "Huggingface Paper",
        "source_id": "e304a0573ea22c1d",
        "max_pages": 5
    },
    {
        "url": "https://docs.browsermcp.io",  # Try root without /welcome
        "name": "BrowserMCP Documentation",
        "source_id": "b0c6ad1afc92b92f",
        "max_pages": 10
    }
]

async def delete_and_recrawl(source_info: dict):
    """Delete the failed source and create fresh"""
    
    print(f"\n{'='*60}")
    print(f"FIXING: {source_info['name']}")
    print(f"URL: {source_info['url']}")
    print(f"Strategy: Delete and fresh crawl")
    print(f"{'='*60}")
    
    # Delete the old source first
    import os
    from supabase import create_client
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
    
    if supabase_url and supabase_key:
        supabase = create_client(supabase_url, supabase_key)
        
        # Delete any orphaned chunks
        del_chunks = supabase.table('archon_crawled_pages').delete().eq('source_id', source_info['source_id']).execute()
        print(f"[CLEANUP] Deleted {len(del_chunks.data) if del_chunks.data else 0} orphaned chunks")
        
        # Delete the source
        del_source = supabase.table('archon_sources').delete().eq('source_id', source_info['source_id']).execute()
        print(f"[CLEANUP] Deleted source record")
    
    # Wait a bit
    await asyncio.sleep(5)
    
    # Now crawl fresh with better settings
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            progress_id = f"final_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Special handling for different source types
            if "huggingface" in source_info['url']:
                # Huggingface might need longer waits
                wait_strategy = "networkidle"
                delay = 5000
            elif "github.com" in source_info['url']:
                # GitHub needs simple strategy
                wait_strategy = "domcontentloaded"
                delay = 3000
            else:
                # Documentation sites
                wait_strategy = "load"
                delay = 2000
            
            response = await client.post(
                "http://localhost:8181/api/knowledge-items/crawl",
                json={
                    "url": source_info['url'],
                    "max_pages": source_info['max_pages'],
                    "progress_id": progress_id,
                    "wait_for": wait_strategy,
                    "delay": delay,
                    "timeout": 60000,  # 60 second timeout
                    "viewport": {"width": 1920, "height": 1080}  # Larger viewport
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"[SUCCESS] Fresh crawl initiated")
                print(f"   Task ID: {result.get('task_id', 'Unknown')}")
                
                # Wait for processing
                print(f"[PROCESSING] Waiting 45 seconds for crawl to complete...")
                await asyncio.sleep(45)
                
                # Verify chunks were created
                supabase = create_client(supabase_url, supabase_key)
                
                # Find the new source
                new_source = supabase.table('archon_sources').select('source_id').eq('source_url', source_info['url']).execute()
                
                if new_source.data:
                    new_id = new_source.data[0]['source_id']
                    chunks = supabase.table('archon_crawled_pages').select('id').eq('source_id', new_id).execute()
                    chunk_count = len(chunks.data)
                    
                    if chunk_count > 0:
                        print(f"[VERIFIED] SUCCESS! {chunk_count} chunks created")
                        return True
                    else:
                        print(f"[WARNING] No chunks yet, may still be processing")
                        return False
                else:
                    print(f"[WARNING] Source not found yet, may still be processing")
                    return False
                    
            else:
                print(f"[FAILED] Status {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                return False
                
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            return False

async def main():
    """Fix the final 3 sources"""
    print("\n" + "="*80)
    print("FIXING FINAL 15% - ACHIEVING 100% SUCCESS")
    print("="*80)
    
    successful = 0
    failed = 0
    
    for source in FINAL_SOURCES:
        success = await delete_and_recrawl(source)
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Wait between sources
        if source != FINAL_SOURCES[-1]:
            print("\n[WAIT] Waiting 30 seconds before next source...")
            await asyncio.sleep(30)
    
    print("\n" + "="*80)
    print("FINAL FIX SUMMARY")
    print("="*80)
    print(f"Total sources: {len(FINAL_SOURCES)}")
    print(f"Successfully fixed: {successful}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎯 100% SUCCESS ACHIEVED!")
    else:
        print(f"\n{successful}/{len(FINAL_SOURCES)} fixed. May need manual intervention for remaining.")

if __name__ == "__main__":
    import os
    if not os.getenv('SUPABASE_URL'):
        from dotenv import load_dotenv
        load_dotenv('C:/Jarvis/AI Workspace/Archon/.env')
    
    asyncio.run(main())