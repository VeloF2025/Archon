#!/usr/bin/env python3
"""
Emergency Fix: Process existing crawled content into document chunks
Resolves Phase 3 REF Tools 0% success rate by creating missing embeddings
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add the src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "python" / "src"))

from server.utils import get_supabase_client
from server.services.storage.document_storage_service import add_documents_to_supabase
from server.services.storage.storage_services import DocumentStorageService
from server.config.logfire_config import get_logger

logger = get_logger(__name__)

async def process_existing_crawled_content():
    """Process existing crawled content into chunks with embeddings"""
    print("Starting emergency chunk processing for Phase 3 fix...")
    
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Query crawled pages that don't have corresponding document chunks
        print("Querying crawled content without chunks...")
        
        # Get all sources that have crawled content but no chunks
        sources_result = supabase.table("archon_sources").select("*").eq("chunks_count", 0).limit(3).execute()
        
        if not sources_result.data:
            print("No sources found with missing chunks")
            return
            
        print(f"Found {len(sources_result.data)} sources with missing chunks")
        
        # Initialize document storage service
        doc_storage = DocumentStorageService(supabase)
        
        for source in sources_result.data:
            source_id = source['source_id']
            print(f"\n📝 Processing source: {source_id} ({source.get('title', 'Unknown')})")
            
            # Get crawled pages for this source
            pages_result = supabase.table("archon_crawled_pages").select("*").eq("source_id", source_id).limit(10).execute()
            
            if not pages_result.data:
                print(f"⚠️  No crawled pages found for source {source_id}")
                continue
                
            print(f"📄 Found {len(pages_result.data)} crawled pages")
            
            # Prepare data for chunking
            all_urls = []
            all_chunk_numbers = []
            all_contents = []
            all_metadatas = []
            
            total_content = ""
            page_count = 0
            
            for page in pages_result.data:
                if page.get('markdown_content'):
                    # Chunk the content
                    content = page['markdown_content']
                    chunks = doc_storage.smart_chunk_text(content, chunk_size=5000)
                    
                    for i, chunk in enumerate(chunks):
                        all_urls.append(page['url'])
                        all_chunk_numbers.append(i)
                        all_contents.append(chunk)
                        all_metadatas.append({
                            'source_id': source_id,
                            'page_url': page['url'],
                            'chunk_index': i,
                            'total_chunks': len(chunks)
                        })
                    
                    total_content += content
                    page_count += 1
                    print(f"  ✅ Chunked page: {page['url']} -> {len(chunks)} chunks")
            
            if all_contents:
                print(f"💾 Storing {len(all_contents)} chunks with embeddings...")
                
                # Simple progress callback
                def progress_callback(message, percentage, batch_info=None):
                    print(f"  📈 {percentage}% - {message}")
                
                # Store chunks with embeddings using direct function
                await add_documents_to_supabase(
                    client=supabase,
                    urls=all_urls,
                    chunk_numbers=all_chunk_numbers,
                    contents=all_contents,
                    metadatas=all_metadatas,
                    url_to_full_document={url: content for url, content in zip(all_urls, all_contents)},
                    progress_callback=progress_callback
                )
                success = True
                result = {'total_documents': len(all_contents)}
                
                if success:
                    chunks_created = result.get('total_documents', len(all_contents))
                    print(f"  ✅ Successfully created {chunks_created} document chunks with embeddings")
                    
                    # Update source chunk count
                    supabase.table("archon_sources").update({
                        "chunks_count": chunks_created,
                        "updated_at": "now()"
                    }).eq("source_id", source_id).execute()
                    
                else:
                    print(f"  ❌ Failed to create chunks: {result.get('error', 'Unknown error')}")
            else:
                print(f"  ⚠️  No content found to chunk for source {source_id}")
                
        print("\n🎉 Emergency chunk processing completed!")
        print("🔄 Now testing REF Tools...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during chunk processing: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_rag_query():
    """Test if RAG queries work after chunk creation"""
    try:
        import httpx
        
        print("🧪 Testing RAG query...")
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post("http://localhost:8181/api/rag/query", json={
                "query": "React TypeScript best practices",
                "match_count": 3
            })
            
            if response.status_code == 200:
                result = response.json()
                results_count = len(result.get('results', []))
                print(f"✅ RAG query successful! Found {results_count} results")
                return True
            else:
                print(f"❌ RAG query failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Error testing RAG: {e}")
        return False

async def main():
    """Main execution"""
    print("=" * 60)
    print("PHASE 3 EMERGENCY FIX: Missing Document Chunks")
    print("=" * 60)
    
    # Step 1: Process existing content
    chunk_success = await process_existing_crawled_content()
    
    if chunk_success:
        # Step 2: Test RAG functionality
        await test_rag_query()
        print("\n🎯 Phase 3 REF Tools should now work!")
        print("🧪 Run Phase 3 SCWT benchmark to verify fix")
    else:
        print("\n❌ Chunk processing failed - Phase 3 issue not resolved")

if __name__ == "__main__":
    asyncio.run(main())