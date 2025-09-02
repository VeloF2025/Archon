#!/usr/bin/env python
"""Check the chunking status of all knowledge base sources"""

import asyncio
import os
from supabase import create_client
from datetime import datetime

async def check_knowledge_base():
    # Get Supabase credentials from environment
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
    
    if not supabase_url or not supabase_key:
        print('ERROR: Missing Supabase credentials in environment')
        return
    
    # Create Supabase client
    supabase = create_client(supabase_url, supabase_key)
    
    # Get all sources
    sources_response = supabase.table('archon_sources').select('*').execute()
    sources = sources_response.data
    
    print(f'=' * 80)
    print(f'KNOWLEDGE BASE CHUNKING STATUS REPORT')
    print(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'=' * 80)
    print(f'Total sources in knowledge base: {len(sources)}')
    print(f'=' * 80)
    
    # Statistics
    total_chunks = 0
    sources_with_chunks = 0
    sources_without_chunks = 0
    sources_with_embeddings = 0
    sources_without_embeddings = 0
    problem_sources = []
    
    # Check each source for chunks
    for idx, source in enumerate(sources, 1):
        source_id = source.get('source_id') or source.get('id')
        if not source_id:
            print(f'WARNING: Source {idx} has no ID field')
            continue
            
        source_url = source.get('source_url', 'N/A')
        display_name = source.get('source_display_name', 'Unnamed')
        source_type = source.get('source_type', 'unknown')
        created_at = source.get('created_at', 'Unknown')
        
        # Count chunks for this source - all use crawled_pages table
        chunks_response = supabase.table('archon_crawled_pages').select('*').eq('source_id', source_id).execute()
        table_name = 'archon_crawled_pages'
        
        chunk_count = len(chunks_response.data)
        total_chunks += chunk_count
        
        # Check for embeddings
        has_embedding = False
        embedding_count = 0
        
        if chunk_count > 0:
            sources_with_chunks += 1
            # Check how many chunks have embeddings
            for chunk in chunks_response.data:
                if chunk.get('embedding'):
                    embedding_count += 1
            
            has_embedding = embedding_count > 0
            if has_embedding:
                sources_with_embeddings += 1
            else:
                sources_without_embeddings += 1
        else:
            sources_without_chunks += 1
            problem_sources.append({
                'name': display_name,
                'type': source_type,
                'id': source_id,
                'url': source_url,
                'created': created_at
            })
        
        # Determine status
        if chunk_count == 0:
            status = '❌ NO CHUNKS'
            status_color = 'CRITICAL'
        elif embedding_count == 0:
            status = '⚠️ NO EMBEDDINGS'
            status_color = 'WARNING'
        elif embedding_count < chunk_count:
            status = f'⚠️ PARTIAL EMBEDDINGS ({embedding_count}/{chunk_count})'
            status_color = 'WARNING'
        else:
            status = '✅ FULLY PROCESSED'
            status_color = 'OK'
        
        print(f'\n[{idx}/{len(sources)}] {display_name[:50]}')
        print(f'  Type: {source_type} | Table: {table_name}')
        print(f'  URL: {source_url[:70]}...' if len(str(source_url)) > 70 else f'  URL: {source_url}')
        print(f'  Created: {created_at}')
        print(f'  Chunks: {chunk_count} | Embeddings: {embedding_count}')
        print(f'  Status: {status}')
        print(f'  Source ID: {source_id}')
    
    # Print summary
    print(f'\n{"=" * 80}')
    print(f'SUMMARY')
    print(f'{"=" * 80}')
    print(f'Total Sources: {len(sources)}')
    print(f'Total Chunks: {total_chunks}')
    print(f'Sources with chunks: {sources_with_chunks} ({sources_with_chunks/len(sources)*100:.1f}%)')
    print(f'Sources without chunks: {sources_without_chunks} ({sources_without_chunks/len(sources)*100:.1f}%)')
    print(f'Sources with embeddings: {sources_with_embeddings}')
    print(f'Sources without embeddings: {sources_without_embeddings}')
    
    if problem_sources:
        print(f'\n{"=" * 80}')
        print(f'PROBLEM SOURCES (No Chunks)')
        print(f'{"=" * 80}')
        for source in problem_sources:
            print(f'\n- {source["name"]}')
            print(f'  Type: {source["type"]}')
            print(f'  URL: {source["url"][:70]}...' if len(source["url"]) > 70 else f'  URL: {source["url"]}')
            print(f'  Created: {source["created"]}')
            print(f'  ID: {source["id"]}')
    
    print(f'\n{"=" * 80}')
    print('RECOMMENDATIONS')
    print(f'{"=" * 80}')
    
    if sources_without_chunks > 0:
        print(f'⚠️ {sources_without_chunks} sources need re-processing/chunking')
        print('  - These sources may have failed during initial upload')
        print('  - Consider re-crawling or re-uploading these sources')
    
    if sources_without_embeddings > 0:
        print(f'⚠️ {sources_without_embeddings} sources need embedding generation')
        print('  - Embeddings are required for semantic search')
        print('  - Run embedding generation for these sources')
    
    if sources_without_chunks == 0 and sources_without_embeddings == 0:
        print('✅ All sources are properly chunked and embedded!')

if __name__ == '__main__':
    asyncio.run(check_knowledge_base())