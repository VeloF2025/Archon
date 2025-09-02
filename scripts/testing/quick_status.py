#!/usr/bin/env python
"""Quick status check for knowledge base"""

import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv('.env')
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
supabase = create_client(supabase_url, supabase_key)

# Get all sources
sources = supabase.table('archon_sources').select('*').execute()
total_sources = len(sources.data)

# Count sources with and without chunks
sources_with_chunks = 0
sources_without_chunks = 0
problem_sources = []

for source in sources.data:
    source_id = source.get('source_id')
    url = source.get('source_url', '')
    name = source.get('source_display_name', 'Unknown')
    
    chunks = supabase.table('archon_crawled_pages').select('id').eq('source_id', source_id).execute()
    chunk_count = len(chunks.data)
    
    if chunk_count > 0:
        sources_with_chunks += 1
    else:
        sources_without_chunks += 1
        problem_sources.append((name[:40], url[:50]))

success_rate = (sources_with_chunks / total_sources * 100) if total_sources > 0 else 0

print('='*70)
print('KNOWLEDGE BASE STATUS SUMMARY')
print('='*70)
print(f'Total Sources: {total_sources}')
print(f'With Chunks: {sources_with_chunks} ({success_rate:.1f}%)')
print(f'Without Chunks: {sources_without_chunks} ({100-success_rate:.1f}%)')
print('='*70)

if problem_sources:
    print('\nSOURCES WITHOUT CHUNKS:')
    print('-'*70)
    for name, url in problem_sources:
        print(f'- {name:40} | {url}')
    print('='*70)

if success_rate >= 100:
    print('\n🎯 100% SUCCESS ACHIEVED!')
elif success_rate >= 95:
    print(f'\n✅ EXCELLENT: {success_rate:.1f}% success rate!')
elif success_rate >= 85:
    print(f'\n👍 GOOD: {success_rate:.1f}% success rate')
else:
    print(f'\n⚠️ NEEDS ATTENTION: Only {success_rate:.1f}% success rate')