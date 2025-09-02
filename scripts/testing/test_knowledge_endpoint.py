#!/usr/bin/env python
"""Test knowledge-items endpoint performance"""

import httpx
import asyncio
import time

async def test_endpoint():
    print('Testing knowledge-items endpoint performance...')
    
    start = time.time()
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                'http://localhost:8181/api/knowledge-items',
                params={'page': 1, 'per_page': 100}
            )
            elapsed = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                print(f'[SUCCESS] Response in {elapsed:.2f}s')
                print(f'Total items: {data.get("total", 0)}')
                print(f'Items returned: {len(data.get("data", []))}')
                
                # Check if there are embeddings being loaded
                if data.get("data"):
                    first_item = data["data"][0]
                    has_embedding = "embedding" in first_item
                    print(f'Items include embeddings: {has_embedding}')
                    if has_embedding and first_item.get("embedding"):
                        print(f'Embedding size: {len(first_item["embedding"])} dimensions')
            else:
                print(f'[FAILED] Status {response.status_code} in {elapsed:.2f}s')
                print(f'Response: {response.text[:200]}')
                
        except httpx.TimeoutException:
            elapsed = time.time() - start
            print(f'[TIMEOUT] After {elapsed:.2f}s')
        except Exception as e:
            elapsed = time.time() - start
            print(f'[ERROR] After {elapsed:.2f}s: {str(e)}')

if __name__ == "__main__":
    asyncio.run(test_endpoint())