#!/usr/bin/env python3
"""
Manual validation script for chunks count fix

This script validates the implementation without requiring pytest.
It simulates the test conditions to verify the fix works correctly.
"""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_chunks_count_service_exists():
    """Test that ChunksCountService can be imported."""
    try:
        from src.server.services.knowledge.chunks_count_service import ChunksCountService
        print("✅ ChunksCountService import successful")
        return True
    except ImportError as e:
        print(f"❌ ChunksCountService import failed: {e}")
        return False

def test_chunks_count_service_functionality():
    """Test basic ChunksCountService functionality."""
    try:
        from src.server.services.knowledge.chunks_count_service import ChunksCountService
        
        # Create mock client
        mock_client = MagicMock()
        
        # Mock single source count query
        mock_result = MagicMock()
        mock_result.count = 84
        mock_query = MagicMock()
        mock_query.execute.return_value = mock_result
        mock_client.table.return_value.select.return_value.eq.return_value = mock_query
        
        # Create service
        service = ChunksCountService(mock_client)
        
        # Test single source count
        count = service.get_chunks_count("test_source")
        if count == 84:
            print("✅ Single source count test passed")
        else:
            print(f"❌ Single source count test failed: expected 84, got {count}")
            return False
            
        # Test batch count
        mock_batch_result = MagicMock()
        mock_batch_result.data = [
            {"source_id": "source_1", "chunk_count": 45},
            {"source_id": "source_2", "chunk_count": 23}
        ]
        mock_batch_query = MagicMock()
        mock_batch_query.execute.return_value = mock_batch_result
        mock_client.rpc.return_value = mock_batch_query
        
        batch_counts = service.get_bulk_chunks_count(["source_1", "source_2"])
        expected = {"source_1": 45, "source_2": 23}
        if batch_counts == expected:
            print("✅ Bulk count test passed")
        else:
            print(f"❌ Bulk count test failed: expected {expected}, got {batch_counts}")
            return False
            
        # Test caching
        # Call again - should use cache
        cached_count = service.get_chunks_count("test_source")
        if cached_count == 84:
            print("✅ Caching test passed")
        else:
            print(f"❌ Caching test failed: expected 84, got {cached_count}")
            return False
            
        # Check cache stats
        stats = service.get_cache_stats()
        if stats['cache_hits'] > 0:
            print("✅ Cache stats test passed")
        else:
            print(f"❌ Cache stats test failed: {stats}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ ChunksCountService functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_knowledge_item_service_integration():
    """Test KnowledgeItemService integration with ChunksCountService."""
    try:
        from src.server.services.knowledge.knowledge_item_service import KnowledgeItemService
        
        # Create mock client
        mock_client = MagicMock()
        
        # Create service
        service = KnowledgeItemService(mock_client)
        
        # Check that chunks_count_service is initialized
        if hasattr(service, 'chunks_count_service'):
            print("✅ KnowledgeItemService has chunks_count_service")
        else:
            print("❌ KnowledgeItemService missing chunks_count_service")
            return False
            
        # Test _get_chunks_count method integration
        # Mock the chunks count service method
        service.chunks_count_service.get_chunks_count = MagicMock(return_value=42)
        
        import asyncio
        
        async def run_async_test():
            count = await service._get_chunks_count("test_source")
            if count == 42:
                print("✅ KnowledgeItemService._get_chunks_count integration test passed")
                return True
            else:
                print(f"❌ KnowledgeItemService._get_chunks_count integration test failed: expected 42, got {count}")
                return False
        
        return asyncio.run(run_async_test())
        
    except Exception as e:
        print(f"❌ KnowledgeItemService integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_database_table_usage():
    """Validate that we're using the correct database table."""
    try:
        from src.server.services.knowledge.chunks_count_service import ChunksCountService
        
        # Create mock client to capture table calls
        mock_client = MagicMock()
        
        # Mock result
        mock_result = MagicMock()
        mock_result.count = 10
        mock_query = MagicMock()
        mock_query.execute.return_value = mock_result
        mock_client.table.return_value.select.return_value.eq.return_value = mock_query
        
        # Create service and call method
        service = ChunksCountService(mock_client)
        service.get_chunks_count("test_source")
        
        # Verify that the correct table was called
        mock_client.table.assert_called_with("archon_documents")
        print("✅ Using correct table: archon_documents (not archon_crawled_pages)")
        return True
        
    except Exception as e:
        print(f"❌ Database table validation failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🧪 Validating Chunks Count Fix Implementation")
    print("=" * 50)
    
    tests = [
        ("ChunksCountService Import", test_chunks_count_service_exists),
        ("ChunksCountService Functionality", test_chunks_count_service_functionality),
        ("KnowledgeItemService Integration", test_knowledge_item_service_integration),
        ("Database Table Usage", validate_database_table_usage)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Implementation is working correctly.")
        print("\n✅ Key fixes verified:")
        print("  - ChunksCountService created and functional")
        print("  - Uses archon_documents table (correct)")
        print("  - Caching implemented and working")
        print("  - KnowledgeItemService integration complete")
        print("  - Batch operations supported")
        return True
    else:
        print("⚠️  Some tests failed. Check implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)