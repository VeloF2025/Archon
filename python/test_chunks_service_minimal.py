#!/usr/bin/env python3
"""
Minimal validation script for chunks count fix
Tests only the ChunksCountService in isolation.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Read and execute the ChunksCountService code directly
chunks_service_path = Path(__file__).parent / "src/server/services/knowledge/chunks_count_service.py"

def test_chunks_count_service():
    """Test ChunksCountService by reading and executing the code directly."""
    try:
        # Read the service code
        with open(chunks_service_path, 'r') as f:
            service_code = f.read()
        
        # Create a minimal environment for execution
        service_env = {}
        
        # Mock the logging imports
        mock_logfire = MagicMock()
        service_env['safe_logfire_error'] = mock_logfire
        service_env['safe_logfire_info'] = mock_logfire
        
        # Execute the service code in our environment
        exec(service_code, service_env)
        
        # Get the ChunksCountService class
        ChunksCountService = service_env['ChunksCountService']
        
        print("✅ ChunksCountService class loaded successfully")
        
        # Test basic functionality
        mock_client = MagicMock()
        
        # Test 1: Single source count
        mock_result = MagicMock()
        mock_result.count = 84
        mock_query = MagicMock()
        mock_query.execute.return_value = mock_result
        mock_client.table.return_value.select.return_value.eq.return_value = mock_query
        
        service = ChunksCountService(mock_client)
        count = service.get_chunks_count("test_source")
        
        if count == 84:
            print("✅ Single source count: PASSED")
        else:
            print(f"❌ Single source count: FAILED (expected 84, got {count})")
            return False
        
        # Verify correct table is called
        mock_client.table.assert_called_with("archon_documents")
        print("✅ Correct table (archon_documents): VERIFIED")
        
        # Test 2: Bulk count
        mock_bulk_result = MagicMock()
        mock_bulk_result.data = [
            {"source_id": "source_1", "chunk_count": 45},
            {"source_id": "source_2", "chunk_count": 23}
        ]
        mock_bulk_query = MagicMock()
        mock_bulk_query.execute.return_value = mock_bulk_result
        mock_client.rpc.return_value = mock_bulk_query
        
        bulk_counts = service.get_bulk_chunks_count(["source_1", "source_2"])
        expected_bulk = {"source_1": 45, "source_2": 23}
        
        if bulk_counts == expected_bulk:
            print("✅ Bulk count operation: PASSED")
        else:
            print(f"❌ Bulk count operation: FAILED (expected {expected_bulk}, got {bulk_counts})")
            return False
        
        # Verify RPC call
        mock_client.rpc.assert_called_with('get_chunks_count_bulk', {'source_ids': ['source_1', 'source_2']})
        print("✅ Bulk RPC call: VERIFIED")
        
        # Test 3: Caching
        # Call the same source again - should hit cache
        cached_count = service.get_chunks_count("test_source")
        if cached_count == 84:
            print("✅ Cache hit: PASSED")
        else:
            print(f"❌ Cache hit: FAILED (expected 84, got {cached_count})")
            return False
        
        # Check cache stats
        stats = service.get_cache_stats()
        if stats['cache_hits'] > 0:
            print(f"✅ Cache statistics: PASSED (hit rate: {stats['hit_rate_percent']}%)")
        else:
            print(f"❌ Cache statistics: FAILED ({stats})")
            return False
        
        # Test 4: Error handling
        mock_client.table.return_value.select.return_value.eq.return_value.execute.side_effect = Exception("DB Error")
        
        try:
            service.get_chunks_count("error_source")
            print("❌ Error handling: FAILED (should have raised exception)")
            return False
        except Exception as e:
            if "DB Error" in str(e):
                print("✅ Error handling: PASSED")
            else:
                print(f"❌ Error handling: FAILED (wrong exception: {e})")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_knowledge_service_changes():
    """Validate that KnowledgeItemService has been updated correctly."""
    try:
        knowledge_service_path = Path(__file__).parent / "src/server/services/knowledge/knowledge_item_service.py"
        
        with open(knowledge_service_path, 'r') as f:
            content = f.read()
        
        # Check for key changes
        checks = [
            ("ChunksCountService import", "from .chunks_count_service import ChunksCountService"),
            ("ChunksCountService initialization", "self.chunks_count_service = ChunksCountService"),
            ("Batch chunks counting", "get_bulk_chunks_count(source_ids)"),
            ("Service usage in _get_chunks_count", "self.chunks_count_service.get_chunks_count")
        ]
        
        all_passed = True
        for check_name, expected_text in checks:
            if expected_text in content:
                print(f"✅ {check_name}: FOUND")
            else:
                print(f"❌ {check_name}: MISSING")
                all_passed = False
        
        # Check that old hardcoded logic is removed
        if "chunk_counts[source_id] = 0" in content:
            print("⚠️  Old hardcoded logic still present - review needed")
            
        if "archon_crawled_pages" in content:
            # Check if it's in the old _get_chunks_count method
            if content.count("archon_crawled_pages") > 1:  # Allow one reference in URLs query
                print("⚠️  archon_crawled_pages still used for counting - review needed")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Knowledge service validation failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🧪 Minimal Chunks Count Fix Validation")
    print("=" * 50)
    
    print("\n🔍 Testing ChunksCountService...")
    service_test_passed = test_chunks_count_service()
    
    print("\n🔍 Validating KnowledgeItemService changes...")
    knowledge_test_passed = validate_knowledge_service_changes()
    
    print("\n" + "=" * 50)
    
    if service_test_passed and knowledge_test_passed:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("\n✅ Implementation Summary:")
        print("  - ChunksCountService: Created and functional")
        print("  - Database table: Fixed to use archon_documents")
        print("  - Caching: Implemented and working")
        print("  - Batch operations: Supported")
        print("  - Error handling: Proper exception propagation")
        print("  - KnowledgeItemService: Updated to use new service")
        print("\n🚀 Ready for production deployment!")
        return True
    else:
        print("⚠️  Some validation checks failed.")
        print("📋 Next steps:")
        if not service_test_passed:
            print("  - Fix ChunksCountService implementation")
        if not knowledge_test_passed:
            print("  - Complete KnowledgeItemService integration")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)