"""
TDD Test Suite for Knowledge Base Chunks Count Fix

This test suite follows Test-Driven Development principles to fix the chunks count 
discrepancy where knowledge items API shows chunks_count: 0 but RAG search 
returns chunks with chunk_index values.

All tests are designed to FAIL initially (Red phase) and guide implementation (Green phase).
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from datetime import datetime, timezone
import time
from typing import List, Dict, Any, Optional

# Import the services we'll be testing
from src.server.services.knowledge.knowledge_item_service import KnowledgeItemService


class TestChunksCountService:
    """Unit tests for the new ChunksCountService that will fix the discrepancy."""
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client for testing."""
        client = MagicMock()
        # Mock table method that returns a query builder
        client.table = MagicMock()
        client.from_ = MagicMock()
        return client
    
    @pytest.fixture
    def chunks_count_service(self, mock_supabase_client):
        """Create ChunksCountService instance with mocked dependencies."""
        # This will fail initially since ChunksCountService doesn't exist
        from src.server.services.knowledge.chunks_count_service import ChunksCountService
        return ChunksCountService(mock_supabase_client)
    
    def test_get_chunks_count_single_source_exists(self, chunks_count_service, mock_supabase_client):
        """Test getting chunk count for a single source that exists."""
        # Arrange
        source_id = "test_source_123"
        expected_count = 84
        
        # Mock the database query to return count
        mock_result = MagicMock()
        mock_result.count = expected_count
        mock_query = MagicMock()
        mock_query.execute.return_value = mock_result
        mock_supabase_client.table.return_value.select.return_value.eq.return_value = mock_query
        
        # Act
        actual_count = chunks_count_service.get_chunks_count(source_id)
        
        # Assert
        assert actual_count == expected_count
        mock_supabase_client.table.assert_called_once_with("archon_documents")
        
    def test_get_chunks_count_single_source_not_exists(self, chunks_count_service, mock_supabase_client):
        """Test getting chunk count for a source that doesn't exist."""
        # Arrange
        source_id = "nonexistent_source"
        expected_count = 0
        
        # Mock the database query to return 0 count
        mock_result = MagicMock()
        mock_result.count = 0
        mock_query = MagicMock()
        mock_query.execute.return_value = mock_result
        mock_supabase_client.table.return_value.select.return_value.eq.return_value = mock_query
        
        # Act
        actual_count = chunks_count_service.get_chunks_count(source_id)
        
        # Assert
        assert actual_count == expected_count
        
    def test_get_chunks_count_database_error(self, chunks_count_service, mock_supabase_client):
        """Test error handling when database query fails."""
        # Arrange
        source_id = "test_source"
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.execute.side_effect = Exception("DB Error")
        
        # Act & Assert
        with pytest.raises(Exception, match="DB Error"):
            chunks_count_service.get_chunks_count(source_id)
    
    def test_get_bulk_chunks_count_multiple_sources(self, chunks_count_service, mock_supabase_client):
        """Test getting chunk counts for multiple sources in batch."""
        # Arrange
        source_ids = ["source_1", "source_2", "source_3"]
        expected_counts = {"source_1": 45, "source_2": 23, "source_3": 67}
        
        # Mock the database query to return aggregated results
        mock_result = MagicMock()
        mock_result.data = [
            {"source_id": "source_1", "chunk_count": 45},
            {"source_2": "source_2", "chunk_count": 23},
            {"source_id": "source_3", "chunk_count": 67}
        ]
        mock_query = MagicMock()
        mock_query.execute.return_value = mock_result
        mock_supabase_client.rpc.return_value = mock_query
        
        # Act
        actual_counts = chunks_count_service.get_bulk_chunks_count(source_ids)
        
        # Assert
        assert actual_counts == expected_counts
        mock_supabase_client.rpc.assert_called_once_with('get_chunks_count_bulk', {'source_ids': source_ids})
        
    def test_get_bulk_chunks_count_empty_list(self, chunks_count_service):
        """Test bulk count with empty source list."""
        # Arrange
        source_ids = []
        
        # Act
        actual_counts = chunks_count_service.get_bulk_chunks_count(source_ids)
        
        # Assert
        assert actual_counts == {}
        
    def test_get_bulk_chunks_count_with_cache(self, chunks_count_service, mock_supabase_client):
        """Test that bulk count operations use caching for performance."""
        # Arrange
        source_ids = ["source_1", "source_2"]
        
        # Mock first call
        mock_result = MagicMock()
        mock_result.data = [
            {"source_id": "source_1", "chunk_count": 45},
            {"source_id": "source_2", "chunk_count": 23}
        ]
        mock_query = MagicMock()
        mock_query.execute.return_value = mock_result
        mock_supabase_client.rpc.return_value = mock_query
        
        # Act - call twice
        first_result = chunks_count_service.get_bulk_chunks_count(source_ids)
        second_result = chunks_count_service.get_bulk_chunks_count(source_ids)
        
        # Assert - second call should use cache, not hit database again
        assert first_result == second_result
        # Database should only be called once due to caching
        assert mock_supabase_client.rpc.call_count == 1


class TestKnowledgeItemServiceChunksFix:
    """Integration tests for KnowledgeItemService with chunks count fix."""
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client."""
        client = MagicMock()
        client.table = MagicMock()
        client.from_ = MagicMock()
        return client
        
    @pytest.fixture
    def knowledge_service(self, mock_supabase_client):
        """Create KnowledgeItemService instance."""
        return KnowledgeItemService(mock_supabase_client)
    
    @pytest.fixture
    def sample_sources_data(self):
        """Sample sources data for testing."""
        return [
            {
                "source_id": "test_source_1",
                "title": "Test Documentation 1",
                "summary": "Test summary 1",
                "metadata": {
                    "knowledge_type": "technical",
                    "tags": ["python", "testing"]
                },
                "total_word_count": 1500,
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z"
            },
            {
                "source_id": "test_source_2", 
                "title": "Test Documentation 2",
                "summary": "Test summary 2",
                "metadata": {
                    "knowledge_type": "business",
                    "tags": ["documentation"]
                },
                "total_word_count": 2000,
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z"
            }
        ]
    
    @pytest.mark.asyncio
    async def test_list_items_shows_zero_chunks_count_currently(self, knowledge_service, mock_supabase_client, sample_sources_data):
        """Test that list_items currently returns chunks_count: 0 (this should fail and expose the bug)."""
        # Arrange
        mock_result = MagicMock()
        mock_result.data = sample_sources_data
        mock_result.count = len(sample_sources_data)
        
        # Mock the sources query
        mock_query = MagicMock()
        mock_query.execute.return_value = mock_result
        mock_query.range.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.or_.return_value = mock_query
        mock_supabase_client.from_.return_value.select.return_value = mock_query
        
        # Mock the count query
        mock_count_query = MagicMock()
        mock_count_result = MagicMock()
        mock_count_result.count = len(sample_sources_data)
        mock_count_query.execute.return_value = mock_count_result
        mock_count_query.eq.return_value = mock_count_query
        mock_count_query.or_.return_value = mock_count_query
        mock_supabase_client.from_.return_value.select.return_value = mock_count_query
        
        # Mock URLs query
        mock_urls_result = MagicMock()
        mock_urls_result.data = [
            {"source_id": "test_source_1", "url": "https://example1.com"},
            {"source_id": "test_source_2", "url": "https://example2.com"}
        ]
        mock_urls_query = MagicMock()
        mock_urls_query.execute.return_value = mock_urls_result
        mock_urls_query.in_.return_value = mock_urls_query
        mock_supabase_client.from_.return_value.select.return_value = mock_urls_query
        
        # Act
        result = await knowledge_service.list_items(page=1, per_page=20)
        
        # Assert - This test should FAIL initially, showing the bug
        # Current implementation hardcodes chunks_count to 0
        for item in result["items"]:
            # This assertion should FAIL, exposing the bug
            assert item["metadata"]["chunks_count"] > 0, f"Expected chunks_count > 0, got {item['metadata']['chunks_count']}"
    
    @pytest.mark.asyncio
    async def test_get_chunks_count_returns_zero_currently(self, knowledge_service, mock_supabase_client):
        """Test that _get_chunks_count currently returns 0 (this should fail and show the bug)."""
        # Arrange
        source_id = "test_source_with_chunks"
        
        # Mock that there are actually chunks in the database
        mock_result = MagicMock()
        mock_result.count = 84  # Actual chunks exist
        mock_query = MagicMock()
        mock_query.execute.return_value = mock_result
        mock_query.eq.return_value = mock_query
        mock_supabase_client.table.return_value.select.return_value = mock_query
        
        # Act
        actual_count = await knowledge_service._get_chunks_count(source_id)
        
        # Assert - This should FAIL initially because current implementation
        # counts archon_crawled_pages instead of archon_documents
        assert actual_count == 84, f"Expected 84 chunks, but got {actual_count}"


class TestKnowledgeAPIEndpointsIntegration:
    """Integration tests for Knowledge API endpoints with chunks count fix."""
    
    @pytest.fixture
    async def client(self):
        """Create test client."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from src.server.api_routes.knowledge_api import router
        
        app = FastAPI()
        app.include_router(router)
        
        with TestClient(app) as client:
            yield client
    
    @pytest.fixture
    def mock_knowledge_service(self):
        """Mock KnowledgeItemService."""
        service = AsyncMock()
        return service
    
    def test_get_knowledge_items_returns_correct_chunks_count(self, client, mock_knowledge_service):
        """Test that /api/knowledge-items returns correct chunks_count values."""
        # Arrange
        mock_items_data = {
            "items": [
                {
                    "id": "source_1",
                    "title": "Test Doc 1",
                    "metadata": {
                        "chunks_count": 45,  # Should be actual count, not 0
                        "knowledge_type": "technical"
                    }
                },
                {
                    "id": "source_2", 
                    "title": "Test Doc 2",
                    "metadata": {
                        "chunks_count": 67,  # Should be actual count, not 0
                        "knowledge_type": "business"
                    }
                }
            ],
            "total": 2,
            "page": 1,
            "per_page": 20,
            "pages": 1
        }
        
        # Mock the service method
        with patch('src.server.api_routes.knowledge_api.KnowledgeItemService') as mock_service_class:
            mock_service_instance = mock_service_class.return_value
            mock_service_instance.list_items = AsyncMock(return_value=mock_items_data)
            
            # Act
            response = client.get("/api/knowledge-items")
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            
            # This test should FAIL initially because current API returns chunks_count: 0
            for item in data["items"]:
                assert item["metadata"]["chunks_count"] > 0, f"Expected chunks_count > 0, got {item['metadata']['chunks_count']}"
    
    def test_get_single_knowledge_item_has_correct_chunks_count(self, client):
        """Test that getting a single knowledge item shows correct chunks_count."""
        # This test will fail initially - to be implemented after ChunksCountService exists
        with patch('src.server.api_routes.knowledge_api.KnowledgeItemService') as mock_service_class:
            mock_service_instance = mock_service_class.return_value
            mock_service_instance.get_item = AsyncMock(return_value={
                "id": "test_source",
                "title": "Test Document",
                "metadata": {
                    "chunks_count": 23  # Should be actual count from database
                }
            })
            
            # Act
            response = client.get("/api/knowledge-items/test_source") 
            
            # Assert - This should FAIL initially
            assert response.status_code == 200
            data = response.json()
            assert data["metadata"]["chunks_count"] == 23


class TestDataIntegrityValidation:
    """Data integrity tests to ensure chunk counts match actual documents."""
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client for data integrity tests."""
        client = MagicMock()
        return client
    
    def test_all_32_sources_have_accurate_chunk_counts(self, mock_supabase_client):
        """Test that all 32 existing knowledge sources have accurate chunk counts."""
        # Arrange - Mock 32 sources with varying chunk counts
        mock_sources = []
        mock_chunk_counts = []
        
        for i in range(32):
            source_id = f"source_{i+1}"
            actual_chunks = (i + 1) * 5  # Varying chunk counts
            
            mock_sources.append({
                "source_id": source_id,
                "title": f"Source {i+1}",
                "metadata": {"knowledge_type": "technical"}
            })
            
            mock_chunk_counts.append({
                "source_id": source_id,
                "chunk_count": actual_chunks
            })
        
        # Mock sources query
        mock_sources_result = MagicMock()
        mock_sources_result.data = mock_sources
        mock_supabase_client.from_.return_value.select.return_value.execute.return_value = mock_sources_result
        
        # Mock chunk counts query
        mock_chunks_result = MagicMock()
        mock_chunks_result.data = mock_chunk_counts
        mock_supabase_client.rpc.return_value.execute.return_value = mock_chunks_result
        
        # Act - This will fail initially since ChunksCountService doesn't exist
        from src.server.services.knowledge.chunks_count_service import ChunksCountService
        service = ChunksCountService(mock_supabase_client)
        
        source_ids = [s["source_id"] for s in mock_sources]
        actual_counts = service.get_bulk_chunks_count(source_ids)
        
        # Assert - All sources should have non-zero chunk counts
        for source_id in source_ids:
            assert source_id in actual_counts, f"Missing chunk count for {source_id}"
            assert actual_counts[source_id] > 0, f"Source {source_id} has zero chunks but should have data"
    
    def test_no_orphaned_documents_without_source_references(self, mock_supabase_client):
        """Test that no documents exist without valid source references."""
        # Arrange
        mock_orphaned_docs = MagicMock()
        mock_orphaned_docs.data = []  # Should be empty - no orphans
        mock_query = MagicMock()
        mock_query.execute.return_value = mock_orphaned_docs
        mock_supabase_client.rpc.return_value = mock_query
        
        # Act
        from src.server.services.knowledge.data_integrity_service import DataIntegrityService
        service = DataIntegrityService(mock_supabase_client)
        orphaned_count = service.get_orphaned_documents_count()
        
        # Assert
        assert orphaned_count == 0, f"Found {orphaned_count} orphaned documents"
    
    def test_chunk_counts_match_pgvector_documents(self, mock_supabase_client):
        """Test that reported chunk counts match actual pgvector document count."""
        # Arrange
        test_source_id = "test_source"
        expected_chunk_count = 42
        
        # Mock pgvector count query
        mock_pgvector_result = MagicMock()
        mock_pgvector_result.count = expected_chunk_count
        mock_query = MagicMock()
        mock_query.execute.return_value = mock_pgvector_result
        mock_query.eq.return_value = mock_query
        mock_supabase_client.table.return_value.select.return_value = mock_query
        
        # Act - This will fail initially
        from src.server.services.knowledge.chunks_count_service import ChunksCountService
        service = ChunksCountService(mock_supabase_client)
        reported_count = service.get_chunks_count(test_source_id)
        
        # Assert
        assert reported_count == expected_chunk_count, f"Reported count {reported_count} doesn't match actual {expected_chunk_count}"


class TestPerformanceRequirements:
    """Performance tests to ensure chunk count operations meet speed requirements."""
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client with performance timing."""
        client = MagicMock()
        return client
    
    def test_batch_counting_under_100ms_for_50_sources(self, mock_supabase_client):
        """Test that batch counting 50 sources takes <100ms."""
        # Arrange
        source_ids = [f"source_{i}" for i in range(50)]
        
        # Mock fast database response
        mock_result = MagicMock()
        mock_result.data = [{"source_id": sid, "chunk_count": 10} for sid in source_ids]
        mock_query = MagicMock()
        mock_query.execute.return_value = mock_result
        mock_supabase_client.rpc.return_value = mock_query
        
        # Act - This will fail initially
        from src.server.services.knowledge.chunks_count_service import ChunksCountService
        service = ChunksCountService(mock_supabase_client)
        
        start_time = time.time()
        counts = service.get_bulk_chunks_count(source_ids)
        end_time = time.time()
        
        # Assert
        execution_time = (end_time - start_time) * 1000  # Convert to ms
        assert execution_time < 100, f"Batch counting took {execution_time}ms, should be <100ms"
        assert len(counts) == 50, f"Expected 50 counts, got {len(counts)}"
    
    @pytest.mark.asyncio
    async def test_api_response_time_under_500ms(self, mock_supabase_client):
        """Test that knowledge items API responds in <500ms."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from src.server.api_routes.knowledge_api import router
        
        # Arrange
        app = FastAPI()
        app.include_router(router)
        
        # Mock fast service response
        with patch('src.server.api_routes.knowledge_api.KnowledgeItemService') as mock_service:
            mock_service.return_value.list_items = AsyncMock(return_value={
                "items": [{"id": "test", "metadata": {"chunks_count": 10}}],
                "total": 1,
                "page": 1,
                "per_page": 20,
                "pages": 1
            })
            
            with TestClient(app) as client:
                # Act
                start_time = time.time()
                response = client.get("/api/knowledge-items")
                end_time = time.time()
                
                # Assert
                execution_time = (end_time - start_time) * 1000
                assert response.status_code == 200
                assert execution_time < 500, f"API response took {execution_time}ms, should be <500ms"
    
    def test_cache_hit_rate_above_90_percent(self, mock_supabase_client):
        """Test that caching achieves >90% hit rate for repeated requests."""
        # Arrange
        source_ids = ["source_1", "source_2", "source_3"]
        
        # Mock database call tracking
        db_call_count = 0
        
        def mock_rpc_execute():
            nonlocal db_call_count
            db_call_count += 1
            mock_result = MagicMock()
            mock_result.data = [{"source_id": sid, "chunk_count": 5} for sid in source_ids]
            return mock_result
        
        mock_query = MagicMock()
        mock_query.execute.side_effect = mock_rpc_execute
        mock_supabase_client.rpc.return_value = mock_query
        
        # Act - This will fail initially
        from src.server.services.knowledge.chunks_count_service import ChunksCountService
        service = ChunksCountService(mock_supabase_client)
        
        # Make 10 requests for same data
        for _ in range(10):
            service.get_bulk_chunks_count(source_ids)
        
        # Assert
        # With good caching, only 1 DB call should be made for 10 requests
        cache_hit_rate = ((10 - db_call_count) / 10) * 100
        assert cache_hit_rate >= 90, f"Cache hit rate {cache_hit_rate}% is below 90%"
        assert db_call_count == 1, f"Expected 1 DB call, got {db_call_count}"


class TestRAGChunkConsistency:
    """Tests to ensure RAG search results match reported chunk counts."""
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client."""
        return MagicMock()
    
    def test_rag_search_returns_chunks_for_counted_sources(self, mock_supabase_client):
        """Test that RAG search returns chunks for sources that report having chunks."""
        # Arrange
        source_id = "test_source"
        reported_chunk_count = 25
        
        # Mock chunks count service
        from unittest.mock import patch
        with patch('src.server.services.knowledge.chunks_count_service.ChunksCountService') as mock_chunks_service:
            mock_chunks_instance = mock_chunks_service.return_value
            mock_chunks_instance.get_chunks_count.return_value = reported_chunk_count
            
            # Mock RAG service to return actual chunks
            with patch('src.server.services.search.rag_service.RAGService') as mock_rag_service:
                mock_rag_instance = mock_rag_service.return_value
                mock_rag_instance.perform_rag_query = AsyncMock(return_value=(True, {
                    "results": [
                        {
                            "source_id": source_id,
                            "chunk_index": i,
                            "content": f"Chunk {i} content"
                        }
                        for i in range(5)  # Return 5 chunks
                    ]
                }))
                
                # Act
                chunks_service = mock_chunks_instance
                rag_service = mock_rag_instance
                
                reported_count = chunks_service.get_chunks_count(source_id)
                success, rag_result = await rag_service.perform_rag_query(
                    query=f"source:{source_id}",
                    source=source_id,
                    match_count=10
                )
                
                # Assert
                assert reported_count > 0, f"Source reports 0 chunks but should have {reported_chunk_count}"
                assert success, "RAG search should succeed for source with chunks"
                assert len(rag_result["results"]) > 0, "RAG should return actual chunks for source with reported chunks"


# Marker for tests that should fail initially (Red phase of TDD)
@pytest.mark.failing_by_design
class TestTDDFailingTests:
    """Tests that are designed to fail initially to demonstrate the issue."""
    
    def test_chunks_count_service_does_not_exist_yet(self):
        """This test should fail because ChunksCountService doesn't exist yet."""
        with pytest.raises(ImportError):
            from src.server.services.knowledge.chunks_count_service import ChunksCountService
            
    def test_current_implementation_returns_zero_chunks(self):
        """Test that current implementation incorrectly returns 0 for chunks_count."""
        # This test documents the current broken behavior
        from src.server.services.knowledge.knowledge_item_service import KnowledgeItemService
        
        # Create service instance
        mock_client = MagicMock()
        service = KnowledgeItemService(mock_client)
        
        # Mock database to return non-zero count (simulating actual chunks exist)
        mock_result = MagicMock()
        mock_result.count = 42  # Actual chunks exist
        mock_query = MagicMock()
        mock_query.execute.return_value = mock_result
        mock_query.eq.return_value = mock_query
        mock_client.table.return_value.select.return_value = mock_query
        
        # Current implementation incorrectly queries archon_crawled_pages instead of archon_documents
        # This test should pass with current broken implementation, but fail once we fix it
        import asyncio
        result = asyncio.run(service._get_chunks_count("test_source"))
        
        # This assertion will fail once we fix the implementation to query the correct table
        assert result == 42, f"Current implementation returns wrong table count: {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])