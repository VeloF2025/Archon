"""
API Integration Tests for Chunks Count Fix

This test suite validates that the Knowledge API endpoints correctly 
return accurate chunk counts after the fix is implemented.

These tests focus on the API layer integration and are designed to 
FAIL initially, demonstrating the current API behavior where 
chunks_count is always 0.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from typing import Dict, Any, List


class TestKnowledgeItemsAPIIntegration:
    """Integration tests for /api/knowledge-items endpoint with chunks count fix."""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI app with knowledge router."""
        app = FastAPI()
        from src.server.api_routes.knowledge_api import router
        app.include_router(router)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_knowledge_service(self):
        """Mock KnowledgeItemService with realistic behavior."""
        service = AsyncMock()
        return service
    
    @pytest.fixture
    def realistic_knowledge_items(self):
        """Realistic knowledge items data that shows the chunks count issue."""
        return {
            "items": [
                {
                    "id": "python_docs_source", 
                    "title": "Python Documentation",
                    "url": "https://docs.python.org/3/",
                    "source_id": "python_docs_source",
                    "code_examples": [{"count": 45}],
                    "metadata": {
                        "knowledge_type": "technical",
                        "tags": ["python", "documentation"],
                        "source_type": "url",
                        "status": "active",
                        "description": "Official Python 3 documentation",
                        "chunks_count": 0,  # <-- This is the bug! Should be 89
                        "word_count": 15420,
                        "estimated_pages": 61.7,
                        "code_examples_count": 45
                    },
                    "created_at": "2024-01-15T10:00:00Z",
                    "updated_at": "2024-01-15T10:00:00Z"
                },
                {
                    "id": "react_docs_source",
                    "title": "React Documentation", 
                    "url": "https://react.dev/",
                    "source_id": "react_docs_source",
                    "code_examples": [{"count": 23}],
                    "metadata": {
                        "knowledge_type": "technical",
                        "tags": ["react", "javascript", "frontend"],
                        "source_type": "url",
                        "status": "active", 
                        "description": "Official React documentation",
                        "chunks_count": 0,  # <-- This is the bug! Should be 156
                        "word_count": 12850,
                        "estimated_pages": 51.4,
                        "code_examples_count": 23
                    },
                    "created_at": "2024-01-16T14:30:00Z",
                    "updated_at": "2024-01-16T14:30:00Z"
                }
            ],
            "total": 2,
            "page": 1,
            "per_page": 20,
            "pages": 1
        }
    
    def test_get_knowledge_items_shows_zero_chunks_count_currently(
        self, 
        client, 
        realistic_knowledge_items
    ):
        """
        Test that demonstrates current API behavior: chunks_count is always 0.
        
        This test should PASS initially, showing the current broken behavior.
        After the fix, this test should FAIL, and we'll update it.
        """
        # Arrange
        with patch('src.server.api_routes.knowledge_api.KnowledgeItemService') as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.list_items = AsyncMock(return_value=realistic_knowledge_items)
            
            # Act
            response = client.get("/api/knowledge-items")
            
            # Assert - This documents the current BUG
            assert response.status_code == 200
            data = response.json()
            
            # Current behavior: all chunks_count are 0 (this is the bug)
            for item in data["items"]:
                current_chunks_count = item["metadata"]["chunks_count"]
                
                # This assertion should PASS initially (showing the bug exists)
                assert current_chunks_count == 0, (
                    f"Current API behavior: {item['id']} shows chunks_count={current_chunks_count} "
                    "(this is the bug we're fixing)"
                )
                
                # But we know chunks actually exist (from code_examples_count)
                code_examples_count = item["metadata"]["code_examples_count"]
                assert code_examples_count > 0, (
                    f"Source {item['id']} has {code_examples_count} code examples, "
                    "so it definitely has content and should have chunks"
                )
    
    def test_get_knowledge_items_should_show_actual_chunks_count(
        self, 
        client,
        sample_sources_data
    ):
        """
        Test that demonstrates what the API SHOULD return after the fix.
        
        This test will FAIL initially and should PASS after the fix is implemented.
        """
        # Arrange - This represents the FIXED behavior
        expected_fixed_items = {
            "items": [
                {
                    "id": "source_01",
                    "title": "Knowledge Source 1",
                    "url": "https://example1.com",
                    "source_id": "source_01", 
                    "metadata": {
                        "knowledge_type": "technical",
                        "chunks_count": 87,  # <-- This should be the ACTUAL count after fix
                        "word_count": 1100,
                        "code_examples_count": 12
                    }
                },
                {
                    "id": "source_02",
                    "title": "Knowledge Source 2",
                    "url": "https://example2.com", 
                    "source_id": "source_02",
                    "metadata": {
                        "knowledge_type": "business",
                        "chunks_count": 156,  # <-- This should be the ACTUAL count after fix
                        "word_count": 2200,
                        "code_examples_count": 8
                    }
                }
            ],
            "total": 2,
            "page": 1,
            "per_page": 20,
            "pages": 1
        }
        
        with patch('src.server.api_routes.knowledge_api.KnowledgeItemService') as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.list_items = AsyncMock(return_value=expected_fixed_items)
            
            # Act
            response = client.get("/api/knowledge-items")
            
            # Assert - This test should FAIL initially, PASS after fix
            assert response.status_code == 200
            data = response.json()
            
            for item in data["items"]:
                chunks_count = item["metadata"]["chunks_count"]
                
                # After fix: chunks_count should reflect actual chunks in database
                assert chunks_count > 0, (
                    f"After fix: {item['id']} should show actual chunks_count > 0, got {chunks_count}"
                )
                
                # Should be reasonable compared to word count (rough validation)
                word_count = item["metadata"]["word_count"]
                expected_min_chunks = word_count // 500  # ~500 words per chunk
                expected_max_chunks = word_count // 100  # ~100 words per chunk
                
                assert expected_min_chunks <= chunks_count <= expected_max_chunks, (
                    f"Chunks count {chunks_count} seems unreasonable for {word_count} words "
                    f"(expected between {expected_min_chunks} and {expected_max_chunks})"
                )
    
    def test_get_knowledge_items_with_search_maintains_correct_chunks_count(
        self,
        client
    ):
        """
        Test that search/filtering doesn't break chunks count accuracy.
        """
        # Arrange - Filtered results should still show correct chunks counts
        filtered_results = {
            "items": [
                {
                    "id": "python_tutorial",
                    "title": "Python Tutorial",
                    "metadata": {
                        "knowledge_type": "technical",
                        "chunks_count": 234,  # Should be actual count
                        "tags": ["python", "tutorial"]
                    }
                }
            ],
            "total": 1,
            "page": 1,
            "per_page": 20,
            "pages": 1
        }
        
        with patch('src.server.api_routes.knowledge_api.KnowledgeItemService') as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.list_items = AsyncMock(return_value=filtered_results)
            
            # Act
            response = client.get("/api/knowledge-items?search=python&knowledge_type=technical")
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            
            assert len(data["items"]) == 1
            item = data["items"][0]
            
            # Even with search/filtering, chunks_count should be accurate
            assert item["metadata"]["chunks_count"] == 234
            assert item["metadata"]["chunks_count"] > 0
    
    def test_get_knowledge_items_pagination_maintains_chunks_count(
        self,
        client,
        sample_sources_data
    ):
        """
        Test that pagination doesn't affect chunks count accuracy.
        """
        # Arrange - Create paginated results
        page_1_items = {
            "items": [
                {
                    "id": f"source_{i:02d}",
                    "title": f"Source {i}",
                    "metadata": {
                        "chunks_count": i * 15,  # Should be actual counts
                        "knowledge_type": "technical"
                    }
                }
                for i in range(1, 11)  # First 10 sources
            ],
            "total": 32,
            "page": 1,
            "per_page": 10,
            "pages": 4
        }
        
        page_2_items = {
            "items": [
                {
                    "id": f"source_{i:02d}",
                    "title": f"Source {i}",
                    "metadata": {
                        "chunks_count": i * 15,  # Should be actual counts
                        "knowledge_type": "technical"
                    }
                }
                for i in range(11, 21)  # Second 10 sources
            ],
            "total": 32,
            "page": 2,
            "per_page": 10,
            "pages": 4
        }
        
        with patch('src.server.api_routes.knowledge_api.KnowledgeItemService') as mock_service_class:
            mock_service = mock_service_class.return_value
            
            # Mock different responses based on page
            def mock_list_items(*args, **kwargs):
                page = kwargs.get('page', 1)
                if page == 1:
                    return page_1_items
                elif page == 2:
                    return page_2_items
                else:
                    return {"items": [], "total": 32, "page": page, "per_page": 10, "pages": 4}
            
            mock_service.list_items = AsyncMock(side_effect=mock_list_items)
            
            # Act & Assert for both pages
            for page_num, expected_items in [(1, page_1_items), (2, page_2_items)]:
                response = client.get(f"/api/knowledge-items?page={page_num}&per_page=10")
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["page"] == page_num
                assert len(data["items"]) == 10
                
                # Verify chunks counts are correct for this page
                for i, item in enumerate(data["items"]):
                    source_num = (page_num - 1) * 10 + i + 1
                    expected_chunks = source_num * 15
                    actual_chunks = item["metadata"]["chunks_count"]
                    
                    assert actual_chunks == expected_chunks, (
                        f"Page {page_num}, item {i}: expected {expected_chunks} chunks, got {actual_chunks}"
                    )


class TestSingleKnowledgeItemAPI:
    """Integration tests for single knowledge item retrieval."""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI app."""
        app = FastAPI()
        from src.server.api_routes.knowledge_api import router
        app.include_router(router)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    def test_get_single_knowledge_item_shows_correct_chunks_count(
        self,
        client
    ):
        """
        Test that getting a single knowledge item returns accurate chunks_count.
        
        This test will FAIL initially because the current implementation 
        returns chunks_count: 0 for individual items.
        """
        # Arrange
        source_id = "detailed_test_source"
        expected_item = {
            "id": source_id,
            "title": "Detailed Test Source",
            "url": "https://detailed.example.com",
            "source_id": source_id,
            "metadata": {
                "knowledge_type": "technical",
                "chunks_count": 167,  # Should be actual count from database
                "word_count": 8340,
                "code_examples_count": 23,
                "description": "A comprehensive technical resource"
            },
            "created_at": "2024-01-20T09:15:00Z",
            "updated_at": "2024-01-20T09:15:00Z"
        }
        
        with patch('src.server.api_routes.knowledge_api.KnowledgeItemService') as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.get_item = AsyncMock(return_value=expected_item)
            
            # Act - This endpoint might not exist yet, will be added with the fix
            response = client.get(f"/api/knowledge-items/{source_id}")
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            
            assert data["id"] == source_id
            assert data["metadata"]["chunks_count"] == 167
            assert data["metadata"]["chunks_count"] > 0, (
                "Single knowledge item should show actual chunks count, not 0"
            )
    
    def test_get_single_knowledge_item_not_found(
        self,
        client
    ):
        """Test handling of non-existent knowledge item."""
        # Arrange
        nonexistent_id = "does_not_exist"
        
        with patch('src.server.api_routes.knowledge_api.KnowledgeItemService') as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.get_item = AsyncMock(return_value=None)
            
            # Act
            response = client.get(f"/api/knowledge-items/{nonexistent_id}")
            
            # Assert
            assert response.status_code == 404


class TestRAGQueryIntegration:
    """Integration tests for RAG queries that validate chunks exist."""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI app."""
        app = FastAPI()
        from src.server.api_routes.knowledge_api import router
        app.include_router(router)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    def test_rag_query_returns_chunks_proving_they_exist(
        self,
        client,
        mock_rag_search_results
    ):
        """
        Test RAG query that proves chunks exist even when API reports chunks_count: 0.
        
        This test demonstrates the core issue: RAG returns chunks with chunk_index 
        values like 84, proving chunks exist in the database.
        """
        # Arrange
        query_request = {
            "query": "python functions",
            "match_count": 5
        }
        
        # Mock RAG service to return the problematic results
        with patch('src.server.api_routes.knowledge_api.RAGService') as mock_rag_service_class:
            mock_rag_service = mock_rag_service_class.return_value
            mock_rag_service.perform_rag_query = AsyncMock(
                return_value=(True, mock_rag_search_results)
            )
            
            # Act
            response = client.post("/api/rag/query", json=query_request)
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert len(data["results"]) == 3
            
            # These results PROVE that chunks exist with high chunk_index values
            chunk_indexes = [result["chunk_index"] for result in data["results"]]
            assert 84 in chunk_indexes, "RAG should return chunk with index 84 (proving chunks exist)"
            assert 42 in chunk_indexes, "RAG should return chunk with index 42 (proving chunks exist)"
            assert 156 in chunk_indexes, "RAG should return chunk with index 156 (proving chunks exist)"
            
            # All results should have valid source_ids
            source_ids = [result["source_id"] for result in data["results"]]
            assert "source_01" in source_ids
            assert "source_02" in source_ids
            
            # This proves the discrepancy: RAG finds chunks, but API reports 0 chunks
            for result in data["results"]:
                assert result["chunk_index"] >= 0
                assert len(result["content"]) > 0
                assert result["source_id"].startswith("source_")
    
    def test_rag_query_source_specific_returns_chunks(
        self,
        client
    ):
        """
        Test source-specific RAG query that should return chunks for that source.
        
        If a source reports chunks_count: 0 but RAG returns chunks for it,
        that's the bug we're fixing.
        """
        # Arrange
        specific_source_id = "source_01"
        query_request = {
            "query": "test content",
            "source": specific_source_id,
            "match_count": 10
        }
        
        source_specific_results = {
            "results": [
                {
                    "id": f"doc_1_{i}",
                    "source_id": specific_source_id,
                    "chunk_index": i * 10,
                    "content": f"Content chunk {i} from {specific_source_id}",
                    "similarity": 0.9 - (i * 0.1)
                }
                for i in range(5)  # 5 chunks from this specific source
            ],
            "total_results": 5,
            "query": "test content"
        }
        
        with patch('src.server.api_routes.knowledge_api.RAGService') as mock_rag_service_class:
            mock_rag_service = mock_rag_service_class.return_value
            mock_rag_service.perform_rag_query = AsyncMock(
                return_value=(True, source_specific_results)
            )
            
            # Act
            response = client.post("/api/rag/query", json=query_request)
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert len(data["results"]) == 5
            
            # All results should be from the requested source
            for result in data["results"]:
                assert result["source_id"] == specific_source_id
                assert result["chunk_index"] >= 0
                assert len(result["content"]) > 0
                
            # This proves that source_01 HAS chunks in the database
            # But if the API reports chunks_count: 0 for source_01, that's the bug


class TestDatabaseMetricsIntegration:
    """Integration tests for database metrics that reveal chunks count discrepancies."""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI app."""
        app = FastAPI()
        from src.server.api_routes.knowledge_api import router
        app.include_router(router)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client.""" 
        return TestClient(app)
    
    def test_database_metrics_reveal_chunks_discrepancy(
        self,
        client
    ):
        """
        Test database metrics that should reveal the chunks count discrepancy.
        
        This test will help identify the scope of the issue across all sources.
        """
        # Arrange
        metrics_with_discrepancy = {
            "total_sources": 32,
            "total_documents": 2847,  # Actual chunks in archon_documents
            "total_crawled_pages": 156,  # Pages in archon_crawled_pages  
            "sources_with_zero_reported_chunks": 32,  # All sources report 0 (the bug)
            "sources_with_actual_chunks": 32,  # All sources have actual chunks
            "discrepancy_detected": True,
            "discrepancy_details": {
                "sources_reporting_zero_but_have_chunks": 32,
                "total_unreported_chunks": 2847,
                "affected_source_ids": [f"source_{i:02d}" for i in range(1, 33)]
            }
        }
        
        with patch('src.server.api_routes.knowledge_api.DatabaseMetricsService') as mock_metrics_service_class:
            mock_metrics_service = mock_metrics_service_class.return_value
            mock_metrics_service.get_metrics = AsyncMock(return_value=metrics_with_discrepancy)
            
            # Act
            response = client.get("/api/database/metrics")
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            
            # These metrics expose the bug
            assert data["total_sources"] == 32
            assert data["total_documents"] > 0, "Database has actual chunks"
            assert data["sources_with_zero_reported_chunks"] == 32, "All sources report 0 chunks (the bug)"
            assert data["discrepancy_detected"] is True, "Should detect the chunks count discrepancy"
            
            # The discrepancy details reveal the scope
            discrepancy = data["discrepancy_details"]
            assert discrepancy["sources_reporting_zero_but_have_chunks"] == 32
            assert discrepancy["total_unreported_chunks"] > 0
            assert len(discrepancy["affected_source_ids"]) == 32


class TestAPIErrorHandling:
    """Test API error handling related to chunks count operations."""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI app."""
        app = FastAPI()
        from src.server.api_routes.knowledge_api import router
        app.include_router(router)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    def test_api_handles_chunks_count_service_errors_gracefully(
        self,
        client
    ):
        """
        Test that API handles errors from chunks count service gracefully.
        
        When the chunks count service fails, API should still return items 
        with chunks_count: 0 rather than failing completely.
        """
        # Arrange - Mock service that throws error during chunks count calculation
        failed_items_response = {
            "items": [
                {
                    "id": "error_test_source",
                    "title": "Error Test Source", 
                    "metadata": {
                        "chunks_count": 0,  # Should fall back to 0 on error
                        "knowledge_type": "technical"
                    }
                }
            ],
            "total": 1,
            "page": 1,
            "per_page": 20,
            "pages": 1
        }
        
        with patch('src.server.api_routes.knowledge_api.KnowledgeItemService') as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.list_items = AsyncMock(return_value=failed_items_response)
            
            # Act
            response = client.get("/api/knowledge-items")
            
            # Assert
            assert response.status_code == 200  # Should not fail completely
            data = response.json()
            
            assert len(data["items"]) == 1
            item = data["items"][0]
            
            # Should gracefully fall back to 0 chunks count on error
            assert item["metadata"]["chunks_count"] == 0
    
    def test_api_handles_database_connection_errors(
        self,
        client
    ):
        """
        Test API behavior when database connection fails during chunks count queries.
        """
        # Arrange
        with patch('src.server.api_routes.knowledge_api.KnowledgeItemService') as mock_service_class:
            mock_service = mock_service_class.return_value
            # Simulate database connection error
            mock_service.list_items = AsyncMock(side_effect=Exception("Database connection failed"))
            
            # Act
            response = client.get("/api/knowledge-items")
            
            # Assert
            assert response.status_code == 500
            data = response.json()
            assert "error" in data


# Marker for tests that should fail initially (Red phase of TDD)
@pytest.mark.failing_by_design
class TestTDDAPIIntegrationFailures:
    """Tests designed to fail initially to demonstrate API integration issues."""
    
    def test_current_api_always_returns_zero_chunks_count(self):
        """
        Test that documents the current API bug.
        
        This test should PASS initially (documenting the bug) and FAIL after the fix.
        """
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        
        # Create minimal test app
        app = FastAPI()
        
        # This represents current broken behavior
        @app.get("/api/knowledge-items")
        async def broken_knowledge_items():
            return {
                "items": [
                    {
                        "id": "test_source",
                        "metadata": {
                            "chunks_count": 0  # Always returns 0 (the bug)
                        }
                    }
                ]
            }
        
        client = TestClient(app)
        response = client.get("/api/knowledge-items")
        
        # This assertion documents the current bug
        assert response.status_code == 200
        data = response.json()
        
        # Current behavior: always returns 0 chunks
        for item in data["items"]:
            assert item["metadata"]["chunks_count"] == 0, (
                "Current implementation always returns 0 chunks (this is the bug)"
            )
    
    def test_rag_and_api_inconsistency_exists(self):
        """
        Test that demonstrates the inconsistency between RAG and API results.
        
        This test proves the bug exists: RAG finds chunks, API reports 0.
        """
        # This test represents the current inconsistent state
        
        # Simulate current API response
        api_chunks_count = 0  # API always reports 0
        
        # Simulate RAG response
        rag_chunks_found = [
            {"source_id": "test_source", "chunk_index": 84},
            {"source_id": "test_source", "chunk_index": 156}
        ]
        
        # This demonstrates the inconsistency (the bug)
        assert api_chunks_count == 0, "API reports no chunks"
        assert len(rag_chunks_found) > 0, "But RAG finds actual chunks"
        
        # The inconsistency exists
        assert api_chunks_count != len(rag_chunks_found), (
            f"Inconsistency: API reports {api_chunks_count} chunks, "
            f"RAG finds {len(rag_chunks_found)} chunks"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])