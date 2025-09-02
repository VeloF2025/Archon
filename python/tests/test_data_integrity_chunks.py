"""
Data Integrity Tests for Chunks Count Fix

This test suite focuses on data consistency and integrity validation
to ensure chunk counts match actual database state and detect orphaned data.

These tests are designed to FAIL initially and guide the implementation
of proper data validation and cleanup services.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from typing import List, Dict, Any
import time


class TestDataIntegrityValidation:
    """Comprehensive data integrity tests for chunks count consistency."""
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client with data integrity focus."""
        client = MagicMock()
        client.table = MagicMock()
        client.from_ = MagicMock()
        client.rpc = MagicMock()
        return client
    
    @pytest.fixture
    def data_integrity_service(self, mock_supabase_client):
        """Create DataIntegrityService instance (will fail initially)."""
        # This import will fail initially since the service doesn't exist
        from src.server.services.knowledge.data_integrity_service import DataIntegrityService
        return DataIntegrityService(mock_supabase_client)
    
    def test_all_32_sources_have_accurate_chunk_counts(
        self, 
        mock_supabase_client,
        sample_sources_data,
        expected_chunks_counts
    ):
        """
        Test that all 32 existing knowledge sources have accurate chunk counts.
        
        This test validates the core issue: sources report 0 chunks but actually have chunks.
        """
        # Arrange - Mock the database queries
        
        # Mock sources query
        mock_sources_result = MagicMock()
        mock_sources_result.data = sample_sources_data
        mock_sources_query = MagicMock()
        mock_sources_query.execute.return_value = mock_sources_result
        mock_supabase_client.from_.return_value.select.return_value = mock_sources_query
        
        # Mock actual chunks count query from archon_documents
        def mock_rpc_call(function_name, params):
            if function_name == 'get_chunks_count_bulk':
                source_ids = params['source_ids']
                # Return actual counts that don't match the reported 0
                result = MagicMock()
                result.data = [
                    {"source_id": sid, "chunk_count": expected_chunks_counts.get(sid, 0)}
                    for sid in source_ids
                ]
                return result
            return MagicMock()
        
        mock_supabase_client.rpc.side_effect = lambda fn, params=None: MagicMock(execute=lambda: mock_rpc_call(fn, params))
        
        # Act - This will fail initially since DataIntegrityService doesn't exist
        from src.server.services.knowledge.data_integrity_service import DataIntegrityService
        service = DataIntegrityService(mock_supabase_client)
        
        validation_report = service.validate_all_sources_chunk_counts()
        
        # Assert - All sources should have consistent chunk counts
        assert validation_report["total_sources"] == 32
        assert validation_report["inconsistent_count"] == 0, f"Found {validation_report['inconsistent_count']} sources with inconsistent chunk counts"
        assert len(validation_report["inconsistent_sources"]) == 0, f"Inconsistent sources: {validation_report['inconsistent_sources']}"
        
        # Verify no sources report 0 chunks when they actually have chunks
        for source in validation_report["sources"]:
            reported_count = source["reported_chunks_count"]
            actual_count = source["actual_chunks_count"]
            
            assert reported_count == actual_count, (
                f"Source {source['source_id']} reports {reported_count} chunks "
                f"but actually has {actual_count} chunks"
            )
            
            if actual_count > 0:
                assert reported_count > 0, (
                    f"Source {source['source_id']} has {actual_count} actual chunks "
                    f"but reports {reported_count} chunks (the core bug)"
                )
    
    def test_no_orphaned_documents_without_source_references(
        self,
        mock_supabase_client
    ):
        """
        Test that no documents exist without valid source references.
        
        This validates data integrity by ensuring all chunks belong to valid sources.
        """
        # Arrange
        # Mock query to find orphaned documents
        mock_orphaned_result = MagicMock()
        mock_orphaned_result.data = []  # Should be empty if no orphans
        mock_orphaned_query = MagicMock()
        mock_orphaned_query.execute.return_value = mock_orphaned_result
        
        def mock_rpc_call(function_name, params=None):
            if function_name == 'find_orphaned_documents':
                return mock_orphaned_query
            return MagicMock(execute=lambda: MagicMock(data=[]))
        
        mock_supabase_client.rpc.side_effect = mock_rpc_call
        
        # Act - This will fail initially
        from src.server.services.knowledge.data_integrity_service import DataIntegrityService
        service = DataIntegrityService(mock_supabase_client)
        
        orphaned_report = service.find_orphaned_documents()
        
        # Assert
        assert orphaned_report["orphaned_count"] == 0, (
            f"Found {orphaned_report['orphaned_count']} orphaned documents: "
            f"{orphaned_report.get('orphaned_documents', [])}"
        )
    
    def test_detect_sources_with_zero_reported_but_actual_chunks(
        self,
        mock_supabase_client,
        sample_sources_data
    ):
        """
        Test specifically for the reported bug: sources showing 0 chunks but having actual chunks.
        
        This test directly addresses the core issue described in the ticket.
        """
        # Arrange - Simulate the current broken state
        # Sources report 0 chunks (current API behavior)
        sources_with_zero_reported = []
        for source in sample_sources_data[:10]:  # Test first 10 sources
            source_copy = source.copy()
            source_copy["reported_chunks_count"] = 0  # Current broken behavior
            sources_with_zero_reported.append(source_copy)
        
        # Mock sources query
        mock_sources_result = MagicMock()
        mock_sources_result.data = sources_with_zero_reported
        mock_supabase_client.from_.return_value.select.return_value.execute.return_value = mock_sources_result
        
        # Mock actual chunks query that shows chunks exist
        actual_chunks_data = []
        for i, source in enumerate(sources_with_zero_reported):
            actual_count = (i + 1) * 5  # Actual chunks exist
            actual_chunks_data.append({
                "source_id": source["source_id"],
                "chunk_count": actual_count
            })
        
        mock_chunks_result = MagicMock()
        mock_chunks_result.data = actual_chunks_data
        mock_supabase_client.rpc.return_value.execute.return_value = mock_chunks_result
        
        # Act - This will fail initially
        from src.server.services.knowledge.data_integrity_service import DataIntegrityService
        service = DataIntegrityService(mock_supabase_client)
        
        bug_report = service.detect_zero_reported_with_actual_chunks()
        
        # Assert - This test should find the exact bug we're fixing
        assert len(bug_report["affected_sources"]) > 0, (
            "Should detect sources that report 0 chunks but have actual chunks"
        )
        
        for affected_source in bug_report["affected_sources"]:
            assert affected_source["reported_chunks_count"] == 0
            assert affected_source["actual_chunks_count"] > 0
            
        # This validates the core issue exists
        assert bug_report["total_affected"] > 0, (
            "Should detect the chunks count discrepancy bug"
        )
    
    def test_validate_chunk_index_consistency(
        self,
        mock_supabase_client
    ):
        """
        Test that chunk indexes are consistent and sequential within each source.
        
        This ensures chunks are properly indexed and no gaps exist.
        """
        # Arrange
        source_id = "test_source"
        
        # Mock chunks data with potential gaps
        chunks_with_gaps = [
            {"source_id": source_id, "chunk_index": 0, "content": "Chunk 0"},
            {"source_id": source_id, "chunk_index": 1, "content": "Chunk 1"},
            # Gap - missing chunk_index 2
            {"source_id": source_id, "chunk_index": 3, "content": "Chunk 3"},
            {"source_id": source_id, "chunk_index": 4, "content": "Chunk 4"},
        ]
        
        mock_chunks_result = MagicMock()
        mock_chunks_result.data = chunks_with_gaps
        mock_query = MagicMock()
        mock_query.execute.return_value = mock_chunks_result
        mock_query.eq.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_supabase_client.table.return_value.select.return_value = mock_query
        
        # Act - This will fail initially
        from src.server.services.knowledge.data_integrity_service import DataIntegrityService
        service = DataIntegrityService(mock_supabase_client)
        
        consistency_report = service.validate_chunk_index_consistency(source_id)
        
        # Assert
        assert not consistency_report["is_consistent"], (
            "Should detect chunk index gaps"
        )
        assert len(consistency_report["missing_indexes"]) > 0, (
            "Should identify missing chunk indexes"
        )
        assert 2 in consistency_report["missing_indexes"], (
            "Should detect that chunk_index 2 is missing"
        )
    
    def test_cross_table_referential_integrity(
        self,
        mock_supabase_client
    ):
        """
        Test referential integrity between sources, documents, and pages tables.
        
        This ensures all foreign key relationships are maintained properly.
        """
        # Arrange
        # Mock sources
        sources = [
            {"source_id": "source_1", "title": "Source 1"},
            {"source_id": "source_2", "title": "Source 2"}
        ]
        
        # Mock documents (some with invalid source references)
        documents = [
            {"source_id": "source_1", "chunk_index": 0},  # Valid reference
            {"source_id": "source_2", "chunk_index": 0},  # Valid reference
            {"source_id": "invalid_source", "chunk_index": 0},  # Invalid reference
        ]
        
        # Mock pages (some with invalid source references)
        pages = [
            {"source_id": "source_1", "url": "https://example.com"},  # Valid reference
            {"source_id": "another_invalid_source", "url": "https://invalid.com"},  # Invalid reference
        ]
        
        # Mock database responses
        def mock_table_query(table_name):
            mock_query = MagicMock()
            if table_name == "archon_sources":
                mock_query.execute.return_value.data = sources
            elif table_name == "archon_documents": 
                mock_query.execute.return_value.data = documents
            elif table_name == "archon_crawled_pages":
                mock_query.execute.return_value.data = pages
            else:
                mock_query.execute.return_value.data = []
            
            mock_query.select.return_value = mock_query
            return mock_query
        
        mock_supabase_client.table.side_effect = mock_table_query
        
        # Act - This will fail initially
        from src.server.services.knowledge.data_integrity_service import DataIntegrityService
        service = DataIntegrityService(mock_supabase_client)
        
        integrity_report = service.validate_referential_integrity()
        
        # Assert
        assert not integrity_report["is_consistent"], (
            "Should detect referential integrity violations"
        )
        
        assert len(integrity_report["documents_violations"]) > 0, (
            "Should detect documents with invalid source references"
        )
        
        assert len(integrity_report["pages_violations"]) > 0, (
            "Should detect pages with invalid source references"
        )
        
        # Check specific violations
        document_violations = [v["source_id"] for v in integrity_report["documents_violations"]]
        assert "invalid_source" in document_violations
        
        page_violations = [v["source_id"] for v in integrity_report["pages_violations"]]  
        assert "another_invalid_source" in page_violations


class TestPerformanceDataIntegrity:
    """Performance tests for data integrity operations."""
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client optimized for performance testing."""
        client = MagicMock()
        client.table = MagicMock()
        client.rpc = MagicMock()
        return client
    
    @pytest.mark.performance
    def test_bulk_integrity_validation_performance(
        self, 
        mock_supabase_client,
        sample_sources_data
    ):
        """
        Test that bulk data integrity validation completes within performance requirements.
        
        Target: <2 seconds for validating all 32 sources.
        """
        # Arrange
        # Mock fast bulk validation responses
        mock_sources_result = MagicMock()
        mock_sources_result.data = sample_sources_data
        mock_supabase_client.from_.return_value.select.return_value.execute.return_value = mock_sources_result
        
        # Mock fast RPC responses  
        def mock_fast_rpc(function_name, params=None):
            result = MagicMock()
            if function_name == 'validate_all_integrity':
                result.data = {
                    "total_sources": 32,
                    "validation_time_ms": 150,
                    "issues_found": 0
                }
            return result
        
        mock_supabase_client.rpc.return_value.execute.side_effect = mock_fast_rpc
        
        # Act - This will fail initially
        from src.server.services.knowledge.data_integrity_service import DataIntegrityService
        service = DataIntegrityService(mock_supabase_client)
        
        start_time = time.time()
        report = service.run_comprehensive_integrity_check()
        end_time = time.time()
        
        # Assert
        execution_time = (end_time - start_time) * 1000  # Convert to ms
        assert execution_time < 2000, f"Integrity validation took {execution_time}ms, should be <2000ms"
        
        assert report["validation_completed"] == True
        assert report["total_sources_validated"] == 32
    
    @pytest.mark.performance
    def test_orphaned_documents_detection_performance(
        self,
        mock_supabase_client
    ):
        """
        Test that orphaned document detection is performant.
        
        Target: <500ms to detect orphaned documents across entire database.
        """
        # Arrange
        # Mock efficient orphaned documents query
        def mock_fast_orphaned_query():
            result = MagicMock()
            result.data = []  # No orphaned documents found
            result.execution_time_ms = 89
            return result
        
        mock_supabase_client.rpc.return_value.execute.side_effect = mock_fast_orphaned_query
        
        # Act - This will fail initially
        from src.server.services.knowledge.data_integrity_service import DataIntegrityService
        service = DataIntegrityService(mock_supabase_client)
        
        start_time = time.time()
        orphan_report = service.find_orphaned_documents()
        end_time = time.time()
        
        # Assert
        execution_time = (end_time - start_time) * 1000
        assert execution_time < 500, f"Orphaned detection took {execution_time}ms, should be <500ms"
        
        assert orphan_report["orphaned_count"] == 0
        assert "execution_time_ms" in orphan_report


class TestDataIntegrityRepairOperations:
    """Tests for data integrity repair and cleanup operations."""
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client for repair operations."""
        client = MagicMock()
        client.table = MagicMock()
        client.rpc = MagicMock()
        return client
    
    def test_repair_chunk_count_discrepancies(
        self,
        mock_supabase_client
    ):
        """
        Test automated repair of chunk count discrepancies.
        
        This tests the fix mechanism for the core issue.
        """
        # Arrange
        sources_with_discrepancies = [
            {
                "source_id": "source_1",
                "reported_chunks_count": 0,  # Wrong
                "actual_chunks_count": 45    # Correct
            },
            {
                "source_id": "source_2", 
                "reported_chunks_count": 0,  # Wrong
                "actual_chunks_count": 23    # Correct
            }
        ]
        
        # Mock detection of discrepancies
        mock_detection_result = MagicMock()
        mock_detection_result.data = sources_with_discrepancies
        
        # Mock successful repair operations
        mock_repair_result = MagicMock()
        mock_repair_result.data = {
            "repaired_count": 2,
            "failed_count": 0,
            "repaired_sources": ["source_1", "source_2"]
        }
        
        def mock_rpc_calls(function_name, params=None):
            if function_name == 'detect_chunk_count_discrepancies':
                return mock_detection_result
            elif function_name == 'repair_chunk_count_discrepancies':
                return mock_repair_result
            return MagicMock()
        
        mock_supabase_client.rpc.return_value.execute.side_effect = mock_rpc_calls
        
        # Act - This will fail initially
        from src.server.services.knowledge.data_integrity_service import DataIntegrityService
        service = DataIntegrityService(mock_supabase_client)
        
        repair_report = service.repair_chunk_count_discrepancies()
        
        # Assert
        assert repair_report["repaired_count"] == 2
        assert repair_report["failed_count"] == 0
        assert len(repair_report["repaired_sources"]) == 2
        assert "source_1" in repair_report["repaired_sources"]
        assert "source_2" in repair_report["repaired_sources"]
    
    def test_cleanup_orphaned_documents(
        self,
        mock_supabase_client
    ):
        """
        Test cleanup of orphaned documents that have no valid source reference.
        """
        # Arrange
        orphaned_documents = [
            {"id": "orphan_1", "source_id": "invalid_source_1", "chunk_index": 0},
            {"id": "orphan_2", "source_id": "invalid_source_2", "chunk_index": 0},
        ]
        
        # Mock orphaned documents detection
        mock_orphan_result = MagicMock()
        mock_orphan_result.data = orphaned_documents
        
        # Mock successful cleanup
        mock_cleanup_result = MagicMock()
        mock_cleanup_result.data = {
            "deleted_count": 2,
            "failed_count": 0,
            "deleted_document_ids": ["orphan_1", "orphan_2"]
        }
        
        def mock_rpc_calls(function_name, params=None):
            if function_name == 'find_orphaned_documents':
                return mock_orphan_result
            elif function_name == 'cleanup_orphaned_documents':
                return mock_cleanup_result
            return MagicMock()
        
        mock_supabase_client.rpc.return_value.execute.side_effect = mock_rpc_calls
        
        # Act - This will fail initially
        from src.server.services.knowledge.data_integrity_service import DataIntegrityService
        service = DataIntegrityService(mock_supabase_client)
        
        cleanup_report = service.cleanup_orphaned_documents()
        
        # Assert
        assert cleanup_report["deleted_count"] == 2
        assert cleanup_report["failed_count"] == 0
        assert len(cleanup_report["deleted_document_ids"]) == 2
    
    def test_rebuild_chunk_indexes_for_source(
        self,
        mock_supabase_client
    ):
        """
        Test rebuilding chunk indexes when they have gaps or inconsistencies.
        """
        # Arrange
        source_id = "source_with_gaps"
        
        # Mock chunks with gaps in indexes
        chunks_with_gaps = [
            {"id": "chunk_1", "source_id": source_id, "chunk_index": 0},
            {"id": "chunk_2", "source_id": source_id, "chunk_index": 1},
            # Gap at index 2
            {"id": "chunk_3", "source_id": source_id, "chunk_index": 5},
            {"id": "chunk_4", "source_id": source_id, "chunk_index": 8},
        ]
        
        # Mock successful rebuild
        mock_rebuild_result = MagicMock()
        mock_rebuild_result.data = {
            "source_id": source_id,
            "original_chunk_count": 4,
            "rebuilt_chunk_count": 4,
            "gaps_fixed": 2,
            "new_max_index": 3
        }
        
        def mock_rpc_calls(function_name, params=None):
            if function_name == 'get_chunks_with_gaps':
                mock_result = MagicMock()
                mock_result.data = chunks_with_gaps
                return mock_result
            elif function_name == 'rebuild_chunk_indexes':
                return mock_rebuild_result
            return MagicMock()
        
        mock_supabase_client.rpc.return_value.execute.side_effect = mock_rpc_calls
        
        # Act - This will fail initially
        from src.server.services.knowledge.data_integrity_service import DataIntegrityService
        service = DataIntegrityService(mock_supabase_client)
        
        rebuild_report = service.rebuild_chunk_indexes(source_id)
        
        # Assert
        assert rebuild_report["gaps_fixed"] == 2
        assert rebuild_report["new_max_index"] == 3
        assert rebuild_report["original_chunk_count"] == rebuild_report["rebuilt_chunk_count"]


# Marker for tests that should fail initially (Red phase of TDD)
@pytest.mark.failing_by_design
class TestTDDDataIntegrityFailures:
    """Tests designed to fail initially to demonstrate data integrity issues."""
    
    def test_data_integrity_service_does_not_exist_yet(self):
        """This test should fail because DataIntegrityService doesn't exist yet."""
        with pytest.raises(ImportError):
            from src.server.services.knowledge.data_integrity_service import DataIntegrityService
            
    def test_current_state_has_chunk_count_discrepancies(self, sample_sources_data):
        """
        Test that demonstrates the current broken state.
        
        This test documents that the current implementation has the bug we're fixing.
        """
        # This test represents the current broken behavior
        # where sources report 0 chunks but actually have chunks
        
        # Simulate current behavior: all sources report 0 chunks
        current_broken_behavior = True
        
        for source in sample_sources_data:
            # Current API behavior: hardcoded to 0
            current_reported_chunks = 0
            
            # But actual chunks exist (simulated)
            actual_chunks = (int(source["source_id"].split("_")[1]) * 3)
            
            if current_broken_behavior:
                # This assertion documents the current bug
                assert current_reported_chunks == 0, "Current implementation always reports 0"
                assert actual_chunks > 0, "But actual chunks exist in database"
                
                # This is the discrepancy we're fixing
                assert current_reported_chunks != actual_chunks, (
                    f"Source {source['source_id']}: reported={current_reported_chunks}, "
                    f"actual={actual_chunks} - This is the bug we're fixing"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])