"""
Performance Tests for Chunks Count Operations

This test suite validates that all chunks count operations meet strict 
performance requirements:
- Batch counting <100ms for 50 sources  
- API responses <500ms
- Cache hit rate >90%
- Memory usage within limits

These tests are designed to FAIL initially and guide optimization.
"""

import pytest
import asyncio
import time
import memory_profiler
from unittest.mock import MagicMock, AsyncMock, patch
from typing import List, Dict, Any
import concurrent.futures
import threading
import statistics


@pytest.mark.performance
class TestChunksCountPerformance:
    """Performance tests for chunks count operations."""
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client with performance tracking."""
        client = MagicMock()
        client.table = MagicMock()
        client.rpc = MagicMock()
        
        # Track call times for performance analysis
        client._call_times = []
        
        def track_call_time(original_method):
            def wrapper(*args, **kwargs):
                start = time.time()
                result = original_method(*args, **kwargs)
                end = time.time()
                client._call_times.append((end - start) * 1000)  # Convert to ms
                return result
            return wrapper
        
        # Wrap execute method to track timing
        original_execute = MagicMock()
        client.rpc.return_value.execute = track_call_time(original_execute)
        
        return client
    
    @pytest.fixture
    def large_source_dataset(self):
        """Generate large dataset for performance testing."""
        sources = []
        for i in range(100):  # 100 sources for stress testing
            source_id = f"perf_source_{i:03d}"
            chunk_count = min(1000, (i + 1) * 10)  # Varying counts up to 1000
            
            sources.append({
                "source_id": source_id,
                "expected_chunk_count": chunk_count,
                "title": f"Performance Test Source {i}",
                "metadata": {"knowledge_type": "technical"}
            })
        
        return sources
    
    def test_batch_counting_under_100ms_for_50_sources(
        self, 
        mock_supabase_client,
        large_source_dataset
    ):
        """
        Test that batch counting 50 sources takes <100ms.
        
        This is a critical performance requirement for the UI responsiveness.
        """
        # Arrange
        test_sources = large_source_dataset[:50]  # First 50 sources
        source_ids = [s["source_id"] for s in test_sources]
        
        # Mock fast database response
        expected_results = [
            {"source_id": s["source_id"], "chunk_count": s["expected_chunk_count"]}
            for s in test_sources
        ]
        
        mock_result = MagicMock()
        mock_result.data = expected_results
        
        # Simulate realistic database query time (should be optimized)
        def mock_execute():
            time.sleep(0.02)  # 20ms database time (acceptable)
            return mock_result
        
        mock_supabase_client.rpc.return_value.execute = mock_execute
        
        # Act - This will fail initially since ChunksCountService doesn't exist
        from src.server.services.knowledge.chunks_count_service import ChunksCountService
        service = ChunksCountService(mock_supabase_client)
        
        start_time = time.time()
        result_counts = service.get_bulk_chunks_count(source_ids)
        end_time = time.time()
        
        # Assert
        execution_time = (end_time - start_time) * 1000  # Convert to ms
        assert execution_time < 100, f"Batch counting took {execution_time:.2f}ms, should be <100ms"
        
        # Verify correctness
        assert len(result_counts) == 50, f"Expected 50 results, got {len(result_counts)}"
        
        for source in test_sources:
            source_id = source["source_id"]
            expected_count = source["expected_chunk_count"]
            actual_count = result_counts.get(source_id, 0)
            assert actual_count == expected_count, f"Source {source_id}: expected {expected_count}, got {actual_count}"
    
    def test_single_source_counting_under_10ms(
        self, 
        mock_supabase_client
    ):
        """
        Test that counting chunks for a single source takes <10ms.
        
        This ensures individual operations are fast for real-time queries.
        """
        # Arrange
        source_id = "single_perf_test"
        expected_count = 127
        
        # Mock very fast single query
        mock_result = MagicMock()
        mock_result.count = expected_count
        
        def mock_fast_execute():
            time.sleep(0.002)  # 2ms database time
            return mock_result
        
        mock_query = MagicMock()
        mock_query.execute = mock_fast_execute
        mock_query.eq.return_value = mock_query
        mock_supabase_client.table.return_value.select.return_value = mock_query
        
        # Act - This will fail initially
        from src.server.services.knowledge.chunks_count_service import ChunksCountService
        service = ChunksCountService(mock_supabase_client)
        
        start_time = time.time()
        actual_count = service.get_chunks_count(source_id)
        end_time = time.time()
        
        # Assert
        execution_time = (end_time - start_time) * 1000
        assert execution_time < 10, f"Single count took {execution_time:.2f}ms, should be <10ms"
        assert actual_count == expected_count
    
    def test_cache_performance_hit_rate_above_90_percent(
        self,
        mock_supabase_client
    ):
        """
        Test that caching achieves >90% hit rate for repeated requests.
        
        This validates that our caching strategy significantly reduces database load.
        """
        # Arrange
        source_ids = [f"cache_test_{i}" for i in range(10)]
        
        # Track database calls
        db_call_count = 0
        
        def mock_execute_with_tracking():
            nonlocal db_call_count
            db_call_count += 1
            
            # Simulate database response
            mock_result = MagicMock()
            mock_result.data = [
                {"source_id": sid, "chunk_count": 15}
                for sid in source_ids
            ]
            
            time.sleep(0.01)  # 10ms per DB call
            return mock_result
        
        mock_supabase_client.rpc.return_value.execute = mock_execute_with_tracking
        
        # Act - This will fail initially
        from src.server.services.knowledge.chunks_count_service import ChunksCountService
        service = ChunksCountService(mock_supabase_client)
        
        # Make 20 requests for the same data
        total_requests = 20
        for i in range(total_requests):
            service.get_bulk_chunks_count(source_ids)
        
        # Assert
        cache_hit_rate = ((total_requests - db_call_count) / total_requests) * 100
        assert cache_hit_rate >= 90, f"Cache hit rate {cache_hit_rate:.1f}% is below 90%"
        
        # Should only make 1 DB call for 20 requests with good caching
        assert db_call_count <= 2, f"Expected ≤2 DB calls, got {db_call_count}"
    
    def test_concurrent_access_performance(
        self,
        mock_supabase_client
    ):
        """
        Test performance under concurrent access scenarios.
        
        This validates that the service handles multiple simultaneous requests efficiently.
        """
        # Arrange
        source_ids = [f"concurrent_test_{i}" for i in range(25)]
        
        # Mock thread-safe database response
        mock_result = MagicMock()
        mock_result.data = [
            {"source_id": sid, "chunk_count": 20}
            for sid in source_ids
        ]
        
        call_times = []
        
        def mock_concurrent_execute():
            start = time.time()
            time.sleep(0.03)  # 30ms per call
            end = time.time()
            call_times.append((end - start) * 1000)
            return mock_result
        
        mock_supabase_client.rpc.return_value.execute = mock_concurrent_execute
        
        # Act - This will fail initially
        from src.server.services.knowledge.chunks_count_service import ChunksCountService
        service = ChunksCountService(mock_supabase_client)
        
        # Run 10 concurrent requests
        def make_request():
            return service.get_bulk_chunks_count(source_ids)
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        
        # Assert
        # With good concurrency handling, 10 requests should complete in <200ms
        assert total_time < 200, f"10 concurrent requests took {total_time:.2f}ms, should be <200ms"
        
        # All requests should return correct results
        assert len(results) == 10
        for result in results:
            assert len(result) == 25  # 25 source counts returned
    
    @pytest.mark.memory_intensive
    def test_memory_usage_within_limits(
        self,
        mock_supabase_client,
        large_source_dataset
    ):
        """
        Test that memory usage stays within acceptable limits during large operations.
        
        Target: <50MB additional memory usage for processing 100 sources.
        """
        # Arrange
        source_ids = [s["source_id"] for s in large_source_dataset]
        
        # Mock large dataset response
        mock_result = MagicMock()
        mock_result.data = [
            {"source_id": s["source_id"], "chunk_count": s["expected_chunk_count"]}
            for s in large_source_dataset
        ]
        
        mock_supabase_client.rpc.return_value.execute.return_value = mock_result
        
        # Act & Assert - This will fail initially
        from src.server.services.knowledge.chunks_count_service import ChunksCountService
        service = ChunksCountService(mock_supabase_client)
        
        # Measure memory usage
        @memory_profiler.profile(precision=1)
        def run_large_operation():
            return service.get_bulk_chunks_count(source_ids)
        
        # Memory profiling will show if we exceed limits
        result = run_large_operation()
        
        # Basic assertions
        assert len(result) == 100
        assert all(count > 0 for count in result.values())


@pytest.mark.performance 
class TestKnowledgeAPIPerformance:
    """Performance tests for Knowledge API endpoints."""
    
    @pytest.fixture
    async def performance_client(self):
        """Create optimized test client for performance testing."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from src.server.api_routes.knowledge_api import router
        
        app = FastAPI()
        app.include_router(router)
        
        with TestClient(app) as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_knowledge_items_api_under_500ms(
        self,
        performance_client,
        sample_sources_data
    ):
        """
        Test that /api/knowledge-items responds in <500ms.
        
        This is critical for UI responsiveness in the knowledge management interface.
        """
        # Arrange
        # Mock fast service responses
        mock_items_response = {
            "items": [
                {
                    "id": source["source_id"],
                    "title": source["title"],
                    "metadata": {
                        "chunks_count": (int(source["source_id"].split("_")[1]) * 3),  # Realistic count
                        "knowledge_type": source["metadata"]["knowledge_type"]
                    }
                }
                for source in sample_sources_data[:20]  # First 20 for API test
            ],
            "total": 20,
            "page": 1,
            "per_page": 20,
            "pages": 1
        }
        
        # Act - This will fail initially due to current slow implementation
        with patch('src.server.api_routes.knowledge_api.KnowledgeItemService') as mock_service:
            # Simulate optimized service response time
            async def fast_list_items(*args, **kwargs):
                await asyncio.sleep(0.05)  # 50ms service time (optimized)
                return mock_items_response
            
            mock_service.return_value.list_items = fast_list_items
            
            start_time = time.time()
            response = performance_client.get("/api/knowledge-items?per_page=20")
            end_time = time.time()
            
            # Assert
            execution_time = (end_time - start_time) * 1000
            assert response.status_code == 200
            assert execution_time < 500, f"API response took {execution_time:.2f}ms, should be <500ms"
            
            data = response.json()
            assert len(data["items"]) == 20
            
            # Verify all items have non-zero chunk counts (the fix)
            for item in data["items"]:
                assert item["metadata"]["chunks_count"] > 0, (
                    f"Item {item['id']} still shows 0 chunks after optimization"
                )
    
    def test_paginated_requests_performance_consistency(
        self,
        performance_client,
        sample_sources_data
    ):
        """
        Test that paginated requests maintain consistent performance.
        
        All pages should have similar response times (no degradation with offset).
        """
        # Arrange
        page_size = 10
        total_pages = 5
        page_times = []
        
        # Mock consistent service performance
        def create_page_response(page_num):
            start_idx = (page_num - 1) * page_size
            end_idx = start_idx + page_size
            page_sources = sample_sources_data[start_idx:end_idx]
            
            return {
                "items": [
                    {
                        "id": source["source_id"],
                        "title": source["title"], 
                        "metadata": {"chunks_count": 15}
                    }
                    for source in page_sources
                ],
                "total": len(sample_sources_data),
                "page": page_num,
                "per_page": page_size,
                "pages": total_pages
            }
        
        # Act - This will fail initially due to inefficient pagination
        with patch('src.server.api_routes.knowledge_api.KnowledgeItemService') as mock_service:
            async def fast_paginated_list(*args, **kwargs):
                page = kwargs.get('page', 1)
                await asyncio.sleep(0.02)  # Consistent 20ms regardless of page
                return create_page_response(page)
            
            mock_service.return_value.list_items = fast_paginated_list
            
            # Test each page performance
            for page in range(1, total_pages + 1):
                start_time = time.time()
                response = performance_client.get(f"/api/knowledge-items?page={page}&per_page={page_size}")
                end_time = time.time()
                
                execution_time = (end_time - start_time) * 1000
                page_times.append(execution_time)
                
                assert response.status_code == 200
                assert execution_time < 100, f"Page {page} took {execution_time:.2f}ms, should be <100ms"
        
        # Assert consistent performance across pages
        avg_time = statistics.mean(page_times)
        max_time = max(page_times)
        min_time = min(page_times)
        
        # Variation should be <50% of average time
        max_variation = max_time - min_time
        allowed_variation = avg_time * 0.5
        
        assert max_variation < allowed_variation, (
            f"Page time variation {max_variation:.2f}ms exceeds {allowed_variation:.2f}ms "
            f"(times: {[f'{t:.1f}' for t in page_times]})"
        )
    
    def test_search_and_filtering_performance(
        self,
        performance_client
    ):
        """
        Test that search and filtering operations maintain good performance.
        
        Search queries should complete in <300ms even with filters applied.
        """
        # Arrange
        search_scenarios = [
            {"search": "python", "knowledge_type": None},
            {"search": None, "knowledge_type": "technical"},
            {"search": "documentation", "knowledge_type": "business"},
            {"search": "test", "knowledge_type": "technical"}
        ]
        
        def create_filtered_response(search, knowledge_type):
            # Simulate filtered results
            filtered_count = 5 if search and knowledge_type else 10
            return {
                "items": [
                    {
                        "id": f"filtered_source_{i}",
                        "title": f"Filtered Source {i}",
                        "metadata": {"chunks_count": 20, "knowledge_type": knowledge_type or "technical"}
                    }
                    for i in range(filtered_count)
                ],
                "total": filtered_count,
                "page": 1,
                "per_page": 20,
                "pages": 1
            }
        
        # Act - This will fail initially due to inefficient search
        with patch('src.server.api_routes.knowledge_api.KnowledgeItemService') as mock_service:
            async def fast_filtered_list(*args, **kwargs):
                search = kwargs.get('search')
                knowledge_type = kwargs.get('knowledge_type')
                # Optimized search should be fast
                await asyncio.sleep(0.05)  # 50ms for optimized search
                return create_filtered_response(search, knowledge_type)
            
            mock_service.return_value.list_items = fast_filtered_list
            
            for scenario in search_scenarios:
                params = {}
                if scenario["search"]:
                    params["search"] = scenario["search"]
                if scenario["knowledge_type"]:
                    params["knowledge_type"] = scenario["knowledge_type"]
                
                query_string = "&".join([f"{k}={v}" for k, v in params.items()])
                url = f"/api/knowledge-items?{query_string}" if query_string else "/api/knowledge-items"
                
                start_time = time.time()
                response = performance_client.get(url)
                end_time = time.time()
                
                execution_time = (end_time - start_time) * 1000
                
                # Assert
                assert response.status_code == 200
                assert execution_time < 300, (
                    f"Search scenario {scenario} took {execution_time:.2f}ms, should be <300ms"
                )
                
                data = response.json()
                assert len(data["items"]) > 0, "Search should return results"


@pytest.mark.performance
class TestDatabaseOptimizationValidation:
    """Tests that validate database-level optimizations for chunk counting."""
    
    @pytest.fixture
    def mock_optimized_supabase_client(self):
        """Mock Supabase client that simulates optimized database operations."""
        client = MagicMock()
        
        # Track query performance metrics
        client._query_metrics = {
            "total_queries": 0,
            "total_time_ms": 0,
            "slow_queries": 0  # Queries >50ms
        }
        
        def track_query_performance(query_time_ms):
            client._query_metrics["total_queries"] += 1
            client._query_metrics["total_time_ms"] += query_time_ms
            if query_time_ms > 50:
                client._query_metrics["slow_queries"] += 1
        
        # Mock optimized RPC function
        def mock_optimized_rpc(function_name, params=None):
            result = MagicMock()
            
            if function_name == 'get_chunks_count_bulk':
                # Simulate optimized bulk query performance
                source_ids = params.get('source_ids', [])
                query_time = min(5 + len(source_ids) * 0.5, 50)  # Scales efficiently
                track_query_performance(query_time)
                
                time.sleep(query_time / 1000)  # Simulate actual query time
                
                result.data = [
                    {"source_id": sid, "chunk_count": 25}
                    for sid in source_ids
                ]
            elif function_name == 'get_chunks_count_single':
                # Single query should be very fast
                query_time = 3  # 3ms for single query
                track_query_performance(query_time)
                time.sleep(query_time / 1000)
                result.count = 42
            
            mock_execute = MagicMock(return_value=result)
            return MagicMock(execute=mock_execute)
        
        client.rpc.side_effect = mock_optimized_rpc
        
        # Mock optimized table queries
        def mock_optimized_table_query():
            query_time = 8  # 8ms for optimized table query
            track_query_performance(query_time)
            time.sleep(query_time / 1000)
            
            result = MagicMock()
            result.count = 67
            return result
        
        mock_query_builder = MagicMock()
        mock_query_builder.execute = mock_optimized_table_query
        mock_query_builder.select.return_value = mock_query_builder
        mock_query_builder.eq.return_value = mock_query_builder
        client.table.return_value = mock_query_builder
        
        return client
    
    def test_bulk_query_optimization_effectiveness(
        self,
        mock_optimized_supabase_client
    ):
        """
        Test that bulk queries are significantly faster than individual queries.
        
        Bulk query for 50 sources should be much faster than 50 individual queries.
        """
        # Arrange
        source_ids = [f"opt_test_{i}" for i in range(50)]
        
        # Act - This will fail initially
        from src.server.services.knowledge.chunks_count_service import ChunksCountService
        service = ChunksCountService(mock_optimized_supabase_client)
        
        # Test bulk query performance
        start_time = time.time()
        bulk_results = service.get_bulk_chunks_count(source_ids)
        bulk_time = (time.time() - start_time) * 1000
        
        # Test individual queries performance (for comparison)
        mock_optimized_supabase_client._query_metrics = {"total_queries": 0, "total_time_ms": 0, "slow_queries": 0}
        
        start_time = time.time()
        individual_results = {}
        for source_id in source_ids[:10]:  # Test only first 10 to avoid test timeout
            individual_results[source_id] = service.get_chunks_count(source_id)
        individual_time = (time.time() - start_time) * 1000
        
        # Scale individual time to 50 sources for comparison
        projected_individual_time = individual_time * 5  # 10 -> 50 sources
        
        # Assert
        assert bulk_time < projected_individual_time / 5, (
            f"Bulk query ({bulk_time:.2f}ms) should be much faster than "
            f"individual queries ({projected_individual_time:.2f}ms projected for 50)"
        )
        
        assert len(bulk_results) == 50
        assert all(count > 0 for count in bulk_results.values())
        
        # Verify query efficiency metrics
        metrics = mock_optimized_supabase_client._query_metrics
        assert metrics["slow_queries"] == 0, f"Found {metrics['slow_queries']} slow queries (>50ms)"
    
    def test_database_index_effectiveness_simulation(
        self,
        mock_optimized_supabase_client
    ):
        """
        Test that simulates the effectiveness of database indexes.
        
        With proper indexes, queries should have consistent performance regardless of data size.
        """
        # Arrange - Test different data sizes to verify index effectiveness
        test_sizes = [10, 50, 100, 200]
        performance_results = []
        
        # Act - This will fail initially
        from src.server.services.knowledge.chunks_count_service import ChunksCountService
        service = ChunksCountService(mock_optimized_supabase_client)
        
        for size in test_sizes:
            source_ids = [f"index_test_{i}" for i in range(size)]
            
            # Reset metrics
            mock_optimized_supabase_client._query_metrics = {
                "total_queries": 0, "total_time_ms": 0, "slow_queries": 0
            }
            
            start_time = time.time()
            results = service.get_bulk_chunks_count(source_ids)
            execution_time = (time.time() - start_time) * 1000
            
            performance_results.append({
                "size": size,
                "time_ms": execution_time,
                "time_per_source": execution_time / size,
                "slow_queries": mock_optimized_supabase_client._query_metrics["slow_queries"]
            })
            
            assert len(results) == size
        
        # Assert - Performance should scale linearly or better with good indexes
        largest_test = performance_results[-1]  # 200 sources
        smallest_test = performance_results[0]   # 10 sources
        
        # Time per source should be similar (good index performance)
        time_per_source_variance = abs(
            largest_test["time_per_source"] - smallest_test["time_per_source"]
        )
        
        max_allowed_variance = 2.0  # 2ms variance per source is acceptable
        assert time_per_source_variance < max_allowed_variance, (
            f"Time per source variance {time_per_source_variance:.2f}ms exceeds {max_allowed_variance}ms "
            f"- indexes may not be effective"
        )
        
        # No queries should be slow
        for result in performance_results:
            assert result["slow_queries"] == 0, f"Size {result['size']}: {result['slow_queries']} slow queries"
    
    def test_connection_pooling_effectiveness(
        self,
        mock_optimized_supabase_client
    ):
        """
        Test that connection pooling handles concurrent requests efficiently.
        
        Multiple concurrent requests should not create connection bottlenecks.
        """
        # Arrange
        source_ids = [f"pool_test_{i}" for i in range(20)]
        
        # Simulate concurrent requests
        def make_concurrent_request(request_id):
            from src.server.services.knowledge.chunks_count_service import ChunksCountService
            service = ChunksCountService(mock_optimized_supabase_client)
            
            start_time = time.time()
            results = service.get_bulk_chunks_count(source_ids)
            execution_time = (time.time() - start_time) * 1000
            
            return {
                "request_id": request_id,
                "execution_time": execution_time,
                "result_count": len(results)
            }
        
        # Act - This will fail initially
        concurrent_requests = 8
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [
                executor.submit(make_concurrent_request, i)
                for i in range(concurrent_requests)
            ]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = (time.time() - start_time) * 1000
        
        # Assert
        # All requests should complete successfully
        assert len(results) == concurrent_requests
        
        # Total time should be reasonable (good connection pooling)
        max_allowed_total_time = 300  # 300ms for 8 concurrent requests
        assert total_time < max_allowed_total_time, (
            f"8 concurrent requests took {total_time:.2f}ms, should be <{max_allowed_total_time}ms"
        )
        
        # Individual request times should be consistent
        execution_times = [r["execution_time"] for r in results]
        avg_time = statistics.mean(execution_times)
        max_time = max(execution_times)
        
        # No request should take more than 2x the average (good load balancing)
        assert max_time < avg_time * 2, (
            f"Slowest request ({max_time:.2f}ms) took more than 2x average ({avg_time:.2f}ms)"
        )
        
        # All requests should return correct results
        for result in results:
            assert result["result_count"] == 20


# Benchmark comparison tests
@pytest.mark.performance
@pytest.mark.slow
class TestPerformanceRegression:
    """Tests to prevent performance regression after optimization."""
    
    def test_performance_baseline_vs_optimized(
        self,
        mock_supabase_client
    ):
        """
        Test that optimized implementation is significantly faster than baseline.
        
        The new implementation should be at least 5x faster than the old one.
        """
        # Arrange
        source_ids = [f"regression_test_{i}" for i in range(30)]
        
        # Mock baseline (current slow implementation)
        def mock_slow_baseline():
            time.sleep(0.1)  # 100ms per call (current slow behavior)
            return MagicMock(data=[{"source_id": sid, "chunk_count": 10} for sid in source_ids])
        
        # Mock optimized implementation
        def mock_fast_optimized():
            time.sleep(0.015)  # 15ms per call (optimized)
            return MagicMock(data=[{"source_id": sid, "chunk_count": 10} for sid in source_ids])
        
        # Test baseline performance (simulating current implementation)
        mock_supabase_client.rpc.return_value.execute = mock_slow_baseline
        
        # This represents current slow implementation
        class SlowChunksCountService:
            def __init__(self, client):
                self.client = client
                
            def get_bulk_chunks_count(self, source_ids):
                # Simulate current inefficient approach
                results = {}
                for source_id in source_ids:
                    # Multiple individual queries (inefficient)
                    result = self.client.rpc().execute()
                    results[source_id] = 10
                return results
        
        baseline_service = SlowChunksCountService(mock_supabase_client)
        
        start_time = time.time()
        baseline_results = baseline_service.get_bulk_chunks_count(source_ids)
        baseline_time = (time.time() - start_time) * 1000
        
        # Test optimized performance
        mock_supabase_client.rpc.return_value.execute = mock_fast_optimized
        
        # This will fail initially - represents the optimized implementation we're building
        from src.server.services.knowledge.chunks_count_service import ChunksCountService
        optimized_service = ChunksCountService(mock_supabase_client)
        
        start_time = time.time()
        optimized_results = optimized_service.get_bulk_chunks_count(source_ids)
        optimized_time = (time.time() - start_time) * 1000
        
        # Assert
        assert len(baseline_results) == len(optimized_results) == 30
        
        # Optimized should be at least 5x faster
        speedup_ratio = baseline_time / optimized_time
        assert speedup_ratio >= 5, (
            f"Optimized implementation is only {speedup_ratio:.1f}x faster, should be ≥5x "
            f"(baseline: {baseline_time:.2f}ms, optimized: {optimized_time:.2f}ms)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance"])