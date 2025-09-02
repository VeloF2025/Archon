# TDD Test Suite: Knowledge Base Chunks Count Fix

## Overview

This Test-Driven Development (TDD) test suite is designed to fix the critical chunks count discrepancy in the Knowledge Base system where:

- **Problem**: API reports `chunks_count: 0` for all knowledge items
- **Reality**: RAG search returns actual chunks with `chunk_index` values like 84
- **Impact**: UI shows incorrect chunk counts, affecting user understanding of content volume

## TDD Methodology

This test suite follows strict TDD principles:

### 1. **RED Phase** 🔴
- Tests are designed to **FAIL initially**
- Each test documents the current broken behavior
- Tests prove the discrepancy exists
- Marked with `@pytest.mark.failing_by_design`

### 2. **GREEN Phase** 🟢  
- Implement minimal code to make tests pass
- Focus on correctness over optimization
- All tests should pass after implementation

### 3. **REFACTOR Phase** 🔄
- Optimize for performance requirements
- Maintain test coverage >95%
- Ensure all tests continue to pass

## Test Categories

### Unit Tests (`test_chunks_count_service.py`)
- **ChunksCountService** - New service for efficient chunk counting
- **KnowledgeItemService** - Updates to existing service
- Coverage: >95% of service methods
- Performance: Single queries <10ms, batch queries <100ms

### Integration Tests (`test_api_integration_chunks.py`)
- **Knowledge Items API** - `/api/knowledge-items` endpoint
- **RAG Query API** - `/api/rag/query` endpoint  
- **Database Metrics API** - `/api/database/metrics` endpoint
- Focus: End-to-end API behavior with correct chunk counts

### Data Integrity Tests (`test_data_integrity_chunks.py`)
- **Discrepancy Detection** - Identify sources with wrong counts
- **Orphaned Documents** - Find chunks without valid sources
- **Referential Integrity** - Validate cross-table relationships
- **Repair Operations** - Automated fix mechanisms

### Performance Tests (`test_performance_chunks_count.py`)
- **Batch Operations** - <100ms for 50 sources
- **API Response Times** - <500ms for knowledge items endpoint
- **Cache Effectiveness** - >90% hit rate
- **Concurrent Access** - Handle multiple simultaneous requests
- **Memory Usage** - <50MB additional for 100 sources

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install pytest pytest-cov pytest-asyncio memory-profiler
```

### 2. Run All TDD Tests
```bash
# Run with TDD-specific configuration
pytest -c pytest.chunks.ini

# Or run individual test files
pytest tests/test_chunks_count_service.py -v
pytest tests/test_api_integration_chunks.py -v
pytest tests/test_data_integrity_chunks.py -v
pytest tests/test_performance_chunks_count.py -v
```

### 3. Run by TDD Phase
```bash
# RED Phase - Tests that should fail initially
pytest -m "failing_by_design" -v

# Performance requirements validation
pytest -m "performance" -v

# Integration validation
pytest -m "integration" -v
```

### 4. Generate Coverage Report
```bash
pytest -c pytest.chunks.ini --cov-report=html
# Open htmlcov/chunks_count/index.html
```

## Test Execution Workflow

### Initial State (RED Phase)
```bash
# These commands should show FAILING tests initially
pytest tests/test_chunks_count_service.py::TestChunksCountService::test_get_chunks_count_single_source_exists -v
# Expected: ImportError - ChunksCountService doesn't exist

pytest tests/test_api_integration_chunks.py::TestKnowledgeItemsAPIIntegration::test_get_knowledge_items_should_show_actual_chunks_count -v  
# Expected: AssertionError - API returns chunks_count: 0

pytest tests/test_data_integrity_chunks.py::TestDataIntegrityValidation::test_all_32_sources_have_accurate_chunk_counts -v
# Expected: ImportError - DataIntegrityService doesn't exist
```

### Implementation Phase (GREEN Phase)
After implementing the fixes, these should pass:
```bash
pytest tests/test_chunks_count_service.py -v
pytest tests/test_api_integration_chunks.py -v
pytest tests/test_data_integrity_chunks.py -v
```

### Performance Validation (REFACTOR Phase)
```bash
pytest tests/test_performance_chunks_count.py -v -m "performance"
```

## Key Test Files

### `test_chunks_count_service.py`
**Purpose**: Unit tests for the new ChunksCountService
**Key Tests**:
- `test_get_chunks_count_single_source_exists` - Single source counting
- `test_get_bulk_chunks_count_multiple_sources` - Batch operations
- `test_get_bulk_chunks_count_with_cache` - Caching effectiveness

### `test_api_integration_chunks.py` 
**Purpose**: API endpoint integration testing
**Key Tests**:
- `test_get_knowledge_items_shows_zero_chunks_count_currently` - Documents current bug
- `test_get_knowledge_items_should_show_actual_chunks_count` - Tests fix
- `test_rag_query_returns_chunks_proving_they_exist` - Proves chunks exist

### `test_data_integrity_chunks.py`
**Purpose**: Data consistency and integrity validation
**Key Tests**:
- `test_all_32_sources_have_accurate_chunk_counts` - Validates 32 sources
- `test_detect_sources_with_zero_reported_but_actual_chunks` - Core bug detection
- `test_repair_chunk_count_discrepancies` - Automated fix testing

### `test_performance_chunks_count.py`
**Purpose**: Performance requirements validation
**Key Tests**:
- `test_batch_counting_under_100ms_for_50_sources` - Batch performance
- `test_knowledge_items_api_under_500ms` - API response time
- `test_cache_hit_rate_above_90_percent` - Caching efficiency

## Performance Requirements

### Response Time Targets
- **Single source count**: <10ms
- **Batch count (50 sources)**: <100ms  
- **Knowledge items API**: <500ms
- **Database integrity check**: <2s for all sources

### Caching Requirements
- **Cache hit rate**: >90% for repeated requests
- **Cache invalidation**: Automatic on data updates
- **Memory usage**: <50MB additional for 100 sources

### Scalability Targets  
- **Concurrent requests**: 10+ simultaneous without degradation
- **Large datasets**: 1000+ sources without timeout
- **Database load**: Minimize queries through efficient batching

## Database Requirements

### Tables Involved
- `archon_sources` - Source metadata (contains incorrect chunks_count)
- `archon_documents` - Actual chunks with embeddings (source of truth)
- `archon_crawled_pages` - Pages data (incorrectly used for counting)

### Required Indexes
```sql
CREATE INDEX idx_archon_documents_source_id ON archon_documents(source_id);
CREATE INDEX idx_archon_documents_source_chunk ON archon_documents(source_id, chunk_index);
CREATE INDEX idx_archon_sources_metadata_chunks ON archon_sources USING gin(metadata);
```

### Database Functions
- `get_chunks_count_bulk(text[])` - Efficient batch counting
- `get_chunks_count_single(text)` - Single source counting
- `validate_chunks_integrity()` - Data integrity validation
- `repair_chunk_count_discrepancies()` - Automated repair

## Implementation Checklist

### Phase 1: Services Implementation
- [ ] Create `ChunksCountService` class
- [ ] Implement `get_chunks_count(source_id)` method  
- [ ] Implement `get_bulk_chunks_count(source_ids)` method
- [ ] Add caching mechanism
- [ ] Update `KnowledgeItemService._get_chunks_count()` method

### Phase 2: API Integration
- [ ] Update knowledge items API to use ChunksCountService
- [ ] Modify `list_items()` to return accurate counts
- [ ] Update `get_item()` for single items
- [ ] Add error handling for count failures

### Phase 3: Database Optimization  
- [ ] Install database functions from SQL files
- [ ] Create performance indexes
- [ ] Run migration script to fix existing data
- [ ] Validate all 32 sources have correct counts

### Phase 4: Data Integrity
- [ ] Create `DataIntegrityService` class
- [ ] Implement discrepancy detection
- [ ] Add automated repair functionality
- [ ] Create monitoring for future issues

## Troubleshooting

### Common Issues

**ImportError: No module named 'chunks_count_service'**
- This is expected in RED phase - the service doesn't exist yet
- Implement the service to make tests pass

**AssertionError: chunks_count is 0, expected > 0**  
- This shows the current bug - API returns 0 chunks
- Update the service to query `archon_documents` table

**Performance tests failing (timing out)**
- Database may need optimization
- Check if indexes are created
- Verify caching is working

### Test Debugging
```bash
# Run with detailed output
pytest -c pytest.chunks.ini -vvv --tb=long

# Run specific failing test
pytest tests/test_chunks_count_service.py::TestChunksCountService::test_get_chunks_count_single_source_exists -vvv

# Run with pdb debugger
pytest tests/test_chunks_count_service.py --pdb

# Skip slow tests during development
pytest -c pytest.chunks.ini -m "not slow"
```

## Success Criteria

### All Tests Passing
```bash
pytest -c pytest.chunks.ini
# Should show: ====== X passed in Y.XXs ======
```

### Coverage Requirements Met
```bash
pytest -c pytest.chunks.ini --cov-report=term
# Should show: Total coverage: >95%
```

### Performance Targets Met
```bash
pytest -c pytest.chunks.ini -m "performance"
# All performance tests should pass
```

### Manual Validation
1. **API Test**: `GET /api/knowledge-items` should show chunks_count > 0
2. **RAG Test**: Chunks returned should match reported counts
3. **UI Test**: Knowledge management UI shows accurate chunk counts

## Next Steps

After all tests pass:

1. **Deploy Database Migration**
   ```bash
   psql -d archon -f sql/migration_chunks_count_fix.sql
   ```

2. **Deploy Code Changes**
   - ChunksCountService implementation
   - KnowledgeItemService updates  
   - API endpoint modifications

3. **Monitor Production**
   - Use `v_chunk_count_status` view for monitoring
   - Set up alerts for discrepancies
   - Regular integrity checks

4. **Performance Monitoring**
   - Track API response times
   - Monitor database query performance  
   - Watch cache hit rates

## Contact

For questions about this TDD test suite or the chunks count fix implementation, please refer to the implementation team or the main project documentation.

---

**Last Updated**: 2024-12-31  
**TDD Methodology**: Red-Green-Refactor  
**Coverage Target**: >95%  
**Performance Target**: <500ms API, <100ms batch operations