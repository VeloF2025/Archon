# Test Coverage Requirements: Knowledge Base Chunks Count Fix

## Coverage Targets and Standards

### Overall Coverage Requirements
- **Minimum Coverage**: 95% overall
- **Critical Modules**: 100% coverage required
- **New Services**: 100% line and branch coverage  
- **Modified Services**: 95% coverage maintenance
- **Integration Points**: 100% API endpoint coverage

## Module-Specific Coverage Requirements

### ChunksCountService (NEW - 100% Required)
**File**: `src/server/services/knowledge/chunks_count_service.py`

**Methods requiring 100% coverage**:
- `__init__(supabase_client)` - Constructor validation
- `get_chunks_count(source_id: str) -> int` - Single source counting
- `get_bulk_chunks_count(source_ids: List[str]) -> Dict[str, int]` - Batch operations
- `_cache_key(source_id: str) -> str` - Cache key generation
- `_get_from_cache(key: str) -> Optional[int]` - Cache retrieval
- `_set_cache(key: str, value: int, ttl: int = 300)` - Cache storage
- `_clear_cache(source_id: str = None)` - Cache invalidation

**Test scenarios for each method**:
1. **Happy path**: Normal operation with valid data
2. **Edge cases**: Empty inputs, boundary conditions, maximum values
3. **Error handling**: Database errors, invalid inputs, cache failures
4. **Performance**: Response time validation, concurrent access

### KnowledgeItemService (MODIFIED - 95% Required)
**File**: `src/server/services/knowledge/knowledge_item_service.py`

**Critical methods requiring 100% coverage**:
- `_get_chunks_count(source_id: str) -> int` - MODIFIED METHOD
- `list_items(...)` - Updated to use new chunks count logic
- `_transform_source_to_item(source)` - Updated chunk count integration

**Test focus areas**:
- Integration with ChunksCountService
- Backward compatibility maintenance
- Error handling for chunk count failures
- Performance impact validation

### DataIntegrityService (NEW - 100% Required)  
**File**: `src/server/services/knowledge/data_integrity_service.py`

**Methods requiring 100% coverage**:
- `validate_all_sources_chunk_counts() -> Dict` - Core validation
- `detect_zero_reported_with_actual_chunks() -> Dict` - Bug detection
- `find_orphaned_documents() -> Dict` - Orphan detection
- `validate_chunk_index_consistency(source_id: str) -> Dict` - Index validation
- `validate_referential_integrity() -> Dict` - Cross-table validation
- `repair_chunk_count_discrepancies() -> Dict` - Automated repair
- `cleanup_orphaned_documents() -> Dict` - Cleanup operations
- `rebuild_chunk_indexes(source_id: str) -> Dict` - Index rebuilding

## API Endpoint Coverage Requirements

### Knowledge Items API (100% Required)
**File**: `src/server/api_routes/knowledge_api.py`

**Endpoints requiring full coverage**:
- `GET /api/knowledge-items` - List with accurate chunk counts
- `GET /api/knowledge-items/{source_id}` - Single item with chunk count
- `PUT /api/knowledge-items/{source_id}` - Update operations
- `GET /api/database/metrics` - Metrics including chunk statistics

**Coverage scenarios for each endpoint**:
1. **Success responses**: 200 OK with correct data
2. **Error responses**: 4xx/5xx with proper error handling
3. **Input validation**: Invalid parameters, malformed requests
4. **Performance**: Response time requirements
5. **Security**: Authentication/authorization checks
6. **Data consistency**: Chunk counts match database state

## Performance Test Coverage Requirements

### Batch Operations (ChunksCountService)
- `get_bulk_chunks_count` with 1, 10, 50, 100+ sources
- Performance under concurrent access (10+ simultaneous requests)
- Cache effectiveness validation (hit rates, invalidation)
- Memory usage profiling during large operations

### API Response Times
- Knowledge items endpoint with various page sizes
- Search/filtering performance impact
- Database query optimization validation
- Caching layer effectiveness

### Database Operations
- Index usage validation
- Query execution plan analysis
- Concurrent access patterns
- Connection pooling effectiveness

## Branch Coverage Requirements

### Conditional Logic Coverage
All conditional branches must be tested:

**ChunksCountService**:
```python
# Both branches must be tested
if source_id in self.cache:
    return self.cache[source_id]  # Cache hit branch
else:
    result = self._query_database(source_id)  # Cache miss branch
```

**Error Handling**:
```python
try:
    result = self.supabase.rpc('get_chunks_count_bulk', params)  # Success branch
except Exception as e:
    self.logger.error(f"Database error: {e}")  # Error branch
    return {}  # Fallback branch
```

**Data Validation**:
```python
if not source_ids:
    return {}  # Empty input branch
elif len(source_ids) == 1:
    return self._get_single(source_ids[0])  # Single item optimization branch  
else:
    return self._get_bulk(source_ids)  # Bulk operation branch
```

## Integration Test Coverage Requirements

### Database Integration
- Connection handling (success/failure scenarios)
- Transaction management
- Concurrent access patterns
- Data consistency validation
- Migration script testing

### Service Layer Integration  
- ChunksCountService ↔ KnowledgeItemService integration
- DataIntegrityService ↔ Database integration
- Caching layer ↔ Service integration
- Error propagation between layers

### API Integration
- Service ↔ API endpoint integration
- Request/response data transformation
- Error handling and status codes
- Authentication/authorization flow
- Rate limiting and throttling

## Test Data Coverage Requirements

### Source Variations
Tests must cover diverse source types:
- **URL sources**: Web crawled content
- **File sources**: Uploaded documents  
- **Mixed sources**: Combination of types
- **Edge cases**: Empty sources, corrupted data
- **Scale scenarios**: 1, 32, 100+ sources

### Chunk Count Variations
- **Zero chunks**: Sources with no content
- **Small counts**: 1-10 chunks per source
- **Medium counts**: 10-100 chunks per source  
- **Large counts**: 100+ chunks per source
- **Maximum counts**: Test system limits

### Data States
- **Consistent state**: Reported = Actual counts
- **Inconsistent state**: Reported ≠ Actual counts (the bug)
- **Corrupted state**: Invalid data, missing references
- **Empty state**: No sources or chunks
- **Mixed state**: Some consistent, some inconsistent

## Error Scenario Coverage Requirements

### Database Errors
- Connection failures
- Query timeouts
- Transaction rollbacks
- Index corruption
- Constraint violations

### Service Errors  
- Invalid parameters
- Memory exhaustion
- Cache failures
- Network timeouts
- Authentication failures

### Data Errors
- Orphaned records
- Circular references
- Data type mismatches
- Encoding issues
- Schema violations

## Coverage Measurement and Reporting

### Coverage Tools Configuration
```ini
[coverage:run]
source = src/server/services/knowledge
branch = true
parallel = true
concurrency = thread,multiprocessing

[coverage:report]
show_missing = true
precision = 2
fail_under = 95
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:
```

### Coverage Report Requirements
- **HTML Report**: Detailed line-by-line coverage
- **JSON Report**: Machine-readable metrics
- **Console Report**: Summary for CI/CD
- **XML Report**: Integration with tools

### Coverage Validation Commands
```bash
# Generate comprehensive coverage report
pytest -c pytest.chunks.ini --cov-report=html --cov-report=json --cov-report=term

# Validate minimum coverage requirements
pytest -c pytest.chunks.ini --cov-fail-under=95

# Branch coverage validation
pytest -c pytest.chunks.ini --cov-branch --cov-report=term-missing
```

## Continuous Integration Coverage Requirements

### Pre-commit Hooks
- Minimum 95% coverage check
- No decrease in coverage from previous commit
- All new files must have 100% coverage
- Critical paths must maintain 100% coverage

### Pull Request Requirements
- Coverage report in PR comments
- Comparison with base branch coverage
- Identification of uncovered lines
- Performance test execution results

### Deployment Gates
- 95% overall coverage required for deployment
- 100% coverage for critical security paths
- All performance tests passing
- No known uncovered error scenarios

## Coverage Exclusions and Exceptions

### Acceptable Exclusions
Lines that may be excluded from coverage:
- Debug logging statements
- Abstract method definitions  
- Platform-specific code paths
- External library integration points
- Development-only code paths

### Unacceptable Exclusions
Code that MUST be covered:
- Error handling logic
- Data validation routines
- Security-related code
- API endpoint logic
- Database operations
- Cache management
- Performance-critical paths

## Coverage Quality Gates

### Quality Metrics
- **Line Coverage**: ≥95% overall
- **Branch Coverage**: ≥90% overall  
- **Function Coverage**: 100% for new functions
- **Class Coverage**: 100% for new classes
- **Module Coverage**: ≥95% per module

### Coverage Trends
- No coverage decrease >2% in single PR
- Monthly coverage improvement target: +1%
- Critical bug fixes require coverage improvement
- Performance optimizations maintain coverage

## Testing Best Practices for Coverage

### Test Design Principles
1. **Meaningful Tests**: Coverage should result from testing actual behavior, not just executing code
2. **Edge Case Focus**: Prioritize testing boundary conditions and error scenarios
3. **Integration Coverage**: Test interactions between components
4. **Performance Coverage**: Include performance requirements in coverage validation

### Coverage-Driven Test Writing
```python
# Example: Comprehensive test for chunk counting
def test_get_chunks_count_comprehensive():
    """Test covers all branches and scenarios for get_chunks_count method."""
    
    # Happy path - source exists with chunks
    count = service.get_chunks_count("source_with_chunks")
    assert count > 0
    
    # Edge case - source exists but no chunks  
    count = service.get_chunks_count("empty_source")
    assert count == 0
    
    # Error case - source doesn't exist
    count = service.get_chunks_count("nonexistent_source")
    assert count == 0
    
    # Performance case - response time validation
    start_time = time.time()
    service.get_chunks_count("performance_test_source")
    duration = time.time() - start_time
    assert duration < 0.01  # <10ms requirement
    
    # Cache validation - second call should be faster
    start_time = time.time()
    service.get_chunks_count("performance_test_source")  # Should hit cache
    cached_duration = time.time() - start_time
    assert cached_duration < duration / 2  # Cache should be significantly faster
```

## Coverage Monitoring and Alerts

### Automated Monitoring
- Daily coverage reports
- Coverage regression alerts  
- Performance impact monitoring
- Error rate correlation with coverage

### Coverage Dashboards
- Real-time coverage metrics
- Trend analysis and projections
- Module-specific coverage breakdown
- Historical coverage data

### Alert Thresholds
- **Critical**: Coverage drops below 90%
- **Warning**: Coverage decreases by >5%
- **Info**: New uncovered code detected
- **Performance**: Coverage collection time >30s

## Documentation Coverage Requirements

### Code Documentation
- All public methods require docstrings
- Complex algorithms need inline comments
- Test methods must describe what they validate
- Coverage reports must be self-explanatory

### Test Documentation  
- Test purpose and scenarios clearly documented
- Expected behavior and edge cases described
- Performance requirements stated
- Integration points identified

This comprehensive coverage strategy ensures that the chunks count fix is thoroughly tested, reliable, and maintainable while meeting all performance and quality requirements.