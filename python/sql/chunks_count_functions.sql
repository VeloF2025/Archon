-- PostgreSQL Functions for Efficient Chunks Count Operations
-- 
-- This SQL file contains optimized database functions to support
-- the chunks count fix. These functions enable fast batch operations
-- and maintain data integrity.
--
-- Performance targets:
-- - get_chunks_count_bulk: <100ms for 50 sources
-- - get_chunks_count_single: <10ms per source
-- - validate_chunks_integrity: <2s for all sources

-- =====================================================
-- Function: get_chunks_count_bulk
-- Purpose: Get chunk counts for multiple sources efficiently
-- Performance: <100ms for 50 sources using optimized JOIN
-- =====================================================

CREATE OR REPLACE FUNCTION get_chunks_count_bulk(source_ids text[])
RETURNS TABLE(source_id text, chunk_count bigint)
LANGUAGE plpgsql
STABLE
PARALLEL SAFE
AS $$
BEGIN
    -- Use optimized query with proper indexes for fast batch counting
    -- This replaces individual queries with a single efficient operation
    RETURN QUERY
    SELECT 
        s.source_id::text,
        COALESCE(d.chunk_count, 0) AS chunk_count
    FROM 
        unnest(source_ids) AS s(source_id)
    LEFT JOIN (
        -- Subquery to count chunks per source efficiently
        SELECT 
            doc.source_id,
            COUNT(*) AS chunk_count
        FROM archon_crawled_pages doc
        WHERE doc.source_id = ANY(source_ids)
        GROUP BY doc.source_id
    ) d ON s.source_id = d.source_id
    ORDER BY s.source_id;
    
    -- Add performance logging if needed
    -- RAISE NOTICE 'Processed % sources in batch', array_length(source_ids, 1);
END;
$$;

-- Add comment for documentation
COMMENT ON FUNCTION get_chunks_count_bulk(text[]) IS 
'Efficiently retrieves chunk counts for multiple sources in a single query. 
Optimized for batch operations with <100ms target for 50 sources.';


-- =====================================================
-- Function: get_chunks_count_single  
-- Purpose: Get chunk count for a single source with caching hints
-- Performance: <10ms per source
-- =====================================================

CREATE OR REPLACE FUNCTION get_chunks_count_single(p_source_id text)
RETURNS bigint
LANGUAGE plpgsql
STABLE
PARALLEL SAFE
AS $$
DECLARE
    chunk_count bigint;
BEGIN
    -- Optimized single source query with index hint
    SELECT COUNT(*)
    INTO chunk_count
    FROM archon_crawled_pages 
    WHERE source_id = p_source_id;
    
    -- Return 0 if no chunks found (handles NULL)
    RETURN COALESCE(chunk_count, 0);
END;
$$;

COMMENT ON FUNCTION get_chunks_count_single(text) IS
'Fast single-source chunk count with <10ms performance target.';


-- =====================================================
-- Function: validate_chunks_integrity
-- Purpose: Comprehensive data integrity validation for chunks
-- Performance: <2s for all sources  
-- =====================================================

CREATE OR REPLACE FUNCTION validate_chunks_integrity()
RETURNS TABLE(
    total_sources bigint,
    sources_with_chunks bigint,
    sources_without_chunks bigint,
    total_chunks bigint,
    orphaned_chunks bigint,
    integrity_issues jsonb
)
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    orphan_count bigint;
    integrity_data jsonb DEFAULT '{}'::jsonb;
BEGIN
    -- Count orphaned documents (chunks without valid source references)
    SELECT COUNT(*)
    INTO orphan_count
    FROM archon_crawled_pages d
    LEFT JOIN archon_sources s ON d.source_id = s.source_id
    WHERE s.source_id IS NULL;
    
    -- Build integrity issues JSON
    integrity_data = jsonb_build_object(
        'orphaned_documents_found', orphan_count > 0,
        'orphaned_document_count', orphan_count,
        'validation_timestamp', EXTRACT(EPOCH FROM NOW())
    );
    
    -- Return comprehensive validation results
    RETURN QUERY
    SELECT 
        -- Total sources count
        (SELECT COUNT(*) FROM archon_sources)::bigint,
        
        -- Sources with chunks count  
        (SELECT COUNT(DISTINCT d.source_id) 
         FROM archon_crawled_pages d 
         INNER JOIN archon_sources s ON d.source_id = s.source_id)::bigint,
        
        -- Sources without chunks count
        (SELECT COUNT(*) 
         FROM archon_sources s
         LEFT JOIN archon_crawled_pages d ON s.source_id = d.source_id
         WHERE d.source_id IS NULL)::bigint,
        
        -- Total chunks count
        (SELECT COUNT(*) FROM archon_crawled_pages)::bigint,
        
        -- Orphaned chunks count
        orphan_count,
        
        -- Integrity issues details
        integrity_data;
END;
$$;

COMMENT ON FUNCTION validate_chunks_integrity() IS
'Comprehensive data integrity validation for chunks count system.
Returns detailed statistics about sources, chunks, and integrity issues.';


-- =====================================================
-- Function: detect_chunk_count_discrepancies  
-- Purpose: Identify sources with reported vs actual chunk count mismatches
-- This directly addresses the bug we're fixing
-- =====================================================

CREATE OR REPLACE FUNCTION detect_chunk_count_discrepancies()
RETURNS TABLE(
    source_id text,
    reported_chunks_count bigint,
    actual_chunks_count bigint,
    discrepancy_size bigint
)
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    -- Find sources where reported count doesn't match actual count
    -- This function identifies the exact sources affected by the bug
    RETURN QUERY
    SELECT 
        s.source_id::text,
        -- Extract reported count from metadata (may be 0 due to bug)
        COALESCE((s.metadata->>'chunks_count')::bigint, 0) AS reported_chunks_count,
        -- Count actual chunks in documents table
        COALESCE(d.actual_count, 0) AS actual_chunks_count,
        -- Calculate discrepancy size
        ABS(COALESCE((s.metadata->>'chunks_count')::bigint, 0) - COALESCE(d.actual_count, 0)) AS discrepancy_size
    FROM archon_sources s
    LEFT JOIN (
        SELECT 
            source_id,
            COUNT(*) AS actual_count
        FROM archon_crawled_pages
        GROUP BY source_id
    ) d ON s.source_id = d.source_id
    WHERE 
        -- Only return sources with discrepancies
        COALESCE((s.metadata->>'chunks_count')::bigint, 0) != COALESCE(d.actual_count, 0)
    ORDER BY discrepancy_size DESC, s.source_id;
END;
$$;

COMMENT ON FUNCTION detect_chunk_count_discrepancies() IS
'Identifies sources where reported chunks_count differs from actual count.
Directly addresses the chunks count discrepancy bug.';


-- =====================================================
-- Function: repair_chunk_count_discrepancies
-- Purpose: Fix chunk count discrepancies by updating metadata
-- This implements the automated fix for the bug
-- =====================================================

CREATE OR REPLACE FUNCTION repair_chunk_count_discrepancies()
RETURNS TABLE(
    source_id text,
    old_count bigint,
    new_count bigint,
    repair_status text
)
LANGUAGE plpgsql
AS $$
DECLARE
    repair_record RECORD;
    repair_count bigint := 0;
    error_count bigint := 0;
BEGIN
    -- Loop through discrepancies and fix them
    FOR repair_record IN 
        SELECT * FROM detect_chunk_count_discrepancies()
    LOOP
        BEGIN
            -- Update the source metadata with correct chunk count
            UPDATE archon_sources 
            SET metadata = jsonb_set(
                COALESCE(metadata, '{}'::jsonb),
                '{chunks_count}',
                to_jsonb(repair_record.actual_chunks_count)
            ),
            updated_at = NOW()
            WHERE archon_sources.source_id = repair_record.source_id;
            
            -- Return successful repair record
            RETURN QUERY SELECT 
                repair_record.source_id::text,
                repair_record.reported_chunks_count,
                repair_record.actual_chunks_count,
                'repaired'::text;
                
            repair_count := repair_count + 1;
            
        EXCEPTION WHEN OTHERS THEN
            -- Log error and continue with other repairs
            error_count := error_count + 1;
            
            RETURN QUERY SELECT 
                repair_record.source_id::text,
                repair_record.reported_chunks_count,
                repair_record.actual_chunks_count,
                ('error: ' || SQLERRM)::text;
        END;
    END LOOP;
    
    -- Log repair summary
    RAISE NOTICE 'Chunk count repair completed: % repaired, % errors', repair_count, error_count;
END;
$$;

COMMENT ON FUNCTION repair_chunk_count_discrepancies() IS
'Automatically repairs chunk count discrepancies by updating source metadata.
Implements the core fix for the chunks count bug.';


-- =====================================================
-- Function: find_orphaned_documents
-- Purpose: Identify documents without valid source references
-- Performance: <500ms for complete database scan
-- =====================================================

CREATE OR REPLACE FUNCTION find_orphaned_documents()
RETURNS TABLE(
    document_id text,
    orphaned_source_id text,
    chunk_index bigint,
    created_at timestamptz
)
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    -- Find documents that reference non-existent sources
    RETURN QUERY
    SELECT 
        d.id::text,
        d.source_id::text,
        d.chunk_index,
        d.created_at
    FROM archon_crawled_pages d
    LEFT JOIN archon_sources s ON d.source_id = s.source_id
    WHERE s.source_id IS NULL
    ORDER BY d.created_at DESC;
END;
$$;

COMMENT ON FUNCTION find_orphaned_documents() IS
'Identifies documents that reference non-existent sources.
Used for data integrity cleanup operations.';


-- =====================================================
-- Function: cleanup_orphaned_documents
-- Purpose: Remove documents that have no valid source reference
-- =====================================================

CREATE OR REPLACE FUNCTION cleanup_orphaned_documents()
RETURNS TABLE(
    deleted_count bigint,
    failed_count bigint,
    deleted_document_ids text[]
)
LANGUAGE plpgsql
AS $$
DECLARE
    orphan_ids text[];
    deleted_ids text[] := '{}';
    delete_count bigint := 0;
    fail_count bigint := 0;
    orphan_id text;
BEGIN
    -- Get list of orphaned document IDs
    SELECT array_agg(document_id)
    INTO orphan_ids
    FROM find_orphaned_documents();
    
    -- If no orphans found, return zeros
    IF orphan_ids IS NULL THEN
        RETURN QUERY SELECT 0::bigint, 0::bigint, '{}'::text[];
        RETURN;
    END IF;
    
    -- Delete orphaned documents one by one for safety
    FOREACH orphan_id IN ARRAY orphan_ids
    LOOP
        BEGIN
            DELETE FROM archon_crawled_pages WHERE id = orphan_id;
            
            IF FOUND THEN
                deleted_ids := array_append(deleted_ids, orphan_id);
                delete_count := delete_count + 1;
            END IF;
            
        EXCEPTION WHEN OTHERS THEN
            fail_count := fail_count + 1;
            RAISE NOTICE 'Failed to delete document %: %', orphan_id, SQLERRM;
        END;
    END LOOP;
    
    RETURN QUERY SELECT delete_count, fail_count, deleted_ids;
    
    RAISE NOTICE 'Orphaned document cleanup: % deleted, % failed', delete_count, fail_count;
END;
$$;

COMMENT ON FUNCTION cleanup_orphaned_documents() IS
'Safely removes orphaned documents that have no valid source reference.
Returns detailed cleanup results.';


-- =====================================================
-- Function: rebuild_chunk_indexes
-- Purpose: Rebuild chunk indexes for a source to fix gaps/inconsistencies
-- =====================================================

CREATE OR REPLACE FUNCTION rebuild_chunk_indexes(p_source_id text)
RETURNS TABLE(
    source_id text,
    original_chunk_count bigint,
    rebuilt_chunk_count bigint,
    gaps_fixed bigint,
    new_max_index bigint
)
LANGUAGE plpgsql
AS $$
DECLARE
    original_count bigint;
    chunk_record RECORD;
    new_index bigint := 0;
    gaps_count bigint := 0;
BEGIN
    -- Get original chunk count
    SELECT COUNT(*) INTO original_count
    FROM archon_crawled_pages
    WHERE archon_crawled_pages.source_id = p_source_id;
    
    -- Rebuild indexes sequentially (0, 1, 2, 3, ...)
    FOR chunk_record IN 
        SELECT id, chunk_index AS old_index
        FROM archon_crawled_pages 
        WHERE archon_crawled_pages.source_id = p_source_id
        ORDER BY chunk_index, id  -- Maintain original order
    LOOP
        -- Check if index needs to be updated
        IF chunk_record.old_index != new_index THEN
            UPDATE archon_crawled_pages 
            SET chunk_index = new_index,
                updated_at = NOW()
            WHERE id = chunk_record.id;
            
            gaps_count := gaps_count + 1;
        END IF;
        
        new_index := new_index + 1;
    END LOOP;
    
    -- Return rebuild results
    RETURN QUERY SELECT 
        p_source_id::text,
        original_count,
        new_index,  -- This is the new count (0-indexed, so max = count-1)
        gaps_count,
        CASE WHEN new_index > 0 THEN new_index - 1 ELSE 0 END; -- Max index (0-based)
END;
$$;

COMMENT ON FUNCTION rebuild_chunk_indexes(text) IS
'Rebuilds chunk indexes for a source to eliminate gaps and inconsistencies.
Ensures sequential chunk indexing starting from 0.';


-- =====================================================
-- Performance Optimization Indexes
-- These indexes support the functions above for optimal performance
-- =====================================================

-- Index for fast source_id lookups in documents table (primary operation)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_crawled_pages_source_id 
ON archon_crawled_pages(source_id);

-- Index for fast chunk counting operations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_crawled_pages_source_chunk 
ON archon_crawled_pages(source_id, chunk_index);

-- Index for metadata operations on sources
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_sources_metadata_chunks 
ON archon_sources USING gin((metadata->>'chunks_count'));

-- Index for finding orphaned documents efficiently  
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_crawled_pages_orphan_check
ON archon_crawled_pages(source_id, id) WHERE source_id IS NOT NULL;

-- Add statistics collection for query optimization
ANALYZE archon_crawled_pages;
ANALYZE archon_sources;


-- =====================================================  
-- Function: get_chunks_count_performance_stats
-- Purpose: Monitor performance of chunk counting operations
-- =====================================================

CREATE OR REPLACE FUNCTION get_chunks_count_performance_stats()
RETURNS TABLE(
    total_sources bigint,
    total_chunks bigint,
    avg_chunks_per_source numeric,
    sources_with_most_chunks jsonb,
    performance_metrics jsonb
)
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    start_time timestamp;
    end_time timestamp;
    perf_data jsonb;
BEGIN
    start_time := clock_timestamp();
    
    -- Calculate comprehensive statistics
    RETURN QUERY
    SELECT 
        (SELECT COUNT(*) FROM archon_sources)::bigint,
        (SELECT COUNT(*) FROM archon_crawled_pages)::bigint,
        (SELECT 
            CASE 
                WHEN COUNT(*) > 0 THEN 
                    ROUND((SELECT COUNT(*) FROM archon_crawled_pages)::numeric / COUNT(*)::numeric, 2)
                ELSE 0 
            END
        FROM archon_sources),
        (SELECT jsonb_agg(
            jsonb_build_object(
                'source_id', source_id,
                'chunk_count', chunk_count
            )
        )
        FROM (
            SELECT source_id, COUNT(*) as chunk_count
            FROM archon_crawled_pages
            GROUP BY source_id
            ORDER BY chunk_count DESC
            LIMIT 10
        ) top_sources),
        jsonb_build_object(
            'query_duration_ms', EXTRACT(MILLISECONDS FROM clock_timestamp() - start_time),
            'indexes_available', (
                SELECT COUNT(*)
                FROM pg_indexes 
                WHERE tablename IN ('archon_crawled_pages', 'archon_sources')
                AND indexname LIKE 'idx_archon_%'
            ),
            'last_analyzed', (
                SELECT MAX(last_analyze)
                FROM pg_stat_user_tables
                WHERE relname IN ('archon_crawled_pages', 'archon_sources')
            )
        );
END;
$$;

COMMENT ON FUNCTION get_chunks_count_performance_stats() IS
'Provides performance statistics and monitoring data for chunk count operations.
Useful for debugging and optimization.';


-- =====================================================
-- Grant permissions for the functions
-- =====================================================

-- Grant execute permissions to the application user
-- Adjust role name as needed for your setup
GRANT EXECUTE ON FUNCTION get_chunks_count_bulk(text[]) TO postgres;
GRANT EXECUTE ON FUNCTION get_chunks_count_single(text) TO postgres;
GRANT EXECUTE ON FUNCTION validate_chunks_integrity() TO postgres;
GRANT EXECUTE ON FUNCTION detect_chunk_count_discrepancies() TO postgres;
GRANT EXECUTE ON FUNCTION repair_chunk_count_discrepancies() TO postgres;
GRANT EXECUTE ON FUNCTION find_orphaned_documents() TO postgres;
GRANT EXECUTE ON FUNCTION cleanup_orphaned_documents() TO postgres;
GRANT EXECUTE ON FUNCTION rebuild_chunk_indexes(text) TO postgres;
GRANT EXECUTE ON FUNCTION get_chunks_count_performance_stats() TO postgres;

-- Add final comment
COMMENT ON SCHEMA public IS 'Schema contains optimized functions for chunks count operations with performance targets: get_chunks_count_bulk <100ms for 50 sources, validate_chunks_integrity <2s for all sources.';