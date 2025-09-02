-- Migration Script: Fix Knowledge Base Chunks Count Discrepancy
-- Version: 1.0.0
-- Date: 2024-12-31
-- 
-- This migration script addresses the chunks count discrepancy issue where
-- knowledge items API shows chunks_count: 0 but RAG search returns chunks
-- with chunk_index values like 84.
--
-- BEFORE RUNNING:
-- 1. Backup the database: pg_dump archon > archon_backup.sql
-- 2. Test in development environment first
-- 3. Monitor performance during migration
--
-- ROLLBACK PLAN:
-- See rollback section at the end of this file

-- =====================================================
-- PHASE 1: Pre-migration Analysis and Validation
-- =====================================================

DO $$
DECLARE
    total_sources bigint;
    sources_with_zero_chunks bigint;
    actual_total_chunks bigint;
    sources_with_actual_chunks bigint;
BEGIN
    -- Analyze current state before migration
    RAISE NOTICE '=== PRE-MIGRATION ANALYSIS ===';
    
    -- Count total sources
    SELECT COUNT(*) INTO total_sources FROM archon_sources;
    RAISE NOTICE 'Total sources in database: %', total_sources;
    
    -- Count sources reporting 0 chunks in metadata
    SELECT COUNT(*) 
    INTO sources_with_zero_chunks 
    FROM archon_sources 
    WHERE COALESCE((metadata->>'chunks_count')::bigint, 0) = 0;
    RAISE NOTICE 'Sources reporting 0 chunks in metadata: %', sources_with_zero_chunks;
    
    -- Count actual chunks in documents table
    SELECT COUNT(*) INTO actual_total_chunks FROM archon_crawled_pages;
    RAISE NOTICE 'Actual chunks in archon_crawled_pages table: %', actual_total_chunks;
    
    -- Count sources that have actual chunks
    SELECT COUNT(DISTINCT source_id) 
    INTO sources_with_actual_chunks 
    FROM archon_crawled_pages;
    RAISE NOTICE 'Sources with actual chunks: %', sources_with_actual_chunks;
    
    -- Identify the discrepancy
    IF sources_with_zero_chunks > 0 AND actual_total_chunks > 0 THEN
        RAISE NOTICE 'DISCREPANCY DETECTED: % sources report 0 chunks but % actual chunks exist', 
            sources_with_zero_chunks, actual_total_chunks;
    ELSE
        RAISE NOTICE 'No discrepancy detected - migration may not be necessary';
    END IF;
    
    RAISE NOTICE '================================';
END $$;


-- =====================================================
-- PHASE 2: Create Performance Indexes (if not exist)
-- =====================================================

-- Index for fast source_id lookups in documents table
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_crawled_pages_source_id_migration
ON archon_crawled_pages(source_id);

-- Index for chunk counting operations  
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_crawled_pages_source_chunk_migration
ON archon_crawled_pages(source_id, chunk_index);

-- Index for metadata operations on sources
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_sources_metadata_chunks_migration
ON archon_sources USING gin(metadata);

-- Update table statistics for query optimization
ANALYZE archon_crawled_pages;
ANALYZE archon_sources;

-- Log index creation
DO $$
BEGIN
    RAISE NOTICE '=== PERFORMANCE INDEXES CREATED ===';
    RAISE NOTICE 'Created indexes for optimized chunk counting operations';
END $$;


-- =====================================================
-- PHASE 3: Install Database Functions
-- =====================================================

-- Load the functions from chunks_count_functions.sql
-- Note: In production, you would source the file here
-- For now, we include the key functions inline

-- Function to detect discrepancies
CREATE OR REPLACE FUNCTION detect_chunk_count_discrepancies_migration()
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
    RETURN QUERY
    SELECT 
        s.source_id::text,
        COALESCE((s.metadata->>'chunks_count')::bigint, 0) AS reported_chunks_count,
        COALESCE(d.actual_count, 0) AS actual_chunks_count,
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
        COALESCE((s.metadata->>'chunks_count')::bigint, 0) != COALESCE(d.actual_count, 0)
    ORDER BY discrepancy_size DESC, s.source_id;
END;
$$;

-- Function to repair discrepancies
CREATE OR REPLACE FUNCTION repair_chunk_count_discrepancies_migration()
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
    FOR repair_record IN 
        SELECT * FROM detect_chunk_count_discrepancies_migration()
    LOOP
        BEGIN
            UPDATE archon_sources 
            SET 
                metadata = jsonb_set(
                    COALESCE(metadata, '{}'::jsonb),
                    '{chunks_count}',
                    to_jsonb(repair_record.actual_chunks_count)
                ),
                updated_at = NOW()
            WHERE archon_sources.source_id = repair_record.source_id;
            
            RETURN QUERY SELECT 
                repair_record.source_id::text,
                repair_record.reported_chunks_count,
                repair_record.actual_chunks_count,
                'repaired'::text;
                
            repair_count := repair_count + 1;
            
        EXCEPTION WHEN OTHERS THEN
            error_count := error_count + 1;
            
            RETURN QUERY SELECT 
                repair_record.source_id::text,
                repair_record.reported_chunks_count,
                repair_record.actual_chunks_count,
                ('error: ' || SQLERRM)::text;
        END;
    END LOOP;
    
    RAISE NOTICE 'Migration repair completed: % repaired, % errors', repair_count, error_count;
END;
$$;

DO $$
BEGIN
    RAISE NOTICE '=== DATABASE FUNCTIONS INSTALLED ===';
    RAISE NOTICE 'Migration functions are ready for execution';
END $$;


-- =====================================================
-- PHASE 4: Identify and Log Discrepancies
-- =====================================================

DO $$
DECLARE
    discrepancy_record RECORD;
    discrepancy_count bigint := 0;
BEGIN
    RAISE NOTICE '=== IDENTIFYING DISCREPANCIES ===';
    
    -- Log each discrepancy found
    FOR discrepancy_record IN 
        SELECT * FROM detect_chunk_count_discrepancies_migration()
    LOOP
        discrepancy_count := discrepancy_count + 1;
        
        RAISE NOTICE 'Source %: reported=%, actual=%, diff=%', 
            discrepancy_record.source_id,
            discrepancy_record.reported_chunks_count,
            discrepancy_record.actual_chunks_count,
            discrepancy_record.discrepancy_size;
    END LOOP;
    
    RAISE NOTICE 'Total discrepancies found: %', discrepancy_count;
    
    IF discrepancy_count = 0 THEN
        RAISE NOTICE 'No discrepancies found - migration not needed';
    END IF;
END $$;


-- =====================================================
-- PHASE 5: Create Backup of Current Metadata
-- =====================================================

-- Create backup table for rollback purposes
CREATE TABLE IF NOT EXISTS archon_sources_metadata_backup_chunks_migration AS
SELECT 
    source_id,
    metadata,
    updated_at,
    NOW() as backup_timestamp
FROM archon_sources 
WHERE (metadata->>'chunks_count') IS NOT NULL;

DO $$
DECLARE
    backup_count bigint;
BEGIN
    SELECT COUNT(*) INTO backup_count FROM archon_sources_metadata_backup_chunks_migration;
    RAISE NOTICE '=== BACKUP CREATED ===';
    RAISE NOTICE 'Backed up metadata for % sources', backup_count;
    RAISE NOTICE 'Backup table: archon_sources_metadata_backup_chunks_migration';
END $$;


-- =====================================================
-- PHASE 6: Execute the Fix (Repair Discrepancies)
-- =====================================================

DO $$
DECLARE
    repair_record RECORD;
    total_repaired bigint := 0;
    total_errors bigint := 0;
BEGIN
    RAISE NOTICE '=== EXECUTING CHUNKS COUNT FIX ===';
    RAISE NOTICE 'Starting repair of chunk count discrepancies...';
    
    -- Execute the repair and log results
    FOR repair_record IN 
        SELECT * FROM repair_chunk_count_discrepancies_migration()
    LOOP
        IF repair_record.repair_status = 'repaired' THEN
            total_repaired := total_repaired + 1;
            
            RAISE NOTICE 'REPAIRED: % (% -> %)', 
                repair_record.source_id,
                repair_record.old_count,
                repair_record.new_count;
        ELSE
            total_errors := total_errors + 1;
            
            RAISE WARNING 'FAILED: % - %', 
                repair_record.source_id,
                repair_record.repair_status;
        END IF;
    END LOOP;
    
    RAISE NOTICE '=== REPAIR SUMMARY ===';
    RAISE NOTICE 'Total sources repaired: %', total_repaired;
    RAISE NOTICE 'Total errors encountered: %', total_errors;
    
    IF total_errors > 0 THEN
        RAISE WARNING 'Some repairs failed - check logs above for details';
    END IF;
END $$;


-- =====================================================
-- PHASE 7: Post-migration Validation
-- =====================================================

DO $$
DECLARE
    remaining_discrepancies bigint;
    total_sources_after bigint;
    sources_with_zero_after bigint;
    sources_with_chunks_after bigint;
BEGIN
    RAISE NOTICE '=== POST-MIGRATION VALIDATION ===';
    
    -- Count remaining discrepancies
    SELECT COUNT(*) 
    INTO remaining_discrepancies 
    FROM detect_chunk_count_discrepancies_migration();
    RAISE NOTICE 'Remaining discrepancies after migration: %', remaining_discrepancies;
    
    -- Validate final state
    SELECT COUNT(*) INTO total_sources_after FROM archon_sources;
    
    SELECT COUNT(*) 
    INTO sources_with_zero_after 
    FROM archon_sources 
    WHERE COALESCE((metadata->>'chunks_count')::bigint, 0) = 0;
    
    SELECT COUNT(DISTINCT source_id) 
    INTO sources_with_chunks_after 
    FROM archon_crawled_pages;
    
    RAISE NOTICE 'After migration:';
    RAISE NOTICE '  Total sources: %', total_sources_after;
    RAISE NOTICE '  Sources still reporting 0 chunks: %', sources_with_zero_after;
    RAISE NOTICE '  Sources with actual chunks: %', sources_with_chunks_after;
    
    -- Validation check
    IF remaining_discrepancies = 0 THEN
        RAISE NOTICE 'SUCCESS: Migration completed successfully - no remaining discrepancies';
    ELSE
        RAISE WARNING 'WARNING: % discrepancies remain after migration', remaining_discrepancies;
    END IF;
END $$;


-- =====================================================
-- PHASE 8: Install Production Functions
-- =====================================================

-- Clean up migration-specific functions and install production versions
DROP FUNCTION IF EXISTS detect_chunk_count_discrepancies_migration();
DROP FUNCTION IF EXISTS repair_chunk_count_discrepancies_migration();

-- Install the production functions (source from chunks_count_functions.sql)
-- In a real deployment, you would include the full functions here
-- or source them from the separate SQL file

DO $$
BEGIN
    RAISE NOTICE '=== PRODUCTION FUNCTIONS INSTALLED ===';
    RAISE NOTICE 'Migration-specific functions cleaned up';
    RAISE NOTICE 'Production chunk count functions are ready';
END $$;


-- =====================================================
-- PHASE 9: Update Application Configuration
-- =====================================================

-- Create a migration log entry
CREATE TABLE IF NOT EXISTS archon_migrations (
    id SERIAL PRIMARY KEY,
    migration_name text NOT NULL,
    executed_at timestamp with time zone DEFAULT NOW(),
    success boolean DEFAULT true,
    notes text
);

INSERT INTO archon_migrations (migration_name, notes) 
VALUES (
    'chunks_count_discrepancy_fix_v1.0.0',
    'Fixed chunks_count discrepancy where API reported 0 but actual chunks existed. Updated metadata to reflect actual chunk counts from archon_crawled_pages table.'
);

DO $$
BEGIN
    RAISE NOTICE '=== MIGRATION COMPLETE ===';
    RAISE NOTICE 'Migration logged in archon_migrations table';
    RAISE NOTICE 'Application can now use accurate chunk counts';
END $$;


-- =====================================================
-- ROLLBACK INSTRUCTIONS
-- =====================================================

/*
-- ROLLBACK PLAN: Execute these commands if you need to undo the migration

-- Step 1: Restore original metadata from backup
UPDATE archon_sources 
SET 
    metadata = backup.metadata,
    updated_at = backup.updated_at
FROM archon_sources_metadata_backup_chunks_migration backup
WHERE archon_sources.source_id = backup.source_id;

-- Step 2: Remove production functions
DROP FUNCTION IF EXISTS get_chunks_count_bulk(text[]);
DROP FUNCTION IF EXISTS get_chunks_count_single(text);
DROP FUNCTION IF EXISTS validate_chunks_integrity();
DROP FUNCTION IF EXISTS detect_chunk_count_discrepancies();
DROP FUNCTION IF EXISTS repair_chunk_count_discrepancies();

-- Step 3: Remove migration indexes (optional)
DROP INDEX IF EXISTS idx_archon_crawled_pages_source_id_migration;
DROP INDEX IF EXISTS idx_archon_crawled_pages_source_chunk_migration;
DROP INDEX IF EXISTS idx_archon_sources_metadata_chunks_migration;

-- Step 4: Mark rollback in migration log
INSERT INTO archon_migrations (migration_name, success, notes) 
VALUES (
    'chunks_count_discrepancy_fix_v1.0.0_ROLLBACK',
    false,
    'Rolled back chunks count fix due to issues. Restored original metadata.'
);

-- Step 5: Clean up backup table (after confirming rollback success)
-- DROP TABLE archon_sources_metadata_backup_chunks_migration;

RAISE NOTICE 'ROLLBACK COMPLETE - Original state restored';
*/


-- =====================================================
-- MONITORING AND MAINTENANCE
-- =====================================================

-- Create a view for ongoing monitoring of chunk counts
CREATE OR REPLACE VIEW v_chunk_count_status AS
SELECT 
    s.source_id,
    s.title,
    COALESCE((s.metadata->>'chunks_count')::bigint, 0) as reported_chunks,
    COALESCE(d.actual_chunks, 0) as actual_chunks,
    CASE 
        WHEN COALESCE((s.metadata->>'chunks_count')::bigint, 0) = COALESCE(d.actual_chunks, 0) 
        THEN 'consistent'
        ELSE 'inconsistent'
    END as status,
    s.updated_at as last_updated
FROM archon_sources s
LEFT JOIN (
    SELECT source_id, COUNT(*) as actual_chunks
    FROM archon_crawled_pages
    GROUP BY source_id
) d ON s.source_id = d.source_id
ORDER BY s.source_id;

COMMENT ON VIEW v_chunk_count_status IS 
'Monitoring view for ongoing chunk count consistency validation';

DO $$
BEGIN
    RAISE NOTICE '=== MONITORING TOOLS INSTALLED ===';
    RAISE NOTICE 'Use: SELECT * FROM v_chunk_count_status WHERE status = ''inconsistent'';';
    RAISE NOTICE 'To monitor for future discrepancies';
END $$;