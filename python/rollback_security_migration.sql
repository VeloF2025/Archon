-- =====================================================
-- EMERGENCY ROLLBACK SCRIPT FOR SECURITY MIGRATION
-- =====================================================
-- ⚠️ WARNING: This script restores security vulnerabilities!
-- Only use in emergency situations where the migration
-- caused critical system failures.
--
-- After rollback, immediately plan to re-apply security fixes!
-- =====================================================

-- Log rollback initiation
DO $$
BEGIN
    RAISE NOTICE '⚠️  EMERGENCY ROLLBACK INITIATED at %', NOW();
    RAISE NOTICE '⚠️  WARNING: This will restore security vulnerabilities!';
    RAISE NOTICE '⚠️  Database: %', current_database();
    RAISE NOTICE '⚠️  User: %', current_user;
END
$$;

-- =====================================================
-- ROLLBACK SECTION 1: RESTORE ORIGINAL FUNCTIONS
-- =====================================================

-- 1. Rollback update_updated_at_column function
-- ⚠️ REMOVES SECURITY - search_path vulnerability restored
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Remove security comment
COMMENT ON FUNCTION update_updated_at_column() IS 'ROLLBACK: Security removed - search_path vulnerability restored';

-- 2. Rollback match_archon_crawled_pages function
-- ⚠️ REMOVES SECURITY - search_path vulnerability restored
CREATE OR REPLACE FUNCTION match_archon_crawled_pages (
  query_embedding VECTOR(1536),
  match_count INT DEFAULT 10,
  filter JSONB DEFAULT '{}'::jsonb,
  source_filter TEXT DEFAULT NULL
) RETURNS TABLE (
  id BIGINT,
  url VARCHAR,
  chunk_number INTEGER,
  content TEXT,
  metadata JSONB,
  source_id TEXT,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
BEGIN
  RETURN QUERY
  SELECT
    id,
    url,
    chunk_number,
    content,
    metadata,
    source_id,
    1 - (archon_crawled_pages.embedding <=> query_embedding) AS similarity
  FROM archon_crawled_pages
  WHERE metadata @> filter
    AND (source_filter IS NULL OR source_id = source_filter)
  ORDER BY archon_crawled_pages.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION match_archon_crawled_pages(VECTOR, INT, JSONB, TEXT) IS 'ROLLBACK: Security removed - search_path vulnerability restored';

-- 3. Rollback match_archon_code_examples function
-- ⚠️ REMOVES SECURITY - search_path vulnerability restored
CREATE OR REPLACE FUNCTION match_archon_code_examples (
  query_embedding VECTOR(1536),
  match_count INT DEFAULT 10,
  filter JSONB DEFAULT '{}'::jsonb,
  source_filter TEXT DEFAULT NULL
) RETURNS TABLE (
  id BIGINT,
  url VARCHAR,
  chunk_number INTEGER,
  content TEXT,
  summary TEXT,
  metadata JSONB,
  source_id TEXT,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
BEGIN
  RETURN QUERY
  SELECT
    id,
    url,
    chunk_number,
    content,
    summary,
    metadata,
    source_id,
    1 - (archon_code_examples.embedding <=> query_embedding) AS similarity
  FROM archon_code_examples
  WHERE metadata @> filter
    AND (source_filter IS NULL OR source_id = source_filter)
  ORDER BY archon_code_examples.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION match_archon_code_examples(VECTOR, INT, JSONB, TEXT) IS 'ROLLBACK: Security removed - search_path vulnerability restored';

-- 4. Rollback archive_task function
-- ⚠️ REMOVES SECURITY - search_path vulnerability restored
CREATE OR REPLACE FUNCTION archive_task(
    task_id_param UUID,
    archived_by_param TEXT DEFAULT 'system'
)
RETURNS BOOLEAN AS $$
DECLARE
    task_exists BOOLEAN;
BEGIN
    -- Check if task exists and is not already archived
    SELECT EXISTS(
        SELECT 1 FROM archon_tasks
        WHERE id = task_id_param AND archived = FALSE
    ) INTO task_exists;

    IF NOT task_exists THEN
        RETURN FALSE;
    END IF;

    -- Archive the task
    UPDATE archon_tasks
    SET
        archived = TRUE,
        archived_at = NOW(),
        archived_by = archived_by_param,
        updated_at = NOW()
    WHERE id = task_id_param;

    -- Also archive all subtasks
    UPDATE archon_tasks
    SET
        archived = TRUE,
        archived_at = NOW(),
        archived_by = archived_by_param,
        updated_at = NOW()
    WHERE parent_task_id = task_id_param AND archived = FALSE;

    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION archive_task(UUID, TEXT) IS 'ROLLBACK: Security removed - search_path vulnerability restored';

-- =====================================================
-- ROLLBACK SECTION 2: REMOVE EXTENSIONS SCHEMA
-- =====================================================

-- Note: We don't remove the extensions schema as it may be used elsewhere
-- and removing it could cause more issues than leaving it

DO $$
BEGIN
    RAISE NOTICE '⚠️  Extensions schema left intact to avoid additional issues';
    RAISE NOTICE '⚠️  Manual cleanup required if extensions schema is no longer needed';
END
$$;

-- =====================================================
-- ROLLBACK SECTION 3: VALIDATION
-- =====================================================

-- Verify rollback completed
DO $$
DECLARE
    rec RECORD;
    vulnerable_count INTEGER := 0;
    total_count INTEGER := 0;
BEGIN
    RAISE NOTICE '=== ROLLBACK VALIDATION ===';
    
    FOR rec IN 
        SELECT 
            p.proname,
            CASE WHEN p.proconfig IS NOT NULL THEN 'SECURE' ELSE 'VULNERABLE' END as security_status
        FROM pg_proc p
        JOIN pg_namespace n ON p.pronamespace = n.oid
        WHERE n.nspname = 'public'
        AND p.proname IN (
            'match_archon_crawled_pages',
            'archive_task', 
            'update_updated_at_column',
            'match_archon_code_examples'
        )
        ORDER BY p.proname
    LOOP
        total_count := total_count + 1;
        
        RAISE NOTICE 'Function % is now %', rec.proname, rec.security_status;
        
        IF rec.security_status = 'VULNERABLE' THEN
            vulnerable_count := vulnerable_count + 1;
        END IF;
    END LOOP;
    
    RAISE NOTICE '=== ROLLBACK SUMMARY ===';
    RAISE NOTICE 'Total functions: %', total_count;
    RAISE NOTICE 'Functions now vulnerable: %', vulnerable_count;
    
    IF vulnerable_count = total_count AND total_count = 4 THEN
        RAISE NOTICE '⚠️  ROLLBACK COMPLETE - All security vulnerabilities restored!';
        RAISE NOTICE '⚠️  CRITICAL: Database is now vulnerable to attack!';
        RAISE NOTICE '⚠️  Plan immediate re-application of security fixes!';
    ELSE
        RAISE WARNING '❓ Rollback may be incomplete - some functions may still be secure';
    END IF;
END
$$;

-- =====================================================
-- ROLLBACK SECTION 4: CRITICAL WARNINGS
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE '====================================';
    RAISE NOTICE '⚠️  ⚠️  ⚠️  CRITICAL WARNING ⚠️  ⚠️  ⚠️ ';
    RAISE NOTICE '====================================';
    RAISE NOTICE '';
    RAISE NOTICE 'ROLLBACK COMPLETE - SECURITY VULNERABILITIES RESTORED!';
    RAISE NOTICE '';
    RAISE NOTICE 'Your database is now vulnerable to:';
    RAISE NOTICE '• Privilege escalation attacks';
    RAISE NOTICE '• SQL injection through function hijacking';  
    RAISE NOTICE '• Data corruption or theft';
    RAISE NOTICE '• Complete system compromise';
    RAISE NOTICE '';
    RAISE NOTICE 'IMMEDIATE ACTIONS REQUIRED:';
    RAISE NOTICE '1. Identify and fix the issues that required rollback';
    RAISE NOTICE '2. Plan re-application of security fixes ASAP';
    RAISE NOTICE '3. Monitor database for suspicious activity';
    RAISE NOTICE '4. Consider restricting database access until secure';
    RAISE NOTICE '';
    RAISE NOTICE 'DO NOT LEAVE DATABASE IN THIS VULNERABLE STATE!';
    RAISE NOTICE '';
    RAISE NOTICE '====================================';
END
$$;

-- =====================================================
-- ROLLBACK SECTION 5: EMERGENCY CONTACT INFO
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE 'EMERGENCY CONTACT INFORMATION:';
    RAISE NOTICE 'Security Team: [Add contact info]';
    RAISE NOTICE 'Database Admin: [Add contact info]';
    RAISE NOTICE 'Development Lead: [Add contact info]';
    RAISE NOTICE '';
    RAISE NOTICE 'Report this rollback immediately!';
    RAISE NOTICE 'Document the issues that required rollback!';
    RAISE NOTICE 'Plan security fix re-application within 24 hours!';
END
$$;

-- Log rollback completion
DO $$
BEGIN
    RAISE NOTICE '=== EMERGENCY ROLLBACK COMPLETE ===';
    RAISE NOTICE 'Rollback completed at %', NOW();
    RAISE NOTICE '⚠️  DATABASE IS NOW VULNERABLE ⚠️ ';
    RAISE NOTICE 'Security fixes must be re-applied ASAP!';
    RAISE NOTICE '===============================';
END
$$;

-- =====================================================
-- ROLLBACK COMPLETE - DATABASE IS NOW VULNERABLE!
-- =====================================================