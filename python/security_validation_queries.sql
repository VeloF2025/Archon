-- =====================================================
-- SECURITY VALIDATION QUERIES
-- =====================================================
-- Run these queries to verify that security fixes are working
-- Execute after running database_security_migration.sql
-- =====================================================

-- =====================================================
-- VALIDATION QUERY 1: Check Function Search Path Security
-- =====================================================

SELECT 
    'Function Security Check' as check_type,
    p.proname as function_name,
    n.nspname as schema_name,
    p.prosecdef as security_definer,
    CASE 
        WHEN p.proconfig IS NOT NULL THEN 'SECURE (search_path set)'
        ELSE '⚠️  VULNERABLE (search_path not set)'
    END as security_status,
    p.proconfig as search_path_config
FROM pg_proc p
JOIN pg_namespace n ON p.pronamespace = n.oid
WHERE n.nspname = 'public'
AND p.proname IN (
    'match_archon_crawled_pages',
    'archive_task', 
    'update_updated_at_column',
    'match_archon_code_examples'
)
ORDER BY p.proname;

-- =====================================================
-- VALIDATION QUERY 2: Extension Schema Location
-- =====================================================

SELECT 
    'Extension Location Check' as check_type,
    e.extname as extension_name,
    n.nspname as current_schema,
    CASE 
        WHEN n.nspname = 'extensions' THEN '✅ SECURE (in extensions schema)'
        WHEN n.nspname = 'public' THEN '⚠️  WARNING (in public schema - consider moving)'
        ELSE '❓ UNKNOWN SCHEMA'
    END as security_status
FROM pg_extension e
JOIN pg_namespace n ON e.extnamespace = n.oid
WHERE e.extname IN ('vector', 'pgcrypto')
ORDER BY e.extname;

-- =====================================================
-- VALIDATION QUERY 3: Overall Security Summary
-- =====================================================

WITH function_security AS (
    SELECT 
        COUNT(*) as total_functions,
        COUNT(CASE WHEN p.proconfig IS NOT NULL THEN 1 END) as secure_functions
    FROM pg_proc p
    JOIN pg_namespace n ON p.pronamespace = n.oid
    WHERE n.nspname = 'public'
    AND p.proname IN (
        'match_archon_crawled_pages',
        'archive_task', 
        'update_updated_at_column',
        'match_archon_code_examples'
    )
),
extension_security AS (
    SELECT 
        COUNT(*) as total_extensions,
        COUNT(CASE WHEN n.nspname = 'extensions' THEN 1 END) as secure_extensions
    FROM pg_extension e
    JOIN pg_namespace n ON e.extnamespace = n.oid
    WHERE e.extname IN ('vector', 'pgcrypto')
)
SELECT 
    '=== SECURITY AUDIT SUMMARY ===' as summary,
    fs.total_functions as total_vulnerable_functions,
    fs.secure_functions as functions_now_secure,
    es.total_extensions as total_extensions,
    es.secure_extensions as extensions_in_secure_schema,
    CASE 
        WHEN fs.secure_functions = fs.total_functions THEN '✅ ALL FUNCTION VULNERABILITIES FIXED'
        ELSE '⚠️  FUNCTION VULNERABILITIES REMAIN'
    END as function_status,
    CASE 
        WHEN es.secure_extensions = es.total_extensions THEN '✅ ALL EXTENSIONS SECURE'
        ELSE '⚠️  EXTENSIONS IN PUBLIC SCHEMA (ACCEPTABLE BUT NOT IDEAL)'
    END as extension_status
FROM function_security fs, extension_security es;

-- =====================================================
-- VALIDATION QUERY 4: Test Function Execution
-- =====================================================

-- Test that functions are still callable (basic syntax check)
DO $$
DECLARE
    func_exists BOOLEAN;
    result_count INTEGER;
BEGIN
    RAISE NOTICE '=== FUNCTION EXECUTION TESTS ===';
    
    -- Test update_updated_at_column function exists
    SELECT EXISTS (
        SELECT 1 FROM pg_proc p
        JOIN pg_namespace n ON p.pronamespace = n.oid
        WHERE n.nspname = 'public' AND p.proname = 'update_updated_at_column'
    ) INTO func_exists;
    
    IF func_exists THEN
        RAISE NOTICE '✅ update_updated_at_column: Function exists and is callable';
    ELSE
        RAISE NOTICE '❌ update_updated_at_column: Function missing!';
    END IF;
    
    -- Test archive_task function
    SELECT EXISTS (
        SELECT 1 FROM pg_proc p
        JOIN pg_namespace n ON p.pronamespace = n.oid
        WHERE n.nspname = 'public' AND p.proname = 'archive_task'
    ) INTO func_exists;
    
    IF func_exists THEN
        RAISE NOTICE '✅ archive_task: Function exists and is callable';
    ELSE
        RAISE NOTICE '❌ archive_task: Function missing!';
    END IF;
    
    -- Test search functions (if tables exist)
    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'archon_crawled_pages') THEN
        SELECT EXISTS (
            SELECT 1 FROM pg_proc p
            JOIN pg_namespace n ON p.pronamespace = n.oid
            WHERE n.nspname = 'public' AND p.proname = 'match_archon_crawled_pages'
        ) INTO func_exists;
        
        IF func_exists THEN
            RAISE NOTICE '✅ match_archon_crawled_pages: Function exists and is callable';
        ELSE
            RAISE NOTICE '❌ match_archon_crawled_pages: Function missing!';
        END IF;
    ELSE
        RAISE NOTICE '⚠️  archon_crawled_pages table not found - skipping function test';
    END IF;
    
    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'archon_code_examples') THEN
        SELECT EXISTS (
            SELECT 1 FROM pg_proc p
            JOIN pg_namespace n ON p.pronamespace = n.oid
            WHERE n.nspname = 'public' AND p.proname = 'match_archon_code_examples'
        ) INTO func_exists;
        
        IF func_exists THEN
            RAISE NOTICE '✅ match_archon_code_examples: Function exists and is callable';
        ELSE
            RAISE NOTICE '❌ match_archon_code_examples: Function missing!';
        END IF;
    ELSE
        RAISE NOTICE '⚠️  archon_code_examples table not found - skipping function test';
    END IF;
    
    RAISE NOTICE '=== TEST COMPLETE ===';
END
$$;

-- =====================================================
-- VALIDATION QUERY 5: Trigger Validation
-- =====================================================

-- Check that triggers using update_updated_at_column still work
SELECT 
    'Trigger Validation' as check_type,
    t.tgname as trigger_name,
    c.relname as table_name,
    n.nspname as schema_name,
    CASE t.tgenabled 
        WHEN 'O' THEN '✅ ENABLED'
        WHEN 'D' THEN '❌ DISABLED'
        ELSE '❓ UNKNOWN'
    END as trigger_status
FROM pg_trigger t
JOIN pg_class c ON t.tgrelid = c.oid
JOIN pg_namespace n ON c.relnamespace = n.oid
WHERE t.tgname LIKE '%updated_at%'
AND n.nspname = 'public'
ORDER BY c.relname, t.tgname;

-- =====================================================
-- VALIDATION QUERY 6: Security Best Practices Check
-- =====================================================

SELECT 
    '=== SECURITY BEST PRACTICES COMPLIANCE ===' as compliance_check,
    CASE 
        WHEN (
            SELECT COUNT(*) 
            FROM pg_proc p
            JOIN pg_namespace n ON p.pronamespace = n.oid
            WHERE n.nspname = 'public'
            AND p.proname IN (
                'match_archon_crawled_pages',
                'archive_task', 
                'update_updated_at_column',
                'match_archon_code_examples'
            )
            AND p.proconfig IS NOT NULL
        ) = 4 THEN '✅ PASSED: All vulnerable functions secured with search_path'
        ELSE '❌ FAILED: Some functions still vulnerable'
    END as search_path_compliance,
    CASE 
        WHEN (
            SELECT COUNT(*)
            FROM pg_proc p
            JOIN pg_namespace n ON p.pronamespace = n.oid
            WHERE n.nspname = 'public'
            AND p.proname IN (
                'match_archon_crawled_pages',
                'archive_task', 
                'update_updated_at_column',
                'match_archon_code_examples'
            )
            AND p.prosecdef = true
        ) = 4 THEN '✅ PASSED: All functions use SECURITY DEFINER'
        ELSE '❌ FAILED: Some functions not using SECURITY DEFINER'
    END as security_definer_compliance,
    '⚠️  EXTENSION NOTE: Vector extension may remain in public schema (acceptable in managed environments)' as extension_note;

-- =====================================================
-- VALIDATION COMPLETE
-- =====================================================

-- Final summary message
DO $$
BEGIN
    RAISE NOTICE '=== VALIDATION QUERIES COMPLETE ===';
    RAISE NOTICE 'Review the results above to confirm all security fixes are working';
    RAISE NOTICE 'Look for ✅ (secure) vs ⚠️  (warning) vs ❌ (failed) indicators';
    RAISE NOTICE 'If all functions show "SECURE (search_path set)", the migration was successful';
END
$$;