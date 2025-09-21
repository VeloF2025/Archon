-- Complete Security Verification Script
-- Verifies all Supabase security fixes are applied correctly

-- =====================================================
-- Security Verification Summary Report
-- =====================================================

SELECT '=== SUPABASE SECURITY VERIFICATION REPORT ===' as report_header;
SELECT 'Generated at: ' || NOW() as timestamp;

-- =====================================================
-- 1. RLS (Row Level Security) Status Check
-- =====================================================

SELECT '--- RLS STATUS CHECK ---' as section;

SELECT 
    'RLS Enabled Tables' as metric,
    COUNT(*) as count,
    'Should be 22+' as expected
FROM pg_tables pt
JOIN pg_class c ON c.relname = pt.tablename
JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = pt.schemaname
WHERE schemaname = 'public' 
  AND (tablename LIKE 'archon_%' OR tablename LIKE 'feature_%' OR tablename = 'user_feature_assignments')
  AND c.relrowsecurity = true;

-- List any tables still missing RLS
SELECT 
    'MISSING RLS' as issue_type,
    tablename as table_name,
    'Enable RLS on this table' as action_needed
FROM pg_tables pt
JOIN pg_class c ON c.relname = pt.tablename
JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = pt.schemaname
WHERE schemaname = 'public' 
  AND (tablename LIKE 'archon_%' OR tablename LIKE 'feature_%' OR tablename = 'user_feature_assignments')
  AND c.relrowsecurity = false;

-- Count RLS policies
SELECT 
    'RLS Policies Created' as metric,
    COUNT(*) as count,
    'Should be 22+' as expected
FROM pg_policies
WHERE schemaname = 'public'
  AND tablename LIKE 'archon_%' OR tablename LIKE 'feature_%';

-- =====================================================
-- 2. Security Definer Views Check
-- =====================================================

SELECT '--- SECURITY DEFINER VIEWS CHECK ---' as section;

SELECT 
    viewname as view_name,
    CASE 
        WHEN definition ILIKE '%security definer%' THEN '❌ HAS SECURITY DEFINER (BAD)' 
        ELSE '✅ SECURITY INVOKER (GOOD)'
    END as security_status
FROM pg_views 
WHERE schemaname = 'public' 
  AND viewname IN (
    'archon_cost_optimization_recommendations',
    'archon_project_intelligence_overview', 
    'archon_agent_performance_dashboard'
  )
ORDER BY viewname;

-- Count secure views
SELECT 
    'Secure Views (No Security Definer)' as metric,
    COUNT(*) as count,
    'Should be 3' as expected
FROM pg_views 
WHERE schemaname = 'public' 
  AND viewname IN (
    'archon_cost_optimization_recommendations',
    'archon_project_intelligence_overview', 
    'archon_agent_performance_dashboard'
  )
  AND definition NOT ILIKE '%security definer%';

-- =====================================================
-- 3. Function Search Path Check
-- =====================================================

SELECT '--- FUNCTION SEARCH PATH CHECK ---' as section;

SELECT 
    routine_name as function_name,
    CASE 
        WHEN proconfig IS NULL OR NOT (proconfig::text LIKE '%search_path%') THEN '❌ MUTABLE SEARCH PATH (BAD)'
        ELSE '✅ FIXED SEARCH PATH (GOOD)'
    END as search_path_status,
    CASE 
        WHEN prosecdef = true THEN 'SECURITY DEFINER'
        ELSE 'SECURITY INVOKER'
    END as security_mode
FROM information_schema.routines r
JOIN pg_proc p ON p.proname = r.routine_name
WHERE routine_schema = 'public' 
AND routine_name IN (
    'update_updated_at_column',
    'log_agent_state_transition', 
    'update_agent_pool_usage',
    'evolve_knowledge_confidence'
)
ORDER BY routine_name;

-- Count functions with fixed search path
SELECT 
    'Functions with Fixed Search Path' as metric,
    COUNT(*) as count,
    'Should be 4' as expected
FROM information_schema.routines r
JOIN pg_proc p ON p.proname = r.routine_name
WHERE routine_schema = 'public' 
AND routine_name IN (
    'update_updated_at_column',
    'log_agent_state_transition', 
    'update_agent_pool_usage',
    'evolve_knowledge_confidence'
)
AND proconfig IS NOT NULL 
AND proconfig::text LIKE '%search_path%';

-- =====================================================
-- 4. Vector Extension Location Check
-- =====================================================

SELECT '--- VECTOR EXTENSION LOCATION CHECK ---' as section;

SELECT 
    e.extname as extension_name,
    n.nspname as current_schema,
    CASE 
        WHEN n.nspname = 'public' THEN '❌ IN PUBLIC SCHEMA (BAD)'
        WHEN n.nspname = 'extensions' THEN '✅ IN EXTENSIONS SCHEMA (GOOD)'
        ELSE '⚠️ IN OTHER SCHEMA: ' || n.nspname
    END as location_status
FROM pg_extension e
JOIN pg_namespace n ON e.extnamespace = n.oid
WHERE e.extname = 'vector';

-- =====================================================
-- 5. Overall Security Score
-- =====================================================

SELECT '--- OVERALL SECURITY SCORE ---' as section;

WITH security_metrics AS (
    -- RLS enabled count
    SELECT 'rls_tables' as metric, COUNT(*) as actual, 22 as expected
    FROM pg_tables pt
    JOIN pg_class c ON c.relname = pt.tablename
    JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = pt.schemaname
    WHERE schemaname = 'public' 
      AND (tablename LIKE 'archon_%' OR tablename LIKE 'feature_%' OR tablename = 'user_feature_assignments')
      AND c.relrowsecurity = true
      
    UNION ALL
    
    -- Secure views count
    SELECT 'secure_views' as metric, COUNT(*) as actual, 3 as expected
    FROM pg_views 
    WHERE schemaname = 'public' 
      AND viewname IN ('archon_cost_optimization_recommendations', 'archon_project_intelligence_overview', 'archon_agent_performance_dashboard')
      AND definition NOT ILIKE '%security definer%'
      
    UNION ALL
    
    -- Fixed search path functions count
    SELECT 'fixed_functions' as metric, COUNT(*) as actual, 4 as expected
    FROM information_schema.routines r
    JOIN pg_proc p ON p.proname = r.routine_name
    WHERE routine_schema = 'public' 
    AND routine_name IN ('update_updated_at_column', 'log_agent_state_transition', 'update_agent_pool_usage', 'evolve_knowledge_confidence')
    AND proconfig IS NOT NULL 
    AND proconfig::text LIKE '%search_path%'
    
    UNION ALL
    
    -- Vector extension properly placed
    SELECT 'vector_extension' as metric, COUNT(*) as actual, 1 as expected
    FROM pg_extension e
    JOIN pg_namespace n ON e.extnamespace = n.oid
    WHERE e.extname = 'vector' 
    AND n.nspname != 'public'
),
score_calculation AS (
    SELECT 
        SUM(CASE WHEN actual >= expected THEN 1 ELSE 0 END) as passed_checks,
        COUNT(*) as total_checks
    FROM security_metrics
)
SELECT 
    'Security Score' as metric,
    ROUND((passed_checks::DECIMAL / total_checks * 100), 1) || '%' as score,
    passed_checks || '/' || total_checks || ' checks passed' as details
FROM score_calculation;

-- =====================================================
-- 6. Remaining Issues Summary
-- =====================================================

SELECT '--- REMAINING ISSUES SUMMARY ---' as section;

-- Any tables still missing RLS
SELECT 
    'RLS Missing' as issue_category,
    tablename as item_name,
    'ALTER TABLE ' || tablename || ' ENABLE ROW LEVEL SECURITY;' as fix_command
FROM pg_tables pt
JOIN pg_class c ON c.relname = pt.tablename
JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = pt.schemaname
WHERE schemaname = 'public' 
  AND (tablename LIKE 'archon_%' OR tablename LIKE 'feature_%' OR tablename = 'user_feature_assignments')
  AND c.relrowsecurity = false

UNION ALL

-- Any views still with security definer
SELECT 
    'Security Definer View' as issue_category,
    viewname as item_name,
    'DROP VIEW ' || viewname || '; CREATE VIEW...' as fix_command
FROM pg_views 
WHERE schemaname = 'public' 
  AND viewname IN ('archon_cost_optimization_recommendations', 'archon_project_intelligence_overview', 'archon_agent_performance_dashboard')
  AND definition ILIKE '%security definer%'

UNION ALL

-- Any functions still with mutable search path
SELECT 
    'Mutable Search Path Function' as issue_category,
    routine_name as item_name,
    'CREATE OR REPLACE FUNCTION ... SET search_path = public;' as fix_command
FROM information_schema.routines r
JOIN pg_proc p ON p.proname = r.routine_name
WHERE routine_schema = 'public' 
AND routine_name IN ('update_updated_at_column', 'log_agent_state_transition', 'update_agent_pool_usage', 'evolve_knowledge_confidence')
AND (proconfig IS NULL OR NOT (proconfig::text LIKE '%search_path%'))

UNION ALL

-- Vector extension in public schema
SELECT 
    'Extension in Public' as issue_category,
    'vector extension' as item_name,
    'Move to extensions schema' as fix_command
FROM pg_extension e
JOIN pg_namespace n ON e.extnamespace = n.oid
WHERE e.extname = 'vector' 
AND n.nspname = 'public';

-- If no issues, show success message
SELECT 
    CASE 
        WHEN EXISTS (
            SELECT 1 FROM pg_tables pt
            JOIN pg_class c ON c.relname = pt.tablename
            WHERE schemaname = 'public' 
            AND (tablename LIKE 'archon_%' OR tablename LIKE 'feature_%')
            AND c.relrowsecurity = false
        ) OR EXISTS (
            SELECT 1 FROM pg_views 
            WHERE schemaname = 'public' 
            AND viewname IN ('archon_cost_optimization_recommendations', 'archon_project_intelligence_overview', 'archon_agent_performance_dashboard')
            AND definition ILIKE '%security definer%'
        ) THEN '⚠️ Some security issues remain - see above'
        ELSE '🎉 ALL SECURITY ISSUES RESOLVED!'
    END as final_status;

SELECT '=== END OF SECURITY VERIFICATION REPORT ===' as report_footer;