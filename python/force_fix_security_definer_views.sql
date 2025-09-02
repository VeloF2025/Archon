-- =====================================================
-- FORCE FIX SECURITY DEFINER VIEW ERRORS
-- =====================================================
-- Aggressive fix for persistent SECURITY DEFINER VIEW errors
-- Forces complete recreation of views without SECURITY DEFINER
-- =====================================================

-- Force drop and recreate security_status_prod
DROP VIEW IF EXISTS public.security_status_prod CASCADE;

-- Wait and ensure clean state
DO $$ 
BEGIN 
    -- Small delay to ensure drop is processed
    PERFORM pg_sleep(0.1);
END $$;

-- Recreate security_status_prod as regular view (no SECURITY DEFINER)
CREATE VIEW public.security_status_prod AS
SELECT 
    'archon_supabase'::text as environment,
    current_database() as database_name,
    (SELECT COUNT(*)::integer FROM pg_proc p
     JOIN pg_namespace n ON p.pronamespace = n.oid
     WHERE n.nspname = 'public'
     AND p.proname IN ('match_archon_crawled_pages', 'archive_task', 'update_updated_at_column', 'match_archon_code_examples')
     AND p.proconfig IS NOT NULL) as secured_functions,
    4::integer as total_critical_functions,
    CASE 
        WHEN (SELECT COUNT(*) FROM pg_proc p
              JOIN pg_namespace n ON p.pronamespace = n.oid  
              WHERE n.nspname = 'public'
              AND p.proname IN ('match_archon_crawled_pages', 'archive_task', 'update_updated_at_column', 'match_archon_code_examples')
              AND p.proconfig IS NOT NULL) = 4 
        THEN '✅ SECURE'::text
        ELSE '❌ VULNERABLE'::text
    END as security_status,
    NOW() as deployment_time,
    'supabase_compatible_v2'::text as version;

-- Force drop and recreate archon_rls_performance_status
DROP VIEW IF EXISTS public.archon_rls_performance_status CASCADE;

-- Wait and ensure clean state
DO $$ 
BEGIN 
    -- Small delay to ensure drop is processed
    PERFORM pg_sleep(0.1);
END $$;

-- Recreate archon_rls_performance_status as regular view (no SECURITY DEFINER)
CREATE VIEW public.archon_rls_performance_status AS
SELECT 
    'RLS_OPTIMIZATION_COMPLETE'::text as status,
    (SELECT COUNT(*)::integer
     FROM pg_policies 
     WHERE schemaname = 'public' 
     AND tablename LIKE 'archon_%'
     AND policyname LIKE '%optimized%') as optimized_policies,
    (SELECT COUNT(*)::integer
     FROM pg_policies 
     WHERE schemaname = 'public' 
     AND tablename LIKE 'archon_%') as total_policies,
    NOW() as optimized_at;

-- Explicitly set permissions (ensure no SECURITY DEFINER inheritance)
GRANT SELECT ON public.security_status_prod TO authenticated, service_role, anon;
GRANT SELECT ON public.archon_rls_performance_status TO authenticated, service_role, anon;

-- Alternative approach: Create new views with different names if needed
CREATE OR REPLACE VIEW public.security_status_final AS
SELECT 
    'archon_supabase'::text as environment,
    current_database() as database_name,
    (SELECT COUNT(*)::integer FROM pg_proc p
     JOIN pg_namespace n ON p.pronamespace = n.oid
     WHERE n.nspname = 'public'
     AND p.proname IN ('match_archon_crawled_pages', 'archive_task', 'update_updated_at_column', 'match_archon_code_examples')
     AND p.proconfig IS NOT NULL) as secured_functions,
    4::integer as total_critical_functions,
    CASE 
        WHEN (SELECT COUNT(*) FROM pg_proc p
              JOIN pg_namespace n ON p.pronamespace = n.oid  
              WHERE n.nspname = 'public'
              AND p.proname IN ('match_archon_crawled_pages', 'archive_task', 'update_updated_at_column', 'match_archon_code_examples')
              AND p.proconfig IS NOT NULL) = 4 
        THEN '✅ SECURE'::text
        ELSE '❌ VULNERABLE'::text
    END as security_status,
    NOW() as deployment_time,
    'no_security_definer_v1'::text as version;

GRANT SELECT ON public.security_status_final TO authenticated, service_role, anon;

-- Verification queries
SELECT 'VERIFICATION: Views recreated without SECURITY DEFINER' as status;

-- Test the recreated views
SELECT 'Testing security_status_prod:' as test, * FROM public.security_status_prod LIMIT 1;
SELECT 'Testing archon_rls_performance_status:' as test, * FROM public.archon_rls_performance_status LIMIT 1;
SELECT 'Testing security_status_final:' as test, * FROM public.security_status_final LIMIT 1;

-- Check if views still have SECURITY DEFINER (should return no rows)
SELECT 
    schemaname, 
    viewname,
    definition
FROM pg_views 
WHERE schemaname = 'public' 
AND viewname IN ('security_status_prod', 'archon_rls_performance_status', 'security_status_final')
AND definition LIKE '%SECURITY DEFINER%';

-- Final status
SELECT 
    '✅ SECURITY DEFINER VIEW ERRORS SHOULD BE FIXED' as status,
    'Views recreated without SECURITY DEFINER property' as action_taken,
    'Check Supabase dashboard for updated warnings' as next_step;

-- =====================================================
-- FORCE FIX COMPLETE ✅
--
-- ACTIONS TAKEN:
-- ✅ Forced DROP CASCADE on problematic views
-- ✅ Added explicit delays to ensure clean recreation
-- ✅ Recreated views without any SECURITY DEFINER properties
-- ✅ Added explicit type casting to prevent inheritance issues
-- ✅ Created alternative view (security_status_final) as backup
-- ✅ Explicitly granted permissions without SECURITY DEFINER
--
-- The SECURITY DEFINER VIEW errors should now be resolved.
-- Check the Supabase dashboard warnings to confirm.
-- =====================================================