-- =====================================================
-- RLS PERFORMANCE VALIDATION SCRIPT
-- =====================================================
-- This script measures and validates RLS performance improvements
-- Run BEFORE and AFTER optimization to measure exact improvements
--
-- Usage:
-- 1. Run this script BEFORE optimization (baseline measurement)
-- 2. Run the optimization script (supabase_rls_performance_fix.sql)  
-- 3. Run this script AFTER optimization (improvement measurement)
-- =====================================================

-- Create validation results table if not exists
CREATE TABLE IF NOT EXISTS rls_performance_validation (
    id SERIAL PRIMARY KEY,
    test_run_id UUID DEFAULT gen_random_uuid(),
    test_timestamp TIMESTAMPTZ DEFAULT NOW(),
    optimization_phase VARCHAR(20) CHECK (optimization_phase IN ('before', 'after')),
    table_name VARCHAR(100),
    test_type VARCHAR(50),
    execution_time_ms INTEGER,
    rows_processed INTEGER,
    policy_count INTEGER,
    notes TEXT
);

-- Function to run comprehensive performance tests
CREATE OR REPLACE FUNCTION run_rls_performance_tests(test_phase VARCHAR(20) DEFAULT 'before')
RETURNS TABLE(
    test_name VARCHAR(100),
    execution_time_ms INTEGER,
    rows_processed INTEGER,
    performance_rating VARCHAR(20),
    improvement_notes TEXT
) AS $$
DECLARE
    test_run_uuid UUID := gen_random_uuid();
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    duration_ms INTEGER;
    row_count INTEGER;
    policy_count INTEGER;
    current_policies INTEGER;
    test_results TEXT := '';
BEGIN
    RAISE NOTICE '=== RLS PERFORMANCE VALIDATION - % OPTIMIZATION ===', UPPER(test_phase);
    RAISE NOTICE 'Test Run ID: %', test_run_uuid;
    RAISE NOTICE 'Timestamp: %', NOW();
    RAISE NOTICE '';
    
    -- Count current RLS policies
    SELECT COUNT(*) INTO current_policies
    FROM pg_policy p
    JOIN pg_class c ON p.polrelid = c.oid
    JOIN pg_namespace n ON c.relnamespace = n.oid
    WHERE n.nspname = 'public' AND c.relname LIKE 'archon_%';
    
    RAISE NOTICE 'Current RLS Policy Count: %', current_policies;
    RAISE NOTICE '';
    
    -- Test 1: archon_settings table performance
    RAISE NOTICE 'Testing archon_settings performance...';
    start_time := clock_timestamp();
    SELECT COUNT(*) INTO row_count FROM archon_settings;
    end_time := clock_timestamp();
    duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    INSERT INTO rls_performance_validation (
        test_run_id, optimization_phase, table_name, test_type,
        execution_time_ms, rows_processed, policy_count, notes
    ) VALUES (
        test_run_uuid, test_phase, 'archon_settings', 'full_table_scan',
        duration_ms, row_count, current_policies, 'Settings table access test'
    );
    
    RETURN QUERY SELECT 
        'archon_settings_scan'::VARCHAR(100),
        duration_ms,
        row_count,
        CASE WHEN duration_ms < 50 THEN 'EXCELLENT'
             WHEN duration_ms < 200 THEN 'GOOD'
             WHEN duration_ms < 500 THEN 'ACCEPTABLE'
             ELSE 'POOR' END::VARCHAR(20),
        format('Settings table: %sms for %s rows', duration_ms, row_count)::TEXT;
    
    -- Test 2: archon_projects table performance  
    RAISE NOTICE 'Testing archon_projects performance...';
    start_time := clock_timestamp();
    SELECT COUNT(*) INTO row_count FROM archon_projects;
    end_time := clock_timestamp();
    duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    INSERT INTO rls_performance_validation (
        test_run_id, optimization_phase, table_name, test_type,
        execution_time_ms, rows_processed, policy_count, notes
    ) VALUES (
        test_run_uuid, test_phase, 'archon_projects', 'full_table_scan',
        duration_ms, row_count, current_policies, 'Projects table access test'
    );
    
    RETURN QUERY SELECT 
        'archon_projects_scan'::VARCHAR(100),
        duration_ms,
        row_count,
        CASE WHEN duration_ms < 50 THEN 'EXCELLENT'
             WHEN duration_ms < 200 THEN 'GOOD' 
             WHEN duration_ms < 500 THEN 'ACCEPTABLE'
             ELSE 'POOR' END::VARCHAR(20),
        format('Projects table: %sms for %s rows', duration_ms, row_count)::TEXT;
    
    -- Test 3: archon_tasks table performance (critical for PM enhancement)
    RAISE NOTICE 'Testing archon_tasks performance (PM enhancement critical path)...';
    start_time := clock_timestamp();
    SELECT COUNT(*) INTO row_count FROM archon_tasks WHERE archived = FALSE;
    end_time := clock_timestamp();
    duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    INSERT INTO rls_performance_validation (
        test_run_id, optimization_phase, table_name, test_type,
        execution_time_ms, rows_processed, policy_count, notes
    ) VALUES (
        test_run_uuid, test_phase, 'archon_tasks', 'filtered_scan',
        duration_ms, row_count, current_policies, 'Tasks table filtered scan (PM enhancement path)'
    );
    
    RETURN QUERY SELECT 
        'archon_tasks_pm_scan'::VARCHAR(100),
        duration_ms,
        row_count,
        CASE WHEN duration_ms < 100 THEN 'EXCELLENT'
             WHEN duration_ms < 300 THEN 'GOOD'
             WHEN duration_ms < 1000 THEN 'ACCEPTABLE' 
             ELSE 'POOR' END::VARCHAR(20),
        format('PM critical path: %sms for %s active tasks', duration_ms, row_count)::TEXT;
    
    -- Test 4: Complex join query (simulates PM enhancement discovery)
    RAISE NOTICE 'Testing complex PM enhancement discovery query...';
    start_time := clock_timestamp();
    SELECT COUNT(DISTINCT t.id) INTO row_count
    FROM archon_tasks t
    LEFT JOIN archon_projects p ON t.project_id = p.id
    LEFT JOIN archon_project_sources ps ON p.id = ps.project_id
    WHERE t.archived = FALSE
    AND t.status IN ('todo', 'doing', 'review');
    end_time := clock_timestamp();
    duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    INSERT INTO rls_performance_validation (
        test_run_id, optimization_phase, table_name, test_type,
        execution_time_ms, rows_processed, policy_count, notes
    ) VALUES (
        test_run_uuid, test_phase, 'multiple_tables', 'pm_enhancement_discovery',
        duration_ms, row_count, current_policies, 'Simulates PM enhancement discovery query with joins'
    );
    
    RETURN QUERY SELECT 
        'pm_enhancement_discovery'::VARCHAR(100),
        duration_ms,
        row_count,
        CASE WHEN duration_ms < 200 THEN 'EXCELLENT'
             WHEN duration_ms < 1000 THEN 'GOOD'
             WHEN duration_ms < 3000 THEN 'ACCEPTABLE'
             ELSE 'POOR' END::VARCHAR(20),
        format('PM enhancement discovery: %sms for %s tasks', duration_ms, row_count)::TEXT;
    
    -- Test 5: archon_document_versions performance
    RAISE NOTICE 'Testing archon_document_versions performance...';
    start_time := clock_timestamp();
    SELECT COUNT(*) INTO row_count FROM archon_document_versions;
    end_time := clock_timestamp();  
    duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    INSERT INTO rls_performance_validation (
        test_run_id, optimization_phase, table_name, test_type,
        execution_time_ms, rows_processed, policy_count, notes
    ) VALUES (
        test_run_uuid, test_phase, 'archon_document_versions', 'full_table_scan',
        duration_ms, row_count, current_policies, 'Document versions table access test'
    );
    
    RETURN QUERY SELECT 
        'archon_document_versions_scan'::VARCHAR(100),
        duration_ms,
        row_count,
        CASE WHEN duration_ms < 50 THEN 'EXCELLENT'
             WHEN duration_ms < 200 THEN 'GOOD'
             WHEN duration_ms < 500 THEN 'ACCEPTABLE'
             ELSE 'POOR' END::VARCHAR(20),
        format('Document versions: %sms for %s versions', duration_ms, row_count)::TEXT;
    
    -- Test 6: Policy evaluation efficiency test
    RAISE NOTICE 'Testing policy evaluation patterns...';
    start_time := clock_timestamp();
    SELECT COUNT(*) INTO row_count FROM (
        SELECT t.id FROM archon_tasks t
        UNION ALL
        SELECT p.id FROM archon_projects p  
        UNION ALL
        SELECT ps.id FROM archon_project_sources ps
        UNION ALL
        SELECT dv.id FROM archon_document_versions dv
        LIMIT 1000
    ) combined;
    end_time := clock_timestamp();
    duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    INSERT INTO rls_performance_validation (
        test_run_id, optimization_phase, table_name, test_type,
        execution_time_ms, rows_processed, policy_count, notes
    ) VALUES (
        test_run_uuid, test_phase, 'multiple_tables', 'policy_evaluation_stress',
        duration_ms, row_count, current_policies, 'Tests policy evaluation under load with multiple table access'
    );
    
    RETURN QUERY SELECT 
        'policy_evaluation_stress'::VARCHAR(100),
        duration_ms,
        row_count,
        CASE WHEN duration_ms < 100 THEN 'EXCELLENT'
             WHEN duration_ms < 400 THEN 'GOOD'
             WHEN duration_ms < 1000 THEN 'ACCEPTABLE'
             ELSE 'POOR' END::VARCHAR(20),
        format('Multi-table policy evaluation: %sms for %s records', duration_ms, row_count)::TEXT;
    
    RAISE NOTICE '';
    RAISE NOTICE '=== PERFORMANCE TEST SUMMARY (%s OPTIMIZATION) ===', UPPER(test_phase);
    RAISE NOTICE 'Total RLS Policies: %', current_policies;
    RAISE NOTICE 'Test results saved with Run ID: %', test_run_uuid;
    RAISE NOTICE '=== END PERFORMANCE VALIDATION ===';
END;
$$ LANGUAGE plpgsql;

-- Function to compare before/after results  
CREATE OR REPLACE FUNCTION compare_rls_performance_results()
RETURNS TABLE(
    test_type VARCHAR(50),
    before_ms INTEGER,
    after_ms INTEGER, 
    improvement_ms INTEGER,
    improvement_factor NUMERIC(10,2),
    performance_gain TEXT
) AS $$
DECLARE
    before_run UUID;
    after_run UUID;
BEGIN
    -- Get most recent before and after test runs
    SELECT test_run_id INTO before_run
    FROM rls_performance_validation 
    WHERE optimization_phase = 'before'
    ORDER BY test_timestamp DESC 
    LIMIT 1;
    
    SELECT test_run_id INTO after_run
    FROM rls_performance_validation
    WHERE optimization_phase = 'after' 
    ORDER BY test_timestamp DESC
    LIMIT 1;
    
    IF before_run IS NULL THEN
        RAISE EXCEPTION 'No BEFORE optimization test results found. Run: SELECT * FROM run_rls_performance_tests(''before'');';
    END IF;
    
    IF after_run IS NULL THEN
        RAISE EXCEPTION 'No AFTER optimization test results found. Run: SELECT * FROM run_rls_performance_tests(''after'');';
    END IF;
    
    RAISE NOTICE '=== RLS PERFORMANCE COMPARISON ===';
    RAISE NOTICE 'Before Run ID: %', before_run;
    RAISE NOTICE 'After Run ID: %', after_run;
    RAISE NOTICE '';
    
    RETURN QUERY
    SELECT 
        b.test_type,
        b.execution_time_ms as before_ms,
        a.execution_time_ms as after_ms,
        (b.execution_time_ms - a.execution_time_ms) as improvement_ms,
        ROUND(b.execution_time_ms::NUMERIC / GREATEST(a.execution_time_ms, 1), 2) as improvement_factor,
        CASE 
            WHEN a.execution_time_ms < b.execution_time_ms * 0.1 THEN '🚀 DRAMATIC (90%+ faster)'
            WHEN a.execution_time_ms < b.execution_time_ms * 0.25 THEN '🔥 EXCELLENT (75%+ faster)' 
            WHEN a.execution_time_ms < b.execution_time_ms * 0.5 THEN '✅ VERY GOOD (50%+ faster)'
            WHEN a.execution_time_ms < b.execution_time_ms * 0.75 THEN '👍 GOOD (25%+ faster)'
            WHEN a.execution_time_ms < b.execution_time_ms THEN '📈 IMPROVED'
            ELSE '⚠️ NO IMPROVEMENT'
        END as performance_gain
    FROM rls_performance_validation b
    JOIN rls_performance_validation a ON b.test_type = a.test_type
    WHERE b.test_run_id = before_run
    AND a.test_run_id = after_run
    ORDER BY improvement_factor DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to check for inefficient RLS patterns
CREATE OR REPLACE FUNCTION check_rls_efficiency()  
RETURNS TABLE(
    table_name TEXT,
    policy_name TEXT,
    efficiency_issue TEXT,
    recommendation TEXT
) AS $$
BEGIN
    RAISE NOTICE '=== CHECKING RLS EFFICIENCY PATTERNS ===';
    
    RETURN QUERY
    SELECT 
        c.relname::TEXT,
        p.polname::TEXT,
        CASE 
            WHEN p.polqual ~ 'auth\.role\(\)' OR p.polwithcheck ~ 'auth\.role\(\)' THEN 
                '🔴 CRITICAL: auth.role() row-by-row evaluation'
            WHEN p.polqual ~ 'auth\.uid\(\)' OR p.polwithcheck ~ 'auth\.uid\(\)' THEN
                '🟡 WARNING: auth.uid() row-by-row evaluation' 
            ELSE
                '✅ EFFICIENT: Optimized pattern detected'
        END::TEXT,
        CASE
            WHEN p.polqual ~ 'auth\.role\(\)' OR p.polwithcheck ~ 'auth\.role\(\)' THEN
                'Replace auth.role() with (SELECT auth.role()) for set-based evaluation'
            WHEN p.polqual ~ 'auth\.uid\(\)' OR p.polwithcheck ~ 'auth\.uid\(\)' THEN  
                'Consider replacing auth.uid() with (SELECT auth.uid()) for better performance'
            ELSE
                'Policy is already optimized'
        END::TEXT
    FROM pg_policy p
    JOIN pg_class c ON p.polrelid = c.oid
    JOIN pg_namespace n ON c.relnamespace = n.oid
    WHERE n.nspname = 'public' AND c.relname LIKE 'archon_%'
    ORDER BY 
        CASE WHEN p.polqual ~ 'auth\.role\(\)' OR p.polwithcheck ~ 'auth\.role\(\)' THEN 1
             WHEN p.polqual ~ 'auth\.uid\(\)' OR p.polwithcheck ~ 'auth\.uid\(\)' THEN 2  
             ELSE 3 END,
        c.relname;
END;
$$ LANGUAGE plpgsql;

-- Display usage instructions
DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '=== RLS PERFORMANCE VALIDATOR LOADED ===';
    RAISE NOTICE '';
    RAISE NOTICE 'USAGE INSTRUCTIONS:';
    RAISE NOTICE '';
    RAISE NOTICE '1. BEFORE Optimization - Run baseline test:';
    RAISE NOTICE '   SELECT * FROM run_rls_performance_tests(''before'');';
    RAISE NOTICE '';  
    RAISE NOTICE '2. Run the optimization script:';
    RAISE NOTICE '   (Execute: supabase_rls_performance_fix.sql)';
    RAISE NOTICE '';
    RAISE NOTICE '3. AFTER Optimization - Run comparison test:';
    RAISE NOTICE '   SELECT * FROM run_rls_performance_tests(''after'');';
    RAISE NOTICE '';
    RAISE NOTICE '4. Compare results to see improvements:';
    RAISE NOTICE '   SELECT * FROM compare_rls_performance_results();';
    RAISE NOTICE '';
    RAISE NOTICE '5. Check for any remaining inefficient patterns:';
    RAISE NOTICE '   SELECT * FROM check_rls_efficiency();';
    RAISE NOTICE '';
    RAISE NOTICE 'ADDITIONAL MONITORING:';
    RAISE NOTICE '• View test history: SELECT * FROM rls_performance_validation ORDER BY test_timestamp DESC;';
    RAISE NOTICE '• Policy count check: SELECT COUNT(*) FROM pg_policy p JOIN pg_class c ON p.polrelid = c.oid WHERE c.relname LIKE ''archon_%'';';
    RAISE NOTICE '';
    RAISE NOTICE '=== READY FOR PERFORMANCE TESTING ===';
END
$$;

-- =====================================================
-- VALIDATION SCRIPT LOADED AND READY
-- =====================================================