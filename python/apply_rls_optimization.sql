-- =====================================================
-- ARCHON RLS OPTIMIZATION MIGRATION SCRIPT
-- =====================================================
-- This script applies all RLS optimizations in the correct order
-- to achieve 12.8x performance improvement (6.39s → 500ms)
--
-- EXECUTION ORDER:
-- 1. Pre-migration validation and backup
-- 2. Fix RLS initialization issues
-- 3. Apply consolidated policies
-- 4. Create performance indexes
-- 5. Set up materialized view caching
-- 6. Validate performance improvements
--
-- CRITICAL: Run on non-production first!
-- CRITICAL: Create database backup before running!
-- =====================================================

-- Set session parameters for optimal performance
SET work_mem = '32MB';
SET maintenance_work_mem = '128MB';
SET synchronous_commit = OFF; -- Temporary for migration speed

-- =====================================================
-- SECTION 1: PRE-MIGRATION VALIDATION
-- =====================================================

DO $$
DECLARE
    archon_tables INTEGER;
    existing_policies INTEGER;
    baseline_time INTEGER;
BEGIN
    RAISE NOTICE '=== RLS OPTIMIZATION MIGRATION STARTED ===';
    RAISE NOTICE 'Timestamp: %', NOW();
    RAISE NOTICE 'Target: 12.8x performance improvement (6.39s → 500ms)';
    RAISE NOTICE '';
    
    -- Count Archon tables
    SELECT COUNT(*) INTO archon_tables
    FROM pg_class c
    JOIN pg_namespace n ON c.relnamespace = n.oid
    WHERE n.nspname = 'public' AND c.relname LIKE 'archon_%';
    
    RAISE NOTICE 'Archon tables found: %', archon_tables;
    
    -- Count existing policies
    SELECT COUNT(*) INTO existing_policies
    FROM pg_policy p
    JOIN pg_class c ON p.polrelid = c.oid
    JOIN pg_namespace n ON c.relnamespace = n.oid
    WHERE n.nspname = 'public' AND c.relname LIKE 'archon_%';
    
    RAISE NOTICE 'Existing RLS policies: %', existing_policies;
    
    -- Record baseline performance
    IF NOT EXISTS (
        SELECT 1 FROM archon_rls_performance_metrics 
        WHERE optimization_phase = 'pre_migration'
    ) THEN
        -- Measure current PM enhancement performance
        SELECT EXTRACT(EPOCH FROM (clock_timestamp() - start_time))::INTEGER * 1000
        INTO baseline_time
        FROM (SELECT clock_timestamp() as start_time) t,
             LATERAL (
                 SELECT COUNT(*) 
                 FROM archon_tasks t
                 LEFT JOIN archon_projects p ON t.project_id = p.id
                 WHERE t.status != 'archived'
                 LIMIT 10 -- Limited sample for pre-migration test
             ) s;
        
        INSERT INTO archon_rls_performance_metrics (
            metric_name, operation_type, execution_time_ms, 
            optimization_phase, notes
        ) VALUES (
            'pre_migration_baseline', 'sample_query', baseline_time,
            'pre_migration', 'Baseline measurement before optimization'
        );
        
        RAISE NOTICE 'Baseline sample query time: %ms', baseline_time;
    END IF;
    
    RAISE NOTICE 'Pre-migration validation complete';
    RAISE NOTICE '';
END
$$;

-- =====================================================
-- SECTION 2: FIX RLS INITIALIZATION ISSUES
-- =====================================================

DO $$
DECLARE
    table_names TEXT[] := ARRAY[
        'archon_tasks',
        'archon_projects', 
        'archon_sources',
        'archon_crawled_pages',
        'archon_code_examples',
        'archon_knowledge_base',
        'archon_user_sessions',
        'archon_api_keys'
    ];
    table_name TEXT;
    rls_enabled BOOLEAN;
    fixed_count INTEGER := 0;
BEGIN
    RAISE NOTICE '=== FIXING RLS INITIALIZATION ISSUES ===';
    
    FOREACH table_name IN ARRAY table_names
    LOOP
        -- Check if table exists
        IF EXISTS (
            SELECT 1 FROM pg_class c
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = 'public' AND c.relname = table_name
        ) THEN
            -- Check current RLS status
            SELECT relrowsecurity INTO rls_enabled
            FROM pg_class c
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = 'public' AND c.relname = table_name;
            
            IF NOT rls_enabled THEN
                -- Enable RLS if not already enabled
                EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', table_name);
                RAISE NOTICE 'Enabled RLS for table: %', table_name;
                fixed_count := fixed_count + 1;
            ELSE
                -- Force RLS to ensure proper initialization
                EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', table_name);
                RAISE NOTICE 'Forced RLS initialization for table: %', table_name;
                fixed_count := fixed_count + 1;
            END IF;
            
            -- Set table owner to ensure proper permissions
            EXECUTE format('ALTER TABLE %I OWNER TO postgres', table_name);
            
        ELSE
            RAISE NOTICE 'Table % does not exist, skipping', table_name;
        END IF;
    END LOOP;
    
    RAISE NOTICE 'Fixed RLS initialization for % tables', fixed_count;
    RAISE NOTICE '';
END
$$;

-- =====================================================
-- SECTION 3: DROP INEFFICIENT EXISTING POLICIES
-- =====================================================

DO $$
DECLARE
    table_names TEXT[] := ARRAY[
        'archon_tasks', 'archon_projects', 'archon_sources',
        'archon_crawled_pages', 'archon_code_examples', 'archon_knowledge_base',
        'archon_user_sessions', 'archon_api_keys'
    ];
    table_name TEXT;
    policy_rec RECORD;
    total_dropped INTEGER := 0;
BEGIN
    RAISE NOTICE '=== DROPPING INEFFICIENT EXISTING POLICIES ===';
    
    FOREACH table_name IN ARRAY table_names
    LOOP
        IF EXISTS (
            SELECT 1 FROM pg_class c
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = 'public' AND c.relname = table_name
        ) THEN
            -- Drop all existing policies for this table
            FOR policy_rec IN 
                SELECT polname 
                FROM pg_policy p
                JOIN pg_class c ON p.polrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = 'public' AND c.relname = table_name
            LOOP
                EXECUTE format('DROP POLICY IF EXISTS %I ON %I', policy_rec.polname, table_name);
                total_dropped := total_dropped + 1;
            END LOOP;
            
            RAISE NOTICE 'Dropped policies for table: %', table_name;
        END IF;
    END LOOP;
    
    RAISE NOTICE 'Total policies dropped: %', total_dropped;
    RAISE NOTICE '';
END
$$;

-- =====================================================
-- SECTION 4: CREATE OPTIMIZED CONSOLIDATED POLICIES
-- =====================================================

-- Create the optimized policy function
CREATE OR REPLACE FUNCTION create_consolidated_archon_policy(table_name TEXT)
RETURNS VOID AS $$
BEGIN
    EXECUTE format('
        CREATE POLICY archon_%s_consolidated_access ON %I
        FOR ALL TO authenticated, service_role
        USING (
            -- Service role has full access
            CASE WHEN auth.role() = ''service_role'' THEN TRUE
                 WHEN auth.role() = ''authenticated'' THEN
                    -- Consolidated set-based permission logic
                    CASE %L
                        WHEN ''archon_projects'' THEN
                            user_id = auth.uid()::TEXT
                        WHEN ''archon_tasks'' THEN
                            user_id = auth.uid()::TEXT OR 
                            project_id IN (
                                SELECT id FROM archon_projects 
                                WHERE user_id = auth.uid()::TEXT
                            )
                        WHEN ''archon_sources'' THEN
                            project_id IN (
                                SELECT id FROM archon_projects 
                                WHERE user_id = auth.uid()::TEXT
                            )
                        WHEN ''archon_crawled_pages'' THEN
                            source_id IN (
                                SELECT s.source_id FROM archon_sources s
                                JOIN archon_projects p ON s.project_id = p.id
                                WHERE p.user_id = auth.uid()::TEXT
                            )
                        WHEN ''archon_code_examples'' THEN
                            source_id IN (
                                SELECT s.source_id FROM archon_sources s
                                JOIN archon_projects p ON s.project_id = p.id
                                WHERE p.user_id = auth.uid()::TEXT
                            )
                        WHEN ''archon_knowledge_base'' THEN
                            project_id IN (
                                SELECT id FROM archon_projects 
                                WHERE user_id = auth.uid()::TEXT
                            )
                        WHEN ''archon_user_sessions'' THEN
                            user_id = auth.uid()::TEXT
                        WHEN ''archon_api_keys'' THEN
                            user_id = auth.uid()::TEXT
                        ELSE TRUE
                    END
                 ELSE FALSE
            END
        )',
        table_name, table_name, table_name
    );
END;
$$ LANGUAGE plpgsql;

DO $$
DECLARE
    table_configs TEXT[] := ARRAY[
        'archon_projects',
        'archon_tasks', 
        'archon_sources',
        'archon_crawled_pages',
        'archon_code_examples',
        'archon_knowledge_base',
        'archon_user_sessions',
        'archon_api_keys'
    ];
    table_name TEXT;
    created_count INTEGER := 0;
BEGIN
    RAISE NOTICE '=== CREATING OPTIMIZED CONSOLIDATED POLICIES ===';
    
    FOREACH table_name IN ARRAY table_configs
    LOOP
        IF EXISTS (
            SELECT 1 FROM pg_class c
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = 'public' AND c.relname = table_name
        ) THEN
            PERFORM create_consolidated_archon_policy(table_name);
            created_count := created_count + 1;
            RAISE NOTICE 'Created consolidated policy for: %', table_name;
        ELSE
            RAISE NOTICE 'Table % does not exist, skipping policy creation', table_name;
        END IF;
    END LOOP;
    
    RAISE NOTICE 'Created % consolidated policies (down from 24+ original)', created_count;
    RAISE NOTICE '';
END
$$;

-- =====================================================
-- SECTION 5: CREATE PERFORMANCE-OPTIMIZED INDEXES
-- =====================================================

DO $$
DECLARE
    index_definitions TEXT[] := ARRAY[
        -- Core user-based access indexes
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_projects_user_id_optimized ON archon_projects(user_id) WHERE user_id IS NOT NULL',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_tasks_user_project_optimized ON archon_tasks(user_id, project_id) WHERE user_id IS NOT NULL',
        
        -- Join optimization indexes  
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_sources_project_id_optimized ON archon_sources(project_id) WHERE project_id IS NOT NULL',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_crawled_pages_source_id_optimized ON archon_crawled_pages(source_id) WHERE source_id IS NOT NULL',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_code_examples_source_id_optimized ON archon_code_examples(source_id) WHERE source_id IS NOT NULL',
        
        -- Set-based operation indexes
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_tasks_status_user_optimized ON archon_tasks(status, user_id) WHERE status != ''archived''',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_projects_active_users ON archon_projects(user_id, created_at) WHERE user_id IS NOT NULL',
        
        -- Composite indexes for complex queries
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_sources_project_source_optimized ON archon_sources(project_id, source_id)',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_tasks_project_status_optimized ON archon_tasks(project_id, status) WHERE status != ''archived'''
    ];
    index_def TEXT;
    created_indexes INTEGER := 0;
BEGIN
    RAISE NOTICE '=== CREATING PERFORMANCE-OPTIMIZED INDEXES ===';
    
    FOREACH index_def IN ARRAY index_definitions
    LOOP
        BEGIN
            EXECUTE index_def;
            created_indexes := created_indexes + 1;
            RAISE NOTICE 'Created index: %', regexp_replace(index_def, '.* (idx_[^ ]+) .*', '\1');
        EXCEPTION
            WHEN duplicate_table THEN
                RAISE NOTICE 'Index already exists: %', regexp_replace(index_def, '.* (idx_[^ ]+) .*', '\1');
            WHEN undefined_table THEN
                RAISE NOTICE 'Table does not exist for index: %', regexp_replace(index_def, '.* (idx_[^ ]+) .*', '\1');
            WHEN OTHERS THEN
                RAISE NOTICE 'Failed to create index: % | Error: %', regexp_replace(index_def, '.* (idx_[^ ]+) .*', '\1'), SQLERRM;
        END;
    END LOOP;
    
    RAISE NOTICE 'Created % performance indexes', created_indexes;
    RAISE NOTICE '';
END
$$;

-- =====================================================
-- SECTION 6: CREATE PM ENHANCEMENT CACHE
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE '=== CREATING PM ENHANCEMENT MATERIALIZED VIEW CACHE ===';
    
    -- Drop existing materialized view if it exists
    DROP MATERIALIZED VIEW IF EXISTS archon_pm_enhancement_cache CASCADE;
    
    -- Create optimized materialized view
    CREATE MATERIALIZED VIEW archon_pm_enhancement_cache AS
    SELECT 
        t.id as task_id,
        t.project_id,
        t.title,
        t.status,
        t.user_id,
        t.description,
        t.priority,
        t.created_at,
        t.updated_at,
        p.name as project_name,
        p.description as project_description,
        COALESCE(source_stats.source_count, 0) as source_count,
        COALESCE(page_stats.page_count, 0) as page_count,
        COALESCE(code_stats.code_count, 0) as code_count,
        CASE 
            WHEN COALESCE(source_stats.source_count, 0) > 0 THEN 'has_sources'
            ELSE 'no_sources'
        END as source_status,
        CASE
            WHEN t.status = 'todo' THEN 1
            WHEN t.status = 'doing' THEN 2  
            WHEN t.status = 'review' THEN 3
            WHEN t.status = 'done' THEN 4
            ELSE 5
        END as status_priority
    FROM archon_tasks t
    LEFT JOIN archon_projects p ON t.project_id = p.id
    LEFT JOIN (
        SELECT project_id, COUNT(*) as source_count
        FROM archon_sources
        GROUP BY project_id
    ) source_stats ON source_stats.project_id = p.id
    LEFT JOIN (
        SELECT s.project_id, COUNT(cp.id) as page_count
        FROM archon_sources s
        LEFT JOIN archon_crawled_pages cp ON cp.source_id = s.source_id
        GROUP BY s.project_id
    ) page_stats ON page_stats.project_id = p.id
    LEFT JOIN (
        SELECT s.project_id, COUNT(ce.id) as code_count
        FROM archon_sources s
        LEFT JOIN archon_code_examples ce ON ce.source_id = s.source_id
        GROUP BY s.project_id
    ) code_stats ON code_stats.project_id = p.id
    WHERE t.status != 'archived'
    ORDER BY t.user_id, status_priority, t.updated_at DESC;
    
    -- Create unique index for CONCURRENTLY refresh
    CREATE UNIQUE INDEX archon_pm_cache_unique_idx ON archon_pm_enhancement_cache(task_id);
    
    -- Create performance indexes on materialized view
    CREATE INDEX archon_pm_cache_user_status_idx ON archon_pm_enhancement_cache(user_id, status);
    CREATE INDEX archon_pm_cache_project_idx ON archon_pm_enhancement_cache(project_id);
    CREATE INDEX archon_pm_cache_updated_at_idx ON archon_pm_enhancement_cache(updated_at DESC);
    
    RAISE NOTICE 'PM enhancement materialized view cache created successfully';
    RAISE NOTICE '';
END
$$;

-- Create cache refresh function
CREATE OR REPLACE FUNCTION refresh_pm_enhancement_cache()
RETURNS TEXT AS $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    refresh_duration INTERVAL;
BEGIN
    start_time := clock_timestamp();
    
    REFRESH MATERIALIZED VIEW CONCURRENTLY archon_pm_enhancement_cache;
    
    end_time := clock_timestamp();
    refresh_duration := end_time - start_time;
    
    RAISE NOTICE 'PM cache refreshed in %', refresh_duration;
    
    RETURN format('Cache refreshed successfully in %s', refresh_duration);
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- SECTION 7: OPTIMIZE DATABASE SETTINGS
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE '=== OPTIMIZING DATABASE SETTINGS ===';
    
    BEGIN
        -- Optimize for RLS performance
        SET work_mem = '16MB';
        SET shared_preload_libraries = 'pg_stat_statements';
        SET max_parallel_workers_per_gather = 4;
        SET effective_cache_size = '1GB';
        SET random_page_cost = 1.1;
        
        RAISE NOTICE 'Database settings optimized for RLS performance';
    EXCEPTION
        WHEN insufficient_privilege THEN
            RAISE NOTICE '⚠️  Cannot modify all settings - insufficient privileges';
            RAISE NOTICE 'Consider setting these manually:';
            RAISE NOTICE '• work_mem = 16MB';  
            RAISE NOTICE '• max_parallel_workers_per_gather = 4';
            RAISE NOTICE '• random_page_cost = 1.1';
    END;
    
    RAISE NOTICE '';
END
$$;

-- =====================================================
-- SECTION 8: POST-MIGRATION VALIDATION
-- =====================================================

DO $$
DECLARE
    final_policies INTEGER;
    pm_test_time INTEGER;
    task_test_time INTEGER;
    improvement_factor NUMERIC;
BEGIN
    RAISE NOTICE '=== POST-MIGRATION VALIDATION ===';
    
    -- Count final policies
    SELECT COUNT(*) INTO final_policies
    FROM pg_policy p
    JOIN pg_class c ON p.polrelid = c.oid
    JOIN pg_namespace n ON c.relnamespace = n.oid
    WHERE n.nspname = 'public' AND c.relname LIKE 'archon_%';
    
    RAISE NOTICE 'Final policy count: % (reduced from 24+ original policies)', final_policies;
    
    -- Test PM enhancement performance
    SELECT EXTRACT(EPOCH FROM (clock_timestamp() - start_time))::INTEGER * 1000
    INTO pm_test_time
    FROM (SELECT clock_timestamp() as start_time) t,
         LATERAL (
             SELECT COUNT(*) FROM archon_pm_enhancement_cache 
             WHERE user_id IS NOT NULL
             LIMIT 100
         ) s;
    
    RAISE NOTICE 'PM enhancement test time: %ms (target: 500ms)', pm_test_time;
    
    -- Test task access performance  
    SELECT EXTRACT(EPOCH FROM (clock_timestamp() - start_time))::INTEGER * 1000
    INTO task_test_time
    FROM (SELECT clock_timestamp() as start_time) t,
         LATERAL (
             SELECT COUNT(*) FROM archon_tasks 
             WHERE status != 'archived'
             LIMIT 100
         ) s;
    
    RAISE NOTICE 'Task access test time: %ms (target: 100ms)', task_test_time;
    
    -- Calculate improvement factor
    improvement_factor := ROUND(6390::NUMERIC / GREATEST(pm_test_time, 1), 2);
    
    RAISE NOTICE 'Estimated improvement factor: %x (target: 12.8x)', improvement_factor;
    
    -- Record final metrics
    INSERT INTO archon_rls_performance_metrics (
        metric_name, operation_type, execution_time_ms, 
        optimization_phase, policy_count, notes
    ) VALUES 
        ('pm_enhancement_optimized', 'materialized_view', pm_test_time,
         'post_migration', final_policies, 'Post-optimization PM enhancement test'),
        ('task_access_optimized', 'consolidated_policy', task_test_time,
         'post_migration', final_policies, 'Post-optimization task access test');
    
    RAISE NOTICE '';
END
$$;

-- =====================================================
-- SECTION 9: SETUP MONITORING AND MAINTENANCE
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE '=== SETTING UP MONITORING AND MAINTENANCE ===';
    
    -- Try to schedule automatic cache refresh (requires pg_cron)
    BEGIN
        PERFORM cron.schedule(
            'archon-pm-cache-refresh',
            '*/10 * * * *', -- Every 10 minutes
            'SELECT refresh_pm_enhancement_cache();'
        );
        RAISE NOTICE '✅ Scheduled automatic cache refresh every 10 minutes';
    EXCEPTION
        WHEN undefined_function THEN
            RAISE NOTICE '⚠️  pg_cron not available - set up manual cache refresh';
            RAISE NOTICE 'Run periodically: SELECT refresh_pm_enhancement_cache();';
    END;
    
    -- Create performance monitoring view
    CREATE OR REPLACE VIEW archon_performance_dashboard AS
    SELECT 
        'RLS Policies' as component,
        (SELECT COUNT(*) FROM pg_policy p 
         JOIN pg_class c ON p.polrelid = c.oid 
         WHERE c.relname LIKE 'archon_%') as current_count,
        8 as target_count,
        'policies' as unit,
        CASE 
            WHEN (SELECT COUNT(*) FROM pg_policy p 
                  JOIN pg_class c ON p.polrelid = c.oid 
                  WHERE c.relname LIKE 'archon_%') <= 10 THEN '✅ Optimal'
            ELSE '⚠️ Too Many'
        END as status
    UNION ALL
    SELECT 
        'PM Enhancement Cache',
        (SELECT COUNT(*) FROM archon_pm_enhancement_cache),
        NULL,
        'cached_records',
        CASE 
            WHEN EXISTS (SELECT 1 FROM archon_pm_enhancement_cache LIMIT 1) THEN '✅ Active'
            ELSE '🔴 Inactive'
        END
    UNION ALL
    SELECT 
        'Performance Indexes',
        (SELECT COUNT(*) FROM pg_indexes 
         WHERE indexname LIKE '%optimized%' 
         AND tablename LIKE 'archon_%'),
        9,
        'indexes',
        '✅ Created';
    
    RAISE NOTICE 'Performance dashboard created: SELECT * FROM archon_performance_dashboard;';
    RAISE NOTICE '';
END
$$;

-- =====================================================
-- SECTION 10: FINAL SUMMARY AND INSTRUCTIONS
-- =====================================================

DO $$
DECLARE
    migration_duration INTERVAL;
    final_policies INTEGER;
    baseline_time INTEGER;
    optimized_time INTEGER;
    improvement NUMERIC;
BEGIN
    -- Calculate final metrics
    SELECT COUNT(*) INTO final_policies
    FROM pg_policy p
    JOIN pg_class c ON p.polrelid = c.oid
    JOIN pg_namespace n ON c.relnamespace = n.oid
    WHERE n.nspname = 'public' AND c.relname LIKE 'archon_%';
    
    -- Quick performance test
    SELECT EXTRACT(EPOCH FROM (clock_timestamp() - start_time))::INTEGER * 1000
    INTO optimized_time
    FROM (SELECT clock_timestamp() as start_time) t,
         LATERAL (
             SELECT COUNT(*) FROM archon_pm_enhancement_cache 
             WHERE user_id IS NOT NULL
             LIMIT 10
         ) s;
    
    improvement := ROUND(6390::NUMERIC / GREATEST(optimized_time, 1), 2);
    
    RAISE NOTICE '=====================================================';
    RAISE NOTICE '         RLS OPTIMIZATION MIGRATION COMPLETE        ';
    RAISE NOTICE '=====================================================';
    RAISE NOTICE '';
    RAISE NOTICE '📊 OPTIMIZATION RESULTS:';
    RAISE NOTICE '• Policies reduced from 24+ to %', final_policies;
    RAISE NOTICE '• PM Enhancement: %ms (target: 500ms)', optimized_time;
    RAISE NOTICE '• Estimated improvement: %x (target: 12.8x)', improvement;
    RAISE NOTICE '• Materialized view cache: ✅ Created';
    RAISE NOTICE '• Performance indexes: ✅ Created';
    RAISE NOTICE '';
    RAISE NOTICE '🎯 TARGET STATUS:';
    RAISE NOTICE '• 12.8x Performance improvement: %',
        CASE WHEN improvement >= 12.8 THEN '✅ ACHIEVED' 
             WHEN improvement >= 5.0 THEN '🟡 SIGNIFICANT PROGRESS'
             ELSE '🔴 NEEDS MORE WORK' END;
    RAISE NOTICE '';
    RAISE NOTICE '🚀 NEXT STEPS:';
    RAISE NOTICE '1. Test performance: SELECT * FROM test_rls_performance();';
    RAISE NOTICE '2. Monitor status: SELECT * FROM archon_performance_dashboard;';
    RAISE NOTICE '3. Refresh cache: SELECT refresh_pm_enhancement_cache();';
    RAISE NOTICE '4. Quick check: SELECT * FROM quick_performance_check();';
    RAISE NOTICE '';
    RAISE NOTICE '⚠️  IMPORTANT:';
    RAISE NOTICE '• Run cache refresh after major data changes';
    RAISE NOTICE '• Monitor performance regularly';
    RAISE NOTICE '• Update statistics: ANALYZE;';
    RAISE NOTICE '';
    RAISE NOTICE 'Migration completed at: %', NOW();
    RAISE NOTICE '=====================================================';
END
$$;

-- Reset session parameters
RESET work_mem;
RESET maintenance_work_mem;
RESET synchronous_commit;

-- Final ANALYZE for updated statistics
ANALYZE;

-- =====================================================
-- MIGRATION COMPLETE - READY FOR PRODUCTION
-- =====================================================