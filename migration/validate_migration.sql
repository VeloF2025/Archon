-- =====================================================
-- Phase 7 DeepConf Migration Validation Script
-- =====================================================
-- Run this script after executing the main migration to verify success
-- Expected: All checks should return ✓ status

DO $$
DECLARE
    table_count INTEGER;
    index_count INTEGER;
    view_count INTEGER;
    function_count INTEGER;
    policy_count INTEGER;
    test_id UUID;
    test_confidence DECIMAL(5,4);
    expected_confidence DECIMAL(5,4) := 0.8267;
    stats_result RECORD;
    perf_id UUID;
BEGIN
    RAISE NOTICE '🔍 Phase 7 DeepConf Migration Validation';
    RAISE NOTICE '=============================================';
    
    -- Check 1: Verify Tables
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables 
    WHERE table_name IN ('archon_confidence_scores', 'archon_performance_metrics', 'archon_confidence_calibration')
    AND table_schema = 'public';
    
    IF table_count = 3 THEN
        RAISE NOTICE '✓ All 3 DeepConf tables created successfully';
    ELSE
        RAISE NOTICE '✗ Expected 3 tables, found %', table_count;
    END IF;
    
    -- Check 2: Verify Indexes
    SELECT COUNT(*) INTO index_count
    FROM pg_indexes 
    WHERE tablename LIKE 'archon_confidence%' OR tablename LIKE 'archon_performance%';
    
    IF index_count >= 15 THEN
        RAISE NOTICE '✓ Performance indexes created successfully (% indexes)', index_count;
    ELSE
        RAISE NOTICE '⚠ Found % indexes, expected at least 15', index_count;
    END IF;
    
    -- Check 3: Verify Views
    SELECT COUNT(*) INTO view_count
    FROM information_schema.views 
    WHERE table_name LIKE 'archon_%confidence%' OR table_name LIKE 'archon_%performance%';
    
    IF view_count = 3 THEN
        RAISE NOTICE '✓ All 3 analysis views created successfully';
    ELSE
        RAISE NOTICE '⚠ Found % views, expected 3', view_count;
    END IF;
    
    -- Check 4: Verify Functions
    SELECT COUNT(*) INTO function_count
    FROM information_schema.routines 
    WHERE (routine_name LIKE '%confidence%' OR routine_name LIKE '%performance%')
    AND routine_schema = 'public'
    AND routine_type = 'FUNCTION';
    
    IF function_count >= 2 THEN
        RAISE NOTICE '✓ DeepConf functions created successfully (% functions)', function_count;
    ELSE
        RAISE NOTICE '⚠ Found % functions, expected at least 2', function_count;
    END IF;
    
    -- Check 5: Verify RLS Policies (Optional - may fail in some environments)
    BEGIN
        SELECT COUNT(*) INTO policy_count
        FROM pg_policies 
        WHERE tablename LIKE 'archon_confidence%' OR tablename LIKE 'archon_performance%';
        
        IF policy_count >= 3 THEN
            RAISE NOTICE '✓ RLS policies active (% policies)', policy_count;
        ELSE
            RAISE NOTICE '⚠ Found % RLS policies, expected at least 3', policy_count;
        END IF;
    EXCEPTION
        WHEN OTHERS THEN
            RAISE NOTICE '⚠ Could not check RLS policies (may not be available in this environment)';
    END;
    
    -- Check 6: Test Basic CRUD Operations
    BEGIN
        -- Insert test record
        INSERT INTO archon_confidence_scores (
            request_id, factual_confidence, reasoning_confidence, contextual_relevance,
            uncertainty_lower, uncertainty_upper, model_consensus, request_type
        ) VALUES (
            gen_random_uuid(), 0.8500, 0.7200, 0.9100,
            0.0500, 0.1200, '{"model_1": 0.85, "model_2": 0.82}', 'validation_test'
        ) RETURNING id INTO test_id;
        
        -- Test computed column
        SELECT overall_confidence INTO test_confidence
        FROM archon_confidence_scores 
        WHERE id = test_id;
        
        IF ABS(test_confidence - expected_confidence) < 0.0001 THEN
            RAISE NOTICE '✓ Computed column working correctly (%.4f)', test_confidence;
        ELSE
            RAISE NOTICE '✗ Computed column error: expected %.4f, got %.4f', expected_confidence, test_confidence;
        END IF;
        
        -- Test performance metric function
        SELECT insert_performance_metric(
            0.8500, 250, 0.7800, 0.0200, 0.6500, 
            'validation_test', 'test-model-v1', '/api/validation'
        ) INTO perf_id;
        
        IF perf_id IS NOT NULL THEN
            RAISE NOTICE '✓ Performance metric function working (ID: %)', LEFT(perf_id::TEXT, 8);
        ELSE
            RAISE NOTICE '✗ Performance metric function failed';
        END IF;
        
        -- Test statistics function
        SELECT * INTO stats_result FROM calculate_confidence_stats(
            NOW() - INTERVAL '1 hour',
            NOW(),
            'validation_test'
        );
        
        IF stats_result.total_requests >= 1 THEN
            RAISE NOTICE '✓ Statistics function working (% requests)', stats_result.total_requests;
        ELSE
            RAISE NOTICE '✗ Statistics function returned no results';
        END IF;
        
        -- Clean up test data
        DELETE FROM archon_confidence_scores WHERE request_type = 'validation_test';
        DELETE FROM archon_performance_metrics WHERE request_type = 'validation_test';
        
        RAISE NOTICE '✓ CRUD operations test completed successfully';
        
    EXCEPTION
        WHEN OTHERS THEN
            RAISE NOTICE '✗ CRUD operations test failed: %', SQLERRM;
            -- Attempt cleanup on error
            DELETE FROM archon_confidence_scores WHERE request_type = 'validation_test';
            DELETE FROM archon_performance_metrics WHERE request_type = 'validation_test';
    END;
    
    -- Final Summary
    RAISE NOTICE '';
    RAISE NOTICE '📊 VALIDATION SUMMARY:';
    RAISE NOTICE '  Tables: %/3', table_count;
    RAISE NOTICE '  Indexes: %', index_count;
    RAISE NOTICE '  Views: %/3', view_count;
    RAISE NOTICE '  Functions: %', function_count;
    RAISE NOTICE '  RLS Policies: %', policy_count;
    
    IF table_count = 3 AND view_count = 3 AND function_count >= 2 THEN
        RAISE NOTICE '';
        RAISE NOTICE '🎉 MIGRATION VALIDATION SUCCESSFUL!';
        RAISE NOTICE '   Phase 7 DeepConf Integration is ready for use.';
        RAISE NOTICE '';
        RAISE NOTICE '📋 NEXT STEPS:';
        RAISE NOTICE '  1. Update storage.py to use new tables';
        RAISE NOTICE '  2. Test DeepConf integration';
        RAISE NOTICE '  3. Monitor confidence scoring in dashboard';
    ELSE
        RAISE NOTICE '';
        RAISE NOTICE '⚠ MIGRATION VALIDATION INCOMPLETE';
        RAISE NOTICE '   Some components may not be working correctly.';
        RAISE NOTICE '   Please review the issues above.';
    END IF;
    
END $$;