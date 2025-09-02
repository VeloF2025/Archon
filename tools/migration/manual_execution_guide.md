# Phase 7 DeepConf Database Migration - Manual Execution Guide

## Overview
This guide provides step-by-step instructions for manually executing the Phase 7 DeepConf database migration in the Supabase SQL Editor.

## Prerequisites
- Access to Supabase Dashboard
- SQL Editor permissions
- Database: PostgreSQL 14+ with Supabase extensions

## Step 1: Execute Prerequisite Functions

Copy and execute the following SQL in the Supabase SQL Editor:

```sql
-- =====================================================
-- Prerequisite Functions for Phase 7 DeepConf Migration
-- =====================================================

BEGIN;

-- Create or replace the update_updated_at_column function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Verify function was created
DO $$ 
BEGIN 
    IF EXISTS (
        SELECT 1 FROM information_schema.routines 
        WHERE routine_name = 'update_updated_at_column'
        AND routine_type = 'FUNCTION'
    ) THEN
        RAISE NOTICE '✓ update_updated_at_column function created successfully';
    ELSE
        RAISE EXCEPTION '✗ Failed to create update_updated_at_column function';
    END IF;
END $$;

COMMIT;
```

**Expected Output:** 
- Notice: "✓ update_updated_at_column function created successfully"

## Step 2: Execute Main Migration

Execute the complete Phase 7 migration script (`migration/phase7_deepconf_schema.sql`) in the Supabase SQL Editor.

**Important Notes:**
- The script is wrapped in a transaction (BEGIN/COMMIT)
- Contains built-in validation at the end
- Should take 30-60 seconds to execute

**Expected Output:**
- Multiple notices about tables, indexes, and views being created
- Final notice: "🎉 Phase 7 DeepConf Integration schema migration completed successfully!"

## Step 3: Validation Queries

After executing the migration, run these validation queries:

### 3.1 Verify Tables Were Created

```sql
SELECT 
    t.table_name,
    t.table_type,
    c.column_count
FROM information_schema.tables t
LEFT JOIN (
    SELECT table_name, COUNT(*) as column_count
    FROM information_schema.columns
    WHERE table_schema = 'public'
    AND table_name IN ('archon_confidence_scores', 'archon_performance_metrics', 'archon_confidence_calibration')
    GROUP BY table_name
) c ON t.table_name = c.table_name
WHERE t.table_schema = 'public'
AND t.table_name IN ('archon_confidence_scores', 'archon_performance_metrics', 'archon_confidence_calibration')
ORDER BY t.table_name;
```

**Expected Result:** 3 tables with the following approximate column counts:
- `archon_confidence_scores`: ~15 columns
- `archon_performance_metrics`: ~15 columns  
- `archon_confidence_calibration`: ~10 columns

### 3.2 Verify Indexes Were Created

```sql
SELECT 
    tablename,
    indexname,
    indexdef
FROM pg_indexes 
WHERE tablename LIKE 'archon_confidence%' OR tablename LIKE 'archon_performance%'
ORDER BY tablename, indexname;
```

**Expected Result:** 15+ indexes across the three tables

### 3.3 Verify Views Were Created

```sql
SELECT 
    table_name,
    view_definition
FROM information_schema.views 
WHERE table_name LIKE 'archon_%confidence%' OR table_name LIKE 'archon_%performance%'
ORDER BY table_name;
```

**Expected Result:** 3 views:
- `archon_confidence_trends`
- `archon_performance_dashboard` 
- `archon_calibration_analysis`

### 3.4 Verify Functions Were Created

```sql
SELECT 
    routine_name,
    routine_type,
    data_type
FROM information_schema.routines 
WHERE (routine_name LIKE '%confidence%' OR routine_name LIKE '%performance%')
AND routine_schema = 'public'
AND routine_type = 'FUNCTION'
ORDER BY routine_name;
```

**Expected Result:** At least 2 functions:
- `calculate_confidence_stats`
- `insert_performance_metric`

### 3.5 Verify RLS Policies

```sql
SELECT 
    schemaname,
    tablename,
    policyname,
    permissive,
    roles,
    cmd,
    qual
FROM pg_policies 
WHERE tablename LIKE 'archon_confidence%' OR tablename LIKE 'archon_performance%'
ORDER BY tablename, policyname;
```

**Expected Result:** 3+ RLS policies across the tables

## Step 4: Test Basic CRUD Operations

### 4.1 Test Confidence Scores Insert

```sql
-- Insert test data
WITH inserted AS (
    INSERT INTO archon_confidence_scores (
        request_id, 
        factual_confidence, 
        reasoning_confidence, 
        contextual_relevance,
        uncertainty_lower, 
        uncertainty_upper, 
        model_consensus, 
        request_type
    ) VALUES (
        gen_random_uuid(), 
        0.8500, 
        0.7200, 
        0.9100,
        0.0500, 
        0.1200, 
        '{"model_1": 0.85, "model_2": 0.82}', 
        'test_migration'
    ) RETURNING id, overall_confidence
)
SELECT 
    id,
    overall_confidence,
    ROUND((0.8500 + 0.7200 + 0.9100) / 3.0, 4) as expected_confidence,
    CASE 
        WHEN ABS(overall_confidence - (0.8500 + 0.7200 + 0.9100) / 3.0) < 0.0001 
        THEN '✓ Computed column working correctly'
        ELSE '✗ Computed column error'
    END as validation_result
FROM inserted;
```

**Expected Result:** Computed column should match expected confidence (≈0.8267)

### 4.2 Test Performance Metrics Function

```sql
-- Test the insert_performance_metric function
SELECT insert_performance_metric(
    0.8500::DECIMAL(5,4),  -- token_efficiency
    250,                   -- response_time_ms
    0.7800::DECIMAL(5,4),  -- confidence_accuracy
    0.0200::DECIMAL(5,4),  -- hallucination_rate
    0.6500::DECIMAL(5,4),  -- system_load
    'test_migration',      -- request_type
    'test-model-v1',       -- model_version
    '/api/test'            -- endpoint
) as inserted_metric_id;
```

**Expected Result:** A UUID value representing the inserted metric ID

### 4.3 Test Statistics Function

```sql
-- Test the calculate_confidence_stats function
SELECT * FROM calculate_confidence_stats(
    NOW() - INTERVAL '1 hour',
    NOW(),
    'test_migration'
);
```

**Expected Result:** Statistics for the test data inserted above

### 4.4 Clean Up Test Data

```sql
-- Clean up test data
DELETE FROM archon_confidence_scores WHERE request_type = 'test_migration';
DELETE FROM archon_performance_metrics WHERE request_type = 'test_migration';

SELECT 
    'Test data cleaned up successfully' as status,
    (SELECT COUNT(*) FROM archon_confidence_scores WHERE request_type = 'test_migration') as remaining_confidence_records,
    (SELECT COUNT(*) FROM archon_performance_metrics WHERE request_type = 'test_migration') as remaining_metric_records;
```

**Expected Result:** 0 remaining test records

## Step 5: Performance Test

Run this query to test query performance on the new indexes:

```sql
-- Performance test with EXPLAIN ANALYZE
EXPLAIN ANALYZE
SELECT 
    DATE_TRUNC('hour', created_at) as hour,
    request_type,
    COUNT(*) as request_count,
    AVG(overall_confidence) as avg_confidence
FROM archon_confidence_scores
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', created_at), request_type
ORDER BY hour DESC, request_type
LIMIT 100;
```

**Expected Result:** Query should execute quickly (< 100ms) even with indexes on empty tables

## Troubleshooting

### Common Issues

1. **"Function update_updated_at_column does not exist"**
   - Solution: Execute Step 1 (Prerequisite Functions) first

2. **"Relation already exists" errors**
   - This is normal if re-running the migration
   - The script uses `IF NOT EXISTS` clauses

3. **RLS Policy errors**
   - These might fail if auth schema is not properly configured
   - This is acceptable for development environments

4. **Validation notices not appearing**
   - Check that you're running the complete script, not individual statements

### Rollback (If Needed)

If you need to rollback the migration:

```sql
BEGIN;

-- Drop views
DROP VIEW IF EXISTS archon_confidence_trends CASCADE;
DROP VIEW IF EXISTS archon_performance_dashboard CASCADE;
DROP VIEW IF EXISTS archon_calibration_analysis CASCADE;

-- Drop functions
DROP FUNCTION IF EXISTS calculate_confidence_stats(TIMESTAMP WITH TIME ZONE, TIMESTAMP WITH TIME ZONE, VARCHAR);
DROP FUNCTION IF EXISTS insert_performance_metric(DECIMAL, INTEGER, DECIMAL, DECIMAL, DECIMAL, VARCHAR, VARCHAR, VARCHAR, JSONB);

-- Drop tables (in reverse dependency order)
DROP TABLE IF EXISTS archon_confidence_calibration CASCADE;
DROP TABLE IF EXISTS archon_performance_metrics CASCADE;
DROP TABLE IF EXISTS archon_confidence_scores CASCADE;

COMMIT;

SELECT 'Rollback completed - DeepConf tables removed' as status;
```

## Next Steps After Successful Migration

1. **Update Application Code**: Modify `storage.py` and related files to use new tables
2. **Configure Monitoring**: Set up dashboards using the new views
3. **Test Integration**: Verify DeepConf integration works with new schema
4. **Performance Monitoring**: Monitor query performance and optimize as needed

## Success Criteria

The migration is successful when:
- ✅ All 3 tables created with proper constraints
- ✅ All 15+ indexes created successfully
- ✅ All 3 views created and queryable
- ✅ Both custom functions work correctly
- ✅ RLS policies are active (may vary by environment)
- ✅ Basic CRUD operations work without errors
- ✅ Computed columns calculate correctly