-- =====================================================
-- Prerequisite Functions for Phase 7 DeepConf Migration
-- =====================================================
-- Date: 2025-09-01
-- Purpose: Create required functions before main migration
-- =====================================================

BEGIN;

-- Create or replace the update_updated_at_column function
-- This is referenced by our migration trigger
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