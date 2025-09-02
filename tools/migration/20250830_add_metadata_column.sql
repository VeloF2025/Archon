-- Migration: Add metadata column to archon_settings
-- Date: 2025-08-30
-- Purpose: Support External Validator API key configuration via UI

-- Add metadata column to archon_settings table
ALTER TABLE archon_settings 
ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}';

-- Create GIN index for efficient JSONB queries
CREATE INDEX IF NOT EXISTS idx_archon_settings_metadata 
ON archon_settings USING gin (metadata);

-- Add documentation
COMMENT ON COLUMN archon_settings.metadata IS 'Flexible JSON storage for additional configuration like useAsValidator flag for External Validator';

-- Verify the column was added
DO $$ 
BEGIN 
    IF EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'archon_settings' 
        AND column_name = 'metadata'
    ) THEN 
        RAISE NOTICE 'metadata column successfully added to archon_settings table';
    ELSE 
        RAISE EXCEPTION 'Failed to add metadata column';
    END IF;
END $$;