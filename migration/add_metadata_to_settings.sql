-- Add metadata column to archon_settings table for storing additional configuration
-- This is used for the External Validator to mark which API key should be used

-- Add metadata column as JSONB for flexible storage
ALTER TABLE archon_settings 
ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}';

-- Add index for metadata searches
CREATE INDEX IF NOT EXISTS idx_archon_settings_metadata 
ON archon_settings USING gin (metadata);

-- Add comment explaining the column
COMMENT ON COLUMN archon_settings.metadata IS 'Flexible JSON storage for additional configuration like useAsValidator flag';