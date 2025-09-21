-- =====================================================
-- FIX FOR EXISTING ARCHON_PROJECTS TABLE
-- =====================================================
-- This script adds the missing 'name' column to the existing archon_projects table
-- Run this BEFORE running the main schema if archon_projects already exists

-- Check if the name column exists, and add it if it doesn't
DO $$
BEGIN
    -- Check if archon_projects table exists
    IF EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'archon_projects'
    ) THEN
        -- Check if name column exists
        IF NOT EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = 'archon_projects' 
            AND column_name = 'name'
        ) THEN
            -- Add the name column
            ALTER TABLE public.archon_projects 
            ADD COLUMN name VARCHAR(255);
            
            -- Update existing rows with a default name based on ID
            UPDATE public.archon_projects 
            SET name = 'Project ' || LEFT(id::text, 8) 
            WHERE name IS NULL;
            
            RAISE NOTICE '✅ Added name column to archon_projects table';
        ELSE
            RAISE NOTICE '✅ archon_projects table already has name column';
        END IF;
        
        -- Check if description column exists
        IF NOT EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = 'archon_projects' 
            AND column_name = 'description'
        ) THEN
            -- Add the description column
            ALTER TABLE public.archon_projects 
            ADD COLUMN description TEXT;
            
            RAISE NOTICE '✅ Added description column to archon_projects table';
        ELSE
            RAISE NOTICE '✅ archon_projects table already has description column';
        END IF;
        
        -- Check if updated_at column exists
        IF NOT EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = 'archon_projects' 
            AND column_name = 'updated_at'
        ) THEN
            -- Add the updated_at column
            ALTER TABLE public.archon_projects 
            ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW();
            
            -- Set initial values
            UPDATE public.archon_projects 
            SET updated_at = COALESCE(created_at, NOW()) 
            WHERE updated_at IS NULL;
            
            RAISE NOTICE '✅ Added updated_at column to archon_projects table';
        ELSE
            RAISE NOTICE '✅ archon_projects table already has updated_at column';
        END IF;
        
    ELSE
        RAISE NOTICE '⚠️  archon_projects table does not exist - it will be created by the main schema';
    END IF;
END
$$;

-- Show the current structure of archon_projects
SELECT 
    column_name, 
    data_type, 
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_schema = 'public' 
AND table_name = 'archon_projects'
ORDER BY ordinal_position;