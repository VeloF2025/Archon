-- =====================================================
-- FIX DEFAULT PROJECT INSERTION
-- =====================================================
-- This script properly inserts or updates the default project
-- handling the existing title column requirement

-- First, check what columns exist in archon_projects
DO $$
DECLARE
    has_title_column BOOLEAN;
    has_name_column BOOLEAN;
BEGIN
    -- Check for title column
    SELECT EXISTS (
        SELECT FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'archon_projects' 
        AND column_name = 'title'
    ) INTO has_title_column;
    
    -- Check for name column
    SELECT EXISTS (
        SELECT FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'archon_projects' 
        AND column_name = 'name'
    ) INTO has_name_column;
    
    -- Handle the default project based on what columns exist
    IF has_title_column THEN
        -- Use title column if it exists
        INSERT INTO public.archon_projects (id, title, description)
        VALUES (
            '00000000-0000-0000-0000-000000000001'::UUID,
            'Default Project',
            'Default project for agent management system'
        )
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title,
            description = EXCLUDED.description;
            
        RAISE NOTICE '✅ Default project created/updated using title column';
        
        -- Also update name column if it exists
        IF has_name_column THEN
            UPDATE public.archon_projects 
            SET name = 'Default Project'
            WHERE id = '00000000-0000-0000-0000-000000000001'::UUID;
            RAISE NOTICE '✅ Also updated name column';
        END IF;
        
    ELSIF has_name_column THEN
        -- Use name column if only it exists
        INSERT INTO public.archon_projects (id, name, description)
        VALUES (
            '00000000-0000-0000-0000-000000000001'::UUID,
            'Default Project',
            'Default project for agent management system'
        )
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            description = EXCLUDED.description;
            
        RAISE NOTICE '✅ Default project created/updated using name column';
    ELSE
        RAISE EXCEPTION 'archon_projects table has neither title nor name column';
    END IF;
END
$$;

-- Verify the default project exists
SELECT 
    id,
    COALESCE(name, title, 'Unknown') as project_name,
    description,
    created_at
FROM public.archon_projects
WHERE id = '00000000-0000-0000-0000-000000000001'::UUID;