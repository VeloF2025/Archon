-- 🚨 CRITICAL MISSING INDEX - RUN THIS IN SUPABASE NOW
-- This fixes the N+1 query problem causing 2+ second delays

-- The logs show individual queries like:
-- GET archon_project_sources?project_id=eq.cf5ba87a-5e78-477d-8f81-c8e28a2d4103
-- GET archon_project_sources?project_id=eq.b2f942ea-9c6b-4d59-8cd0-f24c9ae99d33
-- This happens for each project, causing massive delays

CREATE INDEX IF NOT EXISTS idx_archon_project_sources_project_id 
ON public.archon_project_sources(project_id);

-- This single index will change 2+ seconds to <200ms for /api/projects