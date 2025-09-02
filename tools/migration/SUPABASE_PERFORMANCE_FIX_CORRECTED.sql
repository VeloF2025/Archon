-- 🚨 SUPABASE PERFORMANCE FIX - CORRECTED COLUMN NAMES
-- This will fix the 2.5-second API response time immediately

-- =========================================
-- STEP 1: CREATE MISSING TABLE (CRITICAL)
-- =========================================

-- This fixes the DeepConf "table not found" errors
CREATE TABLE IF NOT EXISTS public.confidence_history (
    id BIGSERIAL PRIMARY KEY,
    task_id VARCHAR(255) NOT NULL,
    confidence_score JSONB NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    model_source VARCHAR(100),
    gaming_score FLOAT DEFAULT 0.0,
    overall_confidence FLOAT,
    factual_confidence FLOAT,
    reasoning_confidence FLOAT,
    contextual_confidence FLOAT
);

-- Enable RLS for security
ALTER TABLE public.confidence_history ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all on confidence_history" ON public.confidence_history FOR ALL USING (true);

-- =========================================
-- STEP 2: CRITICAL PERFORMANCE INDEXES
-- (Using correct column names from actual schema)
-- =========================================

-- PROJECTS TABLE INDEXES (fixes 2.5-second delay)
CREATE INDEX IF NOT EXISTS idx_archon_projects_created_at ON public.archon_projects(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_archon_projects_title ON public.archon_projects(title);
CREATE INDEX IF NOT EXISTS idx_archon_projects_pinned ON public.archon_projects(pinned);

-- TASKS TABLE INDEXES (critical for project relationships)
CREATE INDEX IF NOT EXISTS idx_archon_tasks_project_id ON public.archon_tasks(project_id);
CREATE INDEX IF NOT EXISTS idx_archon_tasks_status ON public.archon_tasks(status);
CREATE INDEX IF NOT EXISTS idx_archon_tasks_created_at ON public.archon_tasks(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_archon_tasks_project_status ON public.archon_tasks(project_id, status);

-- SOURCES TABLE INDEXES (using correct column names)
CREATE INDEX IF NOT EXISTS idx_archon_sources_source_id ON public.archon_sources(source_id);
CREATE INDEX IF NOT EXISTS idx_archon_sources_source_url ON public.archon_sources(source_url);
CREATE INDEX IF NOT EXISTS idx_archon_sources_created_at ON public.archon_sources(created_at DESC);

-- CONFIDENCE HISTORY INDEXES
CREATE INDEX IF NOT EXISTS idx_confidence_history_task_id ON public.confidence_history(task_id);
CREATE INDEX IF NOT EXISTS idx_confidence_history_timestamp ON public.confidence_history(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_confidence_history_model_source ON public.confidence_history(model_source);

-- CRAWLED PAGES INDEXES (if the table exists)
CREATE INDEX IF NOT EXISTS idx_archon_crawled_pages_source_id ON public.archon_crawled_pages(source_id);

-- =========================================
-- STEP 3: UPDATE STATISTICS
-- =========================================

-- Update query planner statistics for better performance
ANALYZE public.archon_projects;
ANALYZE public.archon_tasks;
ANALYZE public.archon_sources;
ANALYZE public.confidence_history;