-- 🚀 ARCHON PERFORMANCE FIXES - CRITICAL DATABASE OPTIMIZATIONS
-- Execute these SQL commands in Supabase SQL Editor to fix performance issues

-- =========================================
-- PART 1: CREATE MISSING TABLES
-- =========================================

-- Create missing confidence_history table for DeepConf system
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
    contextual_confidence FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable Row Level Security (RLS) for the new table
ALTER TABLE public.confidence_history ENABLE ROW LEVEL SECURITY;

-- Create RLS policy to allow all operations (adjust as needed for your security model)
CREATE POLICY "Allow all operations on confidence_history" ON public.confidence_history
    FOR ALL USING (true) WITH CHECK (true);

-- =========================================
-- PART 2: ADD CRITICAL PERFORMANCE INDEXES
-- =========================================

-- Indexes for archon_sources table (frequently queried)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_sources_source_id 
    ON public.archon_sources(source_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_sources_created_at 
    ON public.archon_sources(created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_sources_url 
    ON public.archon_sources(url);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_sources_source_type 
    ON public.archon_sources(source_type);

-- Indexes for projects table (causing 2.5s delays)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_projects_created_at 
    ON public.archon_projects(created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_projects_name 
    ON public.archon_projects(name);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_projects_status 
    ON public.archon_projects(status);

-- Indexes for tasks table (project relationships)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_tasks_project_id 
    ON public.archon_tasks(project_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_tasks_status 
    ON public.archon_tasks(status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_tasks_created_at 
    ON public.archon_tasks(created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_tasks_project_status 
    ON public.archon_tasks(project_id, status);

-- Indexes for document versions (if exists)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_document_versions_document_id 
    ON public.archon_document_versions(document_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_document_versions_created_at 
    ON public.archon_document_versions(created_at DESC);

-- Indexes for crawled pages (source relationships)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_crawled_pages_source_id 
    ON public.archon_crawled_pages(source_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archon_crawled_pages_url 
    ON public.archon_crawled_pages(url);

-- Indexes for confidence_history table (new)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_confidence_history_task_id 
    ON public.confidence_history(task_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_confidence_history_timestamp 
    ON public.confidence_history(timestamp DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_confidence_history_model_source 
    ON public.confidence_history(model_source);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_confidence_history_overall_confidence 
    ON public.confidence_history(overall_confidence);

-- Composite indexes for common query patterns
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_confidence_history_task_timestamp 
    ON public.confidence_history(task_id, timestamp DESC);

-- =========================================
-- PART 3: QUERY OPTIMIZATION VIEWS
-- =========================================

-- Create optimized view for project dashboard (reduces N+1 queries)
CREATE OR REPLACE VIEW public.projects_with_stats AS
SELECT 
    p.*,
    COUNT(t.id) as task_count,
    COUNT(CASE WHEN t.status = 'completed' THEN 1 END) as completed_tasks,
    COUNT(CASE WHEN t.status = 'in_progress' THEN 1 END) as active_tasks,
    MAX(t.updated_at) as last_task_update
FROM public.archon_projects p
LEFT JOIN public.archon_tasks t ON p.id = t.project_id
GROUP BY p.id, p.name, p.description, p.status, p.created_at, p.updated_at;

-- =========================================
-- PART 4: MAINTENANCE COMMANDS
-- =========================================

-- Update table statistics for query planner
ANALYZE public.archon_sources;
ANALYZE public.archon_projects; 
ANALYZE public.archon_tasks;
ANALYZE public.archon_crawled_pages;
ANALYZE public.confidence_history;

-- Show current table sizes and row counts
-- (Run this to verify the fixes worked)
/*
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats 
WHERE schemaname = 'public' 
AND tablename IN ('archon_projects', 'archon_tasks', 'archon_sources')
ORDER BY tablename, attname;
*/