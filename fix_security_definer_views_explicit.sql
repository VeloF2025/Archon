-- Explicit Fix for Security Definer Views
-- This script explicitly removes SECURITY DEFINER from views

-- Drop ALL existing views completely
DROP VIEW IF EXISTS public.archon_cost_optimization_recommendations CASCADE;
DROP VIEW IF EXISTS public.archon_project_intelligence_overview CASCADE;
DROP VIEW IF EXISTS public.archon_agent_performance_dashboard CASCADE;

-- Wait to ensure views are completely dropped
SELECT pg_sleep(1);

-- Recreate cost optimization recommendations view (explicitly WITHOUT SECURITY DEFINER)
CREATE VIEW public.archon_cost_optimization_recommendations 
WITH (security_invoker=true)  -- Explicitly set security invoker
AS
SELECT 
    a.id as agent_id,
    a.name as agent_name,
    a.agent_type,
    a.model_tier as current_tier,
    ct.total_cost,
    ct.avg_cost_per_task,
    a.success_rate,
    CASE 
        WHEN a.model_tier = 'OPUS' AND a.success_rate > 0.95 AND ct.avg_cost_per_task > 0.50 THEN 'CONSIDER_SONNET'
        WHEN a.model_tier = 'SONNET' AND a.success_rate < 0.80 THEN 'CONSIDER_OPUS'
        WHEN a.model_tier = 'SONNET' AND a.success_rate > 0.98 AND ct.avg_cost_per_task < 0.05 THEN 'CONSIDER_HAIKU'
        WHEN a.model_tier = 'HAIKU' AND a.success_rate < 0.70 THEN 'CONSIDER_SONNET'
        ELSE 'OPTIMAL'
    END as recommendation,
    CASE 
        WHEN a.model_tier = 'OPUS' AND a.success_rate > 0.95 THEN ct.total_cost * 0.8
        WHEN a.model_tier = 'SONNET' AND a.success_rate > 0.98 THEN ct.total_cost * 0.83
        ELSE 0
    END as potential_monthly_savings
FROM public.archon_agents_v3 a
JOIN (
    SELECT 
        agent_id,
        SUM(total_cost) as total_cost,
        AVG(total_cost) as avg_cost_per_task,
        COUNT(*) as task_count
    FROM public.archon_cost_tracking
    WHERE recorded_at > NOW() - INTERVAL '30 days'
    GROUP BY agent_id
) ct ON a.id = ct.agent_id;

-- Recreate project intelligence overview view (explicitly WITHOUT SECURITY DEFINER)
CREATE VIEW public.archon_project_intelligence_overview
WITH (security_invoker=true)  -- Explicitly set security invoker
AS
SELECT 
    p.id as project_id,
    COALESCE(p.name, 'Project ' || LEFT(p.id::text, 8)) as project_name,
    COUNT(a.id) as total_agents,
    COUNT(CASE WHEN a.state = 'ACTIVE' THEN 1 END) as active_agents,
    COUNT(CASE WHEN a.model_tier = 'OPUS' THEN 1 END) as opus_agents,
    COUNT(CASE WHEN a.model_tier = 'SONNET' THEN 1 END) as sonnet_agents,
    COUNT(CASE WHEN a.model_tier = 'HAIKU' THEN 1 END) as haiku_agents,
    AVG(a.success_rate) as avg_success_rate,
    SUM(a.tasks_completed) as total_tasks_completed,
    COALESCE(ct.monthly_cost, 0) as monthly_cost,
    bc.monthly_budget,
    CASE 
        WHEN bc.monthly_budget > 0 THEN (ct.monthly_cost / bc.monthly_budget * 100)
        ELSE 0
    END as budget_utilization_percent,
    COALESCE(sc.active_contexts, 0) as active_shared_contexts,
    COALESCE(bm.recent_broadcasts, 0) as recent_broadcasts
FROM public.archon_projects p
LEFT JOIN public.archon_agents_v3 a ON p.id = a.project_id
LEFT JOIN (
    SELECT project_id, SUM(total_cost) as monthly_cost
    FROM public.archon_cost_tracking
    WHERE recorded_at > NOW() - INTERVAL '30 days'
    GROUP BY project_id
) ct ON p.id = ct.project_id
LEFT JOIN public.archon_budget_constraints bc ON p.id = bc.project_id
LEFT JOIN (
    SELECT project_id, COUNT(*) as active_contexts
    FROM public.archon_shared_contexts
    WHERE is_active = true
    GROUP BY project_id
) sc ON p.id = sc.project_id
LEFT JOIN (
    SELECT a_sender.project_id, COUNT(*) as recent_broadcasts
    FROM public.archon_broadcast_messages bm
    JOIN public.archon_agents_v3 a_sender ON bm.sender_id = a_sender.id
    WHERE bm.sent_at > NOW() - INTERVAL '7 days'
    GROUP BY a_sender.project_id
) bm ON p.id = bm.project_id
GROUP BY p.id, p.name, ct.monthly_cost, bc.monthly_budget, sc.active_contexts, bm.recent_broadcasts;

-- Recreate agent performance dashboard view (explicitly WITHOUT SECURITY DEFINER)
CREATE VIEW public.archon_agent_performance_dashboard
WITH (security_invoker=true)  -- Explicitly set security invoker
AS
SELECT 
    a.id,
    a.name,
    a.agent_type,
    a.model_tier,
    a.state,
    a.tasks_completed,
    a.success_rate,
    a.avg_completion_time_seconds,
    COALESCE(ct.cost_last_30_days, 0) as cost_last_30_days,
    COALESCE(ak.knowledge_items_count, 0) as knowledge_items_count,
    CASE 
        WHEN a.last_active_at > NOW() - INTERVAL '1 hour' THEN 'HIGH'
        WHEN a.last_active_at > NOW() - INTERVAL '1 day' THEN 'MEDIUM'
        ELSE 'LOW'
    END as activity_level
FROM public.archon_agents_v3 a
LEFT JOIN (
    SELECT agent_id, SUM(total_cost) as cost_last_30_days
    FROM public.archon_cost_tracking
    WHERE recorded_at > NOW() - INTERVAL '30 days'
    GROUP BY agent_id
) ct ON a.id = ct.agent_id
LEFT JOIN (
    SELECT agent_id, COUNT(*) as knowledge_items_count
    FROM public.archon_agent_knowledge
    GROUP BY agent_id
) ak ON a.id = ak.agent_id;

-- Verify that views were created without SECURITY DEFINER
SELECT 
    schemaname,
    viewname,
    CASE 
        WHEN definition ILIKE '%security definer%' THEN 'HAS SECURITY DEFINER' 
        ELSE 'SECURITY INVOKER (GOOD)'
    END as security_status
FROM pg_views 
WHERE schemaname = 'public' 
  AND viewname IN (
    'archon_cost_optimization_recommendations',
    'archon_project_intelligence_overview', 
    'archon_agent_performance_dashboard'
  )
ORDER BY viewname;

-- Script completed - views recreated without SECURITY DEFINER
-- Check the verification query above to confirm views are secure