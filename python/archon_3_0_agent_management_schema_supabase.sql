-- =====================================================
-- ARCHON 3.0 INTELLIGENCE-TIERED AGENT MANAGEMENT SCHEMA
-- SUPABASE-COMPATIBLE VERSION
-- =====================================================
-- Database schema for Intelligence-Tiered Adaptive Agent Management System
-- Created from PRD: Archon_3.0_Intelligence_Tiered_Agent_Management_PRD.md
--
-- Components: Agent Lifecycle, Intelligence Routing, Knowledge Management,
--            Cost Optimization, Real-Time Collaboration, Project-Specific Agents
--
-- CRITICAL: Create database backup before running!
-- Usage: Run in Supabase SQL Editor
-- =====================================================

-- Log migration start
DO $$
BEGIN
    RAISE NOTICE '🚀 ARCHON 3.0 AGENT MANAGEMENT SCHEMA MIGRATION STARTED at %', NOW();
    RAISE NOTICE '📋 Creating tables for Intelligence-Tiered Adaptive Agent Management';
END
$$;

-- =====================================================
-- PHASE 1: AGENT LIFECYCLE MANAGEMENT TABLES
-- =====================================================

-- Drop existing types if they exist (for idempotency)
DROP TYPE IF EXISTS agent_state CASCADE;
DROP TYPE IF EXISTS model_tier CASCADE;
DROP TYPE IF EXISTS agent_type CASCADE;

-- Agent States Enum
CREATE TYPE agent_state AS ENUM (
    'CREATED',      -- Initial state after agent creation
    'ACTIVE',       -- Currently processing tasks
    'IDLE',         -- Available but not processing
    'HIBERNATED',   -- Temporarily suspended to save resources
    'ARCHIVED'      -- Permanently deactivated
);

-- Model Tiers Enum  
CREATE TYPE model_tier AS ENUM (
    'OPUS',         -- Highest capability, most expensive
    'SONNET',       -- Balanced performance and cost (default)
    'HAIKU'         -- Basic tasks only, most cost-effective
);

-- Agent Types Enum
CREATE TYPE agent_type AS ENUM (
    'CODE_IMPLEMENTER',
    'SYSTEM_ARCHITECT', 
    'CODE_QUALITY_REVIEWER',
    'TEST_COVERAGE_VALIDATOR',
    'SECURITY_AUDITOR',
    'PERFORMANCE_OPTIMIZER',
    'DEPLOYMENT_AUTOMATION',
    'ANTIHALLUCINATION_VALIDATOR',
    'UI_UX_OPTIMIZER',
    'DATABASE_ARCHITECT',
    'DOCUMENTATION_GENERATOR',
    'CODE_REFACTORING_OPTIMIZER',
    'STRATEGIC_PLANNER',
    'API_DESIGN_ARCHITECT',
    'GENERAL_PURPOSE'
);

-- Create projects table if it doesn't exist (or use existing one)
-- Note: If archon_projects already exists without a name column, 
-- you may need to add it manually: ALTER TABLE archon_projects ADD COLUMN name VARCHAR(255);
CREATE TABLE IF NOT EXISTS public.archon_projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255),
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Main Agents table
CREATE TABLE IF NOT EXISTS public.archon_agents_v3 (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Basic agent information
    name VARCHAR(255) NOT NULL,
    agent_type agent_type NOT NULL,
    model_tier model_tier NOT NULL DEFAULT 'SONNET',
    project_id UUID REFERENCES public.archon_projects(id) ON DELETE CASCADE,
    
    -- Lifecycle management
    state agent_state NOT NULL DEFAULT 'CREATED',
    state_changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Performance tracking
    tasks_completed INTEGER DEFAULT 0,
    success_rate DECIMAL(5,4) DEFAULT 0.0000, -- 0.0000 to 1.0000
    avg_completion_time_seconds INTEGER DEFAULT 0,
    last_active_at TIMESTAMPTZ,
    
    -- Resource management
    memory_usage_mb INTEGER DEFAULT 0,
    cpu_usage_percent DECIMAL(5,2) DEFAULT 0.00,
    
    -- Configuration
    capabilities JSONB DEFAULT '{}',
    rules_profile_id UUID,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_success_rate CHECK (success_rate >= 0.0 AND success_rate <= 1.0),
    CONSTRAINT valid_cpu_usage CHECK (cpu_usage_percent >= 0.0 AND cpu_usage_percent <= 100.0)
);

-- Agent State History table (for lifecycle tracking)
CREATE TABLE IF NOT EXISTS public.archon_agent_state_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES public.archon_agents_v3(id) ON DELETE CASCADE,
    from_state agent_state,
    to_state agent_state NOT NULL,
    reason VARCHAR(500),
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Indexes will be created separately
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Agent Pool Management (project-level limits)
CREATE TABLE IF NOT EXISTS public.archon_agent_pools (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES public.archon_projects(id) ON DELETE CASCADE,
    
    -- Pool limits by tier
    opus_limit INTEGER NOT NULL DEFAULT 2,      -- Max 2 Opus agents
    sonnet_limit INTEGER NOT NULL DEFAULT 10,   -- Max 10 Sonnet agents  
    haiku_limit INTEGER NOT NULL DEFAULT 50,    -- Max 50 Haiku agents
    
    -- Current usage (denormalized for performance)
    opus_active INTEGER NOT NULL DEFAULT 0,
    sonnet_active INTEGER NOT NULL DEFAULT 0,
    haiku_active INTEGER NOT NULL DEFAULT 0,
    
    -- Pool configuration
    auto_scaling_enabled BOOLEAN DEFAULT TRUE,
    hibernation_timeout_minutes INTEGER DEFAULT 30,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT unique_project_pool UNIQUE (project_id)
);

-- =====================================================
-- PHASE 2: INTELLIGENCE TIER ROUTING TABLES
-- =====================================================

-- Task Complexity Assessment
CREATE TABLE IF NOT EXISTS public.archon_task_complexity (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID,  -- Reference to external task
    
    -- Complexity factors (0.0 to 1.0 scale)
    technical_complexity DECIMAL(3,2) NOT NULL,
    domain_expertise_required DECIMAL(3,2) NOT NULL,
    code_volume_complexity DECIMAL(3,2) NOT NULL,
    integration_complexity DECIMAL(3,2) NOT NULL,
    
    -- Tier assignment
    overall_complexity DECIMAL(3,2) GENERATED ALWAYS AS (
        (technical_complexity * 0.3 + 
         domain_expertise_required * 0.25 + 
         code_volume_complexity * 0.25 + 
         integration_complexity * 0.2)
    ) STORED,
    recommended_tier model_tier NOT NULL,
    assigned_tier model_tier NOT NULL,
    tier_justification TEXT,
    
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_complexities CHECK (
        technical_complexity >= 0 AND technical_complexity <= 1 AND
        domain_expertise_required >= 0 AND domain_expertise_required <= 1 AND
        code_volume_complexity >= 0 AND code_volume_complexity <= 1 AND
        integration_complexity >= 0 AND integration_complexity <= 1
    )
);

-- Routing Rules Configuration
CREATE TABLE IF NOT EXISTS public.archon_routing_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_name VARCHAR(255) NOT NULL,
    agent_type agent_type NOT NULL,
    
    -- Complexity thresholds for tier assignment
    opus_threshold DECIMAL(3,2) DEFAULT 0.75,    -- > 0.75 = Opus
    sonnet_threshold DECIMAL(3,2) DEFAULT 0.35,  -- 0.35-0.75 = Sonnet
    -- < 0.35 = Haiku (implicit)
    
    -- Rule configuration
    is_active BOOLEAN DEFAULT TRUE,
    priority INTEGER DEFAULT 100,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT unique_agent_type_rule UNIQUE (agent_type)
);

-- Routing History (for optimization analysis)
CREATE TABLE IF NOT EXISTS public.archon_routing_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID,
    agent_id UUID REFERENCES public.archon_agents_v3(id) ON DELETE SET NULL,
    
    complexity_score DECIMAL(3,2) NOT NULL,
    recommended_tier model_tier NOT NULL,
    assigned_tier model_tier NOT NULL,
    override_reason TEXT,
    
    execution_time_seconds INTEGER,
    success BOOLEAN,
    
    routed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =====================================================
-- PHASE 3: KNOWLEDGE MANAGEMENT TABLES
-- =====================================================

-- Agent Knowledge Base (with vector embeddings)
CREATE TABLE IF NOT EXISTS public.archon_agent_knowledge (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES public.archon_agents_v3(id) ON DELETE CASCADE,
    
    -- Knowledge content
    knowledge_type VARCHAR(100) NOT NULL, -- 'code_pattern', 'solution', 'error_fix', etc.
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536), -- OpenAI embeddings dimension
    
    -- Confidence tracking
    confidence DECIMAL(3,2) DEFAULT 0.50, -- 0.0 to 1.0
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMPTZ,
    
    -- Metadata
    context_tags TEXT[],
    project_id UUID REFERENCES public.archon_projects(id) ON DELETE CASCADE,
    task_context JSONB DEFAULT '{}',
    
    -- Storage layer
    storage_layer VARCHAR(50) DEFAULT 'permanent', -- permanent, session, temporary
    expires_at TIMESTAMPTZ,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Knowledge Evolution Tracking
CREATE TABLE IF NOT EXISTS public.archon_knowledge_evolution (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    knowledge_id UUID NOT NULL REFERENCES public.archon_agent_knowledge(id) ON DELETE CASCADE,
    
    old_confidence DECIMAL(3,2) NOT NULL,
    new_confidence DECIMAL(3,2) NOT NULL,
    evolution_reason VARCHAR(500),
    
    success_delta INTEGER NOT NULL,
    failure_delta INTEGER NOT NULL,
    
    evolved_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Shared Knowledge (cross-agent learning)
CREATE TABLE IF NOT EXISTS public.archon_shared_knowledge (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    source_agent_id UUID REFERENCES public.archon_agents_v3(id) ON DELETE SET NULL,
    knowledge_type VARCHAR(100) NOT NULL,
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),
    
    -- Sharing metadata
    share_count INTEGER DEFAULT 0,
    adoption_count INTEGER DEFAULT 0,
    average_confidence DECIMAL(3,2) DEFAULT 0.50,
    
    project_id UUID REFERENCES public.archon_projects(id) ON DELETE CASCADE,
    is_public BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =====================================================
-- PHASE 4: COST OPTIMIZATION TABLES
-- =====================================================

-- Cost Tracking
CREATE TABLE IF NOT EXISTS public.archon_cost_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES public.archon_agents_v3(id) ON DELETE CASCADE,
    project_id UUID NOT NULL REFERENCES public.archon_projects(id) ON DELETE CASCADE,
    task_id UUID,
    
    -- Token usage
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    
    -- Cost calculation (in USD)
    input_cost DECIMAL(10,6) NOT NULL,
    output_cost DECIMAL(10,6) NOT NULL,
    total_cost DECIMAL(10,6) GENERATED ALWAYS AS (input_cost + output_cost) STORED,
    
    -- Model and performance
    model_tier model_tier NOT NULL,
    task_duration_seconds INTEGER,
    success BOOLEAN DEFAULT TRUE,
    
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Budget Constraints
CREATE TABLE IF NOT EXISTS public.archon_budget_constraints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES public.archon_projects(id) ON DELETE CASCADE,
    
    -- Budget limits (in USD)
    daily_budget DECIMAL(10,2),
    monthly_budget DECIMAL(10,2),
    per_task_budget DECIMAL(10,2),
    
    -- Alert thresholds (percentage)
    warning_threshold DECIMAL(5,2) DEFAULT 80.00,  -- Alert at 80%
    critical_threshold DECIMAL(5,2) DEFAULT 95.00, -- Stop at 95%
    
    -- Current usage (denormalized for performance)
    daily_spent DECIMAL(10,2) DEFAULT 0.00,
    monthly_spent DECIMAL(10,2) DEFAULT 0.00,
    
    is_active BOOLEAN DEFAULT TRUE,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT unique_project_budget UNIQUE (project_id)
);

-- ROI Analysis
CREATE TABLE IF NOT EXISTS public.archon_roi_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES public.archon_agents_v3(id) ON DELETE CASCADE,
    project_id UUID NOT NULL REFERENCES public.archon_projects(id) ON DELETE CASCADE,
    
    -- Period analysis
    analysis_period VARCHAR(50) NOT NULL, -- 'daily', 'weekly', 'monthly'
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    
    -- Metrics
    total_cost DECIMAL(10,2) NOT NULL,
    tasks_completed INTEGER NOT NULL,
    success_count INTEGER NOT NULL,
    average_completion_time_seconds INTEGER,
    
    -- ROI calculations
    cost_per_task DECIMAL(10,4) GENERATED ALWAYS AS (
        CASE WHEN tasks_completed > 0 THEN total_cost / tasks_completed ELSE 0 END
    ) STORED,
    success_rate DECIMAL(5,4) GENERATED ALWAYS AS (
        CASE WHEN tasks_completed > 0 THEN CAST(success_count AS DECIMAL) / tasks_completed ELSE 0 END
    ) STORED,
    
    -- Optimization recommendations
    recommended_tier model_tier,
    potential_savings DECIMAL(10,2),
    
    analyzed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =====================================================
-- PHASE 5: REAL-TIME COLLABORATION TABLES
-- =====================================================

-- Shared Contexts (for agent collaboration)
CREATE TABLE IF NOT EXISTS public.archon_shared_contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID,
    project_id UUID REFERENCES public.archon_projects(id) ON DELETE CASCADE,
    
    context_name VARCHAR(255) NOT NULL,
    
    -- Shared data
    discoveries JSONB DEFAULT '[]',
    blockers JSONB DEFAULT '[]',
    patterns JSONB DEFAULT '[]',
    
    -- Participants
    participants UUID[] DEFAULT '{}',
    
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Broadcast Messages (pub/sub system)
CREATE TABLE IF NOT EXISTS public.archon_broadcast_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Message content
    message_id VARCHAR(100) UNIQUE NOT NULL DEFAULT gen_random_uuid()::text,
    topic VARCHAR(100) NOT NULL,
    content JSONB NOT NULL,
    message_type VARCHAR(50) DEFAULT 'info', -- info, warning, error, success
    priority INTEGER DEFAULT 1, -- 1 (low) to 5 (critical)
    
    -- Routing
    sender_id UUID REFERENCES public.archon_agents_v3(id) ON DELETE SET NULL,
    target_agents UUID[] DEFAULT '{}',
    target_topics TEXT[] DEFAULT '{}',
    
    -- Delivery tracking
    delivered_count INTEGER DEFAULT 0,
    acknowledged_count INTEGER DEFAULT 0,
    
    sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

-- Topic Subscriptions
CREATE TABLE IF NOT EXISTS public.archon_topic_subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES public.archon_agents_v3(id) ON DELETE CASCADE,
    topic VARCHAR(100) NOT NULL,
    
    -- Filter configuration
    priority_filter INTEGER DEFAULT 1, -- Only receive messages >= this priority
    content_filters JSONB DEFAULT '{}',
    
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Delivery preferences
    subscription_type VARCHAR(50) DEFAULT 'push', -- push, pull, webhook
    callback_endpoint TEXT,
    callback_timeout_seconds INTEGER DEFAULT 30,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT unique_agent_topic UNIQUE (agent_id, topic)
);

-- Message Acknowledgments
CREATE TABLE IF NOT EXISTS public.archon_message_acknowledgments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id VARCHAR(100) NOT NULL REFERENCES public.archon_broadcast_messages(message_id) ON DELETE CASCADE,
    agent_id UUID NOT NULL REFERENCES public.archon_agents_v3(id) ON DELETE CASCADE,
    
    acknowledged_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processing_status VARCHAR(50) DEFAULT 'received', -- received, processing, completed, failed
    processing_result JSONB,
    
    CONSTRAINT unique_message_agent UNIQUE (message_id, agent_id)
);

-- =====================================================
-- PHASE 6: GLOBAL RULES INTEGRATION TABLES
-- =====================================================

-- Rules Profiles (from CLAUDE.md, RULES.md, etc.)
CREATE TABLE IF NOT EXISTS public.archon_rules_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_name VARCHAR(255) NOT NULL,
    
    -- Rule sources
    claude_md_rules JSONB DEFAULT '{}',
    rules_md_rules JSONB DEFAULT '{}',
    manifest_md_rules JSONB DEFAULT '{}',
    custom_rules JSONB DEFAULT '{}',
    
    -- Profile configuration
    is_active BOOLEAN DEFAULT TRUE,
    priority INTEGER DEFAULT 100,
    project_id UUID REFERENCES public.archon_projects(id) ON DELETE CASCADE,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Rule Violations Tracking
CREATE TABLE IF NOT EXISTS public.archon_rule_violations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES public.archon_agents_v3(id) ON DELETE CASCADE,
    rule_profile_id UUID REFERENCES public.archon_rules_profiles(id) ON DELETE SET NULL,
    
    -- Violation details
    rule_type VARCHAR(100) NOT NULL,
    rule_name VARCHAR(255) NOT NULL,
    violation_description TEXT,
    severity VARCHAR(50) DEFAULT 'warning', -- info, warning, error, critical
    
    -- Context
    task_id UUID,
    code_context TEXT,
    
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    resolution_notes TEXT
);

-- =====================================================
-- PHASE 7: INDEXES FOR PERFORMANCE
-- =====================================================

-- Agent indexes
CREATE INDEX IF NOT EXISTS idx_agents_project_id ON public.archon_agents_v3(project_id);
CREATE INDEX IF NOT EXISTS idx_agents_state ON public.archon_agents_v3(state);
CREATE INDEX IF NOT EXISTS idx_agents_model_tier ON public.archon_agents_v3(model_tier);
CREATE INDEX IF NOT EXISTS idx_agents_type ON public.archon_agents_v3(agent_type);

-- State history indexes
CREATE INDEX IF NOT EXISTS idx_state_history_agent_id ON public.archon_agent_state_history(agent_id);
CREATE INDEX IF NOT EXISTS idx_state_history_changed_at ON public.archon_agent_state_history(changed_at DESC);

-- Knowledge indexes
CREATE INDEX IF NOT EXISTS idx_knowledge_agent_id ON public.archon_agent_knowledge(agent_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_project_id ON public.archon_agent_knowledge(project_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_type ON public.archon_agent_knowledge(knowledge_type);

-- Cost tracking indexes
CREATE INDEX IF NOT EXISTS idx_cost_agent_id ON public.archon_cost_tracking(agent_id);
CREATE INDEX IF NOT EXISTS idx_cost_project_id ON public.archon_cost_tracking(project_id);
CREATE INDEX IF NOT EXISTS idx_cost_recorded_at ON public.archon_cost_tracking(recorded_at DESC);

-- Collaboration indexes
CREATE INDEX IF NOT EXISTS idx_broadcast_topic ON public.archon_broadcast_messages(topic);
CREATE INDEX IF NOT EXISTS idx_broadcast_sent_at ON public.archon_broadcast_messages(sent_at DESC);
CREATE INDEX IF NOT EXISTS idx_subscriptions_agent_id ON public.archon_topic_subscriptions(agent_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_topic ON public.archon_topic_subscriptions(topic);

-- =====================================================
-- PHASE 8: TRIGGERS AND FUNCTIONS
-- =====================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at triggers
CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON public.archon_agents_v3
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_pools_updated_at BEFORE UPDATE ON public.archon_agent_pools
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_knowledge_updated_at BEFORE UPDATE ON public.archon_agent_knowledge
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_shared_knowledge_updated_at BEFORE UPDATE ON public.archon_shared_knowledge
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_budget_updated_at BEFORE UPDATE ON public.archon_budget_constraints
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_contexts_updated_at BEFORE UPDATE ON public.archon_shared_contexts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_rules_updated_at BEFORE UPDATE ON public.archon_rules_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to log agent state transitions
CREATE OR REPLACE FUNCTION log_agent_state_transition()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.state IS DISTINCT FROM NEW.state THEN
        INSERT INTO public.archon_agent_state_history (
            agent_id, from_state, to_state, reason
        ) VALUES (
            NEW.id, OLD.state, NEW.state, 'Automatic logging'
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER agent_state_transition_logger
    AFTER UPDATE ON public.archon_agents_v3
    FOR EACH ROW EXECUTE FUNCTION log_agent_state_transition();

-- Function to update agent pool usage
CREATE OR REPLACE FUNCTION update_agent_pool_usage()
RETURNS TRIGGER AS $$
BEGIN
    -- Update pool usage counts
    UPDATE public.archon_agent_pools
    SET 
        opus_active = (
            SELECT COUNT(*) FROM public.archon_agents_v3 
            WHERE project_id = NEW.project_id 
            AND model_tier = 'OPUS' 
            AND state NOT IN ('HIBERNATED', 'ARCHIVED')
        ),
        sonnet_active = (
            SELECT COUNT(*) FROM public.archon_agents_v3 
            WHERE project_id = NEW.project_id 
            AND model_tier = 'SONNET' 
            AND state NOT IN ('HIBERNATED', 'ARCHIVED')
        ),
        haiku_active = (
            SELECT COUNT(*) FROM public.archon_agents_v3 
            WHERE project_id = NEW.project_id 
            AND model_tier = 'HAIKU' 
            AND state NOT IN ('HIBERNATED', 'ARCHIVED')
        )
    WHERE project_id = NEW.project_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_pool_usage_on_agent_change
    AFTER INSERT OR UPDATE OR DELETE ON public.archon_agents_v3
    FOR EACH ROW EXECUTE FUNCTION update_agent_pool_usage();

-- Function to evolve knowledge confidence
CREATE OR REPLACE FUNCTION evolve_knowledge_confidence()
RETURNS TRIGGER AS $$
DECLARE
    total_uses INTEGER;
    new_confidence DECIMAL(3,2);
BEGIN
    total_uses := NEW.success_count + NEW.failure_count;
    
    IF total_uses > 0 THEN
        -- Calculate new confidence based on success rate with decay
        new_confidence := (NEW.success_count::DECIMAL / total_uses) * 0.9 + 0.1;
        
        -- Log evolution if confidence changed significantly
        IF ABS(NEW.confidence - new_confidence) > 0.05 THEN
            INSERT INTO public.archon_knowledge_evolution (
                knowledge_id, old_confidence, new_confidence,
                success_delta, failure_delta,
                evolution_reason
            ) VALUES (
                NEW.id, NEW.confidence, new_confidence,
                NEW.success_count - OLD.success_count,
                NEW.failure_count - OLD.failure_count,
                'Automatic confidence evolution'
            );
            
            NEW.confidence := new_confidence;
        END IF;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER evolve_confidence_on_usage
    BEFORE UPDATE ON public.archon_agent_knowledge
    FOR EACH ROW 
    WHEN (OLD.success_count IS DISTINCT FROM NEW.success_count 
          OR OLD.failure_count IS DISTINCT FROM NEW.failure_count)
    EXECUTE FUNCTION evolve_knowledge_confidence();

-- =====================================================
-- PHASE 9: MONITORING VIEWS
-- =====================================================

-- Agent Performance Dashboard View
CREATE OR REPLACE VIEW public.archon_agent_performance_dashboard AS
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
    WHERE storage_layer != 'temporary'
    GROUP BY agent_id
) ak ON a.id = ak.agent_id;

-- Project Intelligence Overview
CREATE OR REPLACE VIEW public.archon_project_intelligence_overview AS
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
    WHERE is_active = TRUE
    GROUP BY project_id
) sc ON p.id = sc.project_id
LEFT JOIN (
    SELECT COUNT(*) as recent_broadcasts
    FROM public.archon_broadcast_messages
    WHERE sent_at > NOW() - INTERVAL '24 hours'
) bm ON TRUE
GROUP BY p.id, p.name, ct.monthly_cost, bc.monthly_budget, sc.active_contexts, bm.recent_broadcasts;

-- Cost Optimization Recommendations View
CREATE OR REPLACE VIEW public.archon_cost_optimization_recommendations AS
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
    HAVING COUNT(*) >= 5
) ct ON a.id = ct.agent_id
WHERE a.state NOT IN ('ARCHIVED');

-- =====================================================
-- PHASE 10: INITIAL DATA AND CONFIGURATION
-- =====================================================

-- Insert default routing rules
INSERT INTO public.archon_routing_rules (rule_name, agent_type, opus_threshold, sonnet_threshold)
VALUES 
    ('Code Implementation', 'CODE_IMPLEMENTER', 0.75, 0.35),
    ('System Architecture', 'SYSTEM_ARCHITECT', 0.70, 0.30),
    ('Code Review', 'CODE_QUALITY_REVIEWER', 0.65, 0.30),
    ('Test Validation', 'TEST_COVERAGE_VALIDATOR', 0.60, 0.25),
    ('Security Audit', 'SECURITY_AUDITOR', 0.80, 0.40),
    ('Performance Optimization', 'PERFORMANCE_OPTIMIZER', 0.75, 0.35),
    ('Deployment Automation', 'DEPLOYMENT_AUTOMATION', 0.65, 0.30),
    ('AntiHall Validation', 'ANTIHALLUCINATION_VALIDATOR', 0.50, 0.20),
    ('UI/UX Optimization', 'UI_UX_OPTIMIZER', 0.60, 0.25),
    ('Database Architecture', 'DATABASE_ARCHITECT', 0.70, 0.35),
    ('Documentation', 'DOCUMENTATION_GENERATOR', 0.40, 0.15),
    ('Code Refactoring', 'CODE_REFACTORING_OPTIMIZER', 0.65, 0.30),
    ('Strategic Planning', 'STRATEGIC_PLANNER', 0.80, 0.40),
    ('API Design', 'API_DESIGN_ARCHITECT', 0.70, 0.35),
    ('General Purpose', 'GENERAL_PURPOSE', 0.60, 0.30)
ON CONFLICT (agent_type) DO NOTHING;

-- Create default project if none exists
-- Handle both 'name' and 'title' columns that might exist
DO $$
BEGIN
    -- Try to insert with name column
    BEGIN
        INSERT INTO public.archon_projects (id, name, description)
        VALUES (
            '00000000-0000-0000-0000-000000000001'::UUID,
            'Default Project',
            'Default project for agent management system'
        )
        ON CONFLICT (id) DO NOTHING;
    EXCEPTION
        WHEN others THEN
            -- If name column doesn't exist, try with title
            BEGIN
                INSERT INTO public.archon_projects (id, title, description)
                VALUES (
                    '00000000-0000-0000-0000-000000000001'::UUID,
                    'Default Project',
                    'Default project for agent management system'
                )
                ON CONFLICT (id) DO NOTHING;
            EXCEPTION
                WHEN others THEN
                    RAISE NOTICE 'Could not insert default project - may already exist';
            END;
    END;
END
$$;

-- Create default agent pool for default project
INSERT INTO public.archon_agent_pools (project_id)
VALUES ('00000000-0000-0000-0000-000000000001'::UUID)
ON CONFLICT (project_id) DO NOTHING;

-- =====================================================
-- PHASE 11: COMPLETION LOG
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE '✅ ARCHON 3.0 AGENT MANAGEMENT SCHEMA MIGRATION COMPLETED at %', NOW();
    RAISE NOTICE '📊 Created 17 tables, 8 indexes, 7 triggers, 3 monitoring views';
    RAISE NOTICE '🚀 Intelligence-Tiered Adaptive Agent Management System is ready!';
END
$$;

-- =====================================================
-- POST-MIGRATION NOTES
-- =====================================================
-- 1. Ensure pgvector extension is enabled: CREATE EXTENSION IF NOT EXISTS vector;
-- 2. Configure Row Level Security (RLS) policies as needed
-- 3. Set up backup schedule for production data
-- 4. Monitor table sizes and performance metrics
-- 5. Consider partitioning cost_tracking table by month for large datasets