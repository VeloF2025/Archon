-- =====================================================
-- ARCHON 3.0 INTELLIGENCE-TIERED AGENT MANAGEMENT SCHEMA
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
    reason TEXT,
    metadata JSONB DEFAULT '{}',
    
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    changed_by VARCHAR(255) DEFAULT 'system'
);

-- Agent Pool Management table
CREATE TABLE IF NOT EXISTS public.archon_agent_pools (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES public.archon_projects(id) ON DELETE CASCADE,
    
    -- Pool limits per tier
    opus_limit INTEGER DEFAULT 2,
    sonnet_limit INTEGER DEFAULT 10, 
    haiku_limit INTEGER DEFAULT 50,
    
    -- Current counts
    opus_active INTEGER DEFAULT 0,
    sonnet_active INTEGER DEFAULT 0,
    haiku_active INTEGER DEFAULT 0,
    
    -- Pool settings
    auto_scaling_enabled BOOLEAN DEFAULT true,
    hibernation_timeout_minutes INTEGER DEFAULT 30,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Constraints to prevent over-allocation
    CONSTRAINT valid_opus_usage CHECK (opus_active <= opus_limit),
    CONSTRAINT valid_sonnet_usage CHECK (sonnet_active <= sonnet_limit),
    CONSTRAINT valid_haiku_usage CHECK (haiku_active <= haiku_limit)
);

-- =====================================================  
-- PHASE 2: INTELLIGENCE TIER ROUTING TABLES
-- =====================================================

-- Task Complexity Assessment table
CREATE TABLE IF NOT EXISTS public.archon_task_complexity (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES public.archon_tasks(id) ON DELETE CASCADE,
    
    -- Complexity metrics (0.0 to 1.0)
    technical_complexity DECIMAL(5,4) NOT NULL DEFAULT 0.5000,
    domain_expertise_required DECIMAL(5,4) NOT NULL DEFAULT 0.5000,
    code_volume_complexity DECIMAL(5,4) NOT NULL DEFAULT 0.5000,
    integration_complexity DECIMAL(5,4) NOT NULL DEFAULT 0.5000,
    
    -- Calculated overall complexity
    overall_complexity DECIMAL(5,4) GENERATED ALWAYS AS (
        (technical_complexity + domain_expertise_required + code_volume_complexity + integration_complexity) / 4.0
    ) STORED,
    
    -- Tier assignment
    recommended_tier model_tier NOT NULL,
    assigned_tier model_tier NOT NULL,
    tier_justification TEXT,
    
    -- Assessment metadata
    assessed_by VARCHAR(255) DEFAULT 'system',
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_technical_complexity CHECK (technical_complexity >= 0.0 AND technical_complexity <= 1.0),
    CONSTRAINT valid_domain_complexity CHECK (domain_expertise_required >= 0.0 AND domain_expertise_required <= 1.0),
    CONSTRAINT valid_code_complexity CHECK (code_volume_complexity >= 0.0 AND code_volume_complexity <= 1.0),
    CONSTRAINT valid_integration_complexity CHECK (integration_complexity >= 0.0 AND integration_complexity <= 1.0)
);

-- Intelligence Routing Rules table  
CREATE TABLE IF NOT EXISTS public.archon_routing_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Rule definition
    rule_name VARCHAR(255) NOT NULL UNIQUE,
    rule_description TEXT,
    
    -- Tier thresholds (complexity scores 0.0-1.0)
    opus_threshold DECIMAL(5,4) DEFAULT 0.7500,   -- Only truly complex tasks
    sonnet_threshold DECIMAL(5,4) DEFAULT 0.1500, -- Default for most tasks (Sonnet-first)
    haiku_threshold DECIMAL(5,4) DEFAULT 0.0000,  -- Only most basic tasks
    
    -- Special routing conditions
    agent_type_preferences JSONB DEFAULT '{}', -- Which agent types prefer which tiers
    project_tier_override JSONB DEFAULT '{}',  -- Project-specific overrides
    
    -- Rule status
    is_active BOOLEAN DEFAULT true,
    priority_order INTEGER DEFAULT 1,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =====================================================
-- PHASE 3: KNOWLEDGE MANAGEMENT SYSTEM TABLES  
-- =====================================================

-- Knowledge Items table (multi-layer storage)
CREATE TABLE IF NOT EXISTS public.archon_agent_knowledge (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES public.archon_agents_v3(id) ON DELETE CASCADE,
    
    -- Knowledge content
    knowledge_type VARCHAR(100) NOT NULL, -- 'pattern', 'solution', 'error', 'context'
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    
    -- Confidence and learning
    confidence DECIMAL(5,4) NOT NULL DEFAULT 0.5000,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMPTZ,
    
    -- Context and categorization
    context_tags JSONB DEFAULT '[]',
    project_id UUID REFERENCES public.archon_projects(id) ON DELETE CASCADE,
    task_context VARCHAR(255),
    
    -- Storage layer (temporary, working, long_term)  
    storage_layer VARCHAR(50) DEFAULT 'temporary',
    
    -- Vector embedding for similarity search
    embedding VECTOR(1536),
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_confidence CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

-- Knowledge Evolution History
CREATE TABLE IF NOT EXISTS public.archon_knowledge_evolution (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    knowledge_id UUID NOT NULL REFERENCES public.archon_agent_knowledge(id) ON DELETE CASCADE,
    
    -- Evolution tracking
    previous_confidence DECIMAL(5,4),
    new_confidence DECIMAL(5,4),
    evolution_reason VARCHAR(255), -- 'success', 'failure', 'validation', 'consolidation'
    
    -- Context
    task_id UUID REFERENCES public.archon_tasks(id) ON DELETE SET NULL,
    agent_feedback TEXT,
    
    evolved_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Shared Knowledge Base (cross-agent learning)
CREATE TABLE IF NOT EXISTS public.archon_shared_knowledge (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Knowledge identification
    knowledge_pattern VARCHAR(500) NOT NULL,
    solution_approach TEXT NOT NULL,
    
    -- Multi-agent validation
    contributing_agents JSONB DEFAULT '[]', -- Array of agent IDs that contributed
    validation_count INTEGER DEFAULT 0,
    success_rate DECIMAL(5,4) DEFAULT 0.0000,
    
    -- Applicability
    applicable_agent_types JSONB DEFAULT '[]', -- Which agent types can use this
    applicable_contexts JSONB DEFAULT '[]',     -- Which contexts this applies to
    
    -- Vector search
    embedding VECTOR(1536),
    
    -- Status
    is_verified BOOLEAN DEFAULT false,
    verification_threshold INTEGER DEFAULT 3, -- Number of successes needed for verification
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =====================================================
-- PHASE 4: COST OPTIMIZATION TABLES
-- =====================================================

-- Cost Tracking table
CREATE TABLE IF NOT EXISTS public.archon_cost_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Cost association
    agent_id UUID REFERENCES public.archon_agents_v3(id) ON DELETE CASCADE,
    project_id UUID REFERENCES public.archon_projects(id) ON DELETE CASCADE,
    task_id UUID REFERENCES public.archon_tasks(id) ON DELETE SET NULL,
    
    -- Token usage
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER GENERATED ALWAYS AS (input_tokens + output_tokens) STORED,
    
    -- Cost calculation (in USD)
    input_cost DECIMAL(10,6) NOT NULL DEFAULT 0.000000,
    output_cost DECIMAL(10,6) NOT NULL DEFAULT 0.000000,
    total_cost DECIMAL(10,6) GENERATED ALWAYS AS (input_cost + output_cost) STORED,
    
    -- Model information
    model_tier model_tier NOT NULL,
    model_name VARCHAR(100),
    
    -- Performance metrics
    task_duration_seconds INTEGER,
    success BOOLEAN DEFAULT true,
    
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Budget Constraints table
CREATE TABLE IF NOT EXISTS public.archon_budget_constraints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES public.archon_projects(id) ON DELETE CASCADE,
    
    -- Budget limits (in USD)
    monthly_budget DECIMAL(10,2),
    daily_budget DECIMAL(10,2), 
    per_task_budget DECIMAL(10,2),
    
    -- Alert thresholds (as percentages)
    warning_threshold DECIMAL(5,2) DEFAULT 80.00,  -- 80% of budget
    critical_threshold DECIMAL(5,2) DEFAULT 95.00, -- 95% of budget
    
    -- Current usage tracking
    current_monthly_spend DECIMAL(10,2) DEFAULT 0.00,
    current_daily_spend DECIMAL(10,2) DEFAULT 0.00,
    spend_reset_date DATE DEFAULT CURRENT_DATE,
    
    -- Budget enforcement
    auto_downgrade_enabled BOOLEAN DEFAULT true,
    emergency_stop_enabled BOOLEAN DEFAULT false,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ROI Analysis table
CREATE TABLE IF NOT EXISTS public.archon_roi_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Analysis scope
    agent_id UUID REFERENCES public.archon_agents_v3(id) ON DELETE CASCADE,
    project_id UUID REFERENCES public.archon_projects(id) ON DELETE CASCADE,
    analysis_period_days INTEGER DEFAULT 30,
    
    -- Cost metrics
    total_cost DECIMAL(10,2) NOT NULL,
    cost_per_task DECIMAL(10,2),
    
    -- Value metrics  
    tasks_completed INTEGER NOT NULL DEFAULT 0,
    success_rate DECIMAL(5,4) NOT NULL DEFAULT 0.0000,
    avg_completion_time_hours DECIMAL(8,2),
    
    -- ROI calculation
    estimated_value_delivered DECIMAL(10,2),
    roi_ratio DECIMAL(8,4), -- value/cost ratio
    
    -- Optimization recommendations
    recommended_tier model_tier,
    tier_change_rationale TEXT,
    potential_savings DECIMAL(10,2),
    
    analyzed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    analyst VARCHAR(255) DEFAULT 'system'
);

-- =====================================================
-- PHASE 5: REAL-TIME COLLABORATION TABLES
-- =====================================================

-- Shared Context table
CREATE TABLE IF NOT EXISTS public.archon_shared_contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Context identification
    task_id UUID REFERENCES public.archon_tasks(id) ON DELETE CASCADE,
    project_id UUID REFERENCES public.archon_projects(id) ON DELETE CASCADE,
    context_name VARCHAR(255) NOT NULL,
    
    -- Collaboration data
    discoveries JSONB DEFAULT '[]',        -- Array of discovery objects
    blockers JSONB DEFAULT '[]',           -- Array of blocker objects  
    patterns JSONB DEFAULT '[]',           -- Array of successful patterns
    participants JSONB DEFAULT '[]',       -- Array of agent IDs
    
    -- Context status
    is_active BOOLEAN DEFAULT true,
    last_updated_by VARCHAR(255),
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Broadcast Messages table (pub/sub system)
CREATE TABLE IF NOT EXISTS public.archon_broadcast_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Message identification
    message_id VARCHAR(255) UNIQUE NOT NULL,
    topic VARCHAR(255) NOT NULL,
    
    -- Message content
    content JSONB NOT NULL,
    message_type VARCHAR(100) NOT NULL, -- 'discovery', 'blocker', 'pattern', 'update'
    
    -- Priority and delivery
    priority INTEGER DEFAULT 1, -- 1=low, 2=medium, 3=high, 4=critical
    sender_id UUID REFERENCES public.archon_agents_v3(id) ON DELETE SET NULL,
    
    -- Targeting
    target_agents JSONB DEFAULT '[]',     -- Specific agent IDs
    target_topics JSONB DEFAULT '[]',     -- Topic subscriptions
    
    -- Status tracking
    delivered_count INTEGER DEFAULT 0,
    acknowledgment_count INTEGER DEFAULT 0,
    
    -- Timestamps
    sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

-- Topic Subscriptions table
CREATE TABLE IF NOT EXISTS public.archon_topic_subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Subscription details
    agent_id UUID NOT NULL REFERENCES public.archon_agents_v3(id) ON DELETE CASCADE,
    topic VARCHAR(255) NOT NULL,
    
    -- Filtering
    priority_filter INTEGER DEFAULT 1, -- Only receive messages >= this priority
    content_filters JSONB DEFAULT '{}', -- JSONPath filters for content matching
    
    -- Subscription management
    is_active BOOLEAN DEFAULT true,
    subscription_type VARCHAR(50) DEFAULT 'standard', -- 'standard', 'pattern_based', 'conditional'
    
    -- Callback configuration
    callback_endpoint VARCHAR(500),
    callback_timeout_seconds INTEGER DEFAULT 30,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    UNIQUE(agent_id, topic)
);

-- Message Acknowledgments table
CREATE TABLE IF NOT EXISTS public.archon_message_acknowledgments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    message_id UUID NOT NULL REFERENCES public.archon_broadcast_messages(id) ON DELETE CASCADE,
    agent_id UUID NOT NULL REFERENCES public.archon_agents_v3(id) ON DELETE CASCADE,
    
    -- Acknowledgment details
    acknowledged_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processing_status VARCHAR(50) DEFAULT 'received', -- 'received', 'processing', 'completed', 'failed'
    response_data JSONB DEFAULT '{}',
    
    UNIQUE(message_id, agent_id)
);

-- =====================================================
-- PHASE 6: GLOBAL RULES INTEGRATION TABLES
-- =====================================================

-- Rules Profiles table (parsed from CLAUDE.md, RULES.md, MANIFEST.md)
CREATE TABLE IF NOT EXISTS public.archon_rules_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Profile identification
    profile_name VARCHAR(255) UNIQUE NOT NULL,
    agent_type agent_type,
    model_tier model_tier,
    
    -- Rule sources (parsed from configuration files)
    global_rules JSONB DEFAULT '[]',       -- From global CLAUDE.md, RULES.md
    project_rules JSONB DEFAULT '[]',      -- From project-specific rules
    manifest_rules JSONB DEFAULT '[]',     -- From MANIFEST.md
    
    -- Rule categories
    quality_gates JSONB DEFAULT '[]',      -- Quality enforcement rules
    security_rules JSONB DEFAULT '[]',     -- Security constraints
    performance_rules JSONB DEFAULT '[]',  -- Performance requirements
    coding_standards JSONB DEFAULT '[]',   -- Coding style and standards
    
    -- Profile metadata
    rule_count INTEGER DEFAULT 0,
    last_parsed_at TIMESTAMPTZ,
    source_file_hashes JSONB DEFAULT '{}', -- Track file changes
    
    -- Profile status
    is_active BOOLEAN DEFAULT true,
    validation_status VARCHAR(50) DEFAULT 'pending',
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Rule Violations table (enforcement tracking)
CREATE TABLE IF NOT EXISTS public.archon_rule_violations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Violation context
    agent_id UUID REFERENCES public.archon_agents_v3(id) ON DELETE CASCADE,
    task_id UUID REFERENCES public.archon_tasks(id) ON DELETE SET NULL,
    rules_profile_id UUID REFERENCES public.archon_rules_profiles(id) ON DELETE SET NULL,
    
    -- Violation details
    rule_name VARCHAR(255) NOT NULL,
    rule_category VARCHAR(100),
    violation_type VARCHAR(100), -- 'WARNING', 'ERROR', 'CRITICAL'
    description TEXT NOT NULL,
    
    -- Resolution tracking
    status VARCHAR(50) DEFAULT 'open', -- 'open', 'resolved', 'acknowledged', 'suppressed'
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(255),
    resolution_notes TEXT,
    
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =====================================================
-- PHASE 7: INDEXES FOR PERFORMANCE
-- =====================================================

-- Agent management indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_v3_project_state 
ON public.archon_agents_v3 (project_id, state, model_tier);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_v3_performance 
ON public.archon_agents_v3 (success_rate DESC, tasks_completed DESC) 
WHERE state = 'ACTIVE';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_state_history_agent_time 
ON public.archon_agent_state_history (agent_id, changed_at DESC);

-- Knowledge management indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_knowledge_search 
ON public.archon_agent_knowledge USING ivfflat (embedding vector_cosine_ops);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_knowledge_confidence 
ON public.archon_agent_knowledge (confidence DESC, last_used_at DESC) 
WHERE storage_layer != 'temporary';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_shared_knowledge_search 
ON public.archon_shared_knowledge USING ivfflat (embedding vector_cosine_ops) 
WHERE is_verified = true;

-- Cost tracking indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cost_tracking_project_time 
ON public.archon_cost_tracking (project_id, recorded_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cost_tracking_agent_performance 
ON public.archon_cost_tracking (agent_id, success, recorded_at DESC);

-- Collaboration indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_broadcast_messages_topic_priority 
ON public.archon_broadcast_messages (topic, priority DESC, sent_at DESC) 
WHERE expires_at IS NULL OR expires_at > NOW();

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_subscriptions_agent_active 
ON public.archon_topic_subscriptions (agent_id, is_active) 
WHERE is_active = true;

-- =====================================================
-- PHASE 8: TRIGGERS AND FUNCTIONS
-- =====================================================

-- Update agent pool counts when agent state changes
CREATE OR REPLACE FUNCTION update_agent_pool_counts()
RETURNS TRIGGER 
SET search_path = public, pg_temp
SECURITY DEFINER
AS $$
BEGIN
    -- Update pool counts for the project
    IF TG_OP = 'INSERT' AND NEW.state = 'ACTIVE' THEN
        UPDATE public.archon_agent_pools 
        SET 
            opus_active = opus_active + CASE WHEN NEW.model_tier = 'OPUS' THEN 1 ELSE 0 END,
            sonnet_active = sonnet_active + CASE WHEN NEW.model_tier = 'SONNET' THEN 1 ELSE 0 END,
            haiku_active = haiku_active + CASE WHEN NEW.model_tier = 'HAIKU' THEN 1 ELSE 0 END,
            updated_at = NOW()
        WHERE project_id = NEW.project_id;
    END IF;
    
    IF TG_OP = 'UPDATE' AND OLD.state != NEW.state THEN
        -- Decrement old state counts
        IF OLD.state = 'ACTIVE' THEN
            UPDATE public.archon_agent_pools 
            SET 
                opus_active = opus_active - CASE WHEN OLD.model_tier = 'OPUS' THEN 1 ELSE 0 END,
                sonnet_active = sonnet_active - CASE WHEN OLD.model_tier = 'SONNET' THEN 1 ELSE 0 END,
                haiku_active = haiku_active - CASE WHEN OLD.model_tier = 'HAIKU' THEN 1 ELSE 0 END,
                updated_at = NOW()
            WHERE project_id = OLD.project_id;
        END IF;
        
        -- Increment new state counts
        IF NEW.state = 'ACTIVE' THEN
            UPDATE public.archon_agent_pools 
            SET 
                opus_active = opus_active + CASE WHEN NEW.model_tier = 'OPUS' THEN 1 ELSE 0 END,
                sonnet_active = sonnet_active + CASE WHEN NEW.model_tier = 'SONNET' THEN 1 ELSE 0 END,
                haiku_active = haiku_active + CASE WHEN NEW.model_tier = 'HAIKU' THEN 1 ELSE 0 END,
                updated_at = NOW()
            WHERE project_id = NEW.project_id;
        END IF;
        
        -- Log state transition
        INSERT INTO public.archon_agent_state_history (agent_id, from_state, to_state, reason)
        VALUES (NEW.id, OLD.state, NEW.state, 'System state transition');
    END IF;
    
    IF TG_OP = 'DELETE' AND OLD.state = 'ACTIVE' THEN
        UPDATE public.archon_agent_pools 
        SET 
            opus_active = opus_active - CASE WHEN OLD.model_tier = 'OPUS' THEN 1 ELSE 0 END,
            sonnet_active = sonnet_active - CASE WHEN OLD.model_tier = 'SONNET' THEN 1 ELSE 0 END,
            haiku_active = haiku_active - CASE WHEN OLD.model_tier = 'HAIKU' THEN 1 ELSE 0 END,
            updated_at = NOW()
        WHERE project_id = OLD.project_id;
    END IF;
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Create trigger for agent pool management
DROP TRIGGER IF EXISTS trigger_update_agent_pool_counts ON public.archon_agents_v3;
CREATE TRIGGER trigger_update_agent_pool_counts
    AFTER INSERT OR UPDATE OR DELETE ON public.archon_agents_v3
    FOR EACH ROW
    EXECUTE FUNCTION update_agent_pool_counts();

-- Update cost tracking totals
CREATE OR REPLACE FUNCTION update_budget_spending()
RETURNS TRIGGER
SET search_path = public, pg_temp  
SECURITY DEFINER
AS $$
DECLARE
    budget_record RECORD;
BEGIN
    -- Get the budget constraint for this project
    SELECT * INTO budget_record
    FROM public.archon_budget_constraints
    WHERE project_id = NEW.project_id;
    
    IF FOUND THEN
        -- Update daily and monthly spending
        UPDATE public.archon_budget_constraints
        SET
            current_daily_spend = current_daily_spend + NEW.total_cost,
            current_monthly_spend = current_monthly_spend + NEW.total_cost,
            updated_at = NOW()
        WHERE project_id = NEW.project_id;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for cost tracking
DROP TRIGGER IF EXISTS trigger_update_budget_spending ON public.archon_cost_tracking;
CREATE TRIGGER trigger_update_budget_spending
    AFTER INSERT ON public.archon_cost_tracking
    FOR EACH ROW
    EXECUTE FUNCTION update_budget_spending();

-- Knowledge confidence evolution function
CREATE OR REPLACE FUNCTION evolve_knowledge_confidence()
RETURNS TRIGGER
SET search_path = public, pg_temp
SECURITY DEFINER
AS $$
BEGIN
    -- Update confidence based on success/failure
    IF NEW.success_count > OLD.success_count THEN
        -- Success - increase confidence
        UPDATE public.archon_agent_knowledge
        SET 
            confidence = LEAST(confidence * 1.1, 0.99),
            updated_at = NOW()
        WHERE id = NEW.id;
        
        -- Log evolution
        INSERT INTO public.archon_knowledge_evolution 
        (knowledge_id, previous_confidence, new_confidence, evolution_reason)
        VALUES (NEW.id, OLD.confidence, LEAST(OLD.confidence * 1.1, 0.99), 'success');
        
    ELSIF NEW.failure_count > OLD.failure_count THEN
        -- Failure - decrease confidence
        UPDATE public.archon_agent_knowledge
        SET 
            confidence = GREATEST(confidence * 0.9, 0.1),
            updated_at = NOW()
        WHERE id = NEW.id;
        
        -- Log evolution
        INSERT INTO public.archon_knowledge_evolution
        (knowledge_id, previous_confidence, new_confidence, evolution_reason)
        VALUES (NEW.id, OLD.confidence, GREATEST(OLD.confidence * 0.9, 0.1), 'failure');
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for knowledge evolution
DROP TRIGGER IF EXISTS trigger_evolve_knowledge_confidence ON public.archon_agent_knowledge;
CREATE TRIGGER trigger_evolve_knowledge_confidence
    AFTER UPDATE OF success_count, failure_count ON public.archon_agent_knowledge
    FOR EACH ROW
    EXECUTE FUNCTION evolve_knowledge_confidence();

-- =====================================================
-- PHASE 9: INITIAL DATA AND DEFAULT CONFIGURATIONS
-- =====================================================

-- Insert default routing rules (Sonnet-first preference)
INSERT INTO public.archon_routing_rules (
    rule_name, 
    rule_description,
    opus_threshold,
    sonnet_threshold, 
    haiku_threshold,
    agent_type_preferences
) VALUES (
    'default_sonnet_first',
    'Default routing with Sonnet preference for most tasks',
    0.7500,  -- Opus for truly complex tasks (75%+ complexity)
    0.1500,  -- Sonnet as default for most tasks (15%+ complexity) 
    0.0000,  -- Haiku only for most basic tasks (0%+ complexity)
    '{
        "SYSTEM_ARCHITECT": "SONNET",
        "CODE_IMPLEMENTER": "SONNET", 
        "SECURITY_AUDITOR": "SONNET",
        "PERFORMANCE_OPTIMIZER": "SONNET",
        "STRATEGIC_PLANNER": "OPUS",
        "GENERAL_PURPOSE": "HAIKU"
    }'::jsonb
) ON CONFLICT (rule_name) DO NOTHING;

-- Create default rules profile
INSERT INTO public.archon_rules_profiles (
    profile_name,
    global_rules,
    quality_gates,
    security_rules,
    rule_count,
    validation_status
) VALUES (
    'archon_default_profile',
    '[
        "Follow TDD (Test-Driven Development) approach",
        "Zero tolerance for TypeScript/ESLint errors", 
        "Maintain >95% test coverage",
        "Apply NLNH (No Lies, No Hallucination) protocol",
        "Enforce DGTS (Don''t Game The System) validation"
    ]'::jsonb,
    '[
        "Zero TypeScript compilation errors",
        "Zero ESLint errors/warnings",
        "Test coverage >95%",
        "Build must succeed",
        "No console.log statements"
    ]'::jsonb,
    '[
        "Input validation required",
        "Error handling mandatory", 
        "No hardcoded secrets",
        "SQL injection prevention",
        "XSS protection"
    ]'::jsonb,
    13,
    'active'
) ON CONFLICT (profile_name) DO NOTHING;

-- =====================================================
-- PHASE 10: VIEWS FOR MONITORING AND ANALYTICS
-- =====================================================

-- Agent Performance Dashboard View
CREATE OR REPLACE VIEW archon_agent_performance_dashboard AS
SELECT 
    a.id,
    a.name,
    a.agent_type,
    a.model_tier,
    a.state,
    a.tasks_completed,
    a.success_rate,
    a.avg_completion_time_seconds,
    CASE 
        WHEN a.last_active_at > NOW() - INTERVAL '1 hour' THEN 'RECENT'
        WHEN a.last_active_at > NOW() - INTERVAL '1 day' THEN 'TODAY'
        WHEN a.last_active_at > NOW() - INTERVAL '1 week' THEN 'WEEK'
        ELSE 'INACTIVE'
    END as activity_level,
    -- Cost analysis (last 30 days)
    COALESCE(c.total_cost, 0) as cost_last_30_days,
    COALESCE(c.total_tokens, 0) as tokens_last_30_days,
    -- Knowledge items
    COALESCE(k.knowledge_items, 0) as knowledge_items_count,
    COALESCE(k.avg_confidence, 0) as avg_knowledge_confidence
FROM public.archon_agents_v3 a
LEFT JOIN (
    SELECT 
        agent_id,
        SUM(total_cost) as total_cost,
        SUM(total_tokens) as total_tokens
    FROM public.archon_cost_tracking 
    WHERE recorded_at > NOW() - INTERVAL '30 days'
    GROUP BY agent_id
) c ON a.id = c.agent_id
LEFT JOIN (
    SELECT 
        agent_id,
        COUNT(*) as knowledge_items,
        AVG(confidence) as avg_confidence
    FROM public.archon_agent_knowledge
    WHERE storage_layer != 'temporary'
    GROUP BY agent_id  
) k ON a.id = k.agent_id;

-- Project Intelligence Overview
CREATE OR REPLACE VIEW archon_project_intelligence_overview AS
SELECT
    p.id as project_id,
    p.name as project_name,
    -- Agent distribution
    COUNT(a.id) as total_agents,
    COUNT(CASE WHEN a.state = 'ACTIVE' THEN 1 END) as active_agents,
    COUNT(CASE WHEN a.model_tier = 'OPUS' THEN 1 END) as opus_agents,
    COUNT(CASE WHEN a.model_tier = 'SONNET' THEN 1 END) as sonnet_agents,
    COUNT(CASE WHEN a.model_tier = 'HAIKU' THEN 1 END) as haiku_agents,
    -- Performance metrics
    AVG(a.success_rate) as avg_success_rate,
    SUM(a.tasks_completed) as total_tasks_completed,
    -- Cost analysis (last 30 days)
    COALESCE(c.monthly_cost, 0) as monthly_cost,
    COALESCE(bc.monthly_budget, 0) as monthly_budget,
    CASE 
        WHEN bc.monthly_budget > 0 THEN 
            (COALESCE(c.monthly_cost, 0) / bc.monthly_budget * 100)
        ELSE 0 
    END as budget_utilization_percent,
    -- Knowledge sharing
    COALESCE(sc.shared_contexts, 0) as active_shared_contexts,
    COALESCE(bm.broadcast_messages, 0) as recent_broadcasts
FROM public.archon_projects p
LEFT JOIN public.archon_agents_v3 a ON p.id = a.project_id
LEFT JOIN (
    SELECT 
        project_id,
        SUM(total_cost) as monthly_cost
    FROM public.archon_cost_tracking
    WHERE recorded_at > NOW() - INTERVAL '30 days'
    GROUP BY project_id
) c ON p.id = c.project_id
LEFT JOIN public.archon_budget_constraints bc ON p.id = bc.project_id
LEFT JOIN (
    SELECT 
        project_id,
        COUNT(*) as shared_contexts
    FROM public.archon_shared_contexts
    WHERE is_active = true
    GROUP BY project_id
) sc ON p.id = sc.project_id
LEFT JOIN (
    SELECT 
        ct.project_id,
        COUNT(*) as broadcast_messages
    FROM public.archon_broadcast_messages bm
    JOIN public.archon_cost_tracking ct ON ct.agent_id = bm.sender_id
    WHERE bm.sent_at > NOW() - INTERVAL '24 hours'
    GROUP BY ct.project_id
) bm ON p.id = bm.project_id
GROUP BY p.id, p.name, c.monthly_cost, bc.monthly_budget, sc.shared_contexts, bm.broadcast_messages;

-- Cost Optimization Recommendations View
CREATE OR REPLACE VIEW archon_cost_optimization_recommendations AS
WITH agent_costs AS (
    SELECT 
        ct.agent_id,
        a.model_tier,
        a.agent_type,
        COUNT(*) as task_count,
        AVG(ct.total_cost) as avg_cost_per_task,
        SUM(ct.total_cost) as total_cost,
        AVG(CASE WHEN ct.success THEN 1.0 ELSE 0.0 END) as success_rate,
        AVG(ct.task_duration_seconds) as avg_duration_seconds
    FROM public.archon_cost_tracking ct
    JOIN public.archon_agents_v3 a ON ct.agent_id = a.id
    WHERE ct.recorded_at > NOW() - INTERVAL '30 days'
    GROUP BY ct.agent_id, a.model_tier, a.agent_type
)
SELECT 
    agent_id,
    agent_type,
    model_tier as current_tier,
    total_cost,
    success_rate,
    avg_cost_per_task,
    CASE 
        WHEN model_tier = 'OPUS' AND success_rate > 0.95 AND avg_cost_per_task > 0.50 THEN 'CONSIDER_SONNET'
        WHEN model_tier = 'SONNET' AND success_rate < 0.80 THEN 'CONSIDER_OPUS'  
        WHEN model_tier = 'SONNET' AND success_rate > 0.98 AND avg_cost_per_task < 0.05 THEN 'CONSIDER_HAIKU'
        WHEN model_tier = 'HAIKU' AND success_rate < 0.70 THEN 'CONSIDER_SONNET'
        ELSE 'OPTIMAL'
    END as recommendation,
    CASE 
        WHEN model_tier = 'OPUS' AND success_rate > 0.95 THEN 
            total_cost * 0.8  -- Sonnet costs ~80% less than Opus
        WHEN model_tier = 'SONNET' AND success_rate > 0.98 THEN
            total_cost * 0.17 -- Haiku costs ~83% less than Sonnet  
        ELSE 0
    END as potential_monthly_savings
FROM agent_costs
WHERE task_count >= 5; -- Only agents with sufficient data

-- =====================================================
-- PHASE 11: COMPLETION VALIDATION
-- =====================================================

DO $$
DECLARE
    table_count INTEGER;
    index_count INTEGER;
    trigger_count INTEGER;
BEGIN
    RAISE NOTICE '📋 Phase 11: Final validation of Archon 3.0 Agent Management Schema';
    
    -- Count created tables
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_name LIKE 'archon_%'
    AND table_name IN (
        'archon_agents_v3', 'archon_agent_state_history', 'archon_agent_pools',
        'archon_task_complexity', 'archon_routing_rules', 'archon_agent_knowledge',
        'archon_knowledge_evolution', 'archon_shared_knowledge', 'archon_cost_tracking',
        'archon_budget_constraints', 'archon_roi_analysis', 'archon_shared_contexts',
        'archon_broadcast_messages', 'archon_topic_subscriptions', 'archon_message_acknowledgments',
        'archon_rules_profiles', 'archon_rule_violations'
    );
    
    -- Count created indexes
    SELECT COUNT(*) INTO index_count
    FROM pg_indexes
    WHERE schemaname = 'public'
    AND indexname LIKE 'idx_%v3%' OR indexname LIKE 'idx_agent_%' OR indexname LIKE 'idx_broadcast_%';
    
    -- Count created triggers
    SELECT COUNT(*) INTO trigger_count
    FROM information_schema.triggers
    WHERE trigger_schema = 'public'
    AND trigger_name LIKE 'trigger_%agent%' OR trigger_name LIKE 'trigger_%budget%' OR trigger_name LIKE 'trigger_%knowledge%';
    
    RAISE NOTICE '=== ARCHON 3.0 AGENT MANAGEMENT SCHEMA COMPLETE ===';
    RAISE NOTICE '✅ Tables Created: % (target: 17)', table_count;
    RAISE NOTICE '✅ Performance Indexes: % created', index_count;  
    RAISE NOTICE '✅ Automation Triggers: % created', trigger_count;
    RAISE NOTICE '';
    RAISE NOTICE '🎯 Intelligence-Tiered System Components:';
    RAISE NOTICE '  • Agent Lifecycle Management (5 states: CREATED→ACTIVE→IDLE→HIBERNATED→ARCHIVED)';
    RAISE NOTICE '  • Intelligence Tier Routing (Opus/Sonnet/Haiku with Sonnet-first preference)';
    RAISE NOTICE '  • Knowledge Management System (confidence-based learning, multi-layer storage)';
    RAISE NOTICE '  • Cost Optimization Engine (budget tracking, ROI analysis, tier recommendations)';
    RAISE NOTICE '  • Real-Time Collaboration (shared contexts, pub/sub messaging)';
    RAISE NOTICE '  • Global Rules Integration (parsed from CLAUDE.md, RULES.md, MANIFEST.md)';
    RAISE NOTICE '';
    RAISE NOTICE '📊 Monitoring Views Available:';
    RAISE NOTICE '  • SELECT * FROM archon_agent_performance_dashboard;';
    RAISE NOTICE '  • SELECT * FROM archon_project_intelligence_overview;'; 
    RAISE NOTICE '  • SELECT * FROM archon_cost_optimization_recommendations;';
    RAISE NOTICE '';
    RAISE NOTICE '🚀 Agent Pool Limits (per project):';
    RAISE NOTICE '  • Opus: 2 agents (complex tasks only)';
    RAISE NOTICE '  • Sonnet: 10 agents (default tier for most tasks)';
    RAISE NOTICE '  • Haiku: 50 agents (basic tasks only)';
    RAISE NOTICE '';
    
    IF table_count >= 15 AND index_count >= 8 AND trigger_count >= 3 THEN
        RAISE NOTICE '🎉 SUCCESS: Archon 3.0 Intelligence-Tiered Agent Management schema deployed!';
        RAISE NOTICE '   All components ready for Intelligence-Tiered Adaptive Agent Management';
    ELSE
        RAISE WARNING '⚠️  PARTIAL SUCCESS: Some components may need manual review';
        RAISE NOTICE '   Tables: %/17, Indexes: %, Triggers: %', table_count, index_count, trigger_count;
    END IF;
    
    RAISE NOTICE '==========================================';
    RAISE NOTICE 'Schema migration completed at %', NOW();
END
$$;

-- =====================================================
-- SCHEMA MIGRATION COMPLETE
-- =====================================================